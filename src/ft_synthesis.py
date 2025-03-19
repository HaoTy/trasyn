import json
import os
import subprocess
import warnings
from functools import reduce
from itertools import product
from math import pi
from typing import Literal

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

try:
    import cupy as cp

    asnumpy = cp.asnumpy
except ModuleNotFoundError:
    cp = np
    asnumpy = np.asarray

from mps_tools import sample, trace_target_unitary
from utils import distance, rz, seq2mat, substitute_duplicates, t

ASSETS_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/../assets"


def synthesize(
    target_unitary: NDArray[np.complex128],
    true_state: NDArray[np.complex128] | None = None,
    ft_state: NDArray[np.complex128] | None = None,
    gate_set: Literal["tsh", "tshxyz"] = "tshxyz",
    num_t_in_block: int = 10,
    min_block_count: int = 1,
    max_block_count: int = 4,
    error_threshold: float | None = None,
    num_samples: int = 40000,
    num_attempts: int = 1,
    min_fixed_fraction: float = 0,
    max_fixed_fraction: float = 0,
    ignore_phase: bool = True,
    rng: Generator | int | None = None,
    gpu: bool = False,
    verbose: bool = False,
) -> tuple[str, NDArray[np.complex128], float]:
    hs_tensor = np.load(f"{ASSETS_DIR}/{gate_set}/tensor_{num_t_in_block}.npy")
    t_tensor = np.einsum("ij,jpk->ipk", t(), hs_tensor)
    if true_state is not None:
        num_qubits = int(np.log2(true_state.shape[0])) - 1
        identity = reduce(np.kron, [np.eye(2)] * num_qubits)
        hs_tensor = np.kron(hs_tensor, identity.reshape(2**num_qubits, 1, 2**num_qubits))
        t_tensor = np.kron(t_tensor, identity.reshape(2**num_qubits, 1, 2**num_qubits))
        true_state = true_state.reshape(-1, 1)
        target_unitary = (
            np.kron(target_unitary, identity) @ true_state @ ft_state.reshape(1, -1).conj()
        )
    if gpu:
        if cp is np:
            warnings.warn("cupy not installed, falling back to numpy.")
            gpu = False
        hs_tensor = cp.asarray(hs_tensor)
        t_tensor = cp.asarray(t_tensor)
        target_unitary = cp.asarray(target_unitary)
    best_error, best_string = 2, None
    if max_block_count == 1 and error_threshold is not None:
        for t_count in range(num_t_in_block + 1):
            split_low = max(0, 72 * 2 ** (t_count - 1) - 48)
            split_high = 72 * 2**t_count - 48
            mps = trace_target_unitary([hs_tensor[:, split_low:split_high]], target_unitary)
            bitstring, fidelity = sample(
                mps, num_samples, min_fixed_fraction, max_fixed_fraction, rng
            )
            if true_state is None:
                fidelity /= 2
            fidelity = min(fidelity, 1)
            error = np.sqrt(1 - fidelity**2)
            if verbose:
                print(
                    f"T count: {t_count}, error: {error}, {np.sqrt(1 - fidelity)}, {fidelity}",
                )
            if error < best_error - 1e-5:
                best_error = error
                best_string = bitstring + split_low
            if error < error_threshold:
                break
    else:
        split_low = max(0, 72 * 2 ** (num_t_in_block - 2) - 48)
        split_high = 72 * 2 ** (num_t_in_block - 1) - 48
        for i, j in product(range(min_block_count - 2, max_block_count - 1), range(num_attempts)):
            if num_t_in_block == 1:
                mps = [hs_tensor[:, :split_high]] + [t_tensor[:, :split_high]] * (i + 1)
            elif i == -1:
                mps = [hs_tensor]
            else:
                mps = (
                    [hs_tensor[:, split_low:split_high]]
                    + [t_tensor[:, split_low:split_high]] * i
                    + [t_tensor]
                )
            mps = trace_target_unitary(
                mps,
                target_unitary,
            )
            # if gpu:
            #     cp.get_default_memory_pool().free_all_blocks() # Manual garbage collection causes serious slowdown
            bitstring, fidelity = sample(
                mps,
                num_samples,
                min_fixed_fraction,
                max_fixed_fraction,
                rng + j if isinstance(rng, int) else rng,
            )
            if true_state is None:
                fidelity /= 2
            fidelity = min(fidelity, 1)
            error = np.sqrt(1 - fidelity**2)
            if verbose:
                print(
                    "T count"
                    + (
                        f" = {i + 1}"
                        if num_t_in_block == 1
                        else f" <= {num_t_in_block - 1 + max((i + 1) * num_t_in_block + 1, 0)}"
                    ),
                    f"error: {error}, {np.sqrt(1 - fidelity)}, {fidelity}",
                )
            if error < best_error - 1e-5:
                best_error = error
                best_string = bitstring
            if error_threshold is not None and error <= error_threshold:
                break
    if error_threshold is not None and best_error > error_threshold:
        warnings.warn(
            f"Error threshold {error_threshold} is not reached by the lowest error found: {best_error}."
        )

    with open(
        f"{ASSETS_DIR}/{gate_set}/sequences_{num_t_in_block}.json",
        "r",
        encoding="utf-8",
    ) as f:
        sequences = json.load(f)
    with open(
        f"{ASSETS_DIR}/{gate_set}/duplicates_12.json",
        "r",
        encoding="utf-8",
    ) as f:
        duplicates = json.load(f)

    seqstr = substitute_duplicates(
        "t".join(
            substitute_duplicates(sequences[int(j)], duplicates)
            for j in tuple(best_string[:-1] + split_low) + (best_string[-1],)
        ),
        duplicates,
    )
    if not ignore_phase and true_state is None:
        seqstr += "p" * np.argmin(
            [
                np.linalg.norm(seq2mat(seqstr + "p" * i) - asnumpy(target_unitary), 2)
                for i in range(16)
            ]
        )
    return seqstr, seq2mat(seqstr), best_error


def gridsynth(
    angle: float, error_threshold: float = 1e-3, seed: int = 0
) -> tuple[str, NDArray[np.complex128], float]:
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    seq = (
        (
            subprocess.run(
                [
                    f"{curr_dir}/gridsynth",
                    str(angle % (2 * pi)),
                    "-e",
                    str(error_threshold),
                    "-p",
                    "-r",
                    str(seed),
                ],
                capture_output=True,
                check=False,
            )
            .stdout[:-2]
            .decode()
        )
        .lower()
        .replace("i", "")
    )
    mat = seq2mat(seq)
    return seq, mat, distance(mat, rz(angle))
