import json
import os
import warnings
from itertools import product
from math import log
from typing import Literal, Sequence

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

try:
    import cupy as cp
    from cupy_backends.cuda.api.runtime import CUDARuntimeError

    asnumpy = cp.asnumpy
    MemError = CUDARuntimeError
except ModuleNotFoundError:
    cp = np
    asnumpy = np.asarray
    MemError = MemoryError

from .gates import rz, t, u
from .mps import _sample, _trace_target_unitary
from .utils import distance, get_available_memory, seq2mat

ASSETS_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/assets"
AVAILABLE_GATE_SETS = [
    dir_entry
    for dir_entry in sorted(os.listdir(ASSETS_DIR))
    if os.path.isdir(path := f"{ASSETS_DIR}/{dir_entry}")
]


def _num_candidates(
    nonclifford_count: int | NDArray[np.int64], nonclifford_gate: Literal["t"] = "t"
) -> int | NDArray[np.int64]:
    if nonclifford_gate != "t":
        raise NotImplementedError(f"Non-Clifford gate {nonclifford_gate} is not supported yet.")
    return np.clip(72 * 2.0**nonclifford_count - 48, 0, None).astype(np.int64)


def _substitute_duplicates(target_sequence: str, lookup_table: dict[str, str]) -> str:
    old_sequence = ""
    while old_sequence != target_sequence:
        old_sequence = target_sequence
        for key, value in lookup_table.items():
            target_sequence = target_sequence.replace(key, value)
    return target_sequence


def synthesize(
    target_unitary: NDArray | Sequence[float] | float,
    nonclifford_budget: int | Sequence[int],
    error_threshold: float | None = None,
    gate_set: str = "tshxyz",
    num_attempts: int = 5,
    num_samples: int | None = None,
    context: tuple[NDArray, NDArray] | None = None,
    gpu: bool = True,
    rng: Generator | int | None = None,
    verbose: bool = False,
) -> tuple[str, NDArray[np.complex128], float]:
    """
    Synthesize a gate sequence to approximate a target unitary.

    Parameters
    ----------
    target_unitary : NDArray | Sequence[float] | float
        The target unitary matrix, three angles for U(theta, phi, lam), or a single Rz angle.
    nonclifford_budget : int | Sequence[int]
        The budget for non-Clifford gates (e.g. T gates). Can be a single integer representing
        the total budget or a sequence of integers specifying the budget for each tensor.
    error_threshold : float, optional
        The synthesis error threshold for the method to return once it is met. Note that this is
        not a hard constraint; the method will return the best solution found after all
        attempts if the error threshold is not met or it is not specified. Default is None.
    gate_set : str, optional
        The target gate set. Gates are listed in the order of cost. Additional gate sets can be
        added with the unique_matrices.py script. This process will be made more user-friendly
        in the future. Default is "tshxyz".
    num_attempts : int, optional
        The number of sampling attempts per budget configuration. Default is 5.
    num_samples : int, optional
        The number of samples to in the sampling process. If None, it is calculated based on
        available memory. Default is None.
    context : tuple[NDArray[np.complex128], NDArray[np.complex128]], optional
        Placeholder for an unimplemented feature.
    gpu : bool, optional
        Whether to use GPU for synthesis. Default is True.
    rng : numpy.random.Generator | int, optional
        Random number generator or seed for reproducibility. Default is None.
    verbose : bool, optional
        Whether to print verbose output during synthesis. Default is False.

    Returns
    -------
    str
        The synthesized gate sequence as a string. Gates are listed in the matrix product order.
    NDArray[np.complex128]
        The matrix corresponding to the synthesized gate sequence.
    float
        The error of the synthesized sequence compared to the target unitary.

    Raises
    ------
    ValueError
        If the budget is invalid.
    NotImplementedError
        If the gate set does not have associated assets.

    Examples
    --------

    >>> seq, mat, err = trasyn.synthesize(trasyn.gates.t(), nonclifford_budget=10)
    >>> print(seq, err)
    t 0.0
    >>> seq, mat, err = trasyn.synthesize([0.1, 0.2, 0.3], nonclifford_budget=20) # U(0.1, 0.2, 0.3)
    >>> print(seq, err, seq.count("t"))
    yththyththththxthththythththxththxththxththxthsz 0.0018002056473114445 19
    >>> seq, mat, err = trasyn.synthesize(pi / 16, 30, error_threshold=0.001) # Rz(pi/16)
    >>> print(seq, err, seq.count("t"))
    hththththxththththththxthththxththththththxththths 0.0005551347294707683 22
    """
    gate_set = gate_set.lower()
    if gate_set not in AVAILABLE_GATE_SETS:
        raise NotImplementedError(
            f"Unimplemented gate set: {gate_set}. "
            f"Available gate sets: {', '.join(AVAILABLE_GATE_SETS)}. "
            "(Gates are listed in the order of cost.) "
            "Additional gate sets can be added with the unique_matrices.py script, "
            "This process will be made more user-friendly in the future."
        )

    if isinstance(target_unitary, float):
        target_unitary = rz(target_unitary)
    elif len(target_unitary) == 3:
        target_unitary = u(*target_unitary)
    else:
        target_unitary = np.asarray(target_unitary)

    if gpu and cp is np:
        warnings.warn("cupy not installed, falling back to numpy.")
        gpu = False

    if num_samples is None or isinstance(nonclifford_budget, int):
        memsize = get_available_memory(gpu)
        if verbose:
            print(f"Available memory: {memsize}")

    MAX_COUNT = int(os.listdir(f"{ASSETS_DIR}/{gate_set}")[0].split("_")[1].split(".")[0])
    if isinstance(nonclifford_budget, int):
        tensor_budget = min(11 + int(log(memsize / 2**36, 4)), MAX_COUNT)
        budgets = [[curr_budget + 1] for curr_budget in range(min(nonclifford_budget, MAX_COUNT))]
        for curr_budget in range(MAX_COUNT + 1, nonclifford_budget + 1):
            num_tensors = (curr_budget - 1) // tensor_budget + 1
            budgets.append([curr_budget // num_tensors] * num_tensors)
            budgets[-1][0] = curr_budget - sum(budgets[-1][1:])
    else:
        if max(nonclifford_budget) > MAX_COUNT:
            raise ValueError(
                f"> {MAX_COUNT} non-Clifford gates per tensor for gate set {gate_set} "
                "is not supported."
            )
        budgets = [nonclifford_budget]

    hs_tensor = np.load(f"{ASSETS_DIR}/{gate_set}/tensor_{MAX_COUNT}.npy")
    t_tensor = np.einsum("ipj,jk->ipk", hs_tensor, t())

    if context is not None:
        raise NotImplementedError("Context is not implemented yet.")

    if rng is None or isinstance(rng, int):
        rng = np.random.default_rng(rng)

    if gpu:
        hs_tensor = cp.asarray(hs_tensor)
        t_tensor = cp.asarray(t_tensor)
        target_unitary = cp.asarray(target_unitary)

    best_error, best_string = 2, None
    for budget, _ in product(budgets, range(num_attempts)):
        budget = np.asarray(budget)
        split_low = _num_candidates(budget[:-1] - 2)
        split_high = _num_candidates(budget[:-1] - 1)
        mps = [t_tensor[:, split_low[i] : split_high[i]] for i in range(len(budget) - 1)]
        mps.append(hs_tensor[:, : _num_candidates(budget[-1])])
        mps = _trace_target_unitary(mps, target_unitary)
        if num_samples is None:
            if len(mps) == 1:
                n_samples = 1
            else:
                n_samples = memsize // (
                    max(tsr.shape[1] * tsr.shape[2] for tsr in mps[1:]) * 2 ** (4 + len(budget))
                )
            while n_samples:
                try:
                    bitstring, fidelity = _sample(mps, n_samples, rng=rng)
                    break
                except MemError:
                    n_samples = int(n_samples * 0.9)
        else:
            bitstring, fidelity = _sample(mps, num_samples, rng=rng)
        if context is None:
            fidelity /= 2
        fidelity = min(fidelity, 1)
        error = np.sqrt(1 - fidelity**2)
        if verbose:
            print(f"Budget: {budget}, Num samples: {n_samples}")
            print(f"Error:{error}, Fidelity: {fidelity}")
        if error < best_error - 1e-5:
            best_error = error
            best_string = bitstring
            best_string[:-1] += split_low
        if error_threshold is not None and error <= error_threshold:
            break
    if error_threshold is not None and best_error > error_threshold:
        warnings.warn(
            f"Error threshold {error_threshold} is not reached "
            f"by the lowest error found: {best_error}."
        )

    with open(
        f"{ASSETS_DIR}/{gate_set}/sequences_{MAX_COUNT}.json",
        "r",
        encoding="utf-8",
    ) as f:
        sequences = json.load(f)
    with open(
        f"{ASSETS_DIR}/{gate_set}/duplicates_{MAX_COUNT}.json",
        "r",
        encoding="utf-8",
    ) as f:
        duplicates = json.load(f)

    seqstr = _substitute_duplicates(
        "t".join(_substitute_duplicates(sequences[int(j)], duplicates) for j in best_string),
        duplicates,
    )
    mat = seq2mat(seqstr)
    return seqstr, mat, distance(mat, asnumpy(target_unitary))


try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import HGate, SGate, TGate, XGate, YGate, ZGate
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import Optimize1qGatesSimpleCommutation

    QISKIT_GATES = {
        "h": HGate,
        "s": SGate,
        "t": TGate,
        "x": XGate,
        "y": YGate,
        "z": ZGate,
    }
    CONTINUOUS_GATES = ["rx", "ry", "rz", "u", "u1", "u2", "u3"]
    TENSOR_1T = np.load(
        *[
            f"{ASSETS_DIR}/tshxyz/{dir_entry}"
            for dir_entry in os.listdir(f"{ASSETS_DIR}/tshxyz")
            if dir_entry.endswith(".npy")
        ]
    )[:, : _num_candidates(1)].transpose(1, 0, 2)

    def synthesize_qiskit_circuit(
        circuit: QuantumCircuit, u3_transpile: bool = True, **trasyn_options
    ) -> QuantumCircuit:
        """
        Synthesize a Qiskit circuit to a fault-tolerant gate set.

        Parameters
        ----------
        circuit : qiskit.QuantumCircuit
            The input quantum circuit to be synthesized.
        u3_transpile : bool, optional
            Whether to explore transpilations that can reduce the number of rotations in the
            circuit. Default is True.
        trasyn_options : dict
            Arguments for `synthesize()`.

        Returns
        -------
        qiskit.QuantumCircuit
            The synthesized quantum circuit in the specified fault-tolerant gate set.

        Raises
        ------
        ValueError
            If an unknown gate is encountered in the circuit.

        Notes
        -----
        This function removes final measurements from the circuit.
        """
        circuit.remove_final_measurements()
        if u3_transpile:
            best_circuit, best_num_rotations = None, np.inf
            for opt_lvl in range(4):
                for circ in [
                    circuit,
                    PassManager([Optimize1qGatesSimpleCommutation(run_to_completion=True)]).run(
                        transpile(
                            circuit,
                            basis_gates=["cx", "h", "rz", "rx"],
                            optimization_level=opt_lvl,
                        )
                    ),
                ]:
                    circ = transpile(circ, basis_gates=["cx", "u3"], optimization_level=opt_lvl)
                    num_rotations = 0
                    for op, qbts, _ in circ:
                        if op.name in CONTINUOUS_GATES:
                            matrix = op.to_matrix()
                            duplicate = np.argwhere(
                                np.isclose(
                                    np.abs(
                                        np.dot(TENSOR_1T[:, 0], matrix[0].conj())
                                        + np.dot(TENSOR_1T[:, 1], matrix[1].conj())
                                    ),  # calculate trace without matrix multiplication
                                    2,
                                )
                            )
                            if len(duplicate) == 0:
                                num_rotations += 1
                    if num_rotations < best_num_rotations:
                        best_num_rotations = num_rotations
                        best_circuit = circ
            circuit = best_circuit

        ft_qc = QuantumCircuit(*circuit.qregs, *circuit.cregs)
        synthesized_gates = {}
        for op, qbts, cbts in circuit:
            if op.name in CONTINUOUS_GATES:
                if (key := tuple(op.params)) in synthesized_gates:
                    seq = synthesized_gates[key]
                else:
                    seq = synthesize(op.to_matrix(), **trasyn_options)[0]
                    synthesized_gates[key] = seq
                for gate in seq[::-1]:
                    try:
                        ft_qc.append(QISKIT_GATES[gate](), qbts)
                    except KeyError as err:
                        raise ValueError(f"Unknown gate: {gate}") from err
            else:
                ft_qc.append(op, qbts, cbts)
        return ft_qc

except ImportError:
    pass
