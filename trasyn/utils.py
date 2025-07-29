import json
import os
import subprocess
from collections.abc import Generator, Sequence
from functools import lru_cache
from math import sqrt

import numpy as np
import psutil
from numpy.typing import NDArray

from .gates import GATES

ASSETS_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/assets"
_MAX_CACHE_LEN = 8


@lru_cache(maxsize=3**_MAX_CACHE_LEN)
def _seq2mat_cache(gate_seq: str) -> NDArray[np.complex128]:
    length = len(gate_seq)
    if length == 0:
        return np.eye(2)
    if length == 1:
        try:
            return GATES[gate_seq.lower()]()
        except KeyError as err:
            raise ValueError(
                f"Unknown gate: {gate_seq}. Available gates: {', '.join(GATES)}."
            ) from err
    return _seq2mat_cache(gate_seq[: length // 2]) @ _seq2mat_cache(gate_seq[length // 2 :])


def seq2mat(gate_seq: str) -> NDArray[np.complex128]:
    if len(gate_seq) > _MAX_CACHE_LEN:
        return _seq2mat_cache(gate_seq[:_MAX_CACHE_LEN]) @ seq2mat(gate_seq[_MAX_CACHE_LEN:])
    return _seq2mat_cache(gate_seq)


def to_superop(
    matrix: NDArray[np.complex128], depolar_error_rate: float = 0
) -> NDArray[np.complex128]:
    # conj outer matrix to be consistent with qiskit's convention
    return np.round(
        (1 - 3 * depolar_error_rate / 4) * np.kron(matrix.conj(), matrix)
        + sum(
            depolar_error_rate
            / 4
            * np.kron(GATES[seq]().conj() @ matrix.conj(), GATES[seq]() @ matrix)
            for seq in ("x", "y", "z")
        ),
        16,
    )


@lru_cache(maxsize=3**_MAX_CACHE_LEN)
def _seq2superop_cache(
    gate_seq: str, logical_error_rates: tuple[tuple[str, float]] = ()
) -> NDArray[np.complex128]:
    length = len(gate_seq)
    if length == 0:
        return np.eye(4)
    if length == 1:
        try:
            gate_seq = gate_seq.lower()
            return to_superop(GATES[gate_seq](), dict(logical_error_rates).get(gate_seq, 0))
        except KeyError as err:
            raise ValueError(
                f"Unknown gate: {gate_seq}. Available gates: {', '.join(GATES)}."
            ) from err
    return _seq2superop_cache(gate_seq[: length // 2], logical_error_rates) @ _seq2superop_cache(
        gate_seq[length // 2 :], logical_error_rates
    )


def seq2superop(
    gate_seq: str, logical_error_rates: tuple[tuple[str, float]] = ()
) -> NDArray[np.complex128]:
    if len(gate_seq) > _MAX_CACHE_LEN:
        return _seq2superop_cache(gate_seq[:_MAX_CACHE_LEN], logical_error_rates) @ seq2superop(
            gate_seq[_MAX_CACHE_LEN:], logical_error_rates
        )
    return _seq2superop_cache(gate_seq, logical_error_rates)


def fidelity(
    mat1: NDArray[np.complex128], mat2: NDArray[np.complex128] | None = None, superop: bool = False
) -> float:
    if mat2 is None:
        entries = np.diag(mat1)
    else:
        entries = mat1 * mat2.conj()  # transposes cancel out
    trace_value = np.abs(np.sum(entries)) / mat1.shape[0]
    if not superop:
        trace_value = trace_value**2
    return min(float(trace_value), 1)


def distance(
    mat1: NDArray[np.complex128], mat2: NDArray[np.complex128] | None = None, superop: bool = False
) -> float:
    return sqrt(1 - fidelity(mat1, mat2, superop))


def transpose(
    system: NDArray[np.complex128], first_qubit: int, second_qubit: int
) -> NDArray[np.complex128]:
    if first_qubit == second_qubit:
        return system
    num_qubits = int(np.log2(system.shape[0]))
    source, destination = [first_qubit], [second_qubit]
    original_shape, tsr_shape = system.shape, [2] * num_qubits
    if system.ndim > 1:
        source.append(first_qubit + num_qubits)
        destination.append(second_qubit + num_qubits)
        tsr_shape = [2] * (2 * num_qubits)
    return np.moveaxis(system.reshape(tsr_shape), source, destination).reshape(original_shape)


def find_file_to_read(
    base_dir: str | Sequence[str],
    exclude_keywords: Sequence[str] = (),
    exclude_filenames: Sequence[str] = (),
) -> Generator[str, None, None]:
    if isinstance(base_dir, str):
        base_dir = [base_dir]
    for directory in base_dir:
        if directory.endswith("/"):
            directory = directory[:-1]
        for filename in sorted(os.listdir(directory)):
            if filename in exclude_filenames or any(k in filename for k in exclude_keywords):
                continue
            if os.path.isdir(path := f"{directory}/{filename}"):
                yield from find_file_to_read(path, exclude_keywords, exclude_filenames)
            yield path


def find_json_to_read(
    base_dir: str | Sequence[str],
    exclude_keywords: Sequence[str] = (),
    exclude_filenames: Sequence[str] = (),
) -> Generator[tuple[str, dict], None, None]:
    for path in find_file_to_read(base_dir, exclude_keywords, exclude_filenames):
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
            yield path.split("/")[-1][:-5].lower().replace("_", " "), data


def get_available_memory(gpu: bool = False) -> int:
    if gpu:
        memsize, unit = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader"]
            )
            .decode()
            .strip()
            .split()[:2]
        )
        memsize = int(memsize)
        if unit == "MiB":
            memsize *= 1024**2
        elif unit == "GiB":
            memsize *= 1024**3
        else:
            raise NotImplementedError(
                f"Unknown memory unit {unit}. Only MiB and GiB are supported at this moment."
            )
    else:
        memsize = psutil.virtual_memory().available
    return memsize


try:
    from qiskit import QuantumCircuit, qasm2

    def find_qasm_to_read(
        base_dir: str | Sequence[str],
        exclude_keywords: Sequence[str] = (),
        exclude_filenames: Sequence[str] = (),
    ) -> Generator[tuple[str, QuantumCircuit], None, None]:
        for path in find_file_to_read(base_dir, exclude_keywords, exclude_filenames):
            if (filename := path.split("/")[-1]).endswith(".qasm"):
                yield filename[:-5].lower().replace("_", " "), qasm2.load(
                    path, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS
                )

except ImportError:
    pass
