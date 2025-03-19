import json
import os
from collections.abc import Generator, Sequence
from functools import lru_cache
from math import pi

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit, qasm2

MAX_CACHE_LEN = 10


def h() -> NDArray[np.float64]:
    return np.array([[1, 1], [1, -1]]) / np.sqrt(2)


def s() -> NDArray[np.complex128]:
    return np.array([[1, 0], [0, 1j]])


def t() -> NDArray[np.complex128]:
    return np.array([[1, 0], [0, np.exp(1j * pi / 4)]])


def x() -> NDArray[np.float64]:
    return np.array([[0, 1], [1, 0]])


def y() -> NDArray[np.complex128]:
    return np.array([[0, -1j], [1j, 0]])


def z() -> NDArray[np.float64]:
    return np.array([[1, 0], [0, -1]])


def w() -> NDArray[np.complex128]:
    return np.eye(2) * np.exp(1j * pi / 4)


def p() -> NDArray[np.complex128]:
    return np.eye(2) * np.exp(1j * pi / 8)


def i() -> NDArray[np.float64]:
    return np.eye(2)


def rz(theta: float) -> NDArray[np.complex128]:
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]])


def rx(theta: float) -> NDArray[np.complex128]:
    return np.array(
        [
            [np.cos(theta / 2), -1j * np.sin(theta / 2)],
            [-1j * np.sin(theta / 2), np.cos(theta / 2)],
        ]
    )


def cx() -> NDArray[np.float64]:
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
    )


@lru_cache(maxsize=3**MAX_CACHE_LEN)
def seq2mat_cache(gate_seq: str) -> NDArray[np.complex128]:
    length = len(gate_seq)
    if length == 0:
        return np.eye(2)
    if length == 1:
        return globals()[gate_seq.lower()]()
    return seq2mat_cache(gate_seq[: length // 2]) @ seq2mat(gate_seq[length // 2 :])


def seq2mat(gate_seq: str) -> NDArray[np.complex128]:
    if len(gate_seq) > MAX_CACHE_LEN:
        return seq2mat_cache(gate_seq[:MAX_CACHE_LEN]) @ seq2mat(
            gate_seq[MAX_CACHE_LEN:]
        )
    return seq2mat_cache(gate_seq)


def trace(mat1: NDArray[np.complex128], mat2: NDArray[np.complex128]) -> float:
    return min(np.abs(np.trace(mat1 @ mat2.conj().T) / mat1.shape[0]), 1)


def distance(mat1: NDArray[np.complex128], mat2: NDArray[np.complex128]) -> float:
    return np.sqrt(1 - trace(mat1, mat2) ** 2)


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
    return np.moveaxis(system.reshape(tsr_shape), source, destination).reshape(
        original_shape
    )


def substitute_duplicates(target_sequence: str, lookup_table: dict[str, str]) -> str:
    old_sequence = ""
    while old_sequence != target_sequence:
        old_sequence = target_sequence
        for key, value in lookup_table.items():
            target_sequence = target_sequence.replace(key, value)
    return target_sequence


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
            if filename in exclude_filenames or any(
                k in filename for k in exclude_keywords
            ):
                continue
            if os.path.isdir(path := f"{directory}/{filename}"):
                yield from find_file_to_read(path, exclude_keywords, exclude_filenames)
            yield path


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
