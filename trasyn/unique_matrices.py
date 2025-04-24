import json
import os
from itertools import product

import numpy as np

from .gates import t
from .synthesis import _substitute_duplicates
from .utils import seq2mat, trace

try:
    import cupy as cp

    asnumpy = cp.asnumpy
except ModuleNotFoundError:
    cp = np
    asnumpy = np.asarray

nontrivial_cliffords = "sh"
trivial_cliffords = "xyz"
clifford_gates = nontrivial_cliffords + trivial_cliffords
nonclifford_gate = "t"
ASSETS_DIR = (
    f"{os.path.dirname(os.path.abspath(__file__))}/assets/{nonclifford_gate}{clifford_gates}/"
)

if __name__ == "__main__":
    try:
        matrices = np.load(f"{ASSETS_DIR}tensor_0.npy")
        with open(f"{ASSETS_DIR}sequences_0.json", "r", encoding="utf-8") as file:
            hs_sequences = json.load(file)
    except FileNotFoundError:
        matrices = np.eye(2, dtype=complex).reshape(1, 2, 2)
        sequences = [""]
        duplicates = {}
        for length in range(1, 8):
            print(f"{length = }")
            for seq in product(clifford_gates, repeat=length):
                seqstr = _substitute_duplicates("".join(seq), duplicates)
                if seqstr in sequences:
                    continue
                matrix = seq2mat(seqstr)
                duplicate = False
                for existing_seq, existing_mat in zip(sequences, matrices):
                    if not np.allclose(trace(existing_mat, matrix), 1):
                        continue
                    duplicate = True
                    counts = []
                    for s in (existing_seq, seqstr):
                        counts.append([])
                        for gate in nontrivial_cliffords:
                            counts[-1].append(s.count(gate))
                        counts[-1].append(len(s))
                        counts[-1].append(s)
                    if tuple(counts[0]) > tuple(counts[1]):
                        duplicates[existing_seq] = seqstr
                    else:
                        duplicates[seqstr] = existing_seq
                    break
                if not duplicate:
                    sequences.append(seqstr)
                    matrices = np.vstack([matrices, matrix.reshape(1, 2, 2)])
            print(matrices.shape[0])
        matrices = matrices.transpose(1, 0, 2)
        np.save(f"{ASSETS_DIR}tensor_0.npy", matrices)
        with open(f"{ASSETS_DIR}sequences_0.json", "w", encoding="utf-8") as file:
            json.dump(sequences, file, indent=4)
        with open(f"{ASSETS_DIR}duplicates_0.json", "w", encoding="utf-8") as file:
            json.dump(duplicates, file, indent=4)
        hs_sequences = sequences

    t_block = np.einsum("ij,jpk->pik", t(), matrices)

    for length in range(1, 16):
        print(f"{length = }")
        with open(
            f"{ASSETS_DIR}duplicates_{length-1}.json",
            "r",
            encoding="utf-8",
        ) as file:
            duplicates = json.load(file)
        with open(f"{ASSETS_DIR}sequences_{length-1}.json", "r", encoding="utf-8") as file:
            sequences = json.load(file)
        matrices = np.load(f"{ASSETS_DIR}tensor_{length-1}.npy").transpose(1, 0, 2)
        matrices = cp.asarray(matrices)
        t_block = cp.asarray(t_block)
        for (mat1, seq1), (mat2, seq2) in product(
            zip(matrices, sequences), zip(t_block, hs_sequences)
        ):
            seqstr = _substitute_duplicates(seq1 + "t" + seq2, duplicates)
            if seqstr in sequences:
                continue
            matrix = mat1 @ mat2
            duplicate = cp.argwhere(
                cp.isclose(
                    cp.abs(
                        cp.dot(matrices[:, 0], matrix[0].conj())
                        + cp.dot(matrices[:, 1], matrix[1].conj())
                    ),  # calculate trace without matrix multiplication
                    2,
                )
            )
            if len(duplicate) == 0:
                sequences.append(seqstr)
                matrices = cp.vstack([matrices, matrix.reshape(1, 2, 2)])
            else:
                existing_seq = sequences[duplicate[0][0].item()]
                counts = []
                for s in (existing_seq, seqstr):
                    counts.append([])
                    for gate in nonclifford_gate + nontrivial_cliffords:
                        counts[-1].append(s.count(gate))
                    counts[-1].append(len(s))
                    counts[-1].append(s)
                if tuple(counts[0]) > tuple(counts[1]):
                    duplicates[existing_seq] = seqstr
                elif existing_seq != seqstr:
                    duplicates[seqstr] = existing_seq
        print(len(matrices), len(sequences), len(duplicates))
        np.save(f"{ASSETS_DIR}tensor_{length}.npy", asnumpy(matrices.transpose(1, 0, 2)))
        with open(f"{ASSETS_DIR}sequences_{length}.json", "w", encoding="utf-8") as file:
            json.dump(sequences, file, indent=4)
        with open(
            f"{ASSETS_DIR}duplicates_{length}.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(duplicates, file, indent=4)
