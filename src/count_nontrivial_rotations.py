import csv
import json
import os
import warnings
from itertools import product

import numpy as np
from qiskit.compiler import transpile
from qiskit.qasm2 import dump
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGatesSimpleCommutation

from utils import find_qasm_to_read

dataset = "benchpress"
warnings.filterwarnings("ignore", category=DeprecationWarning)
DATA_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/../data/qasm/{dataset}"
ASSETS_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/../assets/"
tensor = np.load(f"{ASSETS_DIR}tensor_1.npy").transpose(1, 0, 2)

data = []

for name, qc in find_qasm_to_read(DATA_DIR, exclude_keywords=["transpiled"]):
    print("\\midrule")
    print(name, end=" ")
    data.append([name])
    qc.remove_final_measurements()
    circuits = []
    for gate_set, opt_lvl in product((["cx", "h", "rz"], ["cx", "u3"]), (0, 1, 2, 3)):
        for circ in [qc, PassManager(
            [Optimize1qGatesSimpleCommutation(run_to_completion=True)]
        ).run(
            transpile(
                qc, basis_gates=["cx", "h", "rz", "rx"], optimization_level=opt_lvl
            )
        )]:
            circ = transpile(circ, basis_gates=gate_set, optimization_level=opt_lvl)
            circuits.append(circ)
            num_rotations = 0
            for op, qbts, _ in circ:
                if op.name in ["rx", "ry", "rz", "u1", "u2", "u3"]:
                    matrix = op.to_matrix()
                    duplicate = np.argwhere(
                        np.isclose(
                            np.abs(
                                np.dot(tensor[:, 0], matrix[0].conj())
                                + np.dot(tensor[:, 1], matrix[1].conj())
                            ),  # calculate trace without matrix multiplication
                            2,
                        )
                    )
                    if len(duplicate) == 0:
                        num_rotations += 1
            print("&", num_rotations, end=" ")
            data[-1].append(num_rotations)
    
    for gate_type, select_range in zip(("rz", "u3"), (9, 17)):
        transpiled_filename = (
            DATA_DIR.replace("qasm", f"qasm_transpiled_{gate_type}")
            + f"/{name.replace(' ', '_')}_transpiled_{gate_type}.qasm"
        )
        os.makedirs(os.path.dirname(transpiled_filename), exist_ok=True)
        dump(circuits[select_range - np.argmin(data[-1][1:select_range][::-1]) - 2], transpiled_filename)
        print("\\\\")

with open(f"{DATA_DIR}/num_nontrivial_rotations.csv", "w", encoding="utf-8") as file:
    csv.writer(file).writerows(data)

data = {row[0]: row[1:] for row in data}
with open(f"{DATA_DIR}/num_nontrivial_rotations.json", "w", encoding="utf-8") as file:
    json.dump(data, file, indent=4)
