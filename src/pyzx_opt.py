import os
import json
import pyzx as zx
from qiskit import qasm2


def optimize_pyzx(circuit_file):
    circuit = zx.Circuit.from_qasm_file(circuit_file)
    circuit_graph = circuit.to_graph()
    circuit_graph = zx.teleport_reduce(circuit_graph)
    circuit = zx.Circuit.from_graph(circuit_graph)
    return qasm2.loads(circuit.to_qasm())


directories = ["benchpress", "hamlib", "qaoa"]
for directory_name in directories:
    directory = (
        f"./data/circuit_benchmark/trasyn_tshxyz_tb10_ns40000_bc2/{directory_name}"
    )
    for f in sorted(os.listdir(directory)):
        if not f.endswith(".qasm"):
            continue
        print(f)
        file = f"{directory}/{f}"
        data = json.load(open(file.replace(".qasm", ".json")))
        if sum(data["num_gates"].values()) > 50000:
            continue
        qc = optimize_pyzx(file)
        optimized = qc.count_ops()
        optimized_tcount = optimized.get("t", 0) + optimized.get("tdg", 0)
        data["optimized_t_count"] = optimized_tcount
        data["optimized_num_gates"] = optimized
        data["optimized_t_depth"] = qc.depth(lambda x: x.name == "t")
        json.dump(data, open(file.replace(".qasm", ".json"), "w"), indent=4)
        print(data["optimized_t_depth"] / data["t_depth"])
