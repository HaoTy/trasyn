import os
from collections import Counter
from itertools import product
from math import pi

import networkx as nx
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.qasm2 import dump
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGatesSimpleCommutation
from qiskit_optimization.applications import Maxcut

DATA_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/../data"

depth_pool = range(1, 6)
qubit_pool = range(4, 27, 2)
seed_pool = range(10)

for p, n, seed in product(depth_pool, qubit_pool, seed_pool):
    graph = nx.random_regular_graph(3, n, seed)
    rng = np.random.default_rng((p, n, seed))
    weights = rng.normal(1, 0.1, 3 * n // 2)
    for i, (w, v) in enumerate(graph.edges):
        graph.edges[w, v]["weight"] = weights[i]
    problem = Maxcut(graph).to_quadratic_program()
    hamiltonian = problem.to_ising()[0]
    gammas = rng.uniform(0, 2 * pi, p)
    betas = rng.uniform(0, pi, p)
    circuit = QuantumCircuit(n)
    circuit.h(range(n))
    for i in range(p):
        counter = Counter()
        for term, coeff in sorted(
            hamiltonian.chop().to_list(),
            key=lambda x: x[0][::-1],
            reverse=True,
        ):
            qbt1, qbt2 = [idx for idx, pauli in enumerate(term[::-1]) if pauli == "Z"]
            counter[qbt1] += 1
            counter[qbt2] += 1
            if counter[qbt1] == 3:
                qbt2, qbt1 = qbt1, qbt2
            circuit.cx(qbt1, qbt2)
            circuit.rz(2 * gammas[i] * coeff.real, qbt2)
            circuit.cx(qbt1, qbt2)
        circuit.rx(2 * betas[i], range(n))
    filename = f"{DATA_DIR}/qasm/qaoa/qaoa_n{n}/qaoa_p{p}_s{seed}_n{n}.qasm"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    dump(circuit, filename)
