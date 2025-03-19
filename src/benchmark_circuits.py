import json
import os
import warnings
from functools import partial
from time import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate, SGate, TGate, XGate, YGate, ZGate
from qiskit.compiler import transpile
from qiskit.qasm2 import dump, load
from qiskit.quantum_info import state_fidelity
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGatesSimpleCommutation
from qiskit_aer import AerSimulator, StatevectorSimulator, UnitarySimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from ft_synthesis import gridsynth, synthesize
from utils import distance, find_qasm_to_read

dataset = "qaoa"
algorithm = "gridsynth"
epsilon = 0.007
logical_errors = [1e-6, 1e-5, 1e-4]
gate_set = "tshxyz"
num_t_in_block = 10
num_samples = 40000
max_block_count = 2
min_fixed_fraction = 0
max_fixed_fraction = 0
gpu = True
seed = 0

warnings.filterwarnings("ignore", category=DeprecationWarning)
DATA_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/../data"
QISKIT_GATES = {
    "h": HGate,
    "s": SGate,
    "t": TGate,
    "x": XGate,
    "y": YGate,
    "z": ZGate,
}

if algorithm == "gridsynth":
    synthesizer = partial(gridsynth, seed=seed)
    alg_config = f"{algorithm}_eps{epsilon}"
elif algorithm == "trasyn":
    synthesizer = partial(
        synthesize,
        gate_set=gate_set,
        num_t_in_block=num_t_in_block,
        min_block_count=max_block_count,
        max_block_count=max_block_count,
        num_samples=num_samples,
        min_fixed_fraction=min_fixed_fraction,
        max_fixed_fraction=max_fixed_fraction,
        gpu=gpu,
        rng=seed,
    )
    alg_config = f"{algorithm}_{gate_set}_tb{num_t_in_block}_ns{num_samples}_bc{max_block_count}"
else:
    raise ValueError(f"Unknown algorithm: {algorithm}")


with open(
    f"{DATA_DIR}/qasm/{dataset}/num_nontrivial_rotations.json", "r", encoding="utf-8"
) as file:
    num_nontrivial_rotations = json.load(file)

for name, qc in find_qasm_to_read(f"{DATA_DIR}/qasm/{dataset}", exclude_keywords=["transpiled"]):
    print(name)
    # if qc.num_qubits > 12:
    #     continue
    qc.remove_final_measurements()
    filename = (
        f"{DATA_DIR}/circuit_benchmark/{alg_config}/{dataset}"
        + f"/{name.replace(' ', '_')}_{alg_config}_s{seed}"
    )
    if os.path.exists(f"{filename}.json"):
        continue
    num_rotations_list = np.array(num_nontrivial_rotations[name])
    if algorithm == "trasyn":
        (  # pylint: disable=unbalanced-tuple-unpacking
            use_u3,
            opt_lvl,
            use_commutation,
        ) = np.unravel_index(15 - np.argmin(num_rotations_list[::-1]), (2, 4, 2))
        num_rotations = np.min(num_rotations_list)
        effective_epsilon = None
    else:
        use_u3 = False
        opt_lvl, use_commutation = np.unravel_index(  # pylint: disable=unbalanced-tuple-unpacking
            7 - np.argmin(num_rotations_list[:8][::-1]), (4, 2)
        )
        num_rotations = np.min(num_rotations_list[:8])
        effective_epsilon = epsilon * np.min(num_rotations_list) / num_rotations
    if use_u3:
        gate_type = "u3"
        gate_set = ["cx", "u3"]
    else:
        gate_type = "rz"
        gate_set = ["cx", "h", "rz"]
    transpiled_filename = (
        f"{DATA_DIR}/qasm_transpiled_{gate_type}/"
        + f"{dataset}/{name.replace(' ', '_')}_transpiled_{gate_type}.qasm"
    )
    if os.path.exists(transpiled_filename):
        qc = load(transpiled_filename)
    else:
        if use_commutation:
            qc = PassManager([Optimize1qGatesSimpleCommutation(run_to_completion=True)]).run(
                transpile(qc, basis_gates=["cx", "h", "rz", "rx"], optimization_level=opt_lvl)
            )
        qc = transpile(qc, basis_gates=gate_set, optimization_level=opt_lvl)
        os.makedirs(os.path.dirname(transpiled_filename), exist_ok=True)
        dump(qc, transpiled_filename)

    if os.path.exists(filename + ".qasm"):
        ft_qc = load(filename + ".qasm")
        with open(f"{filename}.json", "r", encoding="utf-8") as file:
            duration = json.load(file)["time"]
    else:
        ft_qc = QuantumCircuit(*qc.qregs, *qc.cregs)
        synthesized_gates = {}
        start_time = time()
        for op, qbts, cbts in qc:
            if op.name in ["rz", "u3"]:
                if (key := tuple(op.params)) in synthesized_gates:
                    seq = synthesized_gates[key]
                else:
                    seq = synthesizer(
                        key[0] if algorithm == "gridsynth" else op.to_matrix(),
                        error_threshold=effective_epsilon,
                    )[0]
                    synthesized_gates[key] = seq
                for gate in seq[::-1]:
                    if gate in QISKIT_GATES:
                        ft_qc.append(QISKIT_GATES[gate](), qbts)
                    else:
                        raise ValueError(f"Unknown gate: {gate}")
            else:
                ft_qc.append(op, qbts, cbts)
        duration = time() - start_time

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        dump(ft_qc, f"{filename}.qasm")

    noisy_fidelity, unitary_distance, fidelity = None, None, None
    if qc.num_qubits <= 27:
        simulator = StatevectorSimulator(device="GPU" if gpu else "CPU")
        ideal_sv = simulator.run(qc).result().get_statevector()
        fidelity = state_fidelity(
            ideal_sv,
            simulator.run(ft_qc).result().get_statevector(),
        )
    if qc.num_qubits <= 12:
        simulator = UnitarySimulator(device="GPU" if gpu else "CPU")
        unitary_distance = distance(
            simulator.run(qc).result().get_unitary(),
            simulator.run(ft_qc).result().get_unitary(),
        )
        if logical_errors:
            noisy_fidelity = {}
        for logical_error in logical_errors:
            noise_model = NoiseModel()
            noise_model.add_all_qubit_quantum_error(
                depolarizing_error(logical_error, 1), ["s", "t", "h"]
            )
            noise_model.add_all_qubit_quantum_error(depolarizing_error(logical_error, 2), ["cx"])
            noisy_qc = ft_qc.copy()
            noisy_qc.save_density_matrix()
            noisy_op = (
                AerSimulator(
                    method="density_matrix",
                    noise_model=noise_model,
                    device="GPU" if gpu else "CPU",
                )
                .run(noisy_qc)
                .result()
                .data(0)["density_matrix"]
            )
            noisy_fidelity[logical_error] = state_fidelity(noisy_op, ideal_sv)
    num_gates = ft_qc.count_ops()
    t_depth = ft_qc.depth(lambda x: x.name == "t")
    print(
        f"{unitary_distance=} {fidelity=} {num_rotations=} {duration=} {t_depth=} {effective_epsilon=}"
    )
    print(num_gates)
    print(noisy_fidelity)
    with open(f"{filename}.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "algorithm_config": alg_config,
                "benchmark_name": name,
                "n": qc.num_qubits,
                "seed": seed,
                "unitary_distance": unitary_distance,
                "state_fidelity": fidelity,
                "num_nontrivial_rotations": int(num_rotations),
                "time": duration,
                "num_gates": num_gates,
                "noisy_fidelity": noisy_fidelity,
                "t_depth": t_depth,
            },
            file,
            indent=4,
        )
