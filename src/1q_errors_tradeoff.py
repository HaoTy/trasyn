import json
import os
import warnings
from functools import partial
from itertools import product
from multiprocessing import Pool

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate, SGate, TGate, XGate, YGate, ZGate
from qiskit.quantum_info import process_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from ft_synthesis import gridsynth, synthesize
from utils import rz

algorithm = "gridsynth_t_only"
logical_errors = np.around(
    [j * 10**i for i in range(-7, -3) for j in range(1, 10)] + [0.001], 8
)
synthesis_thresholds = np.around(
    [j * 10**i for i in range(-5, -1) for j in range(1, 10)] + [0.1], 8
)
seed_pool = range(1000)
alg_seed = 0
gpu = False
num_processes = 1

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


def simulate_errors(seed: int) -> None:
    savepath = f"{DATA_DIR}/logical_error/1q/{algorithm}/{algorithm}_{seed}.json"
    # if os.path.exists(savepath):
    #     continue
    angle = np.random.default_rng(seed).uniform(0, 2 * np.pi)
    simulator = AerSimulator(method="superop", device="GPU" if gpu else "CPU")
    qc = QuantumCircuit(1)
    qc.rz(angle, 0)
    qc.save_superop()  # pylint: disable=no-member
    ideal_op = simulator.run(qc).result().data(0)["superop"]

    fidelity, t_counts, synthesis_errors, sequences = [], [], [], []
    for epsilon in synthesis_thresholds:
        print(f"{seed=} {epsilon=}")
        if "gridsynth" in algorithm:
            seq, _, err = gridsynth(angle, error_threshold=epsilon, seed=alg_seed)
        elif "trasyn" in algorithm:
            seq, _, err = synthesize(
                rz(angle),
                num_t_in_block=15,
                max_block_count=1,
                num_samples=0,
                gpu=gpu,
                rng=alg_seed,
                error_threshold=epsilon,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        t_counts.append(seq.count("t"))
        synthesis_errors.append(err)
        sequences.append(seq)
        print(f"{t_counts[-1]=} {synthesis_errors[-1]=}")
        ft_qc = QuantumCircuit(1)
        for gate in seq[::-1]:
            if gate in QISKIT_GATES:
                ft_qc.append(QISKIT_GATES[gate](), [0])
            else:
                raise ValueError(f"Unknown gate: {gate}")
        ft_qc.save_superop()  # pylint: disable=no-member

        for logical_error in logical_errors:
            noise_model = NoiseModel()
            noise_model.add_all_qubit_quantum_error(
                depolarizing_error(logical_error, 1),
                ["t"] if "t_only" in algorithm else ["s", "t", "h"],
            )
            noisy_op = (
                AerSimulator(
                    method="superop",
                    noise_model=noise_model,
                    device="GPU" if gpu else "CPU",
                )
                .run(ft_qc)
                .result()
                .data(0)["superop"]
            )
            fidelity.append(process_fidelity(noisy_op, ideal_op))
            print(f"{logical_error=}", fidelity[-1])

    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    with open(savepath, "w", encoding="utf-8") as file:
        json.dump(
            {
                "algorithm": algorithm,
                "seed": seed,
                "angle": angle,
                "logical_errors": logical_errors.tolist(),
                "synthesis_thresholds": synthesis_thresholds.tolist(),
                "fidelity": fidelity,
                "t_counts": t_counts,
                "synthesis_errors": synthesis_errors,
                "sequences": sequences,
            },
            file,
            indent=4,
        )


with Pool(num_processes) as pool:
    pool.map(simulate_errors, seed_pool)
