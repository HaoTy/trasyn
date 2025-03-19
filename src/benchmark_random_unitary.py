import json
import os
from math import pi
from time import time

from qiskit.quantum_info import random_unitary
from qiskit.synthesis import OneQubitEulerDecomposer

from ft_synthesis import gridsynth, synthesize
from utils import distance, rz, seq2mat

algorithm = "trasyn"
gate_set = "tshxyz"
num_t_in_block = 10
num_samples = 40000
max_fixed_fraction = 0
gpu = True
seed_pool = range(1000)

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
if "trasyn" in algorithm:
    algorithm = f"{algorithm}_{gate_set}_tb{num_t_in_block}_ns{num_samples}"
    print(f"num_t_in_block: {num_t_in_block} num_samples: {num_samples}")
dirpath = f"{CURR_DIR}/../data/random_unitary/{algorithm}"
os.makedirs(os.path.dirname(dirpath), exist_ok=True)

for seed in seed_pool:
    print(f"seed: {seed}")
    operator = random_unitary(2, seed)
    target_unitary = operator.data
    angles = [
        instruction.operation.params[0] % (2 * pi)
        for instruction in OneQubitEulerDecomposer("ZXZ")(operator)
    ][::-1]
    data = []
    if "trasyn" in algorithm:
        synthesize(
            rz(0),
            gate_set=gate_set,
            num_t_in_block=num_t_in_block,
            min_block_count=1,
            max_block_count=1,
            num_samples=num_samples,
            max_fixed_fraction=max_fixed_fraction,
            gpu=gpu,
            rng=seed,
        )
        for i in range(1, 4):
            start_time = time()
            seq, mat, err = synthesize(
                target_unitary,
                gate_set=gate_set,
                num_t_in_block=num_t_in_block,
                min_block_count=i,
                max_block_count=i,
                error_threshold=10 ** (-i),
                num_samples=num_samples,
                num_attempts=100,
                max_fixed_fraction=max_fixed_fraction,
                gpu=gpu,
                rng=seed,
            )
            data.append(
                {
                    "epsilon": 10 ** (-i),
                    "time": time() - start_time,
                    "seqstr": seq,
                    "t_count": seq.count("t"),
                    "error": float(err),
                }
            )
            print(data[-1])
    elif "gridsynth" in algorithm:
        for epsilon in [10 ** (-i) for i in range(1, 4)]:
            start_time = time()
            seq = "h".join(gridsynth(angle, epsilon / 3, seed)[0] for angle in angles)
            data.append(
                {
                    "epsilon": epsilon,
                    "time": time() - start_time,
                    "seqstr": seq,
                    "t_count": seq.count("t"),
                    "error": distance(seq2mat(seq), target_unitary),
                }
            )
            print(data[-1])
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    with open(
        f"{dirpath}/{algorithm}_s{seed}" + ".json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "seed": seed,
                "angles": angles,
                # "num_t_in_block": num_t_in_block,
                # "num_samples": num_samples,
                # "max_fixed_fraction": max_fixed_fraction,
                "data": data,
            },
            f,
            indent=4,
        )
