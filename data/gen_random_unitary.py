from qiskit.quantum_info import random_unitary
import numpy as np
import random

for i in range(1000):
    operator = random_unitary(2, i)
    target_unitary = operator.data
    np.save(f"random_1q/s{i}", target_unitary)
