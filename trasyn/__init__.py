from .synthesis import is_clifford, is_nontrivial_rotation, synthesize
from .utils import distance, fidelity, seq2mat

try:
    from .synthesis import (
        num_nontrivial_rotations,
        synthesize_qiskit_circuit,
        transpile_circuit_to_u3,
    )
except ImportError:
    pass
