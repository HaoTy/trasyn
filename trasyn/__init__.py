from .synthesis import synthesize

try:
    from .synthesis import synthesize_qiskit_circuit
except ImportError:
    pass
