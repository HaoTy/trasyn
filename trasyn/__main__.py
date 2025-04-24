import argparse
import re
from math import pi

import numpy as np

from .gates import GATES
from .synthesis import synthesize

try:
    from qiskit import qasm2

    from .synthesis import synthesize_qiskit_circuit

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


def main():
    """Main function to run trasyn as a command-line tool."""
    parser = argparse.ArgumentParser(
        prog="trasyn",
        description="Synthesize a single-qubit unitary or a qasm circuit to Clifford+T.",
    )
    parser.add_argument(
        "target",
        type=str,
        help="An expression of the target unitary or a filename, e.g. 'Rz(3*pi/8)', "
        + "'u(0.1, pi+0.2, 0.3**0.5)', 'unitary.npy', or 'circuit.qasm'.",
    )
    parser.add_argument("budget", type=int, help="The non-Clifford gate budget for synthesis. ")
    parser.add_argument(
        "-e",
        "--error-threshold",
        type=float,
        help=(
            "The synthesis error threshold for the method to return once it is met. Note that "
            "this is not a hard constraint; the method will return the best solution found after "
            "all attempts if the error threshold is not met or it is not specified. "
            "Default is None."
        ),
    )
    parser.add_argument(
        "-g",
        "--gate-set",
        type=str,
        default="tshxyz",
        help=(
            "The target gate set. Gates are listed in the order of cost. Additional gate sets can"
            " be added with the unique_matrices.py script. This process will be made more "
            "user-friendly in the future. Default is 'tshxyz'."
        ),
    )
    parser.add_argument(
        "--num-attempts",
        "--na",
        type=int,
        default=5,
        help="The number of attempts per budget configuration. Default is 5.",
    )
    parser.add_argument(
        "--num-samples",
        "--ns",
        type=int,
        help=(
            "The number of samples in the sampling process. If None, it is calculated based "
            "on available memory. Default is None."
        ),
    )
    parser.add_argument(
        "-c",
        "--cpu-only",
        action="store_false",
        default=True,
        dest="gpu",
        help="Do not use GPU for synthesis.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output.",
    )
    parser.add_argument(
        "-s",
        "--save-path",
        type=str,
        default=None,
        help="Path to save the synthesized circuit. "
        + "Default is '<target>_synthesized.qasm' if target is a QASM circuit.",
    )
    parser.add_argument(
        "--skip-transpile",
        "--st",
        action="store_false",
        default=True,
        dest="transpile",
        help="Skip the transpilation step for circuit synthesis.",
    )
    args = parser.parse_args()

    if args.target.endswith(".qasm"):
        if not QISKIT_AVAILABLE:
            raise ImportError("Circuit synthesis requires Qiskit to be installed.")
        if args.save_path is None:
            args.save_path = args.target.replace(".qasm", "_synthesized.qasm")
        qasm2.dump(
            synthesize_qiskit_circuit(
                qasm2.load(args.target, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS),
                args.transpile,
                nonclifford_budget=args.budget,
                gate_set=args.gate_set,
                error_threshold=args.error_threshold,
                num_attempts=args.num_attempts,
                num_samples=args.num_samples,
                gpu=args.gpu,
                rng=args.seed,
                verbose=args.verbose,
            ),
            args.save_path,
        )
        exit(0)
    if args.target.endswith(".npy"):
        target = np.load(args.target)
    elif re.match(
        r"^\s*(" + "|".join(GATES) + r")\s*\(\s*([\d\.\s,\/\*\+\-\(\)]|pi)*\s*\)\s*",
        args.target.lower(),
    ):
        target = eval( # pylint: disable=eval-used
            args.target.lower(),
            GATES | {"pi": pi, "__builtins__": None},
        )
    else:
        raise ValueError(
            f"Unknown target unitary format: {args.target}. "
            + "Expected a numpy array file, a gate expression, or a QASM circuit."
        )

    seq, _, err = synthesize(
        target,
        args.budget,
        error_threshold=args.error_threshold,
        gate_set=args.gate_set,
        num_attempts=args.num_attempts,
        num_samples=args.num_samples,
        gpu=args.gpu,
        rng=args.seed,
        verbose=args.verbose,
    )
    print(f"Sequence: {seq}, Error: {err}, T-count: {seq.count('t')}")


if __name__ == "__main__":
    main()
