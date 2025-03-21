# trasyn: TensoR-based Arbitrary unitary SYNthesis

This repo contains the code and data for the paper [Reducing T Gates with Unitary Synthesis](https://arxiv.org/abs/2503.15843). We are actively preparing our method to be a user-friendly package.

## Reproduce Results
Install the dependencies in `requirements.txt`. Download the [gridsynth binary](https://www.mathstat.dal.ca/~selinger/newsynth/) and put it in `src/`.  Optionally, install `cupy` to enable GPU acceleration. For the best performance, please `export CUPY_ACCELERATORS=cub` if `cub` is not the default.

- `src/unique_matrices.py` generates the unique matrices and corresponding gate sequences in `assets/`.
- `src/benchmark_random_unitary.py` synthesizes random single-qubit unitaries. The random unitaries are in `data/random_1q/`, and the results are in `data/random_unitary/`.
- `data/qasm/` contains raw qasm files we perform our circuit experiments on. `src/generate_qaoa.py` generates the QAOA qasm files with the gate ordering that maximizes rotation merging.
- `src/count_nontrivial_rotations.py` transpiles the raw qasm files with in `qiskit` using 16 settings as desribed in the paper and counts the number of nontrivial rotations in the transpiled circuits for each setting. The transpiled circuits with the least rotations for the $R_z$ and $U3$ intermediate representations are stored in `data/qasm_transpiled_rz/` and `data/qasm_transpiled_u3/`, respectively.
- `src/benchmark_circuits.py` synthesizes the $R_z$ or $U3$ in the transpiled circuits. The results and the synthesized qasm files are in `data/circuit_benchmark/`.
- `src/pyzx_opt.py` optimizes the synthesized circuits with `pyzx`. The data after optimization is updated in `data/circuit_benchmark/`.
- `src/1q_errors_tradeoff.py` computes the process fidelity of synthesized single-qubit $R_z$ gates under varying logical error rates and synthesis error thresholds to study their tradeoff. The results are in `data/logical_error/`.


## Reproduce Figures and Tables
Install `matplotlib`, `pandas`, and `scipy`. Run `src/plot_performance.ipynb`.
