# trasyn: tensor-based arbitrary unitary synthesis

trasyn is a gate synthesis algorithm for early fault-tolerant quantum computing. trasyn natively synthesizes arbitrary single-qubit unitaries to reduce the resulting non-Clifford gate count. For details, check out our paper [Reducing T Gates with Unitary Synthesis](https://arxiv.org/abs/2503.15843). For the original code and data for the paper, please refer to the `paper` branch.

## Install
### Minimum Installation
```bash
pip install trasyn
```

### GPU Acceleration
It is highly recommended to install `cupy` for GPU acceleration:
```bash
pip install trasyn[cupy-cuda12] # or [cupy-cuda11] depending on CUDA version
```
The CUDA Toolkit version can be found using `nvcc --version`.

### Qiskit Integration
Optionally, `qiskit` may be installed to support features like full-circuit transpilation and synthesis.
```bash
pip install trasyn[qiskit]
```

## Usage
### Synthesize a single-qubit unitary
`trasyn.synthesize(target, nonclifford_budget)` takes a target unitary matrix (in the form of a numpy array, three angles for U(theta, phi, lam), or a single Rz angle) and a non-Clifford gate count budget and outputs a gate sequence string of the target gate set, the corresponding matrix, and the distance to the target.

```Python console
>>> seq, mat, err = trasyn.synthesize(trasyn.gates.t(), nonclifford_budget=10)
>>> print(seq, err)
t 0.0
>>> seq, mat, err = trasyn.synthesize([0.1, 0.2, 0.3], nonclifford_budget=20) # U(0.1, 0.2, 0.3)
>>> print(seq, err, seq.count("t"))
yththyththththxthththythththxththxththxththxthsz 0.0018002056473114445 19
>>> seq, mat, err = trasyn.synthesize(pi / 16, 30, error_threshold=0.001) # Rz(pi/16)
>>> print(seq, err, seq.count("t"))
hththththxththththththxthththxththththththxththths 0.0005551347294707683 22
```

#### Full list of `synthesize()` arguments

- `target_unitary` : NDArray | Sequence[float] | float
    
    The target unitary matrix, three angles for U(theta, phi, lam), or a single Rz angle.

- `nonclifford_budget` : int | Sequence[int]
    
    The budget for non-Clifford gates (e.g. T gates). Can be a single integer representing the total budget or a sequence of integers specifying the budget for each tensor.

- `error_threshold` : float, optional
    
    The synthesis error threshold for the method to return once it is met. Note that this is
    not a hard constraint; the method will return the best solution found after all
    attempts if the error threshold is not met or it is not specified. Default is None.

- `gate_set` : "tsh" or "tshxyz", optional
    
    The target gate set. Gates are listed in the order of cost. Additional gate sets can be
    added with the unique_matrices.py script. This process will be made more user-friendly 
    in the future. Default is "tshxyz".

- `num_attempts` : int, optional
    
    The number of sampling attempts per budget configuration. Default is 30.

- `num_samples` : int, optional
    
    The number of samples to in the sampling process. If None, it is calculated based on 
    available memory. Default is None.

- `gpu` : bool, optional
    
    Whether to use GPU for synthesis. Default is True.

- `rng` : numpy.random.Generator | int, optional
    
    Random number generator or seed for reproducibility. Default is None.

- `verbose` : bool, optional
    
    Whether to print verbose output during synthesis. Default is False.


### Synthesize a Qiskit circuit
Given a Qiskit circuit, `trasyn.synthesize_qiskit_circuit(circuit, u3_transpile=True)` transpiles it to CNOT+U3 and synthesizes each U3 gate to bring the full circuit to Clifford+T. The transpilation step attempts various configurations to minimize the nontrivial U3 count and thus, the total T count.

### Use in Command-line
Synthesize a single-qubit unitary:
```console
$ trasyn 'rz(pi/16)' 30
Sequence: hththththxththththththxthththxththththththxththths, Error: 0.0005551347294707683, T-count: 22
```

Synthesize a qasm circuit (need Qiskit to be installed):
```console
$ trasyn circuit.qasm 20 -s synthesized.qasm
```

#### Full list of flags
```
usage: trasyn [-h] [-e ERROR_THRESHOLD] [-g GATE_SET] [--num-samples NUM_SAMPLES] [--num-attempts NUM_ATTEMPTS] [-c] [--seed SEED] [-v] [-s SAVE_PATH] [--skip-transpile] target budget

Synthesize a single-qubit unitary or a qasm circuit to Clifford+T.

positional arguments:
  target                An expression of the target unitary or a filename, e.g. 'Rz(3*pi/8)', 'u(0.1, pi+0.2, 0.3**0.5)', 'unitary.npy', or 'circuit.qasm'.
  budget                The non-Clifford gate budget for synthesis.

options:
  -h, --help            show this help message and exit
  -e ERROR_THRESHOLD, --error-threshold ERROR_THRESHOLD
                        The synthesis error threshold for the method to return once it is met. Note that this is not a hard constraint; the method will return the best solution found after all attempts if the error threshold is not
                        met or it is not specified. Default is None.
  -g GATE_SET, --gate-set GATE_SET
                        The target gate set. Gates are listed in the order of cost. Additional gate sets can be added with the unique_matrices.py script. This process will be made more user-friendly in the future. Default is
                        'tshxyz'.
  --num-samples NUM_SAMPLES, --ns NUM_SAMPLES
                        The number of sampling attempts per budget configuration. Default is 30.
  --num-attempts NUM_ATTEMPTS, --na NUM_ATTEMPTS
                        The number of samples in the sampling process. If None, it is calculated based on available memory. Default is None.
  -c, --cpu-only        Do not use GPU for synthesis.
  --seed SEED           Random seed for reproducibility.
  -v, --verbose         Enable verbose output.
  -s SAVE_PATH, --save-path SAVE_PATH
                        Path to save the synthesized circuit. Default is '<target>_synthesized.qasm' if target is a QASM circuit.
  --skip-transpile, --st
                        Skip the transpilation step for circuit synthesis.
```

