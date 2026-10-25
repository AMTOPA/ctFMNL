# Exact ctFMNL Assortment Optimization

Python implementations of **FABS** (Fenwick-Accelerated Breakpoint Search) and **SPAR** (Shared Preprocessing and Assortment Reuse) for ctFMNL assortment optimization.

## Installation

Python 3.10 or newer is required. Run the following commands from the repository root.

```bash
python -m pip install -r requirements.txt
```

`requirements-tested.txt` lists the dependency versions used for testing.

## Repository structure

| Directory | Contents |
| --- | --- |
| `algorithms/` | FABS, SPAR, Guan et al.'s Algorithm A.1, MNL-threshold and component-ablation solvers |
| `data/` | Input-generation code, experimental sizes and seeds, 100 exactness inputs, scale examples and Section 6.2 CSV data |
| `experiments/` | Shared experiment utilities and Section 6.4 batch runners and consistency checks |
| `visualization/` | Section 6.2 figure generation |
| `figures/comparison/` | Runtime and paired-speedup figures in PDF and PNG formats |
| `tests/` | Algorithm, enumeration, data-generation and experiment-integrity tests |

## Algorithm interfaces

- FABS: `solveCtFmnlNLog2N(products, a0, w, q)` in `algorithms/fabs.py`.
- Algorithm A.1: `solveCtFmnlAlgorithmA1(products, a0, w, delta)` in `algorithms/guan_a1.py`.
- SPAR: `RepeatedOptimizer(products, a0).solve(requests)` in `algorithms/spar.py`, with requests supplied as `(w, delta)` pairs.

FABS and the MNL-threshold solver use `q = 1 + delta`; Algorithm A.1 and the ablation solvers take `delta` directly. Product IDs are unique positive integers. ID zero denotes the outside option. Benefits are nonnegative, attractiveness values are positive, and `w` and `delta` are nonnegative.

```python
from algorithms.fabs import Product, solveCtFmnlNLog2N

products = [Product(productId=1, a=1.2, m=10.0),
            Product(productId=2, a=0.8, m=20.0)]
value, assortment = solveCtFmnlNLog2N(products, a0=1.0, w=0.5, q=3.0)
```

## Experimental input data

```bash
python -m data.generate --experiment all
```

This command generates the inputs for the small-instance exactness check and Sections 6.2, 6.3 and 6.4. The scale lists, instance counts, seeds, array formats and generation commands are documented in `data/README.md`. Section 6.3 uses the same product instances and scalar parameters as Section 6.2 at the corresponding sizes. Benefits and reinforcement parameters are retained as unrounded float64 values.

```bash
python -m data.generate --experiment comparison --n 10000 --instances 20
python -m data.generate --experiment batch
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover small-instance enumeration, focal-membership boundaries, tied attractiveness values, zero benefits, objective evaluation, reproducible inputs and experiment integrity.

## Section 6.2 figures

```bash
python -m visualization.plot_comparison
```

The script reads `data/comparison/runtime_summary.csv` and `data/comparison/speedup_summary.csv`, and generates the figures in `figures/comparison/`. `data/comparison/paired_instances.csv` contains paired observations. Runtime summaries report medians and first/third quartiles over successful calls. Paired speedups use instances on which both methods completed successfully. The hard time limit is 600 seconds.

## Section 6.4 batch experiments

```bash
python -m experiments.batch.run_parallel --help
python -m experiments.batch.run_parallel --workers 10
python -m experiments.batch.audit results/batch
```

Each worker evaluates the two methods sequentially on the same batch; independent batches run in parallel. The runner saves generated inputs, the experimental protocol, source snapshots, request-level solutions and timing records. The audit checks recorded inputs, source files and experiment keys. Use an empty output directory for each run.

## Code and data availability

The algorithm implementations, automated tests, experimental input generators, batch experiment scripts, small input examples and Section 6.2 plotting data are available in this repository.

Repository: https://github.com/AMTOPA/ctFMNL

Release: https://github.com/AMTOPA/ctFMNL/releases/tag/v0.1.0

Manuscript title: *Exact ctFMNL Assortment Optimization*.
