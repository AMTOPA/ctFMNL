# Synthetic experiment inputs

Run commands from the repository root. All experiments use the shared
`data.synthetic.generate_synthetic_instance` generator. Batch requests and
files use `data.batch.request_arrays` and `ensure_data`.

## Commands

```sh
python -m data.generate --experiment all
python -m data.generate --experiment exactness
python -m data.generate --experiment comparison
python -m data.generate --experiment ablation
python -m data.generate --experiment batch
python -m data.generate --experiment all --n 16 32 --instances 2 --output ./sample_inputs
python -m data.generate --experiment exactness --n 16 --instances 1 --output data/examples
python -m data.generate --experiment comparison --n 100 500 1000 5000 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 --instances 1 --output data/examples
```

`--experiment` defaults to `all`. `--n` accepts one or more positive integers and
replaces the default product counts for every selected experiment. `--instances`
is a positive count, with IDs starting at 1. `--output` defaults to
`data/generated` relative to the repository, independently of the working
directory. Default inputs land in `data/generated/exactness`,
`data/generated/comparison`, and `data/generated/batch`; the batch runner writes
results to `results/batch`. A supplied relative output path is relative to the
working directory.

## Protocols and file locations

| Experiment | Product counts n | Instance IDs | Master seed | Output subdirectory |
| --- | --- | --- | --- | --- |
| exactness | 16 | 1–100 | 20260929 | `exactness` |
| comparison | 100, 500, 1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000 | 1–20 | 20260929 | `comparison` |
| ablation | 10000, 50000, 100000 | 1–20 | 20260929 | `comparison` |
| batch | 500, 1000, 2000 | 1–20 | 20261013 | `batch` |

Each file is named `n{n}_i{instance:02d}.npz`, for example
`comparison/n10000_i01.npz`. Ablation reads and writes exactly the same input
files as comparison: no ablation copies or separate generation stream are used.
`all` generates exactness, comparison (including the ablation subset), and batch
once each. Default generation produces 100 exactness, 280 comparison, and 60
batch files. The example commands produce `data/examples/exactness/n16_i01.npz` and one file for every comparison size
under `data/examples/comparison/`, each with instance ID 1. They are one-instance
samples of the default exactness and comparison protocols.

Existing files are validated against deterministic regeneration, including every
field's name, shape, dtype, and value. A mismatch raises an error without replacing
the cached file. Valid cached files are not rewritten.

## Single-instance generation

Initialize NumPy `default_rng(SeedSequence([master_seed, n, instance_id]))`.
Draw in this order:

1. `m_i ~ Uniform(1, 100)`, n draws;
2. `a_i = exp(u_i)`, with `u_i ~ Uniform(-1, 1)`, n draws;
3. `delta ~ Uniform(1.5, 3)`, one draw;
4. `w ~ Uniform(0, 3)`, one draw.

The outside-option attraction is `a0 = 1`. Single-instance NPZ fields are:

| Field | Shape | dtype |
| --- | --- | --- |
| `product_id` | `(n,)`, IDs 1 through n | int64 |
| `m`, `a` | `(n,)` | float64 |
| `a0`, `w`, `delta` | scalar | float64 |
| `master_seed`, `instance_id` | scalar | int64 |

All floating-point inputs retain the generator's full float64 values without
rounding. The same `(master_seed, n, instance_id)` reproduces the same arrays.

## Batch requests and Q prefixes

Batch product arrays follow the same product generator with master seed
20261013. Every saved batch contains 500 requests, supporting
`Q = 1, 5, 20, 100, 300, 500`. Its NPZ fields are `product_id`, `m`, `a`, `a0`,
`z`, `delta`, `master_seed`, and `instance_id`. Product and metadata dtypes match
the table above; `z` and `delta` are float64 arrays of shape `(500,)`. There is no
scalar `w` in batch files.

Request generation uses streams independent of the product stream:

- Initialize `default_rng(SeedSequence([20261013, 640000, n, instance_id]))`;
  draw 300 values of `z ~ Uniform(-1, 1)`, then 300 values of
  `delta ~ Uniform(1.5, 3)`.
- Initialize `default_rng(SeedSequence([20261013, 640001, n, instance_id]))`;
  draw another 200 z values, then another 200 delta values, and append them.

For each scenario, request q has `w_q = 1.5 + h * z_q` and the saved `delta_q`:

| Scenario | h | Range of w |
| --- | --- | --- |
| concentrated | 0.01 | [1.49, 1.51] |
| moderate | 0.1 | [1.4, 1.6] |
| broad | 1.5 | [0, 3] |

A Q-request batch uses exactly `z[:Q]` and `delta[:Q]`; scenarios share these
underlying draws. Complete 300/200-value blocks are drawn before slicing, so
request prefixes do not depend on the requested Q. No solver is run by this CLI.
