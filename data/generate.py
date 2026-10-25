"""Generate and validate deterministic experiment inputs without running solvers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data import batch
from data.batch import validate_cached
from data.synthetic import generate_synthetic_instance

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "generated"
SINGLE_SEED = 20260929
EXACTNESS_SCALES = (16,)
COMPARISON_SCALES = (
    100, 500, 1000, 5000, 10000, 20000, 30000,
    40000, 50000, 60000, 70000, 80000, 90000, 100000,
)
ABLATION_SCALES = (10000, 50000, 100000)
EXPERIMENTS = ("all", "exactness", "comparison", "ablation", "batch")


def generate(
    experiment: str = "all",
    *,
    n: list[int] | tuple[int, ...] | None = None,
    instances: int | None = None,
    output: str | Path = DEFAULT_OUTPUT,
) -> list[Path]:
    """Write missing inputs and validate existing files; return their paths.

    Explicit sizes replace each selected experiment's defaults. Ablation uses
    the comparison directory, so both experiment names resolve to identical inputs.
    """
    if experiment not in EXPERIMENTS:
        raise ValueError(f"unknown experiment: {experiment}")
    if n is not None and (not n or any(size < 1 for size in n)):
        raise ValueError("n must contain positive integers")
    if instances is not None and instances < 1:
        raise ValueError("instances must be positive")

    selected = ("exactness", "comparison", "batch") if experiment == "all" else (experiment,)
    paths = []
    for name in selected:
        if name == "batch":
            sizes, count, seed = batch.SCALES, batch.N_INSTANCES, batch.FORMAL_SEED
        elif name == "exactness":
            sizes, count, seed = EXACTNESS_SCALES, 100, SINGLE_SEED
        else:
            sizes = ABLATION_SCALES if name == "ablation" else COMPARISON_SCALES
            count, seed = 20, SINGLE_SEED
        directory = Path(output) / ("comparison" if name == "ablation" else name)
        directory.mkdir(parents=True, exist_ok=True)
        for size in dict.fromkeys(sizes if n is None else n):
            for instance in range(1, (count if instances is None else instances) + 1):
                data = generate_synthetic_instance(size, instance, seed)
                if name == "batch":
                    z, delta = batch.request_arrays(size, instance, seed, max(batch.Q_VALUES))
                    path, _ = batch.ensure_data(size, instance, data, z, delta, seed, directory)
                else:
                    path = directory / f"n{size}_i{instance:02d}.npz"
                    expected = {
                        "product_id": np.arange(1, size + 1, dtype=np.int64),
                        "m": np.asarray(data["m"], dtype=np.float64),
                        "a": np.asarray(data["a"], dtype=np.float64),
                        "a0": np.asarray(1.0, dtype=np.float64),
                        "master_seed": np.asarray(seed, dtype=np.int64),
                        "instance_id": np.asarray(instance, dtype=np.int64),
                        "w": np.asarray(data["w"], dtype=np.float64),
                        "delta": np.asarray(data["delta"], dtype=np.float64),
                    }
                    if path.exists():
                        validate_cached(path, expected)
                    else:
                        with path.open("xb") as stream:
                            np.savez_compressed(stream, **expected)
                paths.append(path)
    return paths


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="all")
    parser.add_argument("--n", type=positive_int, nargs="+", help="override product counts")
    parser.add_argument("--instances", type=positive_int, help="generate IDs 1 through this count")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output root directory")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    paths = generate(args.experiment, n=args.n, instances=args.instances, output=args.output)
    print(f"Validated or generated {len(paths)} input files in {args.output.resolve()}")


if __name__ == "__main__":
    main()
