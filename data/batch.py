"""Synthetic input protocol for Section 6.4 batch experiments."""
from __future__ import annotations

from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from experiments.common import sha256_file

ROOT = Path(__file__).resolve().parents[1]
A0 = 1.0
SCALES = (500, 1000, 2000)
Q_VALUES = (1, 5, 20, 100, 300, 500)
SCENARIOS = {"concentrated": 0.01, "moderate": 0.1, "broad": 1.5}
FORMAL_SEED = 20261013
PILOT_SEED = 20261012
N_INSTANCES = 20
FORMAL_DATA = ROOT / "data" / "generated" / "batch"
PILOT_DATA = ROOT / "data" / "generated" / "pilot"


def validate_cached(path: Path, expected: dict[str, np.ndarray]) -> None:
    """Require the complete schema, shapes, dtypes, and unrounded values."""
    try:
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(expected):
                raise ValueError("field names differ")
            for name, value in expected.items():
                actual = saved[name]
                if (actual.dtype != value.dtype or actual.shape != value.shape
                        or not np.array_equal(actual, value)):
                    raise ValueError(f"field {name!r} differs")
    except (OSError, ValueError, EOFError, BadZipFile) as exc:
        raise RuntimeError(f"Saved input does not match protocol: {path}: {exc}") from exc


def request_arrays(n: int, instance: int, seed: int, max_q: int):
    """Generate length-independent prefixes, preserving historical Q=300/500.

    Draw complete fixed-size blocks before slicing. Drawing only ``max_q``
    z-values before delta would shift delta's random stream at every Q.
    """
    if max_q < 1:
        raise ValueError("max_q must be positive")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 640000, n, instance]))
    z = rng.uniform(-1.0, 1.0, size=300)
    delta = rng.uniform(1.5, 3.0, size=300)
    z_parts, delta_parts = [z], [delta]
    for block in range(max(0, (max_q - 300 + 199) // 200)):
        entropy = [seed, 640001, n, instance]
        if block:
            entropy.append(block)
        suffix_rng = np.random.default_rng(np.random.SeedSequence(entropy))
        z_parts.append(suffix_rng.uniform(-1.0, 1.0, size=200))
        delta_parts.append(suffix_rng.uniform(1.5, 3.0, size=200))
    return np.concatenate(z_parts)[:max_q], np.concatenate(delta_parts)[:max_q]



def ensure_data(n, instance, data, z, delta, seed, data_dir=None):
    data_dir = data_dir or (FORMAL_DATA if seed == FORMAL_SEED else PILOT_DATA)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"n{n}_i{instance:02d}.npz"
    expected = {
        "product_id": np.arange(1, n + 1, dtype=np.int64),
        "m": np.asarray(data["m"], dtype=np.float64),
        "a": np.asarray(data["a"], dtype=np.float64),
        "a0": np.asarray(A0, dtype=np.float64),
        "z": np.asarray(z, dtype=np.float64),
        "delta": np.asarray(delta, dtype=np.float64),
        "master_seed": np.asarray(seed, dtype=np.int64),
        "instance_id": np.asarray(instance, dtype=np.int64),
    }
    if path.exists():
        validate_cached(path, expected)
    else:
        np.savez_compressed(path, **expected)
    return path, sha256_file(path)
