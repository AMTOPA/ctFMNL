"""Deterministic single-instance input generation."""
from __future__ import annotations

import numpy as np


def generate_synthetic_instance(n: int, instance_id: int, master_seed: int):
    """Generate one instance with continuous benefits and shared delta.

    Benefits follow U[1,100] and remain raw float64 values; rounding is only
    permitted when formatting reported results.
    """
    rng = np.random.default_rng(np.random.SeedSequence([master_seed, n, instance_id]))
    m = rng.uniform(1.0, 100.0, size=n)
    a = np.exp(rng.uniform(-1.0, 1.0, size=n))
    delta = float(rng.uniform(1.5, 3.0))
    w = float(rng.uniform(0.0, 3.0))
    return {"m": m, "a": a, "delta": delta, "w": w}
