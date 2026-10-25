"""Shared provenance, request hashing, and protocol-grid helpers for Section 6.4."""
from __future__ import annotations

import hashlib
import shutil
import platform
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from experiments.common import sha256_file


def digest_products(data) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(data["m"], dtype="<f8").tobytes())
    digest.update(np.asarray(data["a"], dtype="<f8").tobytes())
    return digest.hexdigest()


def digest_requests(requests) -> str:
    return hashlib.sha256(np.asarray(requests, dtype="<f8").tobytes()).hexdigest()


def scenario_requests(z, delta, h: float, q: int):
    w = 1.5 + h * z[:q]
    delta_prefix = delta[:q]
    if len(w) != q or len(delta_prefix) != q or np.any((w < 0.0) | (w > 3.0)):
        raise ValueError("persisted request inputs do not cover the protocol prefix")
    return [(float(w[i]), float(delta_prefix[i])) for i in range(q)]


def source_hashes(root: Path, source_files: Sequence[str]) -> dict[str, str]:
    root = Path(root).resolve()
    return {relative: sha256_file(root / relative) for relative in source_files}


def write_source_snapshot(root: Path, out_dir: Path, hashes: Mapping[str, str]) -> None:
    root = Path(root).resolve()
    out_dir = Path(out_dir).resolve()
    snapshot_root = out_dir / "source_snapshot"
    for relative, expected in hashes.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"source path escapes project root: {relative!r}")
        snapshot = snapshot_root / relative_path
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative_path, snapshot)
        if sha256_file(snapshot) != expected:
            raise RuntimeError(f"source changed while snapshotting: {relative}")


def build_protocol(*, scales, q_values, scenarios_h, instances_per_scale,
                   repeats_by_q, formal_seed, active_seed, methods,
                   source_sha256, product_generation, request_generation,
                   timed_task, method_order, verification, workers=None,
                   pilot_seed=None, request_extension=None) -> dict:
    """Build one canonical protocol record for serial and parallel runners."""
    protocol = {
        "scales": tuple(scales),
        "q_values": tuple(q_values),
        "scenarios_h": dict(scenarios_h),
        "instances_per_scale": int(instances_per_scale),
        "repeats_by_q": {str(int(q)): int(repeats_by_q[q]) for q in q_values},
        "formal_seed": int(formal_seed),
        "active_seed": int(active_seed),
        "product_generation": product_generation,
        "request_generation": request_generation,
        "same_request_latents_across_scenarios": True,
        "nested_q_prefixes": True,
        "methods": tuple(methods),
        "timed_task": timed_task,
        "method_order": method_order,
        "verification": verification,
        "source_sha256": dict(source_sha256),
        "python": sys.version,
        "platform": platform.platform(),
    }
    if pilot_seed is not None:
        protocol["pilot_seed"] = int(pilot_seed)
    if workers is not None:
        protocol["workers"] = int(workers)
    if request_extension is not None:
        protocol["request_extension"] = request_extension
    return protocol


def expected_ledger_keys(protocol: Mapping) -> set[tuple]:
    scales = tuple(int(x) for x in protocol["scales"])
    q_values = tuple(int(x) for x in protocol["q_values"])
    scenarios = tuple(protocol["scenarios_h"])
    instances = int(protocol["instances_per_scale"])
    methods = tuple(protocol["methods"])
    repeats = {int(q): int(value) for q, value in protocol["repeats_by_q"].items()}
    return {
        (n, instance, scenario, q, method, repeat)
        for n in scales
        for instance in range(1, instances + 1)
        for scenario in scenarios
        for q in q_values
        for method in methods
        for repeat in range(1, repeats[q] + 1)
    }
