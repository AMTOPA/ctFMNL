"""Read-only provenance audit; never regenerate or rewrite measured results."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from experiments.batch.provenance import (
    digest_products, digest_requests, expected_ledger_keys,
    scenario_requests, sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
_SAMPLE_LIMIT = 10



def _positive_int(value, label: str, errors: list[str]) -> int | None:
    if isinstance(value, bool):
        errors.append(f"{label} must be a positive integer")
        return None
    if isinstance(value, Integral):
        parsed = int(value)
    elif isinstance(value, Real) and math.isfinite(float(value)) and float(value).is_integer():
        parsed = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text or not text.isdecimal():
            errors.append(f"{label} must be a positive integer")
            return None
        parsed = int(text)
    else:
        errors.append(f"{label} must be a positive integer")
        return None
    if parsed < 1:
        errors.append(f"{label} must be a positive integer")
        return None
    return parsed


def protocol_grid(protocol: object):
    errors: list[str] = []
    if not isinstance(protocol, dict):
        return None, ["protocol must be a JSON object"]

    def unique_positive_list(name: str):
        raw = protocol.get(name)
        if not isinstance(raw, (list, tuple)) or not raw:
            errors.append(f"protocol {name} must be a nonempty list")
            return ()
        values = []
        for index, value in enumerate(raw):
            parsed = _positive_int(value, f"protocol {name}[{index}]", errors)
            if parsed is not None:
                values.append(parsed)
        if len(set(values)) != len(values):
            errors.append(f"protocol {name} contains duplicates")
        return tuple(values)

    scales = unique_positive_list("scales")
    q_values = unique_positive_list("q_values")
    instances = _positive_int(
        protocol.get("instances_per_scale"), "protocol instances_per_scale", errors
    )

    raw_scenarios = protocol.get("scenarios_h")
    scenarios = ()
    if not isinstance(raw_scenarios, dict) or not raw_scenarios:
        errors.append("protocol scenarios_h must be a nonempty object")
    else:
        scenarios = tuple(raw_scenarios)
        for name, h in raw_scenarios.items():
            if not isinstance(name, str) or not name.strip():
                errors.append("protocol scenario names must be nonempty strings")
            if isinstance(h, bool) or not isinstance(h, Real) or not math.isfinite(float(h)):
                errors.append(f"protocol scenario {name!r} has invalid h")

    raw_methods = protocol.get("methods")
    methods = ()
    if not isinstance(raw_methods, (list, tuple)) or not raw_methods:
        errors.append("protocol methods must be a nonempty list")
    else:
        methods = tuple(raw_methods)
        if any(not isinstance(method, str) or not method.strip() for method in methods):
            errors.append("protocol methods must be nonempty strings")
        if len(set(methods)) != len(methods):
            errors.append("protocol methods contains duplicates")

    raw_repeats = protocol.get("repeats_by_q")
    repeats: dict[int, int] = {}
    if not isinstance(raw_repeats, dict) or not raw_repeats:
        errors.append("protocol repeats_by_q must be a nonempty object")
    else:
        for raw_q, raw_repeat in raw_repeats.items():
            q = _positive_int(raw_q, f"protocol repeats_by_q key {raw_q!r}", errors)
            repeat = _positive_int(
                raw_repeat, f"protocol repeats_by_q[{raw_q!r}]", errors
            )
            if q is not None and repeat is not None:
                if q in repeats:
                    errors.append(f"protocol repeats_by_q has duplicate normalized Q={q}")
                repeats[q] = repeat
        if set(repeats) != set(q_values):
            errors.append("protocol q_values/repeats_by_q grid mismatch")

    raw_sources = protocol.get("source_sha256")
    if not isinstance(raw_sources, dict) or not raw_sources:
        errors.append("protocol source_sha256 must be a nonempty object")
    else:
        for relative, digest in raw_sources.items():
            if not isinstance(relative, str) or not relative.strip():
                errors.append("protocol source paths must be nonempty strings")
            if not isinstance(digest, str) or len(digest) != 64:
                errors.append(f"protocol source digest is invalid for {relative!r}")

    if errors:
        return None, errors
    expected = expected_ledger_keys(protocol)
    return {
        "scales": scales,
        "q_values": q_values,
        "scenarios": scenarios,
        "instances": instances,
        "methods": methods,
        "repeats": repeats,
        "expected": expected,
    }, []


def _contained_path(base: Path, relative: object) -> Path | None:
    if not isinstance(relative, str) or not relative.strip():
        return None
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        return None
    base = base.resolve()
    candidate = (base / path).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return None
    return candidate


def _key_example(key):
    return list(key)


def audit_results(out_dir: Path, rows=None, protocol=None) -> dict:
    """Validate the complete ledger and its persisted input/source provenance."""
    out_dir = Path(out_dir)
    protocol_errors: list[str] = []
    if protocol is None:
        try:
            protocol = json.loads((out_dir / "protocol.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            protocol = {}
            protocol_errors.append(f"cannot read protocol.json: {exc}")
    grid, grid_errors = protocol_grid(protocol)
    protocol_errors.extend(grid_errors)

    manifest_errors: list[str] = []
    try:
        with (out_dir / "input_manifest.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            manifest = list(csv.DictReader(stream))
    except OSError as exc:
        manifest = []
        manifest_errors.append(f"cannot read input_manifest.csv: {exc}")

    if rows is None:
        try:
            with (out_dir / "runs_6_4.csv").open(
                encoding="utf-8-sig", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
        except OSError as exc:
            rows = []
            manifest_errors.append(f"cannot read runs_6_4.csv: {exc}")
    else:
        rows = list(rows)

    expected_input_keys = set()
    if grid is not None:
        expected_input_keys = {
            (n, instance)
            for n in grid["scales"]
            for instance in range(1, grid["instances"] + 1)
        }

    root = ROOT.resolve()
    snapshot_root = (out_dir / "source_snapshot").resolve()
    inputs = {}
    manifest_keys = set()
    manifest_duplicates = []
    input_path_errors = []
    input_file_hash_errors = []
    input_product_hash_mismatches = []
    for index, item in enumerate(manifest, start=2):
        if not isinstance(item, dict):
            manifest_errors.append(f"manifest row {index} is not a record")
            continue
        parse_errors: list[str] = []
        n = _positive_int(item.get("n"), f"manifest row {index} n", parse_errors)
        instance = _positive_int(
            item.get("instance"), f"manifest row {index} instance", parse_errors
        )
        if parse_errors:
            manifest_errors.extend(parse_errors)
            continue
        key = (n, instance)
        if key in manifest_keys:
            manifest_duplicates.append(_key_example(key))
            continue
        manifest_keys.add(key)

        relative = item.get("input_file")
        path = _contained_path(root, relative)
        if path is None:
            input_path_errors.append(str(relative))
            continue
        try:
            if not path.is_file() or sha256_file(path) != item.get("input_sha256"):
                input_file_hash_errors.append(str(relative))
                continue
            with np.load(path, allow_pickle=False) as data:
                z = np.asarray(data["z"]).copy()
                delta = np.asarray(data["delta"]).copy()
                product_hash = digest_products(data)
        except (OSError, ValueError, KeyError) as exc:
            input_file_hash_errors.append(f"{relative}: {exc}")
            continue
        if item.get("product_sha256") != product_hash:
            input_product_hash_mismatches.append(str(relative))
        inputs[key] = (z, delta, product_hash)

    missing_input_keys = expected_input_keys - manifest_keys
    extra_input_keys = manifest_keys - expected_input_keys if grid is not None else manifest_keys

    expected_keys = set() if grid is None else grid["expected"]
    ledger_keys = set()
    duplicate_keys = []
    ledger_row_errors = []
    ledger_status_errors = []
    expected_hashes = {}
    request_errors, product_errors = Counter(), 0
    unverifiable_rows = []
    for index, row in enumerate(rows, start=2):
        if not isinstance(row, dict):
            ledger_row_errors.append(f"ledger row {index} is not a record")
            continue
        parse_errors: list[str] = []
        n = _positive_int(row.get("n"), f"ledger row {index} n", parse_errors)
        instance = _positive_int(
            row.get("instance"), f"ledger row {index} instance", parse_errors
        )
        q = _positive_int(row.get("Q"), f"ledger row {index} Q", parse_errors)
        repeat = _positive_int(
            row.get("repeat"), f"ledger row {index} repeat", parse_errors
        )
        scenario, method = row.get("scenario"), row.get("method")
        if not isinstance(scenario, str) or not scenario:
            parse_errors.append(f"ledger row {index} has invalid scenario")
        if not isinstance(method, str) or not method:
            parse_errors.append(f"ledger row {index} has invalid method")
        if parse_errors:
            ledger_row_errors.extend(parse_errors)
            continue
        key = (n, instance, scenario, q, method, repeat)
        if key in ledger_keys:
            duplicate_keys.append(_key_example(key))
        ledger_keys.add(key)
        if row.get("status") != "ok":
            ledger_status_errors.append(
                {"row": index, "key": _key_example(key), "status": row.get("status")}
            )

        if grid is not None and key in expected_keys:
            if (n, instance) not in inputs:
                unverifiable_rows.append(_key_example(key))
                continue
            z, delta, product_hash = inputs[n, instance]
            request_key = (n, instance, scenario, q)
            try:
                if request_key not in expected_hashes:
                    h = float(protocol["scenarios_h"][scenario])
                    expected_hashes[request_key] = digest_requests(
                        scenario_requests(z, delta, h, q)
                    )
                request_errors[str(q)] += int(
                    row.get("request_sha256") != expected_hashes[request_key]
                )
                product_errors += int(row.get("product_sha256") != product_hash)
            except (IndexError, KeyError, TypeError, ValueError) as exc:
                unverifiable_rows.append([*_key_example(key), str(exc)])

    missing_keys = expected_keys - ledger_keys
    extra_keys = ledger_keys - expected_keys if grid is not None else ledger_keys

    source_path_errors = []
    source_errors = []
    raw_sources = protocol.get("source_sha256", {}) if isinstance(protocol, dict) else {}
    if isinstance(raw_sources, dict):
        for relative, digest in raw_sources.items():
            snapshot = _contained_path(snapshot_root, relative)
            if snapshot is None:
                source_path_errors.append(str(relative))
                continue
            try:
                if not snapshot.is_file() or sha256_file(snapshot) != digest:
                    source_errors.append(str(relative))
            except OSError:
                source_errors.append(str(relative))

    report = {
        "rows": len(rows),
        "expected_rows": len(expected_keys),
        "protocol_errors": protocol_errors,
        "manifest_errors": manifest_errors,
        "manifest_missing_key_count": len(missing_input_keys),
        "manifest_missing_key_examples": [
            _key_example(key) for key in sorted(missing_input_keys)[:_SAMPLE_LIMIT]
        ],
        "manifest_extra_key_count": len(extra_input_keys),
        "manifest_extra_key_examples": [
            _key_example(key) for key in sorted(extra_input_keys)[:_SAMPLE_LIMIT]
        ],
        "manifest_duplicate_keys": manifest_duplicates[:_SAMPLE_LIMIT],
        "input_path_errors": input_path_errors,
        "input_file_hash_errors": input_file_hash_errors,
        "input_product_hash_mismatches": input_product_hash_mismatches,
        "ledger_empty": not rows,
        "ledger_missing_key_count": len(missing_keys),
        "ledger_missing_key_examples": [
            _key_example(key) for key in sorted(missing_keys)[:_SAMPLE_LIMIT]
        ],
        "ledger_extra_key_count": len(extra_keys),
        "ledger_extra_key_examples": [
            _key_example(key) for key in sorted(extra_keys)[:_SAMPLE_LIMIT]
        ],
        "ledger_duplicate_keys": duplicate_keys[:_SAMPLE_LIMIT],
        "ledger_row_errors": ledger_row_errors,
        "ledger_status_errors": ledger_status_errors[:_SAMPLE_LIMIT],
        "unverifiable_rows": unverifiable_rows[:_SAMPLE_LIMIT],
        "request_hash_mismatches_by_q": dict(
            sorted(request_errors.items(), key=lambda item: int(item[0]))
        ),
        "product_hash_mismatches": product_errors,
        "source_path_errors": source_path_errors,
        "source_hash_mismatches": source_errors,
    }
    failures = (
        protocol_errors
        or manifest_errors
        or missing_input_keys
        or extra_input_keys
        or manifest_duplicates
        or input_path_errors
        or input_file_hash_errors
        or input_product_hash_mismatches
        or not rows
        or missing_keys
        or extra_keys
        or duplicate_keys
        or ledger_row_errors
        or ledger_status_errors
        or unverifiable_rows
        or sum(request_errors.values())
        or product_errors
        or source_path_errors
        or source_errors
    )
    report["provenance_ok"] = not bool(failures)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--report", type=Path, help="write an additional audit report, not original data")
    args = parser.parse_args()
    report = audit_results(args.out_dir)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(0 if report["provenance_ok"] else 1)


if __name__ == "__main__":
    main()

