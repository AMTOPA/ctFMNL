"""Section 6.4: paired batches of independent FABS versus paper Algorithm 2."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import algorithms.fabs as fabs
from algorithms.spar import RepeatedOptimizer
from experiments.common import ObjectiveOracle, close_objectives
from data.synthetic import generate_synthetic_instance
from experiments.batch.provenance import (
    build_protocol, digest_products, digest_requests, scenario_requests,
    source_hashes, write_source_snapshot,
)

from data.batch import (
    A0, FORMAL_DATA, FORMAL_SEED, N_INSTANCES, PILOT_DATA, PILOT_SEED,
    Q_VALUES, SCALES, SCENARIOS, ensure_data, request_arrays,
)

ROOT = Path(__file__).resolve().parents[2]
OBJECTIVE_ERROR_TOL = 1e-6
REPEATS_FOR_SMALL_BATCH = 3
REPEATS_FOR_LARGE_BATCH = 3
OUT = ROOT / "results" / "batch"
METHODS = ("independent_fabs", "algorithm2")
SOURCE_FILES = (
    "data/batch.py",
    "experiments/batch/provenance.py",
    "algorithms/fabs.py",
    "algorithms/spar.py",
    "experiments/common.py",
    "experiments/batch/run_serial.py",
    "experiments/batch/run_parallel.py",
)
FIELDS = [
    "n",
    "instance",
    "scenario",
    "h",
    "Q",
    "method",
    "repeat",
    "runtime_s",
    "status",
    "searches_H",
    "groups_G",
    "benchmarks_T",
    "pairs_M",
    "certificate_hits",
    "H_over_M",
    "G_over_M",
    "value_max_diff_from_fabs",
    "objective_max_error",
    "all_objectives_match",
    "all_selections_match",
    "product_sha256",
    "request_sha256",
    "seed",
    "error",
]



def make_products(n: int, instance: int, seed: int):
    data = generate_synthetic_instance(n, instance, seed)
    products = [
        fabs.Product(i + 1, float(data["a"][i]), float(data["m"][i])) for i in range(n)
    ]
    return products, data


def run_method(method, products, requests):
    started = time.perf_counter()
    if method == "independent_fabs":
        values, selections = [], []
        for w, delta in requests:
            value, selected = fabs.solveCtFmnlNLog2N(products, A0, w, 1.0 + delta)
            values.append(float(value))
            selections.append(sorted(int(i) for i in selected))
        batch = None
    else:
        batch = RepeatedOptimizer(products, A0).solve(requests)
        values = [float(v) for v in batch.values]
        selections = batch.selected
    elapsed = time.perf_counter() - started
    return values, selections, batch, elapsed


def verify_pair(products, requests, fabs_result, batch_result, oracles=None):
    """Verify every request outside timing; reject partial/nonfinite outputs.

    Agreement between solvers is not enough: both returned assortments are
    evaluated independently from the model formula.
    """
    fabs_values, fabs_sets = fabs_result
    batch_values, batch_sets = batch_result
    count = len(requests)
    if not count or any(
        len(x) != count for x in (fabs_values, fabs_sets, batch_values, batch_sets)
    ):
        raise ValueError("request/result lengths must match and be nonempty")
    if not all(math.isfinite(v) for v in (*fabs_values, *batch_values)):
        raise ValueError("nonfinite solver objective")
    max_value_diff = max(abs(x - y) for x, y in zip(fabs_values, batch_values))
    all_values_match = all(
        close_objectives(x, y) for x, y in zip(fabs_values, batch_values)
    )
    max_error = 0.0
    if oracles is None:
        oracles = [ObjectiveOracle(products, A0, w, delta) for w, delta in requests]
    if len(oracles) != count:
        raise ValueError("oracle/request lengths must match")
    for k, oracle in enumerate(oracles):
        for values, sets in ((fabs_values, fabs_sets), (batch_values, batch_sets)):
            evaluated = oracle(sets[k])
            if not math.isfinite(evaluated):
                raise ValueError("nonfinite independent objective")
            max_error = max(max_error, abs(values[k] - evaluated))
    if not all_values_match or max_error > OBJECTIVE_ERROR_TOL:
        raise RuntimeError(
            f"objective verification failed: gap={max_value_diff}, "
            f"oracle_error={max_error}"
        )
    return max_value_diff, all_values_match, fabs_sets == batch_sets, max_error


def save_request_results(path, requests, values, selections, batch):
    payload = {
        "requests_w_delta": requests,
        "values": values,
        "selected_product_ids": selections,
        "best_benchmark_indices": None if batch is None else batch.best_benchmarks,
        "benefit_thresholds": None if batch is None else batch.thresholds.tolist(),
        "searches_H": None if batch is None else batch.searches,
        "groups_G": None if batch is None else batch.groups,
        "benchmarks_T": None if batch is None else batch.benchmark_count,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def write_manifest(scales, n_instances, q_max, out_dir, data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in scales:
        for instance in range(1, n_instances + 1):
            products, data = make_products(n, instance, FORMAL_SEED)
            z, delta = request_arrays(n, instance, FORMAL_SEED, q_max)
            path, digest = ensure_data(
                n, instance, data, z, delta, FORMAL_SEED, data_dir
            )
            rows.append(
                {
                    "n": n,
                    "instance": instance,
                    "seed": FORMAL_SEED,
                    "input_file": str(path.relative_to(ROOT)),
                    "input_sha256": digest,
                    "product_sha256": digest_products(data),
                    "request_seed": f"{FORMAL_SEED},640000,{n},{instance}",
                }
            )
    with (out_dir / "input_manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--formal", action="store_true")
    parser.add_argument("--instances", type=int, default=5)
    parser.add_argument("--scales", type=int, nargs="*", default=list(SCALES))
    parser.add_argument("--q-values", type=int, nargs="*", default=list(Q_VALUES))
    parser.add_argument("--repeats-small", type=int, default=REPEATS_FOR_SMALL_BATCH)
    parser.add_argument("--repeats-large", type=int, default=REPEATS_FOR_LARGE_BATCH)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--append",
        action="store_true",
        help="append only missing repeat keys to an existing ledger",
    )
    args = parser.parse_args()
    if args.instances < 1 or args.repeats_small < 1 or args.repeats_large < 1:
        parser.error("instances and repeats-small must be positive")

    out_dir = args.out_dir.resolve()
    data_dir = (args.data_dir or (FORMAL_DATA if args.formal else PILOT_DATA)).resolve()
    if args.append:
        parser.error("unsafe legacy append is disabled; use a new output directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    scales, q_values = tuple(args.scales), tuple(args.q_values)
    if not scales or not q_values or any(q < 1 for q in q_values):
        parser.error("scales and positive q-values must be nonempty")
    n_instances = N_INSTANCES if args.formal else args.instances
    seed = FORMAL_SEED if args.formal else PILOT_SEED
    q_max = max(q_values)
    repeats_by_q = {
        q: (args.repeats_small if q <= 20 else args.repeats_large) for q in q_values
    }
    output_name = "runs_6_4.csv" if args.formal else "pilot_runs.csv"
    runs_path = out_dir / output_name
    if runs_path.exists():
        raise RuntimeError(
            f"Output already exists; refusing to mix protocols: {runs_path}"
        )

    protocol = build_protocol(
        scales=scales,
        q_values=q_values,
        scenarios_h=SCENARIOS,
        instances_per_scale=n_instances,
        repeats_by_q=repeats_by_q,
        formal_seed=FORMAL_SEED,
        pilot_seed=PILOT_SEED,
        active_seed=seed,
        methods=METHODS,
        source_sha256=source_hashes(ROOT, SOURCE_FILES),
        product_generation="m~U[1,100] raw float64; log(a)~U[-1,1]; a0=1",
        request_generation="z~U[-1,1], delta~U[1.5,3], w=1.5+h*z; full precision",
        timed_task="serial full batch, all values and assortments; includes method preprocessing, grouping, searches, certificate checks and recovery; excludes I/O and independent verification",
        method_order="alternates between paired methods by repeat and batch parity",
        verification="every repetition, outside timed regions, compared to FABS and raw objective oracle",
        request_extension="Q<=300 preserves the original seeded sequence; Q=301..500 uses independent seeded suffix streams",
    )
    protocol_path = out_dir / (
        "protocol.json" if args.formal else "pilot_protocol.json"
    )
    write_source_snapshot(ROOT, out_dir, protocol["source_sha256"])
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    if args.formal:
        write_manifest(scales, n_instances, q_max, out_dir, data_dir)

    with runs_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for n in scales:
            for instance in range(1, n_instances + 1):
                products, data = make_products(n, instance, seed)
                z, delta = request_arrays(n, instance, seed, q_max)
                ensure_data(n, instance, data, z, delta, seed, data_dir)
                product_hash = digest_products(data)
                for scenario_index, (scenario, h) in enumerate(SCENARIOS.items()):
                    for q in q_values:
                        requests = scenario_requests(z, delta, h, q)
                        request_hash = digest_requests(requests)
                        oracles = [
                            ObjectiveOracle(products, A0, w, d) for w, d in requests
                        ]
                        solution_dir = (
                            out_dir
                            / "request_results"
                            / f"n{n}_i{instance:02d}"
                            / scenario
                            / f"Q{q}"
                        )
                        for repeat in range(1, repeats_by_q[q] + 1):
                            reverse = (repeat + instance + scenario_index) % 2 == 0
                            order = tuple(reversed(METHODS)) if reverse else METHODS
                            observed = {}
                            for method in order:
                                values, selections, batch, elapsed = run_method(
                                    method, products, requests
                                )
                                observed[method] = (values, selections, batch, elapsed)
                            fabs_values, fabs_sets, _, _ = observed["independent_fabs"]
                            batch_values, batch_sets, batch, _ = observed["algorithm2"]
                            max_diff, values_match, sets_match, oracle_error = (
                                verify_pair(
                                    products,
                                    requests,
                                    (fabs_values, fabs_sets),
                                    (batch_values, batch_sets),
                                    oracles,
                                )
                            )
                            if repeat == 1:
                                save_request_results(
                                    solution_dir / "independent_fabs.json",
                                    requests,
                                    fabs_values,
                                    fabs_sets,
                                    None,
                                )
                                save_request_results(
                                    solution_dir / "algorithm2.json",
                                    requests,
                                    batch_values,
                                    batch_sets,
                                    batch,
                                )
                            for method in METHODS:
                                values, selections, result, elapsed = observed[method]
                                is_batch = method == "algorithm2"
                                batch_meta = result if is_batch else None
                                writer.writerow(
                                    {
                                        "n": n,
                                        "instance": instance,
                                        "scenario": scenario,
                                        "h": h,
                                        "Q": q,
                                        "method": method,
                                        "repeat": repeat,
                                        "runtime_s": elapsed,
                                        "status": "ok",
                                        "searches_H": ""
                                        if batch_meta is None
                                        else batch_meta.searches,
                                        "groups_G": ""
                                        if batch_meta is None
                                        else batch_meta.groups,
                                        "benchmarks_T": ""
                                        if batch_meta is None
                                        else batch_meta.benchmark_count,
                                        "pairs_M": ""
                                        if batch_meta is None
                                        else batch_meta.pair_count,
                                        "certificate_hits": ""
                                        if batch_meta is None
                                        else batch_meta.certificate_hits,
                                        "H_over_M": ""
                                        if batch_meta is None
                                        else batch_meta.searches
                                        / batch_meta.pair_count,
                                        "G_over_M": ""
                                        if batch_meta is None
                                        else batch_meta.groups / batch_meta.pair_count,
                                        "value_max_diff_from_fabs": max_diff,
                                        "objective_max_error": oracle_error,
                                        "all_objectives_match": int(values_match),
                                        "all_selections_match": int(sets_match),
                                        "product_sha256": product_hash,
                                        "request_sha256": request_hash,
                                        "seed": seed,
                                        "error": "",
                                    }
                                )
                            stream.flush()
                            print(
                                f"n={n} i={instance:02d} {scenario} Q={q} rep={repeat}: "
                                f"FABS={observed['independent_fabs'][3]:.3f}s, "
                                f"Algorithm2={observed['algorithm2'][3]:.3f}s, "
                                f"H/M={batch.searches}/{batch.pair_count}",
                                flush=True,
                            )
    print(f"COMPLETE {runs_path}", flush=True)


if __name__ == "__main__":
    main()
