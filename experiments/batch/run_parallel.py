"""Parallel formal runner for Section 6.4 paired batch experiments."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import algorithms.fabs as fabs
from experiments.common import ObjectiveOracle
from experiments.batch.run_serial import (
    A0,
    OUT as DEFAULT_OUT,
    FORMAL_DATA as DEFAULT_DATA,
    FIELDS,
    SOURCE_FILES,
    FORMAL_SEED,
    METHODS,
    N_INSTANCES,
    Q_VALUES,
    SCENARIOS,
    SCALES,
    digest_products,
    digest_requests,
    scenario_requests,
    save_request_results,
    run_method as serial_run_method,
    verify_pair,
    write_manifest,
)
from experiments.batch.provenance import source_hashes, write_source_snapshot, build_protocol

ROOT = Path(__file__).resolve().parents[2]

_INPUT_CACHE = {}


def load_cached_inputs(data_dir, n, instance):
    key = (str(data_dir), n, instance)
    cached = _INPUT_CACHE.get(key)
    if cached is None:
        path = Path(data_dir) / f"n{n}_i{instance:02d}.npz"
        with np.load(path, allow_pickle=False) as saved:
            m = np.asarray(saved["m"], dtype=np.float64).copy()
            a = np.asarray(saved["a"], dtype=np.float64).copy()
            z = np.asarray(saved["z"], dtype=np.float64).copy()
            delta = np.asarray(saved["delta"], dtype=np.float64).copy()
        products = [fabs.Product(i + 1, float(a[i]), float(m[i])) for i in range(n)]
        cached = products, z, delta, digest_products({"m": m, "a": a})
        _INPUT_CACHE[key] = cached
    return cached


def run_job(job):
    out_dir = Path(job["out_dir"])
    data_dir = Path(job["data_dir"])
    n, instance, scenario_index = job["n"], job["instance"], job["scenario_index"]
    scenario, h, q, repeat, seed = (
        job["scenario"],
        job["h"],
        job["q"],
        job["repeat"],
        job["seed"],
    )
    products, z, delta, product_hash = load_cached_inputs(data_dir, n, instance)
    requests = scenario_requests(z, delta, h, q)
    request_hash = digest_requests(requests)
    oracles = [ObjectiveOracle(products, A0, w, d) for w, d in requests]
    reverse = (repeat + instance + scenario_index) % 2 == 0
    order = tuple(reversed(METHODS)) if reverse else METHODS
    observed = {}
    for method in order:
        observed[method] = serial_run_method(method, products, requests)

    fabs_values, fabs_sets, _, _ = observed["independent_fabs"]
    batch_values, batch_sets, batch, _ = observed["algorithm2"]
    value_diff, values_match, selections_match, oracle_error = verify_pair(
        products,
        requests,
        (fabs_values, fabs_sets),
        (batch_values, batch_sets),
        oracles,
    )

    solution_dir = (
        out_dir / "request_results" / f"n{n}_i{instance:02d}" / scenario / f"Q{q}"
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
            solution_dir / "algorithm2.json", requests, batch_values, batch_sets, batch
        )

    rows = []
    for method in METHODS:
        values, selections, result, elapsed = observed[method]
        batch_meta = result if method == "algorithm2" else None
        rows.append(
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
                "searches_H": "" if batch_meta is None else batch_meta.searches,
                "groups_G": "" if batch_meta is None else batch_meta.groups,
                "benchmarks_T": ""
                if batch_meta is None
                else batch_meta.benchmark_count,
                "pairs_M": "" if batch_meta is None else batch_meta.pair_count,
                "certificate_hits": ""
                if batch_meta is None
                else batch_meta.certificate_hits,
                "H_over_M": ""
                if batch_meta is None
                else batch_meta.searches / batch_meta.pair_count,
                "G_over_M": ""
                if batch_meta is None
                else batch_meta.groups / batch_meta.pair_count,
                "value_max_diff_from_fabs": value_diff,
                "objective_max_error": oracle_error,
                "all_objectives_match": int(values_match),
                "all_selections_match": int(selections_match),
                "product_sha256": product_hash,
                "request_sha256": request_hash,
                "seed": seed,
                "error": "",
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--instances", type=int, default=N_INSTANCES)
    parser.add_argument("--scales", type=int, nargs="*", default=list(SCALES))
    parser.add_argument("--q-values", type=int, nargs="*", default=list(Q_VALUES))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 10 or args.instances < 1 or args.repeats < 1:
        parser.error(
            "workers must be in [1,10]; instances and repeats must be positive"
        )
    scales, q_values = tuple(args.scales), tuple(args.q_values)
    if (
        not scales
        or not q_values
        or any(x < 1 for x in (*scales, *q_values))
        or len(set(scales)) != len(scales)
        or len(set(q_values)) != len(q_values)
    ):
        parser.error("scales and positive q-values must be nonempty")
    out_dir, data_dir = args.out_dir.resolve(), args.data_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"output directory must be empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(scales, args.instances, max(q_values), out_dir, data_dir)
    protocol = build_protocol(
        scales=scales,
        q_values=q_values,
        scenarios_h=SCENARIOS,
        instances_per_scale=args.instances,
        repeats_by_q={q: args.repeats for q in q_values},
        formal_seed=FORMAL_SEED,
        active_seed=FORMAL_SEED,
        methods=METHODS,
        source_sha256=source_hashes(ROOT, SOURCE_FILES),
        product_generation="m~U[1,100] raw float64; log(a)~U[-1,1]; a0=1",
        request_generation="z~U[-1,1], delta~U[1.5,3], w=1.5+h*z; full precision",
        workers=args.workers,
        timed_task="serial paired methods within each worker job; parallelism only across independent batches",
        method_order="alternates between paired methods by repeat and batch parity",
        verification="every timed batch checked against paired FABS and raw objective oracle outside reported runtimes",
        request_extension="Q<=300 preserves the original seeded sequence; Q=301..500 uses independent seeded suffix streams",
    )
    write_source_snapshot(ROOT, out_dir, protocol["source_sha256"])
    (out_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    jobs = [
        {
            "out_dir": str(out_dir),
            "data_dir": str(data_dir),
            "n": n,
            "instance": instance,
            "scenario_index": si,
            "scenario": scenario,
            "h": h,
            "q": q,
            "repeat": repeat,
            "seed": FORMAL_SEED,
        }
        for n in scales
        for instance in range(1, args.instances + 1)
        for si, (scenario, h) in enumerate(SCENARIOS.items())
        for q in q_values
        for repeat in range(1, args.repeats + 1)
    ]
    runs_path = out_dir / "runs_6_4.csv"
    with runs_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            completed = 0
            for offset in range(0, len(jobs), args.workers * 4):
                window = jobs[offset : offset + args.workers * 4]
                futures = {pool.submit(run_job, job): job for job in window}
                for future in as_completed(futures):
                    job = futures[future]
                    rows = future.result()
                    writer.writerows(rows)
                    completed += 1
                    if completed % args.workers == 0 or completed == len(jobs):
                        stream.flush()
                    print(
                        f"[{completed}/{len(jobs)}] n={job['n']} i={job['instance']:02d} "
                        f"{job['scenario']} Q={job['q']} rep={job['repeat']}",
                        flush=True,
                    )
    print(f"COMPLETE {runs_path}", flush=True)


if __name__ == "__main__":
    main()
