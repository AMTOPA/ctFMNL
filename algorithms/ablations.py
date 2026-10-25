"""ctFMNL ablation solvers for Section 6.3.

Variants retain the same incremental attractiveness/focal-exit sweep and model
semantics. They differ only in aggregate queries/root search. ``delta`` is the
model parameter (the corresponding attraction multiplier is ``1 + delta``).
Each public solver returns ``(objective, assortment, telemetry)``.
"""
from __future__ import annotations

import bisect
import math
import random
import time
from itertools import islice
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from algorithms.fabs import FenwickTree, validate_parameters
from experiments.common import objective as _objective


@dataclass(frozen=True)
class _Item:
    product_id: Any
    a: float
    m: float


def _items(products: Iterable[Any]) -> list[_Item]:
    result, seen = [], set()
    for p in products:
        pid = getattr(p, "productId", getattr(p, "product_id", None))
        a, m = float(p.a), float(p.m)
        if pid is None:
            raise ValueError("each product must expose productId or product_id")
        if pid in seen:
            raise ValueError(f"duplicate product id: {pid!r}")
        if not math.isfinite(a) or a <= 0 or not math.isfinite(m) or m < 0:
            raise ValueError("products require finite a > 0 and finite m >= 0")
        seen.add(pid)
        result.append(_Item(pid, a, m))
    return result


def verify_objective(products: Iterable[Any], assortment: Iterable[Any],
                     a0: float, w: float, delta: float) -> float:
    """Backward-compatible access to the shared independent objective oracle."""
    return _objective(products, assortment, a0, w, delta)


class _Fenwick:
    """Zero-based adapter around the shared one-based Fenwick implementation."""

    def __init__(self, values: Sequence[float]):
        self.tree = FenwickTree(len(values))
        self.tree.build(values)

    def add(self, index: int, amount: float) -> None:
        self.tree.add(index + 1, amount)

    def prefix(self, end: int) -> float:
        return self.tree.sum(end)

    def suffix(self, start: int) -> float:
        return self.tree.sum(self.tree.size) - self.tree.sum(start)


def _solve(products, a0, w, delta, method: str, search: str = "bisection",
           construct_each_benchmark: bool = False, max_iterations=1000,
           objective_tolerance=1e-6):
    """Run one implementation variant with an explicit candidate-build policy.

    ``method`` controls aggregate queries (direct scans, rebuilt Fenwick, or
    incrementally maintained Fenwick); ``search`` controls breakpoint bisection
    versus Newton--Dinkelbach.  Candidate assortment construction is kept
    separate so the three component ablations can all build it once after the
    best benchmark is known, while ``candidate_each_benchmark`` measures the
    otherwise identical per-benchmark construction overhead.
    """
    started = time.perf_counter()
    items = _items(products)
    ordered_items = sorted(items, key=lambda x: (x.a, str(x.product_id)))
    a0, w, delta = float(a0), float(w), float(delta)
    validate_parameters(a0, w, delta)
    if search not in ("bisection", "dinkelbach"):
        raise ValueError("search must be bisection or dinkelbach")
    if max_iterations < 1 or objective_tolerance < 0:
        raise ValueError("max_iterations must be positive and objective_tolerance nonnegative")
    q = 1.0 + delta
    marks = sorted({p.m for p in items})
    rank = {m: i for i, m in enumerate(marks)}
    groups: dict[float, list[_Item]] = {}
    for p in ordered_items:
        groups.setdefault(p.a, []).append(p)
    live_a, live_am = [0.0] * len(marks), [0.0] * len(marks)
    live: list[_Item] = []
    focal_exits = sorted(items, key=lambda p: ((1.0 + w) * p.a, str(p.product_id)))
    focal_exit_index = 0
    fenwick_a = fenwick_am = None
    best_value, best_ids = 0.0, []
    best_candidate: _Item | None = None
    best_rho = 0.0
    telemetry = {"method": method, "search": search, "benchmarks": 0,
                 "breakpoint_probes": 0, "root_iterations": 0,
                 "fenwick_builds": 0, "candidate_constructions": 0,
                 "iteration_cap_hits": 0, "root_cycles": 0,
                 "max_normalized_root_residual": 0.0}

    def change(p: _Item, focal: bool, sign: float) -> None:
        wa = p.a * (q if focal else 1.0)
        r = rank[p.m]
        live_a[r] += sign * wa
        live_am[r] += sign * p.m * wa
        if method == "fenwick" and fenwick_a is not None:
            fenwick_a.add(r, sign * wa)
            fenwick_am.add(r, sign * p.m * wa)

    def aggregate_for(candidate: _Item | None):
        if candidate is not None:
            change(candidate, True, -1.0)
        if method == "rebuild":
            telemetry["fenwick_builds"] += 1
            fa, fam = _Fenwick(live_a), _Fenwick(live_am)
            def query(k):
                return fa.suffix(k), fam.suffix(k)
        elif method == "fenwick":
            def query(k):
                return fenwick_a.suffix(k), fenwick_am.suffix(k)
        else:
            def query(k):
                return sum(islice(live_a, k, None)), sum(islice(live_am, k, None))
        return query

    def restore_candidate(candidate: _Item | None) -> None:
        if candidate is not None:
            change(candidate, True, 1.0)

    def build_from_live(candidate: _Item | None, rho: float) -> list[Any]:
        telemetry["candidate_constructions"] += 1
        selected = [p.product_id for p in live if p.m > rho and p is not candidate]
        if candidate is not None:
            selected.append(candidate.product_id)
        return selected

    def build_final(candidate: _Item | None, rho: float) -> list[Any]:
        telemetry["candidate_constructions"] += 1
        benchmark_a = a0 if candidate is None else candidate.a
        selected = [p.product_id for p in ordered_items
                    if p.a <= benchmark_a and p.m > rho and p is not candidate]
        if candidate is not None:
            selected.append(candidate.product_id)
        return selected

    def assess(candidate: _Item | None, effective_a0: float) -> None:
        nonlocal best_value, best_ids, best_candidate, best_rho
        telemetry["benchmarks"] += 1
        query = aggregate_for(candidate)
        ca = candidate.a * q if candidate else 0.0
        cam = candidate.m * ca if candidate else 0.0
        if search == "bisection":
            lo, hi, last = 0, len(marks), 0
            while lo <= hi:
                mid = (lo + hi) // 2
                rho = 0.0 if mid == 0 else marks[mid - 1]
                sa, sam = query(mid)
                f = sam + cam - rho * (effective_a0 + sa + ca)
                telemetry["breakpoint_probes"] += 1
                if f >= 0.0:
                    last, lo = mid, mid + 1
                else:
                    hi = mid - 1
            rho = 0.0 if last == 0 else marks[last - 1]
            sa, sam = query(last)
            value = (sam + cam) / (effective_a0 + sa + ca)
            chosen_rho = rho
        else:
            rho, seen_k = 0.0, set()
            root_best_value, root_best_rho = -math.inf, 0.0
            for _ in range(max_iterations):
                k = bisect.bisect_right(marks, rho)
                sa, sam = query(k)
                denom, numer = effective_a0 + sa + ca, sam + cam
                new_rho = numer / denom
                residual = numer - rho * denom
                telemetry["root_iterations"] += 1
                if new_rho > root_best_value:
                    root_best_value = new_rho
                    root_best_rho = rho
                d_min = effective_a0 + ca
                normalized_residual = abs(residual) / d_min
                telemetry["max_normalized_root_residual"] = max(
                    telemetry.get("max_normalized_root_residual", 0.0),
                    normalized_residual)
                new_k = bisect.bisect_right(marks, new_rho)
                if new_k == k and normalized_residual <= objective_tolerance:
                    break
                if new_k in seen_k:
                    telemetry["root_cycles"] += 1
                    break
                seen_k.add(k)
                rho = new_rho
            else:
                telemetry["iteration_cap_hits"] += 1
            value = max(root_best_value, 0.0)
            chosen_rho = root_best_rho
        chosen = (build_from_live(candidate, chosen_rho)
                  if construct_each_benchmark else None)
        if value > best_value:
            best_value = value
            best_candidate = candidate
            best_rho = chosen_rho
            if chosen is not None:
                best_ids = chosen
        restore_candidate(candidate)

    for p in items:
        if p.a <= a0:
            live.append(p)
            focal = (1.0 + w) * p.a >= a0
            change(p, focal, 1.0)
    if method == "fenwick":
        fenwick_a, fenwick_am = _Fenwick(live_a), _Fenwick(live_am)
        telemetry["fenwick_builds"] += 1
    while (focal_exit_index < len(focal_exits)
           and (1.0 + w) * focal_exits[focal_exit_index].a < a0):
        focal_exit_index += 1
    assess(None, a0 * (q if (1.0 + w) * a0 >= a0 else 1.0))

    for a_t, group in sorted(groups.items()):
        if a_t < a0:
            continue
        if a_t > a0:
            for p in group:
                live.append(p)
                change(p, True, 1.0)
            while (focal_exit_index < len(focal_exits)
                   and (1.0 + w) * focal_exits[focal_exit_index].a < a_t):
                p = focal_exits[focal_exit_index]
                change(p, True, -1.0)
                change(p, False, 1.0)
                focal_exit_index += 1
        for candidate in group:
            effective_a0 = a0 * (q if (1.0 + w) * a0 >= a_t else 1.0)
            assess(candidate, effective_a0)

    if not construct_each_benchmark:
        best_ids = build_final(best_candidate, best_rho)
    telemetry["elapsed_seconds"] = time.perf_counter() - started
    return best_value, best_ids, telemetry

def solve_direct_suffix(products, a0, w, delta, **kwargs):
    """Ablation 1: direct suffix sums in place of Fenwick suffix queries."""
    return _solve(products, a0, w, delta, "direct", **kwargs)


def solve_rebuild_fenwick(products, a0, w, delta, **kwargs):
    """Ablation 2: linear Fenwick builds for each benchmark, then bisection."""
    return _solve(products, a0, w, delta, "rebuild", **kwargs)


def solve_newton_dinkelbach(products, a0, w, delta, **kwargs):
    """Ablation 3: incremental Fenwick queries with root iterations."""
    return _solve(products, a0, w, delta, "fenwick", search="dinkelbach", **kwargs)


def solve_candidate_each_benchmark(products, a0, w, delta, **kwargs):
    """Extra fairness audit: FABS queries, but build candidates at every benchmark."""
    return _solve(products, a0, w, delta, "fenwick", search="bisection",
                  construct_each_benchmark=True, **kwargs)


def solve(products, a0, w, delta, variant="direct", **kwargs):
    """Common interface for the three fair variants and the candidate-build audit."""
    solvers = {"direct": solve_direct_suffix,
               "rebuild": solve_rebuild_fenwick,
               "newton_dinkelbach": solve_newton_dinkelbach,
               "candidate_each_benchmark": solve_candidate_each_benchmark}
    try:
        solver = solvers[variant]
    except KeyError as exc:
        raise ValueError(f"unknown variant {variant!r}; choose {tuple(solvers)}") from exc
    return solver(products, a0, w, delta, **kwargs)


def _self_check():
    """Small randomized tests against independent enumeration and Algorithm A.1."""
    from algorithms.guan_a1 import Product, solveCtFmnlAlgorithmA1

    def enumerate_opt(ps, a0, w, delta):
        best = 0.0
        for mask in range(1 << len(ps)):
            ids = [p.productId for i, p in enumerate(ps) if mask >> i & 1]
            best = max(best, verify_objective(ps, ids, a0, w, delta))
        return best

    rng = random.Random(20260929)
    cases = [
        ([Product(1, 1, 0), Product(2, 1, 0)], 1, 0, 0),
        ([Product(1, 2, 3), Product(2, 1, 1), Product(3, 3, 2)], 4, 0.5, 1.2),
        ([Product(1, 1, 2), Product(2, 2, 2), Product(3, 3, 2)], 1, 0, 0.5),
    ]
    for _ in range(35):
        n = rng.randrange(0, 9)
        ps = [Product(i + 1, rng.choice([0.5, 1, 2, 3]), rng.choice([0, 0.2, 1, 2, 4]))
              for i in range(n)]
        cases.append((ps, rng.choice([0.5, 1, 2]), rng.choice([0, 0.25, 1]),
                      rng.choice([0, 0.1, 0.8, 2])))
    for ps, a0, w, delta in cases:
        oracle = enumerate_opt(ps, a0, w, delta)
        a1 = solveCtFmnlAlgorithmA1(ps, a0, w, delta)["revenue"]
        assert math.isclose(a1, oracle, rel_tol=1e-9, abs_tol=1e-11), ("A.1", ps, oracle, a1)
        for solver in (solve_direct_suffix, solve_rebuild_fenwick, solve_newton_dinkelbach):
            value, ids, telemetry = solver(ps, a0, w, delta)
            checked = verify_objective(ps, ids, a0, w, delta)
            assert math.isclose(value, oracle, rel_tol=1e-9, abs_tol=1e-11), (solver.__name__, ps, oracle, value, ids)
            assert math.isclose(checked, value, rel_tol=1e-9, abs_tol=1e-11), (solver.__name__, checked, value)
            assert telemetry["iteration_cap_hits"] == 0, (solver.__name__, telemetry)
    print(f"self-check passed: {len(cases)} instances x 3 ablations")


if __name__ == "__main__":
    _self_check()



