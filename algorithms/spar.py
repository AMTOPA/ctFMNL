"""Reference implementation of the paper's Algorithm 2 for repeated ctFMNL solves.

The implementation follows Section 5.3: persistent prefix versions over the
attractiveness order provide base and nonfocal suffix queries; requests are
grouped by their focal membership at each benchmark; interval certificates
avoid a new breakpoint search when the cached canonical assortment remains
optimal for the requested delta.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from algorithms.fabs import Product, validate_parameters, validate_products


@dataclass(slots=True)
class _Node:
    left: int = 0
    right: int = 0
    sum_a: float = 0.0
    sum_ma: float = 0.0


class _PersistentBenefitTree:
    def __init__(self, size: int):
        self.size = max(1, int(size))
        self.nodes = [_Node()]

    def _clone(self, index: int) -> int:
        old = self.nodes[index]
        self.nodes.append(_Node(old.left, old.right, old.sum_a, old.sum_ma))
        return len(self.nodes) - 1

    def update(self, previous: int, position: int, a: float, ma: float) -> int:
        def rec(old: int, lo: int, hi: int) -> int:
            new = self._clone(old)
            node = self.nodes[new]
            node.sum_a += a
            node.sum_ma += ma
            if lo != hi:
                mid = (lo + hi) // 2
                if position <= mid:
                    node.left = rec(node.left, lo, mid)
                else:
                    node.right = rec(node.right, mid + 1, hi)
            return new
        return rec(previous, 0, self.size - 1)

    def suffix(self, root: int, first: int) -> tuple[float, float]:
        """Return (sum a, sum m*a) for benefit ranks >= first."""
        if first <= 0:
            node = self.nodes[root]
            return node.sum_a, node.sum_ma
        if first >= self.size:
            return 0.0, 0.0

        def rec(index: int, lo: int, hi: int) -> tuple[float, float]:
            if index == 0 or hi < first:
                return 0.0, 0.0
            if first <= lo:
                node = self.nodes[index]
                return node.sum_a, node.sum_ma
            mid = (lo + hi) // 2
            left = rec(self.nodes[index].left, lo, mid)
            right = rec(self.nodes[index].right, mid + 1, hi)
            return left[0] + right[0], left[1] + right[1]
        return rec(root, 0, self.size - 1)

    def last_present_at_or_before(self, root: int, last: int) -> int | None:
        if root == 0 or last < 0:
            return None
        last = min(last, self.size - 1)
        def rec(index: int, lo: int, hi: int) -> int | None:
            if index == 0 or lo > last:
                return None
            if lo == hi:
                return lo
            mid = (lo + hi) // 2
            result = rec(self.nodes[index].right, mid + 1, hi)
            return result if result is not None else rec(self.nodes[index].left, lo, mid)
        return rec(root, 0, self.size - 1)

    def first_present_at_or_after(self, root: int, first: int) -> int | None:
        if root == 0 or first >= self.size:
            return None
        first = max(first, 0)
        def rec(index: int, lo: int, hi: int) -> int | None:
            if index == 0 or hi < first:
                return None
            if lo == hi:
                return lo
            mid = (lo + hi) // 2
            result = rec(self.nodes[index].left, lo, mid)
            return result if result is not None else rec(self.nodes[index].right, mid + 1, hi)
        return rec(root, 0, self.size - 1)


@dataclass
class _Certificate:
    threshold: float
    lower: float
    upper: float
    n0: float
    nf: float
    d0: float
    df: float

    def contains(self, delta: float) -> bool:
        denominator = self.d0 + delta * self.df
        if denominator <= 0.0:
            return False
        value = (self.n0 + delta * self.nf) / denominator
        return self.lower <= value < self.upper

    def value(self, delta: float) -> float:
        return (self.n0 + delta * self.nf) / (self.d0 + delta * self.df)


@dataclass
class _Benchmark:
    product: Product | None
    benchmark_a: float
    candidate_count: int


@dataclass
class BatchResult:
    values: np.ndarray
    selected: list[list[int]]
    best_benchmarks: list[int | None]
    thresholds: np.ndarray
    searches: int
    groups: int
    benchmark_count: int
    pair_count: int
    certificate_hits: int


class RepeatedOptimizer:
    """Exact repeated optimizer implementing Algorithm 2 for arbitrary w_q."""

    def __init__(self, products: Sequence[Product], a0: float = 1.0):
        self.products = list(products)
        self.a0 = float(a0)
        validate_parameters(self.a0, 0.0, 0.0)
        validate_products(self.products)
        ranked_products = sorted(self.products, key=lambda p: (p.a, p.productId))
        self.products_by_a = ranked_products
        self.a_values = [p.a for p in ranked_products]
        # Keep zero as a genuine breakpoint.  It is also a valid product
        # benefit, so it must not be removed by slicing off the sentinel.
        self.m_values = np.array(sorted({0.0, *(p.m for p in products)}), dtype=float)
        self.unique_m = self.m_values
        self.m_rank = {float(m): i for i, m in enumerate(self.unique_m)}
        self.product_m_rank = [self.m_rank[float(p.m)] for p in ranked_products]
        self.product_ids_by_rank = np.array([p.productId for p in ranked_products], dtype=np.int64)

        tree = _PersistentBenefitTree(len(self.unique_m))
        roots = [0]
        for p, position in zip(ranked_products, self.product_m_rank):
            roots.append(tree.update(roots[-1], position, p.a, p.m * p.a))
        self.tree = tree
        self.roots = roots

        outside_prefix = bisect_left(self.a_values, self.a0)
        self.benchmarks: list[_Benchmark] = []
        self.benchmarks.append(_Benchmark(None, self.a0, outside_prefix))
        for index in range(outside_prefix, len(ranked_products)):
            p = ranked_products[index]
            self.benchmarks.append(_Benchmark(p, p.a, index))

    def _query(self, root: int, rho: float) -> tuple[float, float]:
        first = bisect_right(self.unique_m, rho)
        return self.tree.suffix(root, first)

    def _search(self, benchmark: _Benchmark, p: int, outside_focal: bool, delta: float):
        root_k = self.roots[benchmark.candidate_count]
        root_p = self.roots[p]
        mandatory_base_a = 0.0 if benchmark.product is None else benchmark.product.a
        mandatory_base_b = 0.0 if benchmark.product is None else benchmark.product.m * benchmark.product.a
        mandatory_a = (1.0 + delta) * mandatory_base_a
        mandatory_b = (1.0 + delta) * mandatory_base_b
        effective_a0 = self.a0 * (1.0 + delta * float(outside_focal))

        def phi(index: int) -> float:
            rho = float(self.m_values[index])
            base_a, base_b = self.tree.suffix(root_k, bisect_right(self.unique_m, rho))
            non_a, non_b = self.tree.suffix(root_p, bisect_right(self.unique_m, rho))
            focal_a, focal_b = base_a - non_a, base_b - non_b
            total_a = base_a + delta * focal_a
            total_b = base_b + delta * focal_b
            return total_b + mandatory_b - rho * (effective_a0 + total_a + mandatory_a)

        lo, hi = 0, len(self.m_values) - 1
        last = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if phi(mid) >= 0.0:
                last = mid
                lo = mid + 1
            else:
                hi = mid - 1
        rho = float(self.m_values[last])
        base_a, base_b = self._query(root_k, rho)
        non_a, non_b = self._query(root_p, rho)
        focal_a, focal_b = base_a - non_a, base_b - non_b
        total_a = base_a + delta * focal_a
        total_b = base_b + delta * focal_b
        denominator = effective_a0 + total_a + mandatory_a
        value = (total_b + mandatory_b) / denominator

        first = bisect_right(self.unique_m, rho)
        lower_rank = self.tree.last_present_at_or_before(root_k, first - 1)
        upper_rank = self.tree.first_present_at_or_after(root_k, first)
        lower = 0.0 if lower_rank is None else float(self.unique_m[lower_rank])
        upper = float("inf") if upper_rank is None else float(self.unique_m[upper_rank])
        n0 = base_b + mandatory_base_b
        nf = focal_b + mandatory_base_b
        d0 = self.a0 + mandatory_base_a + base_a
        df = self.a0 * float(outside_focal) + mandatory_base_a + focal_a
        certificate = _Certificate(rho, lower, upper, n0, nf, d0, df)
        return value, rho, certificate

    def _recover(self, benchmark: _Benchmark, rho: float) -> list[int]:
        k = benchmark.candidate_count
        ids = [int(self.product_ids_by_rank[i]) for i in range(k)
               if self.products_by_a[i].m > rho]
        if benchmark.product is not None:
            ids.append(int(benchmark.product.productId))
        return ids

    def _nonfocal_count(self, benchmark: _Benchmark, w: float) -> int:
        """Count free nonfocal products using the actual multiplication boundary."""
        lo, hi = 0, benchmark.candidate_count
        while lo < hi:
            mid = (lo + hi) // 2
            if (1.0 + w) * self.a_values[mid] >= benchmark.benchmark_a:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def _request_groups(self, benchmark, requests, w_order, delta_order):
        """Algorithm 2: O(Q + G_t log n) grouping, not Q separate searches.

        Instead of dividing to form xi_t, test the next focal transition by
        multiplication.  This preserves the inclusive float64 model boundary.
        The outside option must participate in transitions even though it is
        not stored in the product-only persistent versions.
        """
        assigned = [0] * len(requests)
        groups = []
        cursor = 0
        while cursor < len(w_order):
            w = requests[w_order[cursor]][0]
            p = self._nonfocal_count(benchmark, w)
            outside_focal = (1.0 + w) * self.a0 >= benchmark.benchmark_a
            # The next focal transition is the largest attractiveness among
            # currently nonfocal alternatives, including the outside option.
            # Keeping the outside option here is essential when p == 0.
            next_a = self.a_values[p - 1] if p else 0.0
            if not outside_focal:
                next_a = max(next_a, self.a0)
            group_index = len(groups)
            groups.append((p, outside_focal, []))
            # The first request always belongs to the new group. Subsequent
            # requests remain in it strictly before the next focal entry.
            assigned[w_order[cursor]] = group_index
            cursor += 1
            while cursor < len(w_order):
                next_w = requests[w_order[cursor]][0]
                if next_a and (1.0 + next_w) * next_a >= benchmark.benchmark_a:
                    break
                assigned[w_order[cursor]] = group_index
                cursor += 1
        for q_index in delta_order:
            groups[assigned[q_index]][2].append(q_index)
        return groups

    def solve(self, requests: Iterable[tuple[float, float]]) -> BatchResult:
        requests = [(float(w), float(delta)) for w, delta in requests]
        q = len(requests)
        if q == 0:
            raise ValueError("at least one request is required")
        for w, delta in requests:
            validate_parameters(self.a0, w, delta)
        best_values = np.full(q, -np.inf, dtype=float)
        best_benchmarks: list[int | None] = [None] * q
        best_rhos = np.zeros(q, dtype=float)
        searches = 0
        groups = 0
        pair_count = q * len(self.benchmarks)
        w_order = sorted(range(q), key=lambda j: requests[j][0])
        delta_order = sorted(range(q), key=lambda j: requests[j][1])

        for benchmark_index, benchmark in enumerate(self.benchmarks):
            grouped = self._request_groups(benchmark, requests, w_order, delta_order)
            groups += len(grouped)
            for p, outside_focal, indices in grouped:
                certificate: _Certificate | None = None
                for q_index in indices:
                    _w, delta = requests[q_index]
                    if certificate is None or not certificate.contains(delta):
                        value, rho, certificate = self._search(benchmark, p, outside_focal, delta)
                        searches += 1
                    else:
                        value = certificate.value(delta)
                        rho = certificate.threshold
                    if value > best_values[q_index]:
                        best_values[q_index] = value
                        best_benchmarks[q_index] = benchmark_index
                        best_rhos[q_index] = rho

        selected = [self._recover(self.benchmarks[int(best_benchmarks[i])], float(best_rhos[i]))
                    for i in range(q)]
        return BatchResult(
            values=best_values,
            selected=selected,
            best_benchmarks=best_benchmarks,
            thresholds=best_rhos,
            searches=searches,
            groups=groups,
            benchmark_count=len(self.benchmarks),
            pair_count=pair_count,
            certificate_hits=pair_count - searches,
        )
