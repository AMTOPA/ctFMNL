"""
ctFMNL solver in O(n log^2 n), written in the "fixed t -> MNL threshold"
style rather than the "fractional programming / Dinkelbach" style.

Key idea
- Outer loop: enumerate the baseline item t with the largest attraction in S.
- Inner loop: once t is fixed, the problem becomes a forced-inclusion MNL assortment.
- The optimal inner solution is revenue-ordered by m and can be found by binary
  search over revenue breakpoints, with Fenwick trees maintaining suffix sums.

This file keeps the executable logic aligned with the uploaded implementation,
but rewrites the interpretation and comments so the solver no longer relies on
an explicit fractional-programming narrative.
"""

from __future__ import annotations

import argparse
import time
from typing import List, Tuple, Optional


from algorithms.fabs import (
    Product, FenwickTree, generateRandomProducts, validate_parameters,
    validate_products, verifySolution as _verify_solution,
)


def effective_weight(item_a: float, baseline_a: float, w: float, q: float) -> float:
    """Effective attraction under fixed baseline item t."""
    return q * item_a if (1.0 + w) * item_a >= baseline_a else item_a


def solveCtFmnlNLog2N_MNLThreshold(
    products: List[Product],
    a0: float,
    w: float,
    q: float,
) -> Tuple[float, List[int]]:
    """
    Exact O(n log^2 n) ctFMNL solver.

    Interpretation used here:
    1) Enumerate the maximum-attraction item t.
    2) For fixed t, solve a forced-inclusion MNL assortment problem.
    3) The optimal inner solution is an m-threshold set.
    4) Use binary search on the ordered m-breakpoints and Fenwick trees for
       fast suffix-sum queries.
    """
    validate_parameters(a0, w, q - 1.0)
    validate_products(products)
    if not products:
        return 0.0, []

    m_values = sorted({0.0, *(p.m for p in products)})
    m_to_rank = {m: i + 1 for i, m in enumerate(m_values)}

    prod_by_a = sorted(products, key=lambda p: (p.a, p.productId))
    prod_by_m_desc = sorted(products, key=lambda p: (-p.m, p.productId))

    bit_A = FenwickTree(len(m_values))
    bit_B = FenwickTree(len(m_values))
    cur_sum_A = 0.0
    cur_sum_B = 0.0

    best_revenue = 0.0
    best_t: Optional[Product] = None
    best_threshold = 0.0

    ptr_add = 0
    ptr_demote = 0
    best_t_index: Optional[int] = None

    # Case t = 0: the no-purchase option is the largest-attraction alternative.
    # Then the inner MNL problem is still revenue-ordered by m.
    t0_candidates = [p for p in prod_by_m_desc if p.a <= a0]
    a0_eff_t0 = q * a0
    prefix_A = 0.0
    prefix_B = 0.0
    best_rev_t0 = 0.0
    best_k_t0 = 0

    for k, p in enumerate(t0_candidates):
        a_tilde = effective_weight(p.a, a0, w, q)
        prefix_A += a_tilde
        prefix_B += p.m * a_tilde
        denom = a0_eff_t0 + prefix_A
        if denom > 0.0:
            rev = prefix_B / denom
            if rev > best_rev_t0:
                best_rev_t0 = rev
                best_k_t0 = k + 1

    if best_rev_t0 > best_revenue:
        best_revenue = best_rev_t0
        best_t = None

    # Main scan over candidate baseline items t in nondecreasing a_t.
    for idx_t, t in enumerate(prod_by_a):
        a_t = t.a

        # Only candidates with a_t >= a0 can be larger than the outside option.
        if a_t < a0:
            continue

        # Activate all items with a_i <= a_t. They enter as focal first.
        while ptr_add < idx_t:
            p = prod_by_a[ptr_add]
            rank = m_to_rank[p.m]
            a_tilde = q * p.a
            bit_A.add(rank, a_tilde)
            bit_B.add(rank, p.m * a_tilde)
            cur_sum_A += a_tilde
            cur_sum_B += p.m * a_tilde
            ptr_add += 1

        # Items eventually stop being focal when (1+w)a_i < a_t. Each item is
        # demoted from q*a_i to a_i at most once over the full scan.
        while ptr_demote < ptr_add:
            p = prod_by_a[ptr_demote]
            if (1.0 + w) * p.a >= a_t:
                break
            if ptr_demote != idx_t:
                rank = m_to_rank[p.m]
                demote = (q - 1.0) * p.a
                bit_A.add(rank, -demote)
                bit_B.add(rank, -p.m * demote)
                cur_sum_A -= demote
                cur_sum_B -= p.m * demote
            ptr_demote += 1

        outside_is_focal = (1.0 + w) * a0 >= a_t
        a0_eff = q * a0 if outside_is_focal else a0

        if m_values:
            lo, hi = 0, len(m_values) - 1
            first_nonpositive = len(m_values)
            first_nonpositive_prefix_A = 0.0
            first_nonpositive_prefix_B = 0.0

            # For threshold lam, include exactly those items with m_i > lam.
            # Let H_t(lam) = Q(lam) - lam * (a0_eff + P(lam));
            # H_t is monotone over ordered breakpoints, so binary search applies.
            while lo <= hi:
                mid = (lo + hi) // 2
                lam = m_values[mid]
                idx = mid + 1

                prefix_A = bit_A.sum(idx)
                prefix_B = bit_B.sum(idx)

                # Suffix sums over items with m_i > lam.
                suffix_A = cur_sum_A - prefix_A
                suffix_B = cur_sum_B - prefix_B

                # Forced inclusion of baseline item t.
                baseline_A = q * a_t
                baseline_B = t.m * baseline_A

                h_val = (suffix_B + baseline_B) - lam * (a0_eff + suffix_A + baseline_A)

                if h_val <= 0.0:
                    first_nonpositive = mid
                    first_nonpositive_prefix_A = prefix_A
                    first_nonpositive_prefix_B = prefix_B
                    hi = mid - 1
                else:
                    lo = mid + 1

            baseline_A = q * a_t
            baseline_B = t.m * baseline_A

            def evaluate_breakpoint(index: int, prefix=None):
                if not 0 <= index < len(m_values):
                    return None
                if prefix is None:
                    prefix = (bit_A.sum(index + 1), bit_B.sum(index + 1))
                suffix_A = cur_sum_A - prefix[0]
                suffix_B = cur_sum_B - prefix[1]
                denom = a0_eff + suffix_A + baseline_A
                if denom <= 0.0:
                    return None
                return ((suffix_B + baseline_B) / denom, float(m_values[index]))

            candidates = [(first_nonpositive,
                           (first_nonpositive_prefix_A, first_nonpositive_prefix_B))]
            candidates.append((first_nonpositive - 1, None))
            evaluated = [evaluate_breakpoint(index, prefix) for index, prefix in candidates]
            evaluated = [result for result in evaluated if result is not None]
            best_rev_t, best_threshold_t = max(
                evaluated, default=(-1e300, 0.0), key=lambda result: result[0]
            )

            if best_rev_t > best_revenue:
                best_revenue = best_rev_t
                best_t = t
                best_t_index = idx_t
                best_threshold = best_threshold_t

        # After solving the subproblem for current t, insert t so that later
        # baseline items can treat it as an ordinary candidate.
        rank_t = m_to_rank[t.m]
        baseline_A = q * a_t
        baseline_B = t.m * baseline_A
        bit_A.add(rank_t, baseline_A)
        bit_B.add(rank_t, baseline_B)
        cur_sum_A += baseline_A
        cur_sum_B += baseline_B
        ptr_add = idx_t + 1

    if best_t is None:
        return best_revenue, [p.productId for p in t0_candidates[:best_k_t0]]

    selected = []
    assert best_t_index is not None
    for p in prod_by_a[:best_t_index + 1]:
        if p.productId != best_t.productId and p.m > best_threshold:
            selected.append(p.productId)
    selected.append(best_t.productId)
    return best_revenue, selected


def verifySolution(
    products: List[Product],
    selected: List[int],
    a0: float,
    w: float,
    q: float,
) -> float:
    return _verify_solution(products, selected, a0, w, q)


def main() -> None:
    parser = argparse.ArgumentParser(description="ctFMNL O(n log^2 n) solver via MNL threshold structure")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--n", type=int, default=100000)
    parser.add_argument("--tests", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--a0", type=float, default=1.0)
    parser.add_argument("--w", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=1.0)
    args = parser.parse_args()

    q = 1.0 + args.delta

    if args.test:
        max_diff = 0.0
        all_passed = True
        for test_idx in range(args.tests):
            seed = args.seed + test_idx
            products = generateRandomProducts(args.n, seed=seed)
            rev, assort = solveCtFmnlNLog2N_MNLThreshold(products, args.a0, args.w, q)
            verified = verifySolution(products, assort, args.a0, args.w, q)
            diff = abs(rev - verified)
            max_diff = max(max_diff, diff)
            if diff > 1e-6:
                all_passed = False
                print(f"FAILED seed={seed}: algo={rev:.9f}, verify={verified:.9f}")
        print(f"max_diff={max_diff:.6e}")
        if all_passed:
            print("all tests passed")
        return

    products = generateRandomProducts(args.n, seed=args.seed)
    start = time.perf_counter()
    rev, assort = solveCtFmnlNLog2N_MNLThreshold(products, args.a0, args.w, q)
    elapsed = time.perf_counter() - start
    verified = verifySolution(products, assort, args.a0, args.w, q)

    print(f"revenue={rev:.9f}")
    print(f"verify={verified:.9f}")
    print(f"n_selected={len(assort)}")
    print(f"elapsed={elapsed:.6f}s")


if __name__ == "__main__":
    main()
