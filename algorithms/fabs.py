"""
ctFMNL 模型求解算法 - O(n log² n) 实现

关键修正：正确处理外部选项 0 可能属于焦点集的情况
根据论文第 10 页："Note that the no-purchase option 0 can be either focal or non-focal."

日期：2026-03-02
"""

import argparse
import math
from numbers import Integral
import random
import time
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Product:
    productId: int
    a: float
    m: float


def validate_parameters(a0: float, w: float, delta: float) -> None:
    """Validate the paper's attention-parameter domain, including empty instances."""
    if (not math.isfinite(a0) or a0 <= 0
            or not math.isfinite(w) or w < 0
            or not math.isfinite(delta) or delta < 0):
        raise ValueError("require finite a0 > 0, w >= 0, delta >= 0")


def validate_products(products: List[Product]) -> None:
    """Product IDs are unique positive integers; ID zero is the outside option."""
    seen = set()
    for p in products:
        if (not isinstance(p.productId, Integral) or isinstance(p.productId, bool)
                or p.productId <= 0 or p.productId in seen):
            raise ValueError("productId values must be unique positive integers")
        if (not math.isfinite(p.a) or p.a <= 0
                or not math.isfinite(p.m) or p.m < 0):
            raise ValueError("require finite a_i > 0 and m_i >= 0")
        seen.add(p.productId)


class FenwickTree:
    """Fenwick Tree for prefix sums (1-based indexing)"""
    def __init__(self, size: int):
        self.size = size
        self.tree = [0.0] * (size + 1)

    def build(self, values) -> None:
        """Build the Fenwick array from one-based point values in O(n)."""
        self.size = len(values)
        self.tree = [0.0] + list(values)
        for i in range(1, self.size + 1):
            parent = i + (i & -i)
            if parent <= self.size:
                self.tree[parent] += self.tree[i]

    def add(self, index: int, delta: float) -> None:
        """Add delta to position index (1-based)"""
        if index <= 0 or index > self.size:
            return
        i = index
        while i <= self.size:
            self.tree[i] += delta
            i += i & -i

    def sum(self, index: int) -> float:
        """Prefix sum [1, index] (1-based). Returns 0 if index <= 0."""
        if index <= 0:
            return 0.0
        i = min(index, self.size)
        res = 0.0
        while i > 0:
            res += self.tree[i]
            i -= i & -i
        return res


def generateRandomProducts(
    n: int, seed: int,
    aRange: Tuple[float, float] = (0.1, 10.0),
    mRange: Tuple[float, float] = (0.1, 10.0)
) -> List[Product]:
    """生成随机产品数据"""
    rng = random.Random(seed)
    return [
        Product(productId=i+1, a=rng.uniform(*aRange), m=rng.uniform(*mRange))
        for i in range(n)
    ]


def solveCtFmnlNLog2N(
    products: List[Product],
    a0: float,
    w: float,
    q: float,
) -> Tuple[float, List[int]]:
    """Solve ctFMNL assortment optimization by the paper's FABS Algorithm 1.

    The historical project API uses q = 1 + delta; the paper's reinforcement
    parameter is therefore delta = q - 1.
    """
    delta = q - 1.0
    validate_parameters(a0, w, delta)
    validate_products(products)
    if not products:
        return 0.0, []

    n = len(products)
    # Algorithm 1, lines 1-2: rank N union {0}; the outside option precedes
    # products at the same attractiveness under the fixed tie-breaking rule.
    ranked = sorted(
        [(a0, 0, None)] + [(p.a, 1, p) for p in products],
        key=lambda item: (item[0], item[1], item[2].productId if item[2] else 0),
    )
    r0 = next(r for r, (_, _, p) in enumerate(ranked, start=1) if p is None)
    product_rank = {
        p.productId: r for r, (_, _, p) in enumerate(ranked, start=1) if p is not None
    }

    # Algorithm 1, line 3: B = {0} union {m_i}, with q(i) as breakpoint rank.
    breakpoints = sorted({0.0, *(p.m for p in products)})
    breakpoint_position = {b: k for k, b in enumerate(breakpoints, start=1)}
    product_position = {p.productId: breakpoint_position[p.m] for p in products}
    tree_a = FenwickTree(len(breakpoints))
    tree_b = FenwickTree(len(breakpoints))
    total_a = 0.0
    total_b = 0.0

    # Algorithm 1, lines 4-5 and equation (14): construct focal-exit events.
    events = [[] for _ in range(n + 3)]
    pointer = 0
    ranked_attractiveness = [item[0] for item in ranked]
    for p in sorted(products, key=lambda item: (item.a, item.productId)):
        threshold = (1.0 + w) * p.a
        while pointer < len(ranked_attractiveness) and ranked_attractiveness[pointer] <= threshold:
            pointer += 1
        exit_position = pointer + 1 if pointer < len(ranked_attractiveness) else n + 2
        if exit_position <= n + 1:
            events[exit_position].append(p)

    def evaluate_benchmark(t: Product | None, benchmark_a: float):
        outside_focal = (1.0 + w) * a0 >= benchmark_a
        effective_a0 = q * a0 if outside_focal else a0
        mandatory_a = 0.0 if t is None else q * t.a
        mandatory_b = 0.0 if t is None else t.m * mandatory_a

        def phi(k: int) -> float:
            # k is zero-based; strict m_i > b_k queries the suffix after k+1.
            position = k + 1
            suffix_a = total_a - tree_a.sum(position)
            suffix_b = total_b - tree_b.sum(position)
            rho = breakpoints[k]
            return suffix_b - rho * (effective_a0 + suffix_a) + mandatory_b - rho * mandatory_a

        lo, hi = 0, len(breakpoints) - 1
        last_nonnegative = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if phi(mid) >= 0.0:
                last_nonnegative = mid
                lo = mid + 1
            else:
                hi = mid - 1

        position = last_nonnegative + 1
        suffix_a = total_a - tree_a.sum(position)
        suffix_b = total_b - tree_b.sum(position)
        denominator = effective_a0 + suffix_a + mandatory_a
        if denominator <= 0.0:
            raise FloatingPointError("nonpositive ctFMNL denominator")
        return (suffix_b + mandatory_b) / denominator, last_nonnegative

    # Algorithm 1, lines 6-7.
    best_value = float("-inf")
    best_t = None
    best_breakpoint = 0

    # Algorithm 1, lines 8-25: events, benchmark evaluation, then insertion.
    for r, (_, _, t) in enumerate(ranked, start=1):
        for p in events[r]:
            position = product_position[p.productId]
            decrease_a = delta * p.a
            decrease_b = delta * p.m * p.a
            tree_a.add(position, -decrease_a)
            tree_b.add(position, -decrease_b)
            total_a -= decrease_a
            total_b -= decrease_b

        if r >= r0:
            value, breakpoint_index = evaluate_benchmark(t, a0 if t is None else t.a)
            if value > best_value:
                best_value = value
                best_t = t
                best_breakpoint = breakpoint_index

        if t is not None:
            position = product_position[t.productId]
            initial_a = q * t.a
            initial_b = t.m * initial_a
            tree_a.add(position, initial_a)
            tree_b.add(position, initial_b)
            total_a += initial_a
            total_b += initial_b

    # Algorithm 1, line 27: recover S_t*(b_k*) once using equation (9).
    benchmark_rank = r0 if best_t is None else product_rank[best_t.productId]
    rho = breakpoints[best_breakpoint]
    assortment = []
    for r in range(1, benchmark_rank + 1):
        _, _, p = ranked[r - 1]
        if p is not None and (p is best_t or p.m > rho):
            assortment.append(p.productId)
    return best_value, assortment

def verifySolution(
    products: List[Product],
    selected: List[int],
    a0: float,
    w: float,
    q: float,
) -> float:
    """
    验证解的收益
    
    关键修正：正确处理外部选项 0 可能属于焦点集的情况
    """
    validate_parameters(a0, w, q - 1.0)
    validate_products(products)
    if not selected:
        return 0.0
    index = {p.productId: p for p in products}
    if len(selected) != len(set(selected)):
        raise ValueError("assortment contains duplicate product IDs")
    try:
        selected_prods = [index[pid] for pid in selected]
    except KeyError as exc:
        raise ValueError(f"unknown product ID: {exc.args[0]!r}") from exc
    t = max(selected_prods, key=lambda p: p.a)
    a_max = max(a0, t.a)
    outside_focal = (1.0 + w) * a0 >= a_max
    effective_a0 = q * a0 if outside_focal else a0

    total_a = 0.0
    total_ma = 0.0

    for p in selected_prods:
        is_focal = (1.0 + w) * p.a >= a_max
        s = q if is_focal else 1.0
        a_tilde = s * p.a
        total_a += a_tilde
        total_ma += p.m * a_tilde

    denom = effective_a0 + total_a
    if denom <= 0.0:
        raise FloatingPointError("nonpositive ctFMNL denominator")
    return total_ma / denom


def main():
    parser = argparse.ArgumentParser(description="ctFMNL 模型 O(n log² n) 求解算法")
    parser.add_argument("--test", action="store_true", help="运行测试")
    parser.add_argument("--n", type=int, default=100000, help="产品数量")
    parser.add_argument("--tests", type=int, default=100, help="测试次数")
    parser.add_argument("--seed", type=int, default=1, help="随机种子")
    parser.add_argument("--a0", type=float, default=1.0, help="无购买选项吸引力")
    parser.add_argument("--w", type=float, default=0.5, help="阈值参数")
    parser.add_argument("--delta", type=float, default=1.0, help="增量参数")
    args = parser.parse_args()

    q = 1.0 + args.delta

    if args.test:
        print(f"进行 {args.tests} 次测试，每次 n={args.n} 个产品...")
        print("=" * 80)
        
        all_passed = True
        max_diff = 0.0
        
        for test_idx in range(args.tests):
            seed = args.seed + test_idx
            products = generateRandomProducts(args.n, seed=seed)
            a0, w = args.a0, args.w
            
            rev, assort = solveCtFmnlNLog2N(products, a0, w, q)
            
            # 验证收益
            verified = verifySolution(products, assort, a0, w, q)
            diff = abs(rev - verified)
            if diff > max_diff:
                max_diff = diff
            
            if diff > 1e-6:
                print(f"测试 {test_idx + 1} (seed={seed}): FAILED")
                print(f"  算法收益: {rev:.6f}")
                print(f"  验证收益: {verified:.6f}")
                all_passed = False
        
        print(f"最大差异: {max_diff:.6e}")
        if all_passed:
            print(f"所有 {args.tests} 次测试通过！")
    else:
        # 演示
        print("=" * 80)
        print("ctFMNL 模型 O(n log² n) 求解算法演示")
        print("=" * 80)
        
        products = generateRandomProducts(args.n, seed=args.seed)
        a0, w = args.a0, args.w

        print(f"参数: n={args.n}, a0={a0}, w={w}, delta={args.delta}")
        print()

        start = time.perf_counter()
        rev, assort = solveCtFmnlNLog2N(products, a0, w, q)
        elapsed = time.perf_counter() - start

        print("O(n log² n) 算法:")
        print(f"  收益: {rev:.6f}")
        print(f"  选择: {assort}")
        print(f"  时间: {elapsed:.6f}s")
        
        # 验证收益
        verified = verifySolution(products, assort, a0, w, q)
        print(f"  验证收益: {verified:.6f}")
        print("=" * 80)


if __name__ == "__main__":
    main()
