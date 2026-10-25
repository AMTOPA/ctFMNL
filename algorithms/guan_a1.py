"""
ctFMNL 模型求解算法 - O(n²) Algorithm A.1 实现

关键修正：正确处理外部选项 0 可能属于焦点集的情况
根据论文第 10 页："Note that the no-purchase option 0 can be either focal or non-focal."

日期：2026-03-02
"""

import argparse
import math
import time
from typing import List, Dict, Any

from algorithms.fabs import (
    Product, generateRandomProducts, validate_parameters, validate_products,
    verifySolution as _verify_solution,
)


def solveCtFmnlAlgorithmA1(
    products: List[Product],
    a0: float,
    w: float,
    delta: float,
) -> Dict[str, Any]:
    """
    O(n²) 算法 - 严格实现论文 Algorithm A.1
    
    关键修正：正确处理外部选项 0 可能属于焦点集的情况
    - 对于每个候选 t，判断外部选项 0 是否在焦点集中：(1+w)*a0 >= a_t
    - 如果 0 是焦点的，则 effective_a0 = a0 * (1+delta)，否则 effective_a0 = a0
    
    复杂度：O(n²)
    """
    validate_parameters(a0, w, delta)
    validate_products(products)

    if not products:
        return {"revenue": 0.0, "assortment": [], "t": 0, "k": 0}

    productsByMarkup = sorted(products, key=lambda p: (-p.m, p.productId))

    bestRevenue = -math.inf
    bestCandidateIds: List[int] = []
    bestT = 0
    bestK = 0

    tCandidates = [None] + products

    for tProduct in tCandidates:
        if tProduct is None:
            tId = 0
            aT = a0
            mT = 0.0
        else:
            tId = tProduct.productId
            aT = tProduct.a
            mT = tProduct.m
            if aT < a0:
                continue

        # 构建 candidates：a_i <= a_t 的产品（按 markup 降序）
        candidates = []
        for p in productsByMarkup:
            if p.productId == tId:
                continue
            if p.a <= aT:
                candidates.append(p)

        # 关键修正：判断外部选项 0 是否在焦点集中
        # 根据论文，焦点集是 S⁺ = S ∪ {0} 的子集，0 可以是焦点的
        a0_is_focal = (1.0 + w) * a0 >= aT
        effective_a0 = a0 * (1.0 + delta) if a0_is_focal else a0

        # 计算 t 的修正吸引力
        if tId != 0:
            aTildeT = aT * (1.0 + delta)
            amTildeT = aTildeT * mT
        else:
            aTildeT = 0.0
            amTildeT = 0.0

        # 前缀和数组
        prefixA = [0.0]
        prefixAM = [0.0]
        prefixIds = []
        for p in candidates:
            isFocal = (1.0 + w) * p.a >= aT
            aTilde = p.a * (1.0 + delta) if isFocal else p.a
            prefixA.append(prefixA[-1] + aTilde)
            prefixAM.append(prefixAM[-1] + aTilde * p.m)
            prefixIds.append(p.productId)

        # 遍历 k 寻找最优解
        for k in range(len(prefixIds) + 1):
            denom = effective_a0 + prefixA[k] + aTildeT
            if denom <= 0.0:
                raise FloatingPointError("nonpositive ctFMNL denominator")
            numerator = prefixAM[k] + amTildeT
            revenue = numerator / denom
            if revenue > bestRevenue:
                bestRevenue = revenue
                bestK = k
                bestT = tId
                # Keep the current prefix by reference; materialize the final
                # assortment only once after all benchmark candidates are scanned.
                bestCandidateIds = prefixIds

    if bestRevenue == -math.inf:
        bestRevenue = 0.0

    bestAssortment = bestCandidateIds[:bestK]
    if bestT != 0:
        bestAssortment.append(bestT)
    return {"revenue": bestRevenue, "assortment": bestAssortment, "t": bestT, "k": bestK}


def verifySolution(
    products: List[Product],
    selected: List[int],
    a0: float,
    w: float,
    delta: float,
) -> float:
    """
    验证解的收益
    
    关键修正：正确处理外部选项 0 可能属于焦点集的情况
    """
    return _verify_solution(products, selected, a0, w, 1.0 + delta)


def main():
    parser = argparse.ArgumentParser(description="ctFMNL 模型 O(n²) 求解算法")
    parser.add_argument("--test", action="store_true", help="运行测试")
    parser.add_argument("--n", type=int, default=10000, help="产品数量")
    parser.add_argument("--tests", type=int, default=100, help="测试次数")
    parser.add_argument("--seed", type=int, default=1, help="随机种子")
    parser.add_argument("--a0", type=float, default=1.0, help="无购买选项吸引力")
    parser.add_argument("--w", type=float, default=0.5, help="阈值参数")
    parser.add_argument("--delta", type=float, default=1.0, help="增量参数")
    args = parser.parse_args()

    if args.test:
        print(f"进行 {args.tests} 次测试，每次 n={args.n} 个产品...")
        print("=" * 80)
        
        all_passed = True
        max_diff = 0.0
        
        for test_idx in range(args.tests):
            seed = args.seed + test_idx
            products = generateRandomProducts(args.n, seed=seed)
            a0, w, delta = args.a0, args.w, args.delta
            
            result = solveCtFmnlAlgorithmA1(products, a0, w, delta)
            rev = result['revenue']
            assort = result['assortment']
            
            # 验证收益
            verified = verifySolution(products, assort, a0, w, delta)
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
        print("ctFMNL 模型 O(n²) 求解算法演示")
        print("=" * 80)
        
        products = generateRandomProducts(args.n, seed=args.seed)
        a0, w, delta = args.a0, args.w, args.delta

        print(f"参数: n={args.n}, a0={a0}, w={w}, delta={delta}")
        print()

        start = time.perf_counter()
        result = solveCtFmnlAlgorithmA1(products, a0, w, delta)
        elapsed = time.perf_counter() - start

        print("O(n²) 算法:")
        print(f"  收益: {result['revenue']:.6f}")
        print(f"  选择: {result['assortment']}")
        print(f"  t={result['t']}, k={result['k']}")
        print(f"  时间: {elapsed:.6f}s")
        
        # 验证收益
        verified = verifySolution(products, result['assortment'], a0, w, delta)
        print(f"  验证收益: {verified:.6f}")
        print("=" * 80)


if __name__ == "__main__":
    main()
