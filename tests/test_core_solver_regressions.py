"""Regression tests for previously identified core-solver boundary failures."""
from __future__ import annotations

import random
import math
import unittest
from unittest.mock import patch

from algorithms.guan_a1 import solveCtFmnlAlgorithmA1
from algorithms.mnl_threshold import (
    Product as ThresholdProduct,
    solveCtFmnlNLog2N_MNLThreshold,
    verifySolution as verify_threshold,
)
from algorithms.fabs import (
    Product,
    solveCtFmnlNLog2N,
    verifySolution as verify_fabs,
)
from algorithms.ablations import solve_newton_dinkelbach
from algorithms.spar import RepeatedOptimizer
from experiments.common import ObjectiveOracle


class CoreSolverRegressionTests(unittest.TestCase):
    def test_exact_focal_boundary_is_not_tolled(self):
        products = [Product(1, 1.0000000000005, 10.0)]
        expected = ObjectiveOracle(products, 1.0, 0.0, 1.0)([1])
        self.assertAlmostEqual(verify_fabs(products, [1], 1.0, 0.0, 2.0), expected)
        self.assertAlmostEqual(
            solveCtFmnlAlgorithmA1(products, 1.0, 0.0, 1.0)["revenue"],
            expected,
        )

    def test_threshold_solver_handles_tied_attractiveness(self):
        products = [
            ThresholdProduct(1, 2.0, 1.0),
            ThresholdProduct(2, 2.0, 10.0),
        ]
        value, selected = solveCtFmnlNLog2N_MNLThreshold(products, 1.0, 0.0, 2.0)
        self.assertAlmostEqual(value, 8.0)
        self.assertAlmostEqual(verify_threshold(products, selected, 1.0, 0.0, 2.0), value)

    def test_threshold_verifier_includes_outside_option_in_maximum(self):
        products = [ThresholdProduct(1, 1.0, 10.0)]
        self.assertAlmostEqual(
            verify_threshold(products, [1], 10.0, 0.0, 2.0),
            10.0 / 21.0,
        )

    def test_batch_optimizer_accepts_zero_benefit(self):
        products = [Product(1, 1.0, 0.0), Product(2, 2.0, 10.0)]
        result = RepeatedOptimizer(products).solve([(0.0, 1.0)])
        self.assertAlmostEqual(result.values[0], 8.0)
        self.assertAlmostEqual(
            result.values[0], ObjectiveOracle(products, 1.0, 0.0, 1.0)(result.selected[0])
        )

    def test_batch_optimizer_uses_exact_focal_boundary(self):
        products = [Product(1, 7.0, 10.0), Product(2, 8.4, 10.0)]
        result = RepeatedOptimizer(products).solve([(0.2, 1.0)])
        expected = ObjectiveOracle(products, 1.0, 0.2, 1.0)(result.selected[0])
        self.assertAlmostEqual(result.values[0], expected)

    def test_cycle_is_not_counted_as_iteration_cap(self):
        _, _, telemetry = solve_newton_dinkelbach(
            [Product(1, 1.0, 0.2), Product(2, 2.0, 2.0), Product(3, 2.0, 4.0)],
            2.0,
            0.0,
            0.0,
        )
        self.assertEqual(telemetry["iteration_cap_hits"], 0)
        self.assertGreaterEqual(telemetry["root_cycles"], 1)

    def test_rejects_invalid_inputs_even_for_empty_instance(self):
        for a0, w, delta in [(0, 0, 0), (1, -1, 0), (1, 0, -1),
                             (math.nan, 0, 0), (1, math.inf, 0), (1, 0, math.nan)]:
            for products in ([], [Product(1, 1, 1)]):
                with self.subTest(params=(a0, w, delta), products=products):
                    for solve in (solveCtFmnlNLog2N, solveCtFmnlNLog2N_MNLThreshold):
                        with self.assertRaises(ValueError):
                            solve(products, a0, w, 1 + delta)
                    with self.assertRaises(ValueError):
                        solveCtFmnlAlgorithmA1(products, a0, w, delta)
        optimizer = RepeatedOptimizer([Product(1, 1, 1)])
        for requests in ([], [(math.nan, 1)], [(0, -1)], [(0, math.inf)]):
            with self.subTest(requests=requests), self.assertRaises(ValueError):
                optimizer.solve(requests)

    def test_invalid_product_ids_and_verifier_selections(self):
        for products in ([Product(0, 1, 1)], [Product(1, 1, 1), Product(1, 2, 2)],
                         [Product(1, math.nan, 1)], [Product(1, 1, -1)]):
            for solve in (solveCtFmnlNLog2N, solveCtFmnlNLog2N_MNLThreshold):
                with self.assertRaises(ValueError):
                    solve(products, 1, 0, 2)
            with self.assertRaises(ValueError):
                solveCtFmnlAlgorithmA1(products, 1, 0, 1)
            with self.assertRaises(ValueError):
                RepeatedOptimizer(products)
        for selected in ([1, 1], [999], [1, 999]):
            with self.assertRaises(ValueError):
                verify_fabs([Product(1, 1, 1)], selected, 1, 0, 2)

    def test_grouping_searches_once_per_group_and_keeps_outside_transitions(self):
        optimizer = RepeatedOptimizer([Product(1, 0.5, 3), Product(2, 2, 10)])
        requests = [(w, delta) for w in [4., 0., 1., 3., 0.5, 1., 2.]
                    for delta in [2., 0., 1.]]
        with patch.object(optimizer, "_nonfocal_count",
                          wraps=optimizer._nonfocal_count) as search:
            batch = optimizer.solve(requests)
        self.assertEqual(search.call_count, batch.groups)
        self.assertLess(search.call_count, batch.pair_count)
        for j, (w, delta) in enumerate(requests):
            value, _ = solveCtFmnlNLog2N(optimizer.products, 1., w, 1. + delta)
            self.assertAlmostEqual(value, batch.values[j], places=10)

    def test_random_ties_zero_benefits_and_scaled_inputs_against_enumeration(self):
        rng = random.Random(20261018)
        for case in range(150):
            n = rng.randrange(0, 9)
            scale = rng.choice([1., 1e-15, 1e15])
            products = [Product(i+1, scale * rng.choice([0.5, 1., 2., 3.]),
                                rng.choice([0., 0.2, 1., 3., 10.])) for i in range(n)]
            a0 = scale * rng.choice([0.5, 1., 2.])
            requests = [(rng.choice([0., 0.2, 1., 3.]), rng.choice([0., 0.5, 2.]))
                        for _ in range(3)]
            batch = RepeatedOptimizer(products, a0).solve(requests)
            for j, (w, delta) in enumerate(requests):
                with self.subTest(case=case, request=j):
                    oracle = ObjectiveOracle(products, a0, w, delta)
                    optimum = max(oracle([i+1 for i in range(n) if mask & (1 << i)])
                                  for mask in range(1 << n))
                    results = [solveCtFmnlNLog2N(products, a0, w, 1 + delta),
                               solveCtFmnlNLog2N_MNLThreshold(products, a0, w, 1 + delta),
                               (batch.values[j], batch.selected[j])]
                    a1 = solveCtFmnlAlgorithmA1(products, a0, w, delta)
                    results.append((a1["revenue"], a1["assortment"]))
                    for value, selected in results:
                        self.assertTrue(math.isclose(value, optimum, rel_tol=1e-9, abs_tol=1e-10),
                                        (value, optimum, products, a0, w, delta))
                        self.assertTrue(math.isclose(oracle(selected), optimum,
                                                     rel_tol=1e-9, abs_tol=1e-10))


if __name__ == "__main__":
    unittest.main()
