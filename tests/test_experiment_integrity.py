"""Regression tests for input/provenance and verification (no benchmarks)."""
import unittest
from unittest.mock import patch

import numpy as np

import experiments.batch.run_serial as runner
import experiments.batch.run_parallel as parallel
from algorithms.fabs import Product, solveCtFmnlNLog2N
from experiments.common import ObjectiveOracle
from data.synthetic import generate_synthetic_instance
from algorithms.spar import RepeatedOptimizer


class ExperimentIntegrityTests(unittest.TestCase):
    def test_prefixes_do_not_depend_on_q(self):
        z, delta = runner.request_arrays(500, 1, runner.FORMAL_SEED, 900)
        for q in (1, 5, 20, 100, 300, 301, 499, 500, 501, 700, 900):
            with self.subTest(q=q):
                short_z, short_delta = runner.request_arrays(500, 1, runner.FORMAL_SEED, q)
                np.testing.assert_array_equal(short_z, z[:q])
                np.testing.assert_array_equal(short_delta, delta[:q])

    def test_legacy_300_and_500_preserved(self):
        seed, n, instance = runner.FORMAL_SEED, 500, 1
        rng = np.random.default_rng(np.random.SeedSequence([seed, 640000, n, instance]))
        z, d = rng.uniform(-1, 1, 300), rng.uniform(1.5, 3, 300)
        tail = np.random.default_rng(np.random.SeedSequence([seed, 640001, n, instance]))
        z = np.r_[z, tail.uniform(-1, 1, 200)]
        d = np.r_[d, tail.uniform(1.5, 3, 200)]
        actual_z, actual_d = runner.request_arrays(n, instance, seed, 500)
        np.testing.assert_array_equal(actual_z, z)
        np.testing.assert_array_equal(actual_d, d)

    def test_unrounded_benefits(self):
        m = generate_synthetic_instance(10000, 1, 123)["m"]
        self.assertTrue(np.any(m != np.round(m, 2)))
        self.assertEqual(len(np.unique(m)), 10000)

    def test_correct_pair(self):
        p = [Product(1, 1, 10)]
        self.assertEqual(runner.verify_pair(p, [(0, 1)], ([5.], [[1]]), ([5.], [[1]])),
                         (0., True, True, 0.))

    def test_same_wrong_answers_rejected(self):
        with self.assertRaises(RuntimeError):
            runner.verify_pair([Product(1, 1, 10)], [(0, 1)], ([6.], [[1]]), ([6.], [[1]]))

    def test_nonfinite_results_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.verify_pair([Product(1, 1, 10)], [(0, 1)], ([value], [[1]]), ([value], [[1]]))

    def test_partial_results_rejected(self):
        with self.assertRaises(ValueError):
            runner.verify_pair([Product(1, 1, 10)], [(0, 1), (1, 2)], ([5.], [[1]]), ([5.], [[1]]))

    def test_second_repeat_is_independently_verified(self):
        job = dict(out_dir="unused", data_dir="unused", n=1, instance=1,
                   scenario_index=0, scenario="concentrated", h=.01, q=1, repeat=2, seed=1)
        cached = ([Product(1, 1, 10)], np.array([0.]), np.array([1.]), "hash")
        with patch.object(parallel, "load_cached_inputs", return_value=cached), \
             patch.object(parallel, "serial_run_method", return_value=([6.], [[1]], None, .1)):
            with self.assertRaises(RuntimeError):
                parallel.run_job(job)

    def test_small_random_instances_against_enumeration(self):
        for seed in range(30):
            products, data = runner.make_products(7, 1, seed)
            requests = [(0., 0.), (data["w"], data["delta"])]
            batch = RepeatedOptimizer(products).solve(requests)
            for k, (w,d) in enumerate(requests):
                oracle = ObjectiveOracle(products, 1., w, d)
                optimum = max(oracle([i+1 for i in range(7) if mask & (1 << i)])
                              for mask in range(1 << 7))
                value, selected = solveCtFmnlNLog2N(products, 1., w, 1+d)
                self.assertAlmostEqual(value, optimum, places=8)
                self.assertAlmostEqual(oracle(selected), optimum, places=8)
                self.assertAlmostEqual(batch.values[k], optimum, places=8)
                self.assertAlmostEqual(oracle(batch.selected[k]), optimum, places=8)


if __name__ == "__main__":
    unittest.main()
