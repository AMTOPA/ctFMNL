"""Focused tests for deterministic input generation and cache validation."""

from contextlib import redirect_stderr
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from data import generate as generation
from data.synthetic import generate_synthetic_instance
from data import batch


class DataGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)

    def load(self, path):
        with np.load(path, allow_pickle=False) as saved:
            return {name: saved[name] for name in saved.files}

    def test_protocol_defaults(self):
        self.assertEqual(generation.SINGLE_SEED, 20260929)
        self.assertEqual(generation.EXACTNESS_SCALES, (16,))
        self.assertEqual(generation.COMPARISON_SCALES, (
            100, 500, 1000, 5000, 10000, 20000, 30000, 40000,
            50000, 60000, 70000, 80000, 90000, 100000,
        ))
        self.assertEqual(generation.ABLATION_SCALES, (10000, 50000, 100000))
        self.assertEqual(batch.SCALES, (500, 1000, 2000))
        self.assertEqual(batch.Q_VALUES, (1, 5, 20, 100, 300, 500))
        self.assertEqual(batch.FORMAL_SEED, 20261013)
        self.assertEqual(batch.N_INSTANCES, 20)
        self.assertEqual(batch.SCENARIOS, {"concentrated": .01, "moderate": .1, "broad": 1.5})
        self.assertEqual(batch.FORMAL_DATA, ROOT / "data" / "generated" / "batch")
        args = generation.build_parser().parse_args([])
        self.assertEqual(args.experiment, "all")
        self.assertEqual(args.output, ROOT / "data" / "generated")
        self.assertIsNone(args.n)
        self.assertIsNone(args.instances)

    def test_single_schema_values_and_no_rounding(self):
        path, = generation.generate("comparison", n=[37], instances=1, output=self.output)
        saved = self.load(path)
        expected = generate_synthetic_instance(37, 1, 20260929)
        self.assertEqual(set(saved), {
            "product_id", "m", "a", "a0", "w", "delta", "master_seed", "instance_id",
        })
        for name in ("m", "a", "w", "delta"):
            np.testing.assert_array_equal(saved[name], expected[name])
        for name in ("m", "a", "a0", "w", "delta"):
            self.assertEqual(saved[name].dtype, np.dtype("float64"))
        for name in ("product_id", "master_seed", "instance_id"):
            self.assertEqual(saved[name].dtype, np.dtype("int64"))
        for name in ("a0", "w", "delta", "master_seed", "instance_id"):
            self.assertEqual(saved[name].shape, ())
        np.testing.assert_array_equal(saved["product_id"], np.arange(1, 38))
        self.assertEqual(saved["a0"], 1.)
        self.assertEqual(saved["master_seed"], 20260929)
        self.assertEqual(saved["instance_id"], 1)
        self.assertTrue(np.any(saved["m"] != np.round(saved["m"], 2)))

    def test_exactness_default_ids(self):
        paths = generation.generate("exactness", output=self.output)
        self.assertEqual(len(paths), 100)
        self.assertEqual(paths[0].name, "n16_i01.npz")
        self.assertEqual(paths[-1].name, "n16_i100.npz")

    def test_comparison_default_instance_count(self):
        paths = generation.generate("comparison", n=[3], output=self.output)
        self.assertEqual(len(paths), 20)
        self.assertEqual(paths[-1].name, "n3_i20.npz")

    def test_ablation_reuses_comparison_files_without_rewrite(self):
        comparison = generation.generate("comparison", n=[11], instances=1, output=self.output)
        path, = comparison
        before = path.read_bytes(), path.stat().st_mtime_ns
        ablation = generation.generate("ablation", n=[11], instances=1, output=self.output)
        self.assertEqual(ablation, comparison)
        self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)
        self.assertFalse((self.output / "ablation").exists())

    def test_all_and_duplicate_sizes_generate_once(self):
        with patch.object(generation, "generate_synthetic_instance", wraps=generate_synthetic_instance) as helper:
            paths = generation.generate("all", n=[7, 7], instances=2, output=self.output)
        self.assertEqual(len(paths), 6)
        self.assertEqual(len(set(paths)), 6)
        self.assertEqual(helper.call_count, 6)
        self.assertEqual({p.parent.name for p in paths}, {"exactness", "comparison", "batch"})
        first = self.load(self.output / "comparison" / "n7_i01.npz")
        second = self.load(self.output / "comparison" / "n7_i02.npz")
        self.assertFalse(np.array_equal(first["m"], second["m"]))

    def test_batch_uses_helpers_and_fixed_prefixes(self):
        with patch.object(batch, "request_arrays", wraps=batch.request_arrays) as requests, \
                patch.object(batch, "ensure_data", wraps=batch.ensure_data) as ensure:
            path, = generation.generate("batch", n=[9], instances=1, output=self.output)
            ensure.assert_called_once()
            before = path.read_bytes(), path.stat().st_mtime_ns
            generation.generate("batch", n=[9], instances=1, output=self.output)
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)
        requests.assert_any_call(9, 1, 20261013, 500)
        self.assertEqual(requests.call_count, 2)
        self.assertEqual(ensure.call_count, 2)
        saved = self.load(path)
        self.assertEqual(set(saved), {
            "product_id", "m", "a", "a0", "z", "delta", "master_seed", "instance_id",
        })
        for name in ("z", "delta"):
            self.assertEqual(saved[name].shape, (500,))
            self.assertEqual(saved[name].dtype, np.dtype("float64"))
        for q in batch.Q_VALUES:
            z, delta = batch.request_arrays(9, 1, 20261013, q)
            np.testing.assert_array_equal(saved["z"][:q], z)
            np.testing.assert_array_equal(saved["delta"][:q], delta)

    def test_changed_cached_fields_are_rejected_without_replacement(self):
        for experiment in ("comparison", "batch"):
            directory = self.output / experiment
            path, = generation.generate(experiment, n=[5], instances=1, output=directory)
            original = self.load(path)
            for field in original:
                with self.subTest(experiment=experiment, field=field):
                    modified = {key: value.copy() for key, value in original.items()}
                    modified[field] = modified[field] + np.asarray(1, dtype=modified[field].dtype)
                    np.savez_compressed(path, **modified)
                    before = path.read_bytes()
                    with self.assertRaisesRegex(RuntimeError, "does not match protocol"):
                        generation.generate(experiment, n=[5], instances=1, output=directory)
                    self.assertEqual(path.read_bytes(), before)

    def test_bad_schema_dtype_shape_and_corrupt_archive(self):
        path, = generation.generate("exactness", n=[5], instances=1, output=self.output)
        original = self.load(path)
        missing = dict(original)
        del missing["a0"]
        variants = (
            missing,
            dict(original, extra=np.asarray(1)),
            dict(original, a0=np.asarray(1., dtype=np.float32)),
            dict(original, w=original["w"].reshape(1)),
        )
        for modified in variants:
            np.savez_compressed(path, **modified)
            before = path.read_bytes()
            with self.assertRaises(RuntimeError):
                generation.generate("exactness", n=[5], instances=1, output=self.output)
            self.assertEqual(path.read_bytes(), before)
        path.write_bytes(b"not an npz archive")
        with self.assertRaises(RuntimeError):
            generation.generate("exactness", n=[5], instances=1, output=self.output)
        self.assertEqual(path.read_bytes(), b"not an npz archive")

    def test_cli_rejects_nonpositive_counts(self):
        for args in (["--n", "0"], ["--n", "-1"], ["--n"],
                     ["--instances", "0"], ["--instances", "-2"], ["--n", "1.5"]):
            with self.subTest(args=args), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    generation.build_parser().parse_args(args)
                self.assertEqual(error.exception.code, 2)

    def test_api_rejects_invalid_inputs_before_creating_directory(self):
        target = self.output / "invalid"
        for kwargs in ({"experiment": "unknown"}, {"n": []}, {"n": [0]}, {"instances": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generation.generate(output=target, **kwargs)
        self.assertFalse(target.exists())

    def test_module_cli(self):
        result = subprocess.run(
            [sys.executable, "-m", "data.generate", "--experiment", "all",
             "--n", "4", "6", "--instances", "1", "--output", str(self.output)],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("6 input files", result.stdout)
        self.assertEqual(len(list(self.output.rglob("*.npz"))), 6)


if __name__ == "__main__":
    unittest.main()
