"""Completeness and path-safety tests for the independent Section 6.4 audit."""
from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import experiments.batch.audit as audit


class AuditCompletenessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name) / "project"
        self.out = self.project / "results"
        self.out.mkdir(parents=True)
        self.root_patch = patch.object(audit, "ROOT", self.project)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

        source = self.out / "source_snapshot" / "solver.py"
        source.parent.mkdir(parents=True)
        source.write_text("print('snapshot')\n", encoding="utf-8")

        input_path = self.project / "inputs" / "n2_i01.npz"
        input_path.parent.mkdir(parents=True)
        m = np.array([2.0, 3.0])
        a = np.array([0.5, 1.5])
        z = np.array([-0.25, 0.75])
        delta = np.array([1.6, 2.4])
        np.savez(input_path, m=m, a=a, z=z, delta=delta)
        product_hash = audit.digest_products({"m": m, "a": a})

        self.protocol = {
            "scales": [2],
            "q_values": [1, 2],
            "scenarios_h": {"base": 0.1},
            "instances_per_scale": 1,
            "repeats_by_q": {"1": 1, "2": 2},
            "methods": ["independent_fabs", "algorithm2"],
            "source_sha256": {"solver.py": audit.sha256_file(source)},
        }
        (self.out / "protocol.json").write_text(
            json.dumps(self.protocol), encoding="utf-8"
        )

        self.manifest_row = {
            "n": 2,
            "instance": 1,
            "seed": 7,
            "input_file": str(input_path.relative_to(self.project)),
            "input_sha256": audit.sha256_file(input_path),
            "product_sha256": product_hash,
            "request_seed": "fixture",
        }
        self._write_manifest(self.manifest_row)

        self.rows = []
        for q in self.protocol["q_values"]:
            request_hash = audit.digest_requests(
                audit.scenario_requests(z, delta, self.protocol["scenarios_h"]["base"], q)
            )
            for method in self.protocol["methods"]:
                for repeat in range(1, self.protocol["repeats_by_q"][str(q)] + 1):
                    self.rows.append(
                        {
                            "n": 2,
                            "instance": 1,
                            "scenario": "base",
                            "Q": q,
                            "method": method,
                            "repeat": repeat,
                            "status": "ok",
                            "product_sha256": product_hash,
                            "request_sha256": request_hash,
                        }
                    )
        self._write_runs(self.rows)

    def _write_manifest(self, row):
        with (self.out / "input_manifest.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    def _write_runs(self, rows):
        with (self.out / "runs_6_4.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_complete_protocol_grid_passes(self):
        report = audit.audit_results(self.out)
        self.assertTrue(report["provenance_ok"], report)
        self.assertEqual(report["rows"], 6)
        self.assertEqual(report["expected_rows"], 6)

    def test_empty_partial_duplicate_and_unknown_status_ledgers_fail(self):
        cases = {
            "empty": [],
            "partial": self.rows[:-1],
            "duplicate": self.rows + [copy.deepcopy(self.rows[0])],
            "unknown_status": [
                {**row, "status": "mystery"} if index == 0 else copy.deepcopy(row)
                for index, row in enumerate(self.rows)
            ],
        }
        for name, rows in cases.items():
            with self.subTest(name=name):
                report = audit.audit_results(
                    self.out, rows=rows, protocol=self.protocol
                )
                self.assertFalse(report["provenance_ok"], report)
                if name == "empty":
                    self.assertTrue(report["ledger_empty"])
                elif name == "partial":
                    self.assertEqual(report["ledger_missing_key_count"], 1)
                elif name == "duplicate":
                    self.assertTrue(report["ledger_duplicate_keys"])
                else:
                    self.assertTrue(report["ledger_status_errors"])

    def test_input_manifest_path_must_remain_within_root(self):
        outside = Path(self.temporary.name) / "outside.npz"
        outside.write_bytes(b"not an input")
        malicious = {
            **self.manifest_row,
            "input_file": "../outside.npz",
            "input_sha256": audit.sha256_file(outside),
        }
        self._write_manifest(malicious)

        report = audit.audit_results(self.out, rows=self.rows, protocol=self.protocol)
        self.assertFalse(report["provenance_ok"])
        self.assertEqual(report["input_path_errors"], ["../outside.npz"])

    def test_source_snapshot_path_cannot_escape_snapshot_directory(self):
        escaped = self.out / "escape.py"
        escaped.write_text("escaped\n", encoding="utf-8")
        protocol = copy.deepcopy(self.protocol)
        protocol["source_sha256"] = {"../escape.py": audit.sha256_file(escaped)}

        report = audit.audit_results(self.out, rows=self.rows, protocol=protocol)
        self.assertFalse(report["provenance_ok"])
        self.assertEqual(report["source_path_errors"], ["../escape.py"])

    def test_missing_or_unlisted_source_snapshot_fails(self):
        source = self.out / "source_snapshot" / "solver.py"
        source.unlink()
        missing = audit.audit_results(self.out, rows=self.rows, protocol=self.protocol)
        self.assertFalse(missing["provenance_ok"])
        self.assertEqual(missing["source_hash_mismatches"], ["solver.py"])

        no_sources = copy.deepcopy(self.protocol)
        no_sources["source_sha256"] = {}
        unlisted = audit.audit_results(self.out, rows=self.rows, protocol=no_sources)
        self.assertFalse(unlisted["provenance_ok"])
        self.assertTrue(unlisted["protocol_errors"])


if __name__ == "__main__":
    unittest.main()
