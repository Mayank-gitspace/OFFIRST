"""Synthetic software-contract tests; these are not screening evidence."""

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("evaluate_acceptance.py")
FIELDS = next(csv.reader([Path(__file__).with_name("manifest_template.csv").read_text().splitlines()[0]]))
THRESHOLD = "0.3675977844516245"


def record(sample, patient, partition, platform, *, eye="OD", status="unilateral",
           site=None, probability="0.9", threshold=THRESHOLD, digest=None):
    image = digest or hashlib.sha256(("image:" + sample).encode()).hexdigest()
    tensor = hashlib.sha256(("tensor:" + sample).encode()).hexdigest()
    return dict(sample_id=sample, image_sha256=image, image_path="", tensor_sha256=tensor,
                partition=partition, dataset="synthetic-test-only", site_group=site or partition,
                device_group="device-a", patient_group_id=patient, eye=eye,
                dr_severity="moderate_npdr" if probability else "unspecified",
                dr_label="positive", modality="fundus", gradability="gradable",
                quality_issue="none", validation_outcome="accept", quality_outcome="gradable",
                probability=probability, decision="positive" if probability and float(probability) >= float(threshold) else "negative",
                model_version="frozen-test-model", threshold=threshold, platform=platform,
                network_mode="airplane_mode", case_review="reviewed", run_id="synthetic-run",
                repeat_index="1", patient_eye_status=status)


def manifest(rows, path):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class EvaluatorTests(unittest.TestCase):
    def run_eval(self, rows, *args):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest(rows, root / "manifest.csv")
            result = subprocess.run([sys.executable, str(SCRIPT), str(root / "manifest.csv"),
                                     "--model-version", "frozen-test-model", "--locked-threshold", THRESHOLD,
                                     "--bootstrap-reps", "100", "--output", str(root / "report.json"), *args],
                                    text=True, capture_output=True)
            report = json.loads((root / "report.json").read_text()) if (root / "report.json").exists() else None
            return result, report

    def base_rows(self):
        rows = []
        for partition in ("validation", "test"):
            for i in range(4):
                patient = f"{partition}-p{i}"
                for platform in ("python", "android"):
                    rows.append(record(f"{partition}-i{i}", patient, partition, platform,
                                       site=partition))
        return rows

    def test_patient_leakage_and_missing_patient_id_block(self):
        rows = self.base_rows()
        rows[-1]["patient_group_id"] = "validation-p0"
        result, _ = self.run_eval(rows)
        self.assertEqual(result.returncode, 3)
        self.assertIn("both validation and test", result.stderr)
        rows = self.base_rows(); rows[0]["patient_group_id"] = ""
        self.assertEqual(self.run_eval(rows)[0].returncode, 3)

    def test_cross_partition_eye_leakage_and_cross_platform_metadata(self):
        rows = self.base_rows()
        rows[-1]["sample_id"] = "test-p3-os"
        rows[-1]["patient_group_id"] = "validation-p3"
        rows[-1]["eye"] = "OS"
        rows[-1]["patient_eye_status"] = "bilateral"
        self.assertEqual(self.run_eval(rows)[0].returncode, 3)
        rows = self.base_rows(); rows[0]["site_group"] = "different-site"
        self.assertEqual(self.run_eval(rows)[0].returncode, 3)

    def test_external_site_leakage_and_duplicate_hash_block(self):
        rows = self.base_rows()
        rows[-1]["site_group"] = "validation"
        self.assertEqual(self.run_eval(rows, "--external-site-evaluation")[0].returncode, 3)
        rows = self.base_rows(); rows[2]["image_sha256"] = rows[0]["image_sha256"]
        self.assertEqual(self.run_eval(rows)[0].returncode, 3)

    def test_threshold_and_android_python_parity(self):
        rows = self.base_rows(); rows[0]["threshold"] = "0.48"
        self.assertEqual(self.run_eval(rows)[0].returncode, 3)
        rows = self.base_rows(); rows[-1]["probability"] = "0.1"; rows[-1]["decision"] = "negative"
        result, report = self.run_eval(rows)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(report["python_android_parity"]["pass"])

    def test_threshold_boundary_uses_greater_than_or_equal(self):
        rows = self.base_rows()
        for row in rows:
            if row["sample_id"].endswith("i0"):
                row["probability"] = THRESHOLD
                row["decision"] = "positive"
        result, report = self.run_eval(rows)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(report["test_metrics"]["confusion_matrix"]["tp"], 4)

    def test_minimums_cannot_be_lowered_and_malformed_headers_block(self):
        self.assertEqual(self.run_eval(self.base_rows(), "--min-unique-patients", "0")[0].returncode, 2)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); path = root / "bad.csv"
            path.write_text(",".join(FIELDS + [FIELDS[0]]) + "\n", encoding="utf-8")
            proc = subprocess.run([sys.executable, str(SCRIPT), str(path), "--model-version", "m",
                                  "--locked-threshold", THRESHOLD, "--bootstrap-reps", "100"],
                                 text=True, capture_output=True)
            self.assertEqual(proc.returncode, 3)
            self.assertIn("duplicate column", proc.stderr)

    def test_empty_manifest_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); path = root / "empty.csv"
            path.write_text(",".join(FIELDS) + "\n", encoding="utf-8")
            result = subprocess.run([sys.executable, str(SCRIPT), str(path), "--model-version", "m",
                                     "--locked-threshold", THRESHOLD, "--bootstrap-reps", "100",
                                     "--output", str(root / "report.json")], text=True, capture_output=True)
            self.assertEqual(result.returncode, 3)
            self.assertIn("no image rows", result.stderr)

    def test_missing_severity_blocks_gradable_fundus_record(self):
        rows = self.base_rows(); rows[0]["dr_severity"] = ""
        self.assertEqual(self.run_eval(rows)[0].returncode, 3)

    def test_bilateral_unilateral_eye_accounting(self):
        rows = self.base_rows()
        # A bilateral patient must have both eye records in each runtime.
        for row in rows:
            if row["patient_group_id"] == "test-p0":
                row["patient_eye_status"] = "bilateral"
                row["eye"] = "OD"
        rows.extend(record("test-p0-os", "test-p0", "test", platform, eye="OS", status="bilateral")
                    for platform in ("python", "android"))
        result, report = self.run_eval(rows)
        self.assertEqual(result.returncode, 2)  # software-valid but undersized/failing gates
        self.assertEqual(report["test_cohort_counts"]["bilateral_patients"], 1)
        self.assertEqual(report["test_cohort_counts"]["unilateral_patients"], 3)
        self.assertTrue(report["acceptance_gates"]["od_and_os_workflows_present"])
        rows = self.base_rows()
        for row in rows:
            if row["patient_group_id"] == "test-p0": row["patient_eye_status"] = "bilateral"
        self.assertEqual(self.run_eval(rows)[0].returncode, 3)
        rows = self.base_rows()
        for row in rows:
            if row["patient_group_id"] == "test-p0": row["patient_eye_status"] = "missing_eye_record"
        result, report = self.run_eval(rows)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(report["test_cohort_counts"]["missing_eye_records"], 1)

    def test_repeated_images_remain_in_patient_clusters(self):
        rows = self.base_rows()
        for platform in ("python", "android"):
            repeat = record("test-i0-repeat", "test-p0", "test", platform, site="test")
            repeat["image_sha256"] = hashlib.sha256(b"distinct-repeat-image").hexdigest()
            rows.append(repeat)
            rerun = dict(repeat); rerun["repeat_index"] = "2"
            rows.append(rerun)
        result, report = self.run_eval(rows)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(report["test_patient_bootstrap_95ci"]["patient_clusters"], 4)

    def test_insufficient_cohort_and_missing_outputs_fail_closed(self):
        rows = self.base_rows()
        result, report = self.run_eval(rows)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(report["acceptance_gates"]["minimum_sample_sizes_met"])
        self.assertFalse(report["acceptance_gates"]["source_image_hashes_verified"])
        rows[-1]["probability"] = ""; rows[-1]["decision"] = "error"
        result, report = self.run_eval(rows)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(report["python_android_parity"]["pass"])

    def test_abstention_is_end_to_end_error_not_a_negative(self):
        rows = self.base_rows()
        target = next(r for r in rows if r["partition"] == "test" and r["platform"] == "android" and r["sample_id"].endswith("i0"))
        target.update(probability="", decision="error", validation_outcome="error", tensor_sha256="")
        result, report = self.run_eval(rows)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(report["test_metrics"]["confusion_matrix"]["fn"], 1)
        self.assertFalse(report["acceptance_gates"]["gradable_fundus_complete_coverage"])

    def test_missing_class_bootstrap_draws_withhold_interval(self):
        rows = self.base_rows()
        for row in rows:
            if row["partition"] == "test" and not row["sample_id"].endswith("i0"):
                row["dr_label"] = "negative"; row["dr_severity"] = "no_dr"
        result, report = self.run_eval(rows)
        self.assertEqual(result.returncode, 2)
        ci = report["test_patient_bootstrap_95ci"]
        self.assertIsNone(ci["sensitivity"])
        self.assertLess(ci["valid_replicates"]["sensitivity"], 100)

    def test_patient_cluster_confidence_bounds_are_primary(self):
        rows = self.base_rows()
        # Point estimates and cluster interval are both below the sensitivity gate.
        for row in rows:
            if row["partition"] == "test" and row["sample_id"].endswith(("i0", "i1")):
                row["probability"] = "0.1"
                row["decision"] = "negative"
        result, report = self.run_eval(rows)
        self.assertEqual(result.returncode, 2)
        self.assertIn("accuracy_lcb_ge_0.85", report["acceptance_gates"])
        self.assertIsNotNone(report["test_patient_bootstrap_95ci"]["accuracy"])
        self.assertEqual(report["test_patient_bootstrap_95ci"]["valid_replicates"]["sensitivity"], 100)
        self.assertFalse(report["acceptance_gates"]["sensitivity_lcb_ge_0.85"])

    def test_degenerate_perfect_small_cohort_cannot_pass(self):
        result, report = self.run_eval(self.base_rows())
        self.assertEqual(result.returncode, 2)
        self.assertIsNone(report["test_patient_bootstrap_95ci"]["accuracy"])
        self.assertFalse(report["acceptance_gates"]["minimum_sample_sizes_met"])


if __name__ == "__main__":
    unittest.main()
