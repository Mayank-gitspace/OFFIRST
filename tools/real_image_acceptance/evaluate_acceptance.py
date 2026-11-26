#!/usr/bin/env python3
"""Audit a frozen real-image OFFIRST acceptance run (Python 3 stdlib only).

This tool scores already-recorded Python/Android outputs. It does not preprocess
images, run ONNX, tune thresholds, or substitute synthetic data for real cases.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

FIELDS = [
    "sample_id", "image_sha256", "image_path", "tensor_sha256", "partition", "dataset",
    "site_group", "device_group", "patient_group_id", "eye", "dr_severity",
    "dr_label", "modality", "gradability", "quality_issue", "validation_outcome",
    "quality_outcome", "probability", "decision", "model_version",
    "threshold", "platform", "network_mode", "case_review", "run_id", "repeat_index",
]
TOLERANCE = 1e-5


def fail(message: str) -> None:
    raise ValueError(message)


def number(row: dict[str, str], field: str) -> float:
    try:
        result = float(row[field])
    except (ValueError, TypeError):
        fail(f"{row.get('sample_id')}: invalid {field}")
    if not math.isfinite(result):
        fail(f"{row.get('sample_id')}: non-finite {field}")
    return result


def auc(pairs: list[tuple[float, int]]) -> float | None:
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return None
    ordered = sorted((score, label) for score, label in pairs)
    rank_sum = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        average_rank = ((i + 1) + j) / 2
        rank_sum += average_rank * sum(label for _, label in ordered[i:j])
        i = j
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total == 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def rows_for(path: Path, root: Path, check_hashes: bool) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(FIELDS) - set(reader.fieldnames or [])
        if missing:
            fail("manifest missing columns: " + ", ".join(sorted(missing)))
        rows = list(reader)
    if not rows:
        fail("manifest has no image rows; no acceptance result can be computed")

    seen: dict[tuple[str, str], str] = {}
    digest_owner: dict[str, tuple[str, str]] = {}
    repeat_keys: set[tuple[str, str, str, str]] = set()
    for row in rows:
        for key in FIELDS:
            row[key] = (row.get(key) or "").strip()
        sample, part = row["sample_id"], row["partition"]
        if not sample or part not in {"validation", "test"}:
            fail("every row needs a sample_id and partition validation|test")
        key = (part, sample)
        digest = row["image_sha256"].lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            fail(f"{sample}: image_sha256 must be a SHA-256 hex digest")
        if key in seen and seen[key] != digest:
            fail(f"{sample}: repeated rows have different image hashes")
        seen[key] = digest
        if digest in digest_owner and digest_owner[digest] != key:
            fail(f"duplicate image hash across sample IDs or partitions: {sample}")
        digest_owner[digest] = key
        if row["eye"] not in {"OD", "OS", "unknown", "not_applicable"}:
            fail(f"{sample}: eye must be OD, OS, unknown, or not_applicable")
        if row["modality"] not in {"fundus", "non_fundus"}:
            fail(f"{sample}: modality must be fundus or non_fundus")
        if row["gradability"] not in {"gradable", "ungradable", "not_applicable"}:
            fail(f"{sample}: invalid gradability")
        if row["quality_issue"] not in {"none", "blur", "dark", "overexposed", "poor_field", "other", "not_applicable"}:
            fail(f"{sample}: invalid quality_issue")
        if row["modality"] == "non_fundus" and (row["gradability"] != "not_applicable" or row["quality_issue"] != "not_applicable"):
            fail(f"{sample}: non-fundus row must use not_applicable quality labels")
        if row["modality"] == "fundus" and row["gradability"] == "ungradable" and row["quality_issue"] in {"none", "not_applicable"}:
            fail(f"{sample}: ungradable fundus needs a documented quality issue")
        if row["modality"] == "fundus" and row["gradability"] == "gradable" and row["quality_issue"] != "none":
            fail(f"{sample}: gradable fundus must use quality_issue=none")
        if row["dr_label"] not in {"positive", "negative", "not_applicable"}:
            fail(f"{sample}: invalid dr_label")
        if row["partition"] == "test" and not row["patient_group_id"]:
            fail(f"{sample}: test rows require a pseudonymous patient_group_id for CIs")
        if row["decision"] not in {"positive", "negative", "not_classified", "error"}:
            fail(f"{sample}: invalid decision")
        if row["platform"] not in {"python", "android"}:
            fail(f"{sample}: platform must be python or android")
        if row["network_mode"] not in {"airplane_mode", "online", "not_applicable"}:
            fail(f"{sample}: invalid network_mode")
        if row["case_review"] not in {"reviewed", "pending"}:
            fail(f"{sample}: case_review must be reviewed or pending")
        if row["validation_outcome"] not in {"accept", "reject", "retake", "error"}:
            fail(f"{sample}: invalid validation_outcome")
        if row["quality_outcome"] not in {"gradable", "retake", "not_applicable", "error"}:
            fail(f"{sample}: invalid quality_outcome")
        if not row["model_version"] or not row["run_id"]:
            fail(f"{sample}: model_version and run_id are required")
        repeat = row["repeat_index"]
        if not repeat.isdigit() or int(repeat) < 1:
            fail(f"{sample}: repeat_index must start at 1")
        repeat_key = (part, sample, row["platform"], repeat)
        if repeat_key in repeat_keys:
            fail(f"{sample}: duplicate repeat_index for platform {row['platform']}")
        repeat_keys.add(repeat_key)
        if part == "test" and row["platform"] == "android" and not row["device_group"]:
            fail(f"{sample}: Android test row requires device_group")
        if row["image_path"]:
            image = (root / row["image_path"]).resolve()
            if not image.is_file():
                fail(f"{sample}: image file missing: {image}")
            if check_hashes:
                actual = hashlib.sha256(image.read_bytes()).hexdigest()
                if actual != digest:
                    fail(f"{sample}: image hash mismatch")
        elif check_hashes:
            fail(f"{sample}: image_path required when checking image hashes")
        probability = row["probability"]
        if probability:
            score = number(row, "probability")
            if not 0 <= score <= 1:
                fail(f"{sample}: probability outside [0,1]")
            if not row["threshold"]:
                fail(f"{sample}: scored image requires its decision threshold")
            expected = "positive" if score >= number(row, "threshold") else "negative"
            if row["decision"] != expected:
                fail(f"{sample}: decision does not match probability >= recorded threshold")
        if row["threshold"]:
            threshold = number(row, "threshold")
            if not 0 <= threshold <= 1:
                fail(f"{sample}: threshold outside [0,1]")
        if row["partition"] == "test" and row["threshold"] == "":
            fail(f"{sample}: test row has no locked threshold")
        tensor_hash = row["tensor_sha256"].lower()
        if row["probability"] and (len(tensor_hash) != 64 or any(c not in "0123456789abcdef" for c in tensor_hash)):
            fail(f"{sample}: scored rows require a preprocessed tensor SHA-256")
    return rows


def primary_rows(rows: list[dict[str, str]], partition: str, platform: str) -> list[dict[str, str]]:
    selected = [r for r in rows if r["partition"] == partition and r["platform"] == platform and r["repeat_index"] == "1"]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["partition"] == partition and row["platform"] == platform:
            grouped[row["sample_id"]].append(row)
    if set(r["sample_id"] for r in selected) != set(grouped):
        fail(f"{partition}: every image must have a repeat_index=1 row")
    return selected


def audit_input(rows: list[dict[str, str]]) -> dict[str, object]:
    counts: Counter[str] = Counter()
    failures: list[str] = []
    for r in rows:
        kind = r["modality"]
        counts[f"{kind}_n"] += 1
        if kind == "non_fundus":
            good = r["gradability"] == "not_applicable" and r["dr_label"] == "not_applicable" and r["validation_outcome"] == "reject" and r["quality_outcome"] == "not_applicable" and not r["probability"] and r["decision"] == "not_classified"
            if not good:
                failures.append(r["sample_id"])
            counts["non_fundus_correctly_rejected"] += int(good)
        elif r["gradability"] == "ungradable":
            good = r["dr_label"] == "not_applicable" and r["validation_outcome"] == "accept" and r["quality_outcome"] == "retake" and not r["probability"] and r["decision"] == "not_classified"
            if not good:
                failures.append(r["sample_id"])
            counts["ungradable_correctly_retake"] += int(good)
        elif r["gradability"] == "gradable":
            counts["gradable_fundus_n"] += 1
            if r["dr_label"] not in {"positive", "negative"}:
                fail(f"{r['sample_id']}: gradable fundus requires DR ground truth")
            # Any validator/quality abstention remains an end-to-end failure and
            # is not removed from the DR confusion-matrix denominator.
            if r["validation_outcome"] == "accept" and r["quality_outcome"] == "gradable" and r["probability"] and r["decision"] in {"positive", "negative"}:
                counts["gradable_fundus_scored"] += 1
            else:
                counts["gradable_fundus_unscored"] += 1
        else:
            fail(f"{r['sample_id']}: fundus gradability cannot be not_applicable")
    return {
        "counts": dict(counts),
        "non_fundus_rejection_pass": counts["non_fundus_n"] > 0 and counts["non_fundus_correctly_rejected"] == counts["non_fundus_n"],
        "ungradable_retake_pass": counts["ungradable_fundus_n"] > 0 and counts["ungradable_correctly_retake"] == counts["ungradable_fundus_n"],
        "fail_sample_ids": sorted(set(failures)),
    }


def classification(rows: list[dict[str, str]], threshold: float, groups: set[str] | None = None) -> dict[str, object]:
    tp = fp = tn = fn = abstain_pos = abstain_neg = 0
    scored: list[tuple[float, int]] = []
    severity: dict[str, Counter[str]] = defaultdict(Counter)
    for r in rows:
        if r["modality"] != "fundus" or r["gradability"] != "gradable":
            continue
        if groups is not None and r["patient_group_id"] not in groups:
            continue
        label = int(r["dr_label"] == "positive")
        classified = r["validation_outcome"] == "accept" and r["quality_outcome"] == "gradable" and r["decision"] in {"positive", "negative"} and bool(r["probability"])
        if classified:
            score = number(r, "probability")
            decision = int(r["decision"] == "positive")
            scored.append((score, label))
            correct_decision = decision == label
        else:
            decision = 0
            correct_decision = False
            if label:
                abstain_pos += 1
            else:
                abstain_neg += 1
        if label and classified and decision:
            tp += 1
        elif label and classified:
            fn += 1
        elif not label and classified and decision:
            fp += 1
        elif not label and classified:
            tn += 1
        if not classified:
            if label:
                fn += 1
            else:
                fp += 1
        grade = r["dr_severity"] or "unspecified"
        severity[grade]["n"] += 1
        severity[grade]["correct"] += int(classified and correct_decision)
        severity[grade]["scored"] += int(classified)
        if label:
            severity[grade]["positive"] += 1
            severity[grade]["tp"] += int(classified and decision == 1)
            severity[grade]["fn"] += int(not classified or decision == 0)
        else:
            severity[grade]["negative"] += 1
            severity[grade]["tn"] += int(classified and decision == 0)
            severity[grade]["fp"] += int(not classified or decision == 1)
    total = tp + fp + tn + fn
    pos = tp + fn
    neg = tn + fp
    acc_n = tp + tn
    auc_value = auc(scored)
    return {
        "n_gradable_fundus": total,
        "n_scored": len(scored),
        "coverage": len(scored) / total if total else None,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "abstain_positive_included_as_fn": abstain_pos, "abstain_negative_included_as_fp": abstain_neg},
        "accuracy": acc_n / total if total else None,
        "accuracy_95ci_wilson": wilson(acc_n, total),
        "sensitivity": tp / pos if pos else None,
        "sensitivity_95ci_wilson": wilson(tp, pos),
        "specificity": tn / neg if neg else None,
        "specificity_95ci_wilson": wilson(tn, neg),
        "auroc_scored_cases_only": auc_value,
        "severity": {k: {"n": v["n"], "scored": v["scored"], "correct": v["correct"], "accuracy_end_to_end": v["correct"] / v["n"] if v["n"] else None, "sensitivity": v["tp"] / v["positive"] if v["positive"] else None, "specificity": v["tn"] / v["negative"] if v["negative"] else None, "tp": v["tp"], "fn": v["fn"], "tn": v["tn"], "fp": v["fp"]} for k, v in sorted(severity.items())},
    }


def bootstrap_auc(rows: list[dict[str, str]], reps: int, seed: int) -> list[float] | None:
    patients: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for r in rows:
        if r["modality"] == "fundus" and r["gradability"] == "gradable" and r["validation_outcome"] == "accept" and r["quality_outcome"] == "gradable" and r["probability"] and r["decision"] in {"positive", "negative"}:
            patients[r["patient_group_id"]].append((number(r, "probability"), int(r["dr_label"] == "positive")))
    ids = sorted(patients)
    if len(ids) < 2:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(reps):
        sample = [rng.choice(ids) for _ in ids]
        paired = [item for patient in sample for item in patients[patient]]
        value = auc(paired)
        if value is not None:
            values.append(value)
    if not values:
        return None
    values.sort()
    return [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]


def bootstrap_metric_intervals(rows: list[dict[str, str]], threshold: float, reps: int, seed: int) -> dict[str, list[float] | None]:
    patients: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        patients[row["patient_group_id"]].append(row)
    ids = sorted(patients)
    values: dict[str, list[float]] = {name: [] for name in ("accuracy", "sensitivity", "specificity", "auroc_scored_cases_only")}
    if len(ids) < 2:
        return {name: None for name in values}
    rng = random.Random(seed + 1)
    for _ in range(reps):
        sample = [row for patient in (rng.choice(ids) for _ in ids) for row in patients[patient]]
        metrics = classification(sample, threshold)
        for name in values:
            value = metrics[name]
            if value is not None:
                values[name].append(value)
    result: dict[str, list[float] | None] = {}
    for name, data in values.items():
        data.sort()
        result[name] = [data[int(0.025 * (len(data) - 1))], data[int(0.975 * (len(data) - 1))]] if data else None
    return result


def determinism(rows: list[dict[str, str]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        if r["platform"] == "android" and r["modality"] == "fundus" and r["gradability"] == "gradable" and r["probability"]:
            grouped[r["sample_id"]].append(r)
    compared = failures = 0
    maximum = 0.0
    failed_ids = []
    for sample, copies in grouped.items():
        if len(copies) < 2:
            continue
        scores = [number(r, "probability") for r in copies if r["probability"]]
        if len(scores) != len(copies):
            failures += 1
            failed_ids.append(sample)
            continue
        delta = max(scores) - min(scores)
        maximum = max(maximum, delta)
        compared += 1
        if delta > TOLERANCE:
            failures += 1
            failed_ids.append(sample)
    android_scored = {r["sample_id"] for r in rows if r["platform"] == "android" and r["modality"] == "fundus" and r["gradability"] == "gradable" and r["probability"]}
    return {"images_with_repeats": compared, "android_scored_images": len(android_scored), "all_android_scored_images_repeated": compared == len(android_scored), "max_probability_range": maximum, "tolerance": TOLERANCE, "pass": compared > 0 and compared == len(android_scored) and failures == 0, "fail_sample_ids": sorted(failed_ids)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--image-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--locked-threshold", required=True, type=float)
    parser.add_argument("--output", type=Path, default=Path("acceptance_report.json"))
    parser.add_argument("--check-image-hashes", action="store_true")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261126)
    args = parser.parse_args()
    if not 0 <= args.locked_threshold <= 1 or args.bootstrap_reps < 100:
        parser.error("threshold must be in [0,1] and bootstrap reps at least 100")
    try:
        all_rows = rows_for(args.manifest, args.image_root.resolve(), args.check_image_hashes)
        versions = {r["model_version"] for r in all_rows}
        if versions != {args.model_version}:
            fail(f"model versions {sorted(versions)} do not equal requested frozen model {args.model_version}")
        test = primary_rows(all_rows, "test", "android")
        validation = primary_rows(all_rows, "validation", "android")
        test_python = primary_rows(all_rows, "test", "python")
        validation_python = primary_rows(all_rows, "validation", "python")
        if not test or not validation:
            fail("both validation and untouched test partitions are required")
        for r in (row for row in all_rows if row["partition"] == "test"):
            if abs(number(r, "threshold") - args.locked_threshold) > 1e-15:
                fail(f"{r['sample_id']}: test threshold differs from the prelocked threshold")
        # Repeated runs for one image must preserve its partition and labels.
        for partition in ("validation", "test"):
            for rows in (r for r in all_rows if r["partition"] == partition):
                peers = [r for r in all_rows if r["partition"] == partition and r["sample_id"] == rows["sample_id"] and r["platform"] == rows["platform"]]
                invariant = ("image_sha256", "dr_label", "dr_severity", "modality", "gradability", "quality_issue", "patient_group_id", "model_version", "threshold", "tensor_sha256")
                if any(any(p[k] != rows[k] for k in invariant) for p in peers):
                    fail(f"{rows['sample_id']}: repeated rows changed labels, image, model, or threshold")
        test_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for r in test:
            for key in ("dataset", "device_group", "site_group", "eye"):
                test_groups[f"{key}:{r[key] or 'unspecified'}"].append(r)
        report: dict[str, object] = {
            "evidence_type": "real_image_acceptance_manifest; no inference performed by this evaluator",
            "model_version": args.model_version,
            "locked_test_threshold": args.locked_threshold,
            "positive_comparison": ">=",
            "bootstrap": {"unit": "patient_group_id", "repetitions": args.bootstrap_reps, "seed": args.seed},
            "repeat_tolerance": TOLERANCE,
            "validation_input_audit": audit_input(validation),
            "validation_python_input_audit": audit_input(validation_python),
            "test_input_audit": audit_input(test),
            "test_python_input_audit": audit_input(test_python),
            "validation_metrics_descriptive_only": classification(validation, args.locked_threshold),
            "test_metrics": classification(test, args.locked_threshold),
            "test_auroc_patient_bootstrap_95ci": bootstrap_auc(test, args.bootstrap_reps, args.seed),
            "test_patient_bootstrap_95ci": bootstrap_metric_intervals(test, args.locked_threshold, args.bootstrap_reps, args.seed),
            "test_subgroups": {key: classification(group, args.locked_threshold) for key, group in sorted(test_groups.items())},
            "determinism_test_partition": determinism([r for r in all_rows if r["partition"] == "test"]),
        }
        parity: dict[str, list[dict[str, str]]] = defaultdict(list)
        for r in all_rows:
            if r["partition"] == "test" and r["repeat_index"] == "1" and r["probability"]:
                parity[r["sample_id"]].append(r)
        parity_deltas = []
        parity_fail_ids = []
        for sample, pair in parity.items():
            by_platform = {r["platform"]: r for r in pair}
            if set(by_platform) != {"python", "android"}:
                parity_fail_ids.append(sample)
                continue
            py, android = by_platform["python"], by_platform["android"]
            if py["tensor_sha256"] != android["tensor_sha256"]:
                parity_fail_ids.append(sample)
                continue
            delta = abs(number(py, "probability") - number(android, "probability"))
            parity_deltas.append(delta)
            if delta > TOLERANCE:
                parity_fail_ids.append(sample)
        report["python_android_parity"] = {
            "paired_images": len(parity_deltas), "probability_tolerance": TOLERANCE,
            "maximum_absolute_probability_difference": max(parity_deltas, default=None),
            "pass": bool(parity_deltas) and not parity_fail_ids,
            "fail_sample_ids": sorted(set(parity_fail_ids)),
        }
        platform_samples = {
            platform: {r["sample_id"] for r in all_rows if r["partition"] == "test" and r["platform"] == platform and r["repeat_index"] == "1"}
            for platform in ("python", "android")
        }
        report["platform_case_set_parity"] = {
            "pass": bool(platform_samples["python"]) and platform_samples["python"] == platform_samples["android"],
            "python_only_sample_ids": sorted(platform_samples["python"] - platform_samples["android"]),
            "android_only_sample_ids": sorted(platform_samples["android"] - platform_samples["python"]),
        }
        failures = []
        for r in test:
            reason = None
            if r["modality"] == "non_fundus":
                if not (r["validation_outcome"] == "reject" and r["decision"] == "not_classified" and not r["probability"]):
                    reason = "non_fundus_false_acceptance"
            elif r["gradability"] == "ungradable":
                if not (r["quality_outcome"] == "retake" and r["decision"] == "not_classified" and not r["probability"]):
                    reason = "ungradable_not_retaken"
            elif r["gradability"] == "gradable":
                expected = "positive" if r["dr_label"] == "positive" else "negative"
                actual = r["decision"] if r["validation_outcome"] == "accept" and r["quality_outcome"] == "gradable" else "not_classified"
                if actual != expected:
                    reason = "false_negative" if expected == "positive" else "false_positive"
            if reason:
                failures.append({"sample_id": r["sample_id"], "image_sha256": r["image_sha256"], "failure_type": reason, "dr_severity": r["dr_severity"], "dr_label": r["dr_label"], "probability": r["probability"] or None, "decision": r["decision"], "validation_outcome": r["validation_outcome"], "quality_outcome": r["quality_outcome"], "case_review": r["case_review"]})
        report["individual_failure_cases"] = failures
        metrics = report["test_metrics"]
        assert isinstance(metrics, dict)
        input_audit = report["test_input_audit"]
        assert isinstance(input_audit, dict)
        python_input_audit = report["test_python_input_audit"]
        assert isinstance(python_input_audit, dict)
        det = report["determinism_test_partition"]
        assert isinstance(det, dict)
        gates = {
            "accuracy_ge_0.85": (metrics["accuracy"] is not None and metrics["accuracy"] >= 0.85),
            "sensitivity_ge_0.85": (metrics["sensitivity"] is not None and metrics["sensitivity"] >= 0.85),
            "specificity_ge_0.80": (metrics["specificity"] is not None and metrics["specificity"] >= 0.80),
            "auroc_ge_0.85": (metrics["auroc_scored_cases_only"] is not None and metrics["auroc_scored_cases_only"] >= 0.85),
            "non_fundus_never_classified": input_audit["non_fundus_rejection_pass"],
            "ungradable_fundus_retaken": input_audit["ungradable_retake_pass"],
            "python_non_fundus_and_ungradable_handling_matches": python_input_audit["non_fundus_rejection_pass"] and python_input_audit["ungradable_retake_pass"],
            "repeat_inference_deterministic": det["pass"],
            "both_eyes_present": {r["eye"] for r in test} >= {"OD", "OS"},
            "gradable_fundus_complete_coverage": metrics["n_scored"] == metrics["n_gradable_fundus"] and metrics["n_gradable_fundus"] > 0,
            "python_android_same_tensor_probability_parity": report["python_android_parity"]["pass"],
            "all_android_test_in_airplane_mode": all(r["network_mode"] == "airplane_mode" for r in test),
            "multiple_dr_severity_categories_present": len({r["dr_severity"] for r in test if r["modality"] == "fundus" and r["gradability"] == "gradable" and r["dr_severity"] not in {"", "unspecified", "not_applicable"}}) >= 3,
            "blur_dark_overexposed_ungradable_cases_present": {r["quality_issue"] for r in test if r["modality"] == "fundus" and r["gradability"] == "ungradable"} >= {"blur", "dark", "overexposed"},
            "python_android_test_case_sets_match": report["platform_case_set_parity"]["pass"],
            "every_test_image_individually_reviewed": all(r["case_review"] == "reviewed" for r in test),
        }
        report["acceptance_gates"] = gates
        report["acceptance_pass"] = all(gates.values())
        report["limitations"] = [
            "AUROC is computed on gradable fundus cases that produced a probability; end-to-end coverage and abstentions remain visible in confusion counts.",
            "Confidence intervals require pseudonymous patient groups and are not a substitute for external clinical validation.",
            "Android/offline workflow evidence must be entered after actual device testing; this CSV evaluator cannot prove device operation.",
            "The test partition must be locked before any model or threshold optimization; this tool never tunes a threshold.",
        ]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output), "acceptance_pass": report["acceptance_pass"], "gates": gates}, indent=2))
        return 0 if report["acceptance_pass"] else 2
    except (OSError, ValueError) as exc:
        print(f"acceptance audit blocked: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
