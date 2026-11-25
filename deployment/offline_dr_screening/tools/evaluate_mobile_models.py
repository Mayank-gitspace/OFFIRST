"""Prepare labelled evaluation and representative calibration for OFFIRST LiteRT candidates."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

THRESHOLD = 0.48
REFERENCE_HASH = "b8fc10191f69faef16f1dc92d08bf9542ee7e26f885767942e28292c5201dd90"
NHWC = (1, 512, 512, 3)
NCHW = (1, 3, 512, 512)


def records(manifest, labelled=False):
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("The manifest is empty")
    paths = set()
    for row in rows:
        if not row["patient_id"].strip():
            raise ValueError("Patient identifiers are required to check partition separation")
        path = (manifest.parent / row["image"]).resolve(strict=True)
        if path in paths:
            raise ValueError("Duplicate image path in manifest")
        paths.add(path)
        row["path"] = path
        if labelled:
            row["label"] = int(row["label"])
            if row["label"] not in (0, 1):
                raise ValueError("Expected binary ground truth labels")
    return rows


def input_tensor(path):
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
        values = np.asarray(rgb, dtype=np.float32) / np.float32(255)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return ((values - mean) / std)[None, ...]


def candidate(path):
    from ai_edge_litert.interpreter import Interpreter

    runtime = Interpreter(model_path=str(path), num_threads=2)
    runtime.allocate_tensors()
    (input_info,), (output_info,) = runtime.get_input_details(), runtime.get_output_details()

    def predict(nhwc):
        shape = tuple(input_info["shape"])
        value = nhwc if shape == NHWC else nhwc.transpose(0, 3, 1, 2)
        if shape not in (NHWC, NCHW):
            raise ValueError(f"Unexpected candidate input shape: {shape}")
        dtype = input_info["dtype"]
        if np.issubdtype(dtype, np.integer):
            scale, zero = input_info["quantization"]
            if scale <= 0:
                raise ValueError("Missing per-tensor input quantization")
            bounds = np.iinfo(dtype)
            value = np.clip(np.rint(value / scale) + zero, bounds.min, bounds.max)
        runtime.set_tensor(input_info["index"], value.astype(dtype))
        runtime.invoke()
        result = runtime.get_tensor(output_info["index"]).item()
        if np.issubdtype(output_info["dtype"], np.integer):
            scale, zero = output_info["quantization"]
            if scale <= 0:
                raise ValueError("Missing per-tensor output quantization")
            result = (result - zero) * scale
        # A quantized sigmoid can represent an endpoint with small rounding error.
        if not np.isfinite(result) or result < -0.004 or result > 1.004:
            raise ValueError("Candidate did not return a DR probability")
        return float(np.clip(result, 0, 1))

    return predict


def metrics(labels, decisions):
    positive = labels == 1
    negative = ~positive
    if not positive.any() or not negative.any():
        raise ValueError("Evaluation set must contain both ground truth classes")
    return {
        "accuracy": float(np.mean(decisions == positive)),
        "sensitivity": float(np.mean(decisions[positive])),
        "specificity": float(np.mean(~decisions[negative])),
    }


def compare(args):
    import onnxruntime as ort

    if hashlib.sha256(args.onnx.read_bytes()).hexdigest() != REFERENCE_HASH:
        raise ValueError("Unexpected reference ONNX checksum")
    evaluation = records(args.evaluation, labelled=True)
    if args.calibration:
        calibration = records(args.calibration)
        for field in ("patient_id", "path"):
            if {r[field] for r in evaluation} & {r[field] for r in calibration}:
                raise ValueError(f"Evaluation overlaps calibration on {field}")
    reference = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    predictors = {str(path): candidate(path) for path in args.candidates}
    scores = {key: [] for key in predictors}
    expected = []
    for row in evaluation:
        value = input_tensor(row["path"])
        result = reference.run(["dr_probability"], {"fundus_image": value.transpose(0, 3, 1, 2)})[0].item()
        if not np.isfinite(result) or not 0 <= result <= 1:
            raise ValueError("Reference did not return a DR probability")
        expected.append(result)
        for key, predict in predictors.items():
            scores[key].append(predict(value))

    labels = np.array([row["label"] for row in evaluation])
    expected = np.array(expected)
    reference_decisions = expected >= THRESHOLD
    baseline = metrics(labels, reference_decisions)
    report = {"threshold": THRESHOLD, "images": len(labels), "reference": baseline, "candidates": {}}
    for key, values in scores.items():
        values = np.array(values)
        error = np.abs(expected - values)
        decisions = values >= THRESHOLD
        disagreement = decisions != reference_decisions
        observed = metrics(labels, decisions)
        report["candidates"][key] = {
            "bytes": Path(key).stat().st_size,
            "mean_absolute_probability_error": float(error.mean()),
            "max_absolute_probability_error": float(error.max()),
            "threshold_disagreements": int(disagreement.sum()),
            "below_to_at_or_above": int((disagreement & decisions).sum()),
            "at_or_above_to_below": int((disagreement & reference_decisions).sum()),
            **observed,
            "delta_from_onnx": {name: value - baseline[name] for name, value in observed.items()},
        }
    print(json.dumps(report, indent=2))


def smoke(args):
    """Check conversion and output plumbing with synthetic tensors only."""
    import onnxruntime as ort

    if hashlib.sha256(args.onnx.read_bytes()).hexdigest() != REFERENCE_HASH:
        raise ValueError("Unexpected reference ONNX checksum")
    reference = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    gradient = np.broadcast_to(
        np.linspace(-2, 2, 512, dtype=np.float32)[None, None, :, None], NHWC
    ).copy()
    inputs = {
        "zeros": np.zeros(NHWC, dtype=np.float32),
        "gradient": gradient,
        "noise": np.random.default_rng(2026).uniform(-2, 2, NHWC).astype(np.float32),
    }
    predictions = {str(path): candidate(path) for path in args.candidates}
    report = {"scope": "synthetic conversion smoke test", "threshold": THRESHOLD, "candidates": {}}
    for path, predict in predictions.items():
        probes = {}
        for name, value in inputs.items():
            expected = reference.run(
                ["dr_probability"], {"fundus_image": value.transpose(0, 3, 1, 2)}
            )[0].item()
            observed = predict(value)
            probes[name] = {
                "onnx_probability": expected,
                "candidate_probability": observed,
                "absolute_error": abs(expected - observed),
                "threshold_disagreement": (expected >= THRESHOLD) != (observed >= THRESHOLD),
            }
        report["candidates"][path] = {"bytes": Path(path).stat().st_size, "probes": probes}
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("calibrate")
    prepare.add_argument("manifest", type=Path)
    prepare.add_argument("output", type=Path, help="Private output .npy outside packaged assets")
    evaluate = commands.add_parser("compare")
    evaluate.add_argument("onnx", type=Path)
    evaluate.add_argument("evaluation", type=Path)
    evaluate.add_argument("candidates", type=Path, nargs="+")
    evaluate.add_argument("--calibration", type=Path)
    probe = commands.add_parser("smoke")
    probe.add_argument("onnx", type=Path)
    probe.add_argument("candidates", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "compare":
        compare(args)
    elif args.command == "smoke":
        smoke(args)
    else:
        rows = records(args.manifest)
        data = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.float32, shape=(len(rows), *NHWC[1:]))
        for index, row in enumerate(rows):
            data[index] = input_tensor(row["path"])[0]
        data.flush()


if __name__ == "__main__":
    main()
