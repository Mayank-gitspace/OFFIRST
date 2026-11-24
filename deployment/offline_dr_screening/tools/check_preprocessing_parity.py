"""Generate shared Pillow goldens and compare actual Dart tensors via CPU ONNX."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, features

SIZE = 512
THRESHOLD = 0.48
SHA = "b8fc10191f69faef16f1dc92d08bf9542ee7e26f885767942e28292c5201dd90"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def pattern(width, height):
    y, x = np.indices((height, width))
    return np.stack(
        [(43 * x + 13 * y) % 256, (3 * x + 71 * y) % 256, (17 * (x ^ y)) % 256],
        axis=-1,
    ).astype(np.uint8)


def prepare(out, extra):
    out.mkdir(parents=True, exist_ok=True)
    for name, width, height in [
        ("down", 1291, 743), ("up", 53, 79), ("mixed", 301, 827),
        ("one_axis", 512, 539), ("identity", 512, 512), ("one_pixel", 1, 1),
    ]:
        Image.fromarray(pattern(width, height)).save(out / (name + ".png"))
    alpha = np.indices((47, 63)).sum(axis=0).astype(np.uint8) % 3 * 127
    Image.fromarray(np.dstack((pattern(63, 47), alpha))).save(out / "rgba.png")
    Image.fromarray(pattern(223, 99)[:, :, 0]).save(out / "gray.png")
    Image.fromarray((pattern(133, 71)[:, :, 0].astype(np.uint16) * 9)).save(out / "gray16.png")
    palette = Image.fromarray(pattern(73, 22)).quantize(colors=16)
    palette.save(out / "palette.png", transparency=0)
    source = Image.fromarray(pattern(773, 365))
    source.save(out / "normal-420.jpg", quality=91, subsampling=2)
    for orientation in range(2, 9):
        exif = Image.Exif()
        exif[274] = orientation
        source.save(out / f"exif-{orientation}.jpg", quality=90, subsampling=2,
                    progressive=orientation % 2 == 0, exif=exif)
    source.convert("CMYK").save(out / "cmyk.jpg", quality=89)
    generated = sorted(p for p in out.iterdir() if p.suffix in (".png", ".jpg")
                       and not p.name.startswith("retinal-"))
    inputs = generated[:]
    for i, original in enumerate(extra):
        target = out / f"retinal-{i}{original.suffix.lower()}"
        shutil.copyfile(original, target)
        inputs.append(target)
    records = []
    for path in inputs:
        with Image.open(path) as image:
            rgb = image.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
            tensor = ((np.asarray(rgb, dtype=np.float32) / 255.0 - MEAN) / STD)
            tensor.transpose(2, 0, 1).astype("<f4").tofile(out / (path.name + ".pillow.f32"))
            records.append({"name": path.name, "width": image.width, "height": image.height,
                            "mode": image.mode, "exif_orientation": image.getexif().get(274)})
    (out / "fixtures.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Prepared {len(records)} fixtures with libjpeg-turbo {features.version_feature('libjpeg_turbo')}")


def compare(out, model):
    import onnxruntime as ort

    if hashlib.sha256(model.read_bytes()).hexdigest() != SHA:
        raise ValueError("Unexpected ONNX SHA-256")
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    report = []
    for row in json.loads((out / "fixtures.json").read_text(encoding="utf-8")):
        name = row["name"]

        def load(implementation):
            return np.fromfile(out / f"{name}.{implementation}.f32", dtype="<f4").reshape(1, 3, SIZE, SIZE)

        def predict(tensor):
            return session.run(["dr_probability"], {"fundus_image": tensor})[0].item()

        expected, dart = load("pillow"), load("dart")
        reference_probability = predict(expected)
        dart_probability = predict(dart)
        report.append({
            **row,
            "tensor_max_abs_error": float(np.max(np.abs(expected - dart))),
            "tensor_values_different": int(np.count_nonzero(expected != dart)),
            "pillow_probability": reference_probability,
            "dart_probability": dart_probability,
            "probability_abs_error": abs(reference_probability - dart_probability),
            "threshold_disagreement": (reference_probability >= THRESHOLD) != (dart_probability >= THRESHOLD),
        })
    summary = {
        "onnx_sha256": SHA,
        "runtime": ort.__version__,
        "fixtures": len(report),
        "tensor_max_abs_error": max(row["tensor_max_abs_error"] for row in report),
        "tensor_values_different": sum(row["tensor_values_different"] for row in report),
        "probability_max_abs_error": max(row["probability_abs_error"] for row in report),
        "threshold_disagreements": sum(row["threshold_disagreement"] for row in report),
        "results": report,
    }
    (out / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    if summary["tensor_max_abs_error"] != 0 or summary["threshold_disagreements"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    generate = commands.add_parser("prepare")
    generate.add_argument("directory", type=Path)
    generate.add_argument("--image", type=Path, action="append", default=[])
    evaluate = commands.add_parser("compare")
    evaluate.add_argument("directory", type=Path)
    evaluate.add_argument("onnx", type=Path)
    args = parser.parse_args()
    if args.operation == "prepare":
        prepare(args.directory, args.image)
    else:
        compare(args.directory, args.onnx)
