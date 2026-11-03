"""Export trained model to ONNX, quantise to INT8, report size and CPU latency (numbers for the video).
Usage: python export_int8.py qg.pt CALIB_DIR   (CALIB_DIR = ~100 val images, any classes)
"""
import sys, os, time, glob
import numpy as np, torch, torch.nn as nn
from torchvision import models
from PIL import Image
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
from onnxruntime.quantization.shape_inference import quant_pre_process

MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

def prep(path, size):
    im = np.asarray(Image.open(path).convert("RGB").resize((size, size)), np.float32) / 255.0
    return ((im - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)

class Reader(CalibrationDataReader):
    def __init__(self, files, size): self.it = iter([{"input": prep(f, size)} for f in files])
    def get_next(self): return next(self.it, None)

def bench(path, x, n=50):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    for _ in range(5): s.run(None, {"input": x})
    t = time.perf_counter()
    for _ in range(n): s.run(None, {"input": x})
    return (time.perf_counter() - t) / n * 1000

def main(ckpt, calib):
    c = torch.load(ckpt, map_location="cpu"); size = c["size"]
    m = models.mobilenet_v3_small(); m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(c["classes"]))
    m.load_state_dict(c["state"]); m.eval()
    torch.onnx.export(m, torch.randn(1, 3, size, size), "qg_fp32.onnx", input_names=["input"],
                      output_names=["logits"], opset_version=17, dynamic_axes={"input": {0: "b"}})
    quant_pre_process("qg_fp32.onnx", "qg_pre.onnx")
    files = sorted(glob.glob(f"{calib}/**/*.png", recursive=True) + glob.glob(f"{calib}/**/*.jpg", recursive=True))[:100]
    quantize_static("qg_pre.onnx", "qg_int8.onnx", Reader(files, size), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
    x = prep(files[0], size)
    for f in ["qg_fp32.onnx", "qg_int8.onnx"]:
        print(f"{f}: {os.path.getsize(f)/1e6:.2f} MB, {bench(f, x):.1f} ms/image on this CPU")
    print("Now re-check val accuracy with the INT8 model before quoting it in the video.")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
