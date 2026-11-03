"""Gradio demo: drop a fundus photo -> green Upload or red Retake + reason.
Usage: python app.py qg_int8.onnx   (classes order must match training: printed by train.py)
"""
import sys, time
import numpy as np, cv2, gradio as gr, onnxruntime as ort
from quality_checks import check

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qg_int8.onnx"
CLASSES = ["good", "reject"]  # edit to match train.py output, e.g. ["good", "reject", "usable"]
SIZE = 224
sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

def run(img_rgb):
    t = time.perf_counter()
    x = cv2.resize(img_rgb, (SIZE, SIZE)).astype(np.float32) / 255.0
    x = ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    logits = sess.run(None, {"input": x})[0][0]; p = np.exp(logits - logits.max()); p /= p.sum()
    label = CLASSES[int(p.argmax())]
    rules = check(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    ms = (time.perf_counter() - t) * 1000
    retake = label == "reject" or rules["verdict"] == "Retake"
    reasons = ", ".join(rules["reasons"]) or ("model flagged low quality" if retake else "")
    head = f"## {'🔴 Retake: ' + reasons if retake else '🟢 Upload'}"
    body = f"Model: {label} ({p.max():.0%}) | checks: {rules['metrics']} | {ms:.0f} ms on this laptop"
    return head + "\n\n" + body

gr.Interface(run, gr.Image(type="numpy", label="Fundus photo"), gr.Markdown(),
             title="Retina Photo Quality Guard",
             description="Built at NIT Rourkela, for Odisha's villages. Proof of idea on public data, not a clinical device.").launch()
