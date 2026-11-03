# Retina Photo Quality Guard - PoC starter (for Tue 13 - Thu 15)
Built at NIT Rourkela, for Odisha's villages. Proof of idea on public data, not a clinical device.

Setup (laptop or Colab): `pip install -r requirements.txt`

Files
- `quality_checks.py` - explainable rule checks, no training needed: too blurry (Laplacian variance), motion blur (gradient direction ratio), too dark, glare, optic disc out of frame, retina not centred. Gives the "reason" in the Retake message. Thresholds are starting guesses, tune them on 50 Good + 50 Reject real images.
- `degrade.py` - fallback dataset: turns good APTOS photos into labelled bad ones (motion blur, defocus, dark, glare, crop). Each defect = a real capture error, which is the optics angle.
- `prepare_eyeq.py` - main dataset route: sorts EyePACS images into good/usable/reject folders from EyeQ label CSVs (check the CSV column names first).
- `train.py` - MobileNetV3-Small, class-weighted loss, prints val accuracy and Reject recall per epoch. Run on Colab GPU.
- `export_int8.py` - ONNX export, INT8 static quantisation, prints MB and ms/image (the numbers for the demo slide).
- `app.py` - Gradio demo: green Upload or red Retake plus reason and latency.

Suggested order
1. Tue: get data (EyeQ subset, or APTOS + `degrade.py`), run `quality_checks.py` on a few real images and tune thresholds.
2. Wed: `train.py` on Colab, then `export_int8.py`; re-check accuracy with the INT8 model.
3. Thu: `app.py` demo, fill the numbers into `../quality_guard_video_script.md`, record.

Tested so far: `quality_checks.py` + `degrade.py` on synthetic fundus-like images (all 5 defects caught, clean images pass). `train.py`, `export_int8.py`, `app.py` are written but not yet run (no torch on the box).
