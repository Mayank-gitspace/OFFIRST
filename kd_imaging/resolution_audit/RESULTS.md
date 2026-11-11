# IDRiD resolution survival audit

**Purpose:** measure how well annotated lesions remain spatially and contrast-visible after the OFFIRST FOV-crop + `cv2.INTER_AREA` resize to model input sizes 384 / 512 / 768. Informs whether a phone student (MobileNetV3 / EfficientNet) should use 384 or larger.

**Not done here:** no model training; no degrader/threshold tuning; no claim that CNR visibility equals model detectability.

## Data

- Source on box: `<workdir>/data/idrid_seg/A. Segmentation` (IDRiD Diabetic Retinopathy Segmentation subset, CC-BY-4.0).
- Images used: **81** (54 train + 27 test JPG originals). MA masks present for 81 images; HE/EX/SE used when present.
- Jeet policy: IDRiD is **TEST-ONLY**. This audit only measures signal visibility; numbers must not be used to choose degrader parameters or CNR thresholds for training.

## Resize method (matched to pipeline)

- FOV crop + pad-to-square: same as `crop_fundus` in `encoder_byol/v4/kernels/full/team_offirst_encoder_byol_anatomy_v4.py` (threshold max-channel > 15 on a 512-max preview, morphological open, bbox crop, pad to square).
- Resize: `cv2.resize(..., (S,S), interpolation=cv2.INTER_AREA)` — same as `load_prep` / eval extract helpers.
- Masks: same crop meta; exact full-frame INTER_AREA pixel-overlap weights evaluated on each connected-component patch (no pixel-grid phase approximation); lesion pixels at S = coverage ≥ 0.5 (else single max-coverage pixel). Equivalent diameter uses area scaling `area*(S/native)^2` (matches INTER_AREA mass conservation for an isolated component).

## Degrader

- `<workdir>/kd_imaging/degrade.py` with params from `visuscout_params.json` (fitted; JSON exposure=1.04, not the 1.12 code default).
- Order: apply degrader on the **FOV-cropped native square**, then resize (blur sigma scales with width/1024 inside `degrade`). Native-first order follows demo.py, but crop-before-degrade is an audit convention, not an implemented student training pipeline. Demo degrades the uncropped rectangle; this changes spatial calibration slightly.
- Params were **not** modified.

## Visibility proxy

- CNR on green channel: `|mean(G_lesion) − mean(G_ring)| / std(G_ring)`, ring = dilated lesion \\ lesion (tight window around each lesion).
- **CNR ≥ 2 is an arbitrary heuristic** for 'visible'. Also report ≥1 and ≥3.
- This is a **proxy**, not proof a CNN can or cannot detect the lesion.

## Caveats

- Ring radius is 5 native pixels, minimum 2 output pixels. Thus its physical support changes with resolution; resizing also smooths background variance. CNR can rise after blur/downsampling and is NOT a monotonic contrast-survival score. Do not interpret visible fractions as sensitivity or use them to select an input size.
- IDRiD segmentation images are lesion-rich; at-least-one-MA rates saturate and are not mild-DR sensitivity estimates.
- v4 actually caches at 512 then resizes to 384; this audit directly resizes native crops at each size to isolate final resolution, not the exact two-stage cache path.

- IDRiD camera ≠ RFMiD / Visuscout; absolute rates may not transfer.
- Degrader fit on 24 Visuscout frames — approximate portable look only.
- Compute cost scales ~(S/384)^2 (512≈1.78×, 768≈4× vs 384).
- Test-only usage; do not tune on these outcomes.

## Summary table (lesion-level)

| type | size | cond | n | med diam px | % diam<1 | med CNR | % CNR≥1 | % CNR≥2 | % CNR≥3 |
|------|------|------|---|-------------|----------|---------|---------|---------|---------|
| EX | 384 | clean | 11343 | 1.87 | 9.0 | 2.16 | 82.2 | 54.6 | 28.1 |
| EX | 384 | degraded | 11343 | 1.87 | 9.0 | 1.71 | 73.4 | 41.0 | 16.9 |
| EX | 512 | clean | 11343 | 2.50 | 1.9 | 2.34 | 88.7 | 61.5 | 30.6 |
| EX | 512 | degraded | 11343 | 2.50 | 1.9 | 1.79 | 78.4 | 42.5 | 16.0 |
| EX | 768 | clean | 11343 | 3.74 | 0.1 | 2.37 | 92.9 | 64.6 | 29.0 |
| EX | 768 | degraded | 11343 | 3.74 | 0.1 | 1.79 | 80.6 | 41.5 | 14.3 |
| EX | original | clean | 11343 | 16.74 | 0.0 | 1.76 | 90.2 | 37.1 | 10.6 |
| EX | original | degraded | 11343 | 16.74 | 0.0 | 1.59 | 77.7 | 32.6 | 10.2 |
| HE | 384 | clean | 1899 | 5.07 | 0.0 | 1.85 | 84.7 | 43.8 | 12.7 |
| HE | 384 | degraded | 1899 | 5.07 | 0.0 | 1.72 | 81.7 | 38.4 | 9.5 |
| HE | 512 | clean | 1899 | 6.76 | 0.0 | 1.77 | 83.0 | 40.1 | 9.5 |
| HE | 512 | degraded | 1899 | 6.76 | 0.0 | 1.68 | 80.5 | 35.7 | 8.0 |
| HE | 768 | clean | 1899 | 10.14 | 0.0 | 1.66 | 81.1 | 32.9 | 6.4 |
| HE | 768 | degraded | 1899 | 10.14 | 0.0 | 1.58 | 77.8 | 30.4 | 6.0 |
| HE | original | clean | 1899 | 45.22 | 0.0 | 1.14 | 61.4 | 9.3 | 0.8 |
| HE | original | degraded | 1899 | 45.22 | 0.0 | 1.38 | 72.0 | 21.0 | 3.8 |
| MA | 384 | clean | 3497 | 1.93 | 2.3 | 2.30 | 90.7 | 62.2 | 23.7 |
| MA | 384 | degraded | 3497 | 1.93 | 2.3 | 1.95 | 85.6 | 47.8 | 12.7 |
| MA | 512 | clean | 3497 | 2.57 | 0.5 | 2.24 | 93.1 | 61.7 | 19.3 |
| MA | 512 | degraded | 3497 | 2.57 | 0.5 | 1.90 | 87.6 | 45.1 | 9.3 |
| MA | 768 | clean | 3497 | 3.85 | 0.1 | 2.02 | 93.3 | 50.8 | 10.9 |
| MA | 768 | degraded | 3497 | 3.85 | 0.1 | 1.79 | 86.9 | 39.3 | 6.7 |
| MA | original | clean | 3497 | 17.26 | 0.0 | 1.30 | 73.8 | 11.0 | 0.8 |
| MA | original | degraded | 3497 | 17.26 | 0.0 | 1.53 | 80.1 | 25.3 | 3.6 |
| SE | 384 | clean | 150 | 11.81 | 0.0 | 2.17 | 97.3 | 60.7 | 19.3 |
| SE | 384 | degraded | 150 | 11.81 | 0.0 | 2.15 | 95.3 | 60.0 | 20.0 |
| SE | 512 | clean | 150 | 15.75 | 0.0 | 2.12 | 94.7 | 56.0 | 18.7 |
| SE | 512 | degraded | 150 | 15.75 | 0.0 | 2.09 | 94.7 | 55.3 | 18.0 |
| SE | 768 | clean | 150 | 23.62 | 0.0 | 1.98 | 92.7 | 49.3 | 14.7 |
| SE | 768 | degraded | 150 | 23.62 | 0.0 | 2.00 | 92.7 | 50.0 | 16.0 |
| SE | original | clean | 150 | 104.96 | 0.0 | 1.55 | 86.0 | 29.3 | 6.7 |
| SE | original | degraded | 150 | 104.96 | 0.0 | 1.82 | 91.3 | 44.0 | 14.0 |

## Per-image: fraction with ≥1 visible MA (mild-DR relevant)

| size | cond | CNR thr | n images with MA | frac with ≥1 visible MA |
|------|------|---------|------------------|-------------------------|
| original | clean | 1.0 | 81 | 100.0% |
| original | degraded | 1.0 | 81 | 100.0% |
| 384 | clean | 1.0 | 81 | 100.0% |
| 384 | degraded | 1.0 | 81 | 100.0% |
| 512 | clean | 1.0 | 81 | 100.0% |
| 512 | degraded | 1.0 | 81 | 100.0% |
| 768 | clean | 1.0 | 81 | 100.0% |
| 768 | degraded | 1.0 | 81 | 100.0% |
| original | clean | 2.0 | 81 | 74.1% |
| original | degraded | 2.0 | 81 | 96.3% |
| 384 | clean | 2.0 | 81 | 100.0% |
| 384 | degraded | 2.0 | 81 | 100.0% |
| 512 | clean | 2.0 | 81 | 100.0% |
| 512 | degraded | 2.0 | 81 | 100.0% |
| 768 | clean | 2.0 | 81 | 98.8% |
| 768 | degraded | 2.0 | 81 | 100.0% |
| original | clean | 3.0 | 81 | 22.2% |
| original | degraded | 3.0 | 81 | 63.0% |
| 384 | clean | 3.0 | 81 | 96.3% |
| 384 | degraded | 3.0 | 81 | 91.4% |
| 512 | clean | 3.0 | 81 | 100.0% |
| 512 | degraded | 3.0 | 81 | 87.7% |
| 768 | clean | 3.0 | 81 | 81.5% |
| 768 | degraded | 3.0 | 81 | 77.8% |

## Files

- `run_audit.py` — this script
- `lesion_level.csv` — per-lesion rows
- `summary_by_type_size.csv`
- `per_image_ma_visible.csv`
- `example_ma_crops.png` — example MA crops at original/768/512/384 clean+degraded
- `RESULTS.md` — this file
