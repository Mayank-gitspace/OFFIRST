# OFFIRST asymmetric-KD imaging: clean tabletop -> portable-like (classical, no ML)

Teacher sees clean RFMiD; student sees `degrade(clean)` of the SAME image (paired). No GANs/diffusion.

## Run (CPU, seconds)
    python3 demo.py                                   # 6-image grid + stats table -> demo/
    python3 degrade.py in.png out.png --seed 3 --leak 0.5
    python3 stats.py /path/to/folder --out s.json --csv s.csv

## Pipeline (degrade.py, `DegradeParams` dataclass, deterministic given seed+params)
approx CRF tone -> radial vignetting -> defocus blur -> colour gains -> light leak -> noise -> JPEG

| knob | default | effect |
|---|---|---|
| exposure | 0.9 | linear gain before tone curve (underexposure) |
| crf_gamma | 1.25 | parametric CRF; >1 darkens mid-tones |
| crf_toe | 0.04 | lifts blacks (washed-out shadows) |
| crf_shoulder | 0.85 | highlight roll-off start (disc hotspot compression) |
| vignette_strength | 0.45 | rim luminance = 1-strength of centre |
| vignette_power | 2.2 | falloff shape (higher = flat centre, steep rim) |
| blur_sigma | 1.2 px @1024w | defocus/vessel softening |
| gain_r/g/b | 1.08/0.92/0.85 | warm/red cast, lower green peak |
| leak_strength / angle / width | 0 (off) / random / 0.25 | warm crescent at FOV edge (environmental light leak) |
| noise_sigma | 2.0 | sensor noise (8-bit units) |
| jpeg_quality | 70 | compression artefacts |

The CRF is a parametric gamma+toe+shoulder curve, NOT a recovered camera response (that would need multi-exposure data; see Debevec / Mitsunaga-Nayar / Grossberg-Nayar EMoR in the plan).

**Defaults are Visuscout-fit (2026-11-06)** against 24 official RET-Amrita Visuscout 100 frames; see `visuscout_params.json` and `demo/degrader_fit_table.md`. Parametric CRF, not a recovered multi-exposure response.

## Stats (stats.py)
Fundus-circle aware (black corners ignored): RGB+lum 16-bin histograms, green-channel peak, mean RGB, R/G and B/G ratios, Lab L/a/b means, Sobel mean/energy/p95 (edge-gradient MTF proxy), Laplacian var, 10-bin radial profile normalized to centre. `vectorize()` gives a flat vector for distance matching.

## Next step: KD student train
1. Use fitted `DegradeParams` (defaults in `degrade.py` / `visuscout_params.json`) for paired `(clean, degrade(clean))` views.
2. At train time, sample params around the fitted values for augmentation diversity.
3. Optional: re-fit if you add more Visuscout frames under `<workdir>/data/portable/ret_amrita` (fit script not checked in; params already saved).

## Contents / results (pushed to OFFIRST)
- `degrade.py`, `stats.py`, `demo.py`, `visuscout_params.json` — degrader, image statistics, demo; fitted parameters.
- `demo/` — clean-vs-degraded grid on RFMiD + stats tables and the fit table (`degrader_fit_table.md`). The two comparison images that include real Visuscout frames are not published (third-party frames, not redistributed here); their numeric stats are in `rfmid_vs_visuscout_stats.json`.
- `resolution_audit/` — microaneurysm visibility audit on IDRiD lesion masks at portable resolution (`RESULTS.md`, `run_audit.py`, CSVs, example crops).
- Used by `encoder_mnv3_ssl/` (degradation inside SSL views) and `kd_run2/` (student-only degradation). No weights; nothing in releases.
- Absolute box paths in logs/JSON are shown as `<workdir>/...` (= repo root layout); scripts use repo-root-relative paths (`data/rfmid/...` must be supplied by you).
