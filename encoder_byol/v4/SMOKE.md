# OFFIRST BYOL anatomy v4 — smoke

## Build
```bash
cd /workspace/zeiss/encoder_byol/v4
python3 build_kernel.py          # v2 kernel -> kernels/{smoke,full}/team_offirst_encoder_byol_anatomy_v4.py
python3 -m py_compile kernels/*/team_offirst_encoder_byol_anatomy_v4.py
/workspace/kenv/bin/python test_vicreg.py   # VICReg unit check
```

## Local CPU smoke (box; no CUDA) — 16 SSL imgs, 2 ep, bs 4, eval_epochs [1] exercised
```bash
cd /workspace/zeiss/encoder_byol/v4
HF_HOME=/workspace/hfcache TOB_INPUT=/workspace/rfmid_train_dl TOB_OUT=$PWD/localtest/out \
TOB_TMP=/workspace/byol_local/tmp TOB_SMOKE_N=16 TOB_SMOKE_EPOCHS=2 TOB_WORKERS=2 TOB_PRETRAINED=1 \
/workspace/kenv/bin/python kernels/smoke/team_offirst_encoder_byol_anatomy_v4.py 2>&1 | tee localtest/smoke.log
```
Env `kenv` (torch 2.14 CPU) got opencv-python-headless, timm, scikit-learn, pyyaml, iterative-stratification via `uv pip`.
No split CSV is mounted locally -> folds recomputed with iterstrat (seed 42); on Kaggle the baseline kernel's
`split_5fold_seed42.csv` is used.

Check in log: anchor_frac ≈ random 0.5 / disc+macula ≈ 0.4 / fallback ≈ 0.1 (low-conf ~20% of discs);
`var`/`cov` VICReg terms finite and falling; `EVAL[.../ep1_online]` + `ep1_multiregion` + final `multiregion_online` lines;
`local_views_grid.png` shows only wide disc/macula crops.

## Kaggle GPU smoke (needs explicit OK — not launched)
```bash
cd /workspace/zeiss/encoder_byol/v4/kernels/smoke && kaggle kernels push -p .
```
Full: `kernels/full` (120 ep; ~CONFIG estimate from smoke ESTIMATE line). **Do not push full without Jeet's OK.**
