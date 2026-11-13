# MobileNetV3-Large BYOL SSL (anatomy v4 views + portable-camera degradation, RFMiD)

Direct self-supervised training of a phone-sized encoder: BYOL with the v4 anatomy-aware views (`encoder_byol/v4`), MobileNetV3-Large (`timm mobilenetv3_large_100`, ImageNet init, 960-d GAP), 512 globals / 224 locals from a 640 cache, and the classical Visuscout-style degrader (`kd_imaging/`) applied inside the view pipeline. 40 epochs on one Kaggle T4 (1 h 41 min, bs 48 + grad-ckpt). SSL on folds 1–3 (1,152 RFMiD images); fold 4 = inner validation (checkpoint selection), fold 0 = dev.

- `RESULTS.md` — full-run results (source of truth). `PRE_RUN_NOTES.md` — design notes / CPU smoke written before the Kaggle run. `WORKFILE.md` — log.
- `kernels/full`, `kernels/smoke` — Kaggle kernels; `build_kernel.py` + `src/` build them from the v4 parent (`parent_team_offirst_encoder_byol_anatomy_v4.py`); `DIFF_vs_v4.patch`.
- `degrade.py` (copy of `kd_imaging/degrade.py`), `test_degrade_equivalence.py`, `test_model_paths.py`, `bench_views.py` — unit checks / benchmarks.
- `full_out/` — full-run logs, summary/report/history/epoch evals, embeddings (`.npy`, clean + degraded, online + target). `kaggle_smoke_out/` — Kaggle smoke. `localtest/` — CPU smoke / eval-mode outputs.
- `locked_rescore/` — CPU re-score of ep40 on the locked A/B split (`rescore.py`, `evallib.py`).
- `ARGON_*` — archived design review. The maintained kernel sources are in `src/`.

## Key results (from RESULTS.md)
Selected checkpoint: **ep40 target** (pre-registered rule on fold 4). No collapse (std_z 0.95, erank 235/256).

| model (dev fold 0) | LP | LP clean→deg | kNN clean / deg-query | rare ODC/ODP/ARMD 1/5/10 |
|---|---|---|---|---|
| ImageNet MNV3 (no SSL) | 0.886 | 0.865 | 0.845 / 0.804 | 0.543 / 0.562 / 0.618 |
| **MNV3 SSL ep40 target** | **0.905** | 0.875 | 0.880 / 0.872 | 0.551 / 0.589 / 0.634 |

Locked A/B split: MNV3 ep40 LP **0.917** vs v4 ep20 0.921 vs B 0.911; rare 10-shot 0.640 vs 0.725 / 0.734; ARMD@10 0.789 vs 0.943 / 0.969. Read: common-label probe ≈ ConvNeXt teacher; rare few-shot (esp. ARMD) is the weak spot; SSL's clearest gain over ImageNet is degraded-query retrieval (+0.07 kNN). These weights are the KD student init in `kd_run2/`.

## Reproduce
- Kaggle (T4, Internet on, dataset `awsaf49/rfmid-train-dataset` only — do not attach IDRiD; the data guard aborts): push `kernels/full` (`kaggle kernels push -p kernels/full`), or `python build_kernel.py` first to rebuild.
- Locked re-score on CPU: `python encoder_mnv3_ssl/locked_rescore/rescore.py` (run from the repo root; expects `encoder_byol/v4` outputs and these embeddings).
- Box-specific absolute paths in logs are shown as `<workdir>/...` (= repo root layout).

Paths like `full_out/out/byol_mnv3_anatomy_v4/best.pth` refer to the release assets below (`.pth` files are not in git).

## Weights (GitHub Release `encoder-mnv3-ssl`)
Checkpoint dicts follow the v4 format (`online`, `target_backbone`, `target_neck`, `arch`, ...). **Recommended encoder: `encoder_mnv3_ssl_best.pth` (epoch 40, use `target_backbone`)**, scored in RESULTS.md. `last` = epoch 40; `ep010`–`ep030` intermediate. `localtest_*` are from the 1-epoch CPU smoke test (not for use; `localtest/out_smoke_prefix/` checkpoints were byte-identical duplicates and are not uploaded).

| file | size | sha256 |
|---|---|---|
| [encoder_mnv3_ssl_best.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-mnv3-ssl/encoder_mnv3_ssl_best.pth) | 48 MB | 156f4561bef2c6c6e5633d3dbe62d5f01f8040d35046caa83f6b66d59fbbbd94 |
| [encoder_mnv3_ssl_ep010.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-mnv3-ssl/encoder_mnv3_ssl_ep010.pth) | 48 MB | 09c2fba8438800c745f2e6f41e6a14e62bd21c7a80edf0cb6b768ea079745f71 |
| [encoder_mnv3_ssl_ep020.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-mnv3-ssl/encoder_mnv3_ssl_ep020.pth) | 48 MB | 6a8984d5e5d43376f0d5564a4e66d21d6ce88ac80751dea3d327961eb91a1af8 |
| [encoder_mnv3_ssl_ep030.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-mnv3-ssl/encoder_mnv3_ssl_ep030.pth) | 48 MB | f7ee7983c31439ff12cb7841e3e46e338677f84d9ab28d75733adc060a99cf7a |
| [encoder_mnv3_ssl_last.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-mnv3-ssl/encoder_mnv3_ssl_last.pth) | 48 MB | 2a2152237c78628a6472e9576923610afa0ea0f51bd648ac3a2df3810afe16ab |
| [encoder_mnv3_ssl_localtest_best.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-mnv3-ssl/encoder_mnv3_ssl_localtest_best.pth) | 48 MB | f4542a79996beb8c31bc17746cf2a4d1358a7efe6821369fc6eb25e81fa55e12 |
| [encoder_mnv3_ssl_localtest_ep001.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-mnv3-ssl/encoder_mnv3_ssl_localtest_ep001.pth) | 48 MB | 2db206e18dd729e5ea41d900ece44c03cbaa7a94fd05ca9f7a28ab0fb446433f |
| [encoder_mnv3_ssl_localtest_last.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-mnv3-ssl/encoder_mnv3_ssl_localtest_last.pth) | 48 MB | b221ba9337dbf960855a0375b7a30281784879ed4d04eaf06ab43191733b57ca |
