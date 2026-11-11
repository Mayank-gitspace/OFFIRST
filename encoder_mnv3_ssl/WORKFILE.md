
## 2026-11-09 — Kaggle smoke + full launch (IST)

**Pre-push:** `teamoffirst/team-offirst-encoder-baseline` is NOT accessible from teamoffirst (permission denied; the baseline lives under `jeetmuley/`). v4 also ran with `kernel_sources: []` and recomputed the split. Script falls back to iterstrat MultilabelStratifiedKFold(5, shuffle, seed 42) — same call as the baseline script. Dropped the kernel source in `build_kernel.py` → rebuilt; .py files byte-identical (md5 unchanged), only metadata changed. Consequence: no split-CSV cross-check and supervised-baseline reference (c) is skipped in the full run too.

**Smoke** https://www.kaggle.com/code/teamoffirst/team-offirst-encoder-byol-mnv3-smoke — pushed 13:42, COMPLETE 14:27 IST, wall 2551 s, no errors. Outputs in `kaggle_smoke_out/`.
- Tesla T4, torch 2.11.0+cu130, timm 1.0.30, AMP on. Split recomputed(iterstrat), folds 5×384; SSL 1,152 imgs (folds 1–3).
- Cache 640 px built in 185 s.
- Batch probe: bs64 OOM, bs48 OOM, **bs48 + grad ckpt fits** (peak 4.42 GiB). → training batch is 48, not 64 (ladder fallback, no LR rescale).
- Throughput: 12.7 img/s wall (90 s/epoch, 24 steps); GPU-only median step ≈ 35–36 img/s → **dataloader/CPU-degrader bound**.
- Loss 0.897 → 0.751; std_z 0.915/0.905; std_h 0.714/0.717; erank 247/246 (of 256). No collapse.
- SMOKE-ONLY probes (2 epochs, relaxed label thresholds, 3 few-shot draws; not results): inner (fold 4) ep2 target LP clean 0.903, c→deg 0.750, deg→deg 0.896, kNN 0.822/0.716; ep2 online LP 0.886 / c→deg 0.812 / deg→deg 0.889. Dev (fold 0) selected ep1 target LP 0.876 / c→deg 0.739 / deg→deg 0.857; online last 0.881 / 0.797 / 0.861. Few-shot ODC/ODP/ARMD k1/5/10 ≈ 0.55–0.64.
- Each evaluate call 240–285 s on Kaggle. Checkpoints best/last/ep001.pth + embeddings written.
- Script ESTIMATE for full: 1.79 h (excludes ImageNet-init reference evals, which smoke skips, and the heavier full eval settings) → my projection ~2–3 h.

**Full** https://www.kaggle.com/code/teamoffirst/team-offirst-encoder-byol-mnv3 — pushed 14:27 IST, RUNNING at 14:28 IST. Projected finish ~16:30–17:30 IST (well under 12 h limit). Not waited on.

## 2026-11-09 — Full run results (IST)
- Kernel COMPLETE; outputs downloaded to `full_out/` (run.log, summary.json, ckpts ep010/020/030/last/best, embeddings). Wall 6,065 s (1 h 41 min); training 3,700 s; bs48 + grad-ckpt (peak 4.42 GiB), 12.5 img/s wall vs ~30 img/s GPU → degrader-bound. No errors / no collapse (std_z 0.85→0.95, std_h ~0.70, loss 1.29→0.12).
- Selection (target, fold 4, mean LP clean & c→deg): ep20 0.8928 / ep30 0.8981 / **ep40 0.9002 → best = ep40** (= last).
- Dev fold 0 (recomputed split): ep40 target LP 0.905, c→deg 0.875, deg→deg 0.890, kNN 0.880/0.872, rare 0.551/0.589/0.634; online LP 0.912. ImageNet MNV3: LP 0.886, c→deg 0.865, kNN 0.845/0.804, rare 0.543/0.562/0.618.
- Verified: MNV3 split is byte-identical to v4 in-run split (downloaded v4 split_used.csv; md5 78d12095…).
- CPU re-score on locked A/B split (`locked_rescore/`): ep40 target LP 0.917, rare 0.553/0.607/0.640, ARMD 10-shot 0.789 (B 0.911, 0.622/0.705/0.734, ARMD 0.969; v4 ep20 0.921, 0.589/0.688/0.725, ARMD 0.943). 228/384 locked fold-0 imgs were in MNV3 SSL set.
- Takeaway: near-teacher on common LP, clear robustness gain over ImageNet on degraded retrieval, weak on rare/ARMD few-shot. Full write-up + caveats: `RESULTS.md`. KD not run → no KD-vs-SSL claim.
