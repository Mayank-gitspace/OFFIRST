# MobileNetV3-Large BYOL (anatomy v4 views + Visuscout degradation) — full Kaggle run results

Source of truth (all numbers below are copied from these files, nothing estimated):
- Kernel: https://www.kaggle.com/code/teamoffirst/team-offirst-encoder-byol-mnv3 — status COMPLETE (checked 2026-11-09 IST)
- Local copy: `full_out/` → `out/run.log`, `out/summary.json`, `out/byol_mnv3_anatomy_v4/{epoch_evals.json,history.json,report.json}`, kernel log `full_out/team-offirst-encoder-byol-mnv3.log` (decoded: `full_out/kernel_stream.txt`)
- Extra CPU re-score on the locked A/B split (done on box, no training): `locked_rescore/` (`rescore.py`, `rescore.log`, `rescore.json`)

## 1. Run health
| item | value |
|---|---|
| Hardware | Tesla T4 (1 GPU), torch 2.11.0+cu130, timm 1.0.30, AMP |
| Backbone | `mobilenetv3_large_100` ImageNet-pretrained, 960-d GAP; 6.52 M params incl. heads (log says backbone 2.97 M) |
| Batch | probe: bs64 OOM, bs48 OOM, **bs48 + grad-ckpt fits (peak 4.42 GiB)** → trained at **48**, 24 steps/epoch, LR **not** rescaled (config said 64) |
| SSL data | 1,152 RFMiD train images = folds 1–3 of recomputed split; fold 4 (inner val) and fold 0 (dev) excluded from SSL |
| Throughput | 12.25–12.72 img/s wall (mean 12.46), ~91–94 s/epoch; GPU-only median step 29.4–30.2 img/s → **CPU/dataloader (degrader) bound** |
| Runtime | wall **6,065 s (1 h 41 min)**: cache 175 s, ImageNet ref evals 365 s, training+in-run evals 5,486 s (pure training 3,700 s) |
| Errors | none. No OOM during training (only the expected batch-probe OOMs). Only warnings: supervised baseline (c) not mounted → skipped; HF unauthenticated; nbconvert SyntaxWarnings |
| Degradation mix (every epoch) | g0 0.00, g1 0.83–0.88, locals 0.59–0.61, mean alpha 0.49–0.51; views random 0.50 / disc 0.20 / macula 0.20 / fallback 0.10 |
| Checkpoints | `ep010.pth`, `ep020.pth`, `ep030.pth`, `last.pth` (=ep40), `best.pth` (=ep40); embeddings (online/target × clean/degraded) are from best = ep40 |

Loss / collapse monitors (from `run.log` / `history`):
| epoch | 1 | 2 | 5 | 10 | 20 | 30 | 40 |
|---|---|---|---|---|---|---|---|
| loss | 1.291 | 0.804 | 0.448 | 0.214 | 0.146 | 0.125 | 0.119 |
| std_z | 0.849 | 0.904 | 0.912 | 0.930 | 0.944 | 0.948 | 0.948 |
| std_h | 0.710 | 0.716 | 0.713 | 0.704 | 0.696 | 0.696 | 0.696 |

Loss falls fast to ep10 then plateaus (~0.12 from ep30). std_z rises, std_h flat ~0.70 → **no collapse**.

kNN monitor (inner fold 4, every 5 ep): ep5 0.858 / deg-q 0.783 · ep10 0.841 / 0.818 · ep15 0.844 / 0.829 · ep20 0.848 / 0.836 · ep25 0.854 / 0.840 · ep30 0.857 / 0.837 · ep35 0.858 / 0.843 · ep40 0.858 / 0.837. Erank 248.5 (ep5) → 235.1 (ep40) of 256 proj dims.

Embedding diagnostics (960-d backbone feature, query fold):
| encoder | split | erank clean / deg | paired − random cos (centred) | clean→deg top-1 retrieval |
|---|---|---|---|---|
| ImageNet MNV3 | inner | 180.6 / 182.1 | 0.143 (0.631) | 0.734 |
| ep20 target / online | inner | 236.6 / 239.0 · 234.0 / 234.0 | 0.408 (0.720) · 0.441 (0.770) | 0.987 · 0.997 |
| ep30 target / online | inner | 235.5 / 236.0 · 233.7 / 233.1 | 0.428 (0.746) · 0.450 (0.792) | 0.987 · 0.997 |
| ep40 target / online | inner | 235.1 / 236.2 · 233.1 / 232.9 | 0.431 (0.750) · 0.450 (0.796) | 0.990 · 1.000 |
| ImageNet MNV3 | dev | 178.6 / 179.9 | 0.143 (0.632) | 0.706 |
| ep40 target / online | dev | 233.8 / 234.2 · 231.5 / 230.7 | 0.431 (0.752) · 0.450 (0.797) | 0.997 · 1.000 |

## 2. Main table (in-run protocol, 512 eval input)
LP = linear-probe mean AUROC over common labels (10 diseases; Disease_Risk reported separately); c→deg = probe fit on clean support, scored on degraded query; deg→deg = degraded support and query; kNN clean / degraded-query mAUROC; rare = logreg few-shot mean of **ODC / ODP / ARMD**, 20 draws.
Inner = SSL folds 1–3 as support → fold 4 query. Dev = folds 1–4 support → fold 0 query (**disclosed dev split, not a blind test**; fold 0 was never in SSL, but it is the split v4 used for its choices).

| model | split | LP clean | LP c→deg | LP deg→deg | DR-risk | kNN clean / deg | rare 1 / 5 / 10 | rare c→deg 1 / 5 / 10 | ARMD 1 / 5 / 10 |
|---|---|---|---|---|---|---|---|---|---|
| ImageNet MNV3 (0 training) | inner | 0.907 | 0.871 | 0.904 | 0.937 | 0.855 / 0.767 | 0.546 / 0.585 / 0.613 | 0.550 / 0.588 / 0.619 | 0.587 / 0.616 / 0.659 |
| ep20 target | inner | 0.919 | 0.866 | 0.901 | 0.934 | 0.848 / 0.836 | 0.557 / 0.590 / 0.629 | 0.558 / 0.589 / 0.624 | 0.621 / 0.662 / 0.730 |
| ep20 online | inner | 0.921 | 0.883 | 0.905 | 0.934 | 0.859 / 0.843 | 0.557 / 0.591 / 0.632 | 0.558 / 0.591 / 0.630 | 0.624 / 0.671 / 0.738 |
| ep30 target | inner | 0.921 | 0.875 | 0.905 | 0.933 | 0.857 / 0.837 | 0.557 / 0.591 / 0.631 | 0.559 / 0.591 / 0.628 | 0.624 / 0.669 / 0.737 |
| ep30 online | inner | 0.924 | 0.892 | 0.905 | 0.934 | 0.864 / 0.851 | 0.558 / 0.594 / 0.636 | 0.560 / 0.594 / 0.635 | 0.626 / 0.675 / 0.744 |
| ep40 target | inner | 0.922 | 0.879 | 0.907 | 0.933 | 0.858 / 0.837 | 0.558 / 0.592 / 0.633 | 0.560 / 0.593 / 0.633 | 0.626 / 0.671 / 0.740 |
| ep40 online | inner | 0.924 | 0.895 | 0.907 | 0.934 | 0.862 / 0.855 | 0.559 / 0.596 / 0.639 | 0.561 / 0.597 / 0.640 | 0.628 / 0.679 / 0.748 |
| ImageNet MNV3 (0 training) | dev | 0.886 | 0.865 | 0.886 | 0.917 | 0.845 / 0.804 | 0.543 / 0.562 / 0.618 | 0.541 / 0.567 / 0.619 | 0.587 / 0.618 / 0.699 |
| **ep40 target (selected)** | dev | **0.905** | 0.875 | 0.890 | 0.928 | 0.880 / 0.872 | **0.551 / 0.589 / 0.634** | 0.553 / 0.598 / 0.642 | 0.611 / 0.691 / 0.774 |
| ep40 online | dev | 0.912 | 0.886 | 0.894 | 0.928 | 0.887 / 0.864 | 0.555 / 0.595 / 0.643 | 0.556 / 0.602 / 0.650 | 0.616 / 0.698 / 0.786 |

Few-shot std over draws is ~0.04–0.05 (see `rescore.log`), so 0.01 rare differences are noise.

**Checkpoint selection:** pre-registered rule = target encoder, inner fold 4, mean(LP clean, LP clean→deg). Scores ep20 0.8928 → ep30 0.8981 → ep40 0.9002 → **ep40 selected** (= last epoch). Fold 0 was not used for selection. The gain from ep20→40 is ≤0.007 and within fold/draw noise; the curve is flat, not still climbing steeply. Online is ≥ target on almost every metric (esp. LP c→deg +0.01–0.02), but selection was fixed to target before the run.

### What SSL bought over ImageNet-only MobileNetV3 (same split, same protocol)
- Dev, ep40 target − ImageNet: LP +0.019, LP c→deg +0.010, deg→deg +0.004, kNN deg-query +0.068, rare +0.008 / +0.027 / +0.016, ARMD 10-shot +0.075.
- Inner, ep40 target − ImageNet: LP +0.015, c→deg +0.008, deg→deg +0.003, kNN deg-query +0.070, rare +0.012 / +0.007 / +0.020.
- Clearest effects: degraded-query retrieval (kNN deg 0.80→0.87 dev; clean→deg top-1 0.71→0.997) and embedding spread (erank 179→234). Clean LP gain is modest; rare few-shot gain is small relative to draw SD.

## 3. Comparison to ConvNeXt-T BYOL v4 (teacher candidate) and anatomy baseline B
### 3a. Same split as v4's in-run numbers
The MNV3 recomputed split is **byte-identical** to v4's in-run `split_used.csv` (md5 78d12095…, verified by downloading v4's split from Kaggle). So fold-0 dev numbers below share query and support images. Probe/few-shot code is the v4 code (diff only adds the degraded-query variants).

| model (fold 0, v4/MNV3 recomputed split) | LP | DR-risk | rare 1 / 5 / 10 |
|---|---|---|---|
| ImageNet ConvNeXt-T @384 (v4 kernel) | 0.887 | 0.931 | 0.556 / 0.570 / 0.621 |
| ImageNet MNV3 @512 | 0.886 | 0.917 | 0.543 / 0.562 / 0.618 |
| v4 ep20 online (picked post hoc by fold-0 kNN → leaky selection) | 0.928 | 0.950 | 0.608 / 0.686 / 0.742 |
| v4 ep120 target | 0.913 | 0.918 | 0.553 / 0.640 / 0.685 |
| **MNV3 ep40 target (selected on fold 4)** | **0.905** | 0.928 | **0.551 / 0.589 / 0.634** |
| MNV3 ep40 online | 0.912 | 0.928 | 0.555 / 0.595 / 0.643 |

### 3b. Locked A/B split (phase0 protocol, CPU re-score of the saved ep40 embeddings)
Sanity: re-scoring on MNV3's own split reproduces the in-run numbers exactly (target LP 0.905, rare 0.551 / 0.589 / 0.634).

| model (locked fold 0) | LP | DR-risk | rare 1 / 5 / 10 | ARMD 1 / 5 / 10 |
|---|---|---|---|---|
| B anatomy baseline (bar) | 0.911 | 0.930 | 0.622 / 0.705 / 0.734 | 0.799 / 0.922 / 0.969 |
| v4 ep20 target | 0.921 | 0.941 | 0.589 / 0.688 / 0.725 | 0.749 / 0.865 / 0.943 |
| **MNV3 ep40 target** | **0.917** | 0.926 | **0.553 / 0.607 / 0.640** | 0.647 / 0.690 / 0.789 |
| MNV3 ep40 online | 0.916 | 0.931 | 0.556 / 0.612 / 0.648 | 0.651 / 0.698 / 0.801 |

Read: on the common-label probe, MNV3 SSL is close to the ConvNeXt v4 teacher (−0.004 locked; −0.008 vs v4 ep120 target on the shared split) and above B on the locked split. On rare few-shot it is clearly behind both v4 ep20 and B (locked 10-shot 0.640 vs 0.725 / 0.734), driven mainly by **ARMD** (10-shot 0.79 vs 0.94 / 0.97). Rare scores barely beat ImageNet-only MNV3.

### Non-comparabilities (must accompany any comparison)
- **SSL data:** MNV3 SSL used folds 1–3 (1,152 imgs); v4 used folds 1–4 (1,536); B used its own locked split. MNV3 had 25% less SSL data.
- **Split:** recomputed iterstrat split (identical to v4 in-run, verified) — **not** the locked A/B split. Locked-split rows: 228 / 384 locked-fold-0 images were in MNV3's label-free SSL set (v4: 308 / 384). That can only help MNV3/v4 vs B there.
- **Input:** MNV3 trained and evaluated at 512 global / 224 local from a 640 cache; v4 trained at 384 (phase0 v4 rows were re-extracted from a 512 cache; v4 ImageNet ref at 384).
- **Training:** batch 48 + grad-ckpt (LR not rescaled from 64), 40 epochs, no VICReg, Visuscout degradation in views; v4: batch 48, 120 epochs, light VICReg, no Visuscout degrader; v4 ep20 was chosen with fold-0 kNN (leaky), MNV3 ep40 by fold-4 rule.
- **Label sets:** in-run common set = 10 diseases (+Disease_Risk); locked phase0 set = 11 diseases (adds LS) — same for all locked rows.
- **Single run / seed / fold.** Fold-to-fold noise is visible: ImageNet MNV3 LP is 0.907 on fold 4 vs 0.886 on fold 0. Treat ±0.02 LP and ±0.03 rare as noise for unpaired comparisons. No paired bootstrap done for MNV3.
- Supervised reference (c) skipped (baseline kernel under `jeetmuley` not accessible from teamoffirst).
- **Nothing here says anything about KD vs SSL** — KD has not been run. No phone latency measured. No IDRiD / Visuscout-real evaluation; "degraded" = our synthetic Visuscout degrader (eval alpha 1.0, seed 2026).

## 4. Honest take
- Run healthy: no errors, no collapse, finished in 1 h 41 min on one T4, CPU-degrader bound (12.5 img/s vs ~30 img/s GPU).
- Direct SSL helps MNV3 over ImageNet mainly in **degradation robustness of the embedding** (kNN on degraded queries +0.07, near-perfect clean↔degraded matching) and modestly on clean LP (+0.015–0.019).
- Common-label probe is near the ConvNeXt v4 teacher; **rare-disease few-shot (esp. ARMD) is the weak spot**, well below v4 and B.
- ep40 selected by the pre-set rule; the epoch curve is flat (20→40 within noise). Online encoder is marginally better than target.

## Files
- `full_out/out/run.log`, `full_out/out/summary.json`, `full_out/out/byol_mnv3_anatomy_v4/{epoch_evals.json,history.json,report.json,best.pth,last.pth,ep010/020/030.pth,embeddings_*.npy,embedding_ids.csv}`, `full_out/out/split_used.csv`, `full_out/out/{degrade_eval_examples.png,local_views_grid.png}`
- `locked_rescore/{rescore.py,evallib.py,rescore.log,rescore.json,v4_summary_for_cfg.json}`

## Per-disease AUROC, locked A/B fold 0 (added Nov 9 17:2x IST; MNV3 = ep40 target; full JSON locked_rescore/mnv3_ep40_target_locked_full.json)
| disease (query pos) | LP MNV3 | LP v4 ep20 | LP B | 10-shot MNV3 | 10-shot v4 ep20 | 10-shot B |
|---|---|---|---|---|---|---|
| Disease_Risk | 0.926 | 0.941 | 0.930 | – | – | – |
| DR | 0.962 | 0.958 | 0.958 | – | – | – |
| ARMD (20) | 0.968 | 0.989 | 0.992 | 0.789 | 0.943 | 0.969 |
| MH (64) | 0.980 | 0.973 | 0.975 | 0.943 | 0.852 | 0.887 |
| DN (28) | 0.842 | 0.840 | 0.876 | 0.622 | 0.690 | 0.686 |
| MYA (21) | 0.976 | 0.987 | 0.988 | 0.953 | 0.960 | 0.964 |
| BRVO (14) | 0.946 | 0.979 | 0.982 | 0.814 | 0.875 | 0.933 |
| TSLN (37) | 0.947 | 0.928 | 0.945 | 0.836 | 0.821 | 0.875 |
| LS (10) | 0.926 | 0.933 | 0.871 | 0.876 | 0.875 | 0.836 |
| ODC (56) | 0.865 | 0.896 | 0.862 | 0.597 | 0.663 | 0.630 |
| ODP (13) | 0.742 | 0.698 | 0.637 | 0.533 | 0.568 | 0.603 |
| ODE (12) | 0.929 | 0.951 | 0.938 | 0.882 | 0.909 | 0.862 |
