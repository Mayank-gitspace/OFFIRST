# BYOL anatomy v4 results (Kaggle full run)

Source of truth:
- Kernel: [teamoffirst/team-offirst-encoder-byol-anatomy-v4](https://www.kaggle.com/code/teamoffirst/team-offirst-encoder-byol-anatomy-v4) Version 1 · COMPLETE
- Local copy: `full_out/` (`report.json`, `epoch_evals.json`, `summary.json`, `run.log`)
- Protocol: target-EMA embeddings; RFMiD fold 0 query / folds 1–4 support; common linear-probe mAUROC; rare headline few-shot **ODC / ODP / ARMD** (MH = media haze, excluded); logreg readout
- A / B / v2last from `v2/free_checks/check1_report.json` (same protocol; check1 common set also includes LS)

## Setup
- Views: 1 disc + 1 macula + 2 random at 2.5–3.2 DD; low-conf disc/mac → random (~20.5% low-conf)
- Uniform weak colour jitter; camera blur/JPEG kept; light VICReg on projector
- 120 epochs; forced eval @ 70 & 120; batch 48; wall ~3.2 h on T4×2

## Linear probe (target-EMA, common labels, fold 0)
| run | mean AUROC | DiseaseRisk |
|---|---|---|
| A random | 0.898 | 0.927 |
| B anatomy v1 | 0.911 | 0.930 |
| anatomy v2 last | 0.921 | 0.940 |
| **v4 target ep120** | **0.913** | **0.918** |
| v4 online ep70 | 0.916 | 0.922 |
| v4 online ep120 | 0.908 | 0.911 |
| v4 best_ep20 online | 0.928 | 0.950 |
| v4 multi-region online ep120 | 0.904 | 0.909 |
| ImageNet init (this kernel) | 0.887 | 0.931 |

Success bar: ≥ B (0.911), ideally near v2 (0.921). **Clears B by +0.002; short of v2 by −0.008.**

## Rare-disease headline few-shot (ODC / ODP / ARMD, logreg)
| run | 1-shot | 5-shot | 10-shot |
|---|---|---|---|
| A | 0.617 | 0.701 | 0.730 |
| B (bar) | **0.622** | **0.705** | **0.734** |
| v2 last | 0.598 | 0.679 | 0.707 |
| **v4 target ep120** | **0.553** | **0.640** | **0.686** |
| v4 online ep70 | 0.558 | 0.652 | 0.699 |
| v4 online ep120 | 0.547 | 0.630 | 0.677 |
| v4 best_ep20 online | 0.608 | 0.686 | 0.742 |
| v4 multi-region online ep120 | 0.552 | 0.643 | 0.693 |

Success bar: ≥ B 0.622 / 0.705 / 0.734. **Misses at every shot** (target −0.069 / −0.066 / −0.049).

## ep70 vs ep120
Online probe and rare few-shot both **peak earlier**: ep70 > ep120. Target at ep120 is only slightly above B on probe and worse than B/v2 on rare. Multi-region concat does not rescue ep120.

## Collapse / training health
- `std_z` stayed 0.86 → 0.99 (no collapse)
- View mix fixed: random 0.50 / disc 0.20 / macula 0.20 / fallback 0.10 (anatomy_views_frac 0.40)
- Best kNN monitor at **ep20** (mAUROC 0.911); later kNN drifted down to ~0.89

## Verdict vs WORKFILE success bar
- Probe: **pass** (≥ B), not near v2
- Rare headline: **fail**
- No collapse: **pass**
- Honest take: longer training + weak colour jitter + VICReg held the common probe near B, but rare ODC/ODP/ARMD few-shot regressed vs B and even vs v2. If one checkpoint matters for the pitch, **best_ep20 online** is the strongest probe (0.928) and closest rare numbers, still short of B at 1/5-shot.
