# Phase 0 — re-score BYOL anatomy v4 ep20 on the locked A/B split (2026-11-05, IST)

No training. CPU only on box (kenv). Eval funcs = v4 kernel `label_sets/linear_probe/fewshot/evaluate` (byte-identical to v2/check1 funcs).

## Inputs
- Locked split: `split_5fold_seed42_locked.csv` (= `/workspace/offirst_upload/encoder_byol/full_run/split_used.csv`; A/B run.log: "split from …/split_5fold_seed42.csv"). Fold 0 = query (384), folds 1–4 = support.
- v4 ep20 = `../full_out/out/byol_anatomy_v4/best.pth` (epoch 20). Features re-extracted on CPU from the 512 cache (`/workspace/byol_local/tmp/cache_512_1920.npy`): online cos vs in-run `embeddings_best_online.npy` = 0.99999.
- B = `/workspace/offirst_release_assets_byol/encoder_byol_anatomy_last.pth` target (ep150), re-extracted → **reproduces check1 exactly** (LP 0.911, rare 0.622/0.705/0.734, ARMD 10-shot 0.969).
- Sanity: in-run ep20 online numbers reproduce exactly on v4's own split (LP 0.928; rare 0.608/0.686/0.742).
- Multi-region: disc/mac crops = v4 RegionDS (2.5 DD, 224 px, box_table_used.csv; 20.5% low-conf boxes used as-is). L2-mean = mean of L2(f_full), L2(f_disc), L2(f_mac), re-L2.

## Headline (fold 0, locked split; rare = logreg mean of ODC/ODP/ARMD, 20 draws)
| run | LP mAUROC | DiseaseRisk | rare 1 / 5 / 10 | ARMD 1 / 5 / 10 |
|---|---|---|---|---|
| B (bar) | 0.911 | 0.930 | 0.622 / 0.705 / 0.734 | 0.799 / 0.922 / 0.969 |
| v4 ep20 in-run (v4 split) | 0.928 | 0.950 | 0.608 / 0.686 / 0.742 | 0.709 / 0.853 / 0.906 |
| **v4 ep20 target f_full (locked)** | **0.921** | 0.941 | **0.589 / 0.688 / 0.725** | **0.749 / 0.865 / 0.943** |
| v4 ep20 online f_full (locked) | 0.918 | 0.938 | 0.595 / 0.696 / 0.726 | 0.761 / 0.876 / 0.944 |
| v4 ep20 target L2-mean(full,disc,mac) | 0.919 | 0.933 | 0.585 / 0.689 / 0.719 | 0.731 / 0.814 / 0.888 |
| v4 ep20 online L2-mean(full,disc,mac) | 0.917 | 0.930 | 0.587 / 0.694 / 0.720 | 0.742 / 0.820 / 0.888 |
| v4 ep120 target (locked) | 0.871 | 0.884 | 0.539 / 0.624 / 0.669 | 0.622 / 0.732 / 0.853 |
| v4 ep120 multi-region concat (locked) | 0.872 | 0.879 | 0.535 / 0.622 / 0.666 | 0.614 / 0.721 / 0.838 |

10-shot per label (target f_full vs B): ODC 0.663 vs 0.630 (+0.033) · ODP 0.568 vs 0.603 (−0.035) · ARMD 0.943 vs 0.969 (−0.026).

## Paired check vs B (same 20 seeded draws, same query; 1000× query bootstrap) — `paired_vs_B.json`
| ep20 target − B | Δ | 95% CI |
|---|---|---|
| ARMD 1-shot | −0.051 | [−0.080, −0.026] |
| ARMD 5-shot | −0.057 | [−0.080, −0.038] |
| **ARMD 10-shot** | **−0.026** | **[−0.049, −0.008]** |
| ODC 10-shot | +0.033 | [+0.009, +0.057] |
| ODP 10-shot | −0.035 | [−0.098, +0.029] |
| headline 10-shot | −0.009 | [−0.033, +0.013] |
CI covers query sampling only (ARMD has just 20 query positives); not SSL-seed variance.

## Per-label linear probe (locked) — B → ep20 target f_full → ep20 target L2-mean
ARMD 0.992 → 0.989 → 0.968 · DN 0.875 → 0.840 → 0.807 · ODC 0.862 → 0.896 → 0.889 · ODP 0.638 → 0.698 → 0.781 · ODE 0.933 → 0.951 → 0.971 · TSLN 0.945 → 0.928 → 0.900 · LS 0.870 → 0.933 → 0.919

## Gates (Argon checklist)
1. **ARMD 10-shot gate:** 0.943 vs B 0.969 = −0.026 (> 0.02), paired CI excludes 0, and 1/5-shot gaps are larger (−0.05). Not split noise → **RUN v4.1**.
2. **Multi-region gate (disc ↑≥0.02 AND DN ↓≤0.005):** disc partly up (ODP LP +0.083, ODP 10-shot +0.026, ODC 10-shot +0.011) but DN LP −0.033 and ARMD 10-shot −0.054 → **FAIL; keep f_full**.

## Caveats
- Locked split moved v4 ep20 LP 0.928 → 0.921 and rare 10-shot 0.742 → 0.725: the in-run "beats B at 10-shot" was a split artefact.
- 308/384 locked-fold-0 images were in v4's label-free SSL set (v4 used a recomputed split). That can only help v4, so the ARMD gap is if anything understated.
- One fold, one SSL seed; ±0.03 fold noise still applies to unpaired comparisons.
- v4 still wins clearly on disc: ODC few-shot +0.03 to +0.04 and LP ODC/ODP/ODE up; LP mAUROC 0.921 ≥ B 0.911 (= v2 0.921).

## Files
`evallib.py`, `step1_existing.py` (+json/log), `extract_ep20.py`, `extract_B.py`, `step2_eval.py`, `step2_*.json`, `paired_armd.py`, `paired_vs_B.json`, `emb_*.npy`.

## Weight locations (in OFFIRST repo)
- v4 ep20 (`best.pth`): [encoder_byol_anatomy_v4_best.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-byol-v4/encoder_byol_anatomy_v4_best.pth) (release `encoder-byol-v4`; all epochs 10–120 in the same release, checksums in `SHA256SUMS.txt`)
- B (anatomy v1 ep150): [encoder_byol_anatomy_last.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-byol-v1/encoder_byol_anatomy_last.pth) (release `encoder-byol-v1`)
- Locked split: `split_5fold_seed42_locked.csv` (this folder) = `encoder_byol/full_run/split_used.csv`
