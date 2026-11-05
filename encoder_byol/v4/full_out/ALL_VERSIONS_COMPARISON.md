# All BYOL versions vs supervised (2026-11-05 20:05 IST)
Sources:
- Supervised and ImageNet (A/B run): offirst_upload/encoder_byol/full_run/anatomy/summary.json
- A, B, v2: v2/free_checks/check1_report.json (target-EMA, plain)
- v4: v4/full_out/out/summary.json
- Rare = mean of the ODC/ODP/ARMD per-label logreg scores (no MH)

| run | LP mAUROC | DiseaseRisk | rare 1/5/10 |
|---|---|---|---|
| Supervised ConvNeXt-T | 0.962 | 0.967 | 0.715/0.830/0.859 |
| ImageNet (A/B split) | 0.900 | 0.932 | 0.542/0.572/0.611 |
| ImageNet (v4 split) | 0.887 | 0.931 | 0.556/0.570/0.621 |
| A random | 0.898 | 0.927 | 0.617/0.701/0.730 |
| B anatomy | 0.911 | 0.930 | 0.622/0.705/0.734 |
| v2 last | 0.921 | 0.940 | 0.598/0.679/0.707 |
| v4 ep20 | 0.928 | 0.950 | 0.608/0.686/0.742 |
| v4 ep70 | 0.916 | 0.922 | 0.558/0.652/0.699 |
| v4 ep120 tgt | 0.913 | 0.918 | 0.553/0.640/0.685 |

Note: v4 used a recomputed iterstrat split rather than the baseline's split_5fold_seed42.csv. Its fold 0 differs slightly (MH 64→63, MYA 21→20, ODE 12→11 positives), and LS dropped out of the common set. Means on the 10 shared diseases: ImageNet 0.898 (A/B split) vs 0.887 (v4 split).
