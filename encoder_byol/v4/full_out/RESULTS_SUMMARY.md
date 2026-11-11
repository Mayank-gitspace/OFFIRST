# BYOL anatomy v4: full run results

Collected 2026-11-05 19:50 IST. Kernel `teamoffirst/team-offirst-encoder-byol-anatomy-v4` v1 has status **COMPLETE**.
Outputs: `<workdir>/encoder_byol/v4/full_out/` (`out/summary.json`, `out/run.log`, `out/byol_anatomy_v4/{report,history,epoch_evals}.json`, ckpts ep010–ep110 + last + best(ep20), embeddings, kernel log).

Run: 120 epochs, bs **48 + grad-ckpt** (bs64 and bs48 without grad-ckpt hit OOM in the batch probe), 17.4 img/s, ~88 s/epoch, total wall 3.24 h (11,652 s).
The view mix that actually ran was random 0.50, disc 0.20, macula 0.20, fallback-random 0.10, so anatomy views were 40%.
The supervised reference was skipped because its embeddings weren't mounted.

## ep70 vs ep120 (same trajectory)
Rare = ODC/ODP/ARMD headline (no MH), logreg, 20 draws. LP = linear probe mean AUROC on the common labels.

| readout | LP mAUROC | DiseaseRisk | rare 1-shot | 5-shot | 10-shot |
|---|---|---|---|---|---|
| ep70 online | **0.916** | 0.922 | 0.558 | 0.652 | 0.699 |
| ep70 multi-region online | 0.918 | 0.922 | 0.561 | 0.660 | 0.705 |
| ep120 online | 0.908 | 0.911 | 0.547 | 0.630 | 0.677 |
| ep120 target-EMA | 0.913 | 0.918 | 0.553 | 0.640 | 0.685 |
| ep120 multi-region online | 0.904 | 0.909 | 0.552 | 0.643 | 0.693 |
| ep20 "best kNN" online | **0.928** | **0.950** | **0.608** | **0.686** | **0.742** |
| ImageNet init (same run) | 0.887 | 0.931 | 0.556 | 0.570 | 0.621 |

ep70 target-EMA wasn't evaluated; the mid-eval covered online and multi-region only.
Per-label 10-shot for ep120 target: ODC 0.651, ODP 0.582, ARMD 0.811. For ep20 online: ODC 0.718, ODP 0.597, ARMD 0.902.

## vs A / B / v2 (check1 protocol, target-EMA logreg)
| run | LP mAUROC | rare 1 / 5 / 10 |
|---|---|---|
| A random | 0.898 | 0.617 / 0.701 / 0.730 |
| B anatomy v1 | 0.911 | 0.622 / 0.705 / 0.734 |
| v2 last | 0.921 | 0.598 / 0.679 / 0.707 |
| **v4 ep120 target** | 0.913 | 0.553 / 0.640 / 0.685 |
| v4 ep70 online | 0.916 | 0.558 / 0.652 / 0.699 |
| v4 ep20 online (best kNN ckpt) | 0.928 | 0.608 / 0.686 / 0.742 |
| ImageNet | 0.900 (check1 table) / 0.887 (v4 run) | v4 run: 0.556 / 0.570 / 0.621 |

Caveat: the in-run ImageNet LP is 0.887, against 0.900 in the v2 table, so the v4 kernel's numbers aren't exactly the check1 protocol. The fold-0 rare query counts do match (ODC 56, ODP 13, ARMD 20). For a clean comparison, re-run check1 on the v4 last, ep70 and ep20 ckpts.

## Pass/fail vs the locked bars (CONFIG_LOCKED success criteria)
| bar | v4 ep120 target | v4 ep70 online | v4 ep20 online |
|---|---|---|---|
| LP ≥ B 0.911 | PASS (0.913, borderline) | PASS (0.916) | PASS (0.928) |
| LP near or above v2 0.921 | FAIL (−0.008) | FAIL (−0.005) | PASS (+0.007) |
| Rare ≥ B 0.622/0.705/0.734 | **FAIL** (−0.07 / −0.07 / −0.05) | **FAIL** (−0.06 / −0.05 / −0.04) | FAIL at 1/5 (−0.014 / −0.019), **PASS at 10 (0.742)** |
| No collapse | PASS: std_z 0.987 and std_h 0.984 at ep120 (min after ep60 was 0.968 / 0.977) | | |
| ep70 vs ep120 reported | PASS: ep70 > ep120 on every metric | | |

## Training monitors
- Loss went from 1.96 to 0.431. The BYOL term went from 1.31 to 0.019, which is very low late in training: the model is close to solving the invariance task.
- VICReg (weight 0.1): the variance term went from 0.60 to 0.218 and flattened by about ep40. The covariance term went from 0.52 to ~1.94 and stayed flat after ep10.
- kNN mAUROC peaked at **0.911 at ep20**, then sat between 0.886 and 0.905. It was 0.890 at ep70 and 0.892 at ep120.
- LR backbone reached ~0 around ep90, and tau went to 1.0 by ep115.

## Read
Training longer hurt. ep20 > ep70 > ep120 on LP, DiseaseRisk and the rare few-shot. 120 epochs over-fit the SSL objective on N=1536.
v4 does not beat B on the rare headline at ep70 or ep120. Only the early (ep20) checkpoint gets close to B, and it beats B at 10-shot.
v4 does beat ImageNet on rare 5/10-shot by about 0.06–0.08. Multi-region readout adds ~+0.005–0.01 to the rare scores and nothing to LP.
Suggested next step (not run): a check1-protocol re-eval of ep020/ep030/ep040/ep070/last from the saved ckpts. Also consider early-stop or a shorter schedule (~20–40 ep) for v4.1.

## Per-disease (added 2026-11-05 19:50 IST)
Linear-probe AUROC per label. B and v2 are check1 target-EMA (`v2/free_checks/check1_report.json`).

| disease | ep20 online | ep70 online | ep120 target | B | v2 |
|---|---|---|---|---|---|
| Disease_Risk | 0.950 | 0.922 | 0.918 | 0.930 | 0.940 |
| DR | 0.972 | 0.970 | 0.969 | 0.958 | 0.963 |
| ARMD | 0.948 | 0.953 | 0.945 | 0.992 | 0.996 |
| MH (media haze) | 0.978 | 0.964 | 0.962 | 0.975 | 0.977 |
| DN | 0.871 | 0.811 | 0.810 | 0.875 | 0.893 |
| MYA | 0.982 | 0.988 | 0.987 | 0.988 | 0.987 |
| BRVO | 0.966 | 0.959 | 0.962 | 0.981 | 0.974 |
| TSLN | 0.959 | 0.932 | 0.937 | 0.945 | 0.954 |
| ODC | 0.894 | 0.886 | 0.878 | 0.862 | 0.846 |
| ODP | 0.731 | 0.735 | 0.706 | 0.638 | 0.697 |
| ODE | 0.978 | 0.965 | 0.972 | 0.933 | 0.915 |

Rare few-shot logreg per label, 1 / 5 / 10-shot:

| disease | ep20 | ep70 | ep120 target | B |
|---|---|---|---|---|
| ODC | .577/.621/.721 | .526/.598/.667 | .526/.593/.652 | .532/.595/.630 |
| ODP | .538/.584/.598 | .524/.580/.598 | .518/.563/.585 | .533/.598/.603 |
| ARMD | .709/.853/.906 | .625/.777/.833 | .615/.763/.820 | .799/.922/.969 |
| ODE | .773/.925/.954 | .743/.883/.934 | .731/.876/.933 | .658/.779/.863 |
| MH | .643/.769/.861 | .582/.667/.757 | .572/.650/.732 | .668/.832/.887 |
