# BYOL A/B results (full runs, 150 epochs, convnext_tiny)

Evaluation on RFMiD-train fold 0 (held out from SSL). BYOL numbers use the target (EMA) encoder at epoch 150, frozen. Source: `full_run/<mode>/summary.json`.

## Linear probe, mean AUROC (12 common labels)

| Encoder | Mean AUROC | Disease_Risk AUROC |
|---|---|---|
| A: BYOL random crops | 0.898 | 0.927 |
| B: BYOL anatomy crops (disc/macula) | **0.911** | 0.930 |
| ImageNet init (no SSL) | 0.900 | 0.932 |
| Supervised baseline* | 0.962 | 0.967 |

## Rare-disease few-shot (mean AUROC over ODC, ODP, ARMD, MH; logistic-regression head)

| Shots | A random | B anatomy | ImageNet | Supervised* |
|---|---|---|---|---|
| 1 | 0.631 | 0.633 | 0.593 | 0.730 |
| 5 | 0.731 | 0.737 | 0.656 | 0.856 |
| 10 | 0.766 | 0.772 | 0.690 | 0.879 |

Per label at 10-shot (A / B / ImageNet / supervised*): ODC 0.62 / 0.63 / 0.59 / 0.88, ODP 0.59 / 0.60 / 0.52 / 0.71, ARMD 0.97 / 0.97 / 0.73 / 0.98, MH 0.87 / 0.89 / 0.93 / 0.94. Over all 10 rare labels at 10-shot: A 0.817, B 0.824, ImageNet 0.754, supervised* 0.928.

## Training health
- **No collapse** in either run: backbone embedding std (`std_h`) never dropped below 0.38 (A) / 0.41 (B) and rose to ~0.85 by epoch 150; projector std stayed ~0.96. See `full_run/training_curves.png`.
- kNN mean AUROC peaked early (A 0.888 at ep 40, B 0.885 at ep 60) and drifted down to ~0.86 by ep 150, so 150 epochs was more than needed.
- Wall time ~4.3 h per run on a Kaggle T4.

## Takeaways
- B (anatomy crops) beats A on the linear probe (+1.3 pp) and slightly on rare few-shot (+0.2 to +0.6 pp).
- Both BYOL encoders clearly beat ImageNet init on rare few-shot (+8 pp at 10-shot for B), but not on the linear probe.
- Frozen BYOL does not reach the supervised baseline on any metric here. ODP is the weakest label for BYOL.

\* **Caveat:** the supervised baseline was trained on all 28 labels of folds 1-4, including these rare labels, so its "few-shot" score is not a clean held-out comparison. A fair test needs some rare diseases hidden from supervised training entirely.
