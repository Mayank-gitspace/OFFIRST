# ConvNeXt-T baseline (EMA, epoch 10) on IDRiD segmentation subset

Data: IDRiD seg = 81 images (54 train + 27 test), all DR-positive. Negatives = 81 RFMiD-train fold-0 VAL normals (all-zero labels; 12 unzipped from normals_dl). In-domain reference = 72 RFMiD fold-0 val DR-positive eyes. IDRiD grading folder and RFMiD eval/test were never read. CPU inference, same crop+512 px+ImageNet norm as training.
Note: 59 IDRiD seg images are pixel-identical to IDRiD grading images (data_splits/idrid_seg_duplicates_in_grading.tsv).

## 1) Probability distributions
| set | label | min | median | mean | max | % >0.5 |
|---|---|---|---|---|---|---|
| IDRiD seg (n=81) | DR | 0.267 | 0.996 | 0.925 | 1.000 | 95.1 |
| IDRiD seg | Disease_Risk | 0.470 | 0.999 | 0.983 | 1.000 | 98.8 |
| RFMiD val normals (n=81) | DR | 0.120 | 0.183 | 0.207 | 0.589 | 1.2 |
| RFMiD val normals | Disease_Risk | 0.187 | 0.356 | 0.416 | 0.994 | 27.2 |
| RFMiD val DR (n=72) | DR | 0.239 | 0.992 | 0.923 | 1.000 | 93.1 |

## 2) AUROC vs RFMiD val normals
- IDRiD seg DR: **0.996**; Disease_Risk: **0.991**
- In-domain reference (RFMiD val DR vs same normals): DR **0.993**, Disease_Risk 0.999
- Camera confound: cross-domain ≈ in-domain, so no sign the score is driven by camera. Control probe IDRiD-DR vs RFMiD-DR (both diseased) on embeddings: AUROC 0.55/0.69/0.71 at k=1/5/10, i.e. camera is only weakly encoded.
- Caveat: fold-0 val also chose the epoch, so the in-domain number is slightly optimistic.

## 3) Lesion burden (Spearman, lesion area % of image)
DR prob: MA 0.78, HE 0.61, EX 0.48, SE 0.30, total 0.67 (all p<0.01). Disease_Risk: MA 0.44, HE 0.44, EX 0.58, SE 0.17 (ns), total 0.60.
Lowest DR scores are all low-burden eyes (total ≤0.4%): IDRiD_80 0.27 (0.20%), IDRiD_29 0.35 (0.07%), IDRiD_57 0.42 (0.40%, mostly EX), IDRiD_41 0.45 (0.15%), IDRiD_44 0.56 (0.27%). Disease_Risk still ≥0.87 for the bottom 4.

## 4) Few-shot linear probe (L2-norm embeddings, 10 draws)
- IDRiD-seg DR vs RFMiD normals: k=1 0.997±0.001, k=5 0.998±0.001, k=10 0.998±0.000
- Heavy EX (>1% area, 20 vs 61) within IDRiD seg, k=5: 0.936±0.039

Files: report.json, idrid_seg_per_image.csv, idrid_seg_eval.png, preds_embs.npz, run_eval.py
