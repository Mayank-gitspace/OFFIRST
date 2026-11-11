# IDRiD Disease Grading: cross-domain frozen-encoder evaluation, KD Run 2 vs SSL ep40 (box CPU, 2026-11-11 02:08 IST)

Jeet asked for this on 2026-11-11 ("No test on idrid grading set"). It was evaluation only. No training or fine-tuning was run, and no Kaggle GPU was used.

## Data
- Source: Kaggle `aaryapatel98/indian-diabetic-retinopathy-image-dataset`, a mirror of the official IDRiD release, folder `B. Disease Grading` with its official CSVs. Box copy: `/workspace/idrid_grading/`.
- Counts are 413 train and 103 test, matching the official numbers. All 516 images are 4288×2848.
- DR grade (0–4): train {'0': 134, '1': 20, '2': 136, '3': 74, '4': 49}, test {'0': 34, '1': 5, '2': 32, '3': 19, '4': 13}.
- DME risk (0–2): train {'0': 177, '1': 41, '2': 195}, test {'0': 45, '1': 10, '2': 48}.
- Referable DR (grade ≥ 2): train 259/413, test 64/103.
- These label distributions match the published IDRiD grading split.
- None of these images was seen by any encoder. All the training kernels have a data guard that aborts if any IDRiD file is mounted.

## Protocol (as briefed)
- **Encoders:** all frozen.
  - KD768 = `kd_final.pth` (ep40, post BN-recal, with the 960→768 head).
  - SSL960 = `student_init_mnv3_ssl_ep40_target.pth` (GAP).
  - SSL768p = SSL960 projected with the kernel's fixed `procrustes_W.npy`.
  - Teacher = BYOL v4 ConvNeXt-T ep20 target at 384, reference only.
- **Preprocessing:** the KD Run 2 kernel's own code, sliced from the kernel source and exec'd (`crop_fundus`, `fov_geom`, the DEGRADE BLOCK, the config). The path is fundus crop/pad → 640 INTER_AREA cache → INTER_AREA to 512 (MNV3) or 384 (teacher) → ImageNet normalization.
- **Degraded copies:** `degrade_eval` with alpha=1.0 and eval_seed 2026. The per-image seed is the row index in this dataset (train rows 0–412, test rows 413–515).
- **Box vs Kaggle check:** on 16 RFMiD images, the box CPU fp32 path reproduces the kernel's saved Kaggle embeddings for all 4 models, clean and degraded. Min cosine is 0.99988 (`embeddings/extract_meta.json`).
- **Probe:** L2-normalize → StandardScaler (fit on train) → LogisticRegression(class_weight=balanced, multinomial).
  - C is chosen from the protocol grid [0.01, 0.1, 1] by 5-fold stratified CV on the 413 train images only. The CV metric is macro OvR AUROC (AUROC for RDR).
  - The probe is refit on all 413 and scored on the 103 test images.
  - QWK and accuracy are computed on the argmax.
- **kNN (RDR):** k=20, cosine, exp(sim/0.07)-weighted vote (the paired_bootstrap.py convention). The bank is clean train; queries are clean and degraded test.
- **Paired bootstrap:** 1000 resamples of the 103 test images (seed 12345), with one index vector shared by all models and metrics. If a class is absent from a resample, it is dropped from that resample's macro AUROC.

## Per-model results (point estimate [single-model 95% bootstrap CI])

**clean→clean**

| metric | KD768 | SSL960 | SSL768p | Teacher (ref) |
|---|---|---|---|---|
| DR5_macroAUROC | 0.814 [0.756, 0.867] | 0.824 [0.767, 0.878] | 0.817 [0.761, 0.872] | 0.784 [0.727, 0.843] |
| DR5_QWK | 0.616 [0.474, 0.739] | 0.651 [0.505, 0.781] | 0.607 [0.449, 0.748] | 0.562 [0.405, 0.686] |
| DME3_macroAUROC | 0.893 [0.830, 0.948] | 0.885 [0.825, 0.938] | 0.879 [0.816, 0.942] | 0.901 [0.844, 0.953] |
| DME3_acc | 0.796 [0.718, 0.874] | 0.796 [0.709, 0.874] | 0.786 [0.699, 0.864] | 0.748 [0.660, 0.825] |
| RDR_AUROC | 0.925 [0.874, 0.969] | 0.915 [0.863, 0.964] | 0.919 [0.868, 0.967] | 0.935 [0.888, 0.974] |
| kNN_RDR_AUROC | 0.891 [0.828, 0.948] | 0.887 [0.820, 0.947] | 0.887 [0.821, 0.947] | 0.897 [0.835, 0.955] |

**clean probe→degraded test**

| metric | KD768 | SSL960 | SSL768p | Teacher (ref) |
|---|---|---|---|---|
| DR5_macroAUROC | 0.811 [0.757, 0.863] | 0.798 [0.744, 0.857] | 0.820 [0.756, 0.877] | 0.751 [0.696, 0.814] |
| DR5_QWK | 0.576 [0.422, 0.698] | 0.647 [0.506, 0.760] | 0.596 [0.444, 0.728] | 0.511 [0.351, 0.645] |
| DME3_macroAUROC | 0.884 [0.824, 0.938] | 0.857 [0.786, 0.921] | 0.858 [0.797, 0.913] | 0.907 [0.858, 0.952] |
| DME3_acc | 0.738 [0.650, 0.816] | 0.738 [0.650, 0.816] | 0.699 [0.612, 0.777] | 0.767 [0.689, 0.845] |
| RDR_AUROC | 0.929 [0.878, 0.970] | 0.907 [0.850, 0.956] | 0.899 [0.840, 0.953] | 0.921 [0.871, 0.965] |
| kNN_RDR_AUROC | 0.892 [0.831, 0.948] | 0.871 [0.799, 0.939] | 0.877 [0.810, 0.940] | 0.885 [0.820, 0.945] |

**degraded train→degraded test**

| metric | KD768 | SSL960 | SSL768p | Teacher (ref) |
|---|---|---|---|---|
| DR5_macroAUROC | 0.825 [0.758, 0.888] | 0.842 [0.789, 0.891] | 0.840 [0.783, 0.893] | 0.766 [0.708, 0.830] |
| DR5_QWK | 0.595 [0.445, 0.720] | 0.627 [0.486, 0.754] | 0.601 [0.450, 0.739] | 0.605 [0.468, 0.718] |
| DME3_macroAUROC | 0.887 [0.824, 0.944] | 0.889 [0.832, 0.935] | 0.884 [0.829, 0.937] | 0.896 [0.837, 0.949] |
| DME3_acc | 0.767 [0.680, 0.845] | 0.786 [0.699, 0.864] | 0.777 [0.689, 0.854] | 0.777 [0.699, 0.854] |
| RDR_AUROC | 0.935 [0.889, 0.974] | 0.922 [0.871, 0.966] | 0.927 [0.878, 0.968] | 0.924 [0.876, 0.967] |

## Paired: KD768 (B) − SSL960 (A)
| condition | metric | A | B | Δ (B−A) | 95% CI | P(Δ>0) |
|---|---|---|---|---|---|---|
| clean | DR5_macroAUROC | 0.824 | 0.814 | -0.010 | [-0.036, +0.017] | 0.240 |
| clean | DR5_QWK | 0.651 | 0.616 | -0.035 | [-0.156, +0.084] | 0.300 |
| clean | DME3_macroAUROC | 0.885 | 0.893 | +0.009 | [-0.021, +0.035] | 0.748 |
| clean | DME3_acc | 0.796 | 0.796 | +0.000 | [-0.049, +0.049] | 0.421 |
| clean | RDR_AUROC | 0.915 | 0.925 | +0.009 | [-0.015, +0.032] | 0.793 |
| clean | kNN_RDR_AUROC | 0.887 | 0.891 | +0.004 | [-0.026, +0.031] | 0.612 |
| clean_to_deg | DR5_macroAUROC | 0.798 | 0.811 | +0.013 | [-0.021, +0.051] | 0.762 |
| clean_to_deg | DR5_QWK | 0.647 | 0.576 | -0.071 | [-0.192, +0.035] | 0.112 |
| clean_to_deg | DME3_macroAUROC | 0.857 | 0.884 | +0.027 | [-0.025, +0.072] | 0.887 |
| clean_to_deg | DME3_acc | 0.738 | 0.738 | +0.000 | [-0.068, +0.068] | 0.447 |
| clean_to_deg | RDR_AUROC | 0.907 | 0.929 | +0.022 | [+0.001, +0.049] | 0.983 |
| clean_to_deg | kNN_RDR_AUROC | 0.871 | 0.892 | +0.020 | [-0.025, +0.071] | 0.816 |
| deg_to_deg | DR5_macroAUROC | 0.842 | 0.825 | -0.017 | [-0.055, +0.024] | 0.204 |
| deg_to_deg | DR5_QWK | 0.627 | 0.595 | -0.032 | [-0.102, +0.040] | 0.182 |
| deg_to_deg | DME3_macroAUROC | 0.889 | 0.887 | -0.001 | [-0.044, +0.032] | 0.523 |
| deg_to_deg | DME3_acc | 0.786 | 0.767 | -0.019 | [-0.068, +0.029] | 0.193 |
| deg_to_deg | RDR_AUROC | 0.922 | 0.935 | +0.013 | [-0.008, +0.034] | 0.896 |

## Paired: KD768 (B) − SSL768p (A)
| condition | metric | A | B | Δ (B−A) | 95% CI | P(Δ>0) |
|---|---|---|---|---|---|---|
| clean | DR5_macroAUROC | 0.817 | 0.814 | -0.004 | [-0.023, +0.016] | 0.360 |
| clean | DR5_QWK | 0.607 | 0.616 | +0.009 | [-0.075, +0.102] | 0.560 |
| clean | DME3_macroAUROC | 0.879 | 0.893 | +0.015 | [-0.014, +0.043] | 0.839 |
| clean | DME3_acc | 0.786 | 0.796 | +0.010 | [-0.039, +0.058] | 0.589 |
| clean | RDR_AUROC | 0.919 | 0.925 | +0.006 | [-0.013, +0.026] | 0.729 |
| clean | kNN_RDR_AUROC | 0.887 | 0.891 | +0.003 | [-0.027, +0.032] | 0.609 |
| clean_to_deg | DR5_macroAUROC | 0.820 | 0.811 | -0.009 | [-0.040, +0.023] | 0.297 |
| clean_to_deg | DR5_QWK | 0.596 | 0.576 | -0.020 | [-0.121, +0.085] | 0.368 |
| clean_to_deg | DME3_macroAUROC | 0.858 | 0.884 | +0.026 | [-0.014, +0.067] | 0.899 |
| clean_to_deg | DME3_acc | 0.699 | 0.738 | +0.039 | [-0.029, +0.117] | 0.821 |
| clean_to_deg | RDR_AUROC | 0.899 | 0.929 | +0.030 | [+0.005, +0.058] | 0.995 |
| clean_to_deg | kNN_RDR_AUROC | 0.877 | 0.892 | +0.015 | [-0.021, +0.052] | 0.804 |
| deg_to_deg | DR5_macroAUROC | 0.840 | 0.825 | -0.014 | [-0.048, +0.019] | 0.210 |
| deg_to_deg | DR5_QWK | 0.601 | 0.595 | -0.005 | [-0.069, +0.069] | 0.433 |
| deg_to_deg | DME3_macroAUROC | 0.884 | 0.887 | +0.003 | [-0.024, +0.025] | 0.621 |
| deg_to_deg | DME3_acc | 0.777 | 0.767 | -0.010 | [-0.058, +0.039] | 0.292 |
| deg_to_deg | RDR_AUROC | 0.927 | 0.935 | +0.008 | [-0.011, +0.027] | 0.817 |

## Reference: Teacher (B) − KD768 (A)
| condition | metric | A | B | Δ (B−A) | 95% CI | P(Δ>0) |
|---|---|---|---|---|---|---|
| clean | DR5_macroAUROC | 0.814 | 0.784 | -0.029 | [-0.066, +0.005] | 0.048 |
| clean | DR5_QWK | 0.616 | 0.562 | -0.055 | [-0.124, -0.001] | 0.023 |
| clean | DME3_macroAUROC | 0.893 | 0.901 | +0.007 | [-0.016, +0.033] | 0.740 |
| clean | DME3_acc | 0.796 | 0.748 | -0.049 | [-0.126, +0.029] | 0.087 |
| clean | RDR_AUROC | 0.925 | 0.935 | +0.010 | [-0.007, +0.029] | 0.879 |
| clean | kNN_RDR_AUROC | 0.891 | 0.897 | +0.006 | [-0.017, +0.031] | 0.696 |
| clean_to_deg | DR5_macroAUROC | 0.811 | 0.751 | -0.060 | [-0.103, -0.017] | 0.000 |
| clean_to_deg | DR5_QWK | 0.576 | 0.511 | -0.065 | [-0.175, +0.052] | 0.116 |
| clean_to_deg | DME3_macroAUROC | 0.884 | 0.907 | +0.023 | [-0.004, +0.051] | 0.957 |
| clean_to_deg | DME3_acc | 0.738 | 0.767 | +0.029 | [-0.029, +0.087] | 0.808 |
| clean_to_deg | RDR_AUROC | 0.929 | 0.921 | -0.008 | [-0.028, +0.012] | 0.194 |
| clean_to_deg | kNN_RDR_AUROC | 0.892 | 0.885 | -0.007 | [-0.035, +0.022] | 0.334 |
| deg_to_deg | DR5_macroAUROC | 0.825 | 0.766 | -0.059 | [-0.106, -0.016] | 0.004 |
| deg_to_deg | DR5_QWK | 0.595 | 0.605 | +0.010 | [-0.082, +0.096] | 0.544 |
| deg_to_deg | DME3_macroAUROC | 0.887 | 0.896 | +0.008 | [-0.011, +0.030] | 0.825 |
| deg_to_deg | DME3_acc | 0.767 | 0.777 | +0.010 | [-0.039, +0.068] | 0.562 |
| deg_to_deg | RDR_AUROC | 0.935 | 0.924 | -0.011 | [-0.031, +0.004] | 0.090 |

## Reading (no overclaiming)
- **On clean IDRiD images, KD shows no detectable gain over SSL ep40.** Every clean-image Δ is small (−0.035 to +0.009), and every CI includes 0. DR grade AUROC and QWK lean negative (P(Δ>0) 0.24 and 0.30). DME and RDR lean slightly positive.
- **On degraded images, KD looks slightly better for referable DR only.** With the clean probe tested on degraded images, RDR AUROC goes 0.907 → 0.929, Δ +0.022, CI [+0.001, +0.049], P = 0.98. Against SSL768p it is +0.030, CI [+0.005, +0.058]. These are 1–2 metrics out of about 17 tested per comparison, with no multiplicity correction, and the lower bound sits near 0. Treat this as a hint, not a result.
- In the same condition, **DR5 QWK drops for KD: −0.071, CI [−0.192, +0.035], P(Δ>0) 0.11.** Degraded-train→degraded-test shows no KD advantage on DR grade or DME.
- **The large RFMiD rare-disease gains (ARMD@10 +0.16) do not show up as a gain on IDRiD DR/DME grading.** Note these are different tasks: RFMiD measured rare-label few-shot, while this measures DR/DME, which are common and lesion-driven. So this neither confirms nor refutes the RFMiD result. It says KD did not buy DR/DME transfer, and also did not clearly hurt it.
- **The teacher is not uniformly better than the student.** It is better on DME and clean RDR, but worse on DR5 AUROC (0.784 vs KD 0.814; clean-probe→degraded 0.751 vs 0.811, CI excluding 0).

## Caveats
- **Small n.** There are 103 test images, and DR grade 1 has only 5 test (20 train) images, so DR5 macro AUROC and QWK are noisy. The CIs are wide (QWK about ±0.1).
- **Label imbalance.** The probe uses class_weight=balanced. QWK and accuracy depend on the argmax of a balanced probe, and no threshold or ordinal tuning was done.
- **C at the grid edge.** C=0.01, the smallest value in the protocol grid, was selected for almost every model and task. Stronger regularization might score higher. I did not extend the grid because it was not in the protocol. This should be decided, not assumed.
- **"Cross-domain" is weaker than it sounds.** IDRiD is the same camera model family as part of RFMiD: 277 of RFMiD's 1,920 training images are 4288×2848 (the Kowa VX-10α subset per the RFMiD paper), and both datasets come from Nanded, India. This is a new dataset, a new label protocol and new graders, but only a partial camera/population shift.
- **Preprocessing deviation.** The box ran CPU fp32 instead of Kaggle CUDA fp16. The effect is negligible (cosine ≥ 0.9999 vs the kernel embeddings). IDRiD's FOV is clipped top and bottom, just like RFMiD's Kowa images. The crop code is unchanged.
- **Degraded protocol.** The degrader is the Visuscout fit (alpha=1). It emulates Visuscout's image statistics; it is not real Visuscout captures.
- **Not a decision test.** No pre-registered keep rule exists for IDRiD, so this report does not pass or fail KD.

## Files
- `extract.py` → `embeddings/*.npy` and `ids.csv` (rows: train 0–412 then test 413–515, plus `geom.npy`).
- `evaluate.py` → `results.json`.
- Logs: `extract.log`, `evaluate.log`.
- Example clean|degraded images: `example_*.jpg`.
