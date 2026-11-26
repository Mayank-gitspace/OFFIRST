# Hybrid DR Improvement Experiments — Status Report

**Date:** 2026-11-25  
**Decision:** **NO-GO for training claims, candidate selection, export, or deployment.**  
**Scope:** Isolated research work under `kd_run2/hybrid_dr_improvement/`. The published v1.1 app, model, cutoff, APKs, README, and releases were not changed.

## Executive summary

The requested three-way comparison could not be run reproducibly in this checkout. The KD Run 2 checkpoint and original teacher/student weights are referenced by hash in the research records and published release metadata, but their binary files are not present locally. The IDRiD and RFMiD source images are also absent, Kaggle credentials are unavailable, and the available Python installation has no PyTorch. Therefore experiments A, B, and C have **no measured results**; this is a blocked execution, not evidence that any method passed or failed.

As a limited baseline check, I reproduced a balanced logistic probe using the saved KD 768-dimensional IDRiD features and the recorded official 413/103 split. This is not end-to-end supervised fine-tuning and must not be compared as though it were experiment A. The 103-image test partition had already been scored in prior work, patient identifiers are unavailable, and these results are development evidence only.

## Experiment status

| Run | Intended method | Status | Metrics |
|---|---|---|---|
| A | KD epoch 40 + supervised any-DR fine-tuning | Not run | Not measured |
| B | A + explicit Masked InfoNCE + supervised fine-tuning | Not run | Not measured |
| C | B + consistency to frozen BYOL ConvNeXt teacher | Not run | Not measured |

No new checkpoint or ONNX export was produced. Consequently, no conclusion can be drawn about gains from supervised fine-tuning, Masked InfoNCE, teacher consistency, or their interaction. The prescribed 85% accuracy, 85% sensitivity, and 80% specificity gates cannot be assessed for A/B/C.

## Reproduced saved-feature baseline

The baseline script is [`baseline_binary_probe.py`](baseline_binary_probe.py); machine-readable output is [`baseline_binary_probe.json`](baseline_binary_probe.json). It reads cached embeddings and labels only, not retinal images or model weights.

Protocol: L2-normalized cached KD 768-D features; any DR is grade 0 versus grades 1–4; 413 official training rows and 103 official test rows; 5-fold stratified cross-validation on training rows selects logistic-regression `C` by ROC-AUC; `class_weight="balanced"`; fixed probability threshold 0.50. The threshold was not selected on the test partition.

| Metric (103 saved test rows) | Result |
|---|---:|
| Accuracy | 81.55% |
| Sensitivity | 85.51% |
| Specificity | 73.53% |
| Balanced accuracy | 79.52% |
| F1 | 86.13% |
| AUROC | 0.8853 |
| AUPRC | 0.9516 |
| Confusion matrix `[TN, FP, FN, TP]` | `[25, 9, 10, 59]` |

This saved-feature baseline clears only the sensitivity gate. It misses the accuracy and specificity gates. The result is not independent validation: the same 103 IDRiD test images appeared in the previous frozen-encoder grading evaluation, and no patient IDs are available for grouped splitting. Grade-wise detection at the fixed threshold was: grade 1, 2/5 (40.0%); grade 2, 28/32 (87.5%); grade 3, 17/19 (89.5%); grade 4, 12/13 (92.3%). Grade 0 had 9/34 false positives. The very small grade 1 count makes that estimate unstable.

## Integrity and data constraints

- The checked-in KD Run 2 manifest lists SHA-256 values for the original student initialization and BYOL ConvNeXt teacher. The KD final checkpoint digest is recorded in `kd_run2/README.md` and the release metadata. The checkpoint payloads were not locally available to load or re-hash in this run; release metadata matching a recorded digest is not a substitute for loading the checkpoint.
- Cached feature and ID file hashes for this baseline are included in the JSON output.
- The repository's prior evidence records overlap between SSL/KD training exposure and RFMiD locked-fold samples, and notes unavailable patient identifiers. Those RFMiD folds cannot support an independent performance claim. The previously scored DeepDRiD cohort was not used for fitting, calibration, or selection here.
- The original dataset pages identify IDRiD and RFMiD as CC BY 4.0 resources: [IDRiD data and license](https://idrid.grand-challenge.org/Data/) and [RFMiD dataset paper](https://www.mdpi.com/2306-7269/6/2/14). Any new use must still follow the original source terms and dataset access conditions.

## Reproducibility and next execution prerequisites

To run the requested experiments, restore the exact hash-verified KD final, student initialization, and teacher checkpoints; provision the authorized IDRiD/RFMiD source images and labels; install the documented PyTorch/CUDA or CPU environment; and record the concrete image manifests and splits. Use exact/near-duplicate screening, patient grouping only where real IDs exist, and keep all prior DeepDRiD evaluation images out of fit, calibration, and selection. Until those inputs and a reproducible run are available, there is no candidate to export or integrate.

No application model, cutoff, release, APK, root README, or research source history was modified.
