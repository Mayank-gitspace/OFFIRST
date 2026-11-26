# OFFIRST v2.1 reproduction status

## Status: aggregate evidence recovered; full historical reproduction unavailable

The frozen model contract and aggregate assessment summaries can be described, but the original v2.1 per-image assessment cannot be reproduced from the currently recovered data. This audit did not reconstruct or fabricate historical predictions, scores, image identifiers/hashes, or corresponding label rows.

## Recoverable v2.1 model and pipeline

| Item | Evidence |
|---|---|
| ONNX asset | deployment/offline_dr_screening/assets/offirst_v2_1_experimental.onnx, 14,888,805 bytes, SHA-256 43a87f05df7adc6a231a679721e7f0b11b45a3fe6d1582aa4527ed738fa63b3f. |
| Encoder checkpoint reference | kd_run2_kd_final.pth, GitHub release kd-run2, hash 8bc239d22ffc4e786820f2151c99380a754bef930e42edea65022e1246f2096d; matching digest is in assets/model_v2_1_evidence/model_manifest.json. Binary not present locally and not loaded during this audit. |
| Input/output | FP32 RGB NCHW [1,3,512,512] to [1,1] probability. |
| Encoder | MobileNetV3-Large pooled 960-D feature, learned 960→768 projection, L2 normalization. |
| Head | Frozen ONNX includes scaler and a 16-unit MLP. Prediction config records regularization 0.1 and no class weighting. Separate fit checkpoint/serialized head is absent; ONNX is the executable head. |
| Calibration and decision | Platt slope 0.5428962637583218, intercept -0.49103461599559345 integrated in graph. Probability is used directly; decision threshold remains 0.3675977844516245, positive at >=. |
| Training preprocessing / embedding | FOV-aware fundus-square crop, 640 cache, 512 student input, ImageNet normalization and training-time augmentation/degradation are described in kd_run2/src/kd_tail.py and run config. |
| App preprocessing | OpenCV color decode, RGB conversion, crop/pad, 640 then 512 INTER_AREA, ImageNet normalization, NCHW FP32 in assets/model_v2_1_evidence/preprocessing.json. |
| Export parity | model_manifest.json: Python/export parity over 141 examples; max absolute error 1.8341754449702208e-06, mean 2.470365740902819e-07, zero cutoff disagreements. This is export parity, not historical dataset replay or Android/device evidence. |

## Aggregate historical assessment evidence

| Dataset | n | Accuracy | Sensitivity | Specificity | AUROC | v2.1 confusion matrix TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---|
| RFMiD assessment | 69 | 81.16% | 54.55% | 86.21% | 0.875 | 50 / 8 / 5 / 6 |
| IDRiD assessment | 72 | 79.17% | 87.23% | 64.00% | 0.874 | 16 / 9 / 6 / 41 |

These figures are preserved summaries from docs/RESEARCH_RESULTS.md, OFFIRST_MATCHED_MODEL_COMPARISON.csv, and the previous recovery report. They are development evidence. The RFMiD/IDRiD image and label IDs, exact split manifests, source/tensor hashes, probabilities, and patient joins were not recovered. The earlier v1.1 comparison is aggregate; its reconstructed confusion cells are explicitly marked derived in the prior report and are not treated as case-level observations here.

The v2.1 prediction configuration includes a separate development head-selection record (16-unit MLP, Platt parameters, threshold, and aggregate development metrics), but not exact image/patient membership, raw features, labels, or per-case output needed to replay head selection.

## Why reproduction is blocked

1. The exact KD final checkpoint is referenced by hash and available as a GitHub release asset, but is not locally recovered for checkpoint loading or feature re-extraction.
2. The full-run RFMiD embedding sidecar has only image ID and fold; it lacks DR labels, patient IDs, source hashes, and site/device metadata.
3. IDRiD cached features and labels are for a separate 516-image grading analysis, not an identifiable join to the 72-row app assessment. The 103-image IDRiD cache test was already used in other probes.
4. ODIR cache and historical probe predictions are another endpoint, not the app's v2.1 binary evaluation.
5. No per-image v2.1 predictions, labels, assessment identifiers, or image hashes were recovered from repository history, the local downloads audit bundle, or the inspected app evidence folder.

## Reproduction needed

To reproduce the historical assessment, recover the original case-level RFMiD and IDRiD manifests with exact image IDs, labels, split and exposure history; source images or authenticated immutable versions; assessment-time preprocessing/tensors or enough inputs to regenerate them; and saved v2.1 probabilities. Download and hash-check the exact KD checkpoint for independent feature extraction. If original probability files cannot be recovered, rerun the unchanged frozen ONNX on exact authorized images and labels and label that result a new re-evaluation, not recovered historical predictions. Keep the threshold unchanged for that replay.
