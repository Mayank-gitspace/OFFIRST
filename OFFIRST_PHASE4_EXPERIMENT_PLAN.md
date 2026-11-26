# OFFIRST Phase 4 experiment plan

## Recommendation

**Primary experiment: recover the exact KD Run 2 encoder and evaluate a frozen-encoder, head-only binary DR classifier on a newly established patient-grouped cohort.** Begin with cached 768-D KD features only after verifying their row-to-image mapping, checkpoint lineage, and preprocessing. Compare the result to the unchanged v2.1 ONNX classifier. Do not train an encoder, modify production, or change the current threshold as part of this plan.

This is the highest-value next step because KD768 is already the v2.1 representation, it has locally cached features, and a saved IDRiD probe indicates head-only analysis is computationally feasible. Existing evidence is insufficient to claim that the head is the limiting component or that the encoder is adequate. This experiment is designed to distinguish those possibilities.

## Phase 0 — recover and freeze evidence (before any fitting)

1. Obtain the exact `kd_run2_kd_final.pth` release asset and verify SHA-256 `8bc239d22ffc4e786820f2151c99380a754bef930e42edea65022e1246f2096d`. Keep it in a controlled research location; do not replace app assets.
2. Reconcile the encoder's 960→768 projection and normalization with `kd_run2/src/kd_tail.py`, v2.1 export metadata, and the cache extraction metadata. Re-extract a small allowed set and compare against cached vectors before using caches.
3. Recover licensed, authorized source images and ground-truth labels. Create stable pseudonymous IDs, SHA-256 image hashes, patient-group IDs, eye/visit identifiers, label provenance, severity, quality/gradable outcome, site, device, and acquisition metadata where available.
4. Build an exposure ledger per image: BYOL v4 pretraining, MNV3 BYOL pretraining, KD student training, head training, validation/calibration, historical v1.1/v2.1 assessments, and any prior use of the same image or patient.
5. Exclude overlapping or historically scored samples from the untouched test cohort. Do not use the current IDRiD 103-image test partition or historical 69-image RFMiD / 72-image IDRiD app assessments for model/head/threshold selection.

## Phase 1 — partition and endpoint lock

- Define the endpoint before fitting: binary any-DR or referable DR. Do not mix IDRiD grade `>=1` with grade `>=2` or RFMiD labels without an explicit mapping and harmonization rule.
- Partition by patient into development/train, validation, and untouched test. If site-disjoint generalization is an intended claim, hold out sites at the same time; record sites not represented rather than implying generality.
- Keep all eyes and repeated visits from a patient in the same partition. Confirm no duplicate content hashes cross partitions.
- Define the image-quality/fundus validation outcome separately from DR. Non-fundus and ungradable images must be routed to reject/retake and must not be counted as a DR-negative prediction.
- Freeze the manifest, class definitions, candidate heads, selection metric, threshold procedure, and minimum cohort rules before analysis begins.

## Phase 2 — prespecified head-only comparison

1. Reproduce frozen v2.1 Python probabilities on eligible source images, using the exact production preprocessing and frozen graph. Check cached-vs-re-extracted KD768 vectors and row mapping.
2. Fit a single regularized logistic regression head on frozen, L2-normalized KD768 features. Fit any scaler only within the training fold. Use patient-grouped nested cross-validation or a fixed development/validation split for regularization selection.
3. Compare with the existing v2.1 head and its locked cutoff as a fixed baseline. Any alternative cutoff can be selected only on the designated validation partition, with the intended operating criteria prespecified; never optimize it against the final test set.
4. Include severity-stratified error review, especially mild/grade-1 cases, and review every false negative and false positive in the development cohort. Record source domain, acquisition quality, device/site, and whether v1.1 disagrees where available.
5. If probability calibration is needed, fit it only on designated calibration/validation data and include it in the candidate definition. Do not apply calibration twice. Do not alter v1.1 or v2.1 calibration.

## Phase 3 — untouched evaluation

After selecting a candidate and freezing its model plus threshold, evaluate once on the untouched test partition. Report confusion matrix, accuracy, sensitivity, specificity, AUROC, AUPRC, calibration, patient-clustered confidence intervals, severity, site/device subgroups, missing/ungradable outcomes, and every failure case. The existing acceptance evaluator's patient/site safeguards and explicit minimum cohort gates should be used. Insufficient valid cohort size is inconclusive, not a pass.

Separately verify Python/Android probability parity using identical tensors, deterministic repeated inference, both OD and OS flows, offline operation, and input validation. This is application integration verification, not a substitute for the patient-level model evaluation.

## Decision rules

- If head-only KD768 improves operating behavior on validation and then passes the frozen external test with uncertainty bounds and cohort gates, prepare a separate change proposal for deployment review; do not replace the current model as part of Phase 4A.
- If it does not improve, determine whether the failure is attributable to label mapping, preprocessing, gradability/input validation, calibration, or representation using case-level evidence. Do not infer encoder inadequacy from a head-only miss without those checks.
- Only propose targeted encoder fine-tuning or a new representation after the frozen KD768 head comparison and case-level error review show that representation is the limiting factor. BYOL+Masked InfoNCE is not a priority until it has a completed, reproducible checkpoint and evidence of an advantage on the same partitions.
- Retain v1.1 and v2.1 ONNX assets, current thresholds, and checkpoints unchanged throughout this study.

## Inputs still required

Execution is not ready to start until the exact KD checkpoint is recovered; source images and label joins are available; patient-group and preferably site/device metadata are supplied; a fresh validation and untouched test cohort is defined; historical exposure overlap is audited; and the cached feature-to-image mapping is proven. Current repository evidence cannot establish those requirements. No model training was run for this plan.
