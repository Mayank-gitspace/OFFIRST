# OFFIRST encoder and DR head comparison

**Decision context:** choose the lowest-cost evidence-generating path without changing current model assets or treating representation probes as clinical screening results. This comparison is based on artifacts present/referenced at commit `38a0ba31996eef453ead711b845dc5aa593905c5`.

## Candidate comparison

| Path | Encoder and head status | Existing reusable data | Training/recovery cost | Main evidence and risks | Assessment |
|---|---|---|---|---|---|
| **A. v1.1 encoder + DR head** | Complete v1.1 MobileNetV3-Large + logistic head exists in ONNX. Standalone encoder checkpoint and original head-fit files were not recovered. | No identified v1.1 feature matrix joined to labels. | Lowest for inference baseline; high/uncertain for fitting a replacement head because encoder features must be extracted from source images or the graph must be split. | Aggregate app assessment was stronger on the reported small RFMiD set and weaker on IDRiD; no row-level predictions/images or independent patient-group evidence. Different preprocessing from v2.1. | Keep as unchanged frozen comparator/rollback; not the easiest trainable head-only candidate. |
| **B. v2.1 KD768 encoder + existing DR head** | Complete ONNX exists; includes KD768 encoder/projection, scaler, 16-unit MLP, and Platt calibration. Separate KD student checkpoint referenced by hash but absent locally. | KD768 arrays and IDRiD metadata locally present; RFMiD embeddings have image ID/fold only. | Lowest to reproduce current app outputs from ONNX. No new training required. Exact checkpoint recovery is needed to inspect/replace encoder or independently extract features. | Export/Python parity was excellent on 141 examples. Reported app metrics vary substantially by dataset; RFMiD sensitivity 54.55%, IDRiD sensitivity 87.23% and specificity 64%. These are development-only aggregates. | Mandatory frozen baseline. The current head should be diagnosed, not silently replaced; no test-set tuning. |
| **C. BYOL + Masked InfoNCE encoder + new DR head** | No completed Masked InfoNCE encoder/checkpoint or trained DR head was found. The local notebook is an unexecuted experiment outline. | Existing BYOL/KD embeddings are from other encoders and cannot substitute for the missing model's outputs. | Highest: checkpoint/input recovery or new encoder training, then extraction, head fit, mobile export, and validation. | No measured evidence that InfoNCE would solve DR false negatives. Adds architecture/training uncertainty and mobile validation work before simpler head experiments. | Do not prioritize. Reconsider only after a controlled representation comparison shows a head-only ceiling. |
| **D. Existing KD768 encoder + newly fit lightweight DR head** | KD Run 2 checkpoint and v2.1 lineage are defined; checkpoint is remotely referenced and currently absent locally. No separately exported research head suitable for production was found. | 1,920×768 RFMiD KD features with image ID/fold; 516×768 IDRiD features with grade and train/test split; 6,565×768 ODIR features with noisy labels. | Lowest candidate for a future head-only research experiment if labels/IDs/partition and feature provenance are restored; can fit from cached vectors. Checkpoint recovery enables independent encoder verification. | IDRiD cached-feature probe reports AUROC 0.885, sensitivity 85.51%, specificity 73.53% at 0.50; grade 1 detection 2/5. This is reused, non-patient-grouped test evidence, not v2.1 and not acceptance. KD representation tests also disclose RFMiD label-free training exposure on some reported query rows. | **Primary next experiment**, conditional on data/provenance recovery. Compare one prespecified lightweight head family with the unchanged v2.1 head on a clean development cohort, then lock and evaluate once on untouched test. No training is initiated in this phase. |

## Why D is the best next experiment

The local feature cache makes D the most economical way to separate representation from classifier-head limitations. KD768 already powers v2.1, and a saved-feature binary probe shows that a conventional head can rank IDRiD DR cases reasonably while missing the specificity gate and weakly detecting the small grade-1 subset. That observation does not prove the representation is adequate; it does justify testing the head before undertaking more encoder training.

The experiment is **blocked from valid execution** by missing source images and verified patient grouping, no RFMiD label join in the cache, and prior reuse of the only local IDRiD test partition. Recover the exact KD checkpoint and a fresh eligible cohort first. If representation adequacy fails on the clean independent development cohort, then investigate preprocessing and encoder limitations with targeted frozen-encoder comparisons before considering any fine-tuning.

## Evidence versus inference

**Observed:**

- v2.1 uses the KD Run 2 768-D student representation and has a separate calibrated 16-unit MLP head in ONNX.
- Cached KD768 embeddings exist for RFMiD, IDRiD, and ODIR, with different metadata completeness and label suitability.
- The saved IDRiD logistic probe fitted on cached features reports 81.55% accuracy, 85.51% sensitivity, 73.53% specificity, 0.8853 AUROC on 103 rows; its report discloses prior scoring and lack of patient IDs.
- The app's reported development assessments show v2.1 sensitivity 54.55% on RFMiD and 87.23% on IDRiD; exact per-case probabilities are not recovered.
- No Masked InfoNCE output checkpoint or executed notebook was found.

**Inference to test, not a conclusion:** the downstream v2.1 head/calibration/label mapping may be a lower-cost place to investigate before changing the KD encoder. Metrics do not isolate this cause, because the app assessments lack case-level evidence and differ in dataset/domain and endpoint details.

## Comparison controls for the future study

- Freeze the KD encoder SHA, preprocessing, label definition, and all data partitions before fitting any head.
- Use v2.1 ONNX as the unchanged baseline and reproduce its output on the exact eligible images.
- Fit head parameters and select operating threshold only on patient-grouped development/validation data; no historical 69 RFMiD or 72 IDRiD assessment reuse for tuning.
- Preserve a separate untouched patient- and preferably site-disjoint test cohort. Report clustered confidence intervals and site/device/severity outcomes.
- Keep a fixed validation/test partition for every candidate so differences are paired by patient.
- Treat a software parity result as integration evidence only; it does not substitute for clinical performance or gradability validation.
