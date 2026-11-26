# OFFIRST v1.1/v2.1 prediction recovery

## Result

The frozen v1.1 and v2.1 ONNX assets and their inference contracts were recovered and verified. The reported matched assessment summaries for 69 RFMiD and 72 IDRiD images were recovered from the repository's research results. **Neither assessment's original image-level prediction table, exact sample join, or source photographs were recovered.** Therefore no per-image probability CSV, ROC/PR curve, score distribution, individual false-negative list, or matched re-inference is claimed.

No model training, weight changes, threshold changes, production edits, or test-set threshold tuning were performed. The investigation branch baseline is 753428dbdb2b6989bdd1aae52cc4c13e52270632.

## Search coverage and recovered material

| Location/evidence | Recovered | Limitation for the requested 69/72 assessments |
| --- | --- | --- |
| deployment/offline_dr_screening/assets/offirst_dr_model.onnx | v1.1 ONNX, SHA-256 b8fc10191f69faef16f1dc92d08bf9542ee7e26f885767942e28292c5201dd90 | Model only; no assessment image IDs or predictions are embedded as a reviewable table. |
| deployment/offline_dr_screening/assets/offirst_v2_1_experimental.onnx | v2.1 ONNX, SHA-256 43a87f05df7adc6a231a679721e7f0b11b45a3fe6d1582aa4527ed738fa63b3f | Model only; no assessment image IDs or predictions are embedded as a reviewable table. |
| assets/model_v2_1_evidence/ | v2.1 model/preprocessing/prediction metadata and aggregate 141-example export parity record | Saved Python/export probability comparison does not include a public per-image RFMiD/IDRiD prediction join or source image corpus. |
| docs/RESEARCH_RESULTS.md, V2_1_EXPERIMENTAL_INTEGRATION.md | v1.1/v2.1 aggregate metrics for the reported matched 69 RFMiD and 72 IDRiD cases; v2.1 aggregate confusion matrices | No row-level labels, probabilities, false-positive/false-negative IDs, or device/source metadata. |
| kd_run2/idrid_grading_eval/ | 516-row IDRiD grading metadata/embeddings for a different frozen-encoder probe, 413/103 split, and probe evaluation outputs | Not v1.1/v2.1 application classifier predictions; no source images are present; cannot be silently substituted for the app's 72-image subset. |
| C:\Users\doifo\Downloads\OFFIRST_Accuracy_Audit\ | Separate local audit bundle. Its deployment metadata/configuration and offirst_dr_model.onnx inventory match the v1.1 asset hash, RGB 512 preprocessing, positive class, and 0.48 cutoff. It includes pseudonymized RFMiD partitions (1,536 train / 192 validation / 192 test), but explicitly records no RFMiD patient keys and no RFMiD binary final-test prediction table. It also contains ODIR historical predictions for a different 8-class epoch-3 probe. | RFMiD manifest is not the reported 69-image assessment and has no image bytes or patient grouping. ODIR probe probabilities are not the deployed binary model's predictions. Neither source can recreate the target matched assessment. |
| C:\Users\doifo\Downloads\OFFIRST_Hybrid_KD_MaskedInfoNCE_DR.ipynb | Read-only notebook inspection: 9 code cells, none executed and none with saved outputs | It is a training/experiment proposal, not historical v1.1/v2.1 prediction evidence. It was not run. |
| Local Git object history and repository artifact search | Existing commits and paths for model metadata, aggregate summaries, encoder probes, and unrelated prediction artifacts | No matching committed 69/72 per-image v1.1/v2.1 prediction export, label join, image hash manifest, or source image files found. |
| Other accessible local folders checked | C:\Users\doifo\offirst_output contains a short unrelated RFMiD training log; Desktop ZEISS docs contain representation comparisons. | Neither contains the target app-classifier per-image results. No matching assessment photographs were found in the checked Downloads/docs folders. |

The separate accuracy-audit bundle's RFMiD 192-image validation partition is **not** established as a patient-independent threshold-development cohort: its patient keys are absent. The bundle also reports uncertain binary-head fit/selection lineage and no verified image-level overlap checks. Its existence does not qualify it for threshold selection.

## Frozen ONNX assets and inference contracts

The local assets were opened with ONNX Runtime 1.26 in the retained Python environment. Both exposed one float input named fundus_image, type tensor(float), shape [batch,3,512,512], and one float output named dr_probability, shape [batch,1]. This verifies session graph contract/loading, not image inference or Python/Android parity on assessment cases.

| Model | Hash | Positive class / threshold | Calibration and head evidence | Preprocessing |
| --- | --- | --- | --- | --- |
| v1.1 offirst_dr_model.onnx | b8fc10191f69faef16f1dc92d08bf9542ee7e26f885767942e28292c5201dd90 | DR_DETECTED at probability >= 0.48 | Metadata identifies MobileNetV3-Large, 960-dimensional encoder feature, and logistic-regression classifier. The graph returns the probability directly; the available config records no separate Platt step. No feature-order file or per-image activation/head dump was recovered; the bundled ONNX is the executable source of truth. | Pillow RGB decode without EXIF transpose; BILINEAR resize to 512×512; float32 /255; ImageNet mean/std; planar RGB NCHW. The app passes graph probability directly and applies the recorded 0.48 cutoff. |
| v2.1 offirst_v2_1_experimental.onnx | 43a87f05df7adc6a231a679721e7f0b11b45a3fe6d1582aa4527ed738fa63b3f | DR = IDRiD grades 1–4 / RFMiD DR=1 at probability >= 0.3675977844516245 | Platt slope 0.5428962637583218 and intercept -0.49103461599559345 are incorporated in the exported graph. Evidence records a KD768 encoder path (960→768 projection), L2 normalization, graph-resident train-fitted StandardScaler and 16-unit MLP head. Do not recalibrate the probability externally. | OpenCV IMREAD_COLOR/EXIF orientation; BGR→RGB; 512-long-side proxy; max-channel >15 mask, 5×5 opening, crop when ≥100 mask pixels, square black padding; 640 then 512 INTER_AREA; ImageNet normalization; RGB NCHW float32. |

The source app declares the two exact thresholds and uses >=; v1.1 and v2.1 preprocessing are distinct. The complete graph stores feature computation and ordering, but v1.1's separate feature-order/head-fit recipe and v2.1's raw assessment-time classifier outputs are unavailable. Do not assume the two probability scales are calibrated comparably across datasets.

## Aggregate matched assessment evidence

OFFIRST_MATCHED_MODEL_COMPARISON.csv records only published aggregate metrics. v2.1 confusion matrices are directly recorded in docs/RESEARCH_RESULTS.md. v1.1 confusion cells are reconstructed from the rounded reported rates and matched class denominators implied by the v2.1 matrix; the CSV marks these cells as derived, not recovered case-level labels.

| Dataset | Model | n | Accuracy | Sensitivity | Specificity | AUROC | AUPRC | TN, FP, FN, TP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RFMiD | v1.1 | 69 | 86.96% | 100.00% | 84.48% | 0.967 | 0.853 | 49, 9, 0, 11 (derived from reported rates) |
| RFMiD | v2.1 | 69 | 81.16% | 54.55% | 86.21% | 0.875 | 0.640 | 50, 8, 5, 6 (reported) |
| IDRiD | v1.1 | 72 | 70.83% | 57.45% | 96.00% | 0.846 | 0.932 | 24, 1, 20, 27 (derived from reported rates) |
| IDRiD | v2.1 | 72 | 79.17% | 87.23% | 64.00% | 0.874 | 0.944 | 16, 9, 6, 41 (reported) |

Both models were reported on the same assessment set within each dataset, but the exact sample-level match cannot be independently verified without IDs. At the reported aggregate operating points, RFMiD v1.1 numerically meets all three 85% accuracy / 85% sensitivity / 80% specificity targets (86.96% / 100.00% / 84.48%); it also reports AUROC 0.967. This aggregate result is not independently reproducible from the recovered files and cohort independence is unverified. v2.1 RFMiD misses accuracy/sensitivity; both IDRiD rows miss accuracy, and v2.1 IDRiD also misses specificity. No new ROC/PR curves or probability distributions can be computed from AUROC/AUPRC summaries alone.

## Per-image failure investigation

The aggregate v2.1 matrices establish five RFMiD false negatives and six IDRiD false negatives. The rounded v1.1 rates imply zero RFMiD and twenty IDRiD false negatives for the matched denominators; these counts are aggregate-derived. The assessment image IDs, probabilities, DR grades/severity, image quality labels, camera/site data, patient identifiers, source paths, and image/tensor hashes are unavailable for all these cases.

Accordingly, no individual false negative can be reviewed for quality, severity, dataset-specific features, preprocessing mismatch, probability margin, or disagreement with the other model. Dataset-level disagreement is observed in the aggregate sensitivity/AUROC results; case-by-case overlap and disagreement are unknown. OFFIRST_FALSE_NEGATIVE_ANALYSIS.csv contains aggregate counts with unavailable case fields left blank. It does not invent case rows.

The separate saved-feature IDRiD binary probe reports poor grade-1 detection (2/5 on its 103-image test split), but this is not the deployed v1.1/v2.1 classifier. It is only adjacent evidence that mild-grade misses are plausible, not a severity attribution for the app's 72-image assessment.

## Threshold feasibility and required inputs

No existing cohort is demonstrated to be both (a) patient-grouped and independent of v1.1/v2.1 model fitting, calibration, and selection and (b) accompanied by source images and row-level probabilities/labels. The RFMiD partitions found in the separate audit bundle lack patient IDs. The available grouped ODIR validation partition belongs to an eight-class feature-probe experiment with a different endpoint, and cannot be substituted. The IDRiD 413/103 grading split is documented for a different frozen-representation evaluation; images and verified patient grouping are not present here.

Therefore no threshold sweep was run or prepared against the historical assessment data. A qualifying threshold-development cohort requires: authorized original images; adjudicated DR endpoint and severity labels; stable pseudonymous patient IDs and site/device metadata; exact model-exposure manifests proving disjointness from encoder/head fitting and prior model selection; a patient-disjoint validation split; both frozen model outputs and preprocessing/tensor hashes on the same images; and a separately sealed untouched test partition. Select each model's threshold only on validation, lock and hash it, then evaluate once on the untouched test. Retain historical thresholds until that process is complete.

## Files produced

- OFFIRST_PREDICTION_RECOVERY.md — search record, recovered contracts, aggregate evidence and gaps.
- OFFIRST_MATCHED_MODEL_COMPARISON.csv — four aggregate comparison rows; no pseudonymous case rows are implied.
- OFFIRST_FALSE_NEGATIVE_ANALYSIS.csv — aggregate FN counts; individual case fields intentionally blank because the evidence was not recovered.
- OFFIRST_THRESHOLD_FEASIBILITY.md — cohort eligibility and exact threshold-sweep prerequisites.
