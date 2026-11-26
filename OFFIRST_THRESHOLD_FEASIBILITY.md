# OFFIRST threshold feasibility

## Conclusion

Threshold optimization is **not currently justified for either frozen application model**. No recovered cohort is proven patient-grouped, independent of that model's head fitting/calibration/selection, and available with authorized source images. No threshold sweep was run. The reported RFMiD v1.1 aggregate point metrics at its historical 0.48 cutoff numerically meet the 85% accuracy / 85% sensitivity / 80% specificity targets, but the source cases and prediction table are unavailable and independence is unverified. The historical v1.1 0.48 and v2.1 0.3675977844516245 cutoffs remain unchanged.

## Candidate cohorts located, and why they do not qualify

| Candidate | Evidence | Blocking issue |
| --- | --- | --- |
| RFMiD assessment reported in docs/RESEARCH_RESULTS.md (69 images) | Matched aggregate v1.1/v2.1 metrics and v2.1 confusion matrix; v1.1 aggregate point metrics meet all three requested numeric targets at 0.48 | This is the historical assessment set requested for recovery, not a tuning set; image IDs, scores, source images and patient groups are absent. The apparent target pass is not independently reproducible and must not be used for threshold selection. |
| RFMiD partitions in C:\Users\doifo\Downloads\OFFIRST_Accuracy_Audit\data\rfmid_partition_manifest.pseudonymized.csv | Separate model audit records 1,536 train, 192 validation and 192 test labels; 33 positive / 159 negative in validation | The audit explicitly records that RFMiD patient keys are unavailable. The 192-image validation set is not the 69-image assessment. No source images or binary app-model predictions were included, and v1.1/v2.1 exposure/selection disjointness is unverified. |
| ODIR grouped validation in the separate local audit bundle | 1,283 validation eyes from 672 patients; this split is patient-disjoint from its 5,109-eye training partition | It belongs to an eight-class ODIR feature-probe task, not the app's fixed binary endpoint; ODIR predictions are for that historical probe, not the deployed v1.1/v2.1 models. No images are present, and binary-head exposure/selection independence is unproved. |
| IDRiD grading split in kd_run2/idrid_grading_eval/ | 413 train / 103 test labels and 516-row feature metadata | It is a separate frozen-encoder grading evaluation, not the 72-image application assessment; no original image files or verified patient-group IDs are available. App head and threshold exposure overlap cannot be excluded. |
| v2.1 model-selection metadata | Development head metrics: accuracy 0.8493, sensitivity 0.8710, specificity 0.8333, AUROC 0.9414, AUPRC 0.9419 | These are marked development-only. The exact patient/image IDs, split, predictions and provenance needed to qualify a new threshold-development cohort are absent. |

The separate local accuracy-audit bundle identifies its RFMiD validation partition as image-level because patient IDs are unavailable. The ODIR validation partition is grouped, but its endpoint and prediction artifacts are from a different model/task. Neither qualifies as a validated patient-grouped threshold set for the app.

## Data required before any sweep

1. Authorized source photographs and stable hashes for a patient-disjoint validation cohort representative of the intended camera/site population.
2. Adjudicated binary DR endpoint, the exact severity definition (v1.1/v2.1 currently differ in documented label mapping), gradability, and patient/site/device IDs or pseudonymous equivalents.
3. Complete fitting/calibration/model-selection exposure manifests for both frozen models, proving candidate validation cases did not enter encoder/head fitting, calibration, threshold selection, or previously viewed assessments. Remove exact and near duplicates across partitions.
4. v1.1 and v2.1 probabilities on identical source images, with image/tensor hashes and the exact frozen preprocessing. Validate Python/Android parity separately.
5. A predeclared threshold selection rule and safety operating criteria. Use validation only to sweep candidate thresholds for accuracy ≥85%, sensitivity ≥85%, and specificity ≥80%; report the trade-off and uncertainty without changing the frozen cutoff in this investigation.
6. A separate untouched patient/site-held-out test manifest, sealed before selection. Freeze selected threshold(s) and manifest hashes before one final test evaluation. Do not inspect test scores while selecting a threshold.

Until these requirements are met, the 69-image RFMiD and 72-image IDRiD assessments remain reporting-only. Their aggregate scores do not permit ROC/PR reconstruction or threshold feasibility analysis, and the existing dataset audit partitions do not establish an eligible independent validation cohort.
