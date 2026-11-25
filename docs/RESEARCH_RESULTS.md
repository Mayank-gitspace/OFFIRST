# Research results and reproducibility

## Application-level DR classification

The following are development assessment results at the models' recorded operating thresholds. They are not independent clinical validation. Exact patient/exam identifiers are not available for a verified patient-grouped assessment; patient independence is unverified, prior dataset exposure exists, and IDRiD and RFMiD use different label protocols. Interpret metrics per dataset rather than pooling them.

| Model | Dataset | Images | Accuracy | Sensitivity | Specificity | Balanced accuracy | F1 | AUROC | AUPRC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v2.1 | IDRiD | 72 | 79.17% | 87.23% | 64.00% | 75.62% | 84.54% | 0.874 | 0.944 |
| v1.1 | IDRiD | 72 | 70.83% | 57.45% | 96.00% | 76.72% | 72.00% | 0.846 | 0.932 |
| v2.1 | RFMiD | 69 | 81.16% | 54.55% | 86.21% | 70.38% | 48.00% | 0.875 | 0.640 |
| v1.1 | RFMiD | 69 | 86.96% | 100.00% | 84.48% | 92.24% | 70.97% | 0.967 | 0.853 |

The v2.1 IDRiD confusion matrix (TN, FP, FN, TP) was `(16, 9, 6, 41)`; its RFMiD matrix was `(50, 8, 5, 6)`. v2.1's combined working gates of 85% accuracy, 85% sensitivity, and 80% specificity are not met on either dataset. RFMiD sensitivity of 54.55% is a substantial false-negative limitation. The v1.1 results are from the same development assessments and are not an independent comparison.

Model-selection partition metrics must not be confused with assessment results. Thresholds or hyperparameters were not selected using an independent, untouched clinical test cohort. See the [v2.1 model evidence](../deployment/offline_dr_screening/assets/model_v2_1_evidence/model_manifest.json) and [final handoff](../deployment/offline_dr_screening/FINAL_RELEASE_HANDOFF.md) for the app model contract and verification record.

## Frozen representation probes (not app classifier metrics)

The following AUROCs are frozen-encoder linear-probe or few-shot representation evaluations. They are not accuracy, sensitivity, or specificity of the deployed v2.1 application classifier.

| Evaluation | Direct BYOL MobileNetV3 | KD Run 2 MobileNetV3 |
| --- | ---: | ---: |
| RFMiD linear-probe AUROC | 0.917 | 0.932 |
| RFMiD 10-shot AUROC | 0.640 | 0.728 |
| ARMD 10-shot AUROC | 0.789 | 0.949 |

These research evaluations have dataset exposure and split limitations; KD Run 2 used a BYOL-initialized MobileNetV3 student and a BYOL ConvNeXt teacher. Its objective is feature distillation, not explicit Masked InfoNCE. The application head was fitted separately later for the v2.1 binary DR task.

## Distinguishing self-supervised and contrastive work

BYOL is non-contrastive self-supervised learning: it matches related views without explicit negative examples. Masked InfoNCE is an explicit contrastive objective and belongs to a separate research strand; it is not part of the deployed v2.1 model and was not part of KD Run 2. Older v1.1 materials describe a BYOL + Masked InfoNCE model, but the complete training implementation and evidence needed to reproduce that claim are not present in this checkout. The v1.1 ONNX and release remain preserved as historical rollback artifacts.

## Archived external exploratory result

The former v1.1 cutoff-0.48 model was scored on 1,742 DeepDRiD images from 488 patients: accuracy 54.54%, sensitivity 21.47%, specificity 98.27%, AUROC 0.764. Historical training exposure and endpoint/label-definition limitations were unresolved. This is exploratory evidence, not clinical validation, and the low sensitivity indicates a serious false-negative limitation.

## Research directory map

| Research line | Location | Scope |
| --- | --- | --- |
| Supervised encoder baselines | [`encoder_baseline/`](../encoder_baseline/README.md) | Multi-label RFMiD encoder probes, not the app's binary DR head |
| BYOL studies | [`encoder_byol/`](../encoder_byol/README.md) | ConvNeXt self-supervised representation research |
| Direct MobileNetV3 BYOL | [`encoder_mnv3_ssl/`](../encoder_mnv3_ssl/README.md) | MobileNetV3 encoder probes |
| Knowledge distillation | [`kd_run2/`](../kd_run2/README.md) | BYOL teacher/student representation distillation and evaluations |
| Imaging analyses | [`kd_imaging/`](../kd_imaging/README.md) | Separate image-quality and representation studies |
| Lesion segmentation | [`lesion_segmenter/`](../lesion_segmenter/README.md) | Separate segmentation research; not shipped in OFFIRST |
| Utilities | [`tools/`](../tools/README.md) | Experiment and evaluation helpers |

Research results do not imply those models or features are present in the app. Do not redistribute restricted datasets or patient records. Follow each dataset's access terms and citation requirements; use only permitted files for reproduction. Some runs require Kaggle inputs or local artifacts that are not part of a clean checkout.

## Engineering verification scope

For the current v2.1 source, formatting and analysis pass; the Flutter suite recorded 16 passed and 2 skipped in the v2.1 configuration, and 17 passed and 1 skipped for the explicit v1.1 rollback configuration. The retained Python/OpenCV plus ONNX evidence replays 141 saved assessment images with maximum absolute probability difference `1.8341754449702208e-06` and zero threshold disagreements. The saved images are not distributed, and this does not establish Android preprocessing parity.

No Android device or emulator was connected for the final v2.1 publication verification. Device installation, Android-native preprocessing parity, airplane-mode inference, data-preserving upgrade, and real PDF viewing/export remain unverified. These engineering checks do not establish diagnostic reliability.
