# Research results and reproducibility map

## Deployed v1.1 exploratory benchmark

The unchanged release model at cutoff 0.48 was evaluated on 1,742 external exploratory DeepDRiD images from 488 patients:

| Accuracy | Sensitivity | Specificity | AUROC |
| ---: | ---: | ---: | ---: |
| 54.54% | 21.47% | 98.27% | 0.764 |

This is not independent clinical validation. Historical training exposure and endpoint/label-definition limitations remain unresolved. The score ranking did not support both an 85% accuracy and 85% sensitivity operating point in the retrospective sweep. The model and cutoff were not altered for publication.

## Separate experiments

| Line | Directory | Interpretation |
| --- | --- | --- |
| BYOL + Masked InfoNCE epoch 3 | `deployment/offline_dr_screening/` and related model notes | Released v1.1 classifier; explicit contrastive term is Masked InfoNCE |
| Direct BYOL MobileNetV3 epoch 40 | `encoder_mnv3_ssl/` | Research encoder experiment, distinct from deployed epoch 3 classifier |
| Knowledge-distilled MobileNetV3 epoch 40 | `kd_run2/` | Teacher/student feature matching, not explicit InfoNCE |
| Experimental v2 DR classifier | Research evaluation materials | Candidate; not shipped and did not pass all acceptance gates |
| Experimental v2.1 conventional-fundus classifier | Research evaluation materials | Candidate; not shipped and did not pass all acceptance gates |

The original repository README contains the ODIR-5K, IDRiD and RFMiD experiment context. In particular, ODIR results are derived from per-eye clinical keywords; the RFMiD development evaluation includes pretraining overlap and must not be read as an independent result. See [KD Run 2](../kd_run2/README.md), [BYOL v4](../encoder_byol/v4/README.md), [MobileNetV3 SSL](../encoder_mnv3_ssl/README.md), [baseline experiments](../encoder_baseline/README.md), [imaging analysis](../kd_imaging/README.md), and [segmentation research](../lesion_segmenter/README.md).

## Recorded verification (prior runs)

The v1.1 readiness report records 13 Flutter tests passed, 24 preprocessing fixtures with bitwise tensor and probability agreement, and successful split APK builds. Another recorded audit reports an ARM64 emulator offline workflow covering bilateral screening, saved visit reopening and PDF export. These are historical checks documented before this publication; they do not establish clinical reliability or physical-device compatibility. Physical-device testing remains incomplete.

See [release readiness](../deployment/offline_dr_screening/RELEASE_READINESS_V1_1.md), [preprocessing audit](../deployment/offline_dr_screening/PREPROCESSING_AUDIT.md), and [offline mobile evaluation notes](../deployment/offline_dr_screening/MOBILE_OPTIMIZATION.md).

## Reproduce responsibly

Evaluation scripts and saved, permitted embeddings/manifests are located within the research directories. Some runs depend on Kaggle inputs, licensed images or paths unavailable in a fresh checkout. Do not upload restricted datasets, patient records, private images, Kaggle credentials or private absolute paths. Follow the specific experiment README and dataset license before reproducing a metric.
