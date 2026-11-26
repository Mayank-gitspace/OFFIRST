# OFFIRST development data inventory

**Scope:** actual local files and metadata at `38a0ba31996eef453ead711b845dc5aa593905c5`. Dataset availability is based on local artifacts, not documentation or download references alone. No source image, label, split, embedding, or research directory was changed.

## Availability summary

| Dataset / asset | What is locally present | Labels and counts recoverable from local files | Patient/split status | Limitations |
|---|---|---|---|---|
| RFMiD training source | No original image set and no complete joined DR-label CSV found. Local research holds 1,920 embedding rows per encoder family and `ID`/fold IDs. | Five folds of 384 cached image IDs each. Local cache does not itself carry the disease label matrix. | Fold assignment exists; source metadata describes the five-fold construction, but no patient key or verified patient-group split was found in the local embedding ID files. | Cached embeddings are model outputs, not image data. Prior RFMiD representation experiments report exposure overlap; historical 69-image app assessment lacks row-level joins. |
| IDRiD Disease Grading | No original photographs. Cached 516×768 KD features and corresponding metadata file are present. | 516 row labels: grade 0=168, 1=25, 2=168, 3=93, 4=62. The report partitions 413 train / 103 test; train grades: 0=134, 1=20, 2=136, 3=74, 4=49; test grades: 0=34, 1=5, 2=32, 3=19, 4=13. | Metadata uses `IDRiD_###` names and `train`/`test`; no explicit, independently verified patient-group ID is included. | The 103-row partition has already been scored in prior work. It is not a fresh acceptance cohort. Local data consists of features and labels/metadata, not source images. |
| ODIR-5K | No original photographs. Cached 6,565×768 KD features, per-eye metadata, and a 7,000-row label table are present. | Across the 6,565 feature rows: keyword disease flags D=1,778; N=2,818; G=313; C=288; A=280; H=193; M=262; O=1,124 (multi-label, overlapping flags). Full 7,000 rows: D=1,801; N=3,095; G=326; C=313; A=280; H=193; M=268; O=1,200. | 3,358 unique patient IDs in feature rows: train 4,583 eyes / 2,343 patients; test 1,982 eyes / 1,015 patients. Metadata contains patient and eye keys. | Keyword-derived labels are noisy and multi-disease; `D` is not a verified severity-graded DR label. Not a substitute for an approved DR screening dataset. |
| EyeQ / image quality | `tools/quality_guard_poc/` contains scripts for EyeQ-based training and APTOS synthetic degradation; no trained quality model or verified EyeQ image/label bundle is present. | No local gradability cohort was found. | No patient/site/device partition evidence. | The PoC README explicitly says training/export have not run; synthetic defects are software fixtures only. Do not treat those heuristics as a fundus detector. |
| ODIR review images | Five example `.png` sample montages under `kd_run2/odir5k_eval/samples/`. | None beyond the related per-eye metadata. | No complete image collection. | Sample montages cannot support model fitting or independent evaluation. |
| App assessment bundles | Aggregate reports for RFMiD n=69 and IDRiD n=72; no source images or case-level labels/probabilities in the repository. | v1.1/v2.1 aggregate metrics and some confusion matrices only. | Exact sample IDs, patient joins, image hashes, and sites/devices unavailable. | Cannot reproduce individual predictions or infer subgroup distributions. |

## Duplicate and partition evidence

- The local RFMiD embedding ID tables have 1,920 rows, with 384 rows in each of five folds. These are image identifiers and folds, not patient-level split records. Image hashes cannot be recomputed without the source photographs.
- The IDRiD metadata has 516 named rows. It does not carry a patient-group column; the name field is an image identifier, not proof of patient grouping. Duplicate-image checks cannot be performed without image bytes or a hash manifest.
- ODIR feature metadata carries 6,565 unique filenames among 6,565 rows and patient/eye IDs. The train/test report is patient-level (2,343 train patients, 1,015 test patients). Content-level duplicate hashes were not checked because source photos are absent.
- No patient overlap can be evaluated across RFMiD and IDRiD or between historical app assessments and feature datasets using only the local metadata.

## Prior OFFIRST exposure

- RFMiD was used for representation learning and evaluation. BYOL v1/v4, MNV3 BYOL, and KD Run 2 use RFMiD fold-based splits; the saved locked-rescore reports explicitly disclose that some “locked” query images overlap prior label-free encoder training. Therefore those results are not an untouched external clinical test.
- The historical OFFIRST app reports contain 69 RFMiD and 72 IDRiD assessments, but image IDs are unavailable in the local summary. Their overlap with available RFMiD/IDRiD records cannot be established.
- The local IDRiD 413/103 cached-feature evaluation report says the encoders did not train on IDRiD images. However, the 103-image test partition has subsequently been used by more than one downstream evaluation, including the saved binary probe. It is not fresh for future head/threshold selection.
- ODIR-5K was used for patient-level held-out representation probes. It has another endpoint and cannot validate an any-DR head without a verified relabeling protocol and eligible images.

## Missing inputs before a defensible supervised head experiment

1. Authorized source image files for the selected dataset, with stable pseudonymous image IDs and SHA-256 hashes.
2. Ground-truth label source and interpretation rules (binary any-DR vs referable DR vs grade 0–4); the current binary endpoints are not interchangeable.
3. Patient-group IDs and eye/visit mapping, plus site and acquisition-device fields where available.
4. Frozen partition manifest recording pretraining, supervised fitting, validation, calibration, and untouched test exposure for every image/patient.
5. A label join for the cached embeddings that verifies row order against image IDs and the exact encoder/checkpoint hash and preprocessing.
6. Independent real images for external-site/device evaluation and quality/gradable outcomes.

Without these, a head can be fit technically on cached arrays, but resulting metrics cannot establish patient-independent screening performance or satisfy the requested real-image acceptance gates.

## Reproducible local checks

The inventory confirmed the dimensions with `numpy.load(..., mmap_mode='r')` and row labels with the committed CSVs. The checkout contains zero model checkpoint files (`.pt`, `.pth`, `.ckpt`, `.safetensors`), two frozen ONNX assets, and no full RFMiD, IDRiD, ODIR, or EyeQ photo collection. The five ODIR sample montage files and research view/degradation illustrations are not source evaluation cohorts.
