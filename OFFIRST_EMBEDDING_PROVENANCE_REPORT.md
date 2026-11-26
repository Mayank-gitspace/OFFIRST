# OFFIRST embedding provenance report

Audit baseline: 38a0ba31996eef453ead711b845dc5aa593905c5. Arrays were inspected read-only for shape, dtype, sample count, and row L2 norms. Hashes below identify bytes now in the checkout; absent source images prevent independent extraction replay.

## KD Run 2 canonical arrays

| Artifact | Shape / dtype / size | Normalization | ID / label / split sidecar | Provenance status |
|---|---|---|---|---|
| kd_run2/full_out/out/kd_run2/embeddings_kd768.npy | 1,920×768; float32; 5,898,368 bytes; SHA-256 ed9e1fa9d48908f1d5df4c9b3322b891153913ee3c083b257dc8095ab1d097ad | L2-normalized per row (sampled row norms 1.0) | embedding_ids.csv, SHA-256 d7db32ef7f16fe58f1e3ee59f16eacba63ec83dd056de046582914d0626041aa; exact aligned columns ID,fold; 1,920 IDs and 384 rows/fold. split_used.csv is byte-identical. No DR labels or patient identifiers. | Full-run KD student final 768-D features by pipeline name/output code. Run metadata says encoder is 40-epoch BN-recalibrated student. The original kd_final.pth is absent; extraction output has no checkpoint SHA field, so array bytes are not cryptographically bound to the remotely referenced checkpoint hash. |
| embeddings_kd768_degraded.npy | 1,920×768; float32; 5,898,368 bytes; SHA-256 f5e936dcc843d79066e0e0552b5fdd964cb3836a5c95c1e2c81ab38f9a6d14b9 | L2-normalized | Same ID,fold rows | Synthetic Visuscout-style degraded variant for robustness metrics. Not real low-quality acquisition evidence. |
| embeddings_kd960.npy and embeddings_kd960_degraded.npy | Each 1,920×960; float32; 7,372,928 bytes | L2-normalized | Same ID,fold rows | Backbone diagnostic before 768-D projection; not app representation. |
| embeddings_ssl_ep40_960*, embeddings_ssl_ep40_768procrustes*, embeddings_teacher* | Each clean/degraded array has 1,920 rows; dimensions respectively 960, 768, 768; float32 | L2-normalized | Same ID,fold rows | BYOL MNV3 initialization, fixed-Procrustes projection, and BYOL ConvNeXt teacher comparators. Not KD final encoder. |

The 1,920 rows cover RFMiD development IDs and all folds, not just held-out rows. Folds 1–3 were KD student training rows, fold 4 inner validation, and fold 0 development evaluation. Fold tags are image-level, not patient-grouped. The sidecar lacks diagnosis labels, patient ID, site and device. ID is the RFMiD source image ID in the extraction pipeline; joining to DR labels requires the original label file and a verified key join. Do not recreate a join from array order.

## Cross-dataset KD768 arrays

| Artifact | Shape / dtype / size and SHA-256 | Row mapping and labels | Provenance and limitations |
|---|---|---|---|
| kd_run2/idrid_grading_eval/embeddings/kd768.npy | 516×768; float32; 1,585,280 bytes; 134edacd5f7ab083158b853ea90e786c622f13592db5ef2b3cc6de8b75605ac9 | ids.csv, SHA-256 601db22ff8461d91e44fab4b62f3c6363fc6c335f5c9e0534a1d6260afeb6849; fields split,name,dr,dme,raw_h,raw_w,fov_radius_frac; 413 train / 103 test; image-level DR grades 0–4. Extractor writes sidecar in feature-row order. | Extractor explicitly loads kd_final.pth, asserts epoch 40 and BN recalibration, and uses 960→768 student. Cross-check against 16 RFMiD kernel embeddings had cosine minimum 0.999884 for KD768 (CPU vs saved kernel path). No source image hash or verified patient-group field; 103 basenames collide across split namespaces, so use composite split+name, not basename. Original images absent. The 103-row test has prior downstream use. |
| idrid_grading_eval/embeddings/kd768_degraded.npy | 516×768; float32; 1,585,280 bytes | Same sidecar | Synthetic degradation, not real capture-quality outcomes. |
| kd_run2/odir5k_eval/embeddings/kd768.npy | 6,565×768; float32; 20,167,808 bytes; f80e8ae7ac97f7ea54ee5eedc466d6ba4ff7eb96e1e2507eb431d9efacb1236b | ids.csv, SHA-256 be928127e952a1215376eedde427f397b741a211276e1da3859deb7bb4f6d7c0; fields patient,eye,file,keywords and split; 4,583 train eyes / 2,343 patients; 1,982 test eyes / 1,015 patients. | Extractor explicitly makes patient-level multilabel-stratified 70/30 split and asserts each patient's eyes stay in one partition. ODIR labels are keyword-derived and multi-label, not adjudicated DR grades. KD representation report has already evaluated the held-out split. No source photos present. |
| odir5k_eval/embeddings/kd768_degraded.npy | 6,565×768; float32; 20,167,808 bytes | Same ODIR sidecar | Synthetic degradation; not real image-quality or device robustness. |

Other IDRiD and ODIR arrays (teacher, BYOL SSL960, Procrustes 768 and degraded variants) have the same sample counts and row sidecars but belong to separate representation comparisons. The complete scan also found BYOL ConvNeXt arrays at 1,920×768 (plus 2,304-D multi-region concatenations), MNV3 BYOL arrays at 1,920×960, local smoke/test subsets (200/400/750 rows), and geometry arrays. These alternatives are not substitutes for verifying the exact KD final checkpoint.

## Row mapping, labels, patients, and exposure

- RFMiD: exact image ID/fold sidecar; no labels or patient IDs beside features. RFMiD photos and a verified label join are absent. RFMiD rows contributed to KD training, selection, or development according to fold; a fold is not a patient group. Prior locked comparisons disclose overlap with SSL/KD exposure.
- IDRiD: exact sidecar carries grades and train/test split. Binary mappings must be declared (any DR = grades 1–4 vs referable DR = grades 2–4). Sidecar row alignment follows extractor output, but image bytes/hashes and a patient-group key are absent. The 103-row test has prior downstream use.
- ODIR: exact sidecar joins features to per-eye keyword labels and patient groups. The split is patient-disjoint; prior representation probes have already used held-out data. The D flag is keyword-derived, and no photos are available for visual adjudication or quality review. This is limited development/probe evidence, not fresh DR acceptance.
- External acceptance: no current cache establishes source-image hashes, adjudicated patient-level labels, a fresh locked split, site/device labels, and exclusion of prior model exposure together.

## Reproducibility limits

The full-run scripts describe the encoder, extraction steps and intended row order; companion CSVs map feature rows to identifiers. IDRiD and ODIR extraction scripts refer to kd_run2/full_out/out/kd_run2/kd_final.pth, which is absent, and source photos are absent. The v2.1 model manifest records the checkpoint hash, but no hash for each .npy array or source image manifest. Therefore:

1. Existing feature matrices can be joined to local metadata where an explicit companion sidecar exists.
2. The local RFMiD array cannot be joined to DR labels/patients using only its sidecar.
3. No cached feature matrix can be independently regenerated against the exact released checkpoint in this checkout.
4. A feature array is not evidence of a clinical outcome or a substitute for the source image and exposure record.

## Dataset rights and derived artifacts

Dataset permissions follow the underlying source and version. The fact that an embedding or prediction is cached locally, or that its source photograph is no longer present, does not establish permission to reuse or redistribute it. Do not add the arrays, labels, source images, row-level predictions, or derivatives to a public branch until the exact dataset source/version, applicable terms, and permitted derivative/distribution use are recorded.

- RFMiD and IDRiD dataset papers state CC BY 4.0; retain attribution and verify that the local cache is from the corresponding official dataset release. Their permissive terms do not repair the missing RFMiD patient/label joins or IDRiD patient/exposure gaps.
- ODIR-5K is registration-gated in the reviewed official page; no dataset reuse license was located. Treat its cached features as uncleared until organizer terms are confirmed.
- DeepDRiD's reviewed Zenodo record has no license value; a paper license or public download is not a dataset license. Do not reuse cached/derived data until dataset-specific terms are obtained.
- DDR is published under CC BY-NC-SA 4.0. That noncommercial/share-alike condition is not presumed compatible with OFFIRST or any intended distribution.
- APTOS is governed by competition rules restricting use to noncommercial purposes and restricting data sharing. It is not treated as an unrestricted open dataset.

The full source/terms, verification status, exposure, and conservative reuse stance are recorded per artifact in `OFFIRST_DATA_LINEAGE_AND_EXPOSURE_MATRIX.csv`. No images or embeddings were copied, modified, or published as part of this licensing update.
