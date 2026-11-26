# OFFIRST Option D feasibility decision

## Decision: CONDITIONAL GO

Option D may proceed only after the exact encoder asset and a fit-for-purpose patient-grouped DR development cohort are recovered and provenance is verified. Do not start fitting a head now. The model artifact is precisely identified and obtainable, but local caches and historical summaries do not currently supply a scientifically defensible, untouched any-DR development/validation/test arrangement.

## What is resolved

- The exact encoder for v2.1 is identified as kd_run2_kd_final.pth, from the GitHub kd-run2 release, SHA-256 8bc239d22ffc4e786820f2151c99380a754bef930e42edea65022e1246f2096d. The same digest is in deployment/offline_dr_screening/assets/model_v2_1_evidence/model_manifest.json.
- Architecture and extraction pipeline are available in kd_run2/src/kd_tail.py, kd_run2/kernels/full/team_offirst_kd_run2.py, and kd_run2/idrid_grading_eval/extract.py / kd_run2/odir5k_eval/extract.py.
- KD768 features are locally cached for RFMiD, IDRiD, and ODIR. The latter two have explicit row sidecars; ODIR also has patient IDs and a patient-disjoint train/test split.
- The v2.1 classifier head is part of the frozen ONNX and its cutoff/calibration are recorded. It can remain an unchanged comparator.

## Conditions before any head fitting

1. Recover and verify the exact checkpoint. Download [kd_run2_kd_final.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/kd-run2/kd_run2_kd_final.pth), then verify SHA-256 exactly equals 8bc239d22ffc4e786820f2151c99380a754bef930e42edea65022e1246f2096d. Load with strict architecture checks and verify epoch 40 plus BN recalibration. Do not substitute the v1.1 TorchScript .pt, old epoch-3 BYOL+InfoNCE checkpoint, direct BYOL MNV3 checkpoint, or smoke checkpoint.
2. Obtain a suitable fresh development cohort. It must include real image files or immutable authenticated inputs, adjudicated DR grades and a prespecified binary label mapping, patient IDs spanning both eyes/repeated visits, source image hashes, and a patient-grouped train/validation split. Record site/device metadata where available.
3. Reconcile prior use. Get case-level IDs/hashes for historical 69-image RFMiD and 72-image IDRiD v2.1 assessments, and the 1,742-image/488-patient DeepDRiD v1.1 assessment. Exclude those cases/patients from any new untouched test. Existing records do not contain the required joins.
4. Prove cached feature lineage. For any retained feature cache, verify row-to-image mapping, checkpoint digest, preprocessing and normalization. Prefer re-extraction from authorized images using the hash-verified checkpoint. Current RFMiD cache cannot support a clean patient-level head experiment because its sidecar has no label/patient key. IDRiD has labels but lacks verified patient grouping and its 103-row test has already been used.
5. Lock evaluation first. Keep current v2.1 ONNX, calibration, and threshold unchanged as comparator. Fit one lightweight head on frozen KD768 vectors using patient-grouped development only. Choose any candidate threshold on a designated validation cohort, then lock before a fresh untouched patient/site test.

## Dataset-specific feasibility

### Dataset permission gate (added 2026-11-26)

Public download availability is not treated as permission. Source image, annotation, embedding, prediction, and model-release decisions must retain the source dataset and license version; cached features do not become unrestricted merely because the original images are absent. Do not copy source images or derived feature/prediction files into this repository until the specific dataset terms permit that use and distribution.

| Dataset | Published terms reviewed | Phase 4C reuse decision |
|---|---|---|
| RFMiD | Original dataset paper states CC BY 4.0. | Potentially eligible with attribution, exact release/version verification, and exposure/patient-group joins. Current 1,920-row cache has no label or patient mapping and prior RFMiD exposure; it is not eligible as fresh evaluation. |
| IDRiD | Dataset paper states CC BY 4.0. | Potentially eligible with attribution. Obtain the official source package and verify image IDs, patient grouping, license notice, and prior exposure. The local 103-row test is already exposed and cannot serve as untouched evaluation. |
| DDR | Dataset repository states CC BY-NC-SA 4.0 for dataset material. | Noncommercial/share-alike restrictions are not cleared for OFFIRST use or downstream model distribution. Exclude unless project use and intended distribution are confirmed compatible or dataset owners grant permission. |
| DeepDRiD | Reviewed Zenodo dataset record has no populated license field; challenge/article publication terms do not themselves license the data. | Not cleared. Obtain dataset-specific terms from the organizers before use; prior OFFIRST assessment overlap also remains unresolved. |
| APTOS 2019 | Official competition rules limit data use to noncommercial purposes and restrict sharing beyond participants. | Not cleared for general OFFIRST development/distribution. Consider only if the specific work is noncommercial and complies with accepted competition rules; do not redistribute images or derived files. |
| ODIR-5K | Official Grand Challenge page provides registered access; no dataset-specific reuse license was located in reviewed materials. | Not cleared pending organizer terms. Existing local features have prior evaluation exposure and keyword-derived labels, so not an acceptance cohort regardless. |

**Current decision remains CONDITIONAL GO**, now subject to both scientific lineage and permission checks. The least-friction candidate is a verified official IDRiD CC BY 4.0 package for development only, using its training portion after patient grouping and exposure reconciliation. A fresh untouched test still requires a separately cleared, eligible source and case-level overlap checks. RFMiD may be considered only after confirming that the local source version matches the publication's CC BY 4.0 dataset. No training is authorized by this report.

License references: [RFMiD dataset paper](https://www.mdpi.com/2306-5729/6/2/14), [IDRiD dataset paper](https://www.mdpi.com/2306-5729/3/3/25), [DDR dataset repository](https://github.com/nkicsl/DDR-dataset), [DeepDRiD Zenodo record](https://zenodo.org/records/8248825), [APTOS competition rules](https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules), [ODIR-5K dataset page](https://odir2019.grand-challenge.org/dataset/).

| Candidate | Development use | Acceptance use |
|---|---|---|
| RFMiD | Cached KD768 rows have image ID/fold but no local labels or patient keys joined. Obtain authorized image/label files and a verified patient-grouping source; without a patient key, do not call the split patient-disjoint. | Current folds have direct KD exposure and no patient groups; not eligible as untouched patient-level test. |
| IDRiD | Official grading labels and cached KD768 vectors exist for 516 named rows. Current sidecar has 413/103 train/test, but patient-group membership is not established locally and 103-row test has already been evaluated. Could be training/dev only after authoritative image/patient join and exposure review. | Current test is not untouched; no fresh IDRiD acceptance claim. Official source offers grading labels for 516 images, but this checkout lacks original images. |
| ODIR-5K | Cached features, labels, patient/eye IDs, and a patient-disjoint split are present. Its D-vs-rest target is keyword-derived and held-out split was already used for representation evaluation. At most use as exploratory head-pipeline development, with endpoint limitations stated. | Not a clinically adjudicated DR acceptance set, and held-out data have already been viewed. |
| DeepDRiD | Official dataset provides DR grading and image-quality outcomes with multiple images per patient, so it could support development only after source images/labels and patient mapping are recovered. | OFFIRST already reports prior v1.1 screening of 1,742 images/488 patients, but exact cases are unavailable locally. Do not use potentially overlapping patients as untouched test until reconciled; otherwise acquire a different site/cohort. |

The DeepDRiD authors describe 2,000 regular fundus images from 500 patients, four images per patient, adjudicated DR grades and image-quality scoring. The official repository publishes challenge training/validation/external-validation data, and the challenge paper describes patient-level grading results. This makes it a plausible source to investigate, not an automatically eligible cohort: remove OFFIRST's prior evaluation exposure first. [Official DeepDRiD repository](https://github.com/deepdrdoc/DeepDRiD), [challenge paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9214346/).

IDRiD's official data description states image-level DR/DME grading is available for 516 images with 413/103 train/test rows; it does not provide a patient-group crosswalk in the local cache. [IDRiD data page](https://idrid.grand-challenge.org/Data/), [dataset paper](https://www.mdpi.com/2306-5729/3/3/25).

## Exact missing items

- Local hash-verified copy of exact KD final release asset (artifact is obtainable at the link above).
- Authorized, fresh patient-grouped DR development and untouched test data with source images/IDs/hashes, labels/severity, partition manifest and exposure ledger. A fresh site-disjoint test is required for an external-site claim.
- Historical assessment case IDs/hashes and patient mapping for RFMiD n=69, IDRiD n=72, and DeepDRiD n=1,742 / 488 patients to prevent overlap.
- For RFMiD, original label table joined by exact source image ID and a reliable patient grouping source; local RFMiD cache does not provide these.
- Re-extraction or cache-to-checkpoint parity evidence tying each reused feature array to recovered checkpoint hash and exact image preprocessing.

Until these conditions are met, the scientifically valid Option D experiment is not ready. No training, model export, threshold change, or production modification was performed in Phase 4C.
