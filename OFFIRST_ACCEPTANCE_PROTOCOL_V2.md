# OFFIRST real-image acceptance protocol v2

## Status

Acceptance remains blocked until an authorized, patient-grouped real-image cohort and actual Android evidence are supplied and the locked test passes. Synthetic evaluator tests are software checks, not screening evidence. No model or production threshold is changed by this protocol.

## Freeze before collection/scoring

Record protocol revision, clinical endpoint and DR severity mapping, model version and ONNX SHA-256, exact preprocessing/tensor contract, locked threshold and `>=` comparator, reference-label adjudication method, inclusion/exclusion flow, acquisition sites/devices, analysis seed, minimum counts, and whether external-site evaluation is required. Keep test images and predictions sealed until validation work is complete. Do not use the historical RFMiD/IDRiD assessment sets for threshold optimization.

## Data and partition integrity

Use authorized actual retinal photographs with adjudicated positive/negative labels, severity, gradability, modality, site/device and pseudonymous patient group. Keep every image, repeated capture, both eyes, and visits from one patient within one partition. Every validation/test image requires a non-empty `patient_group_id`. The evaluator blocks patient, sample, or exact image-hash leakage across validation/test. If the study claims external-site evaluation, run with `--external-site-evaluation`; validation and test site groups must both be present and disjoint. De-duplicate near-identical acquisitions before locking; exact hashes alone cannot discover resized/recompressed duplicates.

Every gradable fundus image must have an adjudicated severity value. Test acceptance also requires both OD and OS cases to exercise both workflows, without requiring each individual patient to have both eyes imaged.

Manifest `patient_eye_status` explicitly accounts for bilateral, unilateral, missing-eye-record and unknown cases. Bilateral declarations require OD+OS records. A unilateral patient remains eligible when only one eye was clinically available. Missing/unknown eye status is reported and should be resolved or justified before acceptance; software does not invent an absent eye image.

The locked test cohort must include both OD and OS cases to exercise both application workflows, while allowing individual patients to be unilateral or have a missing-eye record.

## Predeclared minimum cohort floors

Default evaluator floors for each partition are: 30 unique patients, 15 patients with at least one positive gradable image, 15 with at least one negative gradable image, 30 gradable images, 1 ungradable image, 2 sites, and 2 devices. These are conservative engineering guardrails against tiny or single-stratum cohorts, not universal sample-size guarantees, confidence-interval precision guarantees, or substitutes for a prospective power calculation. Multiple images/eyes increase image counts but never patient counts. A patient may count in both positive and negative patient totals when eyes have different labels. Larger cohorts may be needed to make the lower confidence-bound criteria estimable and sufficiently narrow. CLI minimums can be raised from defaults but cannot be lowered; record any raised values before outcomes are reviewed and document the justification.

## Run protocol

Record one row per image, platform and inference repeat using `tools/real_image_acceptance/manifest_template.csv`. Store actual images in restricted storage outside Git; retain pseudonymous IDs and hashes in the analysis manifest. For scored entries include the tensor SHA-256. Run frozen Python and Android models on identical FP32 tensor bytes where parity is assessed. Capture actual phone/device build, gallery/camera route, airplane mode, persistence and PDF evidence separately; a manifest cannot prove those actions occurred.

Example:

```powershell
python tools/real_image_acceptance/evaluate_acceptance.py `
  path\to\locked_manifest.csv `
  --image-root path\to\restricted\images `
  --check-image-hashes `
  --model-version offirst-v2.1-experimental-43a87f05 `
  --locked-threshold 0.3675977844516245 `
  --external-site-evaluation `
  --bootstrap-reps 5000 `
  --seed 20261126 `
  --output path\to\restricted\acceptance-report.json
```

Use `--external-site-evaluation` only for a protocol that predeclares held-out sites. Otherwise patient-level split integrity still applies. Minimums can be predeclared via `--min-unique-patients`, `--min-positive-patients`, `--min-negative-patients`, `--min-sites`, `--min-devices`, `--min-gradable-images`, and `--min-ungradable-images`.

## Statistical analysis and pass criteria

Primary accuracy, sensitivity and specificity are patient-clustered bootstrap percentile 95% confidence intervals. Sample eligible patients with replacement and carry all gradable images/eyes/repeats from each draw; patients without gradable fundus images are outside the DR metric sampling frame. The evaluator withholds an interval if any requested replicate lacks that metric's required class/denominator or if the bootstrap values are degenerate. It does not condition on valid draws, and unavailable intervals fail closed. Pass requires lower confidence bounds: accuracy >=0.85, sensitivity >=0.85, specificity >=0.80. AUROC >=0.85 is an additional point-estimate gate on scored gradable fundus cases only. Report point estimates, image-level Wilson intervals, denominators, coverage, confusion matrix, patient counts, valid/invalid bootstrap draw counts, AUROC interval, severity/device/site/dataset/eye strata and all individual failures.

An abstention/failed screening on a gradable reference case remains an end-to-end error: positive becomes FN; negative becomes FP. A positive/negative decision without a probability is malformed and rejected. Abstentions are excluded from scored-case AUROC, with coverage reported. Non-fundus inputs must be rejected without a DR output; ungradable fundus inputs must produce retake without a DR probability. Missing outputs, parity mismatch, repeated-score nondeterminism beyond 1e-5, threshold mismatch, case-set mismatch, unverified image hashes, or missing evidence block acceptance.

Inspect every positive and negative failure for severity, image quality, acquisition domain and systematic patterns. No removing difficult cases, relabeling after predictions, prevalence manipulation, test-set threshold tuning, or claiming performance from software tests. A passing JSON report is necessary but not sufficient for clinical or demonstration readiness: provenance, adjudication, actual device workflows and external clinical review remain required.

## Confidence-interval assumptions and limits

Patient clusters are assumed independent for the patient bootstrap. Within-patient correlation is preserved; site-level correlation is not separately modeled. With few sites, device or patient clusters, the intervals may understate generalization uncertainty. Percentile bootstrap intervals are empirical cohort summaries, not formal guarantees. The stated patient/image/site/device floors are not a power calculation; determine sample size prospectively from expected prevalence, target precision, clustering, subgroup needs and desired confidence-bound power. Report a failed/unavailable interval as unestablished rather than passing based on point estimates.
