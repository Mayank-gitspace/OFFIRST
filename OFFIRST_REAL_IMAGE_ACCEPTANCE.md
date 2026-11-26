# OFFIRST real-image acceptance protocol

## Current status

**Acceptance is blocked; OFFIRST is not cleared for demonstration.** The repository contains no authorized source image corpus suitable for this suite and no joined per-image real-image prediction manifest. Existing application ONNX assets are DR classifiers, not input validators. The device record shows no connected Android device. The local five-image regression fixture is synthetic and is excluded from this acceptance suite.

Prior aggregate development results remain a warning, not acceptance evidence: v2.1 RFMiD sensitivity is 54.55%; the 85%/85%/80% targets are not established. No images were sourced or copied into the repository in this phase; no model, threshold, or training code was changed, and no training was run.

## Reproducible harness

- `tools/real_image_acceptance/manifest_template.csv`: one row per image, runtime platform, and repeat. Use pseudonymous sample and patient-group IDs. Keep source images in an access-controlled directory outside Git and store only a relative path in the manifest. `quality_issue` distinguishes blur, dark, overexposed, poor-field, other, and non-applicable cases.
- `tools/real_image_acceptance/evaluate_acceptance.py`: Python standard-library evaluator for image/tensor hashes, locked test threshold, DR confusion matrix, accuracy/sensitivity/specificity with Wilson 95% intervals, AUROC with patient-group bootstrap interval, severity/device/site/eye/dataset subgroups, input validation, tensor-matched Python/Android probability parity, repeated inference, OD/OS representation, and airplane-mode capture.

The evaluator does not run a model, tune thresholds, validate ground truth, or certify clinical performance. It returns exit code 3 when evidence/manifest is incomplete, 2 when an encoded acceptance gate fails, and 0 only when every encoded gate passes. The report still needs human review of data provenance, adjudication, individual cases, device logs, and study design.

After filling the manifest from real authorized images and recording both Python and Android runs, execute from the repository root. The command below uses the frozen v2.1 model ID and cutoff; it is illustrative and has not been run on real cases.

```powershell
python tools/real_image_acceptance/evaluate_acceptance.py `
  tools/real_image_acceptance/manifest.csv `
  --image-root D:\OFFIRST-RESTRICTED\images `
  --check-image-hashes `
  --model-version offirst-v2.1-experimental-43a87f05 `
  --locked-threshold 0.3675977844516245 `
  --bootstrap-reps 2000 `
  --seed 20261126 `
  --output D:\OFFIRST-RESTRICTED\acceptance-report.json
```

No manifest, real images, or report are currently supplied, so this command cannot yet produce a valid acceptance result. Keep source images and any re-identification key outside Git and reports.

## Cohort and partition plan

Before evaluating the final test partition, freeze a data dictionary, endpoint, model version, ONNX SHA-256, preprocessing version, threshold, and protocol revision. Record model/threshold selection on the designated validation partition. Then lock and hash the untouched test manifest. Do not inspect test predictions or use test errors to tune anything. Do not remove hard cases, change labels post hoc, alter class prevalence, or choose only a favorable device/site.

Use consecutive or prospectively specified representative eligible cases from the intended acquisition domain. DR status and severity must be adjudicated using a prespecified reference standard. Keep both eyes and repeated exams from one patient in the same split. Bootstrap confidence intervals by pseudonymous patient group, not individual image. Preserve natural prevalence for accuracy/PPV/NPV; also report sensitivity and specificity by outcome. Document exclusions before model scoring and retain acquisition failures in an availability/coverage flow diagram.

The locked test must contain:

- Confirmed DR-negative and DR-positive gradable fundus images.
- Multiple documented DR severity levels, including no-DR and positive severity categories.
- Non-fundus photos from realistic gallery/camera sources.
- Adjudicated ungradable fundus images covering blur, underexposure, overexposure/glare, poor field, and other capture faults.
- Representative acquisition devices/sites, including multiple devices where available, with each subgroup reported separately.
- Both OD and OS, repeated inference for each scored test image, and Android workflow evidence captured in airplane mode.

Modality, gradability/quality, DR status, and severity are separate labels. Non-fundus and ungradable images do not receive DR probabilities. Resolve reference-label disagreements before partition lock; never relabel based on model output.

## Per-image evidence

Record all manifest columns for every real image, platform, and repeat: pseudonymous sample ID, original image SHA-256, protected relative path, preprocessed tensor SHA-256, partition, dataset/site/device group, pseudonymous patient group, eye, DR severity and binary label, modality, gradability, quality issue, input-validation outcome, quality outcome, probability when applicable, final decision, model version, exact threshold, platform, network mode, case-review status, run ID, and repeat number. Keep timestamps/device build IDs and raw logs in a separate protected execution record. Exclude names, birth dates, patient numbers, and identifying paths.

For Python/Android probability parity, run both runtimes on byte-identical preprocessed FP32 tensors; record tensor SHA-256 and output probability. Compare preprocessing tensors separately. Repeat Android inference on the same device/tensor; the harness uses `1e-5` as the initial engineering tolerance. Establish any different tolerance from runtime evidence before locking the test protocol.

## Metrics and acceptance rules

Compute primary DR metrics on the untouched test partition at the locked operating point. A gradable reference image rejected, retaken, errored, or left unscored is conservatively counted as an end-to-end failure (positive → FN; negative → FP); it is not silently dropped from accuracy/sensitivity/specificity. AUROC is reported for gradable fundus cases that produced probabilities, with scored coverage shown and a complete coverage gate. Report the full confusion matrix and input-validation/quality outcomes.

Required: accuracy ≥85%, sensitivity ≥85%, specificity ≥80%, AUROC ≥0.85; every non-fundus case rejected without a DR classification; every ungradable fundus case receives retake; individual review of all errors; confidence intervals, severity/device/site/dataset/eye subgroup denominators; same-tensor Python/Android probability delta within the declared tolerance; repeated Android scores within tolerance for each scored test image; OD and OS workflows; actual Android gallery/camera, airplane-mode, persistence, and PDF evidence. The test cohort must include blur, dark, and overexposed ungradable examples when represented by the target use conditions.

The evaluator encodes metric, validation, retake, coverage, parity, repeatability, OD/OS, severity-diversity, and airplane-mode gates. Device workflow, case review, cohort validity, adjudication, external validity, and sample-size sufficiency require human review and physical-device evidence. A JSON `acceptance_pass` alone is not clinical clearance.

## Current outcome and blocker

The harness is implemented but has **no real-image cases to score**, so all performance and end-to-end gates remain unestablished. The next required inputs are an approved de-identified labeled corpus with patient/site/device groups and representative non-fundus/ungradable cases, plus a connected supported Android device. Until a locked untouched test run passes and device acceptance is recorded, do not declare OFFIRST ready for demonstration.
