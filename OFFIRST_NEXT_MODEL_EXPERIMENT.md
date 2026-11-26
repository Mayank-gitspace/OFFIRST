# OFFIRST next model experiment — preregistration proposal

## Decision

Do not start encoder training. The immediate smallest model work is a **separate fundus/gradable-quality input-validation classifier**; in parallel, run the diagnostic parity and error-attribution experiment in `OFFIRST_DR_FAILURE_ANALYSIS.md`. No weights, thresholds, ONNX assets, or training code were changed in preparing this proposal.

## Track A: validate input before DR scoring

**Question:** Can a compact offline model distinguish non-fundus images from fundus photographs, and identify fundus images that should be retaken, across the intended acquisition devices?

**Minimum model:** a separate MobileNetV3-Small-class image classifier with 3 outcomes: non-fundus/reject, fundus but ungradable/retake, gradable fundus/continue. Keep DR inference out of this model. Benchmark a compact pretrained backbone before increasing model size; model size is selected on measured Android memory/latency and error tradeoffs, not name alone.

**Data prerequisites:**

- Acquire the EyeQ image/quality labels and matching images under approved terms; first verify label CSV schema and image mapping. The existing `quality_guard_poc` only contains preparation/training/export scripts, not training results or model weights.
- Build a separate, stratified non-fundus cohort representing anterior-segment/face, other body photographs, screens/documents, unrelated scenes, blank/corrupt inputs and camera/gallery edge cases. EyeQ quality labels alone should not be assumed to supply balanced negatives for every non-fundus class.
- Add representative intended-device fundus images with expert gradability labels; record camera/device/site and acquisition conditions. Preserve image IDs and patient/site grouping for splits without exporting personal metadata.
- Split by patient and acquisition site/device where possible; exact/near-deduplicate before splitting. Reserve a locked external site/device test set. Synthetic degradations may be supplemental only.

**Primary metrics:** non-fundus false-accept rate; gradable fundus false-reject rate; sensitivity for ungradable/retake images; subgroup/site/device metrics; calibration and coverage. Define acceptable margins with stakeholders before training. Any uncertainty or classifier error must fail closed to a clear retry/retake/manual-review path, never to a benign DR result.

**Deployment tests:** pinned Python vs Android preprocessing/output parity; model contract, range, offline loading; corrupted/unsupported input; gallery/camera image flow; memory and latency on supported ARM64 phone; no patient/image data in logs. Keep its model version/hash/decision thresholds separate from the DR metadata.

**Stop condition:** do not train until dataset access/licensing, target capture devices, expert labels, split plan, and success/failure margins are approved. No encoder training is part of this track.

## Track B: diagnose v2.1 DR misses without fitting

Restore the original RFMiD and IDRiD assessment manifests and exact saved v1.1/v2.1 per-image scores, then run matched Python references and (with a device) Android tensor/probability parity. This is evaluation-only. Preserve threshold `0.3675977844516245` for v2.1 and `0.48` for v1.1. Report PR/ROC, calibration, score histograms, FN/FP image IDs/hashes, DR grade, source/domain and acquisition quality. Audit exact/near duplicates and model-training exposure.

If Android parity fails, fix only the demonstrated integration defect and add a real-image parity regression fixture. If parity passes, investigate dataset/domain/label and classifier-head/calibration contributions using an independent, patient/site-held-out cohort. Only after that evidence should a minimal downstream head/calibration comparison be proposed. Do not tune on the current assessment images and do not retrain the encoder by default.

## Track C: independent DR evaluation gate

Before any new DR candidate, freeze a protocol and acquire an independent, clinically representative cohort with documented patient/site grouping, acquisition device, gradability, and agreed binary endpoint. The reported 85/85/80 accuracy/sensitivity/specificity gates apply at a preselected operating point on the untouched evaluation cohort; confidence intervals and per-site/subgroup results must be reported. Separate development/selection/calibration data from the untouched test set. A passing software test suite or development split does not satisfy this gate.

## Required outputs before an implementation decision

1. Approved data and access manifest with image hashes, labels, patient/site split IDs, and license/access notes.
2. Fundus classifier baseline report plus error review and threshold selection plan.
3. Matched v1.1/v2.1 per-image error/parity report on identical authorized images.
4. Android ARM64 latency, memory, session-load, and fixture parity results.
5. Go/no-go decision for the smallest downstream correction. Preserve all original model artifacts and rollback paths.
