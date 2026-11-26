# OFFIRST Android inference repair and validation

## Status

**The application is not validated as a safe DR screening system.** This audit found that decodable non-retinal images enter the DR classifier, and saved v2.1 development results show substantial dataset-dependent false negatives. No validated fundus detector, image-quality model, independent labeled image set, or connected Android device is available in this checkout. Consequently, this work does not claim either reported failure is fully fixed and does not mark the APK ready for screening use.

## Repository safety and baseline

- Starting branch: `main` (the requested fix branch was not created because there is no validated detector or clinical evidence on which to base a safe application behavior change).
- Starting commit: `0399a2b70efb40347d70f4ff27d8b266381078b0`.
- Initial status: clean tracked tree, synchronized with `origin/main`; untracked `.vscode/`, `kd_run2/final_contrastive_distilled/`, and `kd_run2/hybrid_dr_improvement/` were preserved. In particular, the explicitly protected `.vscode/` and `kd_run2/hybrid_dr_improvement/` were not modified or removed.
- v2.1 and v1.1 ONNX assets, model metadata, and historical research/checkpoint directories were preserved.

## Findings by inference stage

| Stage | Audit result |
| --- | --- |
| Gallery/camera | `image_picker` returns cached image files; Dart reads those bytes for the v2.1 method-channel path. |
| Decode and orientation | Android accepts JPEG/PNG and decodes with OpenCV `IMREAD_COLOR`, which applies EXIF orientation. Decode/size failures are staged and surfaced as errors. |
| Fundus and quality validation | **Missing.** Any decodable JPEG/PNG can continue to preprocessing and DR inference. There is no modality/fundus classifier or validated gradability model in the app assets. |
| Preprocessing | The frozen v2.1 Android path uses BGR-to-RGB, the documented mask/crop/square-pad pipeline, sequential 640 then 512 `INTER_AREA` resize, ImageNet normalization and RGB planar NCHW FP32. |
| Tensor transfer | The existing fix checks OpenCV float read byte counts and reconstructs channel bytes as little-endian floats rather than taking an alignment-dependent `Float32List.view`. |
| Model and output | v2.1 uses `fundus_image` FP32 `[1,3,512,512]` and `dr_probability` FP32 `[1,1]`; probability is range-checked. The packaged ONNX graph contains the Platt calibration. |
| Threshold | v2.1 retains threshold `0.3675977844516245`; positive is `probability >= threshold`. No Kotlin calibration or probability inversion was found. |
| Flutter result/persistence/report | Prediction exceptions enter an unresolved failure queue and prevent saving until retry or removal. Successful records retain model version, exact probability and threshold; the report includes model, threshold and screening notice. Below-threshold text now explicitly states that DR is not ruled out. |

### Non-fundus acceptance root cause

The application has only format, size, and decode checks before running the DR classifier. It never tests whether the decoded image is a retinal fundus photograph. The DR classifier output cannot validate modality. The repository's `tools/quality_guard_poc` is a research/demo prototype: its README says its training and export were not run, and its image heuristics have unvalidated starting thresholds. No trained quality/fundus ONNX or TFLite asset, EyeQ dataset, or labeled validation photographs were found in the app or tracked model assets. Using those rules as an acceptance gate would not satisfy the requested evidence standard.

**Outcome:** non-fundus acceptance remains unresolved. A dependable rejection/retake gate needs an approved, appropriately licensed labeled fundus/quality dataset, a validated detector with defined reject/retake behavior, and offline Android parity tests. No training was started.

### False-negative root cause and model performance

No supplied positive retinal image with a documented label was available for a controlled Python-versus-Android replay. Therefore the reported individual false negative cannot be attributed to preprocessing, model loading, classifier, calibration, thresholding, or Flutter handling from this checkout alone.

The strongest existing evidence points to a **model generalization limitation**: the v2.1 integration report records development results of IDRiD accuracy 79.17%, sensitivity 87.23%, specificity 64.00% (72 images), and RFMiD accuracy 81.16%, sensitivity 54.55%, specificity 86.21% (69 images). These are not independent clinical validation. Saved-feature analysis additionally documents grade-1 IDRiD sensitivity of 2/5 at the fixed cutoff, but those images are not available as reproducible local source-image fixtures. The classifier is not demonstrated to meet the requested 85% accuracy, 85% sensitivity and 80% specificity targets.

No change was made to the v2.1 model, calibration, cutoff, encoder, projection, original KD768 checkpoint, or v1.1 rollback model. A new threshold would not fix the missing independent validation and could trade sensitivity against specificity.

## Model, preprocessing, and reproducible evidence

- Selected model: `deployment/offline_dr_screening/assets/offirst_v2_1_experimental.onnx`.
- Model SHA-256: `43a87f05df7adc6a231a679721e7f0b11b45a3fe6d1582aa4527ed738fa63b3f`.
- Input/output: `fundus_image`, float32 RGB NCHW `[1,3,512,512]` → `dr_probability`, float32 `[1,1]`.
- Calibration: Platt slope `0.5428962637583218` and intercept `-0.49103461599559345` are embedded in the graph; apply exactly once by consuming its probability directly.
- Decision: DR-positive at `>= 0.3675977844516245`.
- Frozen preprocessing: OpenCV decode with EXIF orientation; BGR-to-RGB; 512-long-edge proxy; max RGB channel `>15` mask; 5×5 morphological opening; bounding box if at least 100 pixels; centered square black padding; 640 then 512 `INTER_AREA`; ImageNet normalization; NCHW float32.
- Frozen checkpoint reference: KD Run 2 MobileNetV3-Large epoch 40, SHA-256 `8bc239d22ffc4e786820f2151c99380a754bef930e42edea65022e1246f2096d`; checkpoint source remains untouched.
- Existing Python/export evidence: 141-image saved development export reports maximum absolute probability difference `1.8341754449702208e-06`, mean difference `2.470365740902819e-07`, and no threshold disagreements. It validates the saved graph against its saved Python development outputs, not an independent labeled test set or Android execution.
- Existing Android instrumentation fixture manifest contains five **synthetic** images and reproducible tensor hashes/probabilities. It does not establish clinical accuracy and was not run on a device in this audit.

## Changes in this repair pass

- `deployment/offline_dr_screening/lib/screening_page.dart`: updated the global screening notice to state that a below-threshold result does not rule out DR and that qualified assessment is appropriate regardless of score; below-threshold image status now says “DR not ruled out”.
- `OFFIRST_ANDROID_INFERENCE_REPAIR.md`: added this audit record, including limitations and remaining validation work.

## Tests and build evidence

This pass did not run formatting, Flutter analysis/tests, Android instrumentation compilation, or a fresh APK build. The pre-existing `ANDROID_INFERENCE_P0_FIX.md` records earlier successful host Flutter checks, synthetic host ONNX replay, and Android instrumentation **compilation** for the earlier inference fixes; those records are not new verification of this pass. The repository contains release APKs from an earlier build, but this pass did not rebuild or independently validate them as its output.

- Python/Android parity on labeled retinal images: **not measured**.
- Android OpenCV execution and device workflow: **not tested**; `adb devices -l` reported no connected devices.
- Internal performance gates: **not met / not established**. Existing development evaluations miss accuracy and specificity, and RFMiD sensitivity is well below target.
- APK package/signing/ABI/hash verification for a fresh artifact: **not performed in this pass**. Earlier candidate evidence is recorded in `ANDROID_INFERENCE_P0_FIX.md` and is not presented as a new build.

## Remaining work before claiming resolution

1. Provide authorized, labeled and de-identified real fundus images with documented DR labels and an agreed independent split. Include non-fundus images and gradability labels for the input gate.
2. Select or develop a validated offline fundus detector and quality model only after dataset/license review and explicit training approval; validate its accept/reject/retake thresholds separately from DR classification.
3. Replay the same saved preprocessed tensors through pinned Python ONNX Runtime and Android ONNX Runtime on a device; capture tensor hashes, probabilities, threshold, model hash and image IDs without personal information.
4. Investigate the v2.1 classifier against an independent, patient-separated labeled cohort. Do not change threshold or weights based only on the reported failure examples.
5. Add real gradability/modality and labeled-image regression fixtures, then run Flutter tests, analyzer, Android instrumentation, fresh APK manifest/signature/model checks, and physical-device offline/gallery/camera/persistence/PDF acceptance.

## Installation

No new installation artifact is endorsed by this audit. Existing experimental APK candidates and their SHA-256 manifest are under `deployment/offline_dr_screening/build/app/outputs/flutter-apk/`; see `ANDROID_INFERENCE_P0_FIX.md` for their recorded package, version, ABI, signing certificate, hashes, and cautious device-install procedure. Install only as a research prototype after reviewing those limitations; there is no installation instruction that turns the current classifier into a clinically validated screening system.
