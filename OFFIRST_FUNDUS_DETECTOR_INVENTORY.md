# OFFIRST fundus-detector and quality-model inventory

## Finding

No trained fundus-versus-non-fundus classifier or trained image-gradability/quality model suitable for app reuse was found in tracked model assets or application assets. The current app checks file format, size, and decode success, then runs DR inference. Its foreground mask is DR-model preprocessing, not a fundus detector. Do not use it, brightness, resolution, or image dimensions to claim modality validation.

## Reusable assets and limits

| Asset/location | What exists | Reuse assessment |
| --- | --- | --- |
| `deployment/offline_dr_screening/assets/offirst_dr_model.onnx` | Frozen v1.1 DR classifier | DR classification only; cannot validate fundus modality. |
| `deployment/offline_dr_screening/assets/offirst_v2_1_experimental.onnx` | Frozen v2.1 DR classifier | DR classification only; cannot validate fundus modality. |
| `deployment/offline_dr_screening/lib/model_preprocessing.dart`, `android/.../V21Preprocessor.kt`, `assets/model_v2_1_evidence/preprocessing.json` | Image decode/orientation, retinal-field crop heuristic, resize, normalization, NCHW packing | Reusable preprocessing implementation/evidence only. Foreground crop/mask is not a trained or validated fundus detector. |
| `tools/quality_guard_poc/quality_checks.py` | Hand-engineered blur, brightness, glare, centering, and optic-disc heuristics | Research prototype with stated starting thresholds. Cannot reject non-fundus reliably; not app-integrated or validated. |
| `tools/quality_guard_poc/train.py`, `export_int8.py`, `app.py` | Proposed MobileNetV3-Small EyeQ classifier training/export/demo code | README explicitly says training/export/app were not run. No weights or exported ONNX/TFLite are present. Not a trained reusable model. |
| `tools/quality_guard_poc/prepare_eyeq.py` | Script expecting EyeQ labels and EyePACS images on disk | No EyeQ/EyePACS image dataset or label CSV was found in the repository. |
| `tools/quality_guard_poc/degrade.py` | Synthetic capture-degradation generator for otherwise good fundus photos | Can supplement quality testing; synthetic degradations do not provide non-fundus examples or prove clinical gradability. |
| `lesion_segmenter/` | ConvNeXt-T U-Net lesion segmentation research, IDRiD segmentation masks, logs and evaluation summaries; checkpoint hashes documented in README | Lesion masks/localization, not fundus modality/quality classification. Large `.pth` checkpoints are referenced as release downloads and were not found locally. Must not repurpose as a detector without a separate validated design. |
| `kd_imaging/` | Image-resolution/lesion-visibility and representation analyses | Not a modality/gradability model. Numeric analyses include restricted/non-published frames; not a detector dataset. |
| `kd_run2/odir5k_eval/eye_labels_all.csv` | Cached ODIR eye labels and metadata for multi-disease representation evaluation | Labels include normal/disease tags and some quality keywords, but this is not a labeled fundus-vs-non-fundus/gradable dataset. Original images are absent; quality label reconstruction has four normal-label mismatches. Not suitable as a ready-made validated gate. |
| Synthetic v2.1 regression fixtures in ignored app build output | Five generated fundus-like images with tensors and predictions | Software regression only, no clinical or modality-validation value. |

The repository-wide model-file inventory found the two app DR ONNX files. Lesion-segmentation `.pth` weights are described by external release URLs but are not local trained detector assets. No fundus classifier `.onnx`/`.tflite` or quality classifier weights were found. Dataset image files for EyeQ, EyePACS, APTOS, IDRiD grading, and RFMiD are not included as reusable training corpora; the repository documents access/setup constraints.

## Recommended smallest defensible component

Add a separate compact fundus-quality classifier before DR inference, with three explicit outcomes: (1) non-fundus/reject, (2) fundus but ungradable/retake, (3) gradable fundus/continue. A small MobileNetV3-Small-class model is a reasonable deployment starting point, subject to measured latency, memory, calibration, and Android operator support. Keep it separate from the DR classifier and preserve its own version/hash/threshold metadata. It must fail closed on inference errors.

Training/evaluation data should include labeled EyeQ quality categories (good/usable/reject) plus a distinct, diverse non-fundus set covering anterior-segment/face photos, documents/screens, other body images, and blank/corrupt images. EyeQ is the codebase's identified route; acquire source data under its original terms and verify image/label mapping. Add site/camera/phone capture diversity and expert gradability labels for the intended acquisition path. Split by patient and source/site, deduplicate exact and near duplicates, and reserve an untouched external set. APTOS has DR labels but is not a substitute for image-quality labels or a non-fundus cohort; synthetic degradations are supplementary only.

Before training, obtain approval, confirm dataset licenses/access and intended device/camera domain, and preregister reject/retake metrics. Measure non-fundus false acceptance, gradable-image false rejection, ungradable sensitivity, subgroup/site performance, and Android/Python parity. No model training was started in this investigation.
