<div align="center">
  <h1>OFFIRST</h1>
  <h2>Offline AI-Powered Retinal Disease Screening</h2>
  <h3>Learn More From Less Data</h3>
  <p><strong>Team OFFIRST | ZEISS BeyondCode 2026</strong></p>
  <p>
    <img alt="Source version 1.0.5+2005" src="https://img.shields.io/badge/source-1.0.5%2B2005-blue">
    <img alt="Android application" src="https://img.shields.io/badge/platform-Android-green">
    <img alt="Research prototype" src="https://img.shields.io/badge/status-research%20prototype-orange">
  </p>
</div>

OFFIRST is an offline-first Android research prototype for retinal image analysis. It combines a frozen on-device ONNX model with a four-class diabetic retinopathy (DR) severity probe, exploratory multi-disease probes, local patient and visit records, and PDF reports.

The research question is whether useful image representations can be learned from fewer disease-labeled examples and then used in a compact mobile workflow. The system demonstrates this engineering direction; it does not establish clinical effectiveness.

> **Research use only.** OFFIRST is not a certified medical device and has not been clinically validated. Do not use it to diagnose or rule out disease, determine referrals, or guide treatment. A score below a threshold does not rule out DR.

**Current source:** version 1.0.5+2005, package <code>org.offirst.offline_dr_screening</code>, based on application source commit <code>212cf1f012ba862cc05315b353960a9549205e4c</code>. The updated v1.0.5 source is not represented by a verified public APK release.

## Contents

- [Project and research question](#project-and-research-question)
- [Architecture](#architecture)
- [BYOL and knowledge distillation](#byol-and-knowledge-distillation)
- [Current models and outputs](#current-models-and-outputs)
- [Image preprocessing and inference](#image-preprocessing-and-inference)
- [Application workflows](#application-workflows)
- [Research results and evaluation limits](#research-results-and-evaluation-limits)
- [Technology stack](#technology-stack)
- [Android releases and verification](#android-releases-and-verification)
- [Developer setup](#developer-setup)
- [Privacy, licensing, and responsible use](#privacy-licensing-and-responsible-use)
- [Repository map](#repository-map)

## Project and research question

Retinal photographs can vary with camera hardware, illumination, focus, field of view, and capture technique. Building a useful screening workflow is difficult when expert-labeled images are limited, acquisition conditions vary, and connectivity or cloud inference cannot be assumed.

OFFIRST investigates a research path that separates representation learning from downstream classification:

1. Learn visual features from image views without requiring a DR label for every pretraining image.
2. Transfer useful teacher features to a smaller student encoder intended for mobile execution.
3. Attach lightweight downstream heads for binary DR classification, severity categories, and exploratory disease labels.
4. Package inference and visit management in an Android application that can operate locally.

“Learn More From Less Data” describes the research motivation. It is not a claim that the current model achieves a particular sample-efficiency or clinical-performance target.

The project includes three different kinds of output:

- **Binary DR model output:** a sigmoid score from the current ONNX graph compared with its configured threshold.
- **Four-class severity probe:** a separate softmax head over the encoder embedding, with Healthy, Mild, Moderate, and Severe classes.
- **Exploratory multi-disease outputs:** a separate JSON probe bundle. These outputs are not clinical diagnoses and are not independently validated.

## Architecture

### Research and model-development pipeline

The documented KD Run 2 lineage uses BYOL for teacher and student representation learning, then distills teacher features into a mobile student. The binary and JSON heads are downstream components. Separate BYOL experiments, explicit Masked InfoNCE work, lesion segmentation, and quality-guard prototypes are research projects; they are not part of the current Android inference graph.

~~~mermaid
flowchart LR
    A[Retinal image views] --> B[BYOL ConvNeXt-Tiny teacher]
    A --> C[BYOL MobileNetV3-Large student initialization]
    B --> D[Knowledge distillation: KD Run 2]
    C --> D
    D --> E[MobileNetV3-Large student encoder]
    E --> F[960-to-768 projection]
    F --> G[L2-normalized 768-dimensional embedding]
    G --> H[Binary DR head exported in ONNX]
    G --> I[Four-class severity JSON probe]
    G --> J[Exploratory multi-disease JSON probes]
    K[Separate Masked InfoNCE experiments] -. not deployed .-> L[Research-only results]
~~~

**Documented lineage:** ConvNeXt-Tiny BYOL teacher, epoch 20; MobileNetV3-Large student initialized from direct BYOL, epoch 40; KD Run 2 student, epoch 40; 960-to-768 projection. The model manifest records the KD checkpoint digest as <code>8bc239d22ffc4e786820f2151c99380a754bef930e42edea65022e1246f2096d</code>. The checkpoint is available as an asset in the [KD Run 2 GitHub release](https://github.com/Mayank-gitspace/OFFIRST/releases/tag/kd-run2). The application packages the ONNX export, not a separately loaded teacher checkpoint.

### Deployed on-device inference

The v2.1 path performs preprocessing in Kotlin/OpenCV and inference with the packaged ONNX Runtime model. Its normalized embedding feeds the Dart JSON probes. The binary output and the severity/multi-disease probe outputs are distinct results.

~~~mermaid
flowchart LR
    A[Gallery or camera JPEG/PNG] --> B[Native decode and orientation]
    B --> C[Heuristic retinal-field gate]
    C --> D[Crop or retain image field, square pad]
    D --> E[Resize and RGB ImageNet normalization]
    E --> F[Float32 NCHW tensor: 1 x 3 x 512 x 512]
    F --> G[ONNX Runtime: encoder plus binary DR head]
    G --> H[dr_probability]
    G --> I[Normalized 768-dimensional embedding]
    H --> J[Compare with binary threshold]
    I --> K[Four-class severity probe]
    I --> L[Exploratory disease probes]
    H --> M[Per-image local record]
    K --> M
    L --> M
    M --> N[Visit history and local PDF actions]
~~~

Source references: [screening service](deployment/offline_dr_screening/lib/screening_service.dart), [ONNX contract](deployment/offline_dr_screening/lib/model_contract.dart), [native preprocessor](deployment/offline_dr_screening/android/app/src/main/kotlin/org/offirst/offline_dr_screening/V21Preprocessor.kt), and [probe implementation](deployment/offline_dr_screening/lib/screening_probes.dart).

## BYOL and knowledge distillation

**BYOL (Bootstrap Your Own Latent)** is a self-supervised learning method. In broad terms, it trains an online network to predict a target-network representation of another augmented view of the same image. A stop-gradient target and a slowly updated target network are used instead of explicit negative examples. This lets the representation-learning stage use images without a disease label for each example; it does not remove the need for valid labels in downstream DR evaluation.

The research records describe a larger ConvNeXt-Tiny BYOL teacher and a MobileNetV3-Large student initialized with direct BYOL. Knowledge distillation then trains the smaller student to match teacher representations on corresponding views. The motivation is to transfer useful features into an encoder that is more practical to run on mobile hardware. It does not, by itself, establish that the resulting features are clinically adequate.

The student representation is projected from 960 features to 768, then L2-normalized. The current ONNX model exposes that 768-dimensional embedding along with its binary score. The 4-class and multi-disease heads are separate downstream probes evaluated in Dart.

**Masked InfoNCE is not part of the deployed KD Run 2 model.** The repository contains separate contrastive research material, including an incomplete notebook; it must not be presented as a component of this app’s active inference pipeline.

## Current models and outputs

### v2.1 experimental ONNX binary model

| Property | Current source value |
| --- | --- |
| ONNX asset | [offirst_v2_1_experimental.onnx](deployment/offline_dr_screening/assets/offirst_v2_1_experimental.onnx) |
| SHA-256 | <code>b6bcb1182db7717c01ef24181d79a44be7d1a75ddefe8567cfec2fffb9e5c59b</code> |
| Manifest | [model_manifest.json](deployment/offline_dr_screening/assets/model_v2_1_evidence/model_manifest.json) |
| Decision configuration | [prediction_config.json](deployment/offline_dr_screening/assets/model_v2_1_evidence/prediction_config.json) |
| Input | <code>fundus_image</code>, float32, <code>[batch, 3, 512, 512]</code>, RGB/NCHW |
| Outputs | <code>dr_probability</code>, float32 <code>[batch, 1]</code>; normalized <code>embedding</code>, float32 <code>[batch, 768]</code> |
| Binary head | L2 normalization, linear 768-to-1 layer, sigmoid |
| Calibration | The current configuration records no calibration step; the scaler is folded into the linear weights |
| Threshold | <code>0.524373471736908</code>; positive when <code>dr_probability &gt;= threshold</code> |
| Intended evidence status | Development only; the manifest says it is not eligible for external validation |

This is the current source model and configuration. It is **not** the older 16-unit MLP with Platt calibration. The current manifest describes a changed linear head and a different threshold. The head asset’s name references RFMiD, ODIR, and APTOS, but the current manifest does not provide a fully auditable, patient-grouped training manifest or independent test results. The threshold is recorded as a Youden-CV threshold for development use, not as a validated clinical operating point.

The model contract and threshold are defined in [screening_service.dart](deployment/offline_dr_screening/lib/screening_service.dart). Model assets and preprocessing metadata are in [assets/model_v2_1_evidence](deployment/offline_dr_screening/assets/model_v2_1_evidence/).

### Four-class DR severity probe

The [APTOS severity probe](deployment/offline_dr_screening/assets/heads/kd768_lp_dr_severity4_aptos.json) is a separate linear softmax head over the normalized 768-dimensional embedding. It selects the largest of four class scores:

- **Healthy**
- **Mild**
- **Moderate**
- **Severe** — the asset maps this class to ICDR grade 3 severe NPDR and grade 4 proliferative DR

The asset metadata records APTOS-only training with 2,929 examples. Its SHA-256 is <code>b7f64ce49be8a11e2e9abebbdf8ea2b73fe3760b12b413c16df85b80afa2e388</code>. The displayed softmax confidence is not demonstrated to be calibrated clinical risk. The app’s default v2.1 result view uses this probe; it is not the binary ONNX output.

### Fifteen named exploratory multi-disease entries

The [multi-disease probe bundle](deployment/offline_dr_screening/assets/heads/kd768_lp_multidisease_rfmid_odir.json) contains 15 named label entries and has SHA-256 <code>2eaf44e6b3343a5821c6f4988d149766e9c18be10d69a00a609424c66157a0cc</code>:

| Code | Label |
| --- | --- |
| Disease_Risk | Any Disease Risk |
| DR | Diabetic Retinopathy |
| ARMD | Age-Related Macular Degeneration |
| MYA | Pathological Myopia |
| Glaucoma | Glaucoma |
| Cataract | Cataract |
| Hypertension | Hypertensive Retinopathy |
| MH | Media Haze |
| DN | Drusen |
| BRVO | Branch Retinal Vein Occlusion |
| TSLN | Tessellation (Tigroid Fundus) |
| LS | Laser Scars |
| ODC | Optic Disc Cupping |
| ODP | Optic Disc Pallor |
| ODE | Optic Disc Edema |

These are exploratory outputs, **not clinically validated diagnoses**. In app logic, Any Disease Risk is used as a summary, the DR entry is derived from the four-class severity output, and other findings use their probe scores and display thresholds. Labels include keyword-derived ODIR findings; this is not a substitute for adjudicated disease-specific evaluation. The head bundle’s own notes warn that some thresholds may create false alarms.

### Legacy v1.1 model

The v1.1 ONNX file remains packaged as a deliberate rollback path: [offirst_dr_model.onnx](deployment/offline_dr_screening/assets/offirst_dr_model.onnx), SHA-256 <code>b8fc10191f69faef16f1dc92d08bf9542ee7e26f885767942e28292c5201dd90</code>, historical cutoff <code>0.48</code>. It is a separate older model and does not use the current v2.1 embedding/probe path by default.

## Image preprocessing and inference

The v2.1 Android path is implemented in [V21Preprocessor.kt](deployment/offline_dr_screening/android/app/src/main/kotlin/org/offirst/offline_dr_screening/V21Preprocessor.kt). It accepts JPEG and PNG inputs subject to encoded-size and decoded-image limits, decodes through OpenCV, converts BGR to RGB, and uses OpenCV’s default EXIF orientation behavior.

The model preprocessing sequence is:

1. Resize the long edge to a 512-pixel proxy to construct a foreground mask from the maximum RGB channel greater than 15.
2. Apply a 5×5 morphological opening; when at least 100 mask pixels remain, crop to the mask bounding box.
3. Center the crop on a square black canvas, preserving the full crop with padding.
4. Resize through 640×640 and then 512×512 using OpenCV <code>INTER_AREA</code>.
5. Normalize channels using ImageNet means <code>[0.485, 0.456, 0.406]</code> and standard deviations <code>[0.229, 0.224, 0.225]</code>, then pack planar float32 RGB in NCHW layout.

The tensor contract is <code>[1, 3, 512, 512]</code> for the app’s single-image inference call. ONNX Runtime runs locally; no server inference endpoint is used.

### Input gate and quality limits

Before inference, Kotlin also runs <code>requireFundusField</code>, a geometric and intensity heuristic. It checks corner darkness, foreground area, contrast, approximate centering, and connected-component dominance. This gate is not a trained fundus detector: it may accept an ordinary photograph or reject a valid retinal image. It has not been established as a reliable non-fundus classifier.

No validated image-quality or gradability model is packaged. Blur, focus, overexposure, underexposure, and clinical gradability are not assessed by a trained quality component. The PDF therefore labels image quality “Not assessed.” The [quality-guard prototype](tools/quality_guard_poc/) is research code, not a validated deployed model.

Unsupported/corrupt files and inference failures produce errors rather than a negative prediction. Failed images remain unscreened and must be retried or removed before the visit can be saved. This is an application error-handling behavior, not proof of screening accuracy.

## Application workflows

| Workflow | Implemented in source | Verification boundary |
| --- | --- | --- |
| Patient registration, lookup, search, editing, and deletion | Local patient store and Flutter screens | Host test code exists; physical-device workflow has not been established here |
| Visit creation and visit history | Local visit records associated with a patient | No patient-level model aggregation |
| OD and OS selection | Eye-specific capture and result association | No result is generated for an absent eye |
| Multiple images per eye | Gallery picker and per-image processing | Images are stored and scored independently |
| Single-eye or bilateral visit | Visit scope reflects images supplied for one or both eyes | Not evidence that the two eyes are clinically complete |
| Unavailable eye | Optional status for the selected eye | Does not infer disability or create a prediction |
| Camera and gallery | Android image picker integration | Picker availability can depend on Android provider and permissions |
| Local inference | OpenCV preprocessing and ONNX Runtime | Device acceptance and offline operation have not been demonstrated for this source snapshot |
| Severity and disease views | Separate JSON probes over the ONNX embedding | Exploratory; not clinically validated |
| Expert directory | Local user-entered expert records and patient associations | Entries are not verified and association does not mean review |
| Error and retry handling | Failed images are kept without scores; retry/remove is required before save | Software behavior only; no failure can be treated as a validated negative |
| PDF report | Local two-page PDF save, view, and export/share actions | See report limitations below |

The central workflow is in [screening_page.dart](deployment/offline_dr_screening/lib/screening_page.dart), persistence in [patient_store.dart](deployment/offline_dr_screening/lib/patient_store.dart), and visit/report orchestration in [visit_page.dart](deployment/offline_dr_screening/lib/visit_page.dart).

### PDF reports

The Android PDF generator is [ReportPdf.kt](deployment/offline_dr_screening/android/app/src/main/kotlin/org/offirst/offline_dr_screening/ReportPdf.kt). It generates a two-page visit report with patient information, eye-wise severity output, exploratory findings, and image panels. Exporting or sharing a PDF is an explicit user action.

Current report limitations:

- It does not report the stored binary ONNX score, threshold, or model version.
- It does not explicitly include visit scope or the recorded unavailable-eye status.
- Image quality is marked “Not assessed.”
- Some patient fields may display placeholders.
- An existing version-6 report may be returned rather than regenerated after related visit or patient information changes.
- Referral/follow-up wording is driven by the exploratory severity output and is not clinically validated guidance.

## Research results and evaluation limits

The following are historical assessments of earlier v1.1 and v2.1 binary classifiers. **They do not evaluate the changed current v2.1 ONNX head or the JSON probes.** The repository reports prior dataset exposure and does not establish patient independence for these assessments. IDRiD and RFMiD use different label protocols.

| Historical classifier | Dataset | Images | Accuracy | Sensitivity | Specificity | AUROC | Historical threshold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1.1 | IDRiD | 72 | 70.83% | 57.45% | 96.00% | 0.846 | 0.48 |
| Prior v2.1 | IDRiD | 72 | 79.17% | 87.23% | 64.00% | 0.874 | 0.3675977844516245 |
| v1.1 | RFMiD | 69 | 86.96% | 100.00% | 84.48% | 0.967 | 0.48 |
| Prior v2.1 | RFMiD | 69 | 81.16% | 54.55% | 86.21% | 0.875 | 0.3675977844516245 |

The prior v2.1 confusion matrices reported in the project are IDRiD (TN, FP, FN, TP) = (16, 9, 6, 41) and RFMiD = (50, 8, 5, 6). Historical v1.1 confusion counts in the recovery records were derived from rounded aggregate metrics. Per-image predictions, labels, image hashes, and a verified patient-grouped partition were not recovered for these assessments.

The historical v2.1 results did not meet the combined project gates of 85% accuracy, 85% sensitivity, and 80% specificity on either dataset. The v1.1 RFMiD point estimates meet those numerical thresholds on that small reported assessment, but the IDRiD sensitivity is 57.45%, patient independence is unverified, and the result is not independent acceptance evidence. **Performance of the current changed model is unverified.** No current-model sensitivity, specificity, accuracy, AUROC, or clinical acceptance claim is made here.

### Representation-learning probes (not app-classifier metrics)

| Evaluation | Direct BYOL MobileNetV3 | KD Run 2 MobileNetV3 |
| --- | ---: | ---: |
| RFMiD linear-probe AUROC | 0.917 | 0.932 |
| RFMiD 10-shot AUROC | 0.640 | 0.728 |
| ARMD 10-shot AUROC | 0.789 | 0.949 |

These are representation-learning evaluations with exposure and split limitations. They are not accuracy, sensitivity, or specificity for the current app classifier.

See [research results](docs/RESEARCH_RESULTS.md), the [prediction-recovery report](OFFIRST_PREDICTION_RECOVERY.md), and the [data lineage and exposure matrix](OFFIRST_DATA_LINEAGE_AND_EXPOSURE_MATRIX.csv) for dataset and evaluation caveats.

## Technology stack

| Technology | Role in OFFIRST |
| --- | --- |
| Flutter and Dart | Android screens, navigation, patient/visit workflows, inference orchestration, probe scoring, persistence calls, and report actions |
| Kotlin and Android SDK | Native image preprocessing, Android method channels, PDF generation, and APK packaging |
| OpenCV 4.12 | Local image decoding, color conversion, crop/pad processing, resizing, and tensor preparation for v2.1 |
| ONNX Runtime | Local execution of the frozen FP32 ONNX model |
| Python and PyTorch | Encoder, distillation, probe, and model-development research |
| scikit-learn and NumPy | Research data handling, lightweight head fitting, and analysis |
| Kaggle | Documented environment for some model-development runs; datasets and outputs remain subject to their own terms |
| App-private files and JSON | Local patient, expert, visit, image, and report persistence |

The Android application has no cloud inference endpoint. Its main release manifest does not request Android’s INTERNET permission; Flutter debug/profile manifests add it for development tooling. Android image providers and PDF viewers/sharing targets are separate system components.

## Android releases and verification

### Current source

The current application source is version <code>1.0.5+2005</code> and defaults to the v2.1 experimental model. The current model file hash is <code>b6bcb1182db7717c01ef24181d79a44be7d1a75ddefe8567cfec2fffb9e5c59b</code>. The configured binary threshold is <code>0.524373471736908</code>.

Local v1.0.5 APK artifacts have been inspected for embedded model/probe assets, package version, ABI, and signer. Their reproducible relationship to the exact source commit has not been established, and no physical-device acceptance has been recorded. They are **not presented as verified downloads**.

### Older published releases

- [Archived v2.1 release](https://github.com/Mayank-gitspace/OFFIRST/releases/tag/offirst-v2.1): package v1.0.3/code 2003 and APK SHA-256 <code>c165bc145ce2fdf6115585594aa4048e728abd0422576f485d36be587c6c77e2</code>. It contains the earlier model with the 16-unit MLP, embedded Platt calibration, ONNX SHA-256 <code>43a87f05df7adc6a231a679721e7f0b11b45a3fe6d1582aa4527ed738fa63b3f</code>, and threshold <code>0.3675977844516245</code>. It is not the current v1.0.5 source model.
- [Archived v1.1 rollback release](https://github.com/Mayank-gitspace/OFFIRST/releases/tag/offirst-android-prototype-v1.1): older binary model and APKs, retained separately from the current v2.1 path.

No new APK or release is published by this documentation. Do not use an archived APK as though it were a build of the current source.

## Developer setup

### Prerequisites

The project uses the documented baseline of Flutter 3.41.9, Dart 3.11.5, JDK 17, and the Android SDK. The Android project uses Gradle 8.14 and declares a minimum Android API level of 24. Install Flutter and Android tooling using their official setup guides, and connect an Android device or configure an emulator for device tests.

The package dependencies and model assets are declared in [pubspec.yaml](deployment/offline_dr_screening/pubspec.yaml). No model download is required for the packaged inference assets.

### Clone and install dependencies

~~~powershell
git clone https://github.com/Mayank-gitspace/OFFIRST.git
cd OFFIRST/deployment/offline_dr_screening
flutter pub get
~~~

### Analyze and test

~~~powershell
flutter analyze
flutter test

# Optional: run the app test suite with the legacy v1.1 model selection.
flutter test --dart-define=OFFIRST_EXPERIMENTAL_V21=false

# Compile the Android instrumentation-test source.
cd android
.\gradlew.bat :app:compileDebugAndroidTestKotlin

# Run Android instrumentation tests on a connected device or emulator.
.\gradlew.bat :app:connectedDebugAndroidTest
~~~

A test command passing is not evidence of clinical model performance. Physical-device screening, offline operation, patient persistence, and PDF export require separate device verification.

### Build and install

From <code>deployment/offline_dr_screening</code>:

~~~powershell
flutter build apk --release --dart-define=OFFIRST_EXPERIMENTAL_V21=true
flutter install
~~~

The v2.1 model is the default; the explicit define above makes the selection clear. To build the legacy rollback variant deliberately:

~~~powershell
flutter build apk --release --dart-define=OFFIRST_EXPERIMENTAL_V21=false
~~~

The existing [candidate build script](deployment/offline_dr_screening/scripts/build_v2_1_final_candidate.ps1) still contains v1.0.4 checks, so it is not a verified v1.0.5 build path until updated. Release signing is currently configured to use the Android debug signing configuration; this is suitable only for a research prototype build, not a production app-store release.

### Important paths

| Path | Contents |
| --- | --- |
| [deployment/offline_dr_screening/lib](deployment/offline_dr_screening/lib) | Flutter app screens, persistence, inference service, and probe scoring |
| [deployment/offline_dr_screening/android/app/src/main/kotlin](deployment/offline_dr_screening/android/app/src/main/kotlin) | Kotlin preprocessing, method channels, and PDF generator |
| [deployment/offline_dr_screening/assets](deployment/offline_dr_screening/assets) | ONNX models, JSON probes, and model evidence manifests |
| [deployment/offline_dr_screening/test](deployment/offline_dr_screening/test) | Flutter unit, widget, persistence, and report test sources |
| [deployment/offline_dr_screening/android/app/src/androidTest](deployment/offline_dr_screening/android/app/src/androidTest) | Android instrumentation-test source |
| [encoder_byol](encoder_byol) | BYOL research experiments |
| [encoder_mnv3_ssl](encoder_mnv3_ssl) | MobileNetV3 self-supervised experiments |
| [kd_run2](kd_run2) | Teacher/student distillation experiments and evaluation artifacts |
| [tools/quality_guard_poc](tools/quality_guard_poc) | Unvalidated quality-guard research prototype |

## Privacy, licensing, and responsible use

- Inference runs locally using packaged assets; patient records and images are stored in app-private files.
- The application does not add account authentication, cloud sync, or application-level encryption. Android backup is disabled.
- Exported or shared PDFs may contain patient details and retinal photographs. Use synthetic patient profiles and de-identified images during demonstrations.
- The heuristic retinal-field gate is not a trained fundus detector, and the app does not provide validated image-gradability assessment.
- The current changed binary model and probes have no established independent clinical evaluation. Project performance gates are not demonstrated for the current model.
- Project records cite CC BY 4.0 for RFMiD and IDRiD source materials, but do not authenticate every local source version or derivative. APTOS is subject to its competition/source terms. ODIR access is registration-controlled, and the repository review did not establish a dataset-specific redistribution license.
- Redistribution permissions for the current model, JSON probes, cached features, and tracked dataset-derived images have not been fully established. The repository has no root LICENSE file.

Dataset citations and the exact data-lineage limitations are recorded in the [data lineage and exposure matrix](OFFIRST_DATA_LINEAGE_AND_EXPOSURE_MATRIX.csv). See also [architecture](docs/ARCHITECTURE.md), [known limitations](docs/KNOWN_LIMITATIONS.md), and [installation notes](docs/INSTALLATION.md).

## Repository map

| Directory | Purpose |
| --- | --- |
| [deployment/offline_dr_screening](deployment/offline_dr_screening) | Flutter/Android application, packaged models, probes, Kotlin pipeline, and tests |
| [docs](docs) | Architecture, setup, model, evaluation, and limitation documentation |
| [encoder_baseline](encoder_baseline) | Supervised encoder baselines and representation probes |
| [encoder_byol](encoder_byol) | ConvNeXt BYOL research |
| [encoder_mnv3_ssl](encoder_mnv3_ssl) | MobileNetV3 self-supervised research |
| [kd_run2](kd_run2) | Knowledge-distillation experiments |
| [kd_imaging](kd_imaging) | Imaging and representation analysis |
| [lesion_segmenter](lesion_segmenter) | Separate lesion-segmentation research, not shipped in the app |
| [tools](tools) | Evaluation and research utilities |

---

**Project:** Team OFFIRST · ZEISS BeyondCode 2026
**Status:** Offline Android research prototype; current clinical performance and device acceptance are not established.
