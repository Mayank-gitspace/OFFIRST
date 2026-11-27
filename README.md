# OFFIRST

## Offline-first Retinal Screening Research Prototype

**Team OFFIRST · ZEISS BeyondCode 2026 · Source version 1.0.5**

**Project focus:** Explore whether self-supervised visual representations can support learning from fewer labeled retinal images, with inference and patient records kept on an Android phone.

OFFIRST is an experimental Android research prototype for exploring diabetic-retinopathy screening from retinal photographs. The Flutter app runs an ONNX model on-device, stores patient and visit records locally, and can create a PDF visit report. The project demonstrates a mobile workflow; it does not establish clinical effectiveness.

> **Research prototype only.** This project is not clinically validated or a certified medical device. Do not use its outputs to diagnose, rule out disease, make referrals, or guide treatment.

The current source snapshot reviewed here is commit <code>212cf1f012ba862cc05315b353960a9549205e4c</code>. The repository’s published APK releases are older than this source snapshot; see [Current source and archived APKs](#current-source-and-archived-apks).

## Problem

Reliable retinal screening can be difficult where specialist access, connectivity, and labeled training data are limited. OFFIRST explores compact self-supervised representations and a mobile inference workflow that can operate without a cloud inference service.

The current implementation demonstrates local image processing, per-eye visit management, and research-model outputs. It does not establish diagnostic accuracy, clinical gradability, or patient-level screening performance.

## Implemented capabilities

- Register, search, edit, and delete patient records.
- Create visits and review local visit history.
- Select a right eye (OD) or left eye (OS), then add images from the camera or gallery.
- Record one or more images for either eye; each image is evaluated independently.
- Optionally mark the selected eye unavailable for a visit. This is a user-entered status only; the app does not infer a reason or disability.
- Run the packaged ONNX model on-device and store its binary DR probability, threshold, model identifier, eye, and normalized embedding.
- Display a four-class DR severity probe by default, or switch to exploratory multi-disease probe outputs.
- Save, view, or export a two-page PDF report.
- Use a first-run guided walkthrough and stage-specific error messages for image access, decoding, preprocessing, model loading, and inference failures.

The app does not combine images into one patient-level prediction. It has no user accounts, cloud synchronization, live appointment booking, or clinician verification.

## Architecture and inference flow

~~~mermaid
flowchart TD
    A[Camera or gallery image] --> B[Decode and heuristic retinal-field check]
    B --> C[OpenCV crop, resize and normalize]
    C --> D[ONNX Runtime on Android]
    D --> E[Binary DR probability and 768-D embedding]
    E --> F[Store per-image result locally]
    E --> G[Four-class severity probe]
    E --> H[15 exploratory disease probes]
    F --> I[Patient visit history and PDF actions]
    G --> I
    H --> I
~~~

### Android inference

The current v2.1 experimental path uses Kotlin/OpenCV preprocessing and the Flutter ONNX Runtime package. The expected model input is RGB float32 NCHW, <code>[batch, 3, 512, 512]</code>. The app checks the model input/output names and tensor dimensions before inference.

The preprocessing path applies EXIF-aware decode, a heuristic retinal-field crop/pad, OpenCV area resizing through 640 × 640 and 512 × 512, and ImageNet channel normalization. Inputs over 32 MiB or 24 megapixels are rejected.

### Fundus input gate and image quality

The input gate is a hand-coded geometric and intensity heuristic. It uses image corners, foreground area, brightness/contrast, centering, and connected-component shape to reject some likely non-retinal inputs before inference. It is **not a trained or clinically validated fundus detector**. It can accept a non-fundus image or reject a retinal image.

The app does **not** assess blur, exposure, focus, or clinical gradability. An accepted image must not be described as gradable or suitable for diagnosis. The separate quality-guard scripts in <code>tools/quality_guard_poc/</code> are not an integrated, validated quality model.

## Model outputs

### Current binary DR model

The default v2.1 experimental ONNX asset is:

- File: [<code>offirst_v2_1_experimental.onnx</code>](deployment/offline_dr_screening/assets/offirst_v2_1_experimental.onnx)
- Manifest: [<code>model_manifest.json</code>](deployment/offline_dr_screening/assets/model_v2_1_evidence/model_manifest.json)
- Manifest-declared SHA-256: <code>b6bcb1182db7717c01ef24181d79a44be7d1a75ddefe8567cfec2fffb9e5c59b</code>
- Size recorded in the manifest: 14,836,101 bytes
- ONNX outputs: binary <code>dr_probability</code> and a normalized 768-dimensional <code>embedding</code>
- Binary operating threshold: <code>0.524373471736908</code>, positive at <code>probability &gt;= threshold</code>
- Binary head: linear 768-to-1 layer followed by sigmoid; the manifest says the scaler is folded into the weights and calibration is none.

The current model manifest marks this asset as **not eligible for external validation**. The digest above is what the tracked manifest declares; it was not recomputed from the binary ONNX payload during this review.

The Dart app stores the binary probability and its threshold with each current image record. In the current v2.1 interface, the default visible result is produced by the separate four-class severity probe below; the multi-disease option uses separate exploratory probes. Do not present either probe as the ONNX binary head’s validated performance.

### Four-class DR severity probe

[<code>kd768_lp_dr_severity4_aptos.json</code>](deployment/offline_dr_screening/assets/heads/kd768_lp_dr_severity4_aptos.json) contains a separate softmax linear probe over the ONNX embedding. It displays **Healthy**, **Mild**, **Moderate**, or **Severe** using argmax. Its metadata records an APTOS-only training set of 2,929 examples. The displayed class confidence is the probe’s softmax output; it is not evidence of calibrated clinical risk.

### Exploratory multi-disease probes

[<code>kd768_lp_multidisease_rfmid_odir.json</code>](deployment/offline_dr_screening/assets/heads/kd768_lp_multidisease_rfmid_odir.json) contains 15 separate probe heads evaluated in Dart from the same embedding. The app lists:

<code>Disease_Risk</code>, <code>DR</code>, <code>ARMD</code>, <code>MYA</code>, <code>Glaucoma</code>, <code>Cataract</code>, <code>Hypertension</code>, <code>MH</code>, <code>DN</code>, <code>BRVO</code>, <code>TSLN</code>, <code>LS</code>, <code>ODC</code>, <code>ODP</code>, and <code>ODE</code>.

These are exploratory RFMiD/ODIR-derived label probes. Their outputs are not diagnoses, and the repository does not provide independent clinical validation for them. ODIR-derived labels and access conditions require particular care; see [data and licensing](#data-attribution-and-licensing).

## Self-supervised learning, distillation, and contrastive research

The current deployed encoder is the KD Run 2 MobileNetV3-Large student with a 960-to-768 projection. Project research records describe the student as initialized from MobileNetV3 BYOL pretraining and distilled from a frozen ConvNeXt-Tiny BYOL teacher. KD Run 2 is a representation-learning stage; the current binary DR head and the JSON probe heads are downstream tasks.

BYOL is non-contrastive self-supervised learning: it learns from related augmented views without explicit negative examples. Knowledge distillation transfers representations from the teacher to the mobile student. The current KD Run 2 lineage is **not** an explicit Masked InfoNCE model.

Explicit Masked InfoNCE is a separate, incomplete research direction. The current <code>kd_run2/final_contrastive_distilled/</code> notebook is a preflight/input-discovery outline with no executed cells, training outputs, resulting checkpoint, DR head, or ONNX export. It is not part of the deployed model. Earlier v1.1 materials mention BYOL/InfoNCE, but the current repository does not provide a complete reproducible training record that would justify attributing that objective to the current app model.

See [KD Run 2](kd_run2/README.md), [research results](docs/RESEARCH_RESULTS.md), and the [contrastive notebook](kd_run2/final_contrastive_distilled/OFFIRST_final_contrastive_distilled.ipynb) for the separation between application inference and research experiments.

## Patient, eye, visit, and report workflow

A visit can contain multiple OD and OS images. The app shows each eye separately and retains each image’s own probability, model metadata, threshold, and embedding. A visit can be unilateral or bilateral; the app does not average images or produce a patient-level score.

If an eye is marked unavailable, the record stores that selection without a supplied image or model result. If image access, decoding, preprocessing, model loading, or inference fails, that image is left without a score; the UI requires failed images to be retried or removed before saving the visit.

Patient metadata, visits, expert contacts, original selected images, and generated reports are stored on the device. Expert entries are user-entered and unverified; an association does not mean that the expert reviewed the case.

### PDF report behavior and limits

The Android Kotlin report code creates a two-page report that can be saved in the app, opened with a PDF viewer, or exported through Android sharing. It includes patient fields, per-eye severity-probe output where available, exploratory findings, and image panels. Some fields display placeholders when no value was entered.

The current PDF does **not** report the stored binary ONNX probability, binary threshold, or model version. It also does not explicitly report the visit scope or an unavailable-eye status. Image quality is labeled **“Not assessed.”** Existing version-6 PDFs can be reused instead of regenerated, so a previously saved report may not reflect later changes to the visit or patient record.

## Offline operation and privacy

Inference and preprocessing run locally; the release app’s main Android manifest does not request the INTERNET permission. Debug and profile manifests include that permission for Flutter development tooling. Selecting a file through an Android provider may still depend on that provider’s availability.

Patient data are stored in app-private files as ordinary local records, and Android backup is disabled. The app does not add application-level encryption or a login screen. Deleting a patient attempts to remove linked local visit images and known report files, but this is not a secure-erasure guarantee. Exported or shared PDFs may contain personal information.

Use synthetic patient details and de-identified retinal images for demonstrations. Review the root-level <code>offirst_sample_report_mockup.zip</code> before redistributing this repository; its contents have not been cleared here as synthetic.

## Current source and archived APKs

The Flutter package in the current source declares version <code>1.0.5+2005</code> and application ID <code>org.offirst.offline_dr_screening</code>. The current source snapshot and model are not represented by a verified public APK release.

| Artifact | What it represents |
| --- | --- |
| Current GitHub source, <code>main</code> at <code>212cf1f</code> | App source version 1.0.5+2005 and the changed v2.1 experimental model described above. |
| [Archived <code>offirst-v2.1</code> release](https://github.com/Mayank-gitspace/OFFIRST/releases/tag/offirst-v2.1) | Older published candidate: package version 1.0.3, version code 2003, APK SHA-256 <code>c165bc145ce2fdf6115585594aa4048e728abd0422576f485d36be587c6c77e2</code>. Its release notes identify the previous ONNX hash <code>43a87f05df7adc6a231a679721e7f0b11b45a3fe6d1582aa4527ed738fa63b3f</code>, 16-unit MLP head, Platt calibration, and threshold <code>0.3675977844516245</code>. It does **not** match the current source model. |
| [Archived v1.1 prototype release](https://github.com/Mayank-gitspace/OFFIRST/releases/tag/offirst-android-prototype-v1.1) | Older rollback artifact, separate from the current v2.1 source. The app retains a build-time v1.1 selection for compatibility with older records. |

A local inspection report lists APKs built from a working directory, but their reproducible relationship to this exact commit has not been established. They are not linked here as verified release downloads. The current code contains a deliberate v1.1 rollback define: <code>OFFIRST_EXPERIMENTAL_V21=false</code>. The normal source default selects v2.1.

## Performance and research evidence

The following table reports historical assessments of the earlier v1.1 and v2.1 binary models. These small development assessments are not independent clinical validation and do **not** evaluate the current changed binary head.

| Historical model | Dataset | Images | Accuracy | Sensitivity | Specificity | AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| v1.1 | IDRiD | 72 | 70.83% | 57.45% | 96.00% | 0.846 |
| Earlier v2.1 | IDRiD | 72 | 79.17% | 87.23% | 64.00% | 0.874 |
| v1.1 | RFMiD | 69 | 86.96% | 100.00% | 84.48% | 0.967 |
| Earlier v2.1 | RFMiD | 69 | 81.16% | 54.55% | 86.21% | 0.875 |

Dataset protocols differ; patient independence and prior model exposure are unresolved in these historical assessments. The earlier v2.1 assessment did not meet the combined working gates of 85% accuracy, 85% sensitivity, and 80% specificity on either dataset. **There is no valid evaluation in this repository establishing that the current changed model meets those gates.**

Separate representation-learning reports give RFMiD linear-probe AUROC of 0.917 for direct BYOL MobileNetV3 and 0.932 for KD Run 2 MobileNetV3; reported 10-shot AUROC was 0.640 versus 0.728, and ARMD 10-shot AUROC 0.789 versus 0.949. These are research probes with dataset exposure and split limitations—not application-classifier metrics or clinical results. See [research results](docs/RESEARCH_RESULTS.md) and [data exposure notes](OFFIRST_DATA_LINEAGE_AND_EXPOSURE_MATRIX.csv).

## Technology stack

| Layer | Technology |
| --- | --- |
| Android UI and workflow | Flutter, Dart |
| Image acquisition | Flutter <code>image_picker</code> camera/gallery integration |
| Current v2.1 preprocessing | Kotlin, OpenCV 4.12 |
| Local model inference | ONNX Runtime Flutter package |
| Current v2.1 image tensor | RGB, 512 × 512, ImageNet-normalized float32 NCHW |
| Local persistence | App-private files via <code>path_provider</code> |
| Research | Python, PyTorch, timm, NumPy, scikit-learn, Kaggle |

## Repository map

| Path | Contents |
| --- | --- |
| [<code>deployment/offline_dr_screening/</code>](deployment/offline_dr_screening/) | Flutter Android app, frozen ONNX assets, JSON probes, Kotlin preprocessing/reporting, and tests |
| [<code>docs/</code>](docs/) | Setup, architecture, model, installation, research, and limitation notes |
| [<code>encoder_baseline/</code>](encoder_baseline/) | Supervised encoder baselines and representation probes |
| [<code>encoder_byol/</code>](encoder_byol/) | BYOL self-supervised research |
| [<code>encoder_mnv3_ssl/</code>](encoder_mnv3_ssl/) | MobileNetV3 self-supervised research |
| [<code>kd_run2/</code>](kd_run2/) | Mobile student/teacher distillation experiments and evaluations |
| [<code>kd_imaging/</code>](kd_imaging/) | Imaging and representation analyses |
| [<code>lesion_segmenter/</code>](lesion_segmenter/) | Separate lesion-segmentation research; not part of the app inference pipeline |
| [<code>tools/</code>](tools/) | Research and evaluation utilities |

## Setup and run on Android

Documented development toolchain: Flutter 3.41.9, Dart 3.11.5, JDK 17, and an Android SDK. Install Flutter and Android tooling using their official installation instructions.

~~~bash
git clone https://github.com/Mayank-gitspace/OFFIRST.git
cd OFFIRST/deployment/offline_dr_screening
flutter pub get
flutter run -d <android-device> --dart-define=OFFIRST_EXPERIMENTAL_V21=true
~~~

To build a release APK from source:

~~~bash
flutter build apk --release --dart-define=OFFIRST_EXPERIMENTAL_V21=true
~~~

On Windows, the repository also includes <code>scripts/build_v2_1_final_candidate.ps1</code>, which performs additional asset and package checks. The build uses the project’s debug signing configuration and is not production signing.

To build the legacy v1.1 rollback deliberately, use <code>--dart-define=OFFIRST_EXPERIMENTAL_V21=false</code>. Do not use the older published APK as a substitute for a build from this source snapshot.

The app includes Flutter tests and an Android preprocessing instrumentation test. GitHub reports no Actions workflow runs or commit status checks for the current commit; this review did not run tests, build the app, or verify behavior on a physical Android device. Test files are not evidence that the current build passes.

## Data attribution and licensing

The upstream references reviewed for this README are:

- [RFMiD dataset paper](https://www.mdpi.com/2306-5729/6/2/14), which identifies the dataset license as CC BY 4.0.
- [IDRiD data page](https://idrid.grand-challenge.org/Data/), which identifies a CC BY 4.0 license.
- [APTOS 2019 competition rules](https://www.kaggle.com/c/aptos2019-blindness-detection/rules), which limit competition-data use to competition participation, non-commercial purposes, and academic research/education.
- [ODIR-2019 dataset page](https://odir2019.grand-challenge.org/dataset/), which describes the dataset and registration/download route. The repository review did not establish a dataset-specific redistribution license or the exact accepted access terms.

These upstream terms do not authenticate the exact source versions used to produce repository derivatives or establish that every cached feature, head, sample, or model can be redistributed.

Research and evaluation records refer to RFMiD, IDRiD, APTOS, and ODIR-5K. Follow the original source’s citation, access, and redistribution terms for every dataset and derivative. Project records cite CC BY 4.0 for some RFMiD and IDRiD materials, but do not authenticate the exact local source versions or clear every derived asset. ODIR access is registration-controlled and the reviewed repository records do not establish a dataset-specific redistribution license; APTOS use remains subject to its source/Kaggle terms.

The repository contains dataset-derived sample images, cached features, and model assets. Permissions for distributing those samples, cached outputs, and the current model weights have not been fully established. The repository has no root <code>LICENSE</code> file. Do not infer that dataset or model assets are unrestricted for reuse.

## Screenshots

No verified screenshots of the current Android app are included in this README. Add screenshots only after capturing them from the current source build and checking that they contain no patient-identifiable information.

## Team and acknowledgements

**Team OFFIRST** · ZEISS BeyondCode 2026.

The project draws on work and tools from the BYOL authors, the knowledge-distillation research community, Flutter, ONNX Runtime, OpenCV, and the dataset creators cited in the research records. This project is a hackathon research prototype and does not imply ZEISS product endorsement.

## Limitations and responsible use

- Current model performance is unverified; the 85% accuracy, 85% sensitivity, and 80% specificity working targets are not established for this model.
- The geometric/intensity input gate is not a trained detector, and image gradability is not assessed.
- Severity and multi-disease outputs are exploratory probes and are not clinical diagnoses.
- A result below the binary threshold does not rule out DR.
- The PDF omits the stored binary score, threshold, model version, visit scope, and unavailable-eye status, and can reuse a previously generated version-6 report.
- PDF triage text contains follow-up/referral language derived from the unvalidated severity probe; those timeframes are not clinically established.
- Patient data and images are stored locally without application-level encryption; exported reports may contain sensitive information.
- Dataset, sample-image, and model redistribution rights still need review.

For additional details, see [architecture](docs/ARCHITECTURE.md), [installation](docs/INSTALLATION.md), [known limitations](docs/KNOWN_LIMITATIONS.md), [research results](docs/RESEARCH_RESULTS.md), and the [data lineage and exposure matrix](OFFIRST_DATA_LINEAGE_AND_EXPOSURE_MATRIX.csv).
