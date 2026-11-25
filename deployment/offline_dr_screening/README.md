# OFFIRST Android application

This directory contains the Flutter Android application and its two frozen ONNX assets. A normal Android release build selects the experimental v2.1 DR classifier by default. The earlier v1.1 model remains as a rollback and for interpreting legacy records.

## Offline workflow

1. Register or select a patient and create a visit.
2. Add one or more retinal images for the right eye (OD) and/or left eye (OS).
3. Preprocess each image on Android, run local ONNX inference, and review its separate probability and threshold outcome.
4. Save the visit and original image files, then generate, open, or export the report.

The application does not aggregate images into a patient-level score. Each image record stores its eye, probability, model version, and threshold. An unpopulated eye has no model assessment. Patient and visit data remain in app-private storage; exported PDFs may contain sensitive information. Android backup is disabled, but the prototype does not add application-level encryption. Use synthetic patient details and de-identified images.

## Active v2.1 model contract

- Asset: `assets/offirst_v2_1_experimental.onnx`; SHA-256 `43a87f05df7adc6a231a679721e7f0b11b45a3fe6d1582aa4527ed738fa63b3f`.
- Input: `fundus_image`, float32 RGB/NCHW `[1, 3, 512, 512]`.
- Output: `dr_probability`, calibrated float32 probability `[1, 1]`.
- Threshold: `0.3675977844516245`, inclusive (`>=`). Platt calibration is already part of the ONNX graph and is not applied again in the app.
- Preprocessing: OpenCV 4.12 color decode, RGB conversion, fundus mask/crop and centered padding, two-stage resize (640 then 512), ImageNet normalization, float32 NCHW tensor.

The v2.1 classifier is DR-only: a KD Run 2 MobileNetV3-Large epoch 40 student with the existing 960→768 projection and supervised 16-unit MLP head, from a BYOL initialization and teacher/student distillation lineage. It is not an explicit Masked InfoNCE model. See `assets/model_v2_1_evidence/` and [the final release handoff](FINAL_RELEASE_HANDOFF.md).

## v1.1 rollback model

`assets/offirst_dr_model.onnx` has SHA-256 `b8fc10191f69faef16f1dc92d08bf9542ee7e26f885767942e28292c5201dd90` and historical threshold `0.48`. To build the rollback variant deliberately, use `--dart-define=OFFIRST_EXPERIMENTAL_V21=false`. Do not use that define for the primary v2.1 application.

## Build and verify

Use Flutter 3.41.9, Dart 3.11.5, JDK 17, and the Android SDK configured for the project. From this directory:

```powershell
flutter pub get
dart format --output=none --set-exit-if-changed lib test
flutter analyze
flutter test
./scripts/build_v2_1_final_candidate.ps1
```

The build script verifies frozen model hashes, the threshold, APK package and label, version, ARM64/ARMv7 ABIs, signer, packaged model reference, and the absence of Android INTERNET permission. It writes an APK checksum manifest. The candidate uses the debug signing key and is not production signed. See [installation](../../docs/INSTALLATION.md), [architecture](../../docs/ARCHITECTURE.md), and [known limitations](../../docs/KNOWN_LIMITATIONS.md).
