# OFFIRST v2.1 Android release handoff

This document describes the Android application selected for the ZEISS BeyondCode 2026 submission. It is an experimental academic research prototype, not a clinical release.

## A. Application

| Property | Value |
| --- | --- |
| Application | OFFIRST |
| Package ID | `org.offirst.offline_dr_screening` |
| Version / version code | `1.0.3` / `2003` |
| Android | Minimum API 24; APK includes `arm64-v8a` and `armeabi-v7a` |
| Candidate APK | `build/app/outputs/flutter-apk/OFFIRST-v2.1-final-candidate.apk` |
| Size | `67,327,505` bytes |
| APK SHA-256 | `c165bc145ce2fdf6115585594aa4048e728abd0422576f485d36be587c6c77e2` |
| Signing | Debug-signed; certificate SHA-256 `08e09b07b3e9e052535d6e34219b4d50f3e320e41da40dcd764f60982d226bc7` |
| Normal model selection | v2.1, no Dart build flag required |
| Build script | `./scripts/build_v2_1_final_candidate.ps1` from this directory |

Implemented application workflows include local patient registration/search/edit/delete, stable patient IDs, local expert directory CRUD and patient associations, visit history, per-image OD/OS image screening, offline ONNX inference, local persistence, and PDF generation/open/export. Each image stores its own result, model identifier, and threshold; the app does not fuse multiple images into a patient-level score.

The application does not implement account authentication, cloud synchronization, expert verification, live scheduling, absent-eye reason codes, multi-disease classification, or a patient-level aggregate score. Local patient and expert information, visit images, and reports are kept in app-private storage without application-level encryption. Explicitly exported PDFs can leave that storage.

## B. Model

| Contract | Value |
| --- | --- |
| Model | OFFIRST v2.1 experimental DR-only classifier |
| Encoder | MobileNetV3-Large, KD Run 2 epoch 40 student |
| Teacher lineage | Frozen ConvNeXt-Tiny BYOL epoch 20, 768-dimensional features |
| Student initialization | Direct BYOL MobileNetV3-Large epoch 40 |
| Projection and head | Original 960→768 projection; supervised 16-unit MLP binary DR head |
| ONNX | `assets/offirst_v2_1_experimental.onnx`, FP32 |
| ONNX SHA-256 | `43a87f05df7adc6a231a679721e7f0b11b45a3fe6d1582aa4527ed738fa63b3f` |
| Encoder checkpoint SHA-256 | `8bc239d22ffc4e786820f2151c99380a754bef930e42edea65022e1246f2096d` |
| Input/output | `fundus_image` float32 `[batch,3,512,512]` RGB/NCHW → `dr_probability` float32 `[batch,1]` |
| Calibration and cutoff | Platt calibration is embedded in ONNX and applied once; cutoff `0.3675977844516245`, positive at `>=` |
| v1.1 rollback ONNX | `assets/offirst_dr_model.onnx`; SHA-256 `b8fc10191f69faef16f1dc92d08bf9542ee7e26f885767942e28292c5201dd90`; historical threshold `0.48` |

v2.1 preprocessing uses local OpenCV 4.12 color decode, BGR-to-RGB, long-edge 512 proxy, RGB max-value `>15` foreground mask, 5×5 morphological opening, crop where at least 100 mask pixels remain, centered square black padding, sequential 640 then 512 `INTER_AREA` resize, ImageNet normalization, and float32 NCHW packing. The normal build selects this path. Pass `--dart-define=OFFIRST_EXPERIMENTAL_V21=false` only to make an explicit v1.1 rollback build.

The v2.1 model follows BYOL initialization and KD Run 2 feature distillation. KD Run 2 is not explicit Masked InfoNCE and the deployed ONNX is not a Masked InfoNCE model. Separate self-supervised and contrastive research material remains distinct from the application model.

## C. Development metrics

| Model | Dataset | Images | Accuracy | Sensitivity | Specificity | AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| v2.1 | IDRiD | 72 | 79.17% | 87.23% | 64.00% | 0.874 |
| v1.1 | IDRiD | 72 | 70.83% | 57.45% | 96.00% | 0.846 |
| v2.1 | RFMiD | 69 | 81.16% | 54.55% | 86.21% | 0.875 |
| v1.1 | RFMiD | 69 | 86.96% | 100.00% | 84.48% | 0.967 |

These are development assessments, not independent clinical validation. Patient independence is unverified for both datasets, prior dataset exposure exists, and the label protocols differ. v2.1 does not meet the combined working gates (accuracy ≥85%, sensitivity ≥85%, specificity ≥80%) on either dataset. RFMiD sensitivity of 54.55% is a substantial false-negative limitation.

Separate representation probes report RFMiD linear-probe AUROC 0.917 → 0.932 and RFMiD 10-shot AUROC 0.640 → 0.728 for direct BYOL MobileNetV3 versus KD Run 2, and ARMD 10-shot AUROC 0.789 → 0.949. These are frozen-encoder representation metrics, not application classifier accuracy. See [research results](../../docs/RESEARCH_RESULTS.md).

## D. Verification record

| Check | Result | Environment and evidence | Remaining issue |
| --- | --- | --- | --- |
| Dart formatting | PASS; 15 files, no changes required | Flutter 3.41.9 / Dart 3.11.5, Windows | None |
| Flutter analyzer | PASS; no issues | `flutter analyze` | None |
| v2.1 Flutter suite | PASS: 16 passed, 2 skipped | Windows host with native ONNX enabled | v2.1 Android MethodChannel/OpenCV path and missing preprocessing fixtures are skipped |
| v1.1 rollback suite | PASS: 17 passed, 1 skipped | Windows host with `OFFIRST_EXPERIMENTAL_V21=false` | Missing preprocessing fixture is skipped |
| Python/OpenCV + ONNX parity | PASS per saved reference evidence: 141 images; max probability error `1.8341754449702208e-06`; zero decision disagreements | Saved `assets/model_v2_1_evidence/model_manifest.json` metrics | Does not test Android preprocessing; assessment images are not bundled |
| Release APK build | PASS with no Dart model-selection define | `scripts/build_v2_1_final_candidate.ps1` | Debug signing; no device execution |
| APK model, threshold, package, version, ABI, signer and permission inspection | PASS | Build-script post-build inspection | Does not prove app startup or workflow |
| Android device/emulator install and startup | NOT TESTED | `adb devices -l` returned no connected devices | Physical phone or emulator required |
| Android preprocessing parity / offline bilateral screening | NOT TESTED | No Android runtime was available | Device test required |
| Upgrade data retention / PDF open and export | NOT TESTED | No Android runtime was available | Device test required |

Formatting, analyzer, and host test results above were executed after changing the no-flag default to v2.1. Host Python/ONNX replay is not evidence of Android tensor parity. No physical-device pass is claimed.

## E. Privacy and use

The release APK manifest has no INTERNET permission; model inference is local. Patient metadata, images, and reports are app-private, not application-encrypted. An explicitly exported report may contain identifying information. Use synthetic records and de-identified photos. No clinical validation is established.

## F. Readiness

| Gate | Status | Basis |
| --- | --- | --- |
| Build readiness | GO | Rebuilt with v2.1 default and verified APK/model metadata |
| Technical device demonstration | NOT VERIFIED | No connected phone or emulator; Android workflow and parity checks remain undone |
| Public research-prototype distribution | GO with disclosed limitations | User authorized publication; release notes and README identify the experimental use, debug signature, development results, and device-test gaps |
| Clinical use or validation | NO-GO / NOT ESTABLISHED | Performance gates are not met and patient independence is unverified |

## G. GitHub handoff

- Integration branch: `integration/offirst-v2.1-final-experimental`
- Publication PR: [#4](https://github.com/Mayank-gitspace/OFFIRST/pull/4), targeting `main`
- Repository homepage: https://github.com/Mayank-gitspace/OFFIRST
- v2.1 release URL: https://github.com/Mayank-gitspace/OFFIRST/releases/tag/offirst-v2.1
- Direct APK: https://github.com/Mayank-gitspace/OFFIRST/releases/download/offirst-v2.1/OFFIRST-v2.1-final-candidate.apk
- SHA-256 manifest: https://github.com/Mayank-gitspace/OFFIRST/releases/download/offirst-v2.1/OFFIRST-v2.1-final-candidate.sha256
- The v1.1 published APK, model, and release remain available at [the v1.1 release](https://github.com/Mayank-gitspace/OFFIRST/releases/tag/offirst-android-prototype-v1.1).
