# OFFIRST v2.1 final candidate — technical handoff

This document describes the Android experimental candidate on
`integration/offirst-v2.1-final-experimental` and PR #4. The PR remains draft
and unmerged. It is not the repository's default application or a clinical
release. The candidate application source was committed at
`397a083c205ba3a5bd2faad61922404bc0e6c2cb`; the final deterministic build
script and release handoff were committed at
`fc814135789c3482cabed9099556efacda605b4b`. The latest documentation revision
is on the current PR #4 head.

## Application

The Flutter Android app provides locally stored patients (name, optional age
and notes), patient search/edit/delete, random stable patient IDs, a local
expert directory with CRUD and patient association, and visit history. A user
can select multiple images for OD or OS independently, run the selected ONNX
model on-device, save the original images and each image's own score/model/
threshold, and generate, open, save or export a PDF report. Local visits and
expert/patient metadata use app-private storage. Writes are staged and visit
startup reconciliation is covered by storage tests.

The application does not implement accounts/authentication, cloud sync,
clinician verification, live scheduling, anatomically absent-eye or
unable-to-assess reasons, a combined patient score, or multi-disease
classification. Patient fields are intentionally limited to name, optional
age and notes. An unpopulated eye is shown as having no model assessment.
Expert details are locally entered and unverified. Association is not evidence
of clinical review. Records are not encrypted at the application layer.

| Property | Final candidate |
| --- | --- |
| App label | OFFIRST |
| Package | `org.offirst.offline_dr_screening` |
| Build name/code | 1.0.3 / 2003 |
| Supported ABIs | `arm64-v8a`, `armeabi-v7a` |
| Candidate APK | `build/app/outputs/flutter-apk/OFFIRST-v2.1-final-candidate.apk` (67,327,529 bytes) |
| APK SHA-256 | `a73431d6054fa8725d0aead3006a50d5173bf345fb3d58f30db3ca27d3b14e55` |
| Build command | `./scripts/build_v2_1_final_candidate.ps1` from this project directory |
| Signing | Android debug key; SHA-256 `08e09b07b3e9e052535d6e34219b4d50f3e320e41da40dcd764f60982d226bc7`, matches published prototype signer |

The build script checks the frozen model hashes and threshold, builds with the
v2.1 Dart define, validates the packaged model reference, identity, version,
ABIs, signer and absence of Internet permission, then writes an APK/model
checksum manifest. The normal build without the v2.1 Dart define retains the
v1.1 path. The default script mode checks the prototype signing certificate.
For a clean clone without the original local debug key, run
`./scripts/build_v2_1_final_candidate.ps1 -AllowDifferentSigner`; this builds
and verifies a locally signed test APK, but it cannot upgrade an install
signed with the published prototype key. No private signing key is included.

## Model

The selected DR-only candidate is an FP32 ONNX export of the frozen KD Run 2
MobileNetV3-Large epoch 40 encoder, its original 960-to-768 projection, and a
supervised 16-unit MLP binary DR head. It does not include a multi-disease
head. The encoder checkpoint SHA-256 in the evidence manifest is
`8bc239d22ffc4e786820f2151c99380a754bef930e42edea65022e1246f2096d`.

| Contract | Value |
| --- | --- |
| ONNX | `assets/offirst_v2_1_experimental.onnx` |
| SHA-256 | `43a87f05df7adc6a231a679721e7f0b11b45a3fe6d1582aa4527ed738fa63b3f` |
| Input | `fundus_image`, float32 `[batch, 3, 512, 512]`, RGB, NCHW |
| Output | `dr_probability`, float32 `[batch, 1]`, probability in `[0,1]` |
| Calibration and cutoff | Platt calibration is baked into the exported graph; compare its probability once against `0.3675977844516245` using `>=` |
| Runtime | ONNX Runtime, local Android execution |
| Image preprocessing | OpenCV 4.12 decode with color orientation handling; BGR-to-RGB; long-edge 512 proxy; RGB max `>15` foreground mask; 5x5 morphological open; bounding crop when mask has at least 100 pixels; centered square black pad; 640 then 512 `INTER_AREA`; ImageNet normalization; float32 NCHW |

The graph already contains Platt calibration. No app-side calibration is
applied. `model_manifest.json`, `prediction_config.json`, and
`preprocessing.json` are in `assets/model_v2_1_evidence/`. The v1.1 model
remains at `assets/offirst_dr_model.onnx`, SHA-256
`b8fc10191f69faef16f1dc92d08bf9542ee7e26f885767942e28292c5201dd90`, with
its recorded 0.48 threshold. Image records store their own model identifier,
probability and threshold; reports read those per-image values. Legacy records
without model metadata are interpreted as v1.1 at 0.48.

The Android preprocessing implementation has not been compared against
Python-produced tensors on a device. The retained Python/OpenCV plus ONNX
replay evidence covers 141 assessment images: maximum absolute probability
error `1.8341754449702208e-06`, mean error `2.470365740902819e-07`, and zero
decision disagreements. The image closest to the threshold was
`0.0023771032404520387` away. This verifies the saved Python/ONNX reference,
not Android preprocessing parity.

## Development metrics and limitations

These previously recorded assessments are development results, use images
without verified patient identifiers, and do not establish independent
clinical validation.

| Dataset | Images | Accuracy | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: |
| IDRiD | 72 | 79.17% | 87.23% | 64.00% |
| RFMiD | 69 | 81.16% | 54.55% | 86.21% |

The combined accuracy, sensitivity and specificity gates are not met. In
particular, RFMiD sensitivity is 54.55%, a substantial false-negative
limitation. Do not use model output alone for diagnosis, referral, treatment
or clinical decisions.

## Verification record

| Check | Status | Environment/evidence | Remaining issue |
| --- | --- | --- | --- |
| Flutter formatting | PASS; no files needed formatting | `dart format --output=none --set-exit-if-changed lib test` | None |
| Flutter analyzer | PASS; no issues | `flutter analyze` | None |
| v2.1 unit, storage, widget and report-payload tests | PASS: 16 passed, 2 skipped | `OFFIRST_NATIVE_TEST=1 flutter test --dart-define=OFFIRST_EXPERIMENTAL_V21=true` on Windows | Skips: v2.1 Android preprocessing service and unavailable fixture directory; native PDF rendering is not exercised |
| Default v1.1 Flutter suite | PASS: 17 passed, 1 skipped | `OFFIRST_NATIVE_TEST=1 flutter test` on Windows | Skipped preprocessing fixtures are absent |
| ONNX model contract and host execution | PASS | Native Windows ONNX Runtime loaded the selected model; tensor contract and probability-range checks passed | Host runtime is not Android runtime |
| Threshold boundary | PASS | Flutter tests exercise just below, at and above the active cutoff | Android display not checked |
| Python/OpenCV plus ONNX parity | PASS per retained 141-image evidence | `assets/model_v2_1_evidence/model_manifest.json` | Restricted assessment images are not included; does not test Android tensors |
| Android release build and APK checks | PASS | `scripts/build_v2_1_final_candidate.ps1`; package, version, label, ARM64/ARMv7, embedded model hash and signer verified | Build does not prove runtime behavior |
| ADB install, startup, bilateral workflow, airplane mode, persistence, PDF and upgrade retention | NOT TESTED | `adb devices -l` returned no attached devices; no emulator AVD was available | Android device/emulator required |
| Native Android preprocessing parity | NOT TESTED | No Android runtime/fixture capture available | Device-side tensor/probability fixture comparison required |
| Physical-device validation | NOT TESTED | No device attached | Do not report as passed |

Run the commands below from this project directory:

```powershell
dart format --output=none --set-exit-if-changed lib test
flutter analyze
$env:OFFIRST_NATIVE_TEST = '1'
$onnxDll = Join-Path $env:LOCALAPPDATA 'Pub\Cache\hosted\pub.dev\onnxruntime-1.4.1\windows'
$env:PATH = "$onnxDll;$env:PATH"
flutter test --dart-define=OFFIRST_EXPERIMENTAL_V21=true
flutter test
./scripts/build_v2_1_final_candidate.ps1
```

The native ONNX tests on a Windows host were enabled with the pinned package
DLLs in `PATH`. The v2.1 Android MethodChannel/OpenCV screening path still
requires an Android device or emulator.

## Security and privacy

The app manifest does not declare network, camera or broad storage permissions.
Images come from explicit camera/gallery picker actions. Inference executes
locally. Patient and expert metadata, visit images and generated reports are
stored in app-private directories without application-level encryption or an
app lock. Android backup is disabled. PDF sharing uses a non-exported
FileProvider restricted to `screening_reports/` and grants read access for an
explicit user action. Reports copied to an external destination leave the app
sandbox. The source scan found no embedded credential configuration. A device
security review and Android runtime audit were not performed.

## Release readiness

| Gate | Decision | Basis |
| --- | --- | --- |
| Build readiness | GO | Build script completed and verified the final APK |
| Technical demonstration | NO-GO pending Android startup, bilateral offline workflow, report and preprocessing parity checks | No device/emulator evidence |
| Public application release | NO-GO | Device checks pending; development model gates not met; draft PR/release only |
| Clinical validation | NOT ESTABLISHED | Development samples lack verified patient independence and show a false-negative limitation |

## GitHub handoff

- Branch: `integration/offirst-v2.1-final-experimental`
- Source: PR #4, based on `feature/offirst-precision-ui-and-clinical-workflow`
- PR: https://github.com/Mayank-gitspace/OFFIRST/pull/4
- Merge state: open draft, unmerged
- Public release state: v1.1 preserved; v2.1 candidate is not the default download
- Candidate implementation commit: `397a083c205ba3a5bd2faad61922404bc0e6c2cb` (pushed to the PR branch)
- Deterministic build-script and release-target commit: `fc814135789c3482cabed9099556efacda605b4b`
- Draft v2.1 release: [Unpublished draft release](https://github.com/Mayank-gitspace/OFFIRST/releases/tag/untagged-b5149a1608a3dbee6419); intended tag `offirst-v2.1-final-candidate`; target commit `fc814135789c3482cabed9099556efacda605b4b`
- Uploaded APK: [OFFIRST-v2.1-final-candidate.apk](https://github.com/Mayank-gitspace/OFFIRST/releases/download/untagged-b5149a1608a3dbee6419/OFFIRST-v2.1-final-candidate.apk), 67,327,529 bytes; SHA-256 `a73431d6054fa8725d0aead3006a50d5173bf345fb3d58f30db3ca27d3b14e55`
- Uploaded checksum manifest: [OFFIRST-v2.1-final-candidate.sha256](https://github.com/Mayank-gitspace/OFFIRST/releases/download/untagged-b5149a1608a3dbee6419/OFFIRST-v2.1-final-candidate.sha256); GitHub asset state `uploaded` and size verified
- Release state: draft and prerelease; not published. Both assets are attached to the draft.
