# OFFIRST v2.1 experimental Android candidate

This is an isolated, opt-in candidate build. The default application remains
OFFIRST v1.1 with its original ONNX file, preprocessing implementation, and
0.48 threshold. Build this candidate with
`--dart-define=OFFIRST_EXPERIMENTAL_V21=true`; do not promote it as a release
without completing the Android and parity gates below.

## Frozen model evidence

- Artifact: `assets/offirst_v2_1_experimental.onnx`
- SHA-256: `43a87f05df7adc6a231a679721e7f0b11b45a3fe6d1582aa4527ed738fa63b3f`
- Export: FP32, opset 17, 14,888,805 bytes
- Input: `fundus_image`, float32 `[1, 3, 512, 512]`, RGB NCHW
- Output: `dr_probability`, float32 `[1, 1]`
- Development-selected threshold: `0.3675977844516245`, positive at `>=`
- Frozen encoder provenance: KD Run 2 MobileNetV3-Large epoch 40,
  checkpoint SHA-256
  `8bc239d22ffc4e786820f2151c99380a754bef930e42edea65022e1246f2096d`;
  existing 960-to-768 projection followed by the fitted 16-unit MLP.
- Platt slope/intercept are incorporated into the exported graph. The Android
  application must consume its probability output directly and must not apply
  a second calibration step.

The detailed provenance/configuration files are retained under
`assets/model_v2_1_evidence/`. The graph was independently checked against its
141-image development assessment export: recorded maximum probability error
`1.8341754449702208e-06`, mean error `2.470365740902819e-07`, and zero threshold
disagreements. This confirms the exported graph against the saved Python output;
it does not establish Android/Dart preprocessing parity or independent clinical
performance.

## Candidate preprocessing

The candidate calls bundled OpenCV 4.12.0 on Android and follows the saved
Python pipeline: OpenCV color decode, BGR-to-RGB, 512-pixel long-side proxy,
maximum-channel foreground threshold `>15`, 5x5 morphological opening, crop
when at least 100 mask pixels remain, centered square black padding, sequential
640 then 512 `INTER_AREA` resize, ImageNet normalization, and NCHW float32.
This path is separate from v1.1's preprocessing.

## Development-only evaluation

The assessment is not patient-independent and includes historically exposed
images. The candidate did not meet all working gates:

| Dataset | Images | Accuracy | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: |
| IDRiD | 72 | 79.17% | 87.23% | 64.00% |
| RFMiD | 69 | 81.16% | 54.55% | 86.21% |

These are development results, not clinical validation. Sensitivity on RFMiD
shows a substantial false-negative limitation.

## Build

From this directory, use the project's pinned Flutter/Gradle setup:

```powershell
flutter pub get
flutter analyze
flutter test
flutter build apk --release --build-name=1.0.2 --build-number=2002 --target-platform android-arm64,android-arm --dart-define=OFFIRST_EXPERIMENTAL_V21=true
```

The candidate uses the existing Android application ID and debug signing
configuration. The published ARM64 v1.1 APK has version code 2001; this
candidate uses 2002 so Android can attempt an in-place upgrade. A successful build alone does not verify installation over an
existing app, native preprocessing parity, app startup, or the screening/report
workflow. Retain the v1.1 submission APK unless those checks pass on Android.

The built candidate is
`build/app/outputs/flutter-apk/offirst-v2.1-experimental-arm64-arm.apk`.
Its SHA-256 is
`2bbf6710f3ec068cbd83acb7f4833aa8b99130e560ba4688816076bedd6c039d`.
It contains ARM64 and ARMv7 code. The package name is
`org.offirst.offline_dr_screening`; its signing certificate SHA-256 is
`08e09b07b3e9e052535d6e34219b4d50f3e320e41da40dcd764f60982d226bc7`,
matching the published v1.1 ARM64 APK. The APK's Dart snapshot contains the
v2.1 build text and v2.1 model asset reference; the v1.1 build text and native
JPEG helper are absent. Both ONNX assets remain packaged, and the build flag
selects the v2.1 asset in `ScreeningService`.

## Physical phone test procedure

No Android device or emulator was attached during this publication check.
Use a test phone with synthetic patient records and de-identified retinal
images. Before upgrading a phone that contains existing records, record a
patient ID and a saved visit/report for the post-upgrade persistence check.
Do not uninstall the application or clear its data.

From this directory in PowerShell:

```powershell
$apk = 'build/app/outputs/flutter-apk/offirst-v2.1-experimental-arm64-arm.apk'
Get-FileHash $apk -Algorithm SHA256
adb devices -l
adb install -r $apk
adb shell dumpsys package org.offirst.offline_dr_screening | Select-String 'versionCode|versionName'
adb shell monkey -p org.offirst.offline_dr_screening 1
adb logcat -d -s AndroidRuntime:E flutter:E
```

`adb install -r` requests an in-place upgrade and retains app data if Android
accepts the package/signature/version combination. A successful install is
not proof that the pre-upgrade records survived; reopen the previously
recorded patient and visit to check. If installation fails, keep the existing
app installed and record the exact ADB error.

On the phone, verify the About page identifies the experimental v2.1 build.
Create a synthetic patient; screen one OD image and one different OS image;
confirm each image has its own score and eye label. Turn on airplane mode and
repeat an inference, then save the visit, force-close and reopen OFFIRST, and
check the saved scores, model identifiers and report. Generate, open and export
the PDF, checking that each image retains its original threshold. For an
upgrade test, also reopen the pre-upgrade v1.1 visit and confirm its 0.48
threshold and model identifier remain intact.

The normal UI rounds displayed probabilities. It cannot establish exact
Android-versus-Python tensor or probability parity. A device-side fixture
capture/instrumentation check is required before marking that gate passed.

## Current gate status

- Python frozen-reference / ONNX probability parity: verified by the saved
  141-image evidence bundle.
- Android OpenCV preprocessing parity against Python fixtures: not yet verified.
- Candidate Flutter runtime probability parity: not yet verified.
- Android install/startup and bilateral offline workflow: not yet verified.
- Physical-device verification: not yet verified.
- Public release promotion: NO-GO until the above Android gates pass and the
  user separately approves publication given the unmet performance targets.
