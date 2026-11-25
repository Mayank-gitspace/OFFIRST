# OFFIRST v2.1 experimental Android candidate

This is an isolated, opt-in candidate build. The default application remains
OFFIRST v1.1 with its original ONNX file, preprocessing implementation, and
0.48 threshold. Build this candidate with
`--dart-define=OFFIRST_EXPERIMENTAL_V21=true`; the v2.1 application is a
separate experimental candidate. Do not promote it as a public release without
completing the Android and parity gates below.

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

The final candidate is built deterministically from this directory with:

```powershell
./scripts/build_v2_1_final_candidate.ps1
```

The script verifies the frozen v1.1/v2.1 model hashes and v2.1 threshold,
builds with `OFFIRST_EXPERIMENTAL_V21=true`, and checks the packaged asset,
Dart build identity, package, label, version, ABIs and signing certificate. It
also records APK and model hashes. The candidate retains the Android package
ID and uses version 1.0.3 / version code 2003, greater than the v1.1 public
APK's version code 2001 and the earlier 2002 candidate. The APK uses the
Android debug signing certificate, whose SHA-256 is
`08e09b07b3e9e052535d6e34219b4d50f3e320e41da40dcd764f60982d226bc7`.

The verified local candidate is
`build/app/outputs/flutter-apk/OFFIRST-v2.1-final-candidate.apk` (67,327,529
bytes), SHA-256
`a73431d6054fa8725d0aead3006a50d5173bf345fb3d58f30db3ca27d3b14e55`.
It contains ARM64 and ARMv7 code and both model assets; the v2.1 build selects
the v2.1 ONNX model. The v1.1 model, published APK and release remain intact.
The APK signature and metadata were verified locally. This successful build
does not verify device installation, Android preprocessing parity, startup or
the screening/report workflow.

## Physical phone test procedure

No Android device or emulator was attached during this publication check
(`adb devices -l` returned no devices and no available AVD was found).
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
  141-image evidence bundle (maximum error `1.8341754449702208e-06`, mean
  error `2.470365740902819e-07`, zero threshold disagreements; closest sample
  `0.0023771032404520387` from the cutoff).
- Android OpenCV preprocessing parity against Python fixtures: not yet verified.
- Candidate Flutter runtime probability parity: not yet verified.
- Android install/startup and bilateral offline workflow: not yet verified.
- Physical-device verification: not yet verified.
- Final local APK: built and packaging/signature-verified; this is a draft
  experimental candidate, not a public clinical release.
- Public release promotion: NO-GO until the above Android gates pass and the
  user separately approves publication given the unmet performance targets.
