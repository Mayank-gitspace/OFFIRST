# OFFIRST Android device verification — v1.0.4 candidate

## Scope and result

This is a static artifact/source audit plus a device-availability check. No Android device or emulator was connected (`adb devices -l` returned an empty list), so installation, ONNX session initialization on Android, gallery/camera inference, airplane-mode behavior, saved visits, and PDF export were **not tested**. Do not treat static APK checks or previously recorded host/synthetic tests as device acceptance.

## Artifact checked

`deployment/offline_dr_screening/build/app/outputs/flutter-apk/OFFIRST-v2.1-inference-fix-arm64.apk`

| Property | Observed |
| --- | --- |
| APK SHA-256 | `bb0640215f77b07afe1959fff12fcd2a42582bfb8ef52d1a3333bc136f651f5d` |
| Package | `org.offirst.offline_dr_screening` |
| Version | `1.0.4`, version code `4004` |
| Label | `OFFIRST` |
| ABI | `arm64-v8a` only |
| Signer SHA-256 | `08e09b07b3e9e052535d6e34219b4d50f3e320e41da40dcd764f60982d226bc7` (prototype Android debug key) |
| Bundled v2.1 model entry | `assets/flutter_assets/assets/offirst_v2_1_experimental.onnx`, 14,888,805 bytes |
| Extracted bundled model SHA-256 | `43a87f05df7adc6a231a679721e7f0b11b45a3fe6d1582aa4527ed738fa63b3f` |
| Flutter app snapshot strings | v2.1 asset name and v2.1 active-model UI string found |

The matching release manifest records the ARM64 and ARMv7 artifact hashes. This audit independently inspected the ARM64 candidate; the ARMv7 candidate was not the device target because no device was present. The APK manifest and package signature were inspected with Android SDK build tools. Static inspection cannot prove successful runtime loading.

## Source contract cross-check

Current source selects `assets/offirst_v2_1_experimental.onnx` by default. Its model contract is input `fundus_image`, float32 RGB NCHW `[1,3,512,512]`; output `dr_probability`, float32 `[1,1]`. The source threshold is exactly `0.3675977844516245`, with DR-positive at `>=`. Platt calibration is embedded in the ONNX graph and must not be applied again.

The v1.0.4 build script checks the frozen v1.1/v2.1 hashes, selected v2.1 asset, threshold, package, version, ABI, signer, and lack of INTERNET permission. The current candidate's model entry was independently extracted and hashed in this audit. Exact threshold behavior remains verified at source/build-script level; I did not infer a runtime threshold solely from a substring in the APK.

## Existing inference-path evidence

- The earlier `ANDROID_INFERENCE_P0_FIX.md` records prior host Flutter tests, host ONNX load/contract and five synthetic tensor predictions, plus successful Android instrumentation **compilation**.
- `android/app/src/androidTest/.../V21PipelineTest.kt` contains tests for OpenCV float-read byte counts, corrupt/unsupported files, and parity of five synthetic Android tensors against Python tensors. This suite was not executed on a device in this pass.
- The synthetic fixture manifest is not a set of labeled retinal photographs and does not validate classification performance.
- No actual Android runtime model-load or image-to-tensor success is established by this audit.

## Device acceptance checklist (pending)

On a compatible ARM64 phone with a synthetic patient and permitted de-identified test images:

1. Record device model, Android version, build fingerprint, and installed app version.
2. Install/update the candidate while preserving app data; verify startup and model readiness.
3. Run the instrumentation tensor parity tests on device with the generated fixture assets.
4. Select a gallery fundus image and a camera-captured fundus image; verify inference completes and failures are surfaced as failures, never negative scores.
5. Repeat in airplane mode; confirm no network permission/use is needed.
6. Save and reopen OD and OS visits, including multiple images; verify probability, exact threshold, and model version persist.
7. Generate, open, and export a PDF; verify each image and its recorded model/threshold.
8. Try a non-fundus photograph. Current source has no real fundus detector, so document that it is accepted as a known limitation; do not claim this step passes input validation.

For instrumentation execution, generate the synthetic fixtures as documented in `deployment/offline_dr_screening/ANDROID_INFERENCE_P0_FIX.md`, then run `:app:connectedDebugAndroidTest` with the device attached. This does not replace clinical validation.

## Conclusion

The retained ARM64 v1.0.4 candidate contains the expected v2.1 model, and its static package/version/ABI/signature metadata match the recorded candidate. The source and build metadata specify the expected tensor contract and threshold. Android runtime model loading and conversion, gallery/camera use, offline workflow, persistence, and PDF behavior remain unverified on a physical device.
