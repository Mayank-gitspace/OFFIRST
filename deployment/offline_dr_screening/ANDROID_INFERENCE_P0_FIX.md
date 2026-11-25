# OFFIRST Android inference P0 fix

## Confirmed root cause

The v2.1 Kotlin preprocessor passed a `FloatArray` to OpenCV `Mat.get` and compared the return value with the array's element count. OpenCV 4.12 returns the number of bytes copied. For `[1, 3, 512, 512]`, a complete read returns 3,145,728 bytes, while the old check expected 786,432 and threw on every image.

After correcting that check, the old Dart handoff also failed: a `Uint8List` decoded from a `StandardMethodCodec` envelope can have byte offset 7. `Float32List.view` requires an offset aligned to four bytes, so directly viewing the envelope payload raises `ArgumentError`. The regression test exercises an actual codec round trip and proves the original view fails.

No connected Android device was available for logcat capture. These causes are established from the OpenCV Java API behavior, source inspection, and executable Flutter regression coverage; phone logcat confirmation is still pending.

## Changes

- `android/app/src/main/kotlin/org/offirst/offline_dr_screening/V21Preprocessor.kt` contains the extracted v2.1 preprocessing path, checks float read bytes, rejects unsupported/corrupt/oversized images by stage, and releases native matrices on all exits.
- `MainActivity.kt` returns stage-coded preprocessing failures and logs stage, exception type, and code frames without image paths or patient data.
- `lib/screening_service.dart` decodes method-channel floats from little-endian bytes into an aligned `Float32List`, validates finite values, reports sanitized failures, and exposes model readiness only after session initialization and contract validation.
- `lib/main.dart` displays initializing, ready, and unavailable model states. Ready means the ONNX session loaded and its tensor contract passed; it does not mean an image was screened.
- `lib/screening_page.dart` requires patient selection before image selection, keeps OD/OS assignment for each image, retains successful results if another image fails, supports retry for failed images, and prevents saving while failures remain unresolved.
- `test/v21_pipeline_regression_test.dart` covers codec alignment, readiness/retry, patient requirement, per-eye association, independent scores, and mixed success/failure handling.
- `test/native_inference_test.dart` replays five synthetic Python-reference tensors through the actual v2.1 ONNX asset and checks returned probabilities and decisions.
- `android/app/src/androidTest/.../V21PipelineTest.kt` covers OpenCV float-read units, corrupt/unsupported-image stages, and native tensor parity against generated fixtures.
- `scripts/generate_v21_regression_fixtures.py` creates synthetic gallery JPEG, PNG, camera-like landscape JPEG, EXIF-rotated JPEG, and 12 MP JPEG fixtures, Python tensors, ONNX probabilities, and a manifest. No clinical photographs or patient data are included.

Gallery selection uses the pinned `image_picker_android` implementation, which opens Android content URIs and copies their bytes into app cache before returning an `XFile` path. Camera captures are likewise returned as app-cache files. Flutter then reads the cached file bytes and passes those bytes across the channel; native code does not reopen a `content://` URI.

## Frozen model and preprocessing

The selected asset remains `assets/offirst_v2_1_experimental.onnx`, SHA-256 `43a87f05df7adc6a231a679721e7f0b11b45a3fe6d1582aa4527ed738fa63b3f`. The input remains `fundus_image`, FP32 RGB NCHW `[1, 3, 512, 512]`; the output remains `dr_probability`, FP32 `[1, 1]`. Platt calibration is already integrated in the graph. The decision threshold remains `0.3675977844516245`, applied once. The v1.1 asset and its historical threshold `0.48` remain unchanged.

Android uses the frozen Python sequence: OpenCV `IMREAD_COLOR` decode; BGR-to-RGB; long-edge 512 proxy resize with `INTER_AREA`; max-channel `>15` mask; 5×5 morphological opening; bounding-box crop when at least 100 foreground pixels remain; centered square black padding; sequential 640×640 then 512×512 `INTER_AREA` resize; ImageNet normalization; planar RGB FP32 tensor packing. No model calibration or threshold logic runs in Kotlin preprocessing.

The fixture generator confirmed the asset hash and produced finite reference predictions for all five synthetic inputs using ONNX Runtime 1.23.2. The Dart host integration test loads the packaged v2.1 asset, checks its tensor contract, and reproduces all five reference probabilities within `1e-5`. This verifies the ONNX model and Dart service on the Windows host. It does not execute Android OpenCV or Android ONNX Runtime.

## Verification

| Check | Result |
| --- | --- |
| Dart format | PASS; no changes required |
| Flutter analyzer | PASS; no issues |
| Flutter suite | PASS; 20 passed, 2 skipped |
| Method-channel decoder | PASS; misaligned codec view regression covered |
| v2.1 ONNX host load, contract, finite probability | PASS |
| Five synthetic fixture predictions | PASS through actual v2.1 ONNX on host |
| Android instrumentation source compilation | PASS: `:app:assembleDebugAndroidTest` |
| OpenCV instrumentation execution | NOT RUN; ADB reports no device/emulator |
| Gallery, camera, airplane mode, bilateral phone workflow, persistence, PDF export | NOT TESTED on device |

The two skipped Flutter tests are platform-specific native inference and unavailable external parity fixtures. The five fresh fixtures are synthetic and are not evidence of clinical performance or a real fundus photograph's diagnostic value.

## Candidate APKs

To create the fixtures used by the native instrumentation test, install the pinned dependencies and generate local-only synthetic assets:

```powershell
python -m venv build/v21-fixture-env
build/v21-fixture-env/Scripts/python -m pip install -r scripts/requirements-v21-fixtures.txt
build/v21-fixture-env/Scripts/python scripts/generate_v21_regression_fixtures.py
```

On a connected emulator/device, execute the Android OpenCV tests with:

```powershell
Push-Location android
.\gradlew.bat :app:connectedDebugAndroidTest
Pop-Location
```

The reproducible release build command is:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_v2_1_final_candidate.ps1
```

It builds separate ARM64 and ARMv7 release APKs with package `org.offirst.offline_dr_screening`, label `OFFIRST`, version name `1.0.4`, and ABI-specific monotonically increasing version codes (ARM64 `4004`, ARMv7 `3004`). Both use the prototype debug certificate fingerprint `08e09b07b3e9e052535d6e34219b4d50f3e320e41da40dcd764f60982d226bc7`. They are candidate artifacts and have not passed device acceptance testing.

| ABI | APK | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| ARM64 | `build/app/outputs/flutter-apk/OFFIRST-v2.1-inference-fix-arm64.apk` | 47,637,551 | `bb0640215f77b07afe1959fff12fcd2a42582bfb8ef52d1a3333bc136f651f5d` |
| ARMv7 | `build/app/outputs/flutter-apk/OFFIRST-v2.1-inference-fix-armv7.apk` | 45,018,527 | `83edeef6cbaff5a8b5f684e7e1cc3f3c6cb670848f3212b7432012916cb54103` |

`OFFIRST-v2.1-inference-fix.sha256` contains both APK hashes and the frozen ONNX hash. The release candidates must remain draft assets until physical-device gallery/camera inference, offline operation, visit persistence, and PDF export are verified.
