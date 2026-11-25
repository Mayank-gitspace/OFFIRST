# OFFIRST preprocessing audit

## Reference and correction

The deployment archive contains `inference.py`, `preprocessing.json`, and the ONNX and TorchScript exports. Its inference code explicitly uses `Image.open(...).convert("RGB")`, Pillow BILINEAR at 512 × 512, and NumPy float32 scaling and ImageNet normalization before producing NCHW. It does not apply EXIF orientation. The repository research transforms and checkpoints do not establish a different preprocessing contract for this exported classifier.

The former Dart path used the image package's simple linear resize and its JPEG decoder, which bakes EXIF orientation and uses different JPEG chroma reconstruction. The revised path decodes JPEGs with libjpeg-turbo 3.1.1 and preserves encoded orientation. It uses the separable Pillow bilinear sampling geometry, widened antialiasing support on downsampling, clipped and normalized edge weights, 22-bit fixed-point coefficients, uint8 rounding after each pass, and float32 rounding at each normalization operation. PNG uses the existing Dart decoder with explicit RGB conversion, including saturation for unsigned 16-bit grayscale. Other image formats retain the existing decoder path and were not included in this parity run.

The original ONNX SHA-256 is `b8fc10191f69faef16f1dc92d08bf9542ee7e26f885767942e28292c5201dd90`. Session validation checks the original float32 `fundus_image` and `dr_probability` names and shapes `[batch_size,3,512,512]` and `[batch_size,1]`. Runtime output must be exactly one float32 tensor of shape `[1,1]` with a finite probability in [0,1]. The threshold remains 0.48.

## Re-audit review

- The JPEG shim accepts 8-bit JPEGs and caps decoded images at 40 megapixels. It checks the header and output channel count before allocating the scanline buffer, converts native failures to Dart format errors, and frees native buffers on success and failure. The shared Dart test rejects both a malformed header and a truncated JPEG before decoding valid fixtures in the same test.
- JPEGs retain encoded orientation to match Pillow. The EXIF fixtures cover values 2 through 8. Android picker and preview behavior still needs a device test because the displayed orientation can differ from the pixels used for inference.
- The APK links the checksum-pinned libjpeg-turbo source statically into the small ABI-specific shim. The native TurboJPEG API and SIMD extensions are disabled for this shim. The host and APK build use separate configurations. The shim's full-resolution decoded buffer is copied into Dart before resampling. Peak memory on low-memory Android devices remains unmeasured.
- The model asset, candidate and generated binaries, fixtures, checker outputs, credentials and patient records are excluded from the proposed commit. The scope is 15 application, configuration and test files plus two documentation files. The v1.0 GitHub pre-release still refers to the earlier APK.

## Measured parity

Windows host, Pillow 11.3.0 with libjpeg-turbo 3.1.1, NumPy 2.2.6, and ONNX Runtime 1.26.0 CPU for both sides of the comparison. The Flutter test calls the production Dart preprocessing and exports the actual float32 tensors. Python reads those tensors and independently constructed Pillow tensors, then runs both through the same unchanged ONNX model.

| Check | Result |
| --- | ---: |
| Shared inputs | 24 |
| Float32 values compared | 18,874,368 |
| Unequal float32 values | 0 |
| Maximum absolute tensor difference | 0 |
| Maximum absolute ONNX probability difference | 0 |
| Disagreements at 0.48 | 0 |

The fixtures include 19 deterministic synthetic images covering enlargement, reduction, mixed axes, square and nonsquare images, RGBA, palette, grayscale and 16-bit grayscale PNG, CMYK JPEG, and EXIF orientation values 2 through 8 with baseline and progressive 4:2:0 JPEG. The other five are existing repository research montages or previews. No separate cleared original fundus photo set was available. These checks measure preprocessing parity on the listed inputs. They do not measure diagnostic accuracy, mobile latency, or individual acquisition conditions.

## Reproduce on Windows

Use Python with Pillow 11.3.0, NumPy 2.2.6 and ONNX Runtime 1.26.0. Build the native shim with CMake, Ninja and a C compiler. For a fresh checkout, provision the original ONNX as described in [README.md](README.md) and populate the Flutter dependency cache with `flutter pub get`. CMake downloads and verifies the pinned upstream libjpeg-turbo archive by default. The commands below reproduce the re-audit using a local copy of that archive. If you use CMake's download instead, omit the `OFFIRST_JPEG_ARCHIVE` assignment.

From `deployment/offline_dr_screening`:

```powershell
$env:OFFIRST_JPEG_ARCHIVE = (Resolve-Path build/offirst-libjpeg-turbo-3.1.1.tar.gz).Path.Replace('\', '/')
cmake -S native -B build/offirst-jpeg-windows -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/offirst-jpeg-windows --parallel 4
python tools/check_preprocessing_parity.py prepare build/preprocessing-re-audit `
  --image ../../kd_imaging/demo/grid_clean_vs_degraded.jpg `
  --image ../../kd_run2/localtest/view_alignment_example.jpg `
  --image ../../encoder_mnv3_ssl/localtest/grid_preview.jpg `
  --image ../../encoder_byol/v4/localtest/grid_top4.jpg `
  --image ../../encoder_mnv3_ssl/localtest/degrade_equiv_1.jpg
$env:OFFIRST_PARITY_DIR = (Resolve-Path build/preprocessing-re-audit).Path
$env:OFFIRST_NATIVE_TEST = '1'
$env:PATH = "$( (Resolve-Path build/offirst-jpeg-windows).Path );$env:LOCALAPPDATA\Pub\Cache\hosted\pub.dev\onnxruntime-1.4.1\windows;$env:PATH"
flutter pub get --offline
dart format lib test
dart format --output=none --set-exit-if-changed lib test
flutter analyze --no-pub
flutter test --no-pub --reporter expanded
python tools/check_preprocessing_parity.py compare build/preprocessing-re-audit assets/offirst_dr_model.onnx
flutter build apk --release --no-pub --target-platform android-arm64,android-arm --split-per-abi
```

Fixture copies, generated tensors, and the per-image report stay in the ignored build directory. Keep any additional cleared inputs private.

## Audit build artifacts

Re-audit results for the commands above:

| Command or check | Result |
| --- | --- |
| Host CMake configure and build | Both exited 0 |
| Fixture preparation | Exit 0, generated 24 fixtures |
| `flutter pub get --offline` | Exit 0 |
| `dart format --output=none --set-exit-if-changed lib test` | Exit 0, 13 files checked, 0 changed |
| `flutter analyze --no-pub` | Exit 0, no issues |
| `flutter test --no-pub --reporter expanded` | Exit 0, 10 passed, 0 failed, 0 skipped |
| `python tools/check_preprocessing_parity.py compare build/preprocessing-re-audit assets/offirst_dr_model.onnx` | Exit 0, zero tensor and probability difference, zero threshold disagreements |
| Split Android release build | Exit 0, both APKs built, Gradle task took 45.6 s |
| APK size, signature, model and ABI inspection | Both APKs below 50,000,000 bytes. Android SDK `apksigner verify` passed. Both contain only the expected ONNX model, with the original SHA-256 above. `llvm-readelf` confirmed the AArch64 and ARM libraries and expected exported symbols in their respective splits. |

| APK under `build/app/outputs/flutter-apk/` | Bytes | SHA-256 |
| --- | ---: | --- |
| `app-arm64-v8a-release.apk` | 24,420,074 | `e361aa38fefa33cead25477139e768fcf2d230158f8f83907090cd08271e5bc1` |
| `app-armeabi-v7a-release.apk` | 23,524,052 | `39092e21a88c7a2e6a695aa82cb007346466a8ea466b6dd0431c77f7e62ba4fa` |

The ARM64 split is 114,802 bytes larger than the v1.0 pre-release. These are local audit builds, signed with the prototype's debug certificate. The hashes in the original [Android demo handoff](ANDROID_DEMO.md) refer to the earlier published artifact. The original ONNX Runtime remains 4 KiB aligned, and compatibility on 16 KiB Android devices has not been established.

## Persistence and platform verification

Visit copies go into a private staging directory. The app moves the complete image set into place before atomically replacing the metadata file. If a copy or metadata write throws, it removes the new visit's files while leaving earlier visits and metadata intact. Tests cover a missing second source image, a blocked metadata write, and a successful retry. Abrupt process termination or power loss between the directory rename and metadata commit can still require cleanup. The host widget test validates bilateral PDF action payloads. No Android device was attached to run the native PDF renderer, picker or on-device inference.
