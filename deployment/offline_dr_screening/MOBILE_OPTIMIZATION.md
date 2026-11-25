# Offline mobile release and model evaluation

## Android packaging

Use Flutter 3.41.9, JDK 17, and the Android SDK/NDK selected by the Flutter project. From this directory:

```powershell
flutter pub get
flutter build apk --release --no-pub --target-platform android-arm64 --analyze-size
flutter build apk --release --no-pub --target-platform android-arm64,android-arm
flutter build apk --release --no-pub --target-platform android-arm64,android-arm --split-per-abi
flutter build appbundle --release --no-pub --target-platform android-arm64,android-arm
```

Use the ABI split for a known Android device, the universal APK for ARM sideloading, and the AAB for distribution that generates device-specific APKs. The release artifacts currently use the project’s existing debug signing key. Android extracts the compressed native libraries at installation. The APK download is smaller, while its installed footprint includes extracted libraries. x86 runtimes are excluded because OFFIRST ships only ARM builds of ONNX Runtime.

| Release artifact | Bytes | MB (decimal) |
| --- | ---: | ---: |
| Baseline universal APK | 68,598,306 | 68.60 |
| Optimized universal APK | 36,207,500 | 36.21 |
| ARM64 APK | 24,305,272 | 24.31 |
| ARMv7 APK | 23,419,344 | 23.42 |
| Android App Bundle | 58,481,086 | 58.48 |

The universal APK shrank by 47.2%. The ONNX asset contributes 11.03 MB compressed, the two ONNX Runtime libraries 9.58 MB, and the two Flutter engine libraries 9.90 MB. Flutter's size analysis reported roughly 763 KiB of decompressed Dart AOT for `package:image` and 58 KiB for the Dart ONNX bindings. The AAB includes over 21 MB of native symbol metadata for distribution tooling, so its byte count is not the download for a device.

If R8 flags missing picker classes after an interrupted build, run `.\gradlew.bat :image_picker_android:clean` from the Android directory, then rebuild. The baseline retry used `$env:GRADLE_OPTS='-Dorg.gradle.caching=false'` and kept R8 enabled.

The bundled ARM64 ONNX Runtime 1.15.1 has 4 KiB ELF segment alignment, so native 16 KiB page compatibility is unverified. Library compression does not change ELF alignment. Android builds have been evaluated here. iOS remains a platform goal and needs a runner, device verification, and an iOS bridge or cross-platform implementation for PDF actions.

## TFLite promotion

Production releases currently include the original ONNX with the 0.48 threshold and the existing Dart preprocessing. Keep preview decode size separate from model input. A TFLite release should select a single inference dependency and asset at build time and be audited for the absence of the other runtime.

Create an isolated converter environment using [onnx2tf](https://github.com/PINTO0309/onnx2tf) and [LiteRT quantization](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_quantization). Keep candidate files, calibration arrays, and dataset manifests outside packaged assets and version control. Use a calibration CSV with `image,patient_id` columns and a disjoint held-out evaluation CSV with `image,patient_id,label`, where label is 0 or 1. Paths can be relative to the manifest. Sample calibration images from the intended device and acquisition distribution, with no held-out patient overlap.

```powershell
Copy-Item assets/offirst_dr_model.onnx build/offirst_dr_model.onnx
onnx2tf -i build/offirst_dr_model.onnx -o build/candidates-float -nuo -nuonag
python tools/evaluate_mobile_models.py calibrate C:/private/calibration.csv C:/private/calibration.npy
Copy-Item assets/offirst_dr_model.onnx build/offirst_dr_model.onnx -Force
onnx2tf -i build/offirst_dr_model.onnx -o build/candidates-int8 -nuo -nuonag -oiqt -qt per-channel -cind fundus_image C:/private/calibration.npy "[[[[0,0,0]]]]" "[[[[1,1,1]]]]" -iqd int8 -oqd int8
python tools/evaluate_mobile_models.py compare assets/offirst_dr_model.onnx C:/private/held-out.csv build/candidates-float/offirst_dr_model_float16.tflite build/candidates-int8/offirst_dr_model_full_integer_quant.tflite --calibration C:/private/calibration.csv
```

Use a disposable ONNX copy for conversion. onnx2tf can rewrite the input graph in place when it generates operation names. For a conversion plumbing check, run `python tools/evaluate_mobile_models.py smoke assets/offirst_dr_model.onnx <candidate.tflite>`. The three synthetic inputs are unsuitable for calibration or model quality estimates.

The calibration command writes normalized NHWC float32 samples. The unity transform in the converter recipe preserves this normalization. Check candidate filenames and tensor metadata for the installed converter version. The comparison reads one image at a time, feeds a shared Pillow reference tensor to both runtimes, and reports model bytes, absolute probability errors, decision crossings by direction, accuracy, sensitivity, and specificity. Before switching runtimes, also compare with the actual Flutter preprocessing and run the full workflow on each target platform. The evaluator does not approve candidates automatically.

### Conversion results

The onnx2tf 2.6.9 direct converter produced an 11,915,112-byte float32 candidate and a 5,994,936-byte float16 candidate. On three synthetic inputs, float32 had a maximum probability difference of 0.000000621 and zero crossings at 0.48. Float16 failed on the Windows CPU with LiteRT 2.1.2 at `CONV_2D` and with LiteRT 2.4.0 at `RELU_0_TO_1`. The float16 candidate exceeds the 5 MB target. The supplied package does not contain representative calibration images or a labeled holdout set, so no calibrated int8 model or labeled accuracy, sensitivity, or specificity comparison was produced. ONNX remains the sole release runtime.

## Device measurements

On a representative mid-range ARM64 phone, record the model, Android version, ABI, and power and thermal conditions. Disable network access. Use the same permitted, non-identifying input images across variants. In a timing build, pass `--dart-define=OFFIRST_PROFILE_INFERENCE=true`. Read `OFFIRST_INFERENCE` from `adb logcat`. Its elapsed time includes file read, preprocessing and inference, with the first call also including ONNX session initialization. Record cold latency separately, then collect median and p95 over repeated warm calls. App startup time can be collected with `adb shell am start -S -W -n org.offirst.offline_dr_screening/.MainActivity`. Take an idle startup PSS sample using `adb shell dumpsys meminfo org.offirst.offline_dr_screening`; collect peak process memory with a trace during screening and record the metric and sampling interval.

In airplane mode, create a test profile, add two OD images and one OS image, finalize, restart, then verify history, both eye summaries and original images. Save a PDF, verify each image page, reopen it in a PDF viewer, and export it locally. Repeat after force-stopping the process. Compare cold and warm timing and inspect peak memory separately when generating a multi-image report. Android’s PDF writer loads sampled images sequentially but retains the document until it writes the file.

For the host-only inference smoke test on Windows, put the plugin’s shipped `onnxruntime.dll` directory on PATH, set `$env:OFFIRST_NATIVE_TEST='1'`, and run `flutter test --no-pub`. The smoke test uses a synthetic image to compare direct and background inference at the same threshold. It does not measure device latency or clinical performance.
