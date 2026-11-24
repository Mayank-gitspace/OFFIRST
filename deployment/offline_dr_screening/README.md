# OFFIRST Offline DR Screening (Android)

Flutter Android prototype for offline bilateral screening using the supplied ONNX export. The application lives under this directory.

## Offline workflow

1. Register a patient on Home or Patients.
2. Open Screening and choose the patient. Select OD or OS before adding photos from the gallery or camera. Switch eyes and add more images while the visit is open. A phone camera requires a suitable fundus acquisition setup.
3. Review or remove individual results, then finalize the visit. The app copies the original image bytes to its private storage and saves the score, threshold, model version, and eye for each image. There is no score aggregation.
4. Open Patients and select a saved visit to see both eyes and reopen the originals. Save PDF generates the report locally. View PDF opens an installed Android PDF viewer with read access to that report. Export PDF uses Android's document picker so the user chooses the destination.

Patient and visit metadata lives in an app-private JSON file under application support storage. New originals are stored separately by visit and eye, and PDFs live in an app-private reports folder. Android app backup is disabled. This prototype relies on the OS sandbox and device protection for local records. It does not add application-level encryption. Exported PDFs may contain patient details and retinal images. Legacy records retain their scores, but photos referenced by old picker paths may have expired. Unfinalized visits are held in memory.

## Model contract

- Input: `fundus_image`, float32 NCHW `[1, 3, 512, 512]`.
- Output: `dr_probability`, float32 probability (the ONNX graph applies `Sigmoid` after the classifier `Gemm`), not a logit.
- Preprocessing: decode as RGB, bilinear resize to exactly 512×512, convert each channel to float32 in `[0,1]`, normalize with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`, then arrange as NCHW.
- Decision: `DR_DETECTED` when probability `>= 0.48`; otherwise `DR_NOT_DETECTED`.

The reference `inference.py` uses Pillow's `Image.Resampling.BILINEAR`. This app uses the `image` package's linear bilinear resampler on decoded 8-bit RGB pixels before normalization. Both use RGB, 512×512, `[0,1]` scaling and the same normalization constants. Different image decoders/resamplers can vary slightly at pixel boundaries; this Dart pipeline has not yet been numerically compared against Pillow on a shared mobile test corpus.

## ONNX Runtime choice and limits

The app pins the Flutter `onnxruntime` plugin at 1.4.1. It bundles ONNX Runtime 1.15.1 native Android shared libraries for `arm64-v8a` and `armeabi-v7a`; inference uses the default CPU execution provider. The inspected graph consists of standard ONNX operators including Conv, Relu, HardSwish, HardSigmoid, GlobalAveragePool, Flatten, Gemm, Add, Mul, and Sigmoid. The bundled CPU runtime is intended to execute these operators without a network connection.

The app restricts packaged native ABIs to 64-bit and 32-bit ARM. It does not include x86/x86_64 native libraries. No Android device execution or latency measurement has been performed. The 36.8 ms mean in the supplied validation report was measured on CUDA and is not a mobile performance result. The plugin/runtime version is old; upgrading it requires rechecking its Android native library and operator compatibility with this exact model.

The archive manifest lists a missing `deployment_metadata.json` (1177 bytes, SHA-256 `41198ef63864aee5fa486663c57114e36fb4841316338d619fc23618d6819613`). The archive actually contains `model_metadata.json` (614 bytes, SHA-256 `ee0e06802dfa6e8174449a1bb6cda0fd79a495ffa54e117c53266dbdd3fd74ab`). All other manifest entries, including both model artifacts, matched their listed size and checksum. The unused TorchScript export and reference Python script were not copied into this app.

## Model checksum

`assets/offirst_dr_model.onnx`

- Bytes: `11,885,729`
- SHA-256: `b8fc10191f69be4fa90b73f1dc92d08bf9542ee7e26f885767942e28292c5201dd90`

Verify on Windows PowerShell from the repository root:

```powershell
Get-FileHash deployment/offline_dr_screening/assets/offirst_dr_model.onnx -Algorithm SHA256
```

## Build (Windows)

See [Offline mobile release and model evaluation](MOBILE_OPTIMIZATION.md) for measured release sizes, ABI builds, the candidate conversion workflow, and the Android benchmark procedure.
See [Android prototype handoff](ANDROID_DEMO.md) for verified artifact hashes, installation, and the offline demonstration.

Install Flutter, Android Studio/Android SDK Platform 36 (or newer compatible with the installed Flutter template), and a JDK supported by the installed Flutter Android Gradle Plugin. From the repository root:

For a fresh checkout, copy the original `OFFIRST_deployment_package/offirst_dr_model.onnx` from the supplied deployment ZIP to `deployment/offline_dr_screening/assets/offirst_dr_model.onnx`. Create the assets directory if needed. Verify the checksum above before building. The model is provisioned separately from Git. The checked-in Gradle 8.14 wrapper pins the distribution checksum.

```powershell
cd deployment/offline_dr_screening
flutter pub get
dart format lib test
flutter analyze
flutter test
flutter build apk --debug --no-pub --target-platform android-arm64,android-arm
flutter build apk --release --target-platform android-arm64,android-arm
```

For a connected ARM Android device, enable USB debugging and run:

```powershell
flutter devices
flutter run --release
```

The app makes no network requests. Flutter debug builds include internet permission for debugging. Android's pickers may display cloud providers, but the model and local report generation run without an internet connection once image bytes are available.

## Intended use

The displayed result is experimental DR screening only. The model is not clinically validated and the result is not a diagnosis. Do not use it to make treatment decisions. Performance and usability on actual Android hardware remain to be measured.
