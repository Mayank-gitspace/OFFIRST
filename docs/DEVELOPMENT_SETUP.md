# Development setup

## Requirements

- Flutter 3.41.9 and the Dart SDK supplied with it (Dart constraint: `>=3.0.0 <4.0.0`)
- JDK 17
- Android Studio / Android SDK and NDK versions selected by the Flutter project
- CMake 3.22.1 and Ninja for the JPEG native shim
- Windows PowerShell for the audited release procedure

Flutter package versions are pinned in `deployment/offline_dr_screening/pubspec.yaml`; `pubspec.lock` records the resolved graph. Android Gradle wrapper and plugin versions are checked into the app. Avoid upgrading the ONNX Runtime plugin/runtime without rechecking native library and operator compatibility.

## Build and inspect

From repository root:

```powershell
cd deployment/offline_dr_screening
flutter pub get
dart format --output=none --set-exit-if-changed lib test
flutter analyze
flutter test
flutter build apk --release --no-pub --target-platform android-arm64,android-arm --split-per-abi
```

APK outputs are written under `build/app/outputs/flutter-apk/`. Generated files, local SDK paths and credentials are ignored. The release configuration currently uses a debug signing key; do not describe these outputs as production signed.

## Optional preprocessing parity audit

The full Windows audit also needs Python 3.12 with Pillow 11.3.0, NumPy 2.2.6 and ONNX Runtime 1.26.0, plus CMake, Ninja, a C compiler, and populated Flutter dependency caches. Follow the exact fixture preparation, environment variables and compare commands in [PREPROCESSING_AUDIT.md](../deployment/offline_dr_screening/PREPROCESSING_AUDIT.md). This fixture audit measures agreement with the reference preprocessing, not clinical accuracy.

Do not put patient images, private dataset paths, Kaggle credentials, signing keys or local build outputs in commits. Use synthetic demo profiles and de-identified images.
