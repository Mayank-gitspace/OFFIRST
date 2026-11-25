# Development setup

## Requirements

- Flutter 3.41.9 and its Dart SDK (Dart constraint: `>=3.0.0 <4.0.0`)
- JDK 17
- Android Studio / Android SDK and the NDK selected by the Flutter project
- CMake 3.22.1 and Ninja for the v1.1 rollback JPEG shim
- Windows PowerShell for the deterministic candidate build script

Flutter package versions are pinned in `deployment/offline_dr_screening/pubspec.yaml` and `pubspec.lock`. Android Gradle wrapper and plugin versions are checked into the app. Avoid upgrading ONNX Runtime or OpenCV without rechecking model contract, native compatibility, and parity.

## Build and test

The checked-in v2.1 and v1.1 ONNX assets are verified by `scripts/build_v2_1_final_candidate.ps1`. The normal release build selects v2.1 without a Dart define. From the repository root:

```powershell
cd deployment/offline_dr_screening
flutter pub get
dart format --output=none --set-exit-if-changed lib test
flutter analyze
flutter test
./scripts/build_v2_1_final_candidate.ps1
```

The script verifies both frozen model hashes and the v2.1 threshold, builds version 1.0.3 / code 2003 with ARM64 and ARMv7 support, inspects the package and signer, and writes an APK and ONNX checksum manifest. It uses the Android debug signing key available in the build environment. A clean clone without the original prototype signing key can pass `-AllowDifferentSigner` for a locally signed test artifact; that artifact cannot upgrade an install signed with the prototype key.

For the v1.1 rollback build or its test configuration, set `--dart-define=OFFIRST_EXPERIMENTAL_V21=false`. Do not use the rollback define for the primary v2.1 release.

APK outputs are written under `deployment/offline_dr_screening/build/app/outputs/flutter-apk/`. Generated outputs and local SDK paths are ignored. Do not commit patient images, credentials, restricted datasets, or signing keys.

## Verification notes

The v2.1 Flutter suite recorded 16 passed and 2 skipped; the explicit v1.1 rollback suite recorded 17 passed and 1 skipped. The skipped checks cover the v2.1 Android MethodChannel preprocessing path and unavailable preprocessing fixtures. Host ONNX tests and Python/OpenCV replay are not a substitute for Android device tests. See the [final release handoff](../deployment/offline_dr_screening/FINAL_RELEASE_HANDOFF.md) and [preprocessing audit](../deployment/offline_dr_screening/PREPROCESSING_AUDIT.md).
