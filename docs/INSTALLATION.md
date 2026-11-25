# Android installation

## Install the application

Download the [OFFIRST v2.1 universal APK](https://github.com/Mayank-gitspace/OFFIRST/releases/download/offirst-v2.1/OFFIRST-v2.1-final-candidate.apk). It contains ARM64 (`arm64-v8a`) and ARMv7 (`armeabi-v7a`) native libraries and requires Android 7.0 / API 24 or newer. Verify the [SHA-256 checksum manifest](https://github.com/Mayank-gitspace/OFFIRST/releases/download/offirst-v2.1/OFFIRST-v2.1-final-candidate.sha256) before installing. The package ID is `org.offirst.offline_dr_screening`; the app label is **OFFIRST**.

1. Download the APK to the Android device.
2. Open it from Downloads or a file manager. If Android asks, allow that app to install this package.
3. Launch **OFFIRST**.
4. Register or select a synthetic demonstration patient, create a visit, and add locally available de-identified images for OD and/or OS.
5. Review each image's separate result, save the visit, and open or export the PDF report.

The v2.1 model and report generation run on-device; they do not require cloud inference. Android image providers may need connectivity to make a selected file available. Pre-install a PDF viewer if the device has none and test report export using synthetic data.

The APK is a debug-signed research prototype. A physical-device installation, existing-data upgrade, offline inference, and report-open/export workflow were not verified in this release check. Do not uninstall an existing OFFIRST installation or clear application data for upgrade tests. A signer mismatch means this APK cannot update that installation; use a separate test device/profile and preserve the original data.

## Build from source

See [development setup](DEVELOPMENT_SETUP.md). The normal Android release configuration selects v2.1 without a build flag. Use `--dart-define=OFFIRST_EXPERIMENTAL_V21=false` only when deliberately producing a v1.1 rollback build.
