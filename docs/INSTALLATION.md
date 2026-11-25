# Installation

## Android user installation

1. Download the [recommended ARM64 APK](https://github.com/Mayank-gitspace/OFFIRST/releases/download/offirst-android-prototype-v1.1/app-arm64-v8a-release.apk). The [ARMv7 APK](https://github.com/Mayank-gitspace/OFFIRST/releases/download/offirst-android-prototype-v1.1/app-armeabi-v7a-release.apk) is available for 32-bit ARM devices but has not been device-tested.
2. On Android 7.0 or newer, open the APK from Downloads and allow installation from that browser/file manager if prompted.
3. Launch **OFFIRST DR Screening**. No account or network connection is needed for local inference.
4. Register a fictitious patient, start a visit, choose an eye, and select de-identified retinal photographs from local storage. Switch eyes to complete a bilateral visit.
5. Review results and save the visit. Open the patient visit history to reopen the visit and generate a PDF. Use a local destination for export.

The ARM64 release is intended for 4 KiB page-size devices. 16 KiB support is unverified. The APKs use a debug certificate and are experimental research builds. Exported PDFs can include patient details and images; use synthetic data only.

## Permissions and troubleshooting

The app uses Android's system picker for gallery/camera selection and Android's document picker/FileProvider flow for opening and exporting reports. If an image provider is unavailable offline, place a de-identified image on device storage and use the Files picker. A phone camera by itself is not a retinal fundus camera; appropriate acquisition hardware is required for retinal photos.

If installation reports a signature conflict, preserve existing app data and install on a separate test device/profile. If a report cannot open, install an offline PDF viewer before disconnecting. Local inference requires an image file to be available on the device; the app itself does not call a cloud service.

## iOS status

No distributable iOS app exists. A future build requires an Apple build/signing environment, iOS runtime and PDF-flow implementation, signing and provisioning, and testing on supported iOS devices. Do not use Android APK links for iOS.

## Developer build

See [development setup](DEVELOPMENT_SETUP.md) for Flutter, Android SDK, dependencies, analysis and build commands.
