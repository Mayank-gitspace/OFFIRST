# Android installation

## Install OFFIRST 1.0.5

Download [OFFIRST-1.0.5-arm64-v8a.apk](https://github.com/Mayank-gitspace/OFFIRST/releases/download/offirst-1.0.5/OFFIRST-1.0.5-arm64-v8a.apk) from the [1.0.5 release](https://github.com/Mayank-gitspace/OFFIRST/releases/tag/offirst-1.0.5). Verify the [SHA-256 manifest](https://github.com/Mayank-gitspace/OFFIRST/releases/download/offirst-1.0.5/OFFIRST-1.0.5-arm64-v8a.sha256) before installing.

| Item | Value |
| --- | --- |
| Version | 1.0.5, version code 4005 |
| Package | `org.offirst.offline_dr_screening`, label **OFFIRST** |
| Android | 7.0 / API 24 or newer |
| ABI | `arm64-v8a` only |
| APK SHA-256 | `62f319ef5978a13fa645a619cdc05802b6020ebfa085ac882fd9647c0e125a2d` |
| ONNX SHA-256 | `b6bcb1182db7717c01ef24181d79a44be7d1a75ddefe8567cfec2fffb9e5c59b` |
| Binary threshold | `0.524373471736908` |

1. Download the APK to an ARM64 Android phone.
2. Open it from Downloads or a file manager. If Android asks, allow that app to install this package.
3. Launch **OFFIRST**.
4. Register or select a synthetic demonstration patient, create a visit, and add locally available de-identified images for OD and/or OS.
5. Review the DR severity result, switch to Multi-disease, save the visit, and open or export the two-page PDF.

Screening and the PDF run on the device. Android’s file picker may need connectivity to expose a selected image. Install a PDF viewer first if the phone has none.

The APK is a debug-signed research prototype. This file was installed and opened on a Redmi Note 9 Pro Max. That launch is not a clinical acceptance test. Do not clear application data. A signer mismatch means this APK cannot update that installation.

The previous universal download is [OFFIRST 1.0.3](https://github.com/Mayank-gitspace/OFFIRST/releases/download/offirst-v2.1/OFFIRST-v2.1-final-candidate.apk), version code 2003. It is an earlier model.

## Build from source

See [development setup](DEVELOPMENT_SETUP.md). The normal Android release configuration selects v2.1 without a build flag. Use `--dart-define=OFFIRST_EXPERIMENTAL_V21=false` only when deliberately producing a v1.1 rollback build.
