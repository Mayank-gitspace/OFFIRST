# OFFIRST application redesign candidate

**Status:** Local feature-branch candidate. Not merged and not released. Physical-device verification is still required before distribution.

## Implemented in this candidate

- Centralized OFFIRST color and component theme, responsive dashboard, local patient/visit/image counts, recent visits, on-device inference indicator, About and privacy information.
- Patient registration, editing, name/ID search, persistent OFF IDs, ID copy, visit history, and confirmed patient deletion. Deletion removes the patient record, its private visit folders, and app-managed reports for those visits; associated expert records are independent.
- Local expert directory with stable IDs, add/edit/search/delete, locally entered details and schedules, last-updated date, patient associations, associated-patient listing, and explicit unverified/availability wording.
- Version-tolerant patient loading: legacy records without expert associations load with an empty association list. Patient IDs, visit IDs, image IDs, scores, and threshold are preserved when editing patient metadata.
- OFFIRST launcher name and original adaptive/legacy cobalt lens icon.
- PDF report header, stable report identifier, patient and visit IDs, app/model and threshold metadata, per-eye image counts, and explicit no-assessment text for an eye with no saved image.

The current report protocol still does not record anatomical absence, an eye-not-screened status, or an unable-to-assess status. A missing image is described only as “no model assessment performed.” The report content has not been visually validated on a physical Android device; a synthetic sample PDF was not generated in this run.

## Data and model contract

Patient records remain in app-private `offirst_patients.json`; local expert records are in app-private `offirst_experts.json`; images and generated reports remain in app-private directories. These JSON records are not application-level encrypted. There is no login or role-based access. Expert information is user-entered and unverified, and an association does not indicate review or diagnosis.

The package identity remains `org.offirst.offline_dr_screening`. The packaged ONNX SHA-256 remains `b8fc10191f69faef16f1dc92d08bf9542ee7e26f885767942e28292c5201dd90`; the inference threshold remains `0.48`; the model identifier remains `offirst-dr-onnx-b8fc1019`. No weights or preprocessing were modified.

## Verification run

- `flutter analyze --no-pub`: passed, no issues.
- `flutter test --no-pub` with local ONNX/JPEG runtime libraries and `OFFIRST_PARITY_DIR=build/v11-readiness`: **16 passed, 0 skipped**. This includes native ONNX inference, tensor-contract checks, all 24 preprocessing goldens, storage compatibility/edit/delete and expert-association coverage, PDF method-channel payload tests, and navigation smoke coverage.
- `flutter build apk --release --no-pub --build-number 2002 --build-name 1.0.1 --target-platform android-arm64,android-arm --split-per-abi`: passed.
- APK metadata reports app label OFFIRST, package identity unchanged, version name `1.0.1`, ARM64 version code `4002`, ARMv7 version code `3002`, and debug signing. Both APKs contain only their intended ABI and the unchanged model hash. Both passed `apksigner verify` and ZIP alignment checks.
- ADB reported no attached emulator or phone. Upgrade/data-retention, on-device inference, navigation, launcher appearance, APK installation, and PDF viewer/export behavior were not device-tested. The debug signing key matches the local previous debug candidate; no production signing key is present.

## Candidate artifact hashes

Generated locally under `build/app/outputs/flutter-apk/` and ignored by Git:

| ABI | APK | SHA-256 |
|---|---|---|
| ARM64 | `app-arm64-v8a-release.apk` | `0db969437e4a5b2f33f73517b8e16ffe8e12d5c8d17bf77934c077b362556131` |
| ARMv7 | `app-armeabi-v7a-release.apk` | `998c22f131b69bcbd124249dc3fecbc37bee9299753c42c8ce962dda1e78dc20` |

The APKs are verification candidates only. They have not been uploaded as release assets, and the existing published APKs/releases and root README were not changed.
