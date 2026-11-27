# Known limitations

> **Scope.** The numeric rows below are the earlier binary classifier’s development assessment. They are not measurements of the current 1.0.5 folded linear head. The current contract and its open evidence gaps are on the [front page](../README.md).

## Model and intended use

- OFFIRST is an experimental academic prototype, not a certified medical device or diagnosis system.
- The deployed v2.1 DR classifier's development results were IDRiD (72 images: 79.17% accuracy, 87.23% sensitivity, 64.00% specificity) and RFMiD (69 images: 81.16% accuracy, 54.55% sensitivity, 86.21% specificity). It did not meet the combined 85% accuracy, 85% sensitivity, and 80% specificity working gates on either dataset.
- Patient independence is unverified, prior dataset exposure exists, and dataset label protocols differ. Clinical generalization has not been established.
- RFMiD sensitivity of 54.55% is a substantial false-negative limitation. A score below threshold does not rule out disease. Do not use outputs alone for diagnosis, referral, treatment, or other clinical decisions.
- The model performs binary DR screening only. It does not classify other diseases, estimate a disease grade, detect lesions, or produce a patient-level result from multiple images.

## Android verification and compatibility

- The candidate APK passed package, model asset, version, ABI, signer, and permission checks, but no Android device or emulator was connected for the final publication verification. Installation, startup, Android preprocessing parity, airplane-mode workflow, data-preserving upgrade, and actual PDF viewing/export remain unverified.
- The APK includes ARM64 and ARMv7 libraries and requires Android API 24 or newer. 16 KiB page-size device compatibility is not verified.
- No Android inference latency, battery, startup, or peak-memory benchmark is claimed.
- A phone camera without suitable retinal imaging optics cannot produce a fundus photograph.

## Data handling

- Patient metadata, original images, expert records, and reports are stored in app-private files without application-level encryption or an app lock.
- Exported PDFs can contain patient details and retinal photographs and leave the app sandbox. The external destination may manage its own copies.
- Use synthetic demonstration records and de-identified images. The application does not authenticate clinicians or verify expert-directory entries; schedules are locally recorded and not live availability.

## Distribution

- The prototype APK uses a debug signing certificate. It is suitable for experimental demonstration, not production distribution.
- The previous v1.1 APK and ONNX release remain available as rollback artifacts. Each saved image records its model and threshold so historical interpretation can be preserved.
