# Known limitations

## Clinical and evaluation

- OFFIRST is an experimental academic prototype, not an approved medical device or clinical diagnosis system.
- The released v1.1 model's exploratory DeepDRiD results at cutoff 0.48 were 54.54% accuracy, 21.47% sensitivity, 98.27% specificity and AUROC 0.764 on 1,742 images from 488 patients.
- Historical training exposure and label-definition limitations are unresolved. Independent clinical validation and generalization are not established.
- Low sensitivity means false negatives are a serious concern. Do not use the output for treatment, referral or other clinical decisions.
- Experimental v2/v2.1 did not pass all acceptance gates and are not shipped.

## Device and model

- Physical Android device testing is incomplete. Historical host/emulator checks do not establish diagnostic reliability.
- ARM64 is intended for 4 KiB page-size devices. ONNX Runtime native ELF libraries have 4 KiB alignment; 16 KiB page support is unverified.
- ARMv7 APK is published, but a corresponding physical device test is not recorded.
- No mobile latency, battery, startup or peak-memory claim is made. Progressive JPEG decoding can require substantial memory.
- Camera capture needs suitable fundus optics. Ordinary phone camera imagery is not equivalent to retinal fundus acquisition.

## Data handling and platform

- Records and original images are stored in app-private files without application-level encryption, app lock, or individual record deletion.
- Exported PDFs leave app-private storage. Providers may sync selected destinations; exported copies must be managed separately.
- Legacy records can refer to expired picker paths. Abrupt termination or low storage can still need recovery/cleanup handling.
- iOS build, signing, native inference and PDF workflow are unavailable and unverified.
- Release APKs are debug signed and intended for experimental evaluation.

Use synthetic demonstration records and de-identified images only.
