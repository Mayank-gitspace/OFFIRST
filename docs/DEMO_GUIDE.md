# OFFIRST live demonstration guide

> **Historical script.** Use the [1.0.5 APK](../README.md) for a current demonstration. That build shows DR severity and multi-disease grades, not only a binary probability.

Use a test profile, a synthetic patient, and de-identified retinal images already stored on the Android device. Do not use real patient records or upload medical images to external services.

1. Launch **OFFIRST** and open the patient directory.
2. Register a demonstration patient and confirm its stable patient ID.
3. Start a new screening visit and add one image for OD and one for OS.
4. Show that each image has its own experimental DR probability and threshold outcome; the app does not calculate a combined patient score.
5. Save the visit and reopen it from patient history.
6. Generate and open the PDF report. Show the per-image model metadata and the experimental-use disclaimer.
7. If demonstrating offline behavior, put the de-identified images on device storage first, switch to airplane mode, and repeat inference. The Android device workflow was not verified during the final publication check, so describe any live run as a demonstration rather than prior validation.

The deployed threshold is `0.3675977844516245`. Do not alter, round, or re-label the model output. A score below threshold is not a confirmed negative diagnosis. Avoid claiming clinical validation, patient-independent performance, or a fused patient-level result.
