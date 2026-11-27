# App user guide

> **Historical guide.** The install target below is the older universal APK. Reviewers of the current build should follow [Installation](INSTALLATION.md) and the [1.0.5 front page](../README.md).

## Run a synthetic demonstration

Install the v2.1 universal APK on an ARM64 or ARMv7 Android device running Android 7.0 or newer. Use a test profile, fictitious patient data, and de-identified fundus images already available on the device. The model runs locally; image availability through Android's picker depends on the chosen provider.

1. Open **OFFIRST** and register a synthetic patient, or select an existing local patient.
2. Create a screening visit.
3. Choose **Right eye (OD)** and add one or more retinal photographs. Choose **Left eye (OS)** separately if screening that eye.
4. Review each image's own experimental DR probability and threshold outcome. Image scores are not combined into a patient-level result.
5. Save the visit. Open the patient history to review the visit and its original image copies.
6. Generate the PDF report, then choose to open or export it.
7. Close and reopen the application to verify local visit history.

Ordinary phone cameras are not fundus cameras; suitable retinal imaging optics are needed to obtain retinal photographs. A below-threshold score does not rule out disease and is not a diagnosis.

## Local data and reports

Patient and expert information, visits, images, and generated reports are stored in app-private local storage. The app does not add application-level encryption. Exported PDFs can contain patient details and retinal photographs; they leave app-private storage when exported. Use synthetic records and de-identified images. Removing an exported file is separate from deleting app data.

## Offline use

Inference and PDF generation execute locally and do not call a cloud inference service. For an offline demonstration, put de-identified image files on the device and install a PDF viewer before switching to airplane mode. Some Android picker providers may require connectivity.

The v2.1 Android preprocessing and physical-device workflow were not verified on a connected device for this publication. See the [release handoff](../deployment/offline_dr_screening/FINAL_RELEASE_HANDOFF.md) for test scope.
