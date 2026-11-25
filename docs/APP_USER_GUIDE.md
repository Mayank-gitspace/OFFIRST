# App user guide

## Start a synthetic demonstration

Use a test phone/profile, the ARM64 v1.1 APK, and de-identified fundus images stored locally. The app has no cloud inference; image availability through a picker is controlled by Android and its document providers.

1. Open **OFFIRST DR Screening** and register a fictitious patient such as “Demo Patient”.
2. Start a screening visit and select the patient.
3. Choose **Right eye (OD)** and add one or more retinal images. Switch to **Left eye (OS)** and add its images.
4. Review each image result. Each image gets its own score; scores are not combined into one patient-level result.
5. Finalize and save the visit. Open Patients and select the saved visit to review both eyes and original image copies.
6. Generate a PDF locally. View it with an installed PDF reader or export it to a local folder.
7. Close and reopen the app to inspect the saved visit.

The output is experimental DR screening only. It can miss disease and must not guide clinical decisions. A phone camera without suitable retinal imaging optics cannot produce a fundus photograph.

## Data and reports

Patient metadata and visit files are stored in the app's private storage. The prototype does not add application-level encryption. Exported PDFs may include patient details and retinal photographs and can leave app-private storage. Use fictitious records and de-identified images. Removing an exported file is a separate action from clearing app data.

## Offline workflow

Inference and PDF generation execute locally. For a fully offline demonstration, store images on the phone and install a PDF viewer before switching to airplane mode. Some Android picker providers may themselves require connectivity.
