# Screenshots

## Current application

No screenshot of OFFIRST 1.0.5 is checked into this repository.

Tracked images under `deployment/offline_dr_screening/android/app/src/main/res` are launcher icons. Plots under `encoder_byol`, `encoder_mnv3_ssl`, `kd_run2`, `kd_imaging`, and `encoder_baseline` are research figures. They are not the screening UI.

`screen.jpeg` in the working tree is untracked. It shows an older dialog, including a Research Demo tab that the 1.0.5 app does not use. Do not publish it as the current interface.

## What a device pass should capture

Use a synthetic patient and a de-identified retinal photograph. Capture only after the published 1.0.5 APK is the installed package (`versionName` 1.0.5, version code 4005).

1. Home, with the five tabs visible: Home, Patients, Screening, Experts, About.
2. Patient list, using a synthetic name.
3. Screening mode control, on DR severity.
4. A retinal photograph that the app accepts, with the four-class grade visible.
5. The same visit on Multi-disease, with Any Disease Risk visible and Media Haze labeled as Media Haze.
6. The rejection message for an ordinary photograph: “This image does not look like a fundus photograph. Choose a retinal image.”
7. Visit page and the PDF actions.
8. Page 1 and page 2 of `OFFIRST-{visit}-v6.pdf`.

Store accepted captures under `docs/screenshots/` only after a person checks them for names, identifiers, and clinical images that should not be public. This audit did not add that folder because no verified 1.0.5 capture was available.
