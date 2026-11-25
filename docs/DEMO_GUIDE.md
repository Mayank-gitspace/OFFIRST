# Demo guide

## Recording status

**Demo video: unavailable.** The publication environment had no running emulator or connected Android device, and no existing real application recording was located. No fabricated video link is provided.

**Application screenshots: unavailable.** Capture real screens on a device or emulator with a fictitious patient and de-identified images before adding them to the repository. Do not alter model outputs. Review screenshots for names, dates, identifiers, image metadata and PDF contents.

## Suggested real-device recording sequence

1. Show the app launch and state that inference is local.
2. Register or select a fictitious demonstration patient.
3. Start a visit and add right-eye and left-eye de-identified images from local storage.
4. Show each experimental probability and the 0.48 cutoff without modifying the result.
5. Finalize the visit, reopen it from patient history, and generate/view/export the PDF.
6. If demonstrating offline use, show airplane mode and Wi-Fi disabled before inference. Avoid implying clinical validation.

Use a 4 KiB ARM64 Android device for the currently documented release scope. Device testing on real hardware remains pending.
