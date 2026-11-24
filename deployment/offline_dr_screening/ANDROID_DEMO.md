# OFFIRST Android prototype handoff

## Verified artifact

- File: `build/app/outputs/flutter-apk/app-arm64-v8a-release.apk` from the app directory.
- Size: 24,305,272 bytes (24.31 MB). Package: `org.offirst.offline_dr_screening`. Version: 1.0.0, code 2001. Minimum Android API: 24. Target API: 36.
- APK SHA-256: `95ef8a3e73b908984c046137cecb0dafab0f5a3638806000f248e8b256456588`
- Packaged ONNX SHA-256: `b8fc10191f69faef16f1dc92d08bf9542ee7e26f885767942e28292c5201dd90`
- Packaged ARM64 ONNX Runtime SHA-256: `b037739b394517a03c45c2138737504d73594c2deea1cf08f4521b9dec5cc78`
- The APK verified with Android SDK apksigner. The prototype is a release build signed with an Android debug certificate. Distribution requires an approved signing key and device compatibility testing.

## Install over USB

Use an ARM64 Android phone with USB debugging enabled and approve the computer on the phone. Use a 4 KiB page size device for the initial demonstration. The bundled ONNX Runtime ELF segments use 4 KiB alignment, so a 16 KiB device needs separate compatibility testing.

Run from the repository root in PowerShell. If the Android SDK lives elsewhere, change the ADB path.

```powershell
$adb = Join-Path $env:LOCALAPPDATA 'Android\Sdk\platform-tools\adb.exe'
$apk = (Resolve-Path 'deployment/offline_dr_screening/build/app/outputs/flutter-apk/app-arm64-v8a-release.apk').Path
Get-FileHash -Algorithm SHA256 $apk
& $adb devices -l
& $adb shell getprop ro.product.cpu.abilist
& $adb shell getconf PAGE_SIZE
& $adb install -r $apk
& $adb shell am start -W -n org.offirst.offline_dr_screening/.MainActivity
```

Compare the hash above, confirm the device reports `arm64-v8a`, and expect `Success` from installation and `Status: ok` from launch. If multiple devices are listed, use `-s SERIAL` in each ADB command. A signature or version conflict can occur with a previous install. Use a clean demo device or an Android test profile to preserve existing app data.

## Demonstrate offline

1. Put three cleared, de-identified sample fundus images in a local folder on the phone and install an offline PDF viewer before disconnecting. Use a fictitious profile such as **Demo Patient**.
2. Enable airplane mode and turn Wi-Fi off. Launch **OFFIRST DR Screening**. Select **Register patient**, enter the demo name, and save.
3. Select **Start screening**, choose the demo patient and **Right eye (OD)**, then use **Add fundus images** to select two local files. Select **Left eye (OS)** and add the third file. An ordinary phone camera alone cannot acquire a fundus image. Use the gallery for this demo.
4. Confirm each image has an experimental model score and threshold result. Select **Finalize and save visit**. Open **Patients**, expand the demo profile, and open the saved visit. Verify two OD images and one OS image.
5. Select **Save PDF**, then **View PDF**. Verify the summary and each image page. Select **Export PDF**, choose a local Downloads folder, and reopen the exported file offline.
6. Force-stop and relaunch OFFIRST. Verify that the visit, images and report are still available. Describe the result as experimental DR screening. Clinical accuracy and mobile timing have not been established.

## Remaining device verification

- Check clean installation, cold launch, and Android crash logs. Exercise bad or canceled inputs, camera intent, background and resume, a large bilateral visit, low storage, restart, missing PDF viewer, and export cancellation. Verify scores against the reference with cleared fixtures. Android hardware was unavailable for this review.
- Measure startup memory, peak screening and report memory, and cold and warm inference latency on a representative mid-range ARM64 phone. Record the phone, Android version, page size and artifact. See [MOBILE_OPTIMIZATION.md](MOBILE_OPTIMIZATION.md). Test 16 KiB compatibility separately.
- Check camera/gallery selection and PDF actions in airplane mode. iOS compatibility remains unverified.

## Storage and sharing audit

Patient identifiers, scores, original images, and reports are stored in Android app-private files. Records are plain JSON within that sandbox. Android backup is disabled in the manifest. There is no app lock, application-level encryption, or individual record deletion. Use fictitious profiles and de-identified images for this prototype.

The PDF viewer receives read access to a single app-private report via a non-exported FileProvider. Export moves a copy outside OFFIRST’s control. A document provider may sync it, and clearing OFFIRST data does not delete an exported copy. Camera files, gallery source files, and metadata can also survive independently. Choose local destinations for the demo and remove exports separately.

Failed or interrupted saves can leave copied images or staging files in private storage because image copies and metadata updates are separate operations. Legacy records may reference expired picker paths. Test interruption and cleanup before handling sensitive data.
