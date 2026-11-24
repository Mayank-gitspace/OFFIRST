# OFFIRST v1.1 release-readiness verification

This candidate builds on integration commit `2fed437018a0077da3a40adf5a3bb1eb7a8db43e` and includes the JPEG memory and visit reconciliation changes. The unchanged ONNX model SHA-256 is `b8fc10191f69faef16f1dc92d08bf9542ee7e26f885767942e28292c5201dd90`. The threshold remains `0.48`.

## Automated checks

The Windows host rebuilt the pinned JPEG shim with CMake and Ninja. `flutter pub get --offline`, the formatter check, and `flutter analyze --no-pub` all exited 0. The analyzer reported no issues.

The full test command was `flutter test --no-pub --machine`. `OFFIRST_NATIVE_TEST=1` activated native ONNX tests. `OFFIRST_PARITY_DIR` pointed to the freshly prepared `build/v11-readiness` fixture directory. The host JPEG and ONNX DLL directories were added to PATH. Use the five original fixture inputs and pinned environment in [PREPROCESSING_AUDIT.md](PREPROCESSING_AUDIT.md), substituting `build/v11-readiness` for the output directory. The test reporter recorded **13 executed, 0 skipped, 13 passed, 0 failed**:

| Test | Status |
| --- | --- |
| Reject wrong native output type, dimensions and probability values | Pass |
| Background ONNX inference matches direct inference and reuses the session | Pass |
| Remove marked, unreferenced transactions at startup | Pass |
| Preserve marked images when metadata is missing or corrupt | Pass |
| Roll back a failed second image copy and preserve prior visits | Pass |
| Roll back a failed metadata write and preserve prior visits | Pass |
| Reopen bilateral visits, all visits and copied original bytes | Pass |
| Preserve legacy results and refuse writes over unreadable history | Pass |
| RGB PNG yields normalized NCHW float32 | Pass |
| Reject a file over 32 MiB before reading it for preprocessing | Pass |
| Production Dart tensors match Pillow goldens | Pass |
| Bilateral visit sends separate OD and OS data to all three PDF actions | Pass |
| Dashboard opens offline bilateral screening | Pass |

`python tools/check_preprocessing_parity.py compare build/v11-readiness assets/offirst_dr_model.onnx` exited 0. All 24 fixtures were evaluated against the unchanged ONNX with CPU ONNX Runtime 1.26.0. All 18,874,368 float32 values matched. Maximum absolute tensor error and probability error were both **0**. There were **0** threshold disagreements. This is preprocessing equivalence on the fixture set, not clinical accuracy validation. The fixture set includes five existing research previews and 19 synthetic inputs.

## JPEG memory and recovery

The first resize pass now consumes a synchronous view of native RGB and then releases the native image. There is no managed full-resolution RGB copy. A JPEG already at 512 × 512 copies only its 786,432 RGB bytes to leave the native scope. The decoder's encoded-input arena is released before the resize callback starts. Exact Pillow sampling, intermediate rounding, orientation and float32 normalization are preserved. Decoder downsampling was not used because it would change the reference sampling.

The app rejects encoded inputs above 32 MiB before reading them for inference. The JPEG decoder admits at most 24 million pixels and 12,000 pixels on either axis. It sets a 96 MiB budget for libjpeg virtual arrays. That does not cap total process memory, and non-JPEG decoders have no equivalent decoded-pixel cap. The largest permitted native JPEG RGB buffer is 72 MB, compared with up to 240 MB for the two RGB buffers under the previous 40 MP cap. Temporary resize buffers, compressed bytes, codec buffers, Dart, and ONNX memory are additional. Host native smoke checks rejected 5000 × 5000 and 12001 × 8 JPEGs and accepted 12000 × 8.

A separate Windows Dart process measured the increase in peak RSS above a small JPEG warm-up, with three runs per variant. The input was a synthetic 5000 × 4000 RGB JPEG with 4:2:0 subsampling, encoded separately as baseline and progressive. This compared the previous and current Dart preprocessing with the same host native decoder, without ONNX loaded.

| 20 MP synthetic input | Previous median peak RSS increase | Current median peak RSS increase |
| --- | ---: | ---: |
| Baseline JPEG | 97.45 MiB | 41.51 MiB |
| Progressive JPEG | 95.41 MiB | 94.13 MiB |

Progressive JPEG decoding still needs coefficient buffers and can dominate peak memory even after the copy is removed. These Windows process measurements do not predict Android peak RAM or latency.

New visit directories carry a private transaction marker through staging and directory rename. Startup reconciles marked directories only after the complete metadata file parses successfully. It keeps visits referenced by ID or original image path, including legacy absolute paths. It removes an unreferenced marked directory, or removes only the marker from a committed visit. It leaves missing or corrupt metadata, unmarked older directories, and unrelated files alone. Tests simulate interruption on both sides of directory rename, protect a committed visit and a legacy path reference, and exercise two rollback failures. Device force-stop and storage failure behavior still needs a smoke test. Older unmarked orphans require manual review.

## Release APKs and 16 KiB inspection

`flutter build apk --release --no-pub --target-platform android-arm64,android-arm --split-per-abi` exited 0. Both APKs are under `build/app/outputs/flutter-apk/` and below 50 MB. SDK `apksigner verify --print-certs` confirmed both debug certificates. Both APKs contain only the intended ABI, and both embedded ONNX files match the original hash above.

| Split | Bytes | SHA-256 |
| --- | ---: | --- |
| ARM64 | 24,429,910 | `1b9bcafa0712f9dfb46f562bd38bd404545aa3a097fbdabd6dd6fe0181a3f665` |
| ARMv7 | 23,533,088 | `d2ef19197ae6e1ab9c9b21a7d9fe09598effbc800ddf36dcd3408847f4c410e9` |

`zipalign -c -P 16 4` passed for both archives, which ship compressed native library entries. Separately, `llvm-readelf --file-header --program-headers --dynamic --wide` confirmed AArch64 and ARM ELF machine types and these load segment alignments:

| Native library | ARM64 | ARMv7 |
| --- | ---: | ---: |
| `libapp.so` | 64 KiB | 16 KiB |
| `libdartjni.so` | 16 KiB | 16 KiB |
| `libflutter.so` | 64 KiB | 64 KiB |
| `liboffirst_jpeg.so` | 16 KiB | 4 KiB |
| `libonnxruntime.so` | 4 KiB | 4 KiB |

APK ZIP alignment does not fix ELF segment alignment. The ARM64 ONNX Runtime is below the [Android 16 KiB ELF alignment requirement](https://developer.android.com/guide/practices/page-sizes#check-alignment). Neither build is verified for 16 KiB Android devices. Use 4 KiB devices for the experimental demo and treat 16 KiB support as a separate runtime upgrade and device-test task.

## Recommendation

**CONDITIONAL GO** for an experimental GitHub pre-release after a 4 KiB ARM64 device smoke test. Advertise ARMv7 runtime support only after a corresponding 32-bit device smoke test, and label the artifacts debug-signed and experimental. No Android device was attached during this verification.

- Confirm installation, launch and the offline picker, including EXIF-tagged JPEGs. Compare actual device scores on known inputs.
- Screen a large baseline and progressive JPEG on a lower-memory phone. Check graceful rejection of oversized inputs and measure startup memory, peak screening memory, and latency.
- Save and reopen a bilateral visit, force-stop during finalization in a test profile, and verify recovery against actual app storage. Generate, view, and export the native Android PDF offline.
- Verify device page size with `adb shell getconf PAGE_SIZE`. Keep 16 KiB devices outside the stated compatibility scope until the runtime libraries and full workflow pass there.

No device test was run during candidate preparation. The v1.0 release remains the previous artifact.

## ARM64 device smoke test

Use a dedicated test phone or a profile with synthetic records only. The current ARM64 APK requires a **4 KiB (4096-byte) page-size device** for this test. Android 7.0 or later, USB debugging, and an offline PDF viewer are required. The APK is **debug signed**. Native device validation remains incomplete, and the **85% accuracy target is unverified**. This is experimental software for evaluation, with no clinical accuracy claim.

From the app directory, save the script below as `build/offirst-arm64-smoke.ps1`. Use the Python environment from the automated checks (Pillow 11.3.0, NumPy 2.2.6 and ONNX Runtime 1.26.0). The script verifies the canonical candidate APK from the table above and creates synthetic images and reference scores before installing anything.

- Host-only preparation: `.\build\offirst-arm64-smoke.ps1 -PrepareOnly -Python .\build\offirst-python312\python.exe`
- Device run: enable airplane mode, turn off Wi-Fi, and authorize USB debugging, then run `.\build\offirst-arm64-smoke.ps1 -Serial YOUR_ADB_SERIAL -Python .\build\offirst-python312\python.exe`. Omit `-Serial` when exactly one authorized device is connected. Override `-SdkRoot` or `-BuildTools` if needed.

```powershell
param(
    [string]$Serial,
    [string]$Python = 'python',
    [string]$SdkRoot = (Join-Path $env:LOCALAPPDATA 'Android\Sdk'),
    [string]$BuildTools = '36.1.0',
    [switch]$PrepareOnly
)
$ErrorActionPreference = 'Stop'
$report = Get-Content RELEASE_READINESS_V1_1.md -Raw -Encoding UTF8
$apk = (Resolve-Path build/app/outputs/flutter-apk/app-arm64-v8a-release.apk).Path
$model = (Resolve-Path assets/offirst_dr_model.onnx).Path
$expectedApk = [regex]::Match($report, '(?m)^\| ARM64 \| [\d,]+ \| \x60([0-9a-f]{64})\x60 \|').Groups[1].Value
$expectedModel = 'b8fc10191f69faef16f1dc92d08bf9542ee7e26f885767942e28292c5201dd90'
$commit = git rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Could not resolve the source commit.' }
$reportCommit = git log -1 --format=%H -- RELEASE_READINESS_V1_1.md
if ($LASTEXITCODE -ne 0 -or $reportCommit -ne $commit) {
    throw 'Check out the commit containing this candidate report.'
}
$changes = @(git status --porcelain)
if ($LASTEXITCODE -ne 0 -or $changes.Count) {
    throw 'Run from a clean checkout of the candidate source.'
}
if ((Get-FileHash $apk).Hash -ne $expectedApk -or
    (Get-FileHash $model).Hash -ne $expectedModel) {
    throw 'APK or local ONNX checksum differs from the verified candidate.'
}

$certificate = @(& "$SdkRoot\build-tools\$BuildTools\apksigner.bat" verify --print-certs $apk)
if ($LASTEXITCODE -ne 0 -or -not ($certificate -match 'CN=Android Debug')) {
    throw 'Expected a verified Android debug certificate.'
}
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [IO.Compression.ZipFile]::OpenRead($apk)
try {
    $models = @($zip.Entries | Where-Object FullName -match '\.(onnx|tflite)$')
    $libraries = @($zip.Entries | Where-Object FullName -match '^lib/.+\.so$')
    if ($models.Count -ne 1 -or
        $models[0].FullName -ne 'assets/flutter_assets/assets/offirst_dr_model.onnx' -or
        @($libraries | Where-Object FullName -notlike 'lib/arm64-v8a/*').Count -or
        -not $zip.GetEntry('lib/arm64-v8a/libonnxruntime.so') -or
        -not $zip.GetEntry('lib/arm64-v8a/liboffirst_jpeg.so')) {
        throw 'Unexpected APK model or ABI payload.'
    }
    $stream = $models[0].Open()
    try { $modelHash = (Get-FileHash -InputStream $stream).Hash } finally { $stream.Dispose() }
    if ($modelHash -ne $expectedModel) { throw 'Embedded ONNX checksum mismatch.' }
} finally {
    $zip.Dispose()
}

$run = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$evidence = (New-Item -ItemType Directory -Path "build/v11-device-evidence/$run" -Force).FullName
$fixtures = (New-Item -ItemType Directory -Path (Join-Path $evidence 'synthetic')).FullName
@("commit=$commit", "apk_sha256=$expectedApk", "model_sha256=$expectedModel") |
    Set-Content -Encoding ASCII (Join-Path $evidence 'artifact.txt')
$certificate | Set-Content -Encoding UTF8 (Join-Path $evidence 'signing.txt')

@'
import hashlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

out, model = map(Path, sys.argv[1:])
reference = runpy.run_path("tools/check_preprocessing_parity.py")
source = Image.fromarray(reference["pattern"](773, 365))
source.save(out / "normal-420.jpg", quality=91, subsampling=2)
exif = Image.Exif()
exif[274] = 6
source.save(out / "exif-6.jpg", quality=90, subsampling=2, progressive=True, exif=exif)

large = Image.new("RGB", (5000, 4000), (18, 10, 8))
draw = ImageDraw.Draw(large)
for i in range(180):
    x, y = (i * 233) % 5000, (i * 137) % 4000
    draw.ellipse((x, y, x + 300, y + 120),
                 fill=((19 * i) % 256, (31 * i) % 256, (53 * i) % 256))
for name, progressive in [("baseline-20mp.jpg", False), ("progressive-20mp.jpg", True)]:
    large.save(out / name, quality=92, subsampling=2, progressive=progressive)
Image.new("L", (5000, 5000), 30).save(out / "over-pixel-limit.jpg")

session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
rows = []
for path in sorted(out.glob("*.jpg")):
    row = {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if path.name == "over-pixel-limit.jpg":
        row["expected"] = "rejected: exceeds 24 million pixels"
    else:
        with Image.open(path) as opened:
            rgb = opened.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
            normalized = (np.asarray(rgb, dtype=np.float32) / 255.0 - reference["MEAN"]) / reference["STD"]
        tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None, ...])
        p = session.run(["dr_probability"], {"fundus_image": tensor})[0].item()
        row.update(probability=p, displayed_percent=round(100 * p, 1), flagged=p >= 0.48)
    rows.append(row)
(out.parent / "expected-scores.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
'@ | & $Python - $fixtures $model
if ($LASTEXITCODE -ne 0) { throw 'Synthetic fixture preparation failed.' }
if ($PrepareOnly) {
    Write-Output "Prepared host evidence at $evidence. No device test was run."
    return
}

$adb = Join-Path $SdkRoot 'platform-tools\adb.exe'
$devices = @(& $adb devices)
if ($LASTEXITCODE -ne 0) { throw 'adb devices failed.' }
$authorized = @($devices | ForEach-Object {
    if ($_ -match '^(\S+)\s+device\s*$') { $Matches[1] }
})
if (-not $Serial -and $authorized.Count -eq 1) { $Serial = $authorized[0] }
if ($authorized -cnotcontains $Serial -or $Serial -match '^(emulator-)|:') {
    throw 'Select one authorized physical USB device with -Serial.'
}
function Invoke-Phone {
    & $adb -s $Serial @args
    if ($LASTEXITCODE -ne 0) { throw "adb command failed: $($args -join ' ')" }
}

$pageSize = (Invoke-Phone shell getconf PAGE_SIZE).Trim()
$abi = (Invoke-Phone shell getprop ro.product.cpu.abilist).Trim()
$api = (Invoke-Phone shell getprop ro.build.version.sdk).Trim()
if ($pageSize -ne '4096' -or ($abi -split ',') -cnotcontains 'arm64-v8a' -or [int]$api -lt 24) {
    throw 'This smoke test requires a 4 KiB ARM64 Android device at API 24 or newer.'
}
function Assert-Offline {
    $flight = (Invoke-Phone shell settings get global airplane_mode_on).Trim()
    $wifi = (Invoke-Phone shell settings get global wifi_on).Trim()
    # Android stores 3 when Wi-Fi was disabled by airplane mode.
    if ($flight -ne '1' -or @('0', '3') -notcontains $wifi) {
        throw 'Enable airplane mode and turn off Wi-Fi on the test device.'
    }
}
Assert-Offline
$deviceModel = (Invoke-Phone shell getprop ro.product.model).Trim()
$android = (Invoke-Phone shell getprop ro.build.version.release).Trim()
$remoteFixtures = "/sdcard/Download/OFFIRST-v11-smoke-$run"
@("model=$deviceModel", "android=$android", "api=$api", "abis=$abi",
    "page_size=$pageSize", 'network=airplane_mode_wifi_off', "fixtures=$remoteFixtures") |
    Set-Content -Encoding UTF8 (Join-Path $evidence 'device.txt')

$pkg = 'org.offirst.offline_dr_screening'
$install = @(Invoke-Phone install -r $apk)
$install | Set-Content -Encoding UTF8 (Join-Path $evidence 'install.txt')
if (-not ($install -match '^\s*Success\s*$')) { throw 'APK installation did not report success.' }
Invoke-Phone shell mkdir -p $remoteFixtures
Get-ChildItem -LiteralPath $fixtures -Filter '*.jpg' -File | ForEach-Object {
    Invoke-Phone push $_.FullName "$remoteFixtures/$($_.Name)"
}

function Start-OfflineApp($label) {
    Assert-Offline
    $launch = @(Invoke-Phone shell am start -W -S -n "$pkg/.MainActivity")
    $launch | Set-Content -Encoding UTF8 (Join-Path $evidence "$label-launch.txt")
    if (-not ($launch -match '^\s*Status:\s+ok\s*$')) { throw 'OFFIRST launch did not report success.' }
    Invoke-Phone shell dumpsys meminfo $pkg |
        Set-Content -Encoding UTF8 (Join-Path $evidence "$label-meminfo.txt")
}

Start-OfflineApp 'first'
Write-Output "Fixtures on phone: $remoteFixtures"
Write-Output "Evidence and expected scores: $evidence"
[void](Read-Host 'Run checklist rows 1-5, then press Enter to force-stop and reopen OFFIRST')
Start-OfflineApp 'restart'
[void](Read-Host 'Run checklist rows 6-8, record PASS/FAIL/NOT RUN, then press Enter')
Assert-Offline
Invoke-Phone shell dumpsys meminfo $pkg |
    Set-Content -Encoding UTF8 (Join-Path $evidence 'final-meminfo.txt')
Invoke-Phone logcat -b crash -d -t 150 |
    Select-String -SimpleMatch $pkg |
    Set-Content -Encoding UTF8 (Join-Path $evidence 'offirst-crash-matches.txt')
Write-Output 'Evidence collected. Device results require a completed checklist.'
```

### Evidence checklist

Record each row as **PASS**, **FAIL**, or **NOT RUN** with observations and a timestamp in the run directory. Keep the metadata, reference scores, and APK hash with the results. If the gallery does not list newly pushed files, try the picker’s Files or Browse view and confirm the exact filenames. Leave the picker check marked blocked if the files remain unavailable. Preserve only synthetic screenshots, PDFs, or log excerpts.

| # | Check | Evidence and pass condition |
| --- | --- | --- |
| 1 | Install and launch offline | `install.txt`, `first-launch.txt` and `device.txt`. Correct APK hash, signature, device ABI and 4096-byte page size. OFFIRST opens in airplane mode with Wi-Fi off. |
| 2 | Local picker and orientation | Register a fictitious profile. Add `normal-420.jpg` as right eye and `exif-6.jpg` as left eye in a bilateral visit. Record picker behavior and actual displayed scores and threshold sides beside the JSON reference. Require the threshold sides to match and displayed percentages within 0.1 percentage point. Preview rotation may honor EXIF while inference uses encoded orientation. |
| 3 | Save and finalize | Save the bilateral visit from row 2. Confirm both sides appear in Patients and record the image count and scores. |
| 4 | Large JPEGs | In a new draft, screen `baseline-20mp.jpg` and `progressive-20mp.jpg`. Record scores, result labels, elapsed tap-to-result time and responsiveness. A `dumpsys meminfo` snapshot is not a peak measure. Capture a memory trace during decode for a peak claim. |
| 5 | Graceful rejection | Select `over-pixel-limit.jpg`. Record the rejection message and confirm no result is added. Select a valid image afterwards and confirm screening still works. Leave this draft unsaved. |
| 6 | Reopen after restart | Compare the saved visit, sides and scores with row 3 after the script's cold restart. Confirm the completed visit remains intact. |
| 7 | Native PDF | Save, view and export the bilateral report to local Downloads. Reopen it offline and verify the summary and both eye pages. Keep only the synthetic PDF or a redacted screenshot. |
| 8 | Interrupted finalization | On the test profile, interrupt a new visit during Save with `adb -s YOUR_ADB_SERIAL shell am force-stop org.offirst.offline_dr_screening`, then reopen OFFIRST. Verify previously completed visits survive and a partial visit is not shown. Record **NOT RUN** for the precise rename-to-metadata window if it was not observed. Review crash or ANR evidence separately. An empty filtered crash log does not prove the absence of a failure. |

The script writes startup and post-run memory snapshots and checks for recent crash lines matching the app. Physical-device results are pending until a tester records the checklist. A separate ARMv7 device run is needed to claim runtime support on that ABI.
