$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    $modelPath = Join-Path $projectRoot 'assets/offirst_v2_1_experimental.onnx'
    $v1ModelPath = Join-Path $projectRoot 'assets/offirst_dr_model.onnx'
    $servicePath = Join-Path $projectRoot 'lib/screening_service.dart'
    $expectedV21 = '43a87f05df7adc6a231a679721e7f0b11b45a3fe6d1582aa4527ed738fa63b3f'
    $expectedV1 = 'b8fc10191f69faef16f1dc92d08bf9542ee7e26f885767942e28292c5201dd90'
    $threshold = '0.3675977844516245'

    if ((Get-FileHash $modelPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedV21) {
        throw 'v2.1 ONNX SHA-256 mismatch.'
    }
    if ((Get-FileHash $v1ModelPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedV1) {
        throw 'The rollback v1.1 ONNX asset differs from its frozen SHA-256.'
    }
    $service = Get-Content -LiteralPath $servicePath -Raw
    if (-not $service.Contains("const v21Threshold = $threshold;")) {
        throw 'The configured v2.1 threshold differs from the frozen value.'
    }
    if (-not $service.Contains("'assets/offirst_v2_1_experimental.onnx'")) {
        throw 'The v2.1 model asset selection is missing.'
    }

    flutter build apk --release --build-name=1.0.3 --build-number=2003 `
        --target-platform android-arm64,android-arm `
        --dart-define=OFFIRST_EXPERIMENTAL_V21=true
    if ($LASTEXITCODE -ne 0) { throw 'Flutter Android candidate build failed.' }

    $outputDir = Join-Path $projectRoot 'build/app/outputs/flutter-apk'
    $builtApk = Join-Path $outputDir 'app-release.apk'
    $finalApk = Join-Path $outputDir 'OFFIRST-v2.1-final-candidate.apk'
    Copy-Item -LiteralPath $builtApk -Destination $finalApk -Force

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [System.IO.Compression.ZipFile]::OpenRead($finalApk)
    try {
        $assetEntry = $archive.GetEntry('assets/flutter_assets/assets/offirst_v2_1_experimental.onnx')
        if ($null -eq $assetEntry) { throw 'Packaged v2.1 ONNX asset is missing.' }
        $assetStream = $assetEntry.Open()
        try {
            $sha = [System.Security.Cryptography.SHA256]::Create()
            try { $packagedV21 = ([BitConverter]::ToString($sha.ComputeHash($assetStream))).Replace('-', '').ToLowerInvariant() }
            finally { $sha.Dispose() }
        } finally { $assetStream.Dispose() }

        $appEntry = $archive.GetEntry('lib/arm64-v8a/libapp.so')
        if ($null -eq $appEntry) { throw 'ARM64 Flutter app library is missing.' }
        $appStream = $appEntry.Open()
        try {
            $buffer = [byte[]]::new([int]$appEntry.Length)
            $read = $appStream.Read($buffer, 0, $buffer.Length)
            if ($read -ne $buffer.Length) { throw 'Could not read the packaged ARM64 app library.' }
            $snapshotText = [System.Text.Encoding]::GetEncoding(28591).GetString($buffer)
            if (-not $snapshotText.Contains('offirst_v2_1_experimental.onnx')) {
                throw 'Packaged application does not reference the v2.1 model asset.'
            }
            if (-not $snapshotText.Contains('Experimental build: packaged OFFIRST v2.1')) {
                throw 'Packaged application does not contain the v2.1 build identity.'
            }
            if ($snapshotText.Contains('The app uses the packaged OFFIRST v1.1')) {
                throw 'The packaged application still contains the v1.1 model selection.'
            }
        } finally { $appStream.Dispose() }
    } finally { $archive.Dispose() }
    if ($packagedV21 -ne $expectedV21) { throw 'Packaged v2.1 ONNX SHA-256 mismatch.' }

    $sdk = if ($env:ANDROID_HOME) { $env:ANDROID_HOME } else { Join-Path $env:LOCALAPPDATA 'Android/Sdk' }
    $buildTools = Get-ChildItem (Join-Path $sdk 'build-tools') -Directory |
        Sort-Object { [version]$_.Name } -Descending | Select-Object -First 1
    if ($null -eq $buildTools) { throw 'Android build-tools not found.' }
    $aapt = Join-Path $buildTools.FullName 'aapt.exe'
    $apksigner = Join-Path $buildTools.FullName 'apksigner.bat'
    $badging = & $aapt dump badging $finalApk
    if ($LASTEXITCODE -ne 0) { throw 'Could not inspect APK manifest.' }
    $badgingText = $badging -join "`n"
    foreach ($required in @(
        "package: name='org.offirst.offline_dr_screening' versionCode='2003' versionName='1.0.3'",
        "application-label:'OFFIRST'",
        "native-code: 'arm64-v8a' 'armeabi-v7a'"
    )) {
        if (-not $badgingText.Contains($required)) { throw "APK manifest check failed: $required" }
    }
    $signature = & $apksigner verify --print-certs $finalApk
    if ($LASTEXITCODE -ne 0) { throw 'APK signature verification failed.' }
    $signerLine = $signature | Where-Object { $_ -match 'Signer #1 certificate SHA-256 digest:' } | Select-Object -First 1
    $signer = ($signerLine -replace '.*digest:\s*', '').Trim().ToLowerInvariant()
    $expectedSigner = '08e09b07b3e9e052535d6e34219b4d50f3e320e41da40dcd764f60982d226bc7'
    if ($signer -ne $expectedSigner) { throw 'APK signer does not match the published prototype certificate.' }
    $apkHash = (Get-FileHash $finalApk -Algorithm SHA256).Hash.ToLowerInvariant()
    $manifestPath = Join-Path $outputDir 'OFFIRST-v2.1-final-candidate.sha256'
    @(
        "$apkHash  OFFIRST-v2.1-final-candidate.apk"
        "$expectedV21  offirst_v2_1_experimental.onnx"
    ) | Set-Content -LiteralPath $manifestPath -Encoding ASCII

    Write-Output "APK: $finalApk"
    Write-Output "APK bytes: $((Get-Item $finalApk).Length)"
    Write-Output "APK SHA-256: $apkHash"
    Write-Output "v2.1 ONNX SHA-256: $expectedV21"
    Write-Output "Build: OFFIRST 1.0.3 (2003), package org.offirst.offline_dr_screening"
    Write-Output "ABIs: arm64-v8a, armeabi-v7a"
    Write-Output "Threshold: $threshold"
    Write-Output "Checksum manifest: $manifestPath"
    Write-Output 'Signing verification:'
    $signature | Select-String 'Signer #1 certificate SHA-256 digest'
} finally {
    Pop-Location
}
