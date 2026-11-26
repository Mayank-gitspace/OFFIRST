param([switch]$AllowDifferentSigner)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    $modelPath = Join-Path $projectRoot 'assets/offirst_v2_1_experimental.onnx'
    $v1ModelPath = Join-Path $projectRoot 'assets/offirst_dr_model.onnx'
    $servicePath = Join-Path $projectRoot 'lib/screening_service.dart'
    $expectedV21 = 'b6bcb1182db7717c01ef24181d79a44be7d1a75ddefe8567cfec2fffb9e5c59b'
    $expectedV1 = 'b8fc10191f69faef16f1dc92d08bf9542ee7e26f885767942e28292c5201dd90'
    $threshold = '0.524373471736908'
    if ((Get-FileHash $modelPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedV21) { throw 'v2.1 ONNX SHA-256 mismatch.' }
    if ((Get-FileHash $v1ModelPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedV1) { throw 'Rollback v1.1 ONNX changed.' }
    $service = Get-Content -LiteralPath $servicePath -Raw
    if (-not $service.Contains('defaultValue: true,') -or -not $service.Contains("const v21Threshold = $threshold;") -or -not $service.Contains("'assets/offirst_v2_1_experimental.onnx'")) { throw 'v2.1 build selection or threshold check failed.' }

    flutter build apk --release --split-per-abi --build-name=1.0.4 --build-number=2004 --target-platform android-arm64,android-arm
    if ($LASTEXITCODE -ne 0) { throw 'Flutter split APK build failed.' }

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $outputDir = Join-Path $projectRoot 'build/app/outputs/flutter-apk'
    $sdk = if ($env:ANDROID_HOME) { $env:ANDROID_HOME } else { Join-Path $env:LOCALAPPDATA 'Android/Sdk' }
    $buildTools = Get-ChildItem (Join-Path $sdk 'build-tools') -Directory | Sort-Object { [version]$_.Name } -Descending | Select-Object -First 1
    if ($null -eq $buildTools) { throw 'Android build-tools not found.' }
    $aapt = Join-Path $buildTools.FullName 'aapt.exe'
    $apksigner = Join-Path $buildTools.FullName 'apksigner.bat'
    $expectedSigner = '08e09b07b3e9e052535d6e34219b4d50f3e320e41da40dcd764f60982d226bc7'
    $records = @()
    foreach ($variant in @(
        @{ abi = 'arm64-v8a'; versionCode = '4004'; built = 'app-arm64-v8a-release.apk'; name = 'OFFIRST-v2.1-inference-fix-arm64.apk' },
        @{ abi = 'armeabi-v7a'; versionCode = '3004'; built = 'app-armeabi-v7a-release.apk'; name = 'OFFIRST-v2.1-inference-fix-armv7.apk' }
    )) {
        $built = Join-Path $outputDir $variant.built
        if (-not (Test-Path -LiteralPath $built)) { throw "Missing split APK: $($variant.built)" }
        $apk = Join-Path $outputDir $variant.name
        Copy-Item -LiteralPath $built -Destination $apk -Force
        $archive = [System.IO.Compression.ZipFile]::OpenRead($apk)
        try {
            $model = $archive.GetEntry('assets/flutter_assets/assets/offirst_v2_1_experimental.onnx')
            if ($null -eq $model) { throw "v2.1 asset missing in $($variant.name)" }
            $stream = $model.Open()
            try {
                $sha = [System.Security.Cryptography.SHA256]::Create()
                try { $modelHash = ([BitConverter]::ToString($sha.ComputeHash($stream))).Replace('-', '').ToLowerInvariant() }
                finally { $sha.Dispose() }
            } finally { $stream.Dispose() }
            if ($modelHash -ne $expectedV21) { throw "Packaged ONNX hash mismatch in $($variant.name)" }
            $appLibrary = $archive.GetEntry("lib/$($variant.abi)/libapp.so")
            if ($null -eq $appLibrary) { throw "Flutter application library missing for $($variant.abi)" }
            $appStream = $appLibrary.Open()
            try {
                $buffer = [byte[]]::new([int]$appLibrary.Length)
                $read = $appStream.Read($buffer, 0, $buffer.Length)
                if ($read -ne $buffer.Length) { throw "Could not read packaged Flutter code in $($variant.name)" }
                $snapshotText = [System.Text.Encoding]::GetEncoding(28591).GetString($buffer)
                foreach ($identity in @('offirst_v2_1_experimental.onnx', 'Active model: OFFIRST v2.1 experimental DR classifier')) {
                    if (-not $snapshotText.Contains($identity)) { throw "v2.1 selection check failed for $($variant.name)" }
                }
            } finally { $appStream.Dispose() }
        } finally { $archive.Dispose() }
        $badging = (& $aapt dump badging $apk) -join "`n"
        if ($LASTEXITCODE -ne 0) { throw "Could not inspect $($variant.name)" }
        foreach ($required in @(
            "package: name='org.offirst.offline_dr_screening' versionCode='$($variant.versionCode)' versionName='1.0.4'",
            "application-label:'OFFIRST'",
            "native-code: '$($variant.abi)'"
        )) { if (-not $badging.Contains($required)) { throw "APK manifest check failed for $($variant.name): $required" } }
        $signature = & $apksigner verify --print-certs $apk
        if ($LASTEXITCODE -ne 0) { throw "APK signature verification failed for $($variant.name)" }
        $signerLine = $signature | Where-Object { $_ -match 'Signer #1 certificate SHA-256 digest:' } | Select-Object -First 1
        $signer = ($signerLine -replace '.*digest:\s*', '').Trim().ToLowerInvariant()
        if ($signer -ne $expectedSigner -and -not $AllowDifferentSigner) { throw "Signer mismatch for $($variant.name)" }
        $permissions = (& $aapt dump permissions $apk) -join "`n"
        if ($LASTEXITCODE -ne 0 -or $permissions.Contains('android.permission.INTERNET')) { throw "Unexpected permissions in $($variant.name)" }
        $hash = (Get-FileHash -LiteralPath $apk -Algorithm SHA256).Hash.ToLowerInvariant()
        $records += [pscustomobject]@{ path = $apk; name = $variant.name; abi = $variant.abi; size = (Get-Item $apk).Length; sha256 = $hash; signer = $signer }
    }
    $manifestPath = Join-Path $outputDir 'OFFIRST-v2.1-inference-fix.sha256'
    @($records | ForEach-Object { "$($_.sha256)  $($_.name)" }) + @("$expectedV21  offirst_v2_1_experimental.onnx") | Set-Content -LiteralPath $manifestPath -Encoding ASCII
    $records | Format-List
    "ONNX SHA-256: $expectedV21"
    "Threshold: $threshold"
    "Version: 1.0.4 (2004); package org.offirst.offline_dr_screening"
    "Checksum manifest: $manifestPath"
} finally { Pop-Location }
