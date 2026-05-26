param(
    [string]$ProjectRoot = ".",
    [string]$OutputDir = "artifacts/kaggle"
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path $ProjectRoot).Path
$out = Join-Path $root $OutputDir

$codeStage = Join-Path $out "code_stage"
$dataStage = Join-Path $out "data_stage"
$codeZip = Join-Path $out "gkx_code_bundle.zip"
$dataZip = Join-Path $out "gkx_processed_bundle.zip"

New-Item -ItemType Directory -Force -Path $out | Out-Null
if (Test-Path $codeStage) { Remove-Item -Recurse -Force $codeStage }
if (Test-Path $dataStage) { Remove-Item -Recurse -Force $dataStage }
New-Item -ItemType Directory -Force -Path $codeStage | Out-Null
New-Item -ItemType Directory -Force -Path $dataStage | Out-Null

# ---- Build code bundle stage ----
$codeTargets = @(
    "src",
    "docs",
    "requirements.txt",
    "README.md",
    "scripts/run_phase5c.py",
    "scripts/kaggle/kaggle_phase5c_runner.py",
    "scripts/kaggle/kaggle_nn_runner.py",
    "scripts/kaggle/kaggle_bootstrap.py"
)

foreach ($t in $codeTargets) {
    $srcPath = Join-Path $root $t
    if (-not (Test-Path $srcPath)) {
        throw "Missing required code target: $t"
    }

    $destPath = Join-Path $codeStage $t
    $destParent = Split-Path -Parent $destPath
    New-Item -ItemType Directory -Force -Path $destParent | Out-Null

    Copy-Item -Path $srcPath -Destination $destPath -Recurse -Force
}

# ---- Build processed-data bundle stage ----
$dataTargets = @(
    "data/processed/features_panel.parquet"
)

$optionalTargets = @(
    "data/processed/predictions.parquet",
    "data/processed/run_manifest.json"
)

foreach ($t in $dataTargets) {
    $srcPath = Join-Path $root $t
    if (-not (Test-Path $srcPath)) {
        throw "Missing required data target: $t"
    }

    $destPath = Join-Path $dataStage $t
    $destParent = Split-Path -Parent $destPath
    New-Item -ItemType Directory -Force -Path $destParent | Out-Null

    Copy-Item -Path $srcPath -Destination $destPath -Recurse -Force
}

foreach ($t in $optionalTargets) {
    $srcPath = Join-Path $root $t
    if (Test-Path $srcPath) {
        $destPath = Join-Path $dataStage $t
        $destParent = Split-Path -Parent $destPath
        New-Item -ItemType Directory -Force -Path $destParent | Out-Null

        Copy-Item -Path $srcPath -Destination $destPath -Recurse -Force
    }
}

# ---- Create zip archives ----
if (Test-Path $codeZip) { Remove-Item -Force $codeZip }
if (Test-Path $dataZip) { Remove-Item -Force $dataZip }

function New-ZipWithPython {
    param(
        [string]$SourceDir,
        [string]$ZipPath
    )

    $pyCode = @"
from pathlib import Path
import zipfile

src = Path(r'''$SourceDir''')
dst = Path(r'''$ZipPath''')

with zipfile.ZipFile(dst, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    for p in src.rglob('*'):
        if p.is_file():
            zf.write(p, p.relative_to(src))
"@

    python -c $pyCode
}

New-ZipWithPython -SourceDir $codeStage -ZipPath $codeZip
New-ZipWithPython -SourceDir $dataStage -ZipPath $dataZip

Write-Host "Created:" $codeZip
Write-Host "Created:" $dataZip

# Keep staging folders for auditability during deadline crunch.
Write-Host "Staging folders preserved at:" $out
