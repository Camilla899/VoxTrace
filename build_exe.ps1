# Build the portable Windows folder: dist\VoxTrace\VoxTrace.exe
# Prerequisites: Python 3.12 with requirements-dev.txt and PyInstaller installed.
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

$py = $env:VOXTRACE_PY
if (-not $py) {
    $candidates = @(
        (Join-Path $root ".venv\Scripts\python.exe"),
        "C:\Users\Administrator\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe",
        (Get-Command python -ErrorAction SilentlyContinue).Source
    )
    $py = $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
}
if (-not $py) {
    throw "Python not found. Set VOXTRACE_PY to a Python 3.12 executable."
}

$env:PYTHONIOENCODING = "utf-8"
$icon = Join-Path $root "app.ico"

& $py -m PyInstaller --noconfirm --clean --windowed --name VoxTrace `
    --icon $icon --distpath (Join-Path $root "dist") --workpath (Join-Path $root "build") `
    --specpath (Join-Path $root "build") `
    --add-data "$root\models;models" `
    --add-data "$root\demo;demo" `
    --add-data "$root\licenses;licenses" `
    --add-data "$root\voxtrace\ui\assets;voxtrace\ui\assets" `
    --add-data "$root\README.md;." `
    --add-data "$root\THIRD_PARTY_NOTICES.md;." `
    --add-data "$root\app.ico;." `
    --collect-data librosa `
    --hidden-import PySide6.QtMultimedia `
    --hidden-import PySide6.QtSvg `
    --hidden-import PySide6.QtNetwork `
    --hidden-import mido `
    --exclude-module pandas --exclude-module sklearn --exclude-module PIL `
    --exclude-module lxml --exclude-module matplotlib --exclude-module huggingface_hub `
    --exclude-module httpx2 --exclude-module httpcore2 --exclude-module anyio `
    --exclude-module yaml --exclude-module onnx --exclude-module torch `
    --exclude-module tensorflow --exclude-module IPython --exclude-module pytest `
    --exclude-module setuptools --exclude-module tkinter `
    --exclude-module PySide6.QtWebEngineCore `
    --exclude-module PySide6.QtWebEngineWidgets `
    --exclude-module PySide6.QtWebEngineQuick `
    app.py

if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }
Copy-Item (Join-Path $root "README.md") (Join-Path $root "dist\VoxTrace\README.md") -Force
Copy-Item (Join-Path $root "THIRD_PARTY_NOTICES.md") (Join-Path $root "dist\VoxTrace\THIRD_PARTY_NOTICES.md") -Force
Write-Host "Built: $root\dist\VoxTrace\VoxTrace.exe"
