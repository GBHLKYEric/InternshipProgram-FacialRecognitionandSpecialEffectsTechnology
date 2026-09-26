param([switch]$Web, [ValidateRange(1,65535)][int]$Port=8765)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
# Repair restricted child-process environments without changing system settings.
if (-not $env:SystemRoot) { $env:SystemRoot = 'C:\Windows' }
if (-not $env:WINDIR) { $env:WINDIR = $env:SystemRoot }
if (-not $env:COMSPEC) { $env:COMSPEC = "$env:SystemRoot\System32\cmd.exe" }
$env:PYTHONIOENCODING = 'utf-8'
if (-not (Test-Path '.venv/Scripts/python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python virtual environment creation failed.' }
}
& .venv/Scripts/python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& .venv/Scripts/python.exe scripts/fetch_models.py
if ($LASTEXITCODE -ne 0) { throw 'Model download failed.' }
if ($Web) {
    Start-Process "http://127.0.0.1:$Port"
    & .venv/Scripts/python.exe app.py --port $Port
} else {
    & .venv/Scripts/python.exe desktop.py
}
