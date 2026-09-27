param([string]$Python = 'python')
# Uses only official PyTorch XPU wheels; never installs or updates a GPU driver.
$ErrorActionPreference = 'Stop'
$env:SystemRoot = 'C:\Windows'
$env:WINDIR = 'C:\Windows'
$env:COMSPEC = 'C:\Windows\System32\cmd.exe'
$Project = Split-Path $PSScriptRoot -Parent
$Parent = Split-Path $Project -Parent
$Work = if ((Split-Path $Parent -Leaf) -eq 'outputs') { Join-Path (Split-Path $Parent -Parent) 'work' } else { Join-Path $Project 'work' }
$Environment = Join-Path $Work 'xpu-env'
$EnvironmentPython = Join-Path $Environment 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $EnvironmentPython)) {
    & $Python -m venv $Environment
    if ($LASTEXITCODE -ne 0) { throw 'Provide a usable Python 3.12 interpreter with -Python.' }
}
& $EnvironmentPython -m pip install torch==2.14.0+xpu torchvision==0.29.0+xpu --index-url https://download.pytorch.org/whl/xpu
if ($LASTEXITCODE -ne 0) { throw 'Official XPU wheel installation failed.' }
# Install metric tooling from PyPI only after the pinned XPU builds are present.
# Existing torch/torchvision satisfy torch-fidelity; do not replace them with CPU wheels.
& $EnvironmentPython -m pip install torch-fidelity==0.4.0 --index-url https://pypi.org/simple
if ($LASTEXITCODE -ne 0) { throw 'XPU evaluation dependency installation failed.' }
& $EnvironmentPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'XPU dependency check failed.' }
& $EnvironmentPython (Join-Path $PSScriptRoot 'verify_xpu.py')
if ($LASTEXITCODE -ne 0) { throw 'XPU hardware verification failed; see reports/xpu-environment.json.' }
