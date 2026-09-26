param(
    [switch]$InstallAndConfigure,
    [switch]$AcceptAnacondaTerms
)
# The default action only downloads and verifies the official installer.
# Run -InstallAndConfigure -AcceptAnacondaTerms ONLY after the user accepts
# the installer and main repository terms linked in the generated report.
$ErrorActionPreference = 'Stop'
$env:SystemRoot = 'C:\Windows'
$env:WINDIR = 'C:\Windows'
$env:COMSPEC = 'C:\Windows\System32\cmd.exe'
$env:USERNAME = [Environment]::UserName
$env:APPDATA = [Environment]::GetFolderPath('ApplicationData')
$env:LOCALAPPDATA = [Environment]::GetFolderPath('LocalApplicationData')
$Project = Split-Path $PSScriptRoot -Parent
$Parent = Split-Path $Project -Parent
$Workspace = if ((Split-Path $Parent -Leaf) -eq 'outputs') { Split-Path $Parent -Parent } else { $Project }
$Work = Join-Path $Workspace 'work'
$InstallDir = Join-Path $Work 'anaconda'
$EnvDir = Join-Path $Work 'conda-face-vision'
$DownloadDir = Join-Path $Work 'installers'
$InstallerName = 'Anaconda3-2026.07-1-Windows-x86_64.exe'
$Installer = Join-Path $DownloadDir $InstallerName
$ExpectedHash = 'b545f4bd8ab3bf32d99002a0779a887668ebfe479ee32ecbf060375670d5ee09'
$InstallerUrl = "https://repo.anaconda.com/archive/$InstallerName"
New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null
if (-not (Test-Path -LiteralPath $Installer)) {
    & curl.exe --fail --location --retry 3 --continue-at - --output "$Installer.part" $InstallerUrl
    if ($LASTEXITCODE -ne 0) { throw "Installer download failed: $LASTEXITCODE" }
    if ((Get-FileHash -LiteralPath "$Installer.part" -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedHash) {
        throw 'Installer SHA256 mismatch. The unverified .part file was retained.'
    }
    Move-Item -LiteralPath "$Installer.part" -Destination $Installer
}
$ActualHash = (Get-FileHash -LiteralPath $Installer -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ActualHash -ne $ExpectedHash) { throw 'Installer SHA256 mismatch.' }
$Signature = Get-AuthenticodeSignature -LiteralPath $Installer
if ($Signature.Status -ne 'Valid') { throw "Installer signature invalid: $($Signature.Status)" }
$ReportPath = Join-Path $Project 'reports/anaconda-environment.json'
if (-not $InstallAndConfigure -and (Test-Path -LiteralPath $ReportPath)) {
    $Previous = Get-Content -LiteralPath $ReportPath -Raw | ConvertFrom-Json
    if ($Previous.installer.sha256 -eq $ActualHash -and $Previous.terms.accepted) {
        Write-Output "Verified official installer; preserved existing installation report ($($Previous.status))."
        return
    }
}
$Report = [ordered]@{
    status = 'installer_verified_awaiting_terms_acceptance'
    checked_at_utc = [DateTime]::UtcNow.ToString('o')
    installer = [ordered]@{ version='2026.07-1'; url=$InstallerUrl; sha256=$ActualHash; bytes=(Get-Item -LiteralPath $Installer).Length; local_path=$Installer; official_hash_source='https://repo.anaconda.com/archive/'; authenticode_status=$Signature.Status.ToString(); signer=$Signature.SignerCertificate.Subject }
    installation = [ordered]@{ path=$InstallDir; project_environment=$EnvDir; performed=$false; add_to_path=$false; register_default_python=$false; conda_init=$false; no_registry=$true; no_shortcuts=$true; registry_note='The exact installer /S /? confirms support for /NoRegistry=1; this additional flag disables installer registry modifications.' }
    terms = [ordered]@{ accepted=$false; url='https://www.anaconda.com/legal/terms/terms-of-service'; installer_notice_source='https://repo.anaconda.com/archive/'; main_channel='https://repo.anaconda.com/pkgs/main'; scope='Anaconda Distribution Installer and main package repository; no paid plan or account signup'; free_personal_use='Section 1(a)(1): individual personal, non-commercial use. Institutional teaching has separate eligibility and Academic EULA conditions.' }
    validation = $null
}
$Report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ReportPath -Encoding utf8
Write-Output "Verified official installer: $Installer"
Write-Output "SHA256: $ActualHash"
if (-not $InstallAndConfigure) {
    Write-Output 'Prepared only. Installer and legal acceptance have NOT been executed.'
    return
}
if (-not $AcceptAnacondaTerms) {
    throw 'Explicit acceptance required: review https://www.anaconda.com/legal/terms/terms-of-service before passing -AcceptAnacondaTerms.'
}
$Report.terms.accepted = $true
$Report.status = 'installing'
$Report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ReportPath -Encoding utf8
$PhysicalInstallDir = $InstallDir
$PhysicalEnvDir = $EnvDir
$LongPaths = Get-ItemPropertyValue -LiteralPath 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled
if ($LongPaths -eq 0 -and $InstallDir.Length -gt 54) {
    # This installer rejects >54 chars and '~'. Its extractor resolves junctions,
    # so the actual files must occupy a short directory. The thread work paths
    # are convenience junctions pointing TO these physical short directories.
    $ShortWork = Join-Path ([Environment]::GetFolderPath('MyDocuments')) 'Codex/work'
    $PhysicalInstallDir = Join-Path $ShortWork 'fv-ana-260926'
    $PhysicalEnvDir = Join-Path $ShortWork 'fv-env-260926'
    if ($PhysicalInstallDir.Length -gt 54) { throw 'No usable short installation prefix. Machine long-path policy was not modified.' }
    foreach ($Pair in @(@($InstallDir, $PhysicalInstallDir), @($EnvDir, $PhysicalEnvDir))) {
        New-Item -ItemType Directory -Force -Path $Pair[1] | Out-Null
        if (Test-Path -LiteralPath $Pair[0]) {
            $Item = Get-Item -LiteralPath $Pair[0]
            if ($Item.LinkType -ne 'Junction' -or $Item.Target -ne $Pair[1]) {
                throw "Existing path $($Pair[0]) is not the expected junction. It was not modified."
            }
        } else {
            New-Item -ItemType Junction -Path $Pair[0] -Target $Pair[1] | Out-Null
        }
    }
}
$Report.installation.physical_path = $PhysicalInstallDir
$Report.installation.physical_project_environment = $PhysicalEnvDir
$Report.installation.long_paths_policy_modified = $false
$Report.installation.compatibility_notes = @(
    'On this computer LongPathsEnabled=0 and the 2026.07-1 installer rejected the 93-character prefix; its reported maximum was 54.',
    'The installer also rejected NTFS 8.3 names containing ~. A short junction targeting a long physical directory still failed because conda resolved the physical path.',
    'The successful strategy uses physical short directories under Documents/Codex/work and convenience junctions in the thread work directory. No long-path registry policy was changed.'
)
$Report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ReportPath -Encoding utf8
$Conda = Join-Path $PhysicalInstallDir 'Scripts/conda.exe'
if (-not (Test-Path -LiteralPath $Conda)) {
    # /D must be last and unquoted in the NSIS argument string, even for spaces.
    & $Installer /InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /NoRegistry=1 /NoShortcuts=1 /S "/D=$PhysicalInstallDir" | Tee-Object -FilePath (Join-Path $Work 'anaconda-vendor-install.txt')
    if ($LASTEXITCODE -ne 0) { throw "Anaconda installer exited $LASTEXITCODE" }
}
if (-not (Test-Path -LiteralPath $Conda)) { throw 'Installer returned without a usable conda.exe.' }
$Report.installation.performed = $true
$Report.status = 'creating_project_environment'
$Report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ReportPath -Encoding utf8
# Only the main channel is used. No global .condarc or shell initialization.
$env:CONDA_PKGS_DIRS = Join-Path $PhysicalInstallDir 'pkgs'
$env:CONDA_ENVS_PATH = Split-Path $PhysicalEnvDir -Parent
& $Conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
if ($LASTEXITCODE -ne 0) { throw 'Could not record the explicitly authorized repository terms acceptance.' }
if (-not (Test-Path -LiteralPath (Join-Path $EnvDir 'conda-meta/history'))) {
    & $Conda create --yes --prefix $PhysicalEnvDir --override-channels --channel https://repo.anaconda.com/pkgs/main python=3.12 pip
    if ($LASTEXITCODE -ne 0) { throw "conda create failed: $LASTEXITCODE" }
}
$EnvPython = Join-Path $PhysicalEnvDir 'python.exe'
# This process-only activation lets conda's Windows DLLs be found. It does not
# write the user/system PATH or initialize any future shell.
$env:PATH = "$PhysicalEnvDir;$PhysicalEnvDir\Scripts;$PhysicalEnvDir\Library\bin;" + $env:PATH
$env:CONDA_PREFIX = $PhysicalEnvDir
& $EnvPython -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed.' }
& $EnvPython -m pip install -r (Join-Path $Project 'requirements-research.txt') jupyterlab==4.6.4 nbclient==0.11.0 nbformat==5.11.1
if ($LASTEXITCODE -ne 0) { throw 'Project dependencies installation failed.' }
& $EnvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency verification failed.' }
& $EnvPython (Join-Path $PSScriptRoot 'verify_anaconda.py') --environment $EnvDir --report $ReportPath
if ($LASTEXITCODE -ne 0) { throw 'Anaconda environment functional verification failed.' }
