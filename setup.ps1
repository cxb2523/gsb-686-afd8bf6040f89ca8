# setup.ps1 - gets Changeling ready: the psdenv virtual environment (Python 3.9 - 3.12),
# the packages in requirements.txt and the models. Safe to run again: it only redoes what's
# missing or changed. run.ps1 calls it for you.
#
# If PowerShell refuses to run scripts, once:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$VenvDir = 'psdenv'
$VenvPy = Join-Path $VenvDir 'Scripts\python.exe'

function Invoke-Checked {
    & $args[0] $args[1..($args.Count - 1)]
    if ($LASTEXITCODE -ne 0) { throw "Setup failed: $($args -join ' '). Run setup.ps1 again to retry." }
}

# 1) The virtual environment (TensorFlow and PyTorch need Python 3.9 - 3.12).
if (-not (Test-Path $VenvPy)) {
    Write-Host 'Creating the virtual environment...'
    $base = $null
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in '3.11', '3.12', '3.10', '3.9') {
            & py "-$v" -c 'import sys' 2>$null
            if ($LASTEXITCODE -eq 0) { $base = @('py', "-$v"); break }
        }
    }
    if (-not $base -and (Get-Command python -ErrorAction SilentlyContinue)) {
        & python -c 'import sys; sys.exit(0 if (3, 9) <= sys.version_info[:2] <= (3, 12) else 1)' 2>$null
        if ($LASTEXITCODE -eq 0) { $base = @('python') }
    }
    if (-not $base) { throw 'Python 3.9 - 3.12 is needed (3.11 is best). Get it from https://www.python.org/downloads/' }
    Invoke-Checked @base -m venv $VenvDir
}

# 2) The packages, whenever requirements.txt changed since the last install.
#    The first time downloads several GB (PyTorch, TensorFlow) - give it a while.
$stamp = Join-Path $VenvDir '.requirements'
$needInstall = -not (Test-Path $stamp) -or
    (Get-FileHash requirements.txt).Hash -ne (Get-FileHash $stamp).Hash
if ($needInstall) {
    Write-Host 'Installing the packages - the first time takes a while...'
    Invoke-Checked $VenvPy -m pip install --upgrade pip
    Invoke-Checked $VenvPy -m pip install -r requirements.txt
    Copy-Item requirements.txt $stamp -Force
}

# 3) Any missing models.
Invoke-Checked $VenvPy scripts/download_models.py

# 4) Which GPU, and a warning if its driver is too old.
& $VenvPy scripts/check_gpu.py

Write-Host 'Changeling Imago is ready. Start it with run.ps1'
