# run.ps1 - starts Changeling Imago. Sets it up first if it needs it (setup.ps1).
#   .\run.ps1                                       opens the app
#   .\run.ps1 -s face.png -t clip.mp4 -o out.mp4    swaps without the app
# Uses your NVIDIA GPU when there is one; add --execution-provider cpu to force the CPU.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

& (Join-Path $PSScriptRoot 'setup.ps1')

$provider = @()
if (-not ($args -match '^--execution-provider') -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    $provider = @('--execution-provider', 'cuda')
}
& (Join-Path 'psdenv' 'Scripts\python.exe') run.py @provider @args
exit $LASTEXITCODE
