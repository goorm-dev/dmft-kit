# Install dependencies, picking the torch build for this machine
#   .\scripts\setup.ps1              NVIDIA GPU found -> CUDA torch (gpu), otherwise CPU torch (cpu)
#   .\scripts\setup.ps1 -Torch cpu   force a build
param([ValidateSet("auto", "gpu", "cpu")][string]$Torch = "auto")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if ($Torch -eq "auto") {
    $nvidia = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "NVIDIA" }
    $Torch = if ($nvidia) { "gpu" } else { "cpu" }
    $found = if ($nvidia) { $nvidia[0].Name } else { "no NVIDIA GPU" }
    Write-Host "Detected: $found -> torch build '$Torch'"
}

uv sync --extra $Torch
if ($LASTEXITCODE) { exit $LASTEXITCODE }

uv run --no-sync python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
exit $LASTEXITCODE
