# Shared setup, dot-sourced by every script:  . "$PSScriptRoot\_env.ps1"
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
# `uv run` would re-sync to the default (no extra) and drop the gpu/cpu torch chosen by setup.ps1
$env:UV_NO_SYNC = "1"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path .venv)) { Write-Host "No .venv: run .\scripts\setup.ps1 first"; exit 2 }

# GPU by default; CPU when forced (-Cpu) or when torch cannot see a CUDA device
function Resolve-Device([bool]$ForceCpu) {
    if ($ForceCpu) { return "cpu" }
    uv run python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" | Out-Null
    if ($LASTEXITCODE -eq 0) { return "cuda" }
    Write-Host "CUDA not available -> falling back to CPU" -ForegroundColor Yellow
    return "cpu"
}
