# Shared setup, dot-sourced by every script:  . "$PSScriptRoot\_env.ps1"
$ErrorActionPreference = "Stop"
# PowerShell 5.1 turns a native command's stderr (tqdm progress) into an error when output is redirected
# (*> log.txt), which "Stop" would turn into an abort. Relax it for uv only; callers check $LASTEXITCODE.
function uv { $ErrorActionPreference = "Continue"; & uv.exe @args }
$env:PYTHONUTF8 = "1"
# `uv run` would re-sync to the default (no extra) and drop the gpu/cpu torch chosen by setup.ps1
$env:UV_NO_SYNC = "1"
Set-Location (Split-Path $PSScriptRoot -Parent)

# .env (KEY=VALUE) -> environment; values already set in the shell win
if (Test-Path .env) {
    foreach ($line in Get-Content .env -Encoding UTF8) {
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$' -and -not [Environment]::GetEnvironmentVariable($Matches[1])) {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2].Trim('"', "'"))
        }
    }
}

if (-not (Test-Path .venv)) { Write-Host "No .venv: run .\scripts\setup.ps1 first"; exit 2 }

# GPU by default; CPU when forced (-Cpu) or when torch cannot see a CUDA device
function Resolve-Device([bool]$ForceCpu) {
    if ($ForceCpu) { return "cpu" }
    uv run python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" | Out-Null
    if ($LASTEXITCODE -eq 0) { return "cuda" }
    Write-Host "CUDA not available -> falling back to CPU" -ForegroundColor Yellow
    return "cpu"
}
