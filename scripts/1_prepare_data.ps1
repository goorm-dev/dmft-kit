# Step 1: labels -> data/train.jsonl, data/test.jsonl, then validate (no weights loaded, no training)
#   .\scripts\1_prepare_data.ps1                  export from Argilla (Submitted + unreviewed synthetic train)
#   .\scripts\1_prepare_data.ps1 -ReviewedOnly    export only what a human Submitted
#   .\scripts\1_prepare_data.ps1 -Demo            built-in demo data instead of Argilla
param([switch]$Demo, [switch]$ReviewedOnly, [string]$Base = "multilingual")
. "$PSScriptRoot\_env.ps1"

if ($Demo) {
    uv run data/build_data.py
} elseif ($ReviewedOnly) {
    uv run scripts/data/export.py --reviewed-only
} else {
    uv run scripts/data/export.py
}
if ($LASTEXITCODE) { exit $LASTEXITCODE }

uv run laya-evals validate data/train.jsonl
if ($LASTEXITCODE) { exit $LASTEXITCODE }
uv run laya-evals validate data/test.jsonl
if ($LASTEXITCODE) { exit $LASTEXITCODE }

# Tokenizes with the base checkpoint's config and reports skipped questions and the reason
uv run laya-train --data data/train.jsonl --base $Base --dry-run
exit $LASTEXITCODE
