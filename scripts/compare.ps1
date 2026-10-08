# Evaluate base and fine-tuned checkpoints on the same test set, then compare side by side
#   .\scripts\compare.ps1                (GPU if available; -Cpu to force CPU)
#   .\scripts\compare.ps1 -Base english -Model checkpoints/other -Data data/test.jsonl
param([switch]$Cpu, [string]$Base = "multilingual", [string]$Model = "checkpoints/ko-tickets",
      [string]$Data = "data/test.jsonl")
. "$PSScriptRoot\_env.ps1"
$Device = Resolve-Device $Cpu

uv run eval.py $Data --model $Base --device $Device --json reports/base.json
if ($LASTEXITCODE) { exit $LASTEXITCODE }
uv run eval.py $Data --model $Model --device $Device --json reports/finetuned.json
if ($LASTEXITCODE) { exit $LASTEXITCODE }

uv run compare.py reports/base.json reports/finetuned.json $Data reports/compare.md
exit $LASTEXITCODE
