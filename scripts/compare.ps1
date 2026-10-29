# Evaluate base and fine-tuned checkpoints on the same test set, then compare side by side
#   .\scripts\compare.ps1                (GPU if available; -Cpu to force CPU)
#   .\scripts\compare.ps1 -Base english -Model checkpoints/other -Data data/test.jsonl
#   .\scripts\compare.ps1 -Name test     -> reports/compare_test.md, base_test.json, finetuned_test.json (keeps other runs)
param([switch]$Cpu, [string]$Base = "multilingual", [string]$Model = "checkpoints/ko-tickets",
      [string]$Data = "data/test.jsonl", [string]$Name)
. "$PSScriptRoot\_env.ps1"
$Device = Resolve-Device $Cpu
$Suffix = if ($Name) { "_$Name" } else { "" }

uv run eval.py $Data --model $Base --device $Device --json "reports/base$Suffix.json"
if ($LASTEXITCODE) { exit $LASTEXITCODE }
uv run eval.py $Data --model $Model --device $Device --json "reports/finetuned$Suffix.json"
if ($LASTEXITCODE) { exit $LASTEXITCODE }

uv run compare.py "reports/base$Suffix.json" "reports/finetuned$Suffix.json" $Data "reports/compare$Suffix.md"
exit $LASTEXITCODE
