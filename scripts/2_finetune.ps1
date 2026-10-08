# Step 2: baseline eval -> fine-tune -> eval fine-tuned -> compare
#   .\2_finetune.ps1          GPU: full fine-tune (~16 GB VRAM); no CUDA -> falls back to CPU
#   .\2_finetune.ps1 -Cpu     force CPU: decision head only (encoder frozen)
#   .\2_finetune.ps1 -Base english -Out checkpoints/reviews   English data, another classifier
#   -Epochs default: 4 on GPU (laya default), 10 on CPU head-only
param([switch]$Cpu, [int]$Epochs = 0, [string]$Base = "multilingual", [string]$Out = "checkpoints/ko-tickets")
. "$PSScriptRoot\_env.ps1"
$Device = Resolve-Device $Cpu
if (-not $Epochs) { $Epochs = if ($Device -eq "cuda") { 4 } else { 10 } }
Write-Host "device=$Device epochs=$Epochs"

uv run eval.py data/test.jsonl --model $Base --device $Device --json reports/base.json
if ($LASTEXITCODE) { exit $LASTEXITCODE }

if ($Device -eq "cpu") {
    uv run laya-train --data data/train.jsonl --base $Base --out $Out `
        --freeze-encoder --device cpu --epochs $Epochs --micro-batch 4 --grad-accum 4 --head-lr 1e-3
} else {
    uv run laya-train --data data/train.jsonl --base $Base --out $Out `
        --device cuda --epochs $Epochs
}
if ($LASTEXITCODE) { exit $LASTEXITCODE }

uv run eval.py data/test.jsonl --model $Out --device $Device --json reports/finetuned.json
if ($LASTEXITCODE) { exit $LASTEXITCODE }

# Exit code 1 here means fine-tuned is worse than base on some metric (not a crash)
uv run laya-evals compare reports/finetuned.json --baseline reports/base.json
exit $LASTEXITCODE
