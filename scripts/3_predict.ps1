# Step 3: ask the fine-tuned checkpoint about one sentence
#   .\3_predict.ps1 "<sentence>"          (GPU if available; -Cpu to force CPU)
#   .\3_predict.ps1                       (prompts for sentences until empty input)
param([string]$Text, [string]$Checkpoint = "checkpoints/ko-tickets", [switch]$Cpu)
. "$PSScriptRoot\_env.ps1"
$Device = Resolve-Device $Cpu

if ($Text) {
    uv run predict.py $Checkpoint $Text --device $Device
    exit $LASTEXITCODE
}
while ($line = Read-Host "text (empty to quit)") {
    uv run predict.py $Checkpoint $line --device $Device
}
