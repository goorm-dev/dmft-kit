# Step 0: collect data and upload it to Argilla for review (run before labeling)
#   .\scripts\0_collect.ps1 -Csv logs.csv                       real chat logs only
#   .\scripts\0_collect.ps1 -Generate                           LLM-generated data only (needs $env:LLM_BASE_URL, $env:LLM_API_KEY)
#   .\scripts\0_collect.ps1 -Generate -TrainModel gemini-3-5-flash,claude-sonnet-4-5 -TestModel gpt-4-1
#   .\scripts\0_collect.ps1 -Csv logs.csv -Generate -LabelModel checkpoints/ko-tickets   (GPU if available, -Cpu to force)
#   .\scripts\0_collect.ps1 -Csv logs.csv -IngestArgs '--id-col','room_id','--user-role','customer'
param(
    [string]$Csv,
    [string[]]$IngestArgs = @(),
    [switch]$Generate,
    [string[]]$TrainModel = @("gemini-3-5-flash", "claude-sonnet-4-5"),
    [int]$Rounds = 6,
    [string]$TestModel = "gpt-4-1",
    [int]$TestRounds = 1,
    [int]$TestN = 5,
    [string]$LabelModel,
    [switch]$SkipSyntheticTrain,
    [switch]$Cpu
)
. "$PSScriptRoot\_env.ps1"

if (-not $Csv -and -not $Generate) { Write-Host "Nothing to collect: pass -Csv and/or -Generate"; exit 2 }

if ($Csv) {
    uv run scripts/data/ingest.py $Csv @IngestArgs
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
}

if ($Generate) {
    if (-not $env:LLM_API_KEY -or -not $env:LLM_BASE_URL) { Write-Host 'Set $env:LLM_BASE_URL and $env:LLM_API_KEY first'; exit 2 }
    if ($TrainModel -contains $TestModel) { Write-Host "-TestModel $TestModel is also in -TrainModel; test must use a different model"; exit 2 }
    uv run scripts/data/generate.py --split train --model @TrainModel --rounds $Rounds
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    uv run scripts/data/generate.py --split test --model $TestModel --rounds $TestRounds --n $TestN
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
}

$prelabel = @()
if ($LabelModel) { $prelabel += "--model", $LabelModel, "--device", (Resolve-Device $Cpu) }
if ($SkipSyntheticTrain) { $prelabel += "--skip-synthetic-train" }
uv run scripts/data/prelabel.py @prelabel
exit $LASTEXITCODE
