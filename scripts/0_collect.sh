#!/usr/bin/env bash
# Step 0: collect data and upload it to Argilla for review (run before labeling)
#   scripts/0_collect.sh --csv logs.csv                     real chat logs only
#   scripts/0_collect.sh --generate                         LLM-generated data only (needs LLM_BASE_URL, LLM_API_KEY)
#   scripts/0_collect.sh --generate --train-model gemini-3-5-flash,claude-sonnet-4-5 --test-model gpt-4-1
#   scripts/0_collect.sh --csv logs.csv --generate --label-model checkpoints/ko-tickets   (GPU if available, --cpu to force)
#   scripts/0_collect.sh --csv logs.csv -- --id-col room_id --user-role customer         (args after -- go to ingest.py)
. "$(dirname "$0")/_env.sh"

csv="" generate=0 train_model="gemini-3-5-flash,claude-sonnet-4-5" rounds=6 test_model="gpt-4-1" test_rounds=1 test_n=5
label_model="" skip_synth=0 cpu=0
ingest_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --csv) csv="$2"; shift 2 ;;
        --generate) generate=1; shift ;;
        --train-model) train_model="$2"; shift 2 ;;
        --rounds) rounds="$2"; shift 2 ;;
        --test-model) test_model="$2"; shift 2 ;;
        --test-rounds) test_rounds="$2"; shift 2 ;;
        --test-n) test_n="$2"; shift 2 ;;
        --label-model) label_model="$2"; shift 2 ;;
        --skip-synthetic-train) skip_synth=1; shift ;;
        --cpu) cpu=1; shift ;;
        --) shift; ingest_args=("$@"); break ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$csv" ] && [ "$generate" = 0 ]; then echo "Nothing to collect: pass --csv and/or --generate" >&2; exit 2; fi

if [ -n "$csv" ]; then
    uv run scripts/data/ingest.py "$csv" ${ingest_args[@]+"${ingest_args[@]}"}
fi

if [ "$generate" = 1 ]; then
    if [ -z "${LLM_API_KEY:-}" ] || [ -z "${LLM_BASE_URL:-}" ]; then echo "Set LLM_BASE_URL and LLM_API_KEY first" >&2; exit 2; fi
    IFS=, read -r -a train_models <<< "$train_model"
    for m in "${train_models[@]}"; do
        if [ "$m" = "$test_model" ]; then echo "--test-model $test_model is also in --train-model; test must use a different model" >&2; exit 2; fi
    done
    uv run scripts/data/generate.py --split train --model "${train_models[@]}" --rounds "$rounds"
    uv run scripts/data/generate.py --split test --model "$test_model" --rounds "$test_rounds" --n "$test_n"
fi

prelabel=()
if [ -n "$label_model" ]; then prelabel+=(--model "$label_model" --device "$(resolve_device "$cpu")"); fi
if [ "$skip_synth" = 1 ]; then prelabel+=(--skip-synthetic-train); fi
uv run scripts/data/prelabel.py ${prelabel[@]+"${prelabel[@]}"}
