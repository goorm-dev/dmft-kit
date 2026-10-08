#!/usr/bin/env bash
# Step 3: ask the fine-tuned checkpoint about one sentence
#   scripts/3_predict.sh "<sentence>"     (GPU if available; --cpu to force CPU)
#   scripts/3_predict.sh                  (prompts for sentences until empty input)
. "$(dirname "$0")/_env.sh"

cpu=0 checkpoint=checkpoints/ko-tickets text=""
while [ $# -gt 0 ]; do
    case "$1" in
        --cpu) cpu=1; shift ;;
        --checkpoint) checkpoint="$2"; shift 2 ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *) text="$1"; shift ;;
    esac
done

device=$(resolve_device "$cpu")
if [ -n "$text" ]; then
    uv run predict.py "$checkpoint" "$text" --device "$device"
    exit
fi
while IFS= read -r -p "text (empty to quit): " line && [ -n "$line" ]; do
    uv run predict.py "$checkpoint" "$line" --device "$device"
done
