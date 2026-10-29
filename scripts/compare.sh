#!/usr/bin/env bash
# Evaluate base and fine-tuned checkpoints on the same test set, then compare side by side
#   scripts/compare.sh                (GPU if available; --cpu to force CPU)
#   scripts/compare.sh --base english --model checkpoints/other --data data/test.jsonl
#   scripts/compare.sh --name test    -> reports/compare_test.md, base_test.json, finetuned_test.json (keeps other runs)
. "$(dirname "$0")/_env.sh"

cpu=0 base=multilingual model=checkpoints/ko-tickets data=data/test.jsonl suffix=""
while [ $# -gt 0 ]; do
    case "$1" in
        --cpu) cpu=1; shift ;;
        --base) base="$2"; shift 2 ;;
        --model) model="$2"; shift 2 ;;
        --data) data="$2"; shift 2 ;;
        --name) suffix="_$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

device=$(resolve_device "$cpu")
uv run eval.py "$data" --model "$base" --device "$device" --json "reports/base$suffix.json"
uv run eval.py "$data" --model "$model" --device "$device" --json "reports/finetuned$suffix.json"
uv run compare.py "reports/base$suffix.json" "reports/finetuned$suffix.json" "$data" "reports/compare$suffix.md"
