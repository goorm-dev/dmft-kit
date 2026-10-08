#!/usr/bin/env bash
# Step 1: labels -> data/train.jsonl, data/test.jsonl, then validate (no weights loaded, no training)
#   scripts/1_prepare_data.sh                  export from Argilla (Submitted + unreviewed synthetic train)
#   scripts/1_prepare_data.sh --reviewed-only  export only what a human Submitted
#   scripts/1_prepare_data.sh --demo           built-in demo data instead of Argilla
. "$(dirname "$0")/_env.sh"

demo=0 reviewed=0 base=multilingual
while [ $# -gt 0 ]; do
    case "$1" in
        --demo) demo=1; shift ;;
        --reviewed-only) reviewed=1; shift ;;
        --base) base="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [ "$demo" = 1 ]; then
    uv run data/build_data.py
elif [ "$reviewed" = 1 ]; then
    uv run scripts/data/export.py --reviewed-only
else
    uv run scripts/data/export.py
fi

uv run laya-evals validate data/train.jsonl
uv run laya-evals validate data/test.jsonl
# Tokenizes with the base checkpoint's config and reports skipped questions and the reason
uv run laya-train --data data/train.jsonl --base "$base" --dry-run
