#!/usr/bin/env bash
# Step 2: baseline eval -> fine-tune -> eval fine-tuned -> compare
#   scripts/2_finetune.sh            CUDA: full fine-tune (~16 GB VRAM) | Apple MPS: full fine-tune, small batches | else CPU
#   scripts/2_finetune.sh --cpu      force CPU: decision head only (encoder frozen)
#   scripts/2_finetune.sh --base english --out checkpoints/reviews   English data, another classifier
#   --epochs default: 4 on CUDA/MPS (laya default), 10 on CPU head-only
. "$(dirname "$0")/_env.sh"

cpu=0 epochs="" base=multilingual out=checkpoints/ko-tickets
while [ $# -gt 0 ]; do
    case "$1" in
        --cpu) cpu=1; shift ;;
        --epochs) epochs="$2"; shift 2 ;;
        --base) base="$2"; shift 2 ;;
        --out) out="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

device=$(resolve_device "$cpu")
if [ -z "$epochs" ]; then if [ "$device" = cpu ]; then epochs=10; else epochs=4; fi; fi
echo "device=$device epochs=$epochs"

uv run eval.py data/test.jsonl --model "$base" --device "$device" --json reports/base.json

case "$device" in
    cuda) uv run laya-train --data data/train.jsonl --base "$base" --out "$out" --device cuda --epochs "$epochs" ;;
    # same batch settings as laya's Apple Silicon script (notebooks/laya_finetune_typed_decisions_mps.py)
    mps)  uv run laya-train --data data/train.jsonl --base "$base" --out "$out" --device mps --epochs "$epochs" \
              --micro-batch 2 --grad-accum 16 ;;
    *)    uv run laya-train --data data/train.jsonl --base "$base" --out "$out" --device cpu --epochs "$epochs" \
              --freeze-encoder --micro-batch 4 --grad-accum 4 --head-lr 1e-3 ;;
esac

uv run eval.py data/test.jsonl --model "$out" --device "$device" --json reports/finetuned.json

# Exit code 1 here means fine-tuned is worse than base on some metric (not a crash)
uv run laya-evals compare reports/finetuned.json --baseline reports/base.json
