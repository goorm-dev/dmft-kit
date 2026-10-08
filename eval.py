"""체크포인트(내장 이름 또는 로컬 경로)를 라벨된 JSONL 로 평가.

    uv run --no-sync eval.py data/test.jsonl --model multilingual --json reports/base.json
    uv run --no-sync eval.py data/test.jsonl --model ./checkpoints/ko-tickets --json reports/finetuned.json

laya-evals run 은 --model 에 로컬 경로를 못 받아서, 같은 하네스(laya.evals)를
Router(models={...}) 로 감싸 호출한다. 결과 JSON 은 `laya-evals compare` 로 비교 가능.
"""
import argparse
import json
import os

import laya
from laya import evals
from laya.evals_cli import RouterRunner

p = argparse.ArgumentParser()
p.add_argument("dataset")
p.add_argument("--model", default="multilingual", help="english | multilingual | typed-decisions | 로컬 경로")
p.add_argument("--device", default="cuda")
p.add_argument("--json", dest="json_out")
args = p.parse_args()

is_path = os.path.isdir(args.model)
name = "target" if is_path else args.model
router = laya.Router(models={"target": args.model} if is_path else None, device=args.device, preload=False)

dataset = evals.Dataset.from_jsonl(args.dataset)
for ex in dataset.examples:
    ex.model = name

report = evals.evaluate(RouterRunner(router), dataset, config={"dataset": args.dataset, "model": args.model})
print(report.to_markdown())
for case in report.cases:
    a = case.get("answer") or {}
    if case.get("correct") is False:
        print(f"  ✗ [{case.get('qid')}] expected={case.get('expected')} got={a.get('choice', a.get('noul'))} conf={case.get('confidence')}")

if args.json_out:
    os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(report.to_json(), f, ensure_ascii=False, indent=2)
