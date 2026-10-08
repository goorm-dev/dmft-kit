"""파인튜닝된 체크포인트에 직접 문장을 넣어보기.

    uv run --no-sync predict.py ./checkpoints/ko-tickets "결제가 두 번 됐어요. 오늘 환불해 주세요." [--device cpu]
"""
import argparse
import json

import laya

p = argparse.ArgumentParser()
p.add_argument("checkpoint")
p.add_argument("text")
p.add_argument("--device", default="cuda")
args = p.parse_args()

agent = laya.load(args.checkpoint, device=args.device)
questions = json.load(open(f"{args.checkpoint}/questions.json", encoding="utf-8"))
print(json.dumps(agent.predict(args.text, questions)["answers"], ensure_ascii=False, indent=2))
