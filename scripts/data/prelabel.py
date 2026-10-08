"""pool(data/pool.jsonl, data/synthetic.jsonl) 중 Argilla 에 아직 없는 예시를 업로드.

라벨 초안(suggestion): 합성 데이터는 생성 요청의 라벨, 그 외는 --model 을 주면 laya 예측.
메타데이터(source/kind/generator/split)가 붙어 Argilla 화면에서 필터링할 수 있다.
합성 train 은 검수하지 않아도 학습에 쓰이므로(Discard 한 것만 제외) 올리기 싫으면 --skip-synthetic-train.

    uv run --no-sync scripts/data/prelabel.py                                   # 초안 없이 업로드
    uv run --no-sync scripts/data/prelabel.py --model multilingual              # 기본 모델 초안
    uv run --no-sync scripts/data/prelabel.py --model checkpoints/ko-tickets    # 파인튜닝 모델 초안
"""
import argparse

import argilla as rg

from argilla_io import add_args, client, ensure_metadata, from_laya, load_pool, record_metadata, render, settings
from common import QUESTIONS

p = argparse.ArgumentParser()
add_args(p)
p.add_argument("--model", default=None)
p.add_argument("--device", default="cuda")
p.add_argument("--chunk", type=int, default=256)
p.add_argument("--skip-synthetic-train", action="store_true")
args = p.parse_args()

c = client(args)
ds = c.datasets(name=args.dataset, workspace=args.workspace)
if ds is None:
    ds = rg.Dataset(name=args.dataset, workspace=args.workspace, settings=settings(), client=c)
    ds.create()
    existing = set()
else:
    ensure_metadata(ds)
    existing = {r["id"] for r in ds.records.to_list(flatten=True)}

# 이미 올라간 레코드는 건드리지 않는다 (사람이 단 라벨 보호)
pool = load_pool(args.pool)
if args.skip_synthetic_train:
    pool = [r for r in pool if not (r.get("labels") and r.get("split") == "train")]
new = [r for r in pool if r["id"] not in existing]
print(f"pool {len(pool)}, already in Argilla {len(pool) - len(new)}, uploading {len(new)}")

agent = None
if args.model:
    import laya
    agent = laya.load(args.model, device=args.device)


def suggestions(answers):
    out = []
    for qid, a in answers.items():
        if a["type"] == "noul":
            value, score = ("yes" if a["noul"] >= 0.5 else "no"), max(a["noul"], 1 - a["noul"])
        elif a["type"] == "score":
            value, score = max(a["probabilities"].items(), key=lambda kv: kv[1])
        else:
            value, score = a["choice"], a["answer_confidence"]
        out.append(rg.Suggestion(qid, value, score=float(score), agent=args.model))
    return out


def row_suggestions(row, answer):
    if row.get("labels"):
        return [rg.Suggestion(qid, from_laya(QUESTIONS[qid], v), score=1.0, agent=f"generated:{row['tags'][-1]}")
                for qid, v in row["labels"].items()]
    return suggestions(answer["answers"]) if answer else []


for i in range(0, len(new), args.chunk):
    chunk = new[i:i + args.chunk]
    answers = agent.predict_batch([r["state"] for r in chunk], QUESTIONS) if agent else [None] * len(chunk)
    ds.records.log([
        rg.Record(id=r["id"], fields={"conversation": render(r["state"])}, metadata=record_metadata(r),
                  suggestions=row_suggestions(r, a))
        for r, a in zip(chunk, answers)
    ])
    print(f"  {i + len(chunk)}/{len(new)}")
