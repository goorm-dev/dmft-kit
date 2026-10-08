"""라벨 -> data/train.jsonl, data/test.jsonl (+ data/versions/<시각>/ 사본).

- 사람 라벨: Argilla 에서 Submit 된 것. 라벨러 의견이 갈리면 train 은 투표 비율(gold), test 는 다수결(expected).
- 합성 train: 생성 요청의 라벨을 검수 없이 사용 (Argilla 에서 Submit/Discard 한 건 그 결정을 따름).
  --reviewed-only 면 사람이 Submit 한 것만 쓴다.
- 합성 test: 사람이 Submit 한 것만 test 에 들어간다.
- 실제 로그: 대화 시작 시각이 가장 늦은 --test-frac 비율의 대화가 test.

    uv run --no-sync scripts/data/export.py
"""
import argparse
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import argilla as rg

from argilla_io import add_args, client, from_laya, labels, load_pool, to_laya
from common import QUESTIONS

p = argparse.ArgumentParser()
add_args(p)
p.add_argument("--test-frac", type=float, default=0.2)
p.add_argument("--out-dir", default="data")
p.add_argument("--reviewed-only", action="store_true")
args = p.parse_args()

pool = load_pool(args.pool)
ds = client(args).datasets(name=args.dataset, workspace=args.workspace)


def records(status):
    return ds.records(query=rg.Query(filter=rg.Filter(("response.status", "==", status)))) if ds else []


human = {rec.id: {qid: Counter(r.value for r in rec.responses[qid] if r.status == "submitted") for qid in QUESTIONS}
         for rec in records("submitted")}
discarded = {rec.id for rec in records("discarded")} - set(human)

labeled = []
for row in pool:
    if row["id"] in human:
        labeled.append((row, human[row["id"]], "human"))
    elif (not args.reviewed_only and row.get("labels") and row.get("split") == "train"
          and row["id"] not in discarded):
        votes = {qid: Counter([from_laya(QUESTIONS[qid], v)]) for qid, v in row["labels"].items()}
        labeled.append((row, votes, "synthetic"))
missing = len(set(human) - {r["id"] for r in pool})


def gold(q, counter):
    n = sum(counter.values())
    probs = {v: counter.get(v, 0) / n for v in labels(q)}
    if q["type"] == "noul":
        return {"probabilities": {"true": probs["yes"], "false": probs["no"]}}
    return {"probabilities": probs}


def to_row(row, votes, split):
    base = {"state": row["state"], "questions": QUESTIONS, "tags": row.get("tags", [])}
    disagree = any(len(c) > 1 for c in votes.values())
    if split == "train" and disagree:
        return {**base, "gold": {qid: gold(QUESTIONS[qid], c) for qid, c in votes.items()}}, disagree
    return {**base, "expected": {qid: to_laya(QUESTIONS[qid], c.most_common(1)[0][0]) for qid, c in votes.items()}}, disagree


starts = defaultdict(lambda: "\uffff")
for row, _, _ in labeled:
    if "split" not in row:
        starts[row["conversation_id"]] = min(starts[row["conversation_id"]], row["ts"])
ordered = sorted(starts, key=starts.get)
test_convs = set(ordered[len(ordered) - round(len(ordered) * args.test_frac):]) if args.test_frac > 0 else set()

out = {"train": [], "test": []}
count = Counter()
n_disagree = 0
dist = defaultdict(Counter)
for row, votes, source in labeled:
    split = row.get("split") or ("test" if row["conversation_id"] in test_convs else "train")
    r, d = to_row(row, votes, split)
    out[split].append(r)
    count[(split, "synthetic" if row.get("labels") else "real", source)] += 1
    n_disagree += d
    if split == "train":
        for qid, c in votes.items():
            dist[qid][c.most_common(1)[0][0]] += 1

out_dir = Path(args.out_dir)
version = out_dir / "versions" / datetime.now().strftime("%Y%m%d-%H%M%S")
version.mkdir(parents=True)
for split, rows in out.items():
    path = out_dir / f"{split}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.copy(path, version / path.name)

print(f"human-submitted {len(human)} (not in pool: {missing}), discarded {len(discarded)}, disagreements {n_disagree}")
print(f"train {len(out['train'])} / test {len(out['test'])} rows -> {version}")
for (split, kind, source), n in sorted(count.items()):
    print(f"  {split:5} {kind:9} labeled by {source:9}: {n}")
for qid, c in dist.items():
    print(f"  train {qid}: {dict(c)}")
if not out["test"]:
    print("경고: test 가 비었습니다. 합성 test 를 Argilla 에서 검수(Submit)하거나 실제 로그를 넣으세요.")
