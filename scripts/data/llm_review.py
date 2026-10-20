"""LLM 판정 모델로 합성 데이터를 자동 검수해 Argilla 에 Submit 한다 (사람 검수량 줄이기).

1. 판정: pool 의 합성 레코드를 --model 이 QUESTIONS 기준으로 다시 판정 (캐시: data/llm_review/verdicts-<model>.jsonl).
2. 비교: 생성 라벨과 판정이 같고 쓸 수 있는 문장이면 '일치', 아니면 data/llm_review/disagree.jsonl 로.
3. --apply: 일치분은 생성 라벨로 Submit, data/llm_review/decisions.jsonl 의 결정(사람이 불일치분을 보고 작성)을 반영.
   decisions.jsonl 한 줄: {"id": "...", "labels": {"harmful": false, ...}} 또는 {"id": "...", "discard": true}

    uv run --no-sync scripts/data/llm_review.py
    uv run --no-sync scripts/data/llm_review.py --apply
"""
import argparse
import concurrent.futures
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import argilla as rg

from argilla_io import add_args, client, from_laya, labels, load_pool, record_metadata, render
from common import QUESTIONS

SYSTEM = ("너는 AI 가드레일 학습 데이터 검수자다. 각 입력을 아래 질문 기준으로 엄격히 판정한다. "
          "입력이 실제 사용자가 칠 법한 문장이 아니면(생성 조건 설명이 섞임, 번호·작성자 설명이 붙음, 모델의 거절문, 빈 문장) "
          "usable 을 false 로 한다.\n\n[질문]\n"
          + "\n".join(f"- {qid}: {q['instructions']} (답: {' / '.join(labels(q))})" for qid, q in QUESTIONS.items())
          + '\n\n출력은 {"results": [{"id": "...", ' + ", ".join(f'"{qid}": "..."' for qid in QUESTIONS)
          + ', "usable": true}]} 형태의 JSON 만.')


def post(args, items):
    body = json.dumps({"model": args.model, "response_format": {"type": "json_object"},
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": json.dumps(items, ensure_ascii=False)}]}).encode()
    base = args.base_url if "://" in args.base_url else "https://" + args.base_url
    token = args.llm_key.removeprefix("Bearer ").strip()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body, headers={
        "Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                content = json.load(resp)["choices"][0]["message"]["content"]
            match = re.search(r"\{.*\}", content, re.S)
            results = json.loads(match.group(0) if match else content)["results"]
            want = {i["id"] for i in items}
            got = {r["id"]: r for r in results if r.get("id") in want
                   and all(r.get(qid) in labels(q) for qid, q in QUESTIONS.items())}
            if len(got) < len(want):
                raise ValueError(f"{len(want) - len(got)} items missing or invalid")
            return list(got.values())
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, TypeError) as exc:
            detail = exc.read().decode(errors="replace")[:300] if isinstance(exc, urllib.error.HTTPError) else ""
            if attempt == 4 or (isinstance(exc, urllib.error.HTTPError) and exc.code == 400):
                raise RuntimeError(f"{exc} {detail}") from exc
            time.sleep(2 ** attempt * 3)


def judge(args, rows, cache_path):
    # 다른 모델로 이미 판정한 건도 재사용 (모델을 바꿔 이어서 돌릴 때)
    done = {}
    for path in cache_path.parent.glob("verdicts-*.jsonl"):
        done.update({r["id"]: r for r in map(json.loads, open(path, encoding="utf-8"))})
    todo = [] if args.no_judge else [r for r in rows if r["id"] not in done]
    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    print(f"{len(rows)} records, cached {len(rows) - len(todo)}, to judge {len(todo)} in {len(batches)} requests")
    with open(cache_path, "a", encoding="utf-8") as f, concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(post, args, [{"id": r["id"], "text": r["state"]["current"]} for r in b]) for b in batches]
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                for res in fut.result():
                    done[res["id"]] = res
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
            except RuntimeError as exc:
                print(f"  failed: {exc} (다시 실행하면 재시도)")
            if i % 10 == 0 or i == len(batches):
                print(f"  {i}/{len(batches)}")
    return done


def main():
    p = argparse.ArgumentParser()
    add_args(p)
    p.add_argument("--model", default="claude-opus-5-5", help="생성 모델보다 강한 모델을 쓸 것")
    p.add_argument("--batch", type=int, default=20)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--base-url", default=os.getenv("LLM_BASE_URL"))
    p.add_argument("--llm-key", default=os.getenv("LLM_API_KEY"))
    p.add_argument("--out-dir", default="data/llm_review")
    p.add_argument("--apply", action="store_true", help="일치분과 decisions.jsonl 을 Argilla 에 Submit")
    p.add_argument("--limit", type=int, help="처음 N 건만 판정 (시험용)")
    p.add_argument("--no-judge", action="store_true", help="새로 판정하지 않고 캐시된 판정만 사용")
    args = p.parse_args()
    if not args.base_url or not args.llm_key:
        raise SystemExit("LLM_BASE_URL, LLM_API_KEY 가 필요합니다 (.env)")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in load_pool(args.pool) if r.get("labels")][:args.limit]
    verdicts = judge(args, rows, out / f"verdicts-{args.model}.jsonl")

    agree, disagree = [], []
    for r in rows:
        v = verdicts.get(r["id"])
        if v is None:
            continue
        gen = {qid: from_laya(QUESTIONS[qid], val) for qid, val in r["labels"].items()}
        same = v.get("usable", True) and all(v[qid] == gen[qid] for qid in QUESTIONS)
        (agree if same else disagree).append({"id": r["id"], "split": r["split"], "kind": r["tags"][1],
                                              "text": r["state"]["current"], "generated": gen, "judge": v})
    with open(out / "disagree.jsonl", "w", encoding="utf-8") as f:
        for d in disagree:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"agree {len(agree)}, disagree {len(disagree)} -> {out / 'disagree.jsonl'}, not judged {len(rows) - len(verdicts)}")
    if not args.apply:
        return

    decisions = {}
    if (out / "decisions.jsonl").exists():
        decisions = {d["id"]: d for d in map(json.loads, open(out / "decisions.jsonl", encoding="utf-8"))}
    open_ids = {d["id"] for d in disagree} - set(decisions)
    if open_ids:
        print(f"불일치 {len(open_ids)} 건은 decisions.jsonl 에 결정이 없어 Submit 하지 않습니다")

    c = client(args)
    ds = c.datasets(name=args.dataset, workspace=args.workspace)
    user_id = c.me.id
    reviewed = {rec.id for status in ("submitted", "discarded")
                for rec in ds.records(query=rg.Query(filter=rg.Filter(("response.status", "==", status))))}
    # 이미 사람이 Submit/Discard 한 레코드는 덮어쓰지 않는다
    by_id = {r["id"]: r for r in rows if r["id"] not in reviewed}
    print(f"이미 검수된 {len(reviewed)} 건은 건너뜀")
    records = []
    for a in agree:
        if a["id"] not in decisions:
            records.append((a["id"], a["generated"], "submitted"))
    for rid, d in decisions.items():
        if d.get("discard"):
            records.append((rid, {qid: labels(q)[0] for qid, q in QUESTIONS.items()}, "discarded"))
        else:
            records.append((rid, {qid: from_laya(QUESTIONS[qid], v) for qid, v in d["labels"].items()}, "submitted"))
    for i in range(0, len(records), 200):
        ds.records.log([rg.Record(id=rid, fields={"conversation": render(by_id[rid]["state"])},
                                  metadata=record_metadata(by_id[rid]),
                                  responses=[rg.Response(qid, val, user_id=user_id, status=status) for qid, val in vals.items()])
                        for rid, vals, status in records[i:i + 200] if rid in by_id])
    records = [r for r in records if r[0] in by_id]
    print(f"Argilla: submitted {sum(s == 'submitted' for _, _, s in records)}, discarded {sum(s == 'discarded' for _, _, s in records)}")


if __name__ == "__main__":
    main()
