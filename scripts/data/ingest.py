"""CSV -> data/pool.jsonl (마스킹·중복 제거).

채팅 로그(메시지 1행): 고객 메시지마다 1개 예시, 직전 대화 이력 포함.
    uv run --no-sync scripts/data/ingest.py data/sample_chat.csv
    uv run --no-sync scripts/data/ingest.py logs.csv --id-col room_id --time-col created_at --role-col sender --text-col message --user-role customer

글 1개 = 1행 (리뷰, 티켓, 게시글 등): 행마다 1개 예시, 이력 없음. --id-col/--time-col 은 열이 있을 때만 사용.
    uv run --no-sync scripts/data/ingest.py reviews.csv --single-text --text-col body --time-col created_at
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict

from common import HISTORY_TURNS, build_state

p = argparse.ArgumentParser()
p.add_argument("csv")
p.add_argument("--out", default="data/pool.jsonl")
p.add_argument("--id-col", default="conversation_id")
p.add_argument("--time-col", default="timestamp", help="정렬 기준. ISO 형식(2026-10-08T09:00:00)이어야 문자열 정렬이 맞다")
p.add_argument("--role-col", default="role")
p.add_argument("--text-col", default="text")
p.add_argument("--user-role", default="user", help="고객 메시지의 role 값")
p.add_argument("--turns", type=int, default=HISTORY_TURNS)
p.add_argument("--encoding", default="utf-8-sig", help="엑셀 한글 CSV 는 cp949 일 수 있음")
p.add_argument("--single-text", action="store_true", help="대화가 아니라 행마다 독립된 글")
args = p.parse_args()

with open(args.csv, encoding=args.encoding, newline="") as f:
    rows = list(csv.DictReader(f))

if args.single_text:
    for i, r in enumerate(rows):
        r[args.id_col] = f"{r.get(args.id_col) or 'row'}#{i}"
        r[args.time_col] = r.get(args.time_col) or f"{i:09d}"
        r[args.role_col] = args.user_role

convs = defaultdict(list)
for r in rows:
    convs[r[args.id_col]].append(r)

seen, out, dups = set(), [], 0
for cid, msgs in convs.items():
    msgs.sort(key=lambda r: r[args.time_col])
    history = []
    for m in msgs:
        text = (m[args.text_col] or "").strip()
        if not text:
            continue
        if m[args.role_col] == args.user_role:
            state = build_state(history, text, args.turns)
            key = json.dumps(state, ensure_ascii=False, sort_keys=True)
            if key in seen:
                dups += 1
            else:
                seen.add(key)
                out.append({"id": hashlib.sha1(key.encode()).hexdigest()[:16],
                            "conversation_id": cid, "ts": m[args.time_col], "state": state})
        history.append({"role": m[args.role_col], "text": text})

with open(args.out, "w", encoding="utf-8") as f:
    for row in out:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"{len(convs)} conversations, {len(rows)} messages -> {len(out)} examples ({dups} duplicates dropped) -> {args.out}")
