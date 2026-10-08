"""Argilla 연결과 QUESTIONS <-> Argilla 라벨 변환 (prelabel.py, export.py 공용)."""
import argparse
import json
import os

import argilla as rg

from common import QUESTIONS


def add_args(p):
    p.add_argument("--pool", nargs="+", default=["data/pool.jsonl", "data/synthetic.jsonl"],
                   help="없는 파일은 건너뜀")
    p.add_argument("--dataset", default=os.getenv("ARGILLA_DATASET", "ko-tickets"),
                   help="QUESTIONS 를 바꾸면 새 이름으로 ($env:ARGILLA_DATASET)")
    p.add_argument("--workspace", default=None, help="생략하면 기본 워크스페이스")
    p.add_argument("--api-url", default=os.getenv("ARGILLA_API_URL", "http://localhost:6900"))
    p.add_argument("--api-key", default=os.getenv("ARGILLA_API_KEY", "argilla.apikey"))


def client(args: argparse.Namespace):
    return rg.Argilla(api_url=args.api_url, api_key=args.api_key)


def load_pool(paths):
    rows = []
    for path in paths:
        if os.path.exists(path):
            rows += [json.loads(line) for line in open(path, encoding="utf-8")]
    return rows


def labels(q):
    """Argilla 에 보여줄 라벨 값. noul -> yes/no, score -> 레벨 인덱스 문자열."""
    if q["type"] == "noul":
        return ["yes", "no"]
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    return list(q["criteria"])


def to_laya(q, value):
    """Argilla 응답 값 -> laya expected 값."""
    if q["type"] == "noul":
        return value == "yes"
    if q["type"] == "score":
        return int(value)
    return value


def from_laya(q, value):
    """laya expected 값 -> Argilla 라벨 값."""
    if q["type"] == "noul":
        return "yes" if value else "no"
    return str(value)


def settings():
    guide = []
    for qid, q in QUESTIONS.items():
        guide.append(f"## {qid}\n{q['instructions']}")
        crit = q.get("criteria")
        if isinstance(crit, dict):
            guide += [f"- **{k}**: {v}" for k, v in crit.items()]
        elif isinstance(crit, list):
            guide += [f"- **{i}**: {v}" for i, v in enumerate(crit)]
    return rg.Settings(
        guidelines="\n".join(guide),
        fields=[rg.TextField(name="conversation", title="대화 (마지막 >>> 줄이 판단 대상)", use_markdown=False)],
        questions=[rg.LabelQuestion(name=qid, title=q["instructions"], labels=labels(q)) for qid, q in QUESTIONS.items()],
        metadata=metadata_properties(),
    )


METADATA = {"source": "출처 (real/synthetic)", "kind": "생성 유형", "generator": "생성 모델", "split": "용도"}


def metadata_properties():
    return [rg.TermsMetadataProperty(name=name, title=title) for name, title in METADATA.items()]


def ensure_metadata(ds):
    """메타데이터 속성이 없던 기존 데이터셋에 추가."""
    have = {m.name for m in ds.settings.metadata}
    missing = [m for m in metadata_properties() if m.name not in have]
    for m in missing:
        ds.settings.metadata.add(m)
    if missing:
        ds.update()


def record_metadata(row):
    if not row.get("labels"):
        return {"source": "real", "kind": "real", "generator": "-", "split": row.get("split", "auto")}
    _, kind, generator = row["tags"]
    return {"source": "synthetic", "kind": kind, "generator": generator, "split": row["split"]}


def render(state):
    lines = [f"[{m['role']}] {m['text']}" for m in state["history"]]
    return "\n".join(lines + ["", f">>> {state['current']}"])
