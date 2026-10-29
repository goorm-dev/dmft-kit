"""eval.py 리포트 두 개를 나란히 비교: 전체 지표, 질문별 지표, 바뀐 사례(고쳐짐/망가짐).

    uv run --no-sync compare.py reports/base.json reports/finetuned.json data/test.jsonl [reports/compare.md]
"""
import json
import sys

from laya.evals import Dataset

base_path, ft_path, data_path = sys.argv[1:4]
md_path = sys.argv[4] if len(sys.argv) > 4 else None
base = json.load(open(base_path, encoding="utf-8"))
ft = json.load(open(ft_path, encoding="utf-8"))
lines = []


def out(s=""):
    print(s)
    lines.append(s)


LOWER_IS_BETTER = ("score_mae", "ece", "brier", "aurc", "latency_", "cost_")


def verdict(metric, delta):
    if metric == "mean_confidence":
        return "참고 (방향 없음)"
    if abs(delta) < 1e-9:
        return "동일"
    better = delta < 0 if metric.startswith(LOWER_IS_BETTER) else delta > 0
    return "✅ 개선" if better else "❌ 악화"


def table(title, a, b):
    out(f"\n### {title}\n\n| metric | base | finetuned | delta | 판정 |\n|---|---|---|---|---|")
    for m in sorted(set(a) | set(b)):
        x, y = a.get(m), b.get(m)
        both = x is not None and y is not None
        d, v = (f"{y - x:+.4f}", verdict(m, y - x)) if both else ("", "")
        out(f"| {m} | {'' if x is None else f'{x:.4f}'} | {'' if y is None else f'{y:.4f}'} | {d} | {v} |")


table("overall", base["overall"], ft["overall"])
for qid in sorted(set(base["slices"].get("qid", {})) | set(ft["slices"].get("qid", {}))):
    table(f"question: {qid}", base["slices"].get("qid", {}).get(qid, {}), ft["slices"].get("qid", {}).get(qid, {}))

# eval.py 는 on_error=fail 이라 cases 는 (예시 순서 x expected 의 질문 순서) 그대로 쌓인다
keys = [(ex.state, qid) for ex in Dataset.from_jsonl(data_path).examples for qid in ex.expected]
assert len(keys) == len(base["cases"]) == len(ft["cases"]), "리포트와 데이터셋이 다릅니다 (같은 test 파일로 평가했는지 확인)"


def shown(answer):
    if answer.get("type") == "noul":
        return f"{'yes' if answer['noul'] >= 0.5 else 'no'}({answer['noul']:.2f})"
    return str(answer.get("choice", answer.get("score")))


def text(state):
    s = state.get("current", json.dumps(state, ensure_ascii=False)) if isinstance(state, dict) else str(state)
    return s if len(s) <= 60 else s[:57] + "..."


def noul_cases(report, qid):
    return [c for c in report["cases"] if c["qid"] == qid and (c.get("answer") or {}).get("type") == "noul"]


def rates(cases, threshold):
    """noul 질문의 '예' 쪽 재현율·정밀도·오탐률. 분모가 0 이면 None."""
    tp = sum(c["expected"] is True and c["answer"]["noul"] >= threshold for c in cases)
    fp = sum(c["expected"] is False and c["answer"]["noul"] >= threshold for c in cases)
    pos = sum(c["expected"] is True for c in cases)
    neg = sum(c["expected"] is False for c in cases)
    return {"recall": tp / pos if pos else None, "precision": tp / (tp + fp) if tp + fp else None,
            "fpr": fp / neg if neg else None}


def fmt(x):
    return "" if x is None else f"{x:.4f}"


noul_qids = sorted({c["qid"] for c in base["cases"] if (c.get("answer") or {}).get("type") == "noul"})
if noul_qids:
    out("\n### 질문별 재현율·정밀도·오탐률 (기준값 0.5, '예' 기준)\n")
    out("| question | metric | base | finetuned | delta |\n|---|---|---|---|---|")
    for qid in noul_qids:
        a, b = rates(noul_cases(base, qid), 0.5), rates(noul_cases(ft, qid), 0.5)
        for m, name in (("recall", "재현율"), ("precision", "정밀도"), ("fpr", "오탐률")):
            d = f"{b[m] - a[m]:+.4f}" if a[m] is not None and b[m] is not None else ""
            out(f"| {qid} | {name} | {fmt(a[m])} | {fmt(b[m])} | {d} |")

    out("\n### 기준값별 재현율·오탐률\n")
    out("| question | 기준값 | base 재현율 | base 오탐률 | ft 재현율 | ft 오탐률 |\n|---|---|---|---|---|---|")
    for qid in noul_qids:
        for t in (0.5, 0.4, 0.3, 0.2):
            a, b = rates(noul_cases(base, qid), t), rates(noul_cases(ft, qid), t)
            out(f"| {qid} | {t} | {fmt(a['recall'])} | {fmt(a['fpr'])} | {fmt(b['recall'])} | {fmt(b['fpr'])} |")


def kind_of(case):
    tags = case.get("tags") or []
    return tags[1] if len(tags) > 1 else "-"


kinds = sorted({kind_of(c) for c in base["cases"]})
if kinds != ["-"]:
    # 질문을 합치면 한 질문의 좋은 결과가 다른 질문의 약점을 가리므로 질문별로도 나눈다
    out("\n### kind별 오답률\n\n| kind | question | 판정 수 | base | finetuned | delta |\n|---|---|---|---|---|---|")
    for k in kinds:
        for qid in ["전체"] + sorted({c["qid"] for c in base["cases"]}):
            a, b = ([c for c in r["cases"] if kind_of(c) == k and c["correct"] is not None and qid in ("전체", c["qid"])]
                    for r in (base, ft))
            if not a:
                continue
            ea, eb = (sum(c["correct"] is False for c in cs) / len(cs) if cs else None for cs in (a, b))
            d = f"{eb - ea:+.4f}" if ea is not None and eb is not None else ""
            out(f"| {k} | {qid} | {len(a)} | {fmt(ea)} | {fmt(eb)} | {d} |")

fixed, broken = [], []
for (state, qid), b, f in zip(keys, base["cases"], ft["cases"]):
    assert b["qid"] == f["qid"] == qid
    line = f"- [{qid}] expected={b['expected']} base={shown(b['answer'])} ft={shown(f['answer'])} :: {text(state)}"
    if b["correct"] is False and f["correct"] is True:
        fixed.append(line)
    elif b["correct"] is True and f["correct"] is False:
        broken.append(line)

out(f"\n### 고쳐진 사례 (base 오답 -> finetuned 정답): {len(fixed)}")
for line in fixed:
    out(line)
out(f"\n### 망가진 사례 (base 정답 -> finetuned 오답): {len(broken)}")
for line in broken:
    out(line)

if md_path:
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
