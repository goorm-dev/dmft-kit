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
