"""합성 데이터 품질 점검: split 별 라벨·멀티턴 비율, 길이·통제 어휘 편향, 거절 history 라벨,
injection '예' 무작위 표본과 모델 x kind x 라벨 층화 표본 (사람 검수용 markdown).

    uv run --no-sync scripts/data/check.py                          # data/synthetic.jsonl
    uv run --no-sync scripts/data/check.py --sample 50 --out-dir data/check
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

# generate.py 의 '은근하게' 지시와 같은 어휘. 이 단어가 injection 라벨과 엮이면 모델이 단어만 보고 맞힌다.
CONTROL_WORDS = re.compile(r"규칙|제한|무시|필터|검열|모드|권한|관리자|지침|설정|시스템|프롬프트|지시|명령")
REFUSAL = re.compile(r"죄송|도와\s?드릴 수 없|도울 수 없|드릴 수 없|할 수 없습니다|어렵습니다|불가능합니다|응할 수 없")
LENGTH_WARN = 1.5
REFUSAL_POSITIVE = (0.2, 0.8)  # 거절 history 중 '예' 비율이 이 범위를 벗어나면 경고


def combo(labels):
    return "".join(f"{q[0].upper()}{int(bool(v))}" for q, v in labels.items())


def pct(part, whole):
    return f"{part / whole:.1%}" if whole else "-"


def report(split, rows, out_dir, sample, per_group, seed):
    print(f"\n## {split} ({len(rows)}건, 생성 모델 {dict(Counter(r['tags'][2] for r in rows))})")
    neg = sum(not any(r["labels"].values()) for r in rows)
    multi = sum(bool(r["state"]["history"]) for r in rows)
    print(f"- 둘 다 아니오: {pct(neg, len(rows))}, 멀티턴: {pct(multi, len(rows))}")

    lengths = {}
    for r in rows:
        lengths.setdefault(combo(r["labels"]), []).append(len(r["state"]["current"]))
    avg = {c: sum(v) / len(v) for c, v in sorted(lengths.items())}
    print("- current 평균 길이: " + ", ".join(f"{c} {a:.0f}자 ({len(lengths[c])}건)" for c, a in avg.items()))
    if "H1I1" in avg and "H0I0" in avg:
        ratio = avg["H1I1"] / avg["H0I0"]
        print(f"  H1I1 / H0I0 = {ratio:.2f}" + (f"  ⚠ {LENGTH_WARN}배 초과" if ratio > LENGTH_WARN else ""))

    if any("injection" in r["labels"] for r in rows):
        inj = [r for r in rows if r["labels"].get("injection")]
        clean = [r for r in rows if r["labels"].get("injection") is False]
        hit_inj, hit_clean = (sum(bool(CONTROL_WORDS.search(r["state"]["current"])) for r in rs) for rs in (inj, clean))
        print(f"- 통제 어휘 비율: injection 예 {pct(hit_inj, len(inj))}, 아니오 {pct(hit_clean, len(clean))}")

    refused = [r for r in rows if any(h["role"] == "assistant" and REFUSAL.search(h["text"]) for h in r["state"]["history"])]
    print(f"- assistant 거절문이 든 history: {len(refused)}건, 라벨 분포 {dict(Counter(combo(r['labels']) for r in refused))}")
    if refused:
        share = sum(any(r["labels"].values()) for r in refused) / len(refused)
        if not REFUSAL_POSITIVE[0] <= share <= REFUSAL_POSITIVE[1]:
            print(f"  ⚠ 거절 history 중 '예'가 하나라도 있는 비율 {share:.0%}: 거절문 자체가 라벨 신호가 된다")

    if not sample:
        return
    rng = random.Random(seed)
    inj = [r for r in rows if r["labels"].get("injection")]
    write_sample(out_dir / f"injection_sample-{split}.md", f"injection '예' 무작위 표본 ({split})",
                 "오탐(injection 이 아닌데 '예')이면 체크한다.", "오탐", {"": rng.sample(inj, min(sample, len(inj)))})
    # 모델·유형·라벨 조합마다 몇 건씩. 새로 넣은 유형(짧은 승낙형 current 등)이 무작위 표본에서 빠지지 않게 한다
    groups = {}
    for r in rows:
        groups.setdefault(f"{r['tags'][2]} / {r['tags'][1]} / {combo(r['labels'])}", []).append(r)
    short = [r for r in rows if r["state"]["history"] and len(r["state"]["current"]) <= 20]
    if short:
        groups["멀티턴 + 20자 이하 current"] = short
    write_sample(out_dir / f"review_sample-{split}.md", f"층화 표본 ({split}, 그룹당 최대 {per_group}건)",
                 "라벨이 틀리면 체크한다. 그룹 제목의 H/I 는 생성 라벨이다.", "라벨 틀림",
                 {g: rng.sample(rs, min(per_group, len(rs))) for g, rs in sorted(groups.items())})


def write_sample(path, title, guide, check, groups):
    lines, total = [], 0
    for group, picked in groups.items():
        lines += [f"## {group} ({len(picked)}건)", ""] if group else []
        for r in picked:
            total += 1
            lines += [f"### {total}. `{r['id']}` ({r['tags'][1]}, {r['tags'][2]}, {combo(r['labels'])})", f"- [ ] {check}"]
            lines += [f"- [{h['role']}] {h['text']}" for h in r["state"]["history"]]
            lines += [f"- **current**: {r['state']['current']}", ""]
    path.write_text("\n".join([f"# {title}, {total}건", "", guide, ""] + lines), encoding="utf-8")
    print(f"- {title} -> {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", nargs="?", default="data/synthetic.jsonl", help="generate.py 출력 (split·labels·tags 포함)")
    p.add_argument("--sample", type=int, default=50, help="split 별 injection '예' 표본 수 (0 이면 표본 파일 안 만듦)")
    p.add_argument("--per-group", type=int, default=3, help="층화 표본의 그룹(모델 x kind x 라벨)당 건수")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="data/check")
    args = p.parse_args()

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"# {args.data}")
    for split in ("train", "test"):
        if part := [r for r in rows if r["split"] == split]:
            report(split, part, out_dir, args.sample, args.per_group, args.seed)


if __name__ == "__main__":
    main()
