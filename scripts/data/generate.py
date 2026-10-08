"""LLM 으로 합성 고객 메시지 생성 -> data/synthetic.jsonl (prelabel/export 가 pool 로 같이 읽음).

LLM 은 문장만 쓴다. 라벨은 요청 조건(QUESTIONS 의 답 조합 x 유형)에서 정해지므로 LLM 이 라벨을 틀릴 수 없다.
응답은 data/synthetic_raw/ 에 캐시되어, 다시 실행하면 새 요청만 보낸다.

    $env:LLM_BASE_URL = "https://<your-llm-gateway>/v1"
    $env:LLM_API_KEY = "<api-key>"
    uv run --no-sync scripts/data/generate.py --dry-run                                  # 프롬프트 1개만 출력
    uv run --no-sync scripts/data/generate.py --split train --model claude-haiku-5-5 --rounds 6
    uv run --no-sync scripts/data/generate.py --split test  --model gemini-3-5-flash --rounds 1 --n 5
"""
import argparse
import concurrent.futures
import hashlib
import itertools
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from common import QUESTIONS, build_state

# 도메인에 맞게 수정. 모든 서비스에서 QUESTIONS 의 모든 답이 자연스럽게 나올 수 있어야 한다.
SERVICES = ["종합 온라인 쇼핑몰", "전자제품 쇼핑몰", "패션 쇼핑몰", "생활용품 정기배송 서비스",
            "식료품 새벽배송 앱", "가구·인테리어 쇼핑몰", "반려동물용품 쇼핑몰", "화장품 쇼핑몰"]

KINDS = {
    "indirect": "자기 상황이나 겪는 문제를 구체적으로 설명한다. 부서·팀 이름이나 '~로 연결해 주세요' 같은 말은 하지 않는다.",
    "direct": "담당 업무나 부서를 직접 언급하며 처리나 연결을 요청한다. 절반은 아주 짧게 쓴다.",
    "negated": "먼저 '{other}' 관련 문의는 아니라고 분명히 부정한 뒤, 실제로는 위 업무에 해당하는 요청을 한다.",
    "needs_with_no": "'안 돼요', '없어요', '못 해요', '안 와요' 같은 부정 표현으로 문제를 설명하지만, 결국 위 업무 처리가 필요한 요청이다.",
}

AGES = ["10대", "20대 대학생", "30대 직장인", "40대 자영업자", "50대 주부", "60대 이상 어르신", "회사 구매 담당자"]
TONES = ["공손함", "짜증남", "매우 화남", "무뚝뚝함", "다급함", "차분함", "불안해함", "사정을 장황하게 설명함"]
WRITING = ["띄어쓰기·맞춤법이 엉망", "줄임말·초성(ㅠㅠ, ㅇㅇ, ㄱㅅ 등) 섞음", "반말", "한 줄로 아주 짧게",
           "여러 문장으로 길게", "오타 한두 개", "이모티콘·느낌표 많이", "격식 있는 존댓말", "말끝 흐림(...)",
           "질문 없이 통보하듯", "요청을 문장 끝에 덧붙임"]

SYSTEM = ("너는 한국어 고객 상담 채팅 데이터를 만드는 작가다. 실제 사람이 상담 채팅창에 치는 그대로 쓴다. "
          "교과서 같은 문장, 비슷한 문장 구조 반복, '확인 부탁드립니다'·'문의드립니다' 같은 정형 문구 남발, "
          "'안녕하세요'로 시작하는 문장 반복을 피한다. 출력은 {\"messages\": [\"...\", ...]} 형태의 JSON 만.")


def answer_values(q):
    if q["type"] == "noul":
        return [True, False]
    if q["type"] == "score":
        return list(range(len(q["criteria"])))
    return list(q["criteria"])


def describe(qid, q, value):
    if q["type"] == "noul":
        rule = "'예'가 되는 상황이 분명히 드러나야 한다" if value else "'아니오'여야 한다. '예'로 볼 근거를 넣지 않는다"
        return f"- '{q['instructions']}' -> {rule}"
    if q["type"] == "score":
        return f"- '{q['instructions']}' -> '{q['criteria'][value]}' 수준"
    return f"- '{q['instructions']}' -> {value} ({q['criteria'][value]})"


PRIMARY = next((qid for qid, q in QUESTIONS.items() if q["type"] == "choice"), None)


def jobs(split, model, rounds, n):
    kinds = KINDS if PRIMARY else {"indirect": KINDS["indirect"]}
    combos = itertools.product(*[answer_values(q) for q in QUESTIONS.values()])
    for combo, kind, r in itertools.product(list(combos), kinds, range(rounds)):
        labels = dict(zip(QUESTIONS, combo))
        key = hashlib.sha1(json.dumps([split, model, kind, combo, r, n], ensure_ascii=False).encode()).hexdigest()[:16]
        rng = random.Random(key)
        crit = QUESTIONS[PRIMARY]["criteria"] if PRIMARY else {}
        other = rng.choice([k for k in crit if k != labels.get(PRIMARY)] or [""])
        writers = [f"{i + 1}. {rng.choice(AGES)}, {rng.choice(TONES)}, {rng.choice(WRITING)}" for i in range(n)]
        prompt = "\n".join([
            f"서비스: {rng.choice(SERVICES)}",
            f"아래 조건을 모두 만족하는 고객 채팅 메시지 {n}개를 써라.",
            "", "[답이 정해진 조건]",
            *[describe(qid, QUESTIONS[qid], v) for qid, v in labels.items()],
            "", f"[유형] {kinds[kind].format(other=other and f'{other}({crit[other]})')}",
            "", "[작성자: i번째 메시지는 i번째 작성자가 쓴다]", *writers,
            "", "[공통]",
            "- 메시지마다 구체적인 상황, 상품, 첫 단어를 모두 다르게 한다.",
            "- 위 조건과 어긋나는 내용(다른 업무 요청, 반대되는 답)을 섞지 않는다.",
            "- 고객이 보낸 메시지 본문만 쓴다. 번호나 작성자 설명은 붙이지 않는다.",
        ])
        yield {"key": key, "split": split, "model": model, "kind": kind, "labels": labels, "prompt": prompt}


def call(args, job):
    body = json.dumps({"model": job["model"], "temperature": 1.0, "response_format": {"type": "json_object"},
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": job["prompt"]}]}).encode()
    req = urllib.request.Request(args.base_url.rstrip("/") + "/chat/completions", data=body, headers={
        "Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                content = json.load(resp)["choices"][0]["message"]["content"]
            match = re.search(r"\{.*\}", content, re.S)
            messages = json.loads(match.group(0) if match else content)["messages"]
            return [m.strip() for m in messages if isinstance(m, str) and m.strip()]
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, TypeError) as exc:
            if attempt == 4:
                raise RuntimeError(f"{job['key']}: {exc}") from exc
            time.sleep(2 ** attempt * 3)


def norm(text):
    return re.sub(r"[\s\W_]+", "", text.lower())


def build(raw_dir, out_path):
    """캐시 전체 -> synthetic.jsonl. test 를 먼저 읽어, test 와 겹치는 train 문장은 버린다."""
    seen, rows, stale = set(), [], 0
    files = sorted(raw_dir.glob("test/*.json")) + sorted(raw_dir.glob("train/*.json"))
    for path in files:
        job = json.loads(path.read_text(encoding="utf-8"))
        # QUESTIONS 를 바꾼 뒤 남은 이전 분류기의 캐시는 라벨이 맞지 않으므로 제외
        if set(job["labels"]) != set(QUESTIONS) or any(
                v not in answer_values(QUESTIONS[q]) for q, v in job["labels"].items()):
            stale += 1
            continue
        for text in job["messages"]:
            if not 2 <= len(text) <= 500 or norm(text) in seen:
                continue
            seen.add(norm(text))
            state = build_state([], text)
            rid = hashlib.sha1(json.dumps(state, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
            rows.append({"id": rid, "conversation_id": f"syn-{rid}", "ts": job["ts"], "state": state,
                         "split": job["split"], "labels": job["labels"],
                         "tags": ["synthetic", job["kind"], job["model"]]})
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    by = {s: sum(r["split"] == s for r in rows) for s in ("train", "test")}
    print(f"{out_path}: {len(rows)} rows (train {by['train']}, test {by['test']}) from {len(files) - stale} cached requests"
          + (f", {stale} skipped (labels from different QUESTIONS)" if stale else ""))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "test"], default="train",
                   help="test 는 train 과 다른 모델로 만들고 Argilla 에서 사람이 검수할 것")
    p.add_argument("--model", default="claude-haiku-5-5")
    p.add_argument("--rounds", type=int, default=6, help="조합(답 x 유형)마다 요청 횟수")
    p.add_argument("--n", type=int, default=10, help="요청 1회당 메시지 수")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--base-url", default=os.getenv("LLM_BASE_URL"), help="OpenAI 호환 API 주소 (.../v1)")
    p.add_argument("--api-key", default=os.getenv("LLM_API_KEY"))
    p.add_argument("--raw-dir", default="data/synthetic_raw")
    p.add_argument("--out", default="data/synthetic.jsonl")
    p.add_argument("--dry-run", action="store_true", help="요청 수와 첫 프롬프트만 출력")
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    todo = [j for j in jobs(args.split, args.model, args.rounds, args.n)
            if not (raw_dir / j["split"] / f"{j['key']}.json").exists()]
    total = len(list(jobs(args.split, args.model, args.rounds, args.n)))
    print(f"{total} requests ({total * args.n} messages), cached {total - len(todo)}, to send {len(todo)}")
    if args.dry_run:
        if todo:
            print("\n--- system ---\n" + SYSTEM + "\n--- user ---\n" + todo[0]["prompt"])
        return
    if not args.api_key or not args.base_url:
        raise SystemExit("LLM_BASE_URL, LLM_API_KEY 환경변수(또는 --base-url, --api-key)가 필요합니다")

    (raw_dir / args.split).mkdir(parents=True, exist_ok=True)
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(call, args, j): j for j in todo}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            job = futures[fut]
            try:
                job["messages"] = fut.result()
            except RuntimeError as exc:
                failed += 1
                print(f"  failed {exc}")
                continue
            job["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            (raw_dir / job["split"] / f"{job['key']}.json").write_text(
                json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")
    if failed:
        print(f"{failed} requests failed; 다시 실행하면 실패한 것만 재요청합니다")
    build(raw_dir, Path(args.out))


if __name__ == "__main__":
    main()
