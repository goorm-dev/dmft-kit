# 결정 모델 파인튜닝 키트

## 실행 순서 요약

GPU 가 기본이며, CUDA 를 쓸 수 없으면 자동으로 CPU 로 실행된다 (`-Cpu` 는 CPU 강제). 자세한 옵션은 아래 [2. 전체 흐름](#2-전체-흐름).

```powershell
# 최초 1회
.\scripts\setup.ps1                                       # NVIDIA GPU 면 CUDA torch, 아니면 CPU torch 설치
docker compose -f argilla/docker-compose.yaml up -d     # Argilla: http://localhost:6900 (argilla / 12345678)
$env:LLM_BASE_URL = "https://<your-llm-gateway>/v1"     # 합성 데이터를 만들 때만
$env:LLM_API_KEY = "<api-key>"

.\scripts\0_collect.ps1 -Generate         # 0. 수집 → Argilla 업로드   (실제 로그: -Csv logs.csv)
#    ▶ Argilla 화면에서 검수 (Submit / Discard) — 검수를 마친 뒤 다음 단계로
.\scripts\1_prepare_data.ps1              # 1. 검수 결과 → data/train.jsonl, data/test.jsonl + 검증  (데모: -Demo)
.\scripts\2_finetune.ps1                  # 2. 기본 모델 평가 → 파인튜닝 → 평가
.\scripts\compare.ps1                     # 3. 기본 모델과 비교 → reports/compare.md
.\scripts\3_predict.ps1 "문장"            # 4. 직접 넣어보기
```

macOS / Linux / WSL 은 같은 이름의 `.sh` 스크립트 (옵션은 `--소문자-케밥`, [macOS · WSL](#macos--linux--wsl-bash)):

```bash
bash scripts/setup.sh
export LLM_BASE_URL="https://<your-llm-gateway>/v1" LLM_API_KEY="<api-key>"
bash scripts/0_collect.sh --generate      # ▶ 이후 Argilla 검수
bash scripts/1_prepare_data.sh
bash scripts/2_finetune.sh
bash scripts/compare.sh
bash scripts/3_predict.sh "문장"
```

**`0_collect` → Argilla 검수 → `1_prepare_data` 순서.** `1_prepare_data` 실행 시 주의:

- Argilla 서버가 켜져 있어야 한다 (검수 결과를 Argilla 에서 읽어온다).
- **test 는 Submit 한 것만** 들어간다. test 검수를 안 하면 `test 가 비었습니다` 경고가 나온다.
  검수 순서: `split = test` 전부 → `kind = negated`, `needs_with_no` 우선 → 나머지 train 표본.
- 합성 train 은 검수를 다 안 해도 된다. 미검수분은 생성 요청의 라벨로 들어간다 (Discard 한 것은 제외).
  판정 모델과 라벨이 갈린 train 은 `llm_review.py --apply --discard-train-disagree` 로 미리 Discard 해 둔다.
- 검수를 더 했으면 `1_prepare_data` 를 다시 실행하면 된다. 매번 Submit 된 전체를 새로 내보낸다.

Argilla·API 키 없이 흐름만 확인: `1_prepare_data.ps1 -Demo` → `2_finetune.ps1` → `compare.ps1` → `3_predict.ps1`.

다른 데이터·다른 목적의 분류기를 만들려면 [4. 다른 분류기 만들기](#4-다른-분류기-만들기-데이터-갈아끼우기).

---

[laya](https://github.com/NandhaKishorM/laya) 결정 모델을 한국어 고객 문의(부서 분류 + 긴급 여부)에 맞춰
파인튜닝하고 평가하는 프로젝트. 데이터 수집 → 검수 → 학습 → 비교 → 직접 확인까지.
아래 예시는 Windows PowerShell 기준이며, macOS·Linux·WSL 은 같은 이름의 `.sh` 를 쓴다 ([옵션 대응표](#macos--linux--wsl-bash)).

```
 실제 로그 CSV ─┐
                ├─ 0_collect ─▶ Argilla (사람 검수) ─▶ 1_prepare_data ─▶ 2_finetune ─▶ compare ─▶ 3_predict
 LLM 합성 데이터 ┘   ingest/generate/prelabel               export·검증          학습·평가     비교       직접 입력
```

## 1. 준비 (최초 1회)

```powershell
# uv 설치 (이미 있으면 생략)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

Set-ExecutionPolicy -Scope Process Bypass    # .ps1 실행이 막히면 (터미널마다)
.\scripts\setup.ps1                         # 의존성 설치 (uv sync 대신)
$env:PYTHONUTF8 = "1"                        # 스크립트는 자동 설정. python 을 직접 돌릴 때 한글 깨짐 방지
```

**GPU / CPU**: GPU 가 기본, 없으면 CPU 로 자동 전환된다.

- 설치: `setup.ps1` 이 NVIDIA GPU 를 감지하면 `uv sync --extra gpu` (CUDA 12.6 torch, `whl/cu126`),
  없으면 `uv sync --extra cpu` (CPU 전용 torch, `whl/cpu`, 훨씬 작음). 강제: `.\scripts\setup.ps1 -Torch cpu|gpu`.
  끝에 `torch 버전 | cuda available` 을 출력한다.
- 실행: 각 스크립트가 `torch.cuda.is_available()` 을 확인해 `cuda`, 안 되면 경고 후 `cpu` (드라이버가 오래돼 CUDA 를 못 쓰는 경우 포함).
  `2_finetune` 은 CPU 일 때 결정 헤드만 학습하는 설정으로 바뀐다. `-Cpu` 로 CPU 강제.
- 스크립트는 `UV_NO_SYNC=1` 로 실행된다 (`uv run` 이 기본 동기화로 torch 를 지우지 않게). 직접 실행할 때는
  `uv run --no-sync python eval.py ...` 처럼 `--no-sync` 를 붙이고, 의존성을 바꿨으면 `setup.ps1` 을 다시 실행한다.

**Argilla 서버** (검수 화면, Docker Desktop 필요. Elasticsearch 포함 RAM 2~3GB 사용):

```powershell
docker compose -f argilla/docker-compose.yaml up -d      # 중지: ... down
```

브라우저에서 http://localhost:6900 , 로그인 `argilla` / `12345678`.
다른 서버(HF Spaces 등)를 쓰면 `$env:ARGILLA_API_URL`, `$env:ARGILLA_API_KEY` 를 설정한다 (기본 `http://localhost:6900`, `argilla.apikey`).

**LLM API** (합성 데이터 생성 시에만, OpenAI 호환):

```powershell
$env:LLM_BASE_URL = "https://<your-llm-gateway>/v1"    # /chat/completions 를 제공하는 주소
$env:LLM_API_KEY = "<api-key>"                          # 파일·커밋에 저장하지 말 것
```

### macOS / Linux / WSL (bash)

`scripts/*.sh` 가 각 `.ps1` 과 같은 일을 한다. bash 3.2(macOS 기본)에서 동작.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # uv 설치 (이미 있으면 생략)
bash scripts/setup.sh                                     # nvidia-smi 로 GPU 감지 → --extra gpu / cpu
docker compose -f argilla/docker-compose.yaml up -d
export LLM_BASE_URL="https://<your-llm-gateway>/v1" LLM_API_KEY="<api-key>"   # 합성 데이터 생성 시
export ARGILLA_DATASET=reviews                            # 다른 분류기일 때 (4장)
```

| PowerShell | bash |
|---|---|
| `setup.ps1 -Torch cpu` | `setup.sh --torch cpu` |
| `0_collect.ps1 -Csv logs.csv -Generate -Rounds 8` | `0_collect.sh --csv logs.csv --generate --rounds 8` |
| `-TrainModel gemini-3-5-flash,claude-sonnet-4-5` | `--train-model gemini-3-5-flash,claude-sonnet-4-5` |
| `-IngestArgs '--id-col','room_id'` | `0_collect.sh --csv logs.csv -- --id-col room_id` (`--` 뒤는 그대로 ingest 로) |
| `-TrainModel` `-TestModel` `-TestRounds` `-TestN` `-LabelModel` `-SkipSyntheticTrain` | `--train-model` `--test-model` `--test-rounds` `--test-n` `--label-model` `--skip-synthetic-train` |
| `1_prepare_data.ps1 -Demo -ReviewedOnly -Base english` | `1_prepare_data.sh --demo --reviewed-only --base english` |
| `2_finetune.ps1 -Cpu -Epochs 6 -Base english -Out checkpoints/x` | `2_finetune.sh --cpu --epochs 6 --base english --out checkpoints/x` |
| `compare.ps1 -Model checkpoints/x -Data data/test.jsonl -Name test` | `compare.sh --model checkpoints/x --data data/test.jsonl --name test` |
| `3_predict.ps1 "문장" -Checkpoint checkpoints/x` | `3_predict.sh "문장" --checkpoint checkpoints/x` |

PowerShell 판과 다른 점:

- **Apple Silicon**: 장치 선택이 CUDA → **MPS** → CPU. setup 은 `--extra cpu` 를 설치하지만 macOS 휠은 MPS 를 포함한다.
  `2_finetune.sh` 는 MPS 에서 전체 학습을 작은 배치(`--micro-batch 2 --grad-accum 16`, laya 의 Apple Silicon 스크립트와 동일)로 돌린다.
- **가상환경이 OS 별로 분리**: `.venv-linux`, `.venv-darwin` (Windows 는 `.venv`). WSL 에서 Windows 와 같은 폴더를 써도 서로 덮어쓰지 않는다.
  직접 `uv run` 할 때는 `UV_PROJECT_ENVIRONMENT=.venv-linux uv run --no-sync ...` 처럼 지정한다.
- WSL 의 GPU 는 Windows NVIDIA 드라이버가 WSL 을 지원해야 보인다 (`nvidia-smi` 가 WSL 안에서 동작하는지 확인).
- Windows 에서 clone 한 파일이라 실행 권한이 없으면 `bash scripts/x.sh` 로 실행하거나 `chmod +x scripts/*.sh`.
  줄바꿈은 `.gitattributes` 로 LF 고정.

## 2. 전체 흐름

### 0단계: 수집 → Argilla 업로드

```powershell
# 실제 채팅 로그 (메시지 1행 CSV: conversation_id, timestamp, role, text)
.\scripts\0_collect.ps1 -Csv data\sample_chat.csv

# 운영 로그가 없으면 LLM 합성 데이터 (train ~1,260개 + 다른 모델로 test ~105개)
.\scripts\0_collect.ps1 -Generate

# 둘 다 + 파인튜닝 모델 예측을 실제 로그의 라벨 초안으로
.\scripts\0_collect.ps1 -Csv logs.csv -Generate -LabelModel checkpoints/ko-tickets
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `-Csv` | | 실제 로그. 열 이름이 다르면 `-IngestArgs '--id-col','room_id','--text-col','message','--user-role','customer'` (엑셀 한글 CSV 는 `'--encoding','cp949'`) |
| `-Generate` | | LLM 합성 데이터 생성 |
| `-TrainModel` / `-Rounds` | `gemini-3-5-flash,claude-sonnet-4-5` / 6 | train 생성 모델(여러 개면 요청을 모델별로 고르게 나눔), 조합당 요청 횟수. 요청 1회 = 10문장 |
| `-TestModel` / `-TestRounds` / `-TestN` | `gpt-4-1` / 1 / 5 | test 는 **train 과 다른 모델**로. 겹치면 실행하지 않는다 |
| `-LabelModel` | 없음 | 실제 로그에 laya 예측을 라벨 초안으로 (`multilingual` 또는 체크포인트 경로) |
| `-SkipSyntheticTrain` | | 합성 train 을 Argilla 에 올리지 않음 (검수 없이 학습에 사용) |

여러 번 실행해도 안전하다: 이미 Argilla 에 있는 레코드는 다시 올리지 않고(사람 라벨 보호), LLM 응답은
`data/synthetic_raw/` 에 캐시되어 새 요청만 보낸다. 실패한 요청은 다시 실행하면 그것만 재시도한다.

`generate.py` 직접 실행 (`0_collect` 가 내부에서 부르는 명령):

```powershell
uv run --no-sync scripts/data/generate.py --split train --model gemini-3-5-flash claude-sonnet-4-5 --rounds 6
uv run --no-sync scripts/data/generate.py --split test --model gpt-4-1 --rounds 1 --n 5
uv run --no-sync scripts/data/generate.py --dry-run          # 요청 수(모델별)와 첫 프롬프트만 출력
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--model` | `gemini-3-5-flash claude-sonnet-4-5` | 여러 개면 요청을 모델별로 번갈아 나눈다. 다른 split 을 만든 모델이 들어 있으면 실행하지 않는다 |
| `--rounds` / `--n` | 6 / 10 | 조합(답 x 유형)마다 요청 횟수 / 요청 1회당 메시지 수 |
| `--raw-dir` / `--out` | `data/synthetic_raw` / `data/synthetic.jsonl` | 캐시 폴더 / 출력. 시험 생성은 다른 폴더로 (아래) |

- 응답 메시지가 정확히 `--n` 개이고 `finish_reason` 이 정상일 때만 캐시에 저장한다. 아니면 재시도한다.
- 400 류 오류와 `finish_reason=content_filter` 는 같은 프롬프트면 결과가 같아 재시도하지 않는다.
- 실패한 요청은 `<raw-dir>/failures.jsonl` 에 한 줄씩 남는다: `ts`, `key`, `split`, `model`, `kind`,
  `error`(HTTP 응답 본문 또는 `finish_reason` 과 응답 끝부분).
- 실행마다 `<raw-dir>/<split>.keys.json` 에 이번 실행의 요청 목록이 저장된다. `synthetic.jsonl` 은 이 목록의 캐시만으로
  만든다. 모델·`--n`·`--rounds` 를 바꿔 다시 돌리면 이전 실행의 캐시는 출력에서 빠진다 (파일은 남는다).
- `PROMPT_VERSION` 을 올리면 이전 버전 캐시는 쓰지 않는다. 현재 7.

생성 결과 품질 점검과 LLM 자동 검수:

```powershell
uv run --no-sync scripts/data/check.py          # split 별 라벨·멀티턴 비율, 길이·통제 어휘 편향, 거절 history 라벨
                                                #  + data/check/injection_sample-<split>.md (injection '예' 무작위 표본)
                                                #  + data/check/review_sample-<split>.md   (모델 x kind x 라벨 층화 표본)
uv run --no-sync scripts/data/llm_review.py     # 판정 모델로 재판정 (멀티턴은 history 포함) -> data/llm_review/to_review.md
uv run --no-sync scripts/data/llm_review.py --apply --discard-train-disagree   # Argilla 에 반영
```

`check.py` 옵션: 입력 파일(기본 `data/synthetic.jsonl`), `--sample`(injection 표본 수, 기본 50, 0 이면 표본 파일을 만들지 않음),
`--per-group`(층화 표본의 그룹당 건수, 기본 3), `--out-dir`(기본 `data/check`), `--seed`.
H1I1 평균 길이가 H0I0 의 1.5배를 넘거나, 거절문이 든 history 중 '예' 비율이 20~80% 를 벗어나면 경고한다.
층화 표본에는 "멀티턴 + 20자 이하 current" 그룹도 따로 있어, 한 행이 두 그룹에 나올 수 있다.
표본에서 라벨 오류를 세어 10% 를 넘으면 `generate.py` 프롬프트를 고쳐 `PROMPT_VERSION` 을 올리고 다시 생성한다.
전체 생성 전에 `--rounds 1 --n 5 --raw-dir data/pilot/raw --out data/pilot/synthetic.jsonl` 로 작게 먼저 돌려 보고,
`check.py data/pilot/synthetic.jsonl --out-dir data/pilot/check` 로 점검한다.

`llm_review.py` 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--model` | `claude-opus-5-5` | 판정 모델. 생성 모델보다 강한 모델을 쓴다 |
| `--pool` / `--out-dir` | `data/pool.jsonl data/synthetic.jsonl` / `data/llm_review` | 판정할 레코드 / 결과 폴더 |
| `--batch` / `--workers` | 20 / 8 | 요청 1회당 건수 / 동시 요청 수 |
| `--limit` | | 처음 N 건만 판정 (시험용) |
| `--no-judge` | | 새로 판정하지 않고 캐시된 판정만 사용 |
| `--reuse-other-models` | | 다른 판정 모델의 캐시도 재사용. 판정 기준이 같은 것만 |
| `--apply` | | 일치분은 생성 라벨로 Submit, `decisions.jsonl` 의 결정을 반영 (Argilla 서버 필요) |
| `--discard-train-disagree` | | `--apply` 때 결정이 없는 train 불일치를 Discard 해 학습에서 뺀다. 판정 모델이 거부한(보류) 건은 빼지 않는다 |

- 판정 캐시(`verdicts-<model>.jsonl`)에는 판정 기준 해시(`policy`)가 들어간다. `QUESTIONS` 나 판정 지시문을 바꾸면
  예전 판정은 무시되고 다시 판정한다. 형식이 틀린 캐시 줄도 다시 판정한다.
- 판정 모델의 콘텐츠 필터에 걸린 배치는 반씩 나눠 다시 보낸다. 한 건만 남아도 걸리면 그 건은 '보류'로 `to_review.md` 에 들어간다.
- 불일치는 `disagree.jsonl`, 그중 결정이 없는 건은 사람 검수 목록 `to_review.md` 로 나온다.
- 사람이 정한 결정은 같은 폴더의 `decisions.jsonl` 에 한 줄씩 쓴다. 라벨은 `QUESTIONS` 의 모든 질문을 빠짐없이 쓴다:
  ```json
  {"id": "a05423e490001ba8", "labels": {"harmful": true, "injection": false}}
  {"id": "57caeb8c372a95f4", "discard": true}
  ```
  Argilla 에 올리기 전에 검사해, pool 에 없는 id, 빠진 질문, 틀린 값(`"discard": "false"` 등), 중복 id 가 있으면
  해당 줄을 알려 주고 멈춘다.

### 사람 검수 (Argilla 화면)

라벨 초안이 채워져 있으니 확인 후 **Submit**, 쓸 수 없으면 **Discard**. 메타데이터 필터 순서:

1. `split = test` → **전부** 검수. test 는 Submit 한 것만 평가에 쓰인다.
2. `kind = negated`, `needs_with_no` → 부정 표현 문장. 생성 오류가 가장 많은 곳.
3. 나머지 train 은 표본만. 100~200건 중 불량이 10% 를 넘으면 검수보다 `scripts/data/generate.py`
   프롬프트(`KINDS`, `describe()`)를 고쳐 다시 생성하는 게 빠르다.

| 상태 | export 에서 |
|---|---|
| Submit | 사람 라벨 사용 (여럿이 갈리면 train 은 투표 비율, test 는 다수결) |
| Discard | 제외 |
| 미검수 합성 train | 생성 요청의 라벨로 사용 (`-ReviewedOnly` 면 제외) |
| 미검수 test / 실제 로그 | 제외 |

### 1단계: 학습 데이터 내보내기 + 검증

```powershell
.\scripts\1_prepare_data.ps1                 # Argilla → data/train.jsonl, data/test.jsonl
.\scripts\1_prepare_data.ps1 -ReviewedOnly   # 사람이 Submit 한 것만
.\scripts\1_prepare_data.ps1 -Demo           # Argilla 없이 내장 데모 데이터(48/16건)로 흐름만 확인
```

출력에서 볼 것: train/test 건수, train 라벨 분포(너무 적은 클래스), dry-run 의 skip 사유
(`options_collapsed` = 선택지 이름이 토큰상 겹침, `options_beyond_max_len` = 선택지가 너무 김).
실행마다 `data/versions/<시각>/` 에 사본이 남는다.

### 2단계: 파인튜닝 + 평가

```powershell
.\scripts\2_finetune.ps1                # GPU: 인코더까지 전체 학습 (VRAM ~16GB, 4 epoch). CUDA 없으면 CPU 로 자동 전환
.\scripts\2_finetune.ps1 -Cpu           # CPU 강제: 결정 헤드만 학습 (10 epoch)
.\scripts\2_finetune.ps1 -Epochs 6
.\scripts\2_finetune.ps1 -Base english -Out checkpoints/reviews   # 영어 데이터, 다른 분류기
```

epoch 기본값/권장: GPU 전체 학습 **4** (laya 기본값, 데이터 수만 건 이상이면 2~3), CPU 헤드만 학습 **10** (5~10).
데이터가 수백 건 이하면 epoch 를 늘리기보다 데이터를 늘린다. 2~3개 값만 `compare` 로 비교해 정한다.

기본 모델(`multilingual`) 평가 → 학습(`checkpoints/ko-tickets`) → 학습 모델 평가 순서.
마지막 `laya-evals compare` 가 exit code 1 이면 어떤 지표가 기본 모델보다 나빠졌다는 뜻이다(오류 아님).

**CPU 헤드 학습 소요 시간**: train 1,080건 기준 epoch 당 약 80분 (10 epoch 면 13시간 이상).
`laya-train` 은 중간부터 이어서 학습하는 기능이 없어, 중단되면 처음부터 다시 돌려야 한다.
오래 걸리므로 로그를 파일로 남겨 두는 것을 권한다. 진행률(stderr)도 함께 남는다:

```powershell
.\scripts\2_finetune.ps1 -Cpu *> reports\finetune.log
```

### 3단계: 기본 모델과 비교

```powershell
.\scripts\compare.ps1
.\scripts\compare.ps1 -Base english -Model checkpoints/v2 -Data data/versions/20261008-120000/test.jsonl
```

`reports/compare.md` 에 저장된다. 지표별 ✅ 개선 / ❌ 악화 판정, 질문별 재현율·정밀도·오탐률,
기준값별(0.5, 0.4, 0.3, 0.2) 재현율·오탐률, kind별 오답률, 고쳐진 사례(base 오답 → ft 정답)·
망가진 사례(base 정답 → ft 오답) 목록.
`-Name test` 를 주면 `reports/compare_test.md`, `base_test.json`, `finetuned_test.json` 으로 저장해
다른 비교 결과를 덮어쓰지 않는다. 테스트셋이 작으면 한 건에 수 %p 가 움직이니 사례 목록을 같이 본다.
버전끼리 비교할 때는 export 할 때마다 test 구성이 바뀌므로 `data/versions/` 의 test 하나를 고정해 `-Data` 로 넘긴다.

### 4단계: 직접 넣어보기

```powershell
.\scripts\3_predict.ps1 "결제가 두 번 됐어요. 오늘 환불해 주세요."
.\scripts\3_predict.ps1                        # 문장을 계속 입력 (빈 줄이면 종료)
```

## 3. 반복 개선 루프

1. `compare` 의 망가진 사례를 보고 원인을 찾는다 (예: 특정 말투가 한 부서에만 있음, 표현이 학습 데이터에 없음).
2. 데이터 보강: 실제 로그를 더 수집하거나, 생성 프롬프트·`SERVICES` 를 고쳐 `0_collect.ps1 -Generate -Rounds 8` 등으로 추가 생성.
   **test 문장을 학습 데이터에 직접 넣지 말 것** (점수만 오르고 실제 성능은 그대로).
3. 파인튜닝한 모델을 `-LabelModel checkpoints/ko-tickets` 로 넘기면 실제 로그 초안이 정확해져 검수가 빨라진다.
4. `1_prepare_data` → `2_finetune` → `compare` 반복. 데이터가 적을 땐 `--seed` 를 바꿔 같은 사례가 매번 틀리는지 확인.

## 4. 다른 분류기 만들기 (데이터 갈아끼우기)

분류기의 정의는 **`scripts/data/common.py` 의 `QUESTIONS` 하나**다. Argilla 질문 화면, 학습, 평가,
예측(`questions.json` 이 체크포인트에 같이 저장됨)이 모두 여기서 나온다. 나머지는 데이터를 이 형식으로 넣는 문제다.

### 체크리스트

1. **질문 정의** — `common.py` 의 `QUESTIONS` 를 새 목적에 맞게 바꾼다 (아래 "질문 설계").
2. **새 Argilla 데이터셋 이름** — 질문이 바뀌면 기존 데이터셋을 쓸 수 없다. 터미널에서:
   ```powershell
   $env:ARGILLA_DATASET = "reviews"      # prelabel/export 가 이 이름을 쓴다
   ```
3. **이전 작업 데이터** — `data/pool.jsonl`(ingest 가 덮어씀), `data/synthetic.jsonl`(현재 `QUESTIONS` 에 맞지 않는
   캐시는 자동 제외), `data/train.jsonl`·`test.jsonl`(export 가 덮어씀, 과거본은 `data/versions/`) 이라 그대로 둬도 되지만,
   보관하려면 `data/` 를 통째로 다른 이름으로 복사해 둔다.
4. **데이터 넣기** — 원본 형식에 따라 아래 표.
5. **(합성 데이터를 쓰면)** `scripts/data/generate.py` 수정 (아래 "합성 데이터 프롬프트").
6. **베이스 모델** — 한국어·다국어는 `multilingual`, 영어만이면 `english`(더 큼, 영어 정확도 높음).
   `-Base` 를 `1_prepare_data`, `2_finetune`, `compare` 에 같은 값으로 넘긴다.
7. **체크포인트 경로** — 덮어쓰지 않게 새 이름으로:
   ```powershell
   .\scripts\2_finetune.ps1 -Out checkpoints/reviews
   .\scripts\compare.ps1 -Model checkpoints/reviews
   .\scripts\3_predict.ps1 "문장" -Checkpoint checkpoints/reviews
   .\scripts\0_collect.ps1 -Csv new.csv -LabelModel checkpoints/reviews     # 다음 수집의 라벨 초안
   ```

### 질문 설계

| type | 답 | 예 |
|---|---|---|
| `choice` | `criteria` 중 하나. `{"라벨": "설명"}` | 부서, 감정, 주제, 의도 |
| `noul` | 예/아니오 확률 | 긴급한가, 답변이 필요한가, 개인정보가 있는가 |
| `score` | 순서 있는 단계. `["낮음", "보통", "높음"]` (라벨은 인덱스 0,1,2) | 별점, 심각도 |

```python
# 예: 쇼핑몰 리뷰 분류기
QUESTIONS = {
    "sentiment": {"type": "choice", "instructions": "current 리뷰의 전반적인 감정은?",
                  "criteria": {"positive": "만족, 칭찬, 재구매 의사",
                               "negative": "불만, 비판, 환불·반품 언급",
                               "mixed": "장점과 단점이 함께 있음"}},
    "needs_reply": {"type": "noul", "instructions": "판매자의 답변이나 조치가 필요한 리뷰인가?"},
}
```

- 라벨 이름은 짧고 서로 다르게 (`options_collapsed` 방지). 설명문(`criteria` 값)은 모델이 같이 읽으므로 구체적으로.
- 선택지 20개 이상이면 `2_finetune.ps1` 의 `laya-train` 에 `--shuffle-options` 추가.
- 어느 것에도 안 맞는 입력이 들어올 수 있으면 `"other"` 같은 선택지를 둔다.
- `instructions` 는 Argilla 화면 질문 제목으로도 쓰인다. 대화 이력이 없는 데이터면 "current" 대신 그냥 "이 리뷰" 등으로.

### 원본 데이터 형식별 수정 위치

| 원본 형식 | 넣는 방법 | 코드 수정 |
|---|---|---|
| 채팅 로그, 메시지 1행 (`대화ID, 시각, 화자, 내용`) | `0_collect.ps1 -Csv logs.csv -IngestArgs '--id-col','room_id','--time-col','created_at','--role-col','sender','--text-col','message','--user-role','customer'` | 없음 |
| 글 1개 = 1행 (리뷰, 티켓, 메일, 게시글) | `-IngestArgs '--single-text','--text-col','body','--time-col','created_at'` (시각 열이 없으면 행 순서, 뒤쪽 20% 가 test) | 없음 |
| 엑셀에서 저장한 한글 CSV | 위 인자에 `'--encoding','cp949'` 추가 | 없음 |
| 시각이 ISO 형식이 아님 (`2026. 10. 8. 오전 9:00`) | | `ingest.py` 의 `msgs.sort(...)` 와 `"ts"` 를 `datetime.strptime` 로 파싱한 ISO 값으로 |
| 텍스트가 여러 열 (제목 + 본문) | | `ingest.py` 의 `text = (m[args.text_col] ...)` 를 `f"{m['title']}\n{m['body']}"` 처럼 합치기 |
| 대화 전체가 한 칸에 (JSON 배열 등) | | `ingest.py` 에서 `convs[대화ID] = [{role, text, 시각 열}, ...]` 를 만드는 앞부분만 다시 작성 |
| JSON, 엑셀, DB | CSV 로 내보내 위 방법 사용, 또는 아래 스키마로 `data/pool.jsonl` 을 직접 작성 | 변환 스크립트 |
| **이미 라벨이 있음** (텍스트 열 + 라벨 열 1개) | Argilla 없이 바로 학습: `uv run --no-sync laya-train --data labeled.csv --text-column body --label-column sentiment --base multilingual --out checkpoints/reviews` (`questions.json` 자동 생성) | 없음 |
| **이미 라벨이 있음** (질문 여러 개) | 아래 학습 JSONL 형식으로 `data/train.jsonl`, `data/test.jsonl` 을 직접 작성 → `uv run --no-sync laya-evals validate data/train.jsonl` → `2_finetune.ps1` (`1_prepare_data` 는 건너뜀, 실행하면 export 가 덮어씀) | 변환 스크립트 |

**pool 스키마** (`data/pool.jsonl`, Argilla 업로드 전 단계, 한 줄에 하나):

```json
{"id": "고유ID", "conversation_id": "대화ID(분할 단위)", "ts": "2026-10-08T09:00:00",
 "state": {"history": [{"role": "user", "text": "..."}], "current": "판단 대상 텍스트"}}
```

`state` 는 반드시 `common.build_state(history, current)` 로 만든다 (마스킹·길이 제한 포함). 운영에서 laya 에
넣는 입력도 같은 함수로 만들어야 학습과 형식이 맞는다.

**학습 JSONL 형식** (`data/train.jsonl`, `data/test.jsonl`):

```json
{"state": {...}, "questions": {QUESTIONS 그대로}, "expected": {"sentiment": "negative", "needs_reply": true}}
```

`expected` 값: choice 는 라벨 이름, noul 은 `true`/`false`, score 는 단계 인덱스(0부터).
확률 라벨은 `"gold": {"sentiment": {"probabilities": {"positive": 0.2, "negative": 0.8, "mixed": 0.0}}}`
(noul 은 `"true"`/`"false"` 키, score 는 `"0"`, `"1"`... 키).

### 합성 데이터 프롬프트 (`scripts/data/generate.py`)

현재 프롬프트는 "고객 상담 채팅 + 부서 라우팅" 기준으로 쓰여 있다. 목적이 다르면 아래를 바꾼다.

| 항목 | 역할 | 예: 리뷰 분류기 |
|---|---|---|
| `SYSTEM` | 작가 역할, 문체 규칙 | "쇼핑몰 구매 리뷰를 쓰는 작가. 실제 리뷰 말투 그대로..." |
| `SERVICES` | 요청마다 무작위로 고르는 배경 | 상품 카테고리 목록 (의류, 가전, 식품, ...) |
| `KINDS` | 문장 유형 (첫 번째 `choice` 질문 기준) | `indirect`: 감정을 직접 말하지 않고 경험만 서술 / `direct`: 감정을 직접 표현 / `negated`: "나쁘진 않은데..." 같은 부정 표현 / `sarcasm`: 반어 |
| `AGES`, `TONES`, `WRITING` | 문장별 작성자 다양화 | 대부분 그대로 사용 가능 |
| `describe()` | 답 조건 문구 | noul 의 '예/아니오' 기준을 목적에 맞게 엄격히 |

- 요청 수 = (모든 질문 답의 조합 수) × (유형 수) × `-Rounds`. 질문·선택지가 늘면 빠르게 커지므로
  먼저 `uv run --no-sync scripts/data/generate.py --dry-run --rounds 2` 로 요청 수와 프롬프트를 확인하고 `-Rounds` 를 정한다.
- 서로 모순인 조합(예: `sentiment=positive` + `needs_reply=true` 가 드물다면)은 `jobs()` 의 루프에
  `if labels["sentiment"] == "positive" and labels["needs_reply"]: continue` 처럼 건너뛰기를 넣는다.

## 5. 커스터마이즈

| 바꾸려는 것 | 수정할 곳 |
|---|---|
| 질문(부서 목록, 긴급 기준 등) | `scripts/data/common.py` 의 `QUESTIONS` + `$env:ARGILLA_DATASET` 새 이름 (위 4장) |
| 대화 이력 길이, 마스킹 규칙 | `common.py` 의 `HISTORY_TURNS`, `MAX_CHARS`, `_MASKS` (`uv run --no-sync scripts/data/common.py` 로 self-check). 운영 코드도 같은 `build_state` 를 써야 한다 |
| 합성 데이터의 서비스 종류·유형·문체 | `scripts/data/generate.py` 의 `SERVICES`, `KINDS`, `AGES`/`TONES`/`WRITING` |
| CSV 열 이름, 인코딩 | `-IngestArgs` (위 표). 시각이 ISO 형식이 아니면 `ingest.py` 의 정렬 부분 수정 |
| 학습 하이퍼파라미터 | `scripts/2_finetune.ps1` 의 `laya-train` 인자 (`--loss soft-ce`, `--label-smoothing 0.1`, `--shuffle-options` 등) |

## 6. 파일 구조

```
scripts/
  setup  0_collect  1_prepare_data  2_finetune  compare  3_predict   (.ps1: Windows, .sh: macOS/Linux/WSL)
  _env.ps1 / _env.sh   공통 설정 (UTF-8, UV_NO_SYNC, GPU→CPU 장치 선택)
  data/
    common.py      질문 스키마, 마스킹, build_state (학습·운영 공용)
    ingest.py      CSV → data/pool.jsonl
    generate.py    LLM → data/synthetic.jsonl (캐시: data/synthetic_raw/)
    prelabel.py    pool → Argilla (초안 + 메타데이터)
    export.py      Argilla → data/train.jsonl, data/test.jsonl, data/versions/
    check.py       합성 데이터 품질 점검 (편향 수치, injection 표본 → data/check/)
    llm_review.py  판정 모델 자동 검수 → data/llm_review/ (불일치 목록 to_review.md)
    argilla_io.py  Argilla 공용 코드
eval.py          체크포인트 평가 → reports/*.json
compare.py       리포트 두 개 비교 → reports/compare.md
predict.py       문장 하나 예측
data/build_data.py, data/sample_chat.csv   데모 데이터, 입력 형식 예시
argilla/docker-compose.yaml                Argilla 서버
```

## 7. 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| `CUDA not available -> falling back to CPU` | 정상 동작 (GPU 없음). GPU 가 있는데 나오면 `setup.ps1 -Torch gpu` 후 출력의 `cuda available` 확인, NVIDIA 드라이버 560 이상인지 확인 |
| `No .venv: run .\scripts\setup.ps1 first` | 의존성 미설치. `setup.ps1` 실행 |
| 직접 `uv run` 후 torch 가 사라짐 | `--no-sync` 없이 실행해 기본 동기화가 됨. `setup.ps1` 재실행 |
| 학습 중 `calibration: not fitted ... fewer than 10` | 데이터가 적어 확신도 보정이 안 됨. 답은 정상, 확신도 값만 믿지 말 것 |
| export 가 `test 가 비었습니다` | test 를 아직 Submit 하지 않음. Argilla 에서 `split = test` 검수 |
| export 의 `not in pool` 이 0 보다 큼 | Argilla 에 있는 레코드의 원본이 pool 파일에 없음 (pool 파일을 지웠거나 덮어씀) |
| 생성 중 `finish_reason=content_filter` 로 같은 요청이 계속 실패 | 생성 모델의 콘텐츠 필터. 같은 프롬프트면 다시 보내도 걸린다. 빠진 요청의 kind·라벨은 `failures.jsonl` 에서 확인하고, 한 조합에 몰렸으면 다른 모델로 그 조합을 보충한다 |
| `decisions.jsonl 오류` 로 `llm_review.py` 가 멈춤 | 출력된 줄의 id·질문·값을 고친다. `discard` 는 `true` 만, 라벨은 모든 질문을 `true`/`false` 로 |
| 생성 중 `failed ... HTTP 500` | `data/synthetic_raw/failures.jsonl` 의 본문·`finish_reason` 확인. 게이트웨이가 모델의 출력 토큰 한도 초과(`length`)를 500 으로 돌려주기도 한다. 계속 실패하는 모델은 빼고 다시 실행 |
| `*>` 로 로그를 남기면 스크립트가 중단됨 | PowerShell 5.1 이 리다이렉트된 진행률(stderr)을 오류로 보고 `$ErrorActionPreference = "Stop"` 에서 중단하던 문제. 지금은 `_env.ps1` 이 `uv` 호출에만 `Continue` 를 적용해 그대로 리다이렉트하면 된다 |
| 한글이 깨짐 | `$env:PYTHONUTF8 = "1"`. CSV 는 `-IngestArgs '--encoding','cp949'` |
| 이전에 올린 레코드에 메타데이터 필터가 안 먹힘 | 메타데이터는 새로 올리는 레코드에만 붙는다. 검수 전이면 새 `--dataset` 이름으로 다시 올린다 |
