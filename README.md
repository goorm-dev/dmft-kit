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
| `-IngestArgs '--id-col','room_id'` | `0_collect.sh --csv logs.csv -- --id-col room_id` (`--` 뒤는 그대로 ingest 로) |
| `-TrainModel` `-TestModel` `-TestRounds` `-TestN` `-LabelModel` `-SkipSyntheticTrain` | `--train-model` `--test-model` `--test-rounds` `--test-n` `--label-model` `--skip-synthetic-train` |
| `1_prepare_data.ps1 -Demo -ReviewedOnly -Base english` | `1_prepare_data.sh --demo --reviewed-only --base english` |
| `2_finetune.ps1 -Cpu -Epochs 6 -Base english -Out checkpoints/x` | `2_finetune.sh --cpu --epochs 6 --base english --out checkpoints/x` |
| `compare.ps1 -Model checkpoints/x -Data data/test.jsonl` | `compare.sh --model checkpoints/x --data data/test.jsonl` |
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

# 운영 로그가 없으면 LLM 합성 데이터 (train ~1,900개 + 다른 모델로 test ~160개)
.\scripts\0_collect.ps1 -Generate

# 둘 다 + 파인튜닝 모델 예측을 실제 로그의 라벨 초안으로
.\scripts\0_collect.ps1 -Csv logs.csv -Generate -LabelModel checkpoints/ko-tickets
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `-Csv` | | 실제 로그. 열 이름이 다르면 `-IngestArgs '--id-col','room_id','--text-col','message','--user-role','customer'` (엑셀 한글 CSV 는 `'--encoding','cp949'`) |
| `-Generate` | | LLM 합성 데이터 생성 |
| `-TrainModel` / `-Rounds` | `claude-haiku-5-5` / 6 | train 생성 모델, 조합(32개)당 요청 횟수. 요청 1회 = 10문장 |
| `-TestModel` / `-TestRounds` / `-TestN` | `gemini-3-5-flash` / 1 / 5 | test 는 **train 과 다른 모델**로 |
| `-LabelModel` | 없음 | 실제 로그에 laya 예측을 라벨 초안으로 (`multilingual` 또는 체크포인트 경로) |
| `-SkipSyntheticTrain` | | 합성 train 을 Argilla 에 올리지 않음 (검수 없이 학습에 사용) |

여러 번 실행해도 안전하다: 이미 Argilla 에 있는 레코드는 다시 올리지 않고(사람 라벨 보호), LLM 응답은
`data/synthetic_raw/` 에 캐시되어 새 요청만 보낸다. 실패한 요청은 다시 실행하면 그것만 재시도한다.

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

### 3단계: 기본 모델과 비교

```powershell
.\scripts\compare.ps1
.\scripts\compare.ps1 -Base english -Model checkpoints/v2 -Data data/versions/20261008-120000/test.jsonl
```

`reports/compare.md` 에 저장된다. 지표별 ✅ 개선 / ❌ 악화 판정과, 고쳐진 사례(base 오답 → ft 정답)·
망가진 사례(base 정답 → ft 오답) 목록. 테스트셋이 작으면 한 건에 수 %p 가 움직이니 사례 목록을 같이 본다.
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
| 한글이 깨짐 | `$env:PYTHONUTF8 = "1"`. CSV 는 `-IngestArgs '--encoding','cp949'` |
| 이전에 올린 레코드에 메타데이터 필터가 안 먹힘 | 메타데이터는 새로 올리는 레코드에만 붙는다. 검수 전이면 새 `--dataset` 이름으로 다시 올린다 |
