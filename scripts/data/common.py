"""질문 스키마 + state 생성 규칙. 학습 데이터와 운영(추론) 코드가 반드시 같은 build_state 를 써야 한다."""
import re

# UC12 가드레일 1차 판정 (디지털배움터 AI 실습 채팅). 한 입력이 둘 다 해당할 수 있어 noul 2개로 둔다.
# 기밀정보·위험 시나리오(보이스피싱)는 기준 미결(EDU-943)이라 제외. 확정되면 noul 을 추가하고 ARGILLA_DATASET 을 바꾼다.
# 라벨 초안으로 쓸 kanana-safeguard 와 범위를 맞춤: harmful = 8b 의 S1~S6, injection = prompt 모델의 A1(인젝션)·A2(유출).
QUESTIONS = {
    "harmful": {
        "type": "noul",
        "instructions": ("current(사용자 입력, 앞 대화 맥락 포함)에 특정 사람·집단을 향한 욕설·모욕·혐오·차별·괴롭힘, "
                         "성적 표현이 있거나, 범죄·폭력·자해·자살·무단 접근을 실행하도록 돕는 요청이 있는가? 대상 없는 가벼운 "
                         "비속어 감탄, 그런 주제를 예방·신고·상담·학습 목적으로 묻는 정상 질문은 아니오."),
    },
    "injection": {
        "type": "noul",
        "instructions": ("current(앞 대화 맥락 포함)가 AI 의 기존 지시나 안전 규칙을 무시·변경·우회하게 하거나, "
                         "시스템 프롬프트·내부 설정 공개를 요구하는 공격인가? 답변 형식·말투·길이·첫마디·역할을 정해 주는 "
                         "일반 요청, 프롬프트 작성법을 묻는 정상 질문은 아니오."),
    },
}

HISTORY_TURNS = 4
MAX_CHARS = 1000

# 숫자 앞뒤에 한글 조사가 붙어도 잡히도록 \b 대신 숫자 lookaround 사용
_MASKS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<EMAIL>"),
    (re.compile(r"(?<!\d)\d(?:[ -]?\d){14,15}(?!\d)"), "<CARD>"),
    (re.compile(r"(?<!\d)01[016789][ -]?\d{3,4}[ -]?\d{4}(?!\d)"), "<PHONE>"),
    (re.compile(r"(?<!\d)\d{8,}(?!\d)"), "<NUMBER>"),
]


def mask(text):
    for pattern, tag in _MASKS:
        text = pattern.sub(tag, text)
    return text


def build_state(history, current, turns=HISTORY_TURNS, max_chars=MAX_CHARS):
    """history: current 이전 메시지들 [{"role", "text"}], 오래된 것부터."""
    def clip(text):
        return mask(text.strip())[:max_chars]
    recent = history[-turns:] if turns > 0 else []
    return {"history": [{"role": m["role"], "text": clip(m["text"])} for m in recent],
            "current": clip(current)}


if __name__ == "__main__":
    assert mask("메일은 a.b@x.co.kr 입니다") == "메일은 <EMAIL> 입니다"
    assert mask("010-1234-5678로 연락") == "<PHONE>로 연락"
    assert mask("01012345678") == "<PHONE>"
    assert mask("카드 1234 5678 9012 3456 결제") == "카드 <CARD> 결제"
    assert mask("주문번호 2026100812345번") == "주문번호 <NUMBER>번"
    assert mask("3월 2건, 15000원") == "3월 2건, 15000원"
    h = [{"role": "user", "text": f"m{i}"} for i in range(6)]
    s = build_state(h, " 지금 ", turns=2)
    assert [m["text"] for m in s["history"]] == ["m4", "m5"] and s["current"] == "지금"
    assert build_state(h, "x", turns=0)["history"] == []
    print("ok")
