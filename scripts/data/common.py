"""질문 스키마 + state 생성 규칙. 학습 데이터와 운영(추론) 코드가 반드시 같은 build_state 를 써야 한다."""
import re

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "current(마지막 고객 메시지)를 처리할 부서는?",
        "criteria": {
            "billing": "결제, 환불, 요금, 청구서, 영수증",
            "technical": "앱 오류, 버그, 접속 장애, 속도 문제",
            "account": "로그인, 비밀번호, 회원정보, 탈퇴, 본인인증",
            "shipping": "배송, 주문 상태, 교환·반품 수거, 주소 변경",
        },
    },
    "urgent": {"type": "noul", "instructions": "current 요청은 오늘 안에 처리해야 하는 긴급 건인가?"},
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
