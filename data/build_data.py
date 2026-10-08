"""데모 데이터셋(한국어 고객문의 분류) -> data/train.jsonl, data/test.jsonl

laya-train / eval.py 가 읽는 JSONL 스키마: {"state", "questions", "expected"}.
자기 데이터로 바꾸려면 QUESTIONS 와 TRAIN/TEST 만 교체하면 된다.
"""
import json
from pathlib import Path

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "이 문의를 처리할 부서는?",
        "criteria": {
            "billing": "결제, 환불, 요금, 청구서, 영수증",
            "technical": "앱 오류, 버그, 접속 장애, 속도 문제",
            "account": "로그인, 비밀번호, 회원정보, 탈퇴, 본인인증",
            "shipping": "배송, 주문 상태, 교환·반품 수거, 주소 변경",
        },
    },
    "urgent": {"type": "noul", "instructions": "오늘 안에 처리해야 하는 긴급한 요청인가?"},
}

# (문의, 부서, 긴급)
TRAIN = [
    ("3월 요금이 두 번 결제됐어요. 오늘 중으로 중복 결제 환불해 주세요.", "billing", True),
    ("지난달 영수증을 이메일로 다시 받을 수 있을까요?", "billing", False),
    ("카드 결제가 계속 실패해서 지금 구독이 끊기기 직전이에요.", "billing", True),
    ("연간 요금제로 바꾸면 할인되나요?", "billing", False),
    ("해지했는데 이번 달에도 요금이 빠져나갔습니다. 당장 환불해 주세요.", "billing", True),
    ("세금계산서 발행 요청드립니다. 다음 주까지만 주시면 돼요.", "billing", False),
    ("환불 신청한 지 2주 됐는데 아직 입금이 안 됐어요.", "billing", False),
    ("오늘 마감인 법인 결제가 승인 거절됐어요. 급합니다.", "billing", True),
    ("쿠폰을 적용했는데 결제 금액에 반영이 안 됐어요.", "billing", False),
    ("청구서 금액이 계약 금액과 달라요. 확인 부탁드립니다.", "billing", False),
    ("무료 체험 끝나기 전에 자동결제 막고 싶어요. 오늘이 마지막 날이에요.", "billing", True),
    ("부가세 포함 가격인지 궁금합니다.", "billing", False),
    ("앱을 열자마자 바로 꺼져요. 업무를 전혀 못 하고 있습니다.", "technical", True),
    ("다크모드에서 글자가 잘 안 보여요.", "technical", False),
    ("서버 접속이 안 돼서 전 직원이 지금 멈춰 있어요.", "technical", True),
    ("사진 업로드할 때 가끔 오류 메시지가 떠요.", "technical", False),
    ("오늘 오후 시연인데 대시보드가 로딩되지 않습니다. 긴급!", "technical", True),
    ("알림이 하루 늦게 오는 것 같아요.", "technical", False),
    ("업데이트 후 동기화가 안 돼서 데이터가 사라진 것 같아요. 바로 봐주세요.", "technical", True),
    ("검색 결과가 느리게 나와요.", "technical", False),
    ("PDF 내보내기 버튼을 누르면 아무 반응이 없어요.", "technical", False),
    ("결제 페이지에서 500 에러가 나서 고객 주문을 못 받고 있어요.", "technical", True),
    ("태블릿에서 화면 비율이 깨져 보입니다.", "technical", False),
    ("API 응답이 가끔 타임아웃 납니다. 원인을 알고 싶어요.", "technical", False),
    ("비밀번호를 잊어버렸어요. 재설정 메일이 안 와요.", "account", False),
    ("누가 제 계정에 로그인한 것 같아요. 지금 당장 잠가 주세요.", "account", True),
    ("이메일 주소를 회사 메일로 바꾸고 싶어요.", "account", False),
    ("계정이 잠겨서 오늘 보고서 제출을 못 하고 있어요.", "account", True),
    ("회원 탈퇴는 어떻게 하나요?", "account", False),
    ("휴대폰 번호가 바뀌어서 본인인증이 안 됩니다.", "account", False),
    ("관리자 권한이 갑자기 사라져서 팀원 초대를 못 해요. 급해요.", "account", True),
    ("닉네임 변경이 가능한가요?", "account", False),
    ("2단계 인증 기기를 잃어버려서 로그인이 막혔어요. 오늘 꼭 들어가야 합니다.", "account", True),
    ("팀원 계정을 다른 팀으로 옮기고 싶어요.", "account", False),
    ("소셜 로그인 연결을 해제하고 싶습니다.", "account", False),
    ("해킹 의심 알림이 왔어요. 비밀번호가 바뀌어 있어요!", "account", True),
    ("주문한 지 일주일 됐는데 아직 배송 준비중이에요.", "shipping", False),
    ("내일 이사라서 오늘 안에 배송지 주소를 바꿔야 해요.", "shipping", True),
    ("반품 수거 기사님이 언제 오시나요?", "shipping", False),
    ("다른 사람 물건이 배송됐어요.", "shipping", False),
    ("선물용인데 오늘 도착 안 하면 의미가 없어요. 지금 어디쯤인가요?", "shipping", True),
    ("교환 신청한 상품을 받았는데 사이즈가 또 틀려요.", "shipping", False),
    ("송장번호 조회가 안 됩니다.", "shipping", False),
    ("배송 완료라고 뜨는데 물건을 못 받았어요. 오늘 꼭 필요한 약이에요.", "shipping", True),
    ("포장이 찢어진 채로 도착했어요.", "shipping", False),
    ("해외 배송도 가능한가요?", "shipping", False),
    ("주문 취소하고 싶은데 오늘 출고 전에 막아 주세요.", "shipping", True),
    ("부분 배송된 나머지 상품은 언제 오나요?", "shipping", False),
]

TEST = [
    ("같은 주문이 카드에 세 번 청구됐어요. 오늘 안에 취소해 주세요.", "billing", True),
    ("영수증에 사업자번호를 넣을 수 있나요?", "billing", False),
    ("요금제 다운그레이드하면 차액은 환불되나요?", "billing", False),
    ("자동이체 계좌를 변경하고 싶어요.", "billing", False),
    ("로그인하면 흰 화면만 나와요. 지금 고객 응대를 못 하고 있어요.", "technical", True),
    ("엑셀 업로드 시 한글이 깨집니다.", "technical", False),
    ("앱이 배터리를 너무 많이 써요.", "technical", False),
    ("전체 서비스가 다운된 것 같아요. 바로 확인 부탁드립니다!", "technical", True),
    ("퇴사한 직원 계정을 비활성화해 주세요.", "account", False),
    ("제 계정으로 모르는 결제 시도가 있었어요. 즉시 계정 정지해 주세요.", "account", True),
    ("프로필 사진이 변경되지 않아요.", "account", False),
    ("인증 메일 링크가 만료됐다고 나와요.", "account", False),
    ("택배가 3일째 같은 터미널에 머물러 있어요.", "shipping", False),
    ("오늘 출발 전에 받아야 하는데 배송 기사님 연락처 좀 알려주세요. 급해요.", "shipping", True),
    ("반품했는데 수거가 안 됐다고 나옵니다.", "shipping", False),
    ("상품이 파손돼서 왔어요. 교환 받고 싶어요.", "shipping", False),
]


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for text, dept, urgent in rows:
            row = {"state": text, "questions": QUESTIONS,
                   "expected": {"department": dept, "urgent": urgent}, "language": "ko"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{path}: {len(rows)} rows")


if __name__ == "__main__":
    here = Path(__file__).parent
    write(here / "train.jsonl", TRAIN)
    write(here / "test.jsonl", TEST)
