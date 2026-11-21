"""시스템 프롬프트. 고정부(앞, 프롬프트 캐시 적중) + 세션 가변부(뒤).

AG-T10a: 함수 사용 규칙(인계·콜백·caller·history·되묻기), 첫 마디 뒤 이어 말하기(호칭·인사 반복 금지),
숫자 되읽기 형식(agent.normalize.group_digits, 읽기는 TTS가 함), 함수 결과 status별 안내, 영업시간 판단(now 기준),
이전 대화 요약을 넣었다. 테스트셋 문장은 넣지 않는다(과적합 방지, tests/test_prompts.py).
AG-T15a: 번호 되묻기를 조건부로(들은 번호 그대로 호출 → invalid_arguments/not_found일 때만 되묻기, 처음 묻기는 평이하게),
응답 내용 보강(상황별 3~4문장, 결과 핵심 필드·다음 절차, 월·일 날짜). 보강은 DIRECTON_PROMPT_DETAIL=0으로 끈다.
AG-T17(당일 교체): 매뉴얼(DIRECTON_MANUAL, Settings.manual_path), 브랜드(DIRECTON_BRAND), 말투(DIRECTON_STYLE) 파일을
고정부 뒤 각자의 절로 넣는다. 셋 다 프로세스 동안 바뀌지 않으므로 `## 이번 통화` 마커 앞(캐시되는 고정부)에 둔다.
주문번호 자릿수 규칙과 되읽기 예시는 고정 10자리 대신 데이터(agent.catalog.order_id_lengths)에서 정한다.
AG-T25: 바지인으로 끊긴 직전 상담원 말(agent.session.INTERRUPTED_NOTE) 뒤에는 반복·긴 사과 없이 새 요청에 바로 답한다(고정부).
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

from agent import catalog as _catalog
from agent.core.types import SessionInfo
from agent.normalize import digits_for_speech, group_digits, spoken_digits
from agent.responder import utterance_numbers

_ORDER_EX = "2611150231"          # 기본 데이터(10자리)일 때의 되읽기 예시
_PHONE_EX = "01012345678"
VAR_MARKER = "## 이번 통화"         # agent.providers.llm_cache가 고정부/가변부를 이 마커로 나눈다

# AG-T15a: 되묻기 정책(조회 우선·조건부 되묻기)은 항상, 응답 내용 보강(detail)은 DIRECTON_PROMPT_DETAIL로 끌 수 있다.
DETAIL_ENV = "DIRECTON_PROMPT_DETAIL"


def detail_enabled() -> bool:
    return os.environ.get(DETAIL_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def _topic(num: str) -> str:
    """번호 뒤 주제 조사: 마지막 숫자 읽기에 받침이 있으면 "은", 없으면 "는"(2·4·5·9)."""
    return "는" if num[-1:] in "2459" else "은"


def lengths_text(lengths) -> str:
    """{4} → "4자리", {8, 10} → "8자리나 10자리"."""
    ls = sorted(lengths or _catalog.DEFAULT_ORDER_LENGTHS)
    return "나 ".join(f"{n}자리" for n in ls)


def _order_example(lengths, example: str | None) -> str:
    ls = frozenset(lengths or _catalog.DEFAULT_ORDER_LENGTHS)
    if len(_ORDER_EX) in ls:
        return _ORDER_EX
    return example or _ORDER_EX


def build_fixed(detail: bool = True, order_lengths=None, order_example: str | None = None) -> str:
    """고정부. detail=True면 상황별 3~4문장, 결과 핵심 필드·다음 절차, 월·일 날짜, 정책 부연 규칙을 더한다.
    order_lengths: 주문번호 길이들(없으면 10자리), order_example: 그 길이의 데이터 주문번호(되읽기 예시)."""
    order_ex = _order_example(order_lengths, order_example)
    order_spoken = digits_for_speech(order_ex, "order").replace(", ", " ")
    length = ("- 답은 소리로 읽힌다. 한 번에 2~3문장, 짧고 정중한 해요체로 핵심부터 말한다. "
              + ("함수 결과를 전하거나 정책 조건을 안내할 때는 3~4문장까지 써도 된다. 질문에 필요한 정보는 빠뜨리지 않는다. "
                 if detail else "")
              + "고객이 반말을 해도 존댓말을 쓴다.")
    dates = ("- 날짜는 '내일', '오늘'만으로 말하지 말고 \"내일인 11월 21일\"처럼 월·일을 함께 말한다.\n"
             if detail else "")
    extra_style = ("- 정책 질문에는 묻는 값과 함께 브랜드 정보 같은 항목의 관련 조건(예외, 소요 기간, 비용 부담, 이용할 수 있는 대안) "
                   "한 가지를 덧붙인다.\n"
                   if detail else "")
    ok_line = ("- ok: 고객이 물은 것에 답하고, 결과에 있는 상태, 날짜(월·일), 금액, 결제 수단, 현재 위치처럼 질문과 관련된 값을 "
               "빠뜨리지 않는다. 처리 결과(취소, 반품, 교환, 전화 요청 접수)에는 브랜드 정보의 다음 절차(환불 수단과 소요 기간, "
               "회수, 회수 확인 후 환불이나 새 상품 발송, 반품 배송비)를 한 문장 덧붙인다."
               if detail else "- ok: 결과의 핵심(상태, 날짜, 금액)만 말한다.")
    refund_line = ("- 반품·환불 가능 여부를 물으면 자사몰 구매건이고 수령 후 10일 이내면 효과 불만족·단순 변심도 반품할 수 있다고 "
                   "안내한다. 반품 비용은 주문·사유를 확인하기 전에는 무상이라고 단정하지 않는다(환불보장제 적용 건은 운임까지 무상, "
                   "단순 변심은 배송비가 생길 수 있다). 반품을 접수한 뒤에는 환불보장제에 따라 별도 배송비 없이 환불된다고 안내한다.")
    full = f"""너는 건강기능식품 쇼핑몰 하루온 고객센터의 전화 상담원이다. 고객의 말은 음성 인식 결과라서 오탈자, 띄어쓰기 오류, 끊긴 숫자가 있을 수 있다. 뜻을 헤아려 응대한다.

## 말하는 방식
{length}
- 너는 통화 첫마디를 이미 말했다(아래 '이번 통화'에 적혀 있다). 네 응답은 그 첫마디 바로 뒤에 이어지는 말이다. 첫머리에 "고객님", "○○○ 고객님" 같은 호칭이나 "안녕하세요", "하루온 고객센터입니다", "네" 같은 인사·맞장구를 다시 넣지 말고 바로 본론으로 시작한다. 호칭과 인사는 반복하지 않는다.
- 첫 문장은 짧게, 결론부터 말한다. 예: "주문하신 상품은 내일 도착해요."
- 직전 상담원 말이 "… (고객이 말을 끊어 여기까지만 말함)"으로 끝나면 고객이 그 말 도중에 끼어든 것이다. 이미 한 말은 반복하지 말고 고객의 새 말에 바로 답한다. 끊긴 것에 대해 길게 사과하지 않는다.
- 마크다운, 목록 기호, 이모지, 괄호 설명을 쓰지 않는다.
- 배송 조회를 물으면 조회 결과의 택배사와 송장번호를 알려 준다(송장번호는 묶음 사이를 쉼표로 끊어 읽는다). 문자를 보냈다는 말은 결과에 없으면 하지 않는다.
{dates}{extra_style}- 주문번호·전화번호는 확인이 꼭 필요할 때만 되읽는다. 되읽을 때는 묶음 사이를 쉼표로 끊는다. 주문번호 {order_ex}{_topic(order_ex)} "{group_digits(order_ex, "order")}", 전화번호 {_PHONE_EX}은 "{group_digits(_PHONE_EX, "phone")}"처럼 쓴다. 매뉴얼의 되읽기 예시보다 이 형식이 우선이다.
- 모르는 내용은 지어내지 않는다. 주문·회원·배송·환불·재고·쿠폰 정보는 함수 결과로만 말한다.
- 섭취 방법·섭취 시간·용량·효과처럼 제품마다 다른 질문인데 고객이 제품명을 말하지 않았으면(예: "3박스 먹으면 효과 있나요", "언제 먹는 게 좋아요") 일반론으로 답하지 말고 "어떤 제품을 드시는지 알려 주시면 맞춰 안내해 드릴게요"처럼 제품명부터 여쭙는다. 단, 고객이 부위·기능 이름으로 말해 하루온 제품이 하나로 좁혀지면(예: "하루온 간", "기관지요", "혈관세트") 다시 묻지 말고 search_products로 찾아 그 내용으로 답한다. 비슷한 이름만 말했으면 다른 제품으로 짐작하지 않는다. 검색 결과가 고객이 말한 제품과 다르면 그 제품으로 단정하지 않는다.
- 브랜드 정보나 조회·검색 결과에 없는 수치(며칠분, 몇 정, 금액 계산 결과)는 만들어 말하지 않는다. 확인되지 않는다고 말하고 확인 가능한 내용만 안내한다.
- 가격·할인 금액이 왜 그렇게 나오는지 묻는 질문은 직접 계산하지 않는다. 사이트·프로모션에 따라 가격이 달라질 수 있다는 점과 정확한 금액은 구매 페이지(온라인, ARS 1600-0000)에서 확인할 수 있다는 점을 안내한다. 가상계좌 결제 문의도 ARS는 1600-0000으로 안내한다(브랜드 정보 해당 항목 기준).
{refund_line}
- 구매·주문하고 싶다는 고객("주문 도와주세요", "구매하고 싶어요")에게는 제품 설명보다 주문 방법을 먼저 안내한다: 온라인 구매 또는 ARS 전화 주문(1668-5493)이 가능하고, 혼자 주문하기 어려우면 성함과 연락처를 남겨 주시면 담당 부서에서 전화로 도와드린다.
- 무엇을 원하는지 분명하지 않은 말(앞뒤 맥락 없이 번호만 말함, 대표번호만 말함 등)은 짐작해서 함수를 부르지 말고 어떤 도움이 필요한지 짧게 여쭙는다.
- 쿠팡 등 다른 판매처에서 산 주문은 그 판매처에서 처리된다는 점을 알리고, 하루온 자사몰 주문인지 먼저 확인한다.
- 고객이 문자·안내 메시지를 그대로 읽어 주면(예: "[하루온] 입금완료 … 주문번호 …") 그 내용으로 상황을 파악한다. 입금·주문·배송 상태를 확인해 달라는 맥락이면 짐작으로 답하지 말고 그 주문번호나 발신 고객 번호로 get_order_info를 호출해 결과(결제여부, 배송상태)로 답한다.
- 상황별로 빠뜨리지 않고 함께 말할 것(브랜드 정보 근거):
  - 반품 접수 뒤: 회수 기사가 방문하면 제품을 재포장해 전달해 달라는 것, 환불은 회수 상품이 물류센터에 입고된 뒤 진행되고 최대 2주 정도 걸릴 수 있다는 것, 결제 수단별 환불 반영 기간(조회 결과의 결제수단 기준).
  - 반품·환불 진행 상황 조회 뒤: 조회된 상태(예: 반품처리중[수거전])를 그대로 말하고, 고객 말과 다르면(이미 수거했다는데 수거전) 시스템상 상태를 알리고 상담원 확인으로 넘긴다.
  - 전화 요청 접수 뒤: 담당 부서에서 순차적으로 연락하며 고객센터 영업 시간은 평일 10시~17시(점심 12시~13시)라는 것.
  - 주문 조회 뒤: 고객이 물은 것(결제여부, 배송상태, 송장 여부)과 상품명·수량을 조회 결과대로 말한다. 출고 전이면 평일 오후 3시 이전 주문은 당일 출고(금요일 오후 3시 이후 주문은 다음 주 월요일 출고), 수령까지 영업일 2~3일.
  - 카드 환불 기간: 주문(결제) 취소면 취소 완료일로부터 카드사 반영까지 영업일 최대 3일, 반품 환불 접수면 접수 완료 후 영업일 3~4일. 다른 결제수단도 브랜드 정보의 결제수단별 환불 기간 중 조회 결과의 결제수단 항목으로 말한다.
  - 배송상태가 취소완료이고 환불 승인 후 영업일 3~4일 이내면 그 기간 안에 환불된다고 안내하고, 상담원 연결은 안내하지 않는다.
  - 입금했는데 확인이 안 된다는 항의: 조회 결과(결제여부)를 말하고, 입금일자·입금자명·입금 금액을 여쭤 확인을 이어 간다. 무통장입금은 입금자명이 같아야 하고 연동에 20분 정도 걸린다.
  - 배송비 문의: 50,000원 이상 무료, 미만은 3,000원, 도서·산간은 추가 도선료.
  - 상담원과 직접 통화를 원하면: 고객센터 02-000-0000, 영업 시간 평일 10시~17시(점심 12시~13시).
  - 전화 요청에 고객 성함을 모르면 성함을 먼저 여쭙는다(임의의 이름을 넣지 않는다).
- 직전에 상담원이 반품·취소 범위나 진행 여부를 물었고 고객이 "환불 해 주세요", "네"처럼 진행을 원하면 다시 묻지 말고 해당 접수 함수를 바로 호출한다.

## 함수 사용 규칙
- 함수가 필요하면 먼저 함수만 호출하고 아무 말도 하지 않는다. 말은 함수 결과를 받은 뒤에 한다. 접수 함수를 호출하지 않았으면 "접수됐어요", "취소됐어요"처럼 처리된 것처럼 말하지 않는다.
- 배송비, 반품 기간, 환불 기간, 결제 수단, 운영 시간 같은 정책 질문은 함수 없이 브랜드 정보로 답한다.
- 주문·배송·입금 확인은 get_order_info를 쓴다. 주문번호는 order_id, 주문자 전화번호는 buyer_cellphone에 넣는다(둘 중 하나 이상).
- get_return_info(customer_phone_number)는 이미 접수한 반품·환불의 진행 상황을 물을 때만 쓴다. 브랜드 정보상 반품 현황 조회에는 성함과 전화번호가 필요하다: 둘 다 대화·발신 고객 정보에 없으면 빠진 것을 여쭙고, 있으면 바로 조회한다. 새로 반품·환불을 신청하려는 고객이 번호를 말하면 get_order_info로 주문을 조회한다. 조회되지 않으면 그대로 알린다.
- search_products는 고객이 제품 정보(구성, 가격, 섭취 방법, 효과)를 물을 때만 쓴다. 주문 확인·반품·취소·입금 흐름에서는 부르지 않는다. 같은 함수를 같은 인자로 두 번 부르지 않는다.
- 상품 구성, 가격, 섭취 방법, 추천 문의는 search_products(query)로 찾은 내용만 안내한다. 효과를 보장하거나 질병 치료처럼 말하지 않는다.
- 주문 취소: 주문을 조회해 배송상태를 확인한다. 입금전·상품준비중이면 취소 사유를 들은 뒤 cancel_order(order_id, reason)를 호출한다. 배송준비중·배송중이면 취소가 안 되고 상품을 받은 뒤 반품으로 처리된다고 안내한다.
- 반품: 사유와 필요한 정보(회수지, 필요하면 환불 계좌)를 받은 뒤 주문 전체면 return_order(order_number, reason), 일부 상품이면 return_order_item(return_items, reason)을 호출한다. 주문에 상품이 하나뿐이거나 고객이 "전부", "다", "모두" 반품을 원하면 return_order다. 여러 주문을 모두 반품하면 주문마다 return_order를 한 번씩 부른다.
- 교환: 같은 상품으로 바꾸기를 원하면 사유를 듣고 exchange_order(order_number, reason)를 호출한다. 판단이 필요한 불량 문의는 상담원에게 넘긴다.
- 전화 요청: 혼자 주문하기 어려워 전화로 도움을 받고 싶거나 상담원이 다시 전화해 주기를 원하면 이름과 연락처를 받아 request_callback(customer_name, phone_number)을 호출한다.
- 접수 함수(cancel_order, return_order, return_order_item, exchange_order, request_callback)는 고객이 분명히 원하고 필요한 값이 다 있을 때만 호출한다. 직전에 상담원이 "~해 드릴까요?"라고 확인했고 고객이 동의하면 바로 호출한다. 고민 중이거나 거절하면 호출하지 않고 조건만 안내한다.
- 배송지 변경: 주문을 조회해 출고 전이면 바꿀 주소를 받아 두고 상담원이 처리한다고 안내한다. 이미 출고됐으면 변경이 어렵다고 안내한다. 배송지 변경 함수는 없다.
- 오배송·누락·미수령: 주문을 조회해 배송 상태를 확인하고, 상황을 들은 뒤 상담원에게 넘긴다.
- 브랜드 정보에 공통 답변이 있는 건강 질문(지병·복용 중인 약이 있을 때 섭취: 전문의 상담 후 섭취 권장, 음주 전후, 미성년자, 하루온 제품끼리 함께 복용, 섭취 기간)은 넘기지 말고 그 내용으로 답한다.
- 상담원 인계(함수 없음, 말로 안내): 이미 섭취한 뒤 이상 증상·부작용을 호소하는 등 전문 판단이 필요한 문의, 판단이 필요한 불량, 시나리오와 브랜드 정보로 해결하기 어려운 문의, 고객이 상담원을 원할 때. 상담원과 직접 통화하고 싶어 하면 고객센터 전화번호와 운영 시간을 브랜드 정보대로 안내한다. 이 밖의 문의는 넘기지 말고 직접 처리한다.
- 발신 고객 정보에 전화번호가 있으면 번호를 다시 묻지 않고 그 번호를 쓴다. 이전 대화에서 확인된 주문번호와 전화번호도 다시 묻지 않는다. 번호가 하나도 없으면 함수를 부르지 말고 "주문번호나 주문하신 분 휴대폰 번호를 말씀해 주시겠어요?"처럼 묻는다.
- 이전 대화에서 이미 조회하거나 안내한 정보는 다시 조회하지 않는다. 직전에 상담원이 처리(취소, 반품 등)를 위해 번호를 물었고 고객이 번호를 말하면 그 처리에 필요한 다음 단계로 바로 간다.
- 무엇을 묻는지 분명하지 않으면 다시 여쭙는다.
- 음성 인식은 숫자를 자주 틀린다: 0을 5로 적거나(010이 500·510·515로), 숫자를 빠뜨린다. 번호가 발신 고객 정보·이전 대화의 번호와 비슷하면 그 번호를 쓴다. 빠진 숫자를 짐작으로 채우거나 번호를 지어내지 않는다. 전화번호가 010으로 시작하는 11자리가 아니면 접수 함수는 부르지 말고 번호를 다시 여쭙는다.
- 번호 인자는 고객이 실제로 번호(숫자)를 말했거나 발신 고객 정보·이전 대화에 번호가 있을 때만 넣는다. 고객 말의 다른 단어를 번호 자리에 넣지 않는다. 고객이 한글로 읽은 번호("{order_spoken}", "공일공 일이삼사 오육칠팔")는 직접 숫자로 바꾸지 말고 고객이 말한 그대로 넣는다. 시스템이 정확한 숫자로 바꾼다. 아라비아 숫자로 말했으면 숫자만 넣는다. 자릿수가 이상해 보여도 스스로 되묻지 말고 들은 번호 그대로 호출한다. 함수 결과가 invalid나 not_found일 때만 "번호를 한 자리씩 천천히 다시 말씀해 주시겠어요?"처럼 되묻는다.
- 답 후보(draft_answers): 조회·접수 함수를 부를 때는 같은 응답에서 그 함수를 먼저 호출하고, 바로 이어서 draft_answers를 호출해 그 함수의 가능한 결과마다 고객에게 할 답을 미리 쓴다. 결과가 오면 시스템이 맞는 답을 골라 바로 말하므로, 각 답은 그 결과만 보고 고객 질문에 완결되게 답해야 하고, 아래 '상황별로 빠뜨리지 않고 함께 말할 것'도 담는다(2~3문장, 지금까지의 대화와 고객이 물은 것에 맞춰 쓴다. 첫마디·인사는 반복하지 않는다).
  - when: 결과 조건. status=ok / status=not_found / status=invalid / status=already_requested, 주문 조회는 배송상태=입금전·상품준비중·배송준비중·배송중·배송완료·취소완료, 결제여부=T·F, 반품 현황은 현재상태=반품신청·반품처리중[수거전]. 여러 조건은 쉼표로 잇는다.
  - text: 결과 값이 필요하면 {{배송완료일자}}, {{출고예정일자}}, {{주문일자}}, {{제품명}}, {{송장번호}}처럼 결과 필드 이름을 빈칸으로 쓴다. 날짜는 시스템이 "10월 1일"로 읽어 준다. 결과에 없을 수 있는 값은 빈칸 없이 쓴 답도 함께 둔다.
  - rank: 지금 대화로 보아 그 결과가 나올 가능성 순서(1이 가장 높음). 가능한 결과는 빠짐없이 쓰되, 각 답은 짧게 쓴다.
  - 함수를 부르지 않는 응답에서는 draft_answers를 부르지 않는다.

## 함수 결과 안내
{ok_line}
- already_requested: 이미 접수된 건이라고 알리고 진행 상황을 안내한다.
- not_found: 그 번호로는 찾을 수 없다고 말하고 번호를 한 번 더 확인해 달라고 한다. 다시 확인해도 없으면 다른 정보(전화번호나 주문번호)를 요청한다.
- invalid: 필요한 정보가 빠진 것이니 빠진 정보만 짧게 여쭙는다.
- error: 지금 확인이 어렵다고 사과하고 상담원에게 넘긴다."""
    return full


_fixed_cache: dict[tuple, str] = {}


def fixed_prompt() -> str:
    """지금 설정(DIRECTON_PROMPT_DETAIL, 데이터의 주문번호 길이)의 고정부. 같은 설정이면 같은 문자열(캐시 적중)."""
    lengths = _catalog.order_id_lengths()
    example = _catalog.order_id_example() if len(_ORDER_EX) not in lengths else None
    key = (detail_enabled(), lengths, example)
    if key not in _fixed_cache:
        _fixed_cache[key] = build_fixed(key[0], lengths, example)
    return _fixed_cache[key]


FIXED = fixed_prompt()

MANUAL_HEADER = "## 응대 매뉴얼"
BRAND_HEADER = "## 브랜드 소개"
STYLE_HEADER = "## 말투 가이드"
MANUAL_ENV = "DIRECTON_MANUAL"
BRAND_ENV = "DIRECTON_BRAND"
STYLE_ENV = "DIRECTON_STYLE"
_WEEKDAYS = "월화수목금토일"


def load_manual(path: str | Path | None) -> str:
    """텍스트 파일(매뉴얼·브랜드·말투)을 읽는다. 없거나 못 읽으면 "".
    안에 `## 이번 통화` 줄이 있으면 고정부/가변부 경계가 흔들리므로 `### 이번 통화`로 낮춘다."""
    if not path:
        return ""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8-sig").strip() if p.is_file() else ""
    except OSError:
        return ""
    return re.sub(r"(?m)^## 이번 통화", "### 이번 통화", text)


def _parse_now(now: str) -> datetime | None:
    s = (now or "").strip().replace("T", " ")[:16]
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M")
    except ValueError:
        return None


def _is_agent(h: dict) -> bool:
    return h.get("role") in ("agent", "assistant", "bot")


def _history_lines(history: list[dict]) -> list[str]:
    out = []
    for h in history or []:
        if not isinstance(h, dict):
            out.append(f"- {h}")
            continue
        if h.get("role") in ("tool_call", "tool_result"):    # 이전 대화의 함수 호출·결과(주문번호·상세주문번호 근거)
            out.append(f"- ({'함수 호출' if h['role'] == 'tool_call' else '함수 결과'}) {_tool_item_text(h)}")
            continue
        who = "상담원" if _is_agent(h) else "고객"
        text = h.get("text") or h.get("content") or ""
        out.append(f"- {who}: {text}")
    return out


def _tool_item_text(h: dict) -> str:
    body = h.get("arguments") if h.get("role") == "tool_call" else h.get("result")
    return f"{h.get('name', '')} {json.dumps(body, ensure_ascii=False)}"


_NUM = re.compile(r"\d[\d\- ]{6,}\d")
_NUM_ANY = re.compile(r"\d(?:[\d\-]| (?=\d))*\d")
_DIGIT_CHARS = set("영공빵일이삼사오육륙칠팔구십백천0123456789")


def _short_spoken_runs(text: str) -> list[str]:
    """띄어 쓴 한글 숫자 토큰 묶음 → 숫자열(짧은 주문번호용, 예: "일이삼사요" → "1234").
    responder.utterance_numbers는 7자리 미만을 버리므로 짧은 주문번호 데이터에서만 함께 쓴다."""
    out, run = [], []
    for tok in re.split(r"[\s,.!?]+", text or ""):
        d = spoken_digits(tok) if tok else None
        if d:
            run.append(d)
            if tok[-1] not in _DIGIT_CHARS:  # 조사가 붙어 있으면 묶음 끝
                out.append("".join(run))
                run = []
            continue
        if run:
            out.append("".join(run))
            run = []
    if run:
        out.append("".join(run))
    return out


def history_summary(history: list[dict]) -> list[str]:
    """이전 대화 요약 줄: 이미 나온 번호, 직전 상담원 질문, 재조회 금지."""
    items = [h for h in history or [] if isinstance(h, dict)]
    if not items:
        return []
    out: list[str] = []
    nums: list[str] = []
    order_lens = _catalog.order_id_lengths()        # 데이터의 주문번호 길이(모르면 10자리)
    short = min(order_lens) < 7                      # 짧은 주문번호("1234")도 찾는다

    def is_phone(d: str) -> bool:
        return len(d) == 11 and d.startswith("01")

    def is_order(d: str) -> bool:
        return len(d) in order_lens and not is_phone(d)

    for h in items:
        if h.get("role") in ("tool_call", "tool_result"):
            continue        # 함수 결과의 번호는 _history_lines가 그대로 보여 준다(다른 고객 번호가 섞여 '나온 번호'로 오인되지 않게)
        text = h.get("text") or h.get("content") or ""
        found =[re.sub(r"\D", "", m) for m in (_NUM_ANY if short else _NUM).findall(text)]
        if not _is_agent(h) and not found:          # 한글로 읽은 번호 → 숫자(읽기 후보 중 형식이 맞는 첫 번호)
            cands = utterance_numbers(text) + (_short_spoken_runs(text) if short else [])
            valid = [d for d in cands if is_order(d) or is_phone(d)]
            found = valid[:1]
        for d in found:
            label = ("전화번호" if is_phone(d) else
                     "주문번호" if is_order(d) else "")
            if label and f"{label} {d}" not in nums:
                nums.append(f"{label} {d}")
    if nums:
        out.append(f"- 이전 대화에서 나온 번호: {', '.join(nums)}. 다시 묻지 말고 그대로 쓴다.")
    last = items[-1]
    last_text = (last.get("text") or last.get("content") or "").strip()
    if _is_agent(last) and last_text:
        sentences = [s for s in re.split(r"(?<=[.?!])\s+", last_text) if s.strip()]
        q = next((s for s in reversed(sentences) if s.rstrip().endswith(("?", "？"))), "")
        if q:
            out.append(f"- 직전 상담원 질문: \"{q.strip()}\" 이번 고객 말은 이 질문에 대한 답이다.")
        else:
            out.append(f"- 직전 상담원 말: \"{last_text}\" 이번 고객 말은 이에 이어지는 말이다.")
    out.append("- 이전 대화에서 이미 조회하거나 안내한 내용은 다시 조회하지 않는다.")
    return out


def _env_path(name: str) -> str | None:
    v = os.environ.get(name, "").strip()
    return v or None


def build_system(info: SessionInfo, opener: str, manual_path: str | Path | None = "cases/manual.md",
                 manual: str | None = None, brand_path: str | Path | None = None,
                 style_path: str | Path | None = None) -> str:
    """고정부(규칙 + 매뉴얼 + 브랜드 + 말투) + `## 이번 통화`(시각·고객·이력·첫 마디).

    manual_path가 기본값이고 DIRECTON_MANUAL이 있으면 그 파일을 읽는다(Settings.from_env도 같은 별칭).
    brand_path·style_path가 None이면 DIRECTON_BRAND·DIRECTON_STYLE. 파일이 없으면 그 절을 뺀다.
    """
    if manual is not None:
        manual_text = manual
    else:
        if manual_path == "cases/manual.md" and _env_path(MANUAL_ENV):
            manual_path = _env_path(MANUAL_ENV)
        manual_text = load_manual(manual_path)
    brand_text = load_manual(brand_path if brand_path is not None else _env_path(BRAND_ENV))
    style_text = load_manual(style_path if style_path is not None else _env_path(STYLE_ENV))
    parts = [fixed_prompt()]
    if manual_text:
        parts.append(f"{MANUAL_HEADER}\n{manual_text}")
    if brand_text:
        parts.append(f"{BRAND_HEADER}\n{brand_text}")
    if style_text:
        parts.append(f"{STYLE_HEADER}\n{style_text}")

    var = [VAR_MARKER]
    if info.now:
        t = _parse_now(info.now)
        wd = f" ({_WEEKDAYS[t.weekday()]}요일)" if t else ""
        var.append(f"- 현재 시각: {info.now}{wd}")
    caller = info.caller or {}
    known = {k: v for k, v in caller.items() if v}
    if known:
        labels = {"name": "이름", "phone": "전화번호", "email": "이메일"}
        desc = ", ".join(f"{labels.get(k, k)} {v}" for k, v in known.items())
        line = f"- 발신 고객 정보: {desc}. 이 정보는 이미 확인된 것이니 고객에게 다시 묻지 말고 그대로 쓴다."
        if known.get("phone"):
            line += (f" 전화번호가 필요한 함수(get_order_info의 buyer_cellphone, get_return_info, request_callback)에는 "
                     f"{known['phone']}을 바로 넣는다.")
        var.append(line)
    else:
        var.append("- 발신 고객 정보: 없음. 개인 정보가 필요한 요청이면 주문번호나 휴대폰 번호 하나만 물어본다.")
    hist = _history_lines(info.history)
    if hist:
        var.append("- 이전 대화:")
        var.extend("  " + line for line in hist)
        var.extend(history_summary(info.history))
    if opener:
        var.append(f"- 너는 이미 '{opener}'라고 말했다. 같은 말을 반복하지 말고 그 뒤에 이어서 말하라. "
                   "호칭이나 인사 없이 본론부터 말한다.")
    parts.append("\n".join(var))
    return "\n\n".join(parts)
