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
AG-T39: 공식 규격(directon-2026-voice-cs)으로 교체. 브랜드 "하루온", 함수 8개(인계 함수 없음 → 말로 안내), 브랜드 자료
(`data/brand/` scenarios.md·information.json·style.json, DIRECTON_BRAND_DIR)를 고정부에 넣는다. 영업시간 평일 10~17시(점심 12~13시).
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

from agent import catalog as _catalog
from agent.core.types import SessionInfo
from agent.grounding import GROUNDING_RULES, refund_table_for  # AG-T48 근거 없는 사실 금지
from agent.normalize import digits_for_speech, group_digits, spoken_digits
from agent.responder import number_candidates_enabled, utterance_numbers

_ORDER_EX = "20260925-0041152"    # 공식 주문번호 형식(15자리) 되읽기 예시(AG-T39, 공식 EXAMPLES의 가명 번호)
_LEGACY_ORDER_EX = "2611150231"   # 옛 다이렉트몰(10자리) 데이터일 때
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
    for ex in (_ORDER_EX, _LEGACY_ORDER_EX):
        if len(re.sub(r"\D", "", ex)) in ls:
            return ex
    return example or _ORDER_EX


def build_fixed(detail: bool = True, order_lengths=None, order_example: str | None = None) -> str:
    """고정부. detail=True면 상황별 3~4문장, 결과 핵심 필드·다음 절차, 월·일 날짜, 정책 부연 규칙을 더한다.
    order_lengths: 주문번호 길이들(공식 15·16자리), order_example: 그 길이의 주문번호(되읽기 예시)."""
    order_ex = _order_example(order_lengths, order_example)
    order_len = lengths_text(order_lengths)
    order_spoken = digits_for_speech(order_ex, "order").replace(", ", " ")
    length = ("- 답은 소리로 읽힌다. 한 번에 2~3문장, 짧고 정중한 해요체로 핵심부터 말한다. "
              + ("함수 결과를 전하거나 정책 조건을 안내할 때는 3~4문장까지 써도 된다. 질문에 필요한 정보는 빠뜨리지 않는다. "
                 if detail else "")
              + "고객이 반말을 해도 존댓말을 쓴다.")
    dates = ("- 날짜는 '내일', '오늘'만으로 말하지 말고 \"내일인 11월 21일\"처럼 월·일을 함께 말한다.\n"
             if detail else "")
    extra_style = ("- 정책 질문에는 묻는 값과 함께 브랜드 정보 같은 항목의 관련 조건(예외, 소요 기간, 비용 부담, 이용할 수 있는 대안) "
                   "한 가지를 덧붙인다.\n"
                   "- 하루온 업무가 아닌 질문은 정중히 안내한 뒤 하루온에서 도울 수 있는 관련 일을 제안한다.\n"
                   if detail else "")
    ok_line = ("- ok: 고객이 물은 것에 답하고, 결과에 있는 배송상태, 결제 여부, 결제 금액, 날짜(월·일), 제품명, 반품 현재상태처럼 질문과 "
               "관련된 값을 빠뜨리지 않는다. 상태는 결과에 적힌 그대로 말한다(예: '입금전'을 '배송 준비 중'으로 뭉뚱그리지 않는다). "
               "접수 결과(취소, 반품, 교환, 전화 요청)에는 브랜드 정보에 있는 다음 절차(회수 일정, 환불 시점과 소요 기간, "
               "연락 시간)를 한 문장 덧붙인다."
               if detail else "- ok: 결과의 핵심(상태, 날짜, 제품명)만 말한다.")
    return f"""너는 건강기능식품 쇼핑몰 하루온 고객센터의 전화 상담원이다. 고객의 말은 음성 인식 결과라서 오탈자, 띄어쓰기 오류, 끊긴 숫자가 있을 수 있다. 뜻을 헤아려 응대한다. 아래 '상담 시나리오'의 흐름을 따르고, 정책과 사실은 '브랜드 정보'와 함수 결과만 쓴다.

## 말하는 방식
{length}
- 너는 통화 첫마디를 이미 말했다(아래 '이번 통화'에 적혀 있다). 네 응답은 그 첫마디 바로 뒤에 이어지는 말이다. 첫머리에 "고객님", "○○○ 고객님" 같은 호칭이나 "안녕하세요", "하루온 고객센터입니다", "네" 같은 인사·맞장구를 다시 넣지 말고 바로 본론으로 시작한다. 호칭과 인사는 반복하지 않는다.
- 첫 문장은 짧게, 결론부터 말한다. 예: "주문하신 제품은 오늘 출고돼요."
- 직전 상담원 말이 "… (고객이 말을 끊어 여기까지만 말함)"으로 끝나면 고객이 그 말 도중에 끼어든 것이다. 이미 한 말은 반복하지 말고 고객의 새 말에 바로 답한다. 끊긴 것에 대해 길게 사과하지 않는다.
- 마크다운, 목록 기호, 이모지, 괄호 설명을 쓰지 않는다. 브랜드 정보에 있는 링크·이모티콘은 읽지 않고, 링크 대신 찾아가는 메뉴 경로를 말로 풀어 안내한다.
- 송장번호처럼 긴 번호는 읽지 말고 "문자로 보내드린 송장번호"라고 한다.
- 이전 대화의 상담원 말은 채팅 상담 기록이라 링크나 이모티콘이 섞여 있다. 따라 하지 않는다.
{dates}{extra_style}- 주문번호·전화번호는 확인이 꼭 필요할 때만 되읽는다. 되읽을 때는 묶음 사이를 쉼표로 끊는다. 주문번호 {order_ex}{_topic(order_ex)} "{group_digits(order_ex, "order")}", 전화번호 {_PHONE_EX}은 "{group_digits(_PHONE_EX, "phone")}"처럼 쓴다.
- 모르는 내용은 지어내지 않는다. 주문·배송·입금·반품 정보는 함수 결과로만, 정책은 브랜드 정보로만 말한다. 브랜드 정보와 시나리오에 없는 정책은 만들지 말고 상담원에게 넘긴다.
- 건강 상태, 복용 중인 약, 부작용처럼 전문 판단이 필요한 문의는 단정해서 답하지 않는다. 브랜드 정보의 공통 안내가 있으면 그대로 전하고, 그 밖에는 상담원에게 넘긴다. 효과를 보장하거나 질병 치료처럼 말하지 않는다.

## 함수 사용 규칙
- 함수가 필요하면 먼저 함수만 호출하고 아무 말도 하지 않는다. 말은 함수 결과를 받은 뒤에 한다. 함수를 호출하지 않았으면 "접수됐어요", "취소됐어요", "요청해 드렸어요"처럼 처리된 것처럼 말하지 않는다. 함수 이름, 호출 형식, JSON은 글자로 쓰지 않는다. 함수는 함수 호출로만 부른다.
- 배송비, 배송 일정, 반품 기간·비용, 환불 기간, 결제 방법, 쿠폰·적립금, 회원 문의 같은 정책 질문은 함수 없이 브랜드 정보로 답한다.
- 주문·배송·입금·취소 상태 확인은 get_order_info를 쓴다. 주문번호를 알면 order_id, 모르면 주문자 전화번호를 buyer_cellphone에 넣는다. 결과에 주문이 여러 건이면 고객이 말한 제품이나 날짜에 맞는 주문으로 답하고, 정할 수 없으면 어느 주문인지 묻는다.
- 반품·환불 진행 상황은 get_return_info에 전화번호(customer_phone_number)를 넣어 조회한다.
- 제품 정보(섭취 방법, 성분, 효능, 함께 먹어도 되는지, 가격 등)는 search_products로 찾는다. 고객이 제품명을 정확히 말하지 않아도 "간 건강 제품 섭취 방법"처럼 검색어를 만들어 부르고, 검색 결과에 있는 내용만 안내한다.
- 주문 취소: 출고 전인지 모르면 get_order_info로 먼저 확인한다. 출고 전이고 취소 사유를 들었으면 cancel_order(order_id, reason)를 호출한다. 이미 출고됐으면 취소 대신 반품 절차를 안내한다.
- 반품: 반품 사유를 들은 뒤 접수한다. 기본은 return_order(order_number, reason)다. 주문의 상품을 모두 반품하면(상품이 하나뿐인 주문, "전체", "다", "전부", 이미 몇 개 먹었거나 뜯었어도 남은 것을 모두 보내는 경우 포함) return_order를 쓴다. return_order_item(return_items, reason)은 고객이 주문 상품 중 일부만 반품하거나 "일부 반품"이라고 분명히 말할 때만 쓰고, return_items의 order_item_code에는 주문 조회 결과의 상세주문번호를 넣는다. 무통장입금·가상계좌처럼 환불 계좌가 필요한 결제면 계좌를 받아 refund_account에 넣고, 회수지를 바꾸는 경우에만 pickup_address를 넣는다.
- 교환: 같은 제품으로 바꾸기를 원하고 사유를 들었으면 exchange_order(order_number, reason)를 호출한다. 불량 여부처럼 판단이 필요하면 상담원에게 넘긴다.
- 배송지 변경은 함수가 없다. 주문을 조회해 출고 전이면 바꿀 주소를 받아 두고 상담원이 처리한다고 안내하고, 이미 출고됐으면 변경이 어렵다고 안내한다.
- 상담원 전화 요청: 혼자 주문하기 어렵거나 상담원이 다시 전화해 주기를 원하면 이름과 연락처를 받아 request_callback(customer_name, phone_number)을 호출한다. 이름이나 연락처가 없으면 함수를 부르지 말고 먼저 묻는다.
- 상담원 연결은 함수가 아니라 말로 안내한다. 연결을 원하거나 해결하기 어려운 문의면 상담원에게 넘겨 도와드리겠다고 말하고, 직접 통화를 원하면 브랜드 정보의 [전화 문의 방법]에 있는 고객센터 번호와 영업 시간을 안내한다.
- 발신 고객 정보에 전화번호가 있으면 번호를 다시 묻지 않는다. 그 번호로 바로 조회한다. 고객이 다른 번호로 주문했다고 하면 고객이 말한 번호를 쓴다.
- 발신 고객 정보가 없고 고객도 주문번호나 전화번호를 말하지 않았으면 함수를 부르지 말고 필요한 번호를 짧게 묻는다. 번호를 처음 물을 때는 "한 자리씩 천천히"를 붙이지 않고 "주문번호나 주문하신 분 연락처를 말씀해 주시겠어요?"처럼 묻는다. 처리에 다른 정보(취소·반품·교환 사유, 전화 요청의 이름, 배송지 변경의 새 주소)도 필요하면 함께 말씀해 달라고 한다. 취소나 반품 같은 실행 요청이어도 되묻기만 한다.
- 이전 대화에서 이미 조회하거나 안내한 정보는 다시 조회하지 않는다. 이전 대화의 조회 결과를 그대로 쓰고, 이미 안내한 처리에 대한 후속 질문(회수 일정, 환불 기간, 배송비 같은 정책)은 함수 없이 브랜드 정보로 답한다. 이전 대화에서 확인된 주문번호와 전화번호는 다시 묻지 않고 그대로 쓴다. 직전에 상담원이 처리를 위해 정보를 물었고 고객이 답하면 그 처리 함수를 바로 호출한다. 직전에 상담원이 "~해 드릴까요?"라고 물었고 고객이 동의하면 그 처리를 바로 실행한다. 고객이 거절하거나 인사로 마무리하면 함수를 부르지 않는다.
- 되돌리기 어려운 처리(취소, 반품, 교환, 전화 요청)는 고객이 분명히 요청하고 필요한 정보가 다 있을 때만 호출한다. 고민 중이라고 하면 조건만 안내한다.
- 번호 인자: 고객이 한글로 읽은 번호("{order_spoken}", "공일공 일이삼사 오육칠팔")는 직접 숫자로 바꾸지 말고 고객이 말한 그대로 넣는다. 시스템이 정확한 숫자로 바꾼다. 아라비아 숫자로 말했으면 숫자만 넣는다. 고객이 말한 주문번호가 {order_len}가 아니거나 전화번호가 010으로 시작하는 11자리가 아니어도 스스로 되묻지 말고 들은 번호 그대로 함수를 호출한다. 음성 인식이 숫자를 틀리게 받아 적는 일이 많아서 시스템이 자릿수를 확인하고 확인된 고객 정보로 번호를 맞춘다. 함수 결과가 invalid_arguments나 not_found일 때만 "번호를 한 자리씩 천천히 다시 말씀해 주시겠어요?"처럼 되묻는다. 자릿수가 맞으면 되묻지 말고 바로 호출한다.

## 함수 결과 안내
{ok_line}
- already_requested: 이미 접수된 건이라고 안내하고, 브랜드 정보에 따라 이후 진행을 알려 준다.
- not_found: 그 번호로는 찾을 수 없다고 말하고 번호를 한 번 더 확인해 달라고 한다. 다시 확인해도 없으면 다른 정보(전화번호나 주문번호)를 요청한다. 반품 현황이 조회되지 않으면 그대로 알리고 필요하면 상담원에게 넘긴다.
- invalid: 처리에 필요한 정보(사유, 번호, 이름 등)가 빠진 것이다. 빠진 정보를 짧게 묻는다.
- error: 지금 확인이 어렵다고 사과하고 상담원에게 넘긴다.
- invalid_arguments: 자리 수 규칙은 설명하지 않고 "번호가 정확히 들리지 않았어요. 번호를 한 자리씩 천천히 다시 말씀해 주시겠어요?"처럼 짧게 다시 묻는다. 고객이 이번 말에서 번호 없이 브랜드 정보로 답할 수 있는 정책을 직접 물었을 때만 그 답을 한 문장으로 먼저 하고, 묻지 않은 안내는 덧붙이지 않는다."""


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
SCENARIO_HEADER = "## 상담 시나리오"
INFO_HEADER = "## 브랜드 정보"
BRAND_DIR_ENV = "DIRECTON_BRAND_DIR"
# 공식 브랜드 자료(AD14와 같은 이유로 저장소에 복사하지 않는다. 제출 묶음에는 이 폴더가 함께 들어간다)
DEFAULT_BRAND_DIR = Path(__file__).resolve().parent.parent / "directon-2026-voice-cs" / "data" / "brand"
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


def _strip_links(text: str) -> str:
    """전화로 읽을 수 없는 URL은 지운다(공식: 링크 대신 찾아가는 길을 말한다)."""
    return re.sub(r"https?://\S+", "(링크)", text)


def load_brand(brand_dir: str | Path | None = None) -> dict[str, str]:
    """공식 브랜드 자료 → {"scenarios", "info", "style", "greeting"}. 없는 파일은 빈 문자열.
    brand_dir가 None이면 DIRECTON_BRAND_DIR → 공식 폴더."""
    d = Path(brand_dir) if brand_dir else Path(_env_path(BRAND_DIR_ENV) or DEFAULT_BRAND_DIR)
    out = {"scenarios": "", "info": "", "style": "", "greeting": ""}
    out["scenarios"] = load_manual(d / "scenarios.md")
    try:
        info = json.loads((d / "information.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        info = []
    blocks = []
    for item in info if isinstance(info, list) else []:
        if isinstance(item, dict) and item.get("content"):
            tag = str(item.get("tag") or "").strip().strip("[]")
            blocks.append(f"### {tag}\n{_strip_links(str(item['content']).strip())}")
    out["info"] = re.sub(r"(?m)^## 이번 통화", "### 이번 통화", "\n\n".join(blocks))
    try:
        style = json.loads((d / "style.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        style = {}
    if isinstance(style, dict):
        out["style"] = str(style.get("style") or "").strip()
        out["greeting"] = str(style.get("greeting") or "").strip()
    return out


def _parse_now(now: str) -> datetime | None:
    """ "YYYY-MM-DD HH:MM" 또는 공식 형식 "YYYY-MM-DD (수) HH:MM"(AG-T39)."""
    s = re.sub(r"\s*\([^)]*\)\s*", " ", (now or "").strip().replace("T", " "))
    s = re.sub(r"\s+", " ", s)[:16]
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M")
    except ValueError:
        return None


def business_hours_note(now: str) -> str:
    """now 기준 상담원 연결 가능 여부(하루온 브랜드 정보 [전화 문의 방법]: 평일 10~17시, 점심 12~13시 제외, 주말 휴무). 모르면 ""."""
    t = _parse_now(now)
    if t is None:
        return ""
    wd = _WEEKDAYS[t.weekday()]
    when = f"{wd}요일 {t:%H:%M}"
    hm = t.hour * 60 + t.minute
    if t.weekday() >= 5:
        why = "주말 휴무"
    elif hm < 10 * 60:
        why = "평일 오전 10시 이전"
    elif hm >= 17 * 60:
        why = "평일 오후 5시 이후"
    elif 12 * 60 <= hm < 13 * 60:
        why = "점심시간 12시~1시"
    else:
        return (f"- 고객센터 영업 시간: 지금({when})은 영업 시간이다. 상담원에게 넘길 때는 말로 안내한다.")
    return (f"- 고객센터 영업 시간: 지금({when})은 영업 시간이 아니다({why}). 상담원에게 넘기거나 직접 통화를 원하면 "
            "영업 시간(평일 오전 10시~오후 5시, 점심 12시~1시 제외)에 연락드린다고 안내한다.")


def _is_agent(h: dict) -> bool:
    return h.get("role") in ("agent", "assistant", "bot")


TOOL_ROLES = ("tool_call", "tool_result")
TOOL_RESULT_MAX_CHARS = 2000          # 공식 연습 history 결과 최대 약 1,230자(search_products)
_PHONE_ARG_HINTS = ("phone", "cellphone")
_ORDER_ARGS = ("order_id", "order_number", "order_item_code")


def _is_tool(h: dict) -> bool:
    return h.get("role") in TOOL_ROLES


def _compact(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _tool_line(h: dict) -> str:
    """AG-T37: 공식 history의 함수 기록 한 줄. 인자는 값이 있는 것만, 결과는 압축 JSON(상한)."""
    name = h.get("name") or "?"
    if h.get("role") == "tool_call":
        args = {k: v for k, v in (h.get("arguments") or {}).items() if v not in (None, "", [], {})}
        return f"- (상담원이 함수 호출) {name} {_compact(args)}"
    body = _compact(h.get("result"))
    if len(body) > TOOL_RESULT_MAX_CHARS:
        body = body[:TOOL_RESULT_MAX_CHARS] + "…(생략)"
    return f"- (조회 결과) {name} → {body}"


def _history_lines(history: list[dict]) -> list[str]:
    out = []
    for h in history or []:
        if not isinstance(h, dict):
            out.append(f"- {h}")
            continue
        if _is_tool(h):
            out.append(_tool_line(h))
            continue
        who = "상담원" if _is_agent(h) else "고객"
        text = h.get("text") or h.get("content") or ""
        out.append(f"- {who}: {text}")
    return out


def _tool_arg_numbers(args: dict) -> list[str]:
    """함수 인자에서 번호(키 이름으로 전화/주문 판단, 자릿수 가정 없음)."""
    out = []
    for k, v in (args or {}).items():
        if isinstance(v, list):                     # return_items: [{"order_item_code": ...}]
            for it in v:
                if isinstance(it, dict):
                    out.extend(_tool_arg_numbers(it))
            continue
        if not isinstance(v, str) or not re.search(r"\d", v):
            continue
        kl = k.lower()
        if any(p in kl for p in _PHONE_ARG_HINTS):
            out.append("전화번호 " + re.sub(r"\D", "", v))
        elif kl in _ORDER_ARGS:
            out.append("주문번호 " + re.sub(r"\s", "", v))
    return out


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


def _is_phone_digits(d: str) -> bool:
    return len(d) == 11 and d.startswith("01")


def _text_numbers(h: dict) -> list[tuple[str, str]]:
    """대화 한 줄의 번호 → [("전화번호"|"주문번호", 숫자열)]. 숫자로 쓴 번호, 없으면 고객이 한글로 읽은 번호
    (읽기 후보 중 형식이 맞는 첫 번호). 형식(전화 01로 시작 11자리, 주문 데이터 길이)이 아니면 버린다."""
    order_lens = _catalog.order_id_lengths()        # 데이터의 주문번호 길이(모르면 10자리)
    short = min(order_lens) < 7                      # 짧은 주문번호("1234")도 찾는다

    def is_order(d: str) -> bool:
        return len(d) in order_lens and not _is_phone_digits(d)

    text = h.get("text") or h.get("content") or ""
    found = [re.sub(r"\D", "", m) for m in (_NUM_ANY if short else _NUM).findall(text)]
    if not _is_agent(h) and not found:
        cands = utterance_numbers(text) + (_short_spoken_runs(text) if short else [])
        found = [d for d in cands if is_order(d) or _is_phone_digits(d)][:1]
    out = []
    for d in found:
        label = "전화번호" if _is_phone_digits(d) else "주문번호" if is_order(d) else ""
        if label:
            out.append((label, d))
    return out


def _empty_result(result) -> bool:
    """조회 결과가 없다: not_found, 또는 ok인데 data의 목록이 모두 비었다."""
    if not isinstance(result, dict):
        return False
    if result.get("status") == "not_found":
        return True
    data = result.get("data")
    if result.get("status") != "ok" or not isinstance(data, dict):
        return False
    lists = [v for v in data.values() if isinstance(v, list)]
    return bool(lists) and not any(lists)


def _failed_lookup_numbers(items: list[dict]) -> list[str]:
    """AG-T49: 이전 대화에서 조회(get_*)했는데 결과가 없던 번호(바로 뒤 같은 함수 결과 기준)."""
    out: list[str] = []
    for i, h in enumerate(items):
        if h.get("role") != "tool_call" or not str(h.get("name") or "").startswith("get_"):
            continue
        res = next((x for x in items[i + 1:] if x.get("role") == "tool_result"), None)
        if res is None or res.get("name") not in (None, h.get("name")) or not _empty_result(res.get("result")):
            continue
        for n in _tool_arg_numbers(h.get("arguments") or {}):
            if n not in out:
                out.append(n)
    return out


def history_summary(history: list[dict]) -> list[str]:
    """이전 대화 요약 줄: 이미 나온 번호, 직전 상담원 질문, 재조회 금지."""
    items = [h for h in history or [] if isinstance(h, dict)]
    if not items:
        return []
    out: list[str] = []
    nums: list[str] = []
    tool_seen = False
    for h in items:
        if _is_tool(h):
            tool_seen = True
            if h.get("role") == "tool_call":
                for n in _tool_arg_numbers(h.get("arguments") or {}):
                    if n not in nums:
                        nums.append(n)
            continue
        for label, d in _text_numbers(h):
            if f"{label} {d}" not in nums:
                nums.append(f"{label} {d}")
    if nums:
        out.append(f"- 이전 대화에서 나온 번호: {', '.join(nums)}. 다시 묻지 말고 그대로 쓴다.")
    talk = [h for h in items if not _is_tool(h)]      # 직전 말은 함수 기록을 건너뛰고 찾는다
    last = talk[-1] if talk else {}
    last_text = (last.get("text") or last.get("content") or "").strip()
    if _is_agent(last) and last_text:
        sentences = [s for s in re.split(r"(?<=[.?!])\s+", last_text) if s.strip()]
        q = next((s for s in reversed(sentences) if s.rstrip().endswith(("?", "？"))), "")
        if q:
            out.append(f"- 직전 상담원 질문: \"{q.strip()}\" 이번 고객 말은 이 질문에 대한 답이다.")
        else:
            out.append(f"- 직전 상담원 말: \"{last_text}\" 이번 고객 말은 이에 이어지는 말이다.")
    out.append("- 이전 대화에서 이미 조회하거나 안내한 내용은 다시 조회하지 않는다.")
    failed = _failed_lookup_numbers(items)               # AG-T49: 결과가 없던 번호(P0095-06 "번호는 동일")
    if failed:
        out.append(f"- 이전 대화에서 {', '.join(failed)}(으)로 조회했지만 결과가 없었다. 고객이 같은 번호·동일하다고 하면 "
                   "그 번호를 다시 묻지 말고(다시 조회하지도 않는다) 그 번호로는 찾지 못했다고 짧게 말한 뒤 주문번호나 "
                   "주문하실 때 쓰신 다른 연락처를 묻는다.")
    if tool_seen:
        out.append("- 이전 대화의 조회 결과에 이미 있는 내용(주문 상태, 송장, 반품 현황, 상품 정보)은 다시 조회하지 않고 "
                   "그 결과로 바로 답한다. 결과에 없는 내용만 새로 조회한다.")
    return out


# ── 번호 후보(AG-T43) ─────────────────────────────────
# 음성 인식이 숫자를 자주 틀리게 받아 적는다(AG-T40). 이미 아는 번호(발신번호, 이전 대화의 번호·조회 결과 주문번호)를
# 후보로 알려 주면 LLM이 들린 숫자를 가장 가까운 후보로 맞추거나(강한 확신) 그 후보로 확인한다(약한 확신).
# 응답기(agent.responder.number_candidates_from_system)도 이 줄을 읽어 같은 후보로 교정·확인한다.
_ORDER_KEYS = ("order_id", "order_number")


def _note(text) -> str:
    return re.sub(r"[()\[\],]", " ", str(text or "")).strip()


def number_candidates(info: SessionInfo) -> list[dict]:
    """이미 아는 번호 → [{"kind": "phone"|"order", "number"(숫자열), "raw"(원래 표기), "note"}]. 같은 번호는 한 번."""
    out: list[dict] = []

    def add(kind: str, raw, note: str = "") -> None:
        raw = str(raw or "").strip()
        d = re.sub(r"\D", "", raw)
        if not d or (kind == "phone" and not _is_phone_digits(d)) or (kind == "order" and len(d) < 4):
            return
        for c in out:
            if c["number"] == d:
                c["note"] = c["note"] or _note(note)
                return
        out.append({"kind": kind, "number": d, "raw": raw, "note": _note(note)})

    phone = ((info.caller or {}).get("phone") or "")
    if phone:
        add("phone", phone, "발신번호")
    for h in info.history or []:
        if not isinstance(h, dict):
            continue
        role = h.get("role")
        if role == "tool_call":
            for k, v in (h.get("arguments") or {}).items():
                if isinstance(v, str) and any(p in k.lower() for p in _PHONE_ARG_HINTS):
                    add("phone", v)
                elif isinstance(v, str) and k.lower() in _ORDER_KEYS:
                    add("order", v)
        elif role == "tool_result":
            res = h.get("result")
            data = res.get("data") if isinstance(res, dict) else None
            for o in (data.get("orders") if isinstance(data, dict) else None) or []:
                if isinstance(o, dict):
                    add("order", o.get("주문번호", o.get("order_id")), o.get("상품명", o.get("product_name")) or "")
        else:
            for label, d in _text_numbers(h):
                add("phone" if label == "전화번호" else "order", d)
    return out


def candidates_line(cands: list[dict]) -> str:
    if not cands:
        return ""
    items = ", ".join(f"{'전화번호' if c['kind'] == 'phone' else '주문번호'} {c['raw']}"
                      + (f"({c['note']})" if c["note"] else "") for c in cands)
    tail = cands[0]["number"][-4:]
    return (f"- 번호 후보: {items}. 음성 인식이 숫자를 자주 틀리게 받아 적는다. 고객이 말한 번호가 이 후보 중 하나와 "
            "거의 같으면(몇 자리만 다르거나 빠지거나 끊겨 들림) 그 후보 번호를 그대로 함수 인자로 쓴다. 비슷한 후보가 있지만 "
            f"확신이 서지 않으면(절반쯤만 겹치거나 비슷한 후보가 둘) 함수를 부르지 말고 '끝자리 {tail} 번호 맞으실까요?'처럼 "
            "그 후보의 끝 네 자리로 한 문장만 확인하고, 번호 전체는 읽지 않는다. 고객이 맞다고 하면 그 후보 번호 전체를 쓴다. "
            "후보와 전혀 다르거나 고객이 다른 번호라고 하면 고객이 말한 번호를 쓴다. 이 규칙이 번호 인자 규칙보다 앞선다.")


def _env_path(name: str) -> str | None:
    v = os.environ.get(name, "").strip()
    return v or None


def build_system(info: SessionInfo, opener: str, manual_path: str | Path | None = "",
                 manual: str | None = None, brand_path: str | Path | None = None,
                 style_path: str | Path | None = None, brand_dir: str | Path | None = None) -> str:
    """고정부(규칙 + 매뉴얼 + 브랜드 + 말투) + `## 이번 통화`(시각·고객·이력·첫 마디).

    manual_path가 기본값이고 DIRECTON_MANUAL이 있으면 그 파일을 읽는다(Settings.from_env도 같은 별칭).
    brand_path·style_path가 None이면 DIRECTON_BRAND·DIRECTON_STYLE. 파일이 없으면 그 절을 뺀다.
    """
    if manual is not None:
        manual_text = manual
    else:
        if not manual_path and _env_path(MANUAL_ENV):
            manual_path = _env_path(MANUAL_ENV)
        manual_text = load_manual(manual_path)
    brand_text = load_manual(brand_path if brand_path is not None else _env_path(BRAND_ENV))
    style_text = load_manual(style_path if style_path is not None else _env_path(STYLE_ENV))
    brand = load_brand(brand_dir)
    parts = [fixed_prompt(), GROUNDING_RULES]
    if brand["scenarios"]:
        parts.append(f"{SCENARIO_HEADER}\n{brand['scenarios']}")
    if brand["info"]:
        parts.append(f"{INFO_HEADER}\n{brand['info']}")
        if refund_table_for(brand["info"]):
            parts.append(refund_table_for(brand["info"]))
    if brand["style"]:
        parts.append(f"{STYLE_HEADER}\n{brand['style']}")
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
        note = business_hours_note(info.now)
        if note:
            var.append(note)
    caller = info.caller or {}
    known = {k: v for k, v in caller.items() if v}
    if known:
        labels = {"name": "이름", "phone": "전화번호", "email": "이메일"}
        desc = ", ".join(f"{labels.get(k, k)} {v}" for k, v in known.items())
        line = f"- 발신 고객 정보: {desc}. 이 정보는 이미 확인된 것이니 고객에게 다시 묻지 말고 그대로 쓴다."
        if known.get("phone"):
            line += (f" 주문 조회(get_order_info의 buyer_cellphone), 반품 현황(get_return_info), 전화 요청(request_callback)에 "
                     f"전화번호가 필요하면 {known['phone']}을 바로 넣는다. 고객이 다른 번호로 주문했다고 하면 그 번호를 쓴다.")
        var.append(line)
    else:
        var.append("- 발신 고객 정보: 없음. 개인 정보가 필요한 요청이면 주문번호나 주문하신 분 연락처 하나만 물어본다.")
    hist = _history_lines(info.history)
    if hist:
        var.append("- 이전 대화:")
        var.extend("  " + line for line in hist)
        var.extend(history_summary(info.history))
    if number_candidates_enabled():
        line = candidates_line(number_candidates(info))
        if line:
            var.append(line)
    if opener:
        var.append(f"- 너는 이미 '{opener}'라고 말했다. 같은 말을 반복하지 말고 그 뒤에 이어서 말하라. "
                   "호칭이나 인사 없이 본론부터 말한다.")
    parts.append("\n".join(var))
    return "\n\n".join(parts)
