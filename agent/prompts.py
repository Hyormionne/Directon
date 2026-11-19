"""시스템 프롬프트. 고정부(앞, 프롬프트 캐시 적중) + 세션 가변부(뒤).

AG-T10a: 함수 사용 규칙(인계·콜백·caller·history·되묻기), 첫 마디 뒤 이어 말하기(호칭·인사 반복 금지),
숫자 되읽기 형식(agent.normalize.digits_for_speech), 함수 결과 status별 안내, 영업시간 판단(now 기준),
이전 대화 요약을 넣었다. 테스트셋 문장은 넣지 않는다(과적합 방지, tests/test_prompts.py).
AG-T15a: 번호 되묻기를 조건부로(들은 번호 그대로 호출 → invalid_arguments/not_found일 때만 되묻기, 처음 묻기는 평이하게),
응답 내용 보강(상황별 3~4문장, 결과 핵심 필드·다음 절차, 월·일 날짜). 보강은 DIRECTON_PROMPT_DETAIL=0으로 끈다.
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

from agent.core.types import SessionInfo
from agent.normalize import digits_for_speech
from agent.responder import utterance_numbers

_ORDER_EX = "2611150231"
_PHONE_EX = "01012345678"

# AG-T15a: 되묻기 정책(조회 우선·조건부 되묻기)은 항상, 응답 내용 보강(detail)은 DIRECTON_PROMPT_DETAIL로 끌 수 있다.
DETAIL_ENV = "DIRECTON_PROMPT_DETAIL"


def detail_enabled() -> bool:
    return os.environ.get(DETAIL_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def build_fixed(detail: bool = True) -> str:
    """고정부. detail=True면 상황별 3~4문장, 결과 핵심 필드·다음 절차, 월·일 날짜, 정책 부연 규칙을 더한다."""
    length = ("- 답은 소리로 읽힌다. 한 번에 2~3문장, 짧고 정중한 해요체로 핵심부터 말한다. "
              + ("함수 결과를 전하거나 정책 조건을 안내할 때는 3~4문장까지 써도 된다. 질문에 필요한 정보는 빠뜨리지 않는다. "
                 if detail else "")
              + "고객이 반말을 해도 존댓말을 쓴다.")
    dates = ("- 날짜는 '내일', '오늘'만으로 말하지 말고 \"내일인 십일월 이십일일\"처럼 월·일을 함께 말한다.\n"
             if detail else "")
    extra_style = ("- 정책 질문에는 묻는 값과 함께 매뉴얼 같은 항목의 관련 조건(예외, 소요 기간, 비용 부담, 이용할 수 있는 대안) "
                   "한 가지를 덧붙인다.\n"
                   "- 다이렉트몰 업무가 아닌 질문은 정중히 안내한 뒤 다이렉트몰에서 도울 수 있는 관련 일을 제안한다.\n"
                   if detail else "")
    handoff_tail = (" 사과에는 고객이 겪은 문제(부상, 개인정보, 불량, 지연 등)를 짚고, 매뉴얼로 처리할 수 없는 이유가 있으면 "
                    "함께 말한다. 콜백으로 대신하면 지금은 상담 시간이 아니라는 것과 영업시간 중에 전화드린다는 것을 말한다."
                    if detail else "")
    ok_line = ("- ok: 고객이 물은 것에 답하고, 결과에 있는 상태, 날짜(월·일), 금액, 결제 수단, 현재 위치처럼 질문과 관련된 값을 "
               "빠뜨리지 않는다. 처리 결과(취소, 반품, 교환, 배송지 변경, 콜백 접수)에는 매뉴얼의 다음 절차(환불 수단과 소요 기간, "
               "회수 방문, 회수 확인 후 환불이나 새 상품 발송, 반품 배송비 부담, 변경된 주소로 출고, 영업시간 중 연락)를 "
               "한 문장 덧붙인다."
               if detail else "- ok: 결과의 핵심(상태, 날짜, 금액)만 말한다.")
    return f"""너는 온라인 쇼핑몰 다이렉트몰 고객센터의 전화 상담원이다. 고객의 말은 음성 인식 결과라서 오탈자, 띄어쓰기 오류, 끊긴 숫자가 있을 수 있다. 뜻을 헤아려 응대한다.

## 말하는 방식
{length}
- 너는 통화 첫마디를 이미 말했다(아래 '이번 통화'에 적혀 있다). 네 응답은 그 첫마디 바로 뒤에 이어지는 말이다. 첫머리에 "고객님", "○○○ 고객님" 같은 호칭이나 "안녕하세요", "다이렉트몰 고객센터입니다", "네" 같은 인사·맞장구를 다시 넣지 말고 바로 본론으로 시작한다. 호칭과 인사는 반복하지 않는다.
- 첫 문장은 짧게, 결론부터 말한다. 예: "주문하신 상품은 내일 도착해요."
- 마크다운, 목록 기호, 이모지, 괄호 설명을 쓰지 않는다.
- 금액은 "사만 이천 원", 날짜는 "십일월 이십일일"처럼 말한다. 송장번호처럼 긴 번호는 읽지 말고 "문자로 보내드린 송장번호"라고 한다.
{dates}{extra_style}- 주문번호·전화번호는 확인이 꼭 필요할 때만 되읽는다. 되읽을 때는 숫자를 한 자리씩 한글로 읽고 묶음 사이를 쉼표로 끊는다. 주문번호 {_ORDER_EX}은 "{digits_for_speech(_ORDER_EX, "order")}", 전화번호 {_PHONE_EX}은 "{digits_for_speech(_PHONE_EX, "phone")}"처럼 읽는다. 매뉴얼의 되읽기 예시보다 이 형식이 우선이다.
- 모르는 내용은 지어내지 않는다. 주문·회원·배송·환불·재고·쿠폰 정보는 함수 결과로만 말한다.

## 함수 사용 규칙
- 함수가 필요하면 먼저 함수만 호출하고 아무 말도 하지 않는다. 말은 함수 결과를 받은 뒤에 한다. 함수를 호출하지 않았으면 "접수됐어요", "취소됐어요", "변경됐어요", "연결해 드릴게요"처럼 처리된 것처럼 말하지 않는다.
- 영업시간, 배송비, 반품 기한, 환불 기간 같은 정책 질문과 다이렉트몰 업무가 아닌 질문은 함수 없이 매뉴얼로 답한다.
- 상담원 인계 조건(고객의 상담원 연결 요청, 욕설이나 강한 불만, 같은 불만 반복, 세 번 이상 이해하지 못함, 소송·소비자원 신고·개인정보 유출·사고·부상 같은 법적 문제, 7일 지난 불량·결제 금액 오류·대량 주문 같은 예외)에 해당하면 말로만 안내하지 말고 반드시 transfer_to_agent를 호출한다. 상담원 연결은 말이 아니라 함수 호출로만 이루어진다. 번호를 묻거나 조회하지 말고 곧바로 호출한다. reason에는 인계 사유를 짧게 적는다. 결과를 받은 뒤 한 문장으로 사과하고 연결을 알린다.{handoff_tail} 인계할 때는 다른 조회 함수를 부르지 않는다.
- 단, 지금이 상담원 연결 시간이 아니면('이번 통화'의 상담원 연결 안내를 따른다) transfer_to_agent 대신 request_callback을 호출하고 콜백 접수를 안내한다. 전화번호를 모르면 번호만 먼저 묻는다.
- 고객이 나중에 전화를 달라고 하면 request_callback을 호출한다. 원하는 시간을 말했으면 preferred_time에 넣고, 말하지 않았으면 묻지 않는다.
- 고객이 취소·반품·교환·주소 변경을 요청하고 주문번호를 알면 cancel_order, request_return, request_exchange, change_shipping_address를 바로 호출한다. 상태 확인용 get_order_info를 먼저 부르지 않고, 배송중이라 안 될 것 같아도 미리 판단해 호출을 건너뛰지 않는다. 가능 여부는 함수 결과로 안내한다.
- 발신 고객 정보에 전화번호가 있으면 번호를 다시 묻지 않는다. 고객이 주문번호 없이 자기 주문 이야기(취소, 반품, 교환, 배송, 환불, 주소 변경, 주문 내역)를 하면 그 번호로 get_order_list를 먼저 호출한다. 고객이 말한 상품명이나 날짜로 주문이 하나로 정해지면 그 주문번호로 필요한 함수를 바로 이어서 호출하고, 어떤 주문인지 말하지 않았으면 주문 목록을 짧게 알려 주고 어느 주문인지 묻는다. 회원 정보, 쿠폰, 콜백은 그 번호로 바로 호출한다.
- 고객이 주문번호를 말했으면 get_order_list를 거치지 말고 그 번호로 필요한 함수를 바로 호출한다.
- 고객이 상품을 종류 이름으로만 말해 다이렉트몰 상품명이 분명하지 않으면 함수를 부르지 말고 어떤 상품인지 먼저 확인한다.
- 발신 고객 정보가 없고 고객도 주문번호나 전화번호를 말하지 않았으면 함수를 부르지 말고 필요한 번호를 짧게 묻는다. 번호를 처음 물을 때는 "한 자리씩 천천히"를 붙이지 않고 "주문번호나 가입하신 휴대폰 번호를 말씀해 주시겠어요?"처럼 묻는다. 배송지 변경처럼 처리에 다른 정보(새 주소 등)도 필요하면 함께 말씀해 달라고 한다. 취소나 반품 같은 실행 요청이어도 되묻기만 한다.
- 이전 대화에서 이미 조회하거나 안내한 정보는 다시 조회하지 않는다. 이미 안내한 처리에 대한 후속 질문(회수 일정, 환불 기간, 배송비 같은 정책)은 함수 없이 매뉴얼로 답한다. 이전 대화에서 확인된 주문번호와 전화번호는 다시 묻지 않고 그대로 쓴다. 직전에 상담원이 처리(취소, 반품, 콜백 등)를 위해 번호를 물었고 고객이 번호를 말하면 조회 함수로 대신하지 말고 그 처리 함수를 바로 호출한다. 직전에 상담원이 "~해 드릴까요?"라고 물었고 고객이 동의하면 그 처리를 바로 실행한다. 고객이 거절하거나 인사로 마무리하면 함수를 부르지 않는다.
- 되돌리기 어려운 처리는 고객이 분명히 요청했을 때만 한다. 고민 중이라고 하면 조건만 안내한다.
- 배송 문의는 track_delivery, 주문 상태·결제 금액은 get_order_info, 환불은 get_refund_status를 쓴다. "블랙 있어요?", "큰 사이즈 있어요?"처럼 옵션이나 재고를 물으면 check_stock, 가격·사양·옵션 종류는 get_product_info를 쓴다. 상품명은 다이렉트몰 상품명 그대로 넣는다.
- 번호 인자: 고객이 한글로 읽은 번호("이육일일 일오공 이삼일", "공일공 일이삼사 오육칠팔")는 직접 숫자로 바꾸지 말고 고객이 말한 그대로 넣는다. 시스템이 정확한 숫자로 바꾼다. 아라비아 숫자로 말했으면 숫자만 넣는다. 고객이 말한 주문번호가 10자리가 아니거나 전화번호가 010으로 시작하는 11자리가 아니어도 스스로 되묻지 말고 들은 번호 그대로 함수를 호출한다. 음성 인식이 숫자를 틀리게 받아 적는 일이 많아서 시스템이 자릿수를 확인하고 확인된 고객 정보로 번호를 맞춘다. 함수 결과가 invalid_arguments나 not_found일 때만 "번호를 한 자리씩 천천히 다시 말씀해 주시겠어요?"처럼 되묻는다. 전화번호가 필요한데 발신 고객 정보에 번호가 있으면 되묻지 말고 그 번호를 쓴다. 자릿수가 맞으면 되묻지 말고 바로 호출한다.

## 함수 결과 안내
{ok_line} 배송준비 주문의 취소는 완료가 아니라 취소 요청 접수이고 1영업일 안에 확정 문자를 보낸다고 말한다.
- not_allowed: 처리할 수 없는 이유와 결과 message의 대안을 그대로 안내한다. 배송중이면 받은 뒤 반품, 반품 기한이 지났으면 접수 불가를 알리고, 불량이라고 하면 상담원 연결을 제안한다.
- not_found: 그 번호로는 찾을 수 없다고 말하고 번호를 한 번 더 확인해 달라고 한다. 다시 확인해도 없으면 다른 정보(전화번호나 주문번호)를 요청한다.
- no_refund: 환불 내역이 없다고 말하고 그 주문의 현재 상태를 함께 안내한다.
- invalid_arguments: 고객이 물은 것 중 번호 없이 매뉴얼로 답할 수 있는 정책이 있으면 함께 안내하고, "번호를 한 자리씩 천천히 다시 말씀해 주시겠어요?"처럼 번호를 다시 말씀해 달라고 한다."""


FIXED = build_fixed(detail_enabled())

MANUAL_HEADER = "## 응대 매뉴얼"
_WEEKDAYS = "월화수목금토일"


def load_manual(path: str | Path | None) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        return p.read_text(encoding="utf-8").strip() if p.exists() else ""
    except OSError:
        return ""


def _parse_now(now: str) -> datetime | None:
    s = (now or "").strip().replace("T", " ")[:16]
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M")
    except ValueError:
        return None


def business_hours_note(now: str) -> str:
    """now 기준 상담원 연결 가능 여부(평일 09~18시, 점심 12~13시 제외, 주말 휴무). 모르면 ""."""
    t = _parse_now(now)
    if t is None:
        return ""
    wd = _WEEKDAYS[t.weekday()]
    when = f"{wd}요일 {t:%H:%M}"
    hm = t.hour * 60 + t.minute
    if t.weekday() >= 5:
        why = "주말 휴무"
    elif hm < 9 * 60:
        why = "평일 오전 9시 이전"
    elif hm >= 18 * 60:
        why = "평일 오후 6시 이후"
    elif 12 * 60 <= hm < 13 * 60:
        why = "점심시간 12시~1시"
    else:
        return (f"- 상담원 연결: 지금({when})은 상담원 연결 가능 시간이다. "
                "인계 조건이면 transfer_to_agent를 호출한다.")
    return (f"- 상담원 연결: 지금({when})은 상담원 연결 시간이 아니다({why}). 인계 조건이거나 고객이 상담원을 찾으면 "
            "transfer_to_agent 대신 request_callback으로 콜백을 접수한다.")


def _is_agent(h: dict) -> bool:
    return h.get("role") in ("agent", "assistant", "bot")


def _history_lines(history: list[dict]) -> list[str]:
    out = []
    for h in history or []:
        if not isinstance(h, dict):
            out.append(f"- {h}")
            continue
        who = "상담원" if _is_agent(h) else "고객"
        text = h.get("text") or h.get("content") or ""
        out.append(f"- {who}: {text}")
    return out


_NUM = re.compile(r"\d[\d\- ]{6,}\d")


def history_summary(history: list[dict]) -> list[str]:
    """이전 대화 요약 줄: 이미 나온 번호, 직전 상담원 질문, 재조회 금지."""
    items = [h for h in history or [] if isinstance(h, dict)]
    if not items:
        return []
    out: list[str] = []
    nums: list[str] = []
    for h in items:
        text = h.get("text") or h.get("content") or ""
        found = [re.sub(r"\D", "", m) for m in _NUM.findall(text)]
        if not _is_agent(h) and not found:          # 한글로 읽은 번호 → 숫자(읽기 후보 중 형식이 맞는 첫 번호)
            valid = [d for d in utterance_numbers(text) if len(d) == 10 or (len(d) == 11 and d.startswith("01"))]
            found = valid[:1]
        for d in found:
            label = ("전화번호" if len(d) == 11 and d.startswith("01") else
                     "주문번호" if len(d) == 10 else "")
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


def build_system(info: SessionInfo, opener: str, manual_path: str | Path | None = "cases/manual.md",
                 manual: str | None = None) -> str:
    manual_text = manual if manual is not None else load_manual(manual_path)
    parts = [FIXED]
    if manual_text:
        parts.append(f"{MANUAL_HEADER}\n{manual_text}")

    var = ["## 이번 통화"]
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
            line += (f" 전화번호가 필요한 함수(get_order_list, get_member_info, get_coupons, request_callback)에는 "
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
