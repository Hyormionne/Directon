"""함수 결과 → 바로 읽을 답 문장.

조회 뒤 답 시간 = (결과 수신 → 답 생성 → TTS 첫 조각). 흔한 결과는 LLM 을 다시 부르지 않고
여기서 문장을 만들어 바로 TTS 로 보낸다. 템플릿이 None 을 돌려주면 LLM 으로 넘긴다.

문장 내용은 information.json 의 배송·반품 안내를 따른다. 링크·이모티콘은 읽지 않는다.
"""
from __future__ import annotations

import re
from datetime import datetime

from .numbers_ko import parse_now, to_spoken_digits

WEEK = "월화수목금토일"


def _md(s: str | None) -> str | None:
    """'2026-09-28' → '9월 28일'"""
    if not s:
        return None
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    return f"{int(m.group(2))}월 {int(m.group(3))}일" if m else None


_NO_BATCHIM_DIGIT = set("2459")      # 이·사·오·구 → 받침 없음


def josa(word: str, pair: str) -> str:
    """josa('맥스2', '은/는') → '맥스2는'"""
    a, b = pair.split("/")
    ch = word.strip()[-1:] if word.strip() else ""
    if "가" <= ch <= "힣":
        has = (ord(ch) - 0xAC00) % 28 != 0
    elif ch.isdigit():
        has = ch not in _NO_BATCHIM_DIGIT
    else:
        has = False
    return word + (a if has else b)


def _product(rows: list[dict]) -> str:
    names = []
    for r in rows:
        n = re.sub(r"[^\w\s&+가-힣]", "", r.get("제품명") or r.get("상품명") or "").strip()
        n = re.sub(r"\s*비밀특가\s*", " ", n).strip()
        if n and n not in names:
            names.append(n)
    if not names:
        return "주문하신 상품"
    return names[0] if len(names) == 1 else f"{names[0]} 외 {len(names) - 1}건"


def _ship_eta(now: datetime | None) -> str:
    """[출고전 배송 안내]: 평일 15시 이전 주문 당일 출고, 금 15시 이후 → 월요일 출고."""
    if now is None:
        return "평일 오후 3시 이전 주문은 당일 출고되고, 받아 보시기까지 영업일 기준 2~3일 정도 걸려요."
    wd, h = now.weekday(), now.hour
    if wd < 5 and h < 15:
        when = "오늘"
    elif wd < 4:
        when = "내일"
    else:
        when = "다음 주 월요일"
    return f"{when} 출고될 예정이고, 받아 보시기까지 영업일 기준 2~3일 정도 걸려요."


PAY_Q = re.compile(r"입금|결제|이체|송금|예금자|입금자")
# 주문 템플릿을 써도 되는 주제 (허용 목록). 주제를 모르거나(None) 접수 흐름이면 LLM 이 문맥을 보고 답한다.
ORDER_TEMPLATE_INTENTS = {"order_status"}


def unpaid_stale(rows: list[dict], now: datetime | None, hours: float = 1.0) -> bool:
    """미입금(입금전) 주문이 주문 시각으로부터 hours 이상 지났나.
    매뉴얼 [입금 연동 시간 안내]의 '20분'은 방금 입금한 경우에만 맞는 안내다."""
    if now is None:
        return False
    for r in rows or []:
        if r.get("배송상태") == "입금전" or r.get("결제여부") == "F":
            try:
                t = datetime.strptime(str(r.get("주문일자"))[:16], "%Y-%m-%d %H:%M")
            except ValueError:
                continue
            if (now - t).total_seconds() >= hours * 3600:
                return True
    return False


def render_order(result: dict, session: dict, intent: str | None = None, utterance: str = "") -> str | None:
    st = result.get("status")
    if st == "not_found":
        return "말씀해 주신 번호로는 주문이 확인되지 않아요. 주문하실 때 쓰신 연락처나 주문번호를 한 번 더 불러 주시겠어요?"
    if st != "ok":
        return None
    rows = (result.get("data") or {}).get("orders") or []
    if not rows:
        return None
    order_ids = {r.get("주문번호") for r in rows}
    if len(order_ids) > 1:
        return None                                    # 여러 주문은 어떤 건인지 LLM 이 묻게 한다
    if intent not in ORDER_TEMPLATE_INTENTS:
        return None                                    # 반품·취소·주제 불명 등은 시나리오 판단이 필요 → LLM
    r = rows[0]
    prod, state = _product(rows), r.get("배송상태") or ""
    now = parse_now(session.get("now"))
    # 입금·결제를 물었으면 결제 상태를 먼저 답한다
    if PAY_Q.search(utterance or "") and r.get("결제여부") == "T" and state not in {"취소완료", "입금전취소"}:
        d = _md(r.get("결제일자"))
        head = f"{d}에 입금 확인됐어요." if d else "입금은 정상으로 확인됐어요."
        tail = {"배송완료": "상품은 배송 완료됐어요.", "배송중": "상품은 지금 배송 중이에요.",
                "배송준비중": "지금 배송 준비 중이에요.", "상품준비중": "지금 배송 준비 중이에요."}.get(state, "")
        return f"{prod} 주문은 {head} {tail}".strip()
    if state == "배송완료":
        d = _md(r.get("배송완료일자"))
        return f"확인해 보니 {prod}, {d + '에 ' if d else ''}배송 완료된 걸로 나와요. CJ대한통운으로 배송됐어요."
    if state == "배송중":
        inv = re.sub(r"\D", "", str(r.get("송장번호") or ""))
        tail = f" 송장번호는 {to_spoken_digits(inv, (4, 4, 4))}예요." if len(inv) >= 10 else ""
        return f"{josa(prod, '은/는')} 지금 배송 중이에요. CJ대한통운에서 송장번호로 배송 현황을 보실 수 있어요.{tail}"
    if state in {"배송준비중", "상품준비중"}:
        d = _md(r.get("출고예정일자"))
        eta = f"{d} 출고 예정이고, 받아 보시기까지 영업일 기준 2~3일 정도 걸려요." if d else _ship_eta(now)
        return f"{josa(prod, '은/는')} 아직 출고 전이고 배송 준비 중이에요. {eta}"
    if state == "입금전" or r.get("결제여부") == "F":
        if unpaid_stale(rows, now):
            return None                                # 주문한 지 오래됐는데 미입금 → '20분' 안내는 틀린 답. LLM 이 입금 정보를 묻는다
        return (f"{prod} 주문은 아직 입금 전으로 확인돼요. 입금하시고 시스템에 반영되기까지 20분 정도 걸릴 수 있어요.")
    if state in {"취소완료", "입금전취소"}:
        return f"{prod} 주문은 취소 완료된 상태예요. 환불은 승인 후 영업일 기준 3~4일 안에 진행돼요."
    return None


def render_return_info(result: dict, session: dict) -> str | None:
    st = result.get("status")
    if st == "not_found":
        return "말씀해 주신 번호로는 접수된 반품 내역이 확인되지 않아요. 확인을 위해 상담원에게 연결해 드릴게요."
    if st != "ok":
        return None
    rows = (result.get("data") or {}).get("반품정보") or []
    if len(rows) != 1:
        return None
    r = rows[0]
    prod, state = _product(rows), r.get("현재상태") or ""
    if "수거전" in state:
        return (f"{prod} 반품은 접수돼 있고, 아직 회수 전이에요. 택배 기사님이 방문해 회수하신 뒤 물류센터에 입고되면 환불이 진행되는데, "
                "최대 2주 정도 걸리고 결제 수단에 반영되기까지 2~3일 더 걸릴 수 있어요.")
    if "수거" in state or "회수" in state or "입고" in state:
        return f"{prod} 반품 상품은 회수됐어요. 물류센터 입고 후 환불이 진행되고, 최대 2주 정도 걸릴 수 있어요."
    if "완료" in state:
        return f"{prod} 반품은 처리 완료됐어요. 결제 수단에 반영되기까지 2~3일 정도 더 걸릴 수 있어요."
    return None


def _pay(orders: list[dict]) -> str:
    out = []
    for r in orders or []:
        v = r.get("결제수단")
        out += v if isinstance(v, list) else [str(v or "")]
    return " ".join(out)


def _refund_line(orders) -> str:
    """결제수단이 확인될 때만 환불 소요 안내 (매뉴얼 [결제수단 별 환불기간 일정])."""
    p = _pay(orders)
    if re.search(r"무통장|가상계좌", p):
        return "환불은 남겨 주신 계좌로 영업일 기준 3~4일 안에 들어가요."
    if re.search(r"카드", p):
        return "카드 결제는 취소 완료 후 영업일 기준 3일 안에 카드사에서 취소돼요."
    return ""


def _within_10_days(orders, now) -> bool:
    """수령(배송완료)일로부터 10일 이내가 '확인될 때만' True (매뉴얼 [환불보장제도]·[반품 가능 기간])."""
    if now is None:
        return False
    for r in orders or []:
        d = r.get("배송완료일자")
        if d:
            try:
                got = datetime.strptime(d[:10], "%Y-%m-%d")
            except ValueError:
                continue
            return 0 <= (now - got).days <= 10
    return False


def render_commit(name: str, result: dict, orders=None, now=None) -> str | None:
    st = result.get("status")
    if st == "already_requested":
        return "이미 접수된 건으로 확인돼요. 진행되는 대로 안내해 드릴게요."
    if st in {"invalid", "error"}:
        return None                                    # LLM 이 빠진 정보를 묻도록
    if st != "ok":
        return None
    if name in ("return_order", "return_order_item"):
        head = "반품 접수 완료됐어요." if name == "return_order" else "말씀하신 상품만 반품 접수됐어요."
        parts = [head]
        if _within_10_days(orders, now):        # 조건이 확인될 때만 (모르면 말하지 않는다)
            parts.append("수령하신 지 10일 이내라 환불보장제도로 배송비 없이 무상 반품돼요.")
        parts.append("CJ대한통운 기사님이 회수하신 뒤 물류센터에 입고되면 환불이 진행되고, 최대 2주 정도 걸릴 수 있어요.")
        rl = _refund_line(orders)
        if rl:
            parts.append(rl)
        return " ".join(parts)
    if name == "cancel_order":
        return " ".join(x for x in ["주문 취소 접수됐어요.", _refund_line(orders) or "환불은 승인 후 영업일 기준 3~4일 안에 진행돼요."] if x)
    return {
        "exchange_order": "교환 접수 완료됐어요. 회수 후 같은 상품으로 다시 보내 드릴게요.",
        "request_callback": "네, 연락처 남겨 두었어요. 담당 부서에서 평일 오전 10시부터 오후 5시 사이에 다시 전화드릴게요.",
    }.get(name)


def render(name: str, result: dict, session: dict, intent: str | None = None, utterance: str = "", orders=None) -> str | None:
    if name == "get_order_info":
        return render_order(result, session, intent, utterance)
    if name == "get_return_info":
        return render_return_info(result, session)
    return render_commit(name, result, orders, parse_now(session.get("now")))


# 함수 결과를 기다리는 동안 바로 낼 채움말 (미리 TTS 로 합성해 PCM 으로 들고 있으면 TTS 왕복이 없다)
FILLERS = {
    "get_order_info": ["네, 주문 내역 확인해 볼게요.", "네, 잠시만요. 주문 확인해 드릴게요."],
    "get_return_info": ["네, 반품 진행 상황 확인해 볼게요.", "잠시만요, 반품 내역 확인해 드릴게요."],
    "commit": ["네, 바로 접수해 드릴게요.", "네, 처리해 드릴게요. 잠시만요."],
    "request_callback": ["네, 연락처 남겨 드릴게요."],
}
