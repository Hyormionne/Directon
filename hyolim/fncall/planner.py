"""어떤 함수를 언제 부를지 정한다.

- plan_lookups(): LLM 을 기다리지 않고 바로 쏠 '조회' 호출 목록. 조회는 더 불러도 감점이 없으므로
  번호가 보이거나 조회가 필요한 문의면 즉시 호출해 함수 지연(1.0~1.5초)을 LLM 생성과 겹친다.
- gate_commit(): '접수' 함수(취소·반품·교환·전화요청)를 통과시킬지 결정하고 번호 인수를 코드가 고른 값으로
  덮어쓴다. 정답에 없는 접수 호출은 함수 점수 0점이라 보수적으로 막는다. LLM 이 고른 호출도 반드시 여기를 거친다.
- suggest_commit(): LLM 없이 규칙만으로 접수 여부를 추정한다(오프라인 평가, LLM 판단 교차 검증용).

문맥 판단 순서: 이번 손님 말 → 상담원의 마지막 질문 → 이전 손님 말.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .slots import Slots, digits

LOOKUP_FUNCS = {"get_order_info", "get_return_info"}
COMMIT_FUNCS = {"cancel_order", "return_order", "return_order_item", "exchange_order", "request_callback"}

R = re.compile
# 손님 말 의도
INTENT_RULES = [   # 위에 있는 규칙이 우선한다
    ("callback", R(r"전화\s*(로)?\s*(주문|구매|연락\s*(주|부탁)|주세요|주실|부탁|해\s*주|연결|통화)|연락\s*(주세요|주실|부탁|바랍니다|해\s*주)|"
                   r"통화\s*(가능|원해|하고\s*싶|부탁|요청|해\s*주|할\s*수|돼|되)|유선|전화\s*(상담|주문)|콜백|다시\s*전화")),
    ("return_status", R(r"(환불|반품).{0,15}(언제|아직|늦|안\s*(들어|됐|되었|왔|와요|와서)|현황|진행\s*(상황|중|됐|되)|됐나|되었나|처리\s*(됐|되었|중))|수거|회수\s*(해\s*갔|됐|되었|해갔|했)|가져\s*갔")),
    ("cancel", R(r"취소")),
    ("exchange", R(r"교환|바꿔\s*주|다른\s*(걸|거|제품)으로")),
    ("return_request", R(r"반품|환불|환풀|환뷸|환블|반풍|반퓸|리턴")),
    ("purchase", R(r"(구매|구입)\s*(하고\s*싶|하려|할래|원해|희망|하고\s*파|라고\s*싶|요망)|사고\s*싶|살\s*(게요|래요|줄\s*몰)|주문\s*(하고\s*싶|하려|방법|할게)|\d\s*(박스|통)\s*주문")),
    ("address_change", R(r"(배송지|주소).{0,8}(변경|바꾸|바꿔|수정)")),
    ("order_status", R(r"배송|언제\s*(와|오|도착|받)|도착|출고|발송|택배|주문\s*(확인|됐|되었|들어|조회|내역)|입금|결제\s*(확인|됐|되었)|송장|못\s*받|안\s*(왔|와요|오)")),
    ("product", R(r"성분|효능|효과|섭취|먹는\s*법|복용|가격|얼마|구성|추천|유통기한|소비기한")),
]
# 상담원의 마지막 말이 어떤 흐름인지
AGENT_RULES = [
    ("return_status", R(r"(환불|반품)[·\s/,]*(반품|환불)?\s*(진행|처리)?\s*(현황|상황|내역)")),
    ("return_request", R(r"반품.{0,12}(접수|진행)|전체\s*반품|부분\s*반품|일부(만|\s*수량)|환불\s*계좌|계좌\s*정보|반품.{0,6}사유|회수.{0,15}주소|수거.{0,15}주소|배송지\s*주소가")),
    ("cancel", R(r"취소.{0,8}(접수|진행|사유|원하)")),
    ("exchange", R(r"교환.{0,8}(접수|진행|원하)")),
    ("callback", R(r"다시\s*(전화|연락)\s*드|연락\s*드리겠|전화\s*(상담|주문)\s*(도와|요청|접수)|유선\s*통화|성함.{0,12}(남겨|알려)")),
    ("purchase", R(r"구매는|ARS|주문\s*방법|온라인\s*구매")),
]
CLOSING = R(r"(좋은|편안한|행복한|즐거운)\s*하루|안내\s*도움\s*되셨|감사합니다\.?\s*$")
AUTO_CLOSE = R(r"자동으로\s*종료")
AFFIRM = R(r"^\s*(네|예|넵|응|맞아요|맞습니다|그렇게|좋아요)|전체|전부|모두|다\s*(환불|반품)|해\s*주세요|부탁드|진행해|접수해|신청합니다|처리.*(부탁|해)|"
           r"주소(지)?\s*(는|가|도)?\s*(동일|같|똑같|그대로|맞|변경\s*없|바뀐\s*거\s*없)|맞아요|일부|부분")
# '한 개만 주문했는데'는 일부 반품이 아니다. '한 개만 반품/환불'처럼 반품 대상일 때만 일부로 본다
PARTIAL = R(r"일부|부분|(한|두|세)\s*(개|박스|통|포)만\s*(반품|환불|보내|돌려|회수)|남은\s*(것|거)만")

# 의도 판단 전에 군말을 지운다 — '통화 좀 가능할까요', '혹시 전화 연결 돼요' 같은 표현이 규칙을 깨지 않게
FILLER = R(r"(?<![가-힣])(좀|혹시|한번|그냥|지금|저기|음|어|아|그|막|진짜|정말|빨리|일단|그러니까)(?![가-힣])")


def soften(text: str) -> str:
    return re.sub(r"\s+", " ", FILLER.sub(" ", text or "")).strip()
NEGATE = R(r"(안\s*할|안\s*해도|안\s*할래|말고|안\s*하고)|아니요|아닙니다|괜찮아요|됐어요|그냥\s*둘")
EXPLICIT = R(r"(반품|환불|취소|교환).{0,6}(해\s*주|처리|신청|접수|부탁|진행|입니다|할게)|전체|전부|모두|다\s*환불")
REASON_Q = R(r"사유|이유|어떤\s*점")
CONFIRM_Q = R(r"(전체|일부|부분).{0,25}(반품|진행|접수)|접수\s*해\s*드릴까요|진행\s*해\s*드릴까요|도와\s*드릴까요|원하실까요|원하시는지|주소.{0,20}(같은지|동일|맞으신지)|맞으시면|이어서\s*진행|진행할지|진행하실지")


def classify(text: str) -> list[str]:
    t = soften(text)
    return [name for name, rx in INTENT_RULES if rx.search(t)]


def agent_flow(text: str) -> str | None:
    for name, rx in AGENT_RULES:
        if rx.search(text or ""):
            return name
    return None


def has_purchase_ctx(session: dict, utterance: str) -> bool:
    """전화 요청(콜백)을 받을 만한 문맥인가: 이번 말이나 이전 손님 말에 구매 의사, 또는 상담원이 콜백 흐름을 진행 중."""
    if "purchase" in classify(utterance):
        return True
    if any(m.get("role") == "customer" and "purchase" in classify(m.get("text") or "") for m in session.get("history") or []):
        return True
    return agent_flow(last_agent(session)) in {"callback", "purchase"}


def last_agent(session: dict) -> str:
    """자동 종료 안내는 건너뛴 상담원의 마지막 말."""
    for m in reversed(session.get("history") or []):
        if m.get("role") == "agent" and not AUTO_CLOSE.search(m.get("text") or ""):
            return m.get("text") or ""
    return ""


def fresh_history(session: dict) -> list[dict]:
    """상담원이 인사로 마무리한 뒤 새로 시작된 부분만. 그 앞의 조회 결과로는 접수하지 않는다."""
    hist = session.get("history") or []
    cut = 0
    for i, m in enumerate(hist):
        t = m.get("text") or ""
        if m.get("role") == "agent" and CLOSING.search(t) and not re.search(r"\?|까요|주시겠|주세요", t):
            cut = i + 1
    return hist[cut:]


def topic(session: dict, utterance: str) -> str | None:
    it = classify(utterance)
    if it and it[0] == "callback" and not has_purchase_ctx(session, utterance):
        it = it[1:]          # 환불·조회 문맥의 "이 번호로 연락 주세요"는 전화 요청이 아니다 → 다른 주제로 판단
    # 번호만 불러 준 경우처럼 이번 말에 주제가 없거나, '구매'처럼 상담원 맥락이 더 정확한 경우
    flow = agent_flow(last_agent(session))
    if it and not (it[0] in {"return_request", "product"} and flow in {"return_status", "callback"}):
        return it[0]
    if flow:
        return flow
    for m in reversed(session.get("history") or []):
        if m.get("role") == "customer":
            it = classify(m.get("text") or "")
            if it:
                return it[0]
    return None


@dataclass
class Call:
    name: str
    arguments: dict
    why: str = ""


def _product_words() -> str:
    """브랜드 정보의 [제품명 리스트]에서 제품을 가리키는 낱말을 뽑는다 (조인플렉스, 루테잔틴, 프로폴리스 …)."""
    import json, os
    try:
        info = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "brand", "information.json"), encoding="utf-8"))
        names = next(x["content"] for x in info if x["tag"] == "[제품명 리스트]")
    except Exception:
        return ""
    generic = {"하루온", "상품명", "가드", "세트", "함유", "식물성", "포", "맨", "올인원", "프로", "PICK"}
    toks = {t for t in re.findall(r"[가-힣A-Za-z]{2,}", names) if t not in generic}
    return "|".join(sorted(toks, key=len, reverse=True))


PRODUCT_WORD = R(r"영양제|보호제|가드|세트|제품|상품|유산균|맥스|눈\s*약|간\s*약|하루온\s*[가-힣]|" + _product_words())


PRODUCT_INFO_Q = R(r"섭취|복용|먹(는|어|으면|을)|성분|효능|효과|구성|며칠|몇\s*(정|포|알|개)|추천|어떤\s*(게|거|제품)|뭐가|차이|"
                   r"유통기한|소비기한|용량|함량|맛|\?")
PAY_OR_FEE = R(r"결제|카드|무통장|입금|가상계좌|현금영수증|영수증|배송비|적립금|쿠폰")


def wants_search(session: dict, utterance: str) -> bool:
    """상품 정보가 필요한 말이면 search_products 를 같이 부른다 (함수 점수와 무관, 응대 사실 확인용)."""
    t = topic(session, utterance)
    if t in {"order_status", "return_status", "cancel", "return_request", "exchange", "address_change", "callback"}:
        return False
    from .risk import is_high_risk
    if is_high_risk(session, utterance):
        return False                                    # 건강 관련 고위험 문의는 매뉴얼 문장으로 답한다 — 검색 불필요
    u = soften(utterance)
    if PAY_OR_FEE.search(u) and not PRODUCT_INFO_Q.search(u):
        return False                                    # 결제·배송비 문의는 브랜드 정보로 답한다
    if t == "purchase" or has_purchase_ctx(session, utterance) and not PRODUCT_INFO_Q.search(u):
        return False                                    # 구매·전화 주문 요청엔 구매 방법 안내가 핵심 — 검색 불필요
    # 제품 정보를 묻거나, 다른 용건 없이 제품 이름만 말한 경우("대장영양제 하루온 대장영양제")
    return t == "product" or (bool(PRODUCT_WORD.search(u)) and (bool(PRODUCT_INFO_Q.search(u)) or t is None))


def plan_lookups(session: dict, utterance: str, slots: Slots, max_variants: int | None = None) -> list[Call]:
    """지금 바로 보낼 조회 호출들. 후보가 여럿이면 여러 개를 같이 보내 검증한다(verify.py).
    들은 번호가 확실하면(신뢰도 0.9 이상) 1개만, 애매하면 변형 후보를 최대 8개까지 동시에 보낸다.
    조회는 더 불러도 감점이 없고, 병렬이라 지연도 한 번만 든다."""
    if max_variants is None:
        top = (slots.utter_phones or slots.utter_orders or [None])[0]
        max_variants = 1 if top is None or top.conf >= 0.9 else 8
    t = topic(session, utterance)
    has_new_id = bool(slots.utter_phones or slots.utter_orders)
    fresh_seen = {m.get("name") for m in fresh_history(session) if m.get("role") == "tool_result"}
    calls: list[Call] = []

    if t in {"callback", "purchase"} and not slots.utter_orders:
        return calls                                    # 전화 요청은 조회가 필요 없다
    if t == "product" and not has_new_id:
        return calls

    if t == "return_status":
        phones = [c.value for c in slots.phones[:max_variants]] if has_new_id else [c.value for c in slots.phones[:1]]
        for p in phones:
            calls.append(Call("get_return_info", {"customer_phone_number": p}, "반품 현황"))
        return calls

    need_order = t in {"order_status", "cancel", "return_request", "exchange", "address_change"}
    if has_new_id or (need_order and "get_order_info" not in fresh_seen):
        orders = [c.value for c in slots.utter_orders[:max_variants]] or [None]
        phones = [c.value for c in slots.utter_phones[:max_variants]] or [slots.phone]
        pairs = {(o, phones[0]) for o in orders} | {(orders[0], p) for p in phones}
        for o, p in sorted(pairs, key=lambda x: (x[0] is None, str(x))):
            if o or p:
                calls.append(Call("get_order_info", {"order_id": o, "buyer_cellphone": p}, f"topic={t}"))
    return calls


def _reason_from_history(session: dict, utterance: str) -> str:
    texts = [m.get("text") or "" for m in (session.get("history") or []) if m.get("role") == "customer"]
    texts.append(utterance)
    for t in reversed(texts):
        if re.search(r"효과|반응|불량|파손|맛|냄새|가스|배변|화장실|안\s*맞|변심|잘못|누락|덜\s*왔|안\s*왔|개만", t):
            return t.strip()[:120]
    return (texts[-1] or "고객 요청").strip()[:120]


def gate_commit(name: str, llm_args: dict, session: dict, utterance: str, slots: Slots) -> tuple[dict | None, str]:
    """접수 호출 검사. (최종 인수 또는 None, 사유). None 이면 호출하지 말고 사유에 맞게 되묻는다.
    사유가 'lookup_first' 면 주문 조회 후 slots.known_orders 를 채워 다시 부르면 된다."""
    a = dict(llm_args or {})
    if name == "request_callback":
        phone = slots.utter_phones[0].value if slots.utter_phones else slots.phone
        if not phone:
            return None, "need_phone"
        a["phone_number"] = phone
        a["customer_name"] = a.get("customer_name") or slots.name or "고객"
        return a, "ok"
    if name not in {"cancel_order", "return_order", "return_order_item", "exchange_order"}:
        return a, "not_commit"
    if NEGATE.search(utterance) and not EXPLICIT.search(utterance):
        return None, "customer_declined"
    if not slots.known_orders:
        return None, "lookup_first"
    order = _pick_order(slots)
    if not order:
        return None, "need_order"
    distinct = {digits(r.get("주문번호")) for r in slots.known_orders if r.get("주문번호")}
    if len(distinct) > 1 and not any(digits(c.value) in distinct for c in slots.utter_orders) \
            and not any(digits(c.value) in distinct for c in slots.orders if c.source == "history_call"):
        return None, "multi_order"                     # 어느 주문인지 먼저 여쭌다
    if not str(a.get("reason") or "").strip():
        a["reason"] = _reason_from_history(session, utterance)
    if name == "cancel_order":
        a["order_id"] = order
        a.setdefault("customer_phone_number", slots.phone)
    elif name == "return_order_item":
        codes = sorted({r.get("상세주문번호") for r in slots.known_orders
                        if digits(r.get("주문번호")) == digits(order) and r.get("상세주문번호")})
        items = [it for it in (a.get("return_items") or []) if it.get("order_item_code") in codes]
        items = items or [{"order_item_code": c, "quantity": 1} for c in codes[:1]]
        if not items:
            return None, "need_item"
        a["return_items"] = items
    else:
        a["order_number"] = order
    if name in ("return_order", "return_order_item", "cancel_order") and not a.get("refund_account"):
        acc = refund_account(session, utterance, slots)
        if acc:
            a["refund_account"] = acc
    return a, "ok"


BANK = R(r"(카카오뱅크|토스뱅크|케이뱅크|새마을금고|우체국|농협|신협|수협|[가-힣]{1,6}은행|[가-힣]{1,4}뱅크)")


def refund_account(session: dict, utterance: str, slots: Slots) -> dict | None:
    """무통장·가상계좌 결제 반품이면 대화에서 환불 계좌(은행·계좌번호·예금주)를 찾는다.
    상담원이 '- 은행명: … - 계좌번호: … - 예금주: …' 로 되짚어 준 말이 있으면 그것을 먼저 쓴다."""
    pays = " ".join(" ".join(r.get("결제수단") or []) if isinstance(r.get("결제수단"), list) else str(r.get("결제수단") or "")
                    for r in slots.known_orders)
    hist = session.get("history") or []
    asked = any(m.get("role") == "agent" and re.search(r"계좌", m.get("text") or "") for m in hist)
    if not (re.search(r"무통장|가상계좌|계좌이체", pays) or asked):
        return None
    for m in reversed(hist):
        t = m.get("text") or ""
        if m.get("role") == "agent":
            b = re.search(r"은행명\s*[:：]\s*([^\s\-]+)", t)
            a = re.search(r"계좌번호\s*[:：]\s*([\d\s\-]{6,})", t)
            h = re.search(r"예금주\s*[:：]\s*([가-힣A-Za-z]+)", t)
            if b and a:
                return {"bank_name": b.group(1), "account_no": digits(a.group(1)), "holder": h.group(1) if h else (slots.name or "")}
    for t in [utterance] + [m.get("text") or "" for m in reversed(hist) if m.get("role") == "customer"]:
        b = BANK.search(t)
        if not b:
            continue
        rest = t[b.end():]
        a = re.search(r"\d[\d\s\-]{6,}\d", rest)
        if not a:
            continue
        h = re.search(r"([가-힣]{2,4})", rest[a.end():])
        return {"bank_name": b.group(1), "account_no": digits(a.group(0)),
                "holder": h.group(1) if h and h.group(1) not in ("입니다", "이에요", "예금주") else (slots.name or "")}
    return None


def _pick_order(slots: Slots) -> str | None:
    known = [r.get("주문번호") for r in slots.known_orders if r.get("주문번호")]
    kd = {digits(k) for k in known}
    for c in slots.orders:
        if digits(c.value) in kd:
            return c.value
    return known[0] if known else None


def suggest_commit(session: dict, utterance: str, slots: Slots, after_lookup: bool = False) -> tuple[Call | None, str]:
    """규칙만으로 '이번 턴에 접수해야 하나'를 추정한다. (호출 또는 None, 사유)
    after_lookup=True 는 이번 턴에 방금 주문 조회를 해서 known_orders 가 채워진 상태라는 뜻."""
    t = topic(session, utterance)
    agent = last_agent(session)
    flow = agent_flow(agent)
    intents = classify(utterance)

    # 전화 요청: 구매·통화 문맥에서 번호(또는 이름)를 받았을 때
    if t in {"callback", "purchase"} or "callback" in intents:
        # 시나리오 9: 전화 요청은 '혼자 주문하기 어려워 전화로 도움받고 싶은' 경우다. 구매 문맥이 있어야 한다.
        # (환불 문의 중 "이 번호로 연락 주세요", "상담원하고 통화하고 싶어요"는 전화 요청 접수가 아니다)
        ctx_callback = has_purchase_ctx(session, utterance)
        if ctx_callback and (slots.utter_phones or "callback" in intents):
            args, why = gate_commit("request_callback", {}, session, utterance, slots)
            return (Call("request_callback", args, why) if args else None), why
        return None, "no_callback_info"

    if t not in {"return_request", "cancel", "exchange"}:
        return None, "not_commit_topic"
    asked = bool(CONFIRM_Q.search(agent) or REASON_Q.search(agent)) and flow in {"return_request", "cancel", "exchange", None}
    affirm = bool(AFFIRM.search(soften(utterance)))
    explicit = bool(EXPLICIT.search(soften(utterance)))
    fresh_has_order = any(m.get("role") == "tool_result" and m.get("name") == "get_order_info"
                          for m in fresh_history(session))
    if asked and affirm:
        pass                                            # 상담원이 확인을 구했고 손님이 답함 → 접수
    elif explicit and fresh_has_order and not after_lookup:
        pass                                            # 조회 뒤 손님이 분명히 요청
    else:
        return None, "no_consent"
    if not slots.known_orders or (not fresh_has_order and not after_lookup):
        if not after_lookup:
            return None, "lookup_first"
    name = {"cancel": "cancel_order", "exchange": "exchange_order"}.get(t, "return_order")
    if name == "return_order" and PARTIAL.search(soften(utterance)):
        name = "return_order_item"
    args, why = gate_commit(name, {}, session, utterance, slots)
    return (Call(name, args, why) if args else None), why
