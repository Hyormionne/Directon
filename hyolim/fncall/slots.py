"""대화 상태에서 함수 인수(전화번호·주문번호·이름)를 코드로 정한다.

LLM 이 번호를 옮겨 적다 틀리는 일을 막기 위해, 번호는 항상 여기서 고른 값으로 덮어쓴다.

우선순위
  1. 이번 손님 발화에서 뽑은 번호
  2. 이전 대화(history)의 함수 호출 인수·조회 결과·손님 말에 있는 번호 (최근 것 우선)
  3. 발신자 정보(caller.phone) — 손님이 다른 번호를 말하지 않았을 때만
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from .numbers_ko import Cand, extract_ids, parse_now

digits = lambda v: re.sub(r"\D", "", str(v or ""))

PHONE_ARGS = ("buyer_cellphone", "customer_phone_number", "phone_number")
ORDER_ARGS = ("order_id", "order_number")

NAME_STOP = set("""고객 고객님 주문 주문확인 주문번호 연락처 전화 전화번호 핸드폰 휴대폰 번호 입니다 이에요 예요 감사합니다
안녕하세요 네 예 반품 환불 교환 취소 배송 문의 확인 부탁 부탁드려요 부탁드립니다 하루온 유선통화 통화 제품구매하고
싶어요 예금자 패키지 성함 이름 저는 제 맞아요 주소 주소지 동일 연락 연락주세요 수고하세요""".split())


@dataclass
class Slots:
    now: datetime | None
    phones: list[Cand] = field(default_factory=list)     # 신뢰도 순
    orders: list[Cand] = field(default_factory=list)
    name: str | None = None
    caller_phone: str | None = None
    known_orders: list[dict] = field(default_factory=list)   # 이전 조회 결과의 주문 행들 (최근 조회 것)
    seen_results: dict = field(default_factory=dict)          # {함수명: 마지막 result}
    utter_phones: list[Cand] = field(default_factory=list)
    utter_orders: list[Cand] = field(default_factory=list)

    @property
    def phone(self) -> str | None:
        return self.phones[0].value if self.phones else None

    @property
    def order(self) -> str | None:
        return self.orders[0].value if self.orders else None


def _edit1(a: str, b: str) -> bool:
    """편집 거리 1 이하 (한 자리 누락·추가·바뀜)"""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    if len(a) > len(b):
        a, b = b, a
    return any(b[:i] + b[i + 1:] == a for i in range(len(b)))


SURNAMES = set("김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노하곽성차주우구민진나지엄채원천방공현함변"
               "염여추도소석선설마길연위표명기반왕금옥육인맹제모탁국어은편용예경봉사부가복태목형피두감음빈동온호범좌팽승간상갈단견당화창")
NAME_END_NOT = re.compile(r"(요|니다|세요|어요|아요|죠|까|는데|해서|려고|려구|하고|합니다|해요|님|씩|째|쯤|들|됨|함|음|임)$")
NAME_WORD_NOT = re.compile(r"(주문|구매|전화|통화|영양|가드|세트|박스|배송|반품|환불|교환|취소|확인|문의|연락|가능|감사|안녕|"
                           r"하루온|은행|제품|상품|고객|번호|주소|입금|결제|수거|회수|택배|아파트|정도|이거|그거|저거|지금|오늘|내일|어제|"
                           r"방법|금액|상담|유선|복용|섭취|구입|원짜리|이였|였으|시즌|이뤄|제풀|두개|한개|세개)")


def _extract_name(text: str) -> str | None:
    """손님 이름(2~4글자) 고르기. 전화 요청 함수의 customer_name 으로 들어가고, 응대 채점 AI 가 이 값을 본다.

    규칙: 띄어쓰기·숫자로 떨어진 '한 단어 전체'만 후보로 본다(단어 일부를 잘라 쓰지 않는다).
    흔한 성씨로 시작하고, 동사 어미·일반 명사가 아니어야 한다. 번호 바로 옆 단어, 3글자 단어를 우선한다."""
    clean = re.sub(r"[^\w가-힣]", " ", text)
    words = re.findall(r"[가-힣]+|\d+", clean)
    best, best_score = None, -1.0
    short = len(words) <= 4                     # '통화 가능한가요? 은하진' 처럼 짧은 말
    for i, w in enumerate(words):
        if not re.fullmatch(r"[가-힣]{2,4}", w) or w in NAME_STOP or w[0] not in SURNAMES \
                or NAME_END_NOT.search(w) or NAME_WORD_NOT.search(w):
            continue
        near_num = any(j >= 0 and j < len(words) and words[j].isdigit() for j in (i - 1, i + 1))
        if not (near_num or short):
            continue                            # 긴 문장 속 일반 단어('모두', '어디서')를 이름으로 오인하지 않게
        score = (2.0 if near_num else 0.0) + (1.0 if len(w) == 3 else 0.3) + 0.01 * i   # 뒤쪽(자기소개 끝) 약간 우선
        if score > best_score:
            best, best_score = w, score
    return best


def _merge(dst: list[Cand], src: list[Cand], conf_scale: float = 1.0, source: str | None = None):
    have = {c.value for c in dst}
    for c in src:
        if c.value not in have:
            dst.append(Cand(c.value, c.kind, c.conf * conf_scale, source or c.source, c.note))
            have.add(c.value)


def resolve(session: dict, utterance: str) -> Slots:
    """session = session.start 메시지(dict). utterance = 이번 손님 말(STT 결과)."""
    now = parse_now(session.get("now"))
    hist = session.get("history") or []
    caller = session.get("caller") or {}
    s = Slots(now=now, caller_phone=digits(caller.get("phone")) or None)

    # 1) 이번 발화
    u = extract_ids(utterance, now, "utterance")
    s.utter_phones = [c for c in u if c.kind == "phone"]
    s.utter_orders = [c for c in u if c.kind == "order"]
    _merge(s.phones, s.utter_phones)
    _merge(s.orders, s.utter_orders)

    # 2) 이전 대화 — 뒤에서부터(최근 것 우선)
    h_phones, h_orders = [], []
    for i, m in enumerate(reversed(hist)):
        decay = max(0.5, 0.9 - 0.02 * i)
        role = m.get("role")
        if role == "tool_call":
            a = m.get("arguments") or {}
            for k in PHONE_ARGS:
                if digits(a.get(k)):
                    _merge(h_phones, [Cand(digits(a[k]), "phone", 0.9, "history_call")], decay)
            for k in ORDER_ARGS:
                if digits(a.get(k)):
                    _merge(h_orders, [Cand(str(a[k]).strip(), "order", 0.9, "history_call")], decay)
        elif role == "tool_result":
            name, res = m.get("name"), m.get("result") or {}
            s.seen_results.setdefault(name, res)
            data = res.get("data") if isinstance(res.get("data"), dict) else {}
            rows = data.get("orders") or data.get("반품정보") or []
            if name == "get_order_info" and rows and not s.known_orders:
                s.known_orders = rows
            ids = []
            for r in rows:
                if r.get("주문번호") and r["주문번호"] not in ids:
                    ids.append(r["주문번호"])
            # 조회 결과가 한 건이면 그 주문이 대화의 대상일 가능성이 높다
            conf = 0.85 if len(ids) == 1 else 0.6
            _merge(h_orders, [Cand(x, "order", conf, "history_result") for x in ids], decay)
        elif role == "customer":
            t = m.get("text") or ""
            c = extract_ids(t, now, "history_customer")
            _merge(h_phones, [x for x in c if x.kind == "phone"], decay * 0.95)
            _merge(h_orders, [x for x in c if x.kind == "order"], decay * 0.95)
            if not s.name and c:
                s.name = _extract_name(t)
        elif role == "agent":
            t = m.get("text") or ""
            c = extract_ids(t, now, "history_agent")
            _merge(h_orders, [x for x in c if x.kind == "order"], decay * 0.7)

    # 여러 주문 중 '손님이 직접 말한 번호'가 있으면 그것을 앞으로
    h_orders.sort(key=lambda c: -c.conf)
    _merge(s.phones, h_phones)
    _merge(s.orders, h_orders)

    # 3) 발신번호 — 들은 번호가 발신번호와 한 글자만 다르면 오인식으로 보고 발신번호를 앞세운다
    if s.caller_phone and s.utter_phones and s.utter_phones[0].conf < 0.9 and \
            any(_edit1(c.value, s.caller_phone) for c in s.utter_phones):
        boosted = Cand(s.caller_phone, "phone", 0.92, "caller~utterance")
        s.utter_phones.insert(0, boosted)
        s.phones = [boosted] + [c for c in s.phones if c.value != s.caller_phone]
    if s.caller_phone:
        _merge(s.phones, [Cand(s.caller_phone, "phone", 0.5 if s.phones else 0.8, "caller")])

    s.name = _extract_name(utterance)                 # 이번 말에 자기 이름을 말했으면 그것이 최우선
    if not s.name:
        for m in reversed(hist):
            if m.get("role") == "customer" and extract_ids(m.get("text") or "", now):
                s.name = _extract_name(m.get("text") or "")
                if s.name:
                    break
    if not s.name:
        s.name = caller.get("name")
    return s
