"""응대 위험도 판단 + 문장 검사.

빠른 응대보다 '잘못 말해서 회사가 책임질 일'을 막는 게 먼저인 경우를 코드로 걸러 낸다.
LLM 에게 "조심해"라고 부탁하는 것(프롬프트)만으로는 매번 지켜진다는 보장이 없어서, 두 겹으로 막는다.

1. assess()  — 답을 만들기 '전'에 위험도를 정한다 (규칙, 1ms 이하).
   HIGH   : LLM 이 자유롭게 답하지 않는다. 미리 정한 안전 문장 + 상담원 연결. (복약 병용, 이상 증상, 임신·질환 등)
   MEDIUM : LLM 이 답하되 제약을 건다. (제품을 특정 못 함, 수치·가격 계산, 다른 판매처, 효과 보장 요구)
   LOW    : 일반 응대.
2. guard_sentence() — LLM 이 만든 문장을 TTS 로 보내기 '직전'에 한 문장씩 검사한다 (정규식, 지연 거의 없음).
   - 효능 보장·치료 주장·'부작용 없음' 같은 금지 표현 → 안전 문장으로 교체
   - 근거(브랜드 정보·함수 결과·대화)에 없는 숫자+단위(원·정·포·일분·%·박스) → 그 문장을 빼고 '확인 후 안내'로 교체

판단 기록(Risk.reasons)은 로그로 남겨 발표·사후 검토에 쓴다.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

R = re.compile

# ── HIGH: 의료 판단이 필요한 질문 ───────────────────────────────────────────
MED_DRUG = R(r"(병원|처방|혈압약|당뇨약|고지혈증약|항생제|진통제|소염제|수면제|항응고|와파린|아스피린|갑상선약|스테로이드|약국\s*약|양약|한약|"
             r"먹는\s*약|복용\s*중인\s*약|약을?\s*(먹고|복용))")
TOGETHER = R(r"같이|함께|병용|먹어도|드셔도|복용해도|괜찮|상관|문제")
SYMPTOM = R(r"두드러기|발진|가려|간지러|복통|배가\s*(너무\s*|많이\s*|좀\s*|계속\s*)?아|아파요|아프고|아팠|설사|구토|토했|메스꺼|어지러|숨\s*(이|쉬)|호흡|부었|붓|통증|알레르기|피가|혈변|"
            r"열이\s*나|가슴이|심장|두통|머리가\s*아|이상\s*반응|부작용")
CONDITION = R(r"임신|임산부|수유\s*(중|부)|모유|수술|투석|항암|암\s*환자|간경화|신장\s*(병|질환)|질환|지병|입원\s*(중|했|해\s*있)|"
              r"당뇨|고혈압|혈압이\s*(높|낮)|고지혈|갑상선|간\s*수치|위염|위궤양|천식|아토피")
TAKE_Q = R(r"먹어도|드셔도|복용해도|섭취해도|괜찮|상관|문제\s*(없|있)|부작용|같이|함께")

# ── MEDIUM ─────────────────────────────────────────────────────────────────
OTHER_SELLER = R(r"쿠팡|네이버|스마트스토어|11번가|지마켓|G마켓|옥션|위메프|티몬|홈쇼핑|오픈마켓|올리브영|카카오\s*선물")
NUMERIC_Q = R(r"며칠\s*분|몇\s*(정|포|알|개|일|박스|달)|얼마|가격|금액|원짜리|할인|계산|총\s*\d|\d+\s*%")
EFFICACY_Q = R(r"효과\s*(있|보장|확실)|나아|낫나|낫겠|완치|치료|빠지나|살\s*빠|확실히")

# ── 문장 검사 ───────────────────────────────────────────────────────────────
FORBIDDEN = R(r"완치|치료(가\s*됩|됩니다|해\s*드|에\s*효과)|효과(를|가)?\s*(보장|확실)|100\s*%\s*효과|반드시\s*효과|"
              r"부작용(은|이)?\s*(전혀\s*)?없|의약품(과|처럼)\s*같|약\s*대신|약을?\s*끊|병이\s*낫|질병(을|이)?\s*(예방|치료)")
NUM_UNIT = R(r"(\d[\d,\.]*)\s*(원|정|포|알|일분|일\s*분|%|퍼센트|박스|개월|캡슐)")

# 안전 문장은 매뉴얼(브랜드 정보) 문구를 전화 말투로 옮긴 것이다. 지어낸 의학 판단을 넣지 않는다.
SAFE = {
    # [지병 및 복용 중인 약이 있는 경우 섭취 안내(공통)] — 상담원에게 넘기지 않고 매뉴얼 문구로 답한다
    "medical": "지병이 있으시거나 복용 중인 약이 있으시면, 고객님 안전을 위해 전문의와 상담하신 뒤에 섭취하시는 걸 권장드려요. "
               "제품 주성분은 판매 페이지 하단에서 확인하실 수 있어요.",
    # 이미 먹고 이상 증상: 위 매뉴얼 항목은 '이상 증상 호소에는 쓰지 않는다'고 적혀 있고, 시나리오는 상담원에게 넘기라고 한다.
    # → 이 경우만 상담원 연결을 유지하되, 섭취 중단 안내를 먼저 한다.
    "symptom": "많이 불편하셨겠어요. 우선 제품 섭취는 멈춰 주세요. 증상에 대한 부분은 제가 판단해 드리기 어려워서, "
               "전문 상담원이 확인해 드리도록 연결해 드릴게요.",
    "forbidden": "효과는 개인차가 있어서 단정해서 말씀드리기는 어려워요.",
    "ungrounded_number": "정확한 수치는 확인 후 다시 안내드릴게요.",
}


def medical_reply(text: str) -> str:
    """[지병 및 복용 중인 약이 있는 경우 섭취 안내(공통)] 문구를 손님 상황에 맞춰 첫머리만 바꾼다."""
    if re.search(r"임신|임산부|수유|모유", text):
        who = "임신 중이시거나 수유 중이시면"
    elif MED_DRUG.search(text):
        who = "복용 중인 약이 있으시면"
    else:
        who = "지병이 있으시면"
    return (f"{who}, 고객님 안전을 위해 전문의와 상담하신 뒤에 섭취하시는 걸 권장드려요. "
            "제품 주성분은 판매 페이지 하단에서 확인하실 수 있어요.")


def is_high_risk(session: dict, utterance: str) -> bool:
    """함수 호출 전 빠른 판단 (HIGH 면 상품 검색 같은 불필요한 호출을 하지 않는다)."""
    return assess(session, utterance).level == "high"


@dataclass
class Risk:
    level: str = "low"                         # low | medium | high
    reasons: list[str] = field(default_factory=list)
    safe_reply: str | None = None             # HIGH 일 때 LLM 대신 말할 문장
    hints: list[str] = field(default_factory=list)   # MEDIUM 일 때 LLM 에 줄 제약

    def bump(self, level: str, reason: str):
        order = {"low": 0, "medium": 1, "high": 2}
        if order[level] > order[self.level]:
            self.level = level
        self.reasons.append(reason)


def _search_names(calls) -> list[str]:
    names = []
    for c, r in calls or []:
        if getattr(c, "name", "") == "search_products":
            for row in ((r.get("data") or {}).get("결과") or []):
                n = row.get("제품명")
                if n and n not in names:
                    names.append(n)
    return names


def _norm(s: str) -> str:
    return re.sub(r"[^가-힣a-z0-9]", "", s.lower())


def product_matches(utterance: str, names: list[str], product_words: str) -> bool | None:
    """손님이 말한 제품 낱말이 검색 결과 제품명과 맞는지. 제품 낱말이 없으면 None(판단 안 함)."""
    # '대장영양제', '혈관청소약', '간 보호제'처럼 종류를 말한 단어는 특정 제품명이 아니라 검색으로 찾아 안내할 대상이다
    toks = [t for t in re.findall(r"[가-힣A-Za-z]{2,}", utterance)
            if (re.fullmatch(product_words, t) or re.search(r"(가드|세트|프로젝트|케어|캐어)$", t))
            and not re.search(r"(영양제|보호제|약|제품|상품)$", t)]
    if not toks:
        return None
    joined = _norm(" ".join(names))
    return any(_norm(t) in joined or any(_norm(t)[:2] in _norm(n) and len(_norm(t)) >= 3 for n in names) for t in toks)


def assess(session: dict, utterance: str, calls=(), committed: bool = False) -> Risk:
    """답을 만들기 전 위험도. committed=True 면 이번 턴에 접수 함수가 실행됨(접수 안내가 우선)."""
    from .planner import _product_words
    rk = Risk()
    u = utterance or ""
    # 이전 손님 말까지 같이 본다 ('혈압약 먹는데' → '같이 먹어도 돼요?')
    prev = " ".join(m.get("text") or "" for m in (session.get("history") or [])[-4:] if m.get("role") == "customer")
    ctx = prev + " " + u

    if MED_DRUG.search(ctx) and TOGETHER.search(u):
        rk.bump("high", "복약 병용 질문")
        rk.safe_reply = medical_reply(ctx)
    if SYMPTOM.search(u) and not committed and not re.search(r"부작용\s*(이|은)?\s*(없|있)\s*(나|을까|어요|는지)", u):
        rk.bump("high", "섭취 후 이상 증상")
        rk.safe_reply = SAFE["symptom"]          # 증상 호소는 매뉴얼상 상담원 연결 대상
        if re.search(r"환불|반품|교환", u):
            rk.safe_reply += " 반품이나 환불은 주문을 확인한 뒤 도와드릴 수 있으니, 주문번호나 주문하신 분 연락처도 알려 주세요."
    if CONDITION.search(ctx) and TAKE_Q.search(u):
        rk.bump("high", "지병·임신 등 건강 상태에서 섭취 문의")
        rk.safe_reply = rk.safe_reply or medical_reply(ctx)

    names = _search_names(calls)
    pm = product_matches(u, names, _product_words() or "x^") if names else None   # 검색을 했을 때만 판단
    if pm is False:
        rk.bump("medium", "제품 특정 불가")
        rk.hints.append("손님이 말한 제품이 검색 결과의 제품과 같은지 확실하지 않다. 제품 정보를 말하지 말고, 정확한 제품명을 먼저 여쭤라.")
    if NUMERIC_Q.search(u):
        rk.bump("medium", "수치·가격 질문")
        rk.hints.append("수치(가격·할인 금액·정 수·며칠분)는 브랜드 정보나 검색 결과에 그대로 적힌 숫자만 말하라. 계산하거나 짐작하지 마라.")
    if OTHER_SELLER.search(ctx):
        rk.bump("medium", "다른 판매처 구매")
        rk.hints.append("다른 판매처 구매 건이다. 자사몰 정책(환불보장 등)을 적용해 약속하지 말고, 구매처와 문의 내용을 확인한 뒤 상담원 확인을 안내하라.")
    if EFFICACY_Q.search(u):
        rk.bump("medium", "효과 보장 요구")
        rk.hints.append("효과를 보장하거나 치료처럼 말하지 마라. 개인차가 있다고 안내하라.")
    return rk


def grounding_text(session: dict, calls=(), extra: str = "") -> str:
    """숫자 검사에 쓸 근거: 브랜드 정보 + 이번 함수 결과 + 대화 + 이번 손님 말."""
    import os
    base = os.path.join(os.path.dirname(__file__), "..", "data", "brand", "information.json")
    try:
        info = " ".join(x["content"] for x in json.load(open(base, encoding="utf-8")))
    except Exception:
        info = ""
    hist = " ".join(json.dumps(m, ensure_ascii=False) for m in session.get("history") or [])
    res = " ".join(json.dumps(r, ensure_ascii=False) for _, r in (calls or []))
    return " ".join([info, hist, res, extra])


_UNIT_NORM = {"일 분": "일분", "퍼센트": "%"}


def _num_in(num: str, unit: str, text: str) -> bool:
    """'30일분'처럼 숫자와 단위를 '함께' 근거에서 찾는다 (숫자 30 만 어딘가 있다고 통과시키지 않는다)."""
    n = num.replace(",", "")
    u = _UNIT_NORM.get(unit, unit).replace(" ", "")
    t = re.sub(r"[\s,]", "", text)
    return f"{n}{u}" in t


def guard_sentence(sentence: str, grounding: str) -> tuple[str, str | None]:
    """(내보낼 문장, 바꾼 이유 또는 None)"""
    if FORBIDDEN.search(sentence):
        return SAFE["forbidden"], "금지 표현"
    for m in NUM_UNIT.finditer(sentence):
        if not _num_in(m.group(1), m.group(2), grounding):
            return SAFE["ungrounded_number"], f"근거 없는 수치 {m.group(0)}"
    return sentence, None
