"""AG-T24 Jev(TypeSafe System One) 사람 인계 판정. 근거·지표: docs/decisions/jev_handoff.md

확정된 고객 발화 + 최근 대화(+ 호출자가 세는 이해 실패 횟수)로 인계 기준 6개를 Noul로 한 요청에 묻는다(병렬 평가).
정책은 코드가 가진다: 기준별 임계값(THRESHOLDS), "하나라도 넘으면 인계".
LLM 선호출과 함께 부르면 지연이 더해지지 않는다(응답 p50 ~200ms < LLM 첫 토큰).
실패·시간 초과는 None = 강제 인계 없음(LLM 판단 그대로).
기본 꺼짐: DIRECTON_JEV_HANDOFF=1일 때만 maybe_judge가 호출한다. 세션·응답기 연결안은 결정 문서 6절.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from agent.turn.jev_turn import SystemOneClient, _noul, make_client

log = logging.getLogger(__name__)

MODEL = "jev-1.14.0"
ENV_SWITCH = "DIRECTON_JEV_HANDOFF"

SITUATION = ("Korean phone customer-service call handled by an AI agent for an online shop that also sells health "
             "supplements. Human agents are available, and the AI must hand the call to a human in specific cases. "
             "`customer_utterance` is a speech-to-text transcript of what the customer just said (it may contain "
             "recognition errors); `recent_conversation` is the earlier part of the call; "
             "`agent_failed_to_understand_count` is how many times in this call the AI agent has already failed to "
             "understand or resolve the customer's request.")

CRITERIA: dict[str, dict] = {
    "health": {
        "type": "noul",
        "instructions": ("Does the customer report a physical symptom, discomfort, injury or adverse reaction after "
                         "taking or using a product, or keep asking for a medical judgment (illness, medication, "
                         "pregnancy) after the agent has already given the product's caution?"),
        "criteria": {
            "true": ("Body discomfort, rash, pain, nausea, dizziness, burn, injury or other reaction linked to a "
                     "product, or a repeated request for medical judgment after the caution was given."),
            "false": ("Ordinary product, order, delivery, payment or policy questions, including ingredients, how to "
                      "take it, or a first question whether it is safe with a condition or medication (the agent "
                      "answers that with the product's caution)."),
        },
    },
    "human_request": {
        "type": "noul",
        "instructions": ("Is the customer asking to be connected to a human agent, a manager or someone in charge, "
                         "or accepting the AI agent's offer to connect them to one?"),
        "criteria": {
            "true": "An explicit request for, or acceptance of, a connection to a human agent, manager or person in charge.",
            "false": ("No request to be connected. Only asking whether or when human agents are available "
                      "(business hours, weekends, lunch time) is not a request."),
        },
    },
    "abuse_anger": {
        "type": "noul",
        "instructions": ("Is the customer swearing, insulting, or strongly angry or agitated (heated complaint, "
                         "rhetorical outrage such as 'does this make sense?', saying they are fed up)?"),
        "criteria": {
            "true": "Profanity, insults, or a strongly angry, agitated complaint.",
            "false": "Calm or only mildly dissatisfied, including polite complaints and ordinary requests.",
        },
    },
    "repeated_failure": {
        "type": "noul",
        "instructions": ("Has communication with the AI agent repeatedly failed: the customer says they have asked "
                         "several times or that the agent keeps misunderstanding or not solving it, or "
                         "`agent_failed_to_understand_count` is 2 or more?"),
        "criteria": {
            "true": "Repeated misunderstanding or unresolved repetition in this or earlier contacts.",
            "false": "No sign of repeated failure.",
        },
    },
    "legal_dispute": {
        "type": "noul",
        "instructions": ("Does the customer raise a legal or formal dispute: threatening to report to a consumer "
                         "agency or the police, a lawsuit, a card-payment dispute or chargeback, or fraud or "
                         "unauthorized use?"),
        "criteria": {
            "true": "A legal threat, formal complaint to an authority, payment dispute, or fraud claim.",
            "false": "No legal or formal dispute.",
        },
    },
    "out_of_policy": {
        "type": "noul",
        "instructions": ("Does the customer need an exception or a case-by-case decision that a scripted agent "
                         "following the store manual cannot make: e.g. a refund or return after the deadline, "
                         "compensation, a bulk or corporate order, a payment error such as a double charge, a "
                         "privacy or security incident, or a problem the manual does not cover?"),
        "criteria": {
            "true": "A situation that needs a human's investigation or discretion beyond standard processes.",
            "false": ("A request the standard processes cover (order lookup, delivery, cancel, return, exchange, "
                      "address change, FAQ)."),
        },
    },
}

# 기준별 임계값(docs/decisions/jev_handoff.md 4절에서 고름). 오탐(불필요한 인계) 비용이 큰 기준일수록 높게.
THRESHOLDS: dict[str, float] = {
    "health": 0.7,
    "human_request": 0.8,
    "abuse_anger": 0.8,
    "repeated_failure": 0.8,
    "legal_dispute": 0.7,
    "out_of_policy": 0.8,
}

REASON_KO = {
    "health": "제품 섭취·사용 후 몸 불편 또는 의학적 판단이 필요한 문의",
    "human_request": "고객의 상담원 연결 요청",
    "abuse_anger": "강한 불만·격앙",
    "repeated_failure": "반복된 이해 실패",
    "legal_dispute": "법적 문제·분쟁 제기",
    "out_of_policy": "매뉴얼 밖 예외 처리 요구",
}


def build_state(utterance: str, history: list[dict] | None = None, failures: int = 0,
                caller_known: bool = False) -> dict:
    return {
        "situation": SITUATION,
        "recent_conversation": [{"role": m.get("role"), "text": m.get("text", m.get("content"))}
                                for m in (history or [])
                                if isinstance(m, dict) and m.get("role") not in ("tool_call", "tool_result")][-6:],
        "agent_failed_to_understand_count": int(failures),
        "caller_already_identified": bool(caller_known),
        "customer_utterance": (utterance or "").strip(),
    }


def build_body(utterance: str, history: list[dict] | None = None, failures: int = 0, caller_known: bool = False,
               model: str = MODEL) -> dict:
    return {"model": model, "state": build_state(utterance, history, failures, caller_known), "questions": CRITERIA}


def decide(probs: dict[str, float], thresholds: dict[str, float] | None = None) -> tuple[bool, list[str]]:
    """하나라도 임계값 이상이면 인계. 근거는 CRITERIA 순서. 빠진 답은 0."""
    th = THRESHOLDS if thresholds is None else thresholds
    why = [k for k in CRITERIA if float(probs.get(k) or 0.0) >= th[k]]
    return bool(why), why


@dataclass(frozen=True)
class HandoffJudgment:
    probs: dict[str, float]
    handoff: bool
    reasons: list[str] = field(default_factory=list)
    ms: float = 0.0


def parse(resp: Any, thresholds: dict[str, float] | None = None, ms: float = 0.0) -> HandoffJudgment | None:
    ans = resp.get("answers") if isinstance(resp, dict) else (getattr(resp, "nouls", None) or getattr(resp, "answers", None))
    ans = ans or {}
    probs = {k: _noul(ans.get(k)) for k in CRITERIA}
    if all(v is None for v in probs.values()):
        return None
    probs = {k: (0.0 if v is None else v) for k, v in probs.items()}
    ok, why = decide(probs, thresholds)
    return HandoffJudgment(probs, ok, why, ms)


class JevHandoff:
    def __init__(self, client: SystemOneClient | None = None, *, model: str = MODEL, timeout_s: float = 1.0,
                 thresholds: dict[str, float] | None = None):
        self._client = client
        self.model = model
        self.timeout_s = timeout_s
        self.thresholds = dict(THRESHOLDS if thresholds is None else thresholds)

    def _get(self) -> SystemOneClient:
        if self._client is None:
            self._client = make_client(self.timeout_s)
        return self._client

    async def judge(self, utterance: str, history: list[dict] | None = None, failures: int = 0,
                    caller_known: bool = False) -> HandoffJudgment | None:
        if not (utterance or "").strip():
            return None
        t0 = time.perf_counter()
        try:
            resp = await asyncio.wait_for(
                self._get().system_one(state=build_state(utterance, history, failures, caller_known),
                                       questions=CRITERIA, model=self.model),
                self.timeout_s)
        except (Exception, asyncio.TimeoutError) as e:  # noqa: BLE001
            log.info("jev handoff failed: %r", e)
            return None
        return parse(resp, self.thresholds, (time.perf_counter() - t0) * 1000)


def in_business_hours(now: str) -> bool | None:
    """상담원 연결 가능 시각(매뉴얼: 평일 09:00~18:00, 점심 12:00~13:00 제외, 토·일 휴무). now = "YYYY-MM-DD HH:MM".
    해석할 수 없으면 None. 공휴일은 모른다(대회 날짜 2026-11-20 금·21 토에는 공휴일 없음)."""
    from datetime import datetime
    try:
        t = datetime.strptime((now or "").strip()[:16], "%Y-%m-%d %H:%M")
    except ValueError:
        return None
    if t.weekday() >= 5:
        return False
    m = t.hour * 60 + t.minute
    return 9 * 60 <= m < 18 * 60 and not (12 * 60 <= m < 13 * 60)


def route(j: HandoffJudgment | None, now: str = "") -> str | None:
    """인계 판정 → 행동. None(그대로 LLM) | "transfer"(영업시간 중) | "callback"(영업시간 외).
    공식 함수에는 인계 함수가 없어 둘 다 말로 안내한다(AG-T38). 영업시간 값은 AG-T39에서 브랜드 정보로."""
    if j is None or not j.handoff:
        return None
    return "callback" if in_business_hours(now) is False else "transfer"


def handoff_enabled() -> bool:
    return os.environ.get(ENV_SWITCH, "").strip() == "1"


async def maybe_judge(j: JevHandoff | None, utterance: str, history: list[dict] | None = None, failures: int = 0,
                      caller_known: bool = False) -> HandoffJudgment | None:
    """스위치가 꺼져 있으면 부르지 않는다(None = 기존 LLM 판단 그대로)."""
    if j is None or not handoff_enabled():
        return None
    return await j.judge(utterance, history, failures, caller_known)


def transfer_reason(j: HandoffJudgment) -> str:
    return ", ".join(REASON_KO[k] for k in j.reasons) or "상담원 연결 필요"


def system_hint(j: HandoffJudgment | None, now: str = "") -> str:
    """LLM 시스템 프롬프트 끝에 붙일 지시. 인계가 아니면 빈 문자열."""
    r = route(j, now)
    if r is None:
        return ""
    head = (f"\n\n[인계 판정] 이번 고객 발화는 상담원 인계 대상입니다(사유: {transfer_reason(j)}). "
            "첫 마디로 이미 사과·공감했으면 다시 사과하지 말고 공감 문장은 생략하세요. ")
    tail = (" 몸이 불편하다는 내용이면 '증상이 심하시면 섭취를 멈추시고 가까운 병원 진료를 받아 보세요.' 한 문장을 더하세요."
            if "health" in j.reasons else "")
    no_fn = "상담원 연결은 함수가 아니라 말로 안내하므로 함수를 부르지 마세요. "
    if r == "callback":
        return head + no_fn + ("지금은 상담원 연결 시간이 아니므로 공감 한 문장 뒤 브랜드 정보의 전화 문의 방법(고객센터 번호와 "
                               "영업 시간)을 안내하세요." + tail)
    return head + no_fn + ("다른 안내나 조회 없이 공감 한 문장 뒤 상담원에게 넘겨 도와드리겠다고 말하고, 브랜드 정보의 "
                           "전화 문의 방법을 안내하세요." + tail)
