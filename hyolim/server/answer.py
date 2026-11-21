"""답을 어떻게 만들지 정한다 — 서버(app.py)와 글 평가(eval_text.py)가 같은 함수를 쓴다.

순서
  1. 위험도 판단(fncall.risk.assess)
     HIGH 이고 이번 턴에 접수가 없었으면 → 정해 둔 안전 문장 (LLM 자유 생성 금지)
  2. 템플릿이 있으면 → 템플릿 (LLM 없음, 가장 빠름)
  3. 그 밖 → LLM. MEDIUM 이면 제약(힌트)을 붙이고, 나오는 문장마다 guard_sentence 로 검사
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator

from fncall.planner import COMMIT_FUNCS
from fncall.risk import Risk, assess, grounding_text, guard_sentence

from . import llm


@dataclass
class Plan:
    mode: str                      # "safe" | "template" | "llm"
    text: str | None = None        # safe / template 일 때 말할 문장
    messages: list | None = None   # llm 일 때 프롬프트
    risk: Risk = field(default_factory=Risk)
    grounding: str = ""
    guarded: list[str] = field(default_factory=list)   # 문장 검사로 바뀐 이유들 (로그용)


def plan_answer(session: dict, utt: str, out) -> Plan:
    committed = any(c.name in COMMIT_FUNCS for c, _ in out.calls)
    rk = assess(session, utt, out.calls, committed)
    if rk.level == "high" and not committed and rk.safe_reply:
        return Plan("safe", text=rk.safe_reply, risk=rk)
    if out.calls and out.speech:                     # 템플릿은 조회 결과로만 만든 문장이라 근거가 확실하다
        return Plan("template", text=out.speech, risk=rk)
    msgs = llm.build_messages(session, utt, out.calls, out.commit_reason, extra_hints=rk.hints)
    return Plan("llm", messages=msgs, risk=rk, grounding=grounding_text(session, out.calls, utt))


async def guarded(plan: Plan, chunks: AsyncIterator[str], ack_played: bool) -> AsyncIterator[str]:
    """LLM 문장을 하나씩 검사해서 내보낸다. 같은 안전 문장이 연달아 나오면 한 번만."""
    last = None
    async for s in llm.sentences(chunks, ack_played=ack_played):
        s2, why = guard_sentence(s, plan.grounding)
        if why:
            plan.guarded.append(f"{why}: {s[:40]}")
        if s2 != last:
            yield s2
        last = s2
