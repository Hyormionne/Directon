"""LLM(GPT-5.4 mini) 스트리밍 답 + 프롬프트.

함수 호출은 fncall 이 코드로 처리하므로 LLM 에는 도구를 주지 않는다. LLM 은 '무슨 말을 할지'만 정한다.
→ 도구 정의가 빠져 프롬프트가 짧아지고, 함수 결과 뒤 LLM 왕복이 1번으로 고정된다.

프롬프트 앞부분(말투·시나리오·브랜드 정보)은 모든 문항에서 똑같이 둬서 프롬프트 캐시가 걸리게 한다.
문항마다 바뀌는 대화·함수 결과·손님 말은 맨 뒤에 둔다.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import AsyncIterator

import httpx

from . import config as C

BRAND = os.path.join(C.ROOT, "data", "brand")

RULES = """너는 건강기능식품 쇼핑몰 '하루온' 고객센터의 전화 상담원이다. 지금은 전화 통화 중이다.
- 답은 1~2문장, 소리 내어 읽기 좋게 짧게. 링크·이모티콘·기호·목록을 쓰지 않는다. 숫자는 말로 읽기 쉽게 쓴다.
- 이미 "네, 고객님" 같은 짧은 맞장구를 먼저 말했다. 답을 "네"나 "고객님"으로 시작하지 않는다.
- 손님이 물은 것에 먼저 답한다. 정책 질문(환불 가능 여부·비용·기간 등)은 브랜드 정보로 바로 안내하고, 그다음에 필요한 정보를 묻는다.
- 시나리오·브랜드 정보·함수 결과에 있는 사실만 말한다. 없는 정책을 지어내지 않는다. 모르면 상담원 연결을 안내한다.
- 손님 말이 짧거나 무슨 문의인지 분명하지 않으면(번호만 말함 등) 짐작해서 처리하지 말고 어떤 문의인지 여쭙는다.
- 함수 결과가 주어지면 그 결과대로 안내한다. 결과가 없거나 '조회 안 됨'이면 그대로 알리고 번호를 다시 여쭙는다.
- 주문 조회·반품 접수 등에 필요한 주문번호나 전화번호가 없으면 먼저 여쭙는다. 이미 대화에 있으면 다시 묻지 않는다.
- 반품·취소·교환·환불 요청인데 아직 주문을 조회하지 않았으면, 사유보다 먼저 주문번호나 주문하신 분 연락처를 여쭙는다.
- 주문을 조회한 뒤에는 사유를 듣고, 필요하면 전체인지 일부인지, 회수지 주소가 주문 주소와 같은지 확인한 뒤 접수한다.
- 입금·배송·환불 상태를 물었는데 아직 주문을 조회하지 않았으면, 먼저 주문번호나 주문하신 분 연락처를 여쭙는다.
- 주문을 조회했는데 안내된 처리 시간(입금 연동 20분, 환불 3~4영업일 등)보다 오래 지연됐다고 하면,
  공감한 뒤 확인에 필요한 정보(날짜·입금자명·금액 등)를 여쭙는다.
- 쿠팡·네이버 등 다른 판매처에서 산 주문은 자사몰 주문 조회로 확인되지 않고, 환불보장제도는 자사몰 구매 건에만 적용된다.
  다른 판매처 구매라고 하면 자사몰처럼 처리하겠다고 하지 말고, 구매처와 문의 내용을 확인한 뒤 상담원 확인으로 넘긴다.
- 지병·복용 중인 약·임신이 있는 상태에서 먹어도 되는지 묻는 질문은 상담원에게 넘기지 말고, 브랜드 정보
  [지병 및 복용 중인 약이 있는 경우 섭취 안내(공통)]대로 "안전을 위해 전문의와 상담 후 섭취를 권장"한다고 안내한다.
- 이미 먹고 이상 증상(두드러기·통증 등)이 생겼다고 하면, 섭취를 멈추라고 안내한 뒤 단정하지 말고 상담원에게 넘긴다.
- 질병이 낫는지·치료되는지 물으면, 의약품이 아닌 건강 보조 식품이라 치료를 단정할 수 없고 효과는 개인차가 있다고 안내한다.
  하루온 제품이나 영양제를 먹고 있다고 말하는 것만으로는 넘기지 않는다. 하루온 제품끼리 함께 먹는 것은 브랜드 정보에 따라 안내한다.
- 상품 검색 결과가 주어지면 그 안의 제품명·구성·섭취 방법 그대로 안내한다. 구성품 이름도 결과에 적힌 그대로 말한다.
- 검색 결과·브랜드 정보에 없는 수치(한 박스 정 수, 며칠분, 할인 적용 금액 등)는 계산하거나 짐작해 말하지 않는다.
  확인되지 않으면 확인이 어렵다고 말하고 상담원 확인을 안내한다.
- 어떤 제품인지 대화에서 알 수 없으면(예: "먹는 시간은 언제가 좋아요?") 섭취법을 일반론으로 말하지 말고 제품명부터 여쭙는다.
  검색 결과의 제품명이 손님이 말한 제품과 다르면 그 제품이라고 단정하지 말고, 정확한 제품명을 여쭙는다.
- 손님이 구매하겠다고 하면 제품 설명을 늘어놓지 말고 구매 방법(온라인 구매, ARS 전화 주문)을 안내하고,
  직접 주문이 어려우면 성함과 연락처를 남기면 상담원이 전화드린다고 안내한다.
- 손님이 상담원 연결을 원하면 고객센터 번호와 운영 시간을 안내하고, 지금 대신 확인해 드릴 수 있는 것(주문 조회 등)도 짧게 제안한다.
- 효과를 보장하거나 질병 치료처럼 말하지 않는다.
- 손님이 상담원과 직접 통화를 원하면 고객센터 번호(02-000-0000)와 운영 시간(평일 10시~17시, 점심 12시~13시)을 안내한다.
- 같은 요청이 반복되면 같은 문장을 되풀이하지 말고, 할 수 있는 대안을 짧게 제시한다."""

HINTS = {
    "need_phone": "전화 요청을 등록하려면 연락처가 필요하다. 성함과 연락처를 여쭤라.",
    "need_order": "주문번호 또는 주문자 전화번호가 필요하다. 여쭤라.",
    "lookup_first": "주문 조회가 되지 않았다. 주문번호나 주문하신 분 연락처를 여쭤라.",
    "no_consent": "아직 접수하지 않았다. 필요한 확인(사유·전체/일부·회수지)을 하나만 물어라.",
    "customer_declined": "손님이 접수를 원하지 않는다. 접수하지 말고 다른 도움이 필요한지 여쭤라.",
    "multi_order": "조회된 주문이 여러 건이다. 아직 접수하지 않았다. 제품명과 주문일로 어느 주문인지 짧게 여쭤라(두세 건만 말한다).",
    "unclear": "손님이 번호나 짧은 말만 해서 무슨 문의인지 분명하지 않다. 번호를 안내하거나 짐작하지 말고 어떤 문의인지 여쭤라.",
    "unpaid_stale": "조회 결과 아직 입금 확인이 안 된 주문이 있는데, 주문한 지 시간이 꽤 지났다. '입금 연동 20분' 안내로 끝내지 마라. "
                    "공감한 뒤, 확인을 위해 입금하신 날짜·입금자명·입금 금액을 여쭤라. 상태가 '입금전취소'인 주문은 조회 결과대로 취소된 상태라고 알린다.",
    "ambiguous_number": "들은 번호가 정확하지 않아 비슷한 번호가 여러 개 조회됐다. 다른 사람 정보일 수 있으니 조회 결과를 말하지 말고, 번호를 천천히 한 번 더 불러 달라고 부탁하라.",
    "caller_not_found": "손님은 번호를 말하지 않았고, 발신번호로 조회했지만 기록이 없었다. '말씀하신 번호'라고 하지 말고, 확인을 위해 주문하신 분 성함과 연락처(또는 주문번호)를 여쭤라.",
}


@lru_cache(maxsize=1)
def system_prompt() -> str:
    style = json.load(open(os.path.join(BRAND, "style.json"), encoding="utf-8"))
    scen = open(os.path.join(BRAND, "scenarios.md"), encoding="utf-8").read()
    info = json.load(open(os.path.join(BRAND, "information.json"), encoding="utf-8"))
    strip = lambda s: re.sub(r"<[^>]+>|https?://\S+", "", s).strip()
    info_txt = "\n".join(f"{x['tag']}: {strip(x['content'])}" for x in info)
    return f"{RULES}\n\n# 말투\n{style.get('style', '')}\n\n# 상담 시나리오\n{scen}\n\n# 브랜드 정보\n{info_txt}"


def _clip(s: str, n: int = 300) -> str:
    s = re.sub(r"<[^>]+>|https?://\S+", "", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s[:n]


def _fmt_history(hist: list[dict], max_items: int = 14) -> str:
    lines = []
    for m in hist[-max_items:]:
        r = m.get("role")
        if r == "customer":
            lines.append(f"손님: {_clip(m.get('text'))}")
        elif r == "agent":
            lines.append(f"상담원: {_clip(m.get('text'))}")
        elif r == "tool_call":
            lines.append(f"[함수 호출] {m.get('name')} {json.dumps(m.get('arguments'), ensure_ascii=False)}")
        elif r == "tool_result":
            lines.append(f"[함수 결과] {m.get('name')} {_clip(json.dumps(m.get('result'), ensure_ascii=False), 900)}")
    return "\n".join(lines)


def build_messages(session: dict, utterance: str, tool_calls=(), commit_reason: str = "", extra_hints=()) -> list[dict]:
    parts = [f"현재 시각: {session.get('now')}"]
    caller = session.get("caller")
    if caller:
        parts.append(f"발신자 정보(참고용, 다른 번호로 주문했을 수 있음): {caller.get('name')} {caller.get('phone')}")
    parts.append("# 지금까지의 대화\n" + (_fmt_history(session.get("history") or []) or "(없음)"))
    if tool_calls:
        parts.append("# 이번에 실행한 함수와 결과\n" + "\n".join(
            f"{c.name} {json.dumps(c.arguments, ensure_ascii=False)} → {_clip(json.dumps(r, ensure_ascii=False), 1200)}"
            for c, r in tool_calls))
    notes = ([HINTS[commit_reason]] if commit_reason in HINTS else []) + list(extra_hints or [])
    if notes:
        parts.append("# 참고 (반드시 지킬 것)\n" + "\n".join(f"- {n}" for n in notes))
    parts.append(f"# 손님의 이번 말\n{utterance}\n\n상담원으로서 다음에 할 말만 써라.")
    return [{"role": "system", "content": system_prompt()}, {"role": "user", "content": "\n\n".join(parts)}]


async def stream_answer(http: httpx.AsyncClient, messages: list[dict]) -> AsyncIterator[str]:
    """텍스트 조각을 흘려보낸다."""
    if C.FAKE:
        yield "확인해 드릴게요. 더 궁금하신 점 있으실까요?"
        return
    body = {"model": C.LLM_MODEL, "messages": messages, "stream": True, "max_completion_tokens": 300}
    if C.LLM_REASONING:
        body["reasoning_effort"] = C.LLM_REASONING
    for attempt in range(2):
        async with http.stream("POST", C.LLM_URL, json=body, timeout=15,
                               headers={"Authorization": f"Bearer {C.OPENAI_API_KEY}"}) as r:
            if r.status_code == 400 and attempt == 0 and "reasoning_effort" in body:
                await r.aread()
                body.pop("reasoning_effort")                 # 모델이 이 값을 지원하지 않으면 빼고 다시
                continue
            if r.status_code != 200:
                raise RuntimeError(f"LLM {r.status_code}: {(await r.aread())[:300]!r}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    return
                try:
                    delta = json.loads(d)["choices"][0]["delta"].get("content")
                except (ValueError, KeyError, IndexError):
                    continue
                if delta:
                    yield delta
            return


ACK_RE = re.compile(r"^\s*(?:(?:네|예|넵)(?:\s*[,.!~]\s*|\s+|$))?(?:고객님(?:\s*[,.!]\s*|\s+|$))?")


def strip_ack(s: str) -> str:
    """채움말('네, 고객님.')을 이미 말했으니 답 첫머리의 '네, 고객님,'을 뗀다."""
    while True:                                            # '네, 고객님. 네, …' 처럼 겹친 경우까지
        t = ACK_RE.sub("", s, count=1)
        if t == s:
            return s
        s = t


async def sentences(chunks: AsyncIterator[str], first_min: int = 8, ack_played: bool = False) -> AsyncIterator[str]:
    """스트리밍 텍스트를 문장 단위로 끊는다. 첫 조각은 짧아도 빨리 보낸다(TTS 첫 소리를 당기려고)."""
    buf, first, head = "", True, ack_played
    async for c in chunks:
        buf += c
        if head:
            if len(buf) < 8 and not re.search(r"[^\s네예넵고객님,.!]", buf):
                continue                                   # '네, 고객님' 이 다 올 때까지 기다린다
            buf, head = strip_ack(buf), False
        while True:
            m = re.search(r"[.?!](\s|$)|[,，]\s", buf) if first else re.search(r"[.?!](\s|$)", buf)
            if not m or (first and m.end() < first_min):
                break
            cut = m.end()
            out, buf = buf[:cut].strip(), buf[cut:]
            if out:
                yield out
                first = False
    if head:
        buf = strip_ack(buf)
    if buf.strip():
        yield buf.strip()
