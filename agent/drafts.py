"""답 후보(draft_answers): 함수 결과 전에 LLM이 결과별 답을 미리 써 두고, 결과가 오면 코드가 골라 바로 말한다.

LLM은 같은 응답에서 진짜 함수 호출을 먼저 내고, 이어서 내부 함수 draft_answers를 부른다(측정기·Mock으로 보내지 않는다).
  answers: [{"when": "배송상태=배송완료", "text": "{배송완료일자}에 배송 완료됐어요.", "rank": 1}, ...]
- when: 조건 목록(쉼표로 구분, 모두 만족). "status=not_found"는 결과 status, 그 밖의 "키=값"은 결과 data 안 행들의 값.
  빈 값이면 무엇이든 맞는다(가장 낮은 우선순위).
- text: 말할 문장. {필드명} 빈칸은 결과 값으로 채운다(날짜는 "10월 1일", 상품명은 이모지·특가 표시 제거).
- rank: 결과가 나올 가능성 순서(1이 가장 높음). 미리 합성할 후보를 고를 때 쓴다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

DRAFT_NAME = "draft_answers"
DRAFT_TOOL = {"type": "function", "function": {
    "name": DRAFT_NAME,
    "description": ("내부용. 조회·접수 함수를 호출한 같은 응답에서, 그 함수 결과별로 고객에게 할 답을 미리 써 둔다. "
                    "결과가 오면 시스템이 맞는 답을 골라 바로 말한다. 고객 기록 함수 없이 단독으로 부르지 않는다."),
    "parameters": {"type": "object", "properties": {"answers": {"type": "array", "items": {
        "type": "object", "properties": {
            "when": {"type": "string", "description": "결과 조건. 예: 'status=not_found', '배송상태=배송중', "
                                                      "'status=ok'. 여러 조건은 쉼표로."},
            "text": {"type": "string", "description": "그 결과일 때 할 말 1~2문장. 결과 값 자리는 {필드명}."},
            "rank": {"type": "integer", "description": "그 결과가 나올 가능성 순서(1이 가장 높음)"}},
        "required": ["when", "text"]}}}, "required": ["answers"]}}}

_SLOT = re.compile(r"\{([^{}]+)\}")
_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:[ T]\d{2}:\d{2}(?::\d{2})?)?$")
_BADGE = re.compile(r"🔒[^🔒]*🔒")
_NON_SPEECH = re.compile(r"[^\w\s가-힣.,·()~%-]")


@dataclass
class Draft:
    when: list[tuple[str, str]]
    text: str
    rank: int

    @property
    def has_slots(self) -> bool:
        return bool(_SLOT.search(self.text))


def parse(args: Any) -> list[Draft]:
    out: list[Draft] = []
    items = (args or {}).get("answers") if isinstance(args, dict) else None
    for i, it in enumerate(items or []):
        if not isinstance(it, dict) or not str(it.get("text") or "").strip():
            continue
        conds = []
        for part in str(it.get("when") or "").split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                if k.strip():
                    conds.append((k.strip(), v.strip()))
        try:
            rank = int(it.get("rank") or i + 1)
        except (TypeError, ValueError):
            rank = i + 1
        out.append(Draft(conds, str(it["text"]).strip(), rank))
    return out


def presynth_targets(drafts: list[Draft], n: int) -> list[Draft]:
    """미리 합성할 후보: 빈칸 없는 것 중 rank 순 상위 n개."""
    return sorted((d for d in drafts if not d.has_slots), key=lambda d: d.rank)[:max(0, n)]


def _rows(result: dict) -> list[dict]:
    data = result.get("data")
    rows: list[dict] = []
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                rows += [x for x in v if isinstance(x, dict)]
        if not rows:
            rows = [data]
    elif isinstance(data, list):
        rows = [x for x in data if isinstance(x, dict)]
    return rows


def _field(result: dict, key: str) -> str | None:
    """결과에서 key 값. 행이 여럿이면 모두 같을 때만(다르면 None — 어느 상품 이야기인지 모른다)."""
    if key in ("status", "message"):
        v = result.get(key)
        return None if v is None else str(v).strip()
    vals = {str(r[key]).strip() for r in _rows(result) if key in r and r[key] is not None}
    return vals.pop() if len(vals) == 1 else None


def _matches(d: Draft, result: dict) -> bool:
    if not d.when:                                   # 조건 없는 후보는 정상 결과(ok)에만
        return result.get("status") == "ok"
    return all(_field(result, k) == v for k, v in d.when)


def speakable(key: str, value: str) -> str:
    m = _DATE.match(value)
    if m:
        return f"{int(m.group(2))}월 {int(m.group(3))}일"
    v = _BADGE.sub("", value)
    if "제품명" in key or "상품명" in key:
        v = re.sub(r"^옵션 선택=", "", v)
    return re.sub(r"\s+", " ", _NON_SPEECH.sub("", v)).strip()


def fill(text: str, result: dict) -> str | None:
    """빈칸을 결과 값으로 채운다. 값이 없거나 비어 있으면 None(이 후보는 못 쓴다)."""
    def rep(m: re.Match) -> str:
        v = _field(result, m.group(1).strip())
        if not v:
            raise KeyError(m.group(1))
        return speakable(m.group(1), v)
    try:
        return _SLOT.sub(rep, text)
    except KeyError:
        return None


def choose(drafts: list[Draft], result: Any) -> tuple[Draft, str] | None:
    """결과에 맞는 후보(조건이 구체적인 것 우선, 같으면 rank 순)와 채운 문장. 없으면 None."""
    if not isinstance(result, dict):
        return None
    for d in sorted(drafts, key=lambda d: (-len(d.when), d.rank)):
        if _matches(d, result):
            text = fill(d.text, result)
            if text:
                return d, text
    return None
