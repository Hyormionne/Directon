"""함수 호출 가드 (AG-T33). LLM이 낸 함수 호출을 실행하기 전에 검사·정리한다.

순수 함수 모듈이다(입출력·전역 상태 없음). 환경변수는 DIRECTON_GUARD_GROUNDING 하나만 check 안에서 읽는다.

check 규칙(이 순서):
1. 정의에 없는 함수(예: transfer_to_agent) → 막음("undefined").
2. 번호 인수 정리: 전화번호는 숫자만, 주문번호는 format_order_id, 상세주문번호 17자리는 YYYYMMDD-NNNNNNN-NN.
   값이 null·빈 문자열·숫자 없음이면 키를 뺀다.
3. 아는 번호로 맞춤: 믿는 출처(이전 대화, 함수 결과, 발신자 전화번호)에 정확히 없고, 같은 종류·같은 길이에
   편집 거리 2 이하인 번호가 하나뿐이면 그 번호로 바꾼다(STT가 숫자를 잘못 들은 경우).
4. 필수 인수 빠짐 → 막음("missing"). get_order_info는 주문번호·전화번호 중 하나가 있어야 한다.
5. 접수 함수의 번호 근거: 주문번호·상세주문번호·전화번호가 아는 번호가 아니면 막음("ungrounded").
   조회·검색은 막지 않는다. 필수가 아닌 번호 인수(cancel_order의 customer_phone_number)는 막지 않고 뺀다.
6. 한 턴에 접수 함수는 하나: ctx.called에 접수 함수가 이미 있으면 막음("second_action").

판단한 것:
- 이번 고객 말(user_text)은 STT 결과라 틀릴 수 있다. 그래서 3번에서 "정확히 아는 번호"는 믿는 출처만 본다.
  고객 말에 있는 번호라도 믿는 출처의 번호 하나와 거의 같으면 그쪽으로 맞춘다. 5번 근거에는 고객 말도 넣는다.
- 상세주문번호는 맞추지 않는다(-01, -02처럼 한 자리만 다른 것이 서로 다른 상품이다).
- 번호가 붙어 들리거나 끊어 들려도 근거를 찾도록, 글마다 숫자만 이어 붙인 문자열에 인수가 들어 있으면 근거로 본다.
- ctx.tools가 비어 있으면 대회 기본 8개 함수와 그 필수 인수로 본다(정의를 못 받았다고 전부 막지 않는다).

agent.responder는 함수 안에서 늦게 불러온다(responder가 이 모듈을 불러올 때 순환을 피한다).
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from agent.normalize import spoken_digits

log = logging.getLogger(__name__)

__all__ = [
    "ORDER_KEYS", "PHONE_KEYS", "LOOKUP_FUNCS", "SEARCH_FUNCS", "ACTION_FUNCS",
    "GuardContext", "Verdict", "known_numbers", "format_order_id", "check",
]

ORDER_KEYS = ("order_id", "order_number")
PHONE_KEYS = ("buyer_cellphone", "customer_phone_number", "phone_number")
ITEMS_KEY = "return_items"
ITEM_CODE_KEY = "order_item_code"

LOOKUP_FUNCS = frozenset({"get_order_info", "get_return_info"})
SEARCH_FUNCS = frozenset({"search_products"})
ACTION_FUNCS = frozenset({"cancel_order", "return_order", "return_order_item", "exchange_order",
                          "request_callback"})

GROUNDING_ENV = "DIRECTON_GUARD_GROUNDING"
SNAP_MAX_DISTANCE = 2           # 이 거리 이하인 아는 번호가 하나뿐일 때만 맞춘다
SNAP_HEARD_TO_TRUSTED = True    # 고객 말(STT)에 있는 번호도 믿는 출처의 번호로 맞춘다
MIN_DIGITS = 7                  # 이보다 짧은 숫자 묶음은 번호로 보지 않는다
ORDER_DIGITS = 15               # YYYYMMDD + NNNNNNN
ITEM_DIGITS = 17                # 주문번호 + NN
_MAX_DEPTH = 8

# ctx.tools가 비었을 때만 쓰는 기본 필수 인수(대회 함수 정의와 같다)
_DEFAULT_REQUIRED: dict[str, tuple[str, ...]] = {
    "get_order_info": (),
    "get_return_info": ("customer_phone_number",),
    "search_products": ("query",),
    "cancel_order": ("order_id", "reason"),
    "return_order": ("order_number", "reason"),
    "return_order_item": ("return_items", "reason"),
    "exchange_order": ("order_number", "reason"),
    "request_callback": ("customer_name", "phone_number"),
}
_DEFAULT_ITEM_REQUIRED = (ITEM_CODE_KEY, "quantity")

_LABELS = {
    "order_id": "주문번호", "order_number": "주문번호",
    "buyer_cellphone": "전화번호", "customer_phone_number": "전화번호", "phone_number": "전화번호",
    "customer_name": "이름", "reason": "사유", "query": "검색어",
    ITEMS_KEY: "반품할 상품", ITEM_CODE_KEY: "상세주문번호", "quantity": "수량",
}
_REASON_LABELS = {"cancel_order": "취소 사유", "return_order": "반품 사유",
                  "return_order_item": "반품 사유", "exchange_order": "교환 사유"}
_NOT_DONE = "아직 아무것도 접수·처리되지 않았으니 처리됐다고 말하지 마세요."


@dataclass
class GuardContext:
    tools: list[dict]                                   # 함수 정의 [{name, description, parameters}] (OpenAI 래핑도 받는다)
    user_text: str = ""                                 # 이번 고객 말(STT 결과)
    history: list[dict] = field(default_factory=list)   # session.start history + 이번 연결의 앞선 말
    caller: dict | None = None                          # {"name","phone","email"} 또는 None
    results: list[dict] = field(default_factory=list)   # 이번 턴에 이미 받은 함수 결과(미리 조회한 주문 포함)
    called: list[str] = field(default_factory=list)     # 이번 턴에 이미 실행한 함수 이름


@dataclass
class Verdict:
    args: dict                      # 정리된 인수(통과 시 이걸로 호출)
    blocked: dict | None = None     # 막았으면 LLM에 돌려줄 결과 {"status": "blocked", "message": "..."}
    reason: str = ""                # 로그용 짧은 코드: "", "undefined", "missing", "ungrounded", "second_action"


# ── 함수 정의 ─────────────────────────────────────────
def _tool_defs(tools: Any) -> dict[str, dict]:
    """함수 정의 목록 → {이름: parameters}. OpenAI 래핑({"type":"function","function":{...}})도 푼다."""
    out: dict[str, dict] = {}
    for t in tools if isinstance(tools, (list, tuple)) else []:
        if not isinstance(t, dict):
            continue
        f = t["function"] if isinstance(t.get("function"), dict) else t
        name = f.get("name")
        if isinstance(name, str) and name:
            params = f.get("parameters")
            out[name] = params if isinstance(params, dict) else {}
    return out


def _required(name: str, params: dict | None) -> list[str]:
    if params is None:
        return list(_DEFAULT_REQUIRED.get(name, ()))
    req = params.get("required")
    return [k for k in req if isinstance(k, str)] if isinstance(req, (list, tuple)) else []


def _item_required(params: dict | None) -> list[str]:
    """return_items 원소의 필수 인수."""
    if params is None:
        return list(_DEFAULT_ITEM_REQUIRED)
    props = params.get("properties")
    spec = props.get(ITEMS_KEY) if isinstance(props, dict) else None
    items = spec.get("items") if isinstance(spec, dict) else None
    req = items.get("required") if isinstance(items, dict) else None
    return [k for k in req if isinstance(k, str)] if isinstance(req, (list, tuple)) else [ITEM_CODE_KEY]


# ── 숫자 뽑기 ─────────────────────────────────────────
_HANGUL = re.compile(r"[가-힣]")
_NON_DIGIT = re.compile(r"\D")
_DIGIT_CH = "0-9공영빵일이삼사오육륙칠팔구"
# 숫자 바로 뒤에 붙은 어미. agent.normalize·agent.responder는 "구입니다"처럼 받침 없는 숫자 뒤의 "입니다"를
# 떼지 못해 마지막 묶음을 통째로 놓친다. 여기서 미리 뗀다("이에요"·"이요"는 숫자 2와 헷갈려 기존 규칙에 맡긴다).
_ENDING = re.compile(rf"(?<=[{_DIGIT_CH}])(입니다만|입니다요|입니다|인데요|인데|거든요|예요|라고요|라고|랍니다)(?![가-힣])")
# 하이픈을 읽은 말. responder.utterance_numbers는 "다시"를 정정으로 보고 앞 번호의 끝을 바꿔 버린다.
_DASH_WORD = re.compile(r"(?<![가-힣])(다시|대시|하이픈|빼기|마이너스)(?![가-힣])")
_SEP = r"[\s.,\-]{0,3}"
_ITEM_RE = re.compile(r"(?<!\d)\d{8}-\d{7}-\d{2}(?!\d)")
_ORDER_RE = re.compile(r"(?<!\d)\d{8}-\d{7}(?!\d)")
_PLAIN_ORDER_RE = re.compile(r"(?<!\d)\d{15,16}(?!\d)")
_PHONE_RE = re.compile(rf"(?<!\d)01[016-9]{_SEP}\d{{3,4}}{_SEP}\d{{4}}(?!\d)")
_PHONE_SHAPE = re.compile(r"0\d{8,10}")
_MERGED = (re.compile(r"(\d{15,16})(01[016-9]\d{7,8})"), re.compile(r"(01[016-9]\d{7,8})(\d{15,16})"))


def _digits(s: Any) -> str:
    return _NON_DIGIT.sub("", str(s or ""))


def _clean_spoken(text: str) -> str:
    return _ENDING.sub(" ", text or "")


def _spoken(text: str) -> str:
    """한국어로 읽은 숫자·아라비아 숫자 혼합 → 숫자열(없으면 "")."""
    try:
        return spoken_digits(_DASH_WORD.sub(" ", _clean_spoken(text))) or ""
    except Exception:  # noqa: BLE001
        return _digits(text)


def _utterance_numbers(text: str) -> list[str]:
    try:
        from agent.responder import utterance_numbers  # 순환 import를 피하려고 늦게 불러온다
        return utterance_numbers(text, MIN_DIGITS)
    except Exception as e:  # noqa: BLE001
        log.debug("utterance_numbers 실패: %r", e)
        return []


def _distance(a: str, b: str) -> int:
    from agent.responder import _edit_distance  # 순환 import를 피하려고 늦게 불러온다
    return _edit_distance(a, b)


def _kind_of(d: str) -> str:
    """자유로운 글에서 나온 숫자열의 종류: 17자리=상세주문번호, 0으로 시작하는 9~11자리=전화번호, 그 밖=주문번호."""
    if len(d) == ITEM_DIGITS:
        return "items"
    if _PHONE_SHAPE.fullmatch(d):
        return "phones"
    return "orders"


def _new_bucket() -> dict[str, list[str]]:
    return {"orders": [], "phones": [], "items": []}


def _add(bucket: dict[str, list[str]], kind: str, d: str) -> None:
    if len(d) < MIN_DIGITS:
        return
    if d not in bucket[kind]:
        bucket[kind].append(d)
    if kind == "items" and len(d) == ITEM_DIGITS and d[:ORDER_DIGITS] not in bucket["orders"]:
        bucket["orders"].append(d[:ORDER_DIGITS])      # 상세주문번호 앞 15자리는 주문번호


def _scan_strict(text: str, bucket: dict[str, list[str]]) -> None:
    """아라비아 숫자로 적힌 번호만 모양으로 찾는다(날짜·금액·송장번호는 넣지 않는다)."""
    for m in _ITEM_RE.finditer(text):
        _add(bucket, "items", _digits(m.group()))
    for m in _ORDER_RE.finditer(text):
        _add(bucket, "orders", _digits(m.group()))
    for m in _PLAIN_ORDER_RE.finditer(text):
        _add(bucket, "orders", m.group())
    for m in _PHONE_RE.finditer(text):
        _add(bucket, "phones", _digits(m.group()))


def _scan_text(text: Any, bucket: dict[str, list[str]], soups: list[str]) -> None:
    """고객·상담원 말에서 번호를 찾는다(한국어로 읽은 숫자 포함)."""
    if not isinstance(text, str) or not text.strip():
        return
    _scan_strict(text, bucket)
    cleaned = _clean_spoken(text)
    for variant in dict.fromkeys((text, cleaned, _DASH_WORD.sub(" ", cleaned))):
        for run in _utterance_numbers(variant):
            _add(bucket, _kind_of(run), run)
            for pat in _MERGED:                     # 주문번호와 전화번호가 붙어 한 묶음이 된 경우
                m = pat.fullmatch(run)
                if m:
                    for part in m.groups():
                        _add(bucket, _kind_of(part), part)
    for soup in (_digits(text), _spoken(text)):
        if len(soup) >= MIN_DIGITS and soup not in soups:
            soups.append(soup)


def _key_kind(key: str) -> str:
    k = key.lower()
    if k == ITEM_CODE_KEY or "item_code" in k or "상세주문" in key:
        return "items"
    if k in ORDER_KEYS or "주문번호" in key or ("order" in k and ("id" in k or "number" in k or k.endswith("_no"))):
        return "orders"
    if "phone" in k or any(w in key for w in ("전화", "연락처", "휴대폰", "핸드폰")):
        return "phones"
    return ""


def _scan_struct(obj: Any, bucket: dict[str, list[str]], key: str = "", depth: int = 0) -> None:
    """함수 인수·결과를 끝까지 내려가며 번호를 찾는다. 키 이름으로 종류를 정하고, 그 밖의 문자열은 모양으로만 본다."""
    if depth > _MAX_DEPTH or obj is None or isinstance(obj, bool):
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _scan_struct(v, bucket, str(k), depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _scan_struct(v, bucket, key, depth + 1)
    elif isinstance(obj, (str, int)):
        kind = _key_kind(key)
        if kind:
            d = _id_digits(obj, phone=kind == "phones")
            if len(d) <= ITEM_DIGITS:
                _add(bucket, kind, d)
        if isinstance(obj, str):
            _scan_strict(obj, bucket)


@dataclass
class _Known:
    trusted: dict[str, list[str]]       # 이전 대화, 함수 인수·결과, 발신자 전화번호
    heard: dict[str, list[str]]         # 이번 고객 말(STT)
    trusted_soups: list[str]            # 글마다 숫자만 이어 붙인 문자열
    heard_soups: list[str]

    def all(self, kind: str) -> list[str]:
        return list(dict.fromkeys([*self.trusted[kind], *self.heard[kind]]))


def _collect(ctx: GuardContext) -> _Known:
    kn = _Known(_new_bucket(), _new_bucket(), [], [])
    history = ctx.history if isinstance(ctx.history, (list, tuple)) else []
    for h in history:
        if isinstance(h, str):
            _scan_text(h, kn.trusted, kn.trusted_soups)
            continue
        if not isinstance(h, dict):
            continue
        for k in ("text", "content"):
            _scan_text(h.get(k), kn.trusted, kn.trusted_soups)
        for k in ("arguments", "result"):
            _scan_struct(h.get(k), kn.trusted)
    _scan_struct(ctx.results if isinstance(ctx.results, (list, tuple, dict)) else None, kn.trusted)
    if isinstance(ctx.caller, dict):
        _add(kn.trusted, "phones", _id_digits(ctx.caller.get("phone"), phone=True))
    _scan_text(ctx.user_text, kn.heard, kn.heard_soups)
    return kn


def known_numbers(ctx: GuardContext) -> dict:
    """모든 출처(고객 말, 이전 대화, 함수 결과, 발신자)에서 아는 번호. {"orders", "phones", "items"} 숫자 문자열."""
    kn = _collect(ctx)
    return {kind: kn.all(kind) for kind in ("orders", "phones", "items")}


def _in_soup(d: str, soups: list[str]) -> bool:
    return len(d) >= MIN_DIGITS and any(d in s for s in soups)


# ── 인수 정리 ─────────────────────────────────────────
def format_order_id(d: str) -> str:
    """15자리 → "YYYYMMDD-NNNNNNN", 그 밖은 숫자 그대로."""
    d = _digits(d)
    return f"{d[:8]}-{d[8:]}" if len(d) == ORDER_DIGITS else d


def _format_item_code(d: str) -> str:
    return f"{d[:8]}-{d[8:15]}-{d[15:]}"


def _id_digits(value: Any, phone: bool = False) -> str:
    """번호 인수 값 → 숫자열. 한국어로 읽은 숫자도 바꾼다. 번호로 볼 수 없으면 ""."""
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, float):
        if not value.is_integer():
            return ""
        value = int(value)
    if isinstance(value, int):
        d = str(abs(value))
        return "0" + d if phone and len(d) == 10 and d.startswith("10") else d   # 정수로 와서 빠진 앞 0
    if not isinstance(value, str):
        return ""
    d = _spoken(value) if _HANGUL.search(value) else _digits(value)
    if phone and d.startswith("82"):                # 국가번호
        if len(d) == 12:
            d = "0" + d[2:]
        elif len(d) == 13 and d[2] == "0":
            d = d[2:]
    return d


def _empty(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return not v.strip()
    if isinstance(v, (list, tuple, dict)):
        return len(v) == 0
    return False


def _copy_args(args: Any) -> dict:
    if not isinstance(args, dict):
        return {}
    try:
        return copy.deepcopy(args)
    except Exception:  # noqa: BLE001
        return dict(args)


def _snap(d: str, kind: str, kn: _Known) -> str:
    """아는 번호와 거의 같으면 그 번호로. 후보가 없거나 둘 이상이면 그대로 둔다."""
    if len(d) < MIN_DIGITS or d in kn.trusted[kind] or _in_soup(d, kn.trusted_soups):
        return d
    heard = d in kn.heard[kind] or _in_soup(d, kn.heard_soups)
    if heard and not SNAP_HEARD_TO_TRUSTED:
        return d
    pool = kn.trusted[kind] if heard else kn.all(kind)
    cands = [c for c in dict.fromkeys(pool)
             if c != d and len(c) == len(d) and _distance(d, c) <= SNAP_MAX_DISTANCE]
    return cands[0] if len(cands) == 1 else d


def _normalize_ids(out: dict, kn: _Known) -> None:
    """규칙 2·3: 번호 인수를 정리하고 아는 번호에 맞춘다(out을 바꾼다)."""
    for keys, kind in ((ORDER_KEYS, "orders"), (PHONE_KEYS, "phones")):
        for key in keys:
            if key not in out:
                continue
            d = _id_digits(out[key], phone=kind == "phones")
            if not d:
                del out[key]                        # null·빈 값·숫자 없는 값이 진짜 번호를 가리지 않게
                continue
            d = _snap(d, kind, kn)
            out[key] = format_order_id(d) if kind == "orders" else d
    if ITEMS_KEY not in out:
        return
    items = out[ITEMS_KEY]
    if isinstance(items, str):                      # JSON 문자열로 온 목록
        try:
            items = json.loads(items)
        except ValueError:
            return
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        return
    for it in items:
        if isinstance(it, dict) and isinstance(it.get(ITEM_CODE_KEY), (str, int)):
            d = _id_digits(it[ITEM_CODE_KEY])
            if len(d) == ITEM_DIGITS:
                it[ITEM_CODE_KEY] = _format_item_code(d)
            elif isinstance(it[ITEM_CODE_KEY], str):
                it[ITEM_CODE_KEY] = it[ITEM_CODE_KEY].strip()
    out[ITEMS_KEY] = items


# ── 형식에 안 맞는 번호는 믿는 출처의 번호로 ─────────
_VALID_PHONE = re.compile(r"01[016-9]\d{7,8}")
_VALID_ORDER = re.compile(r"20\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{7}|\d{16}")


def valid_phone(d: str) -> bool:
    return bool(_VALID_PHONE.fullmatch(d or ""))


def valid_order(d: str) -> bool:
    """YYYYMMDD + 7자리(15자리) 또는 16자리 숫자."""
    return bool(_VALID_ORDER.fullmatch(d or ""))


def _lookups_done(ctx: GuardContext) -> list[tuple[str, set[str], str]]:
    """이번 통화에서 이미 한 조회: [(함수 이름, 쓴 번호 숫자들, 결과 status)]. 이전 대화와 이번 턴 결과를 모두 본다."""
    out: list[tuple[str, set[str], str]] = []
    hist = ctx.history if isinstance(ctx.history, (list, tuple)) else []
    pending: dict[str, set[str]] = {}
    for h in hist:
        if not isinstance(h, dict):
            continue
        name = h.get("name")
        if h.get("role") == "tool_call" and name in LOOKUP_FUNCS and isinstance(h.get("arguments"), dict):
            pending[name] = {_digits(v) for k, v in h["arguments"].items() if k in ORDER_KEYS + PHONE_KEYS} - {""}
        elif h.get("role") == "tool_result" and name in LOOKUP_FUNCS:
            res = h.get("result")
            out.append((name, pending.pop(name, set()), str(res.get("status") if isinstance(res, dict) else "")))
    results = ctx.results if isinstance(ctx.results, (list, tuple)) else []
    for r in results:
        if isinstance(r, dict) and r.get("name") in LOOKUP_FUNCS and isinstance(r.get("arguments"), dict):
            res = r.get("result")
            nums = {_digits(v) for k, v in r["arguments"].items() if k in ORDER_KEYS + PHONE_KEYS} - {""}
            out.append((r["name"], nums, str(res.get("status") if isinstance(res, dict) else "")))
    return out


def _trusted_phones(kn: _Known, caller_phone: str) -> list[str]:
    """믿는 전화번호들, 쓸 순서대로: 발신자 번호 → 이전 대화에서 나중에 나온 번호부터."""
    hist = [p for p in dict.fromkeys(kn.trusted["phones"]) if p != caller_phone and valid_phone(p)]
    return ([caller_phone] if valid_phone(caller_phone) else []) + hist[::-1]


def _lookup_numbers(name: str, out: dict, kn: _Known, ctx: GuardContext, caller_phone: str) -> str:
    """조회 함수의 번호 인수를 코드가 정한다(out을 바꾼다). 조회하지 말아야 하면 LLM에 돌려줄 안내문을 반환한다.

    사용자 확정 규칙(2026-11-21): 번호는 발신자 번호 → 이전 대화 → 이번에 들은 번호 순서로 쓴다.
    시험 음성은 붙여 쓴 번호를 뭉개 읽어 STT가 틀리게 적으므로(음성에만 있는 번호 17개 중 5개만 맞음),
    믿는 번호가 있으면 들은 번호 대신 그 번호로 조회한다. 믿는 번호로 이미 조회했는데 고객이 다른 번호를 말하면:
      형식에 맞는 번호 → 그 번호로 한 번만 더 조회 / 형식에 안 맞는 번호 → 조회하지 않고 다시 묻는다.
    """
    done = [d for d in _lookups_done(ctx) if d[0] == name]
    used = set().union(*(nums for _, nums, _ in done)) if done else set()
    # 믿는 번호는 이번 턴 이전에 알던 것만(이번 턴에 들은 번호로 조회한 기록이 믿는 번호가 되지 않게)
    before = _collect(GuardContext(tools=[], user_text="", history=ctx.history, caller=ctx.caller, results=[]))
    trusted = _trusted_phones(before, caller_phone)
    phone = next((p for p in trusted if p not in used), "")     # 아직 조회에 쓰지 않은 믿는 번호
    tried = bool(trusted) and not phone                         # 믿는 번호는 이미 다 조회해 봤다
    trusted_orders = [o for o in dict.fromkeys(kn.trusted["orders"]) if valid_order(o)]
    phone_key = "buyer_cellphone" if name == "get_order_info" else "customer_phone_number"

    heard_phones = [(k, _digits(out[k])) for k in PHONE_KEYS if k in out]
    heard_orders = [(k, _digits(out[k])) for k in ORDER_KEYS if k in out]

    if phone:
        # 믿는 전화번호로 아직 조회하지 않았다 → 그 번호로. 들은 번호는 쓰지 않는다(이전 대화에 있던 주문번호만 함께 둔다).
        for k, _ in heard_phones:
            del out[k]
        out[phone_key] = phone
        for k, d in heard_orders:
            if d not in trusted_orders or name != "get_order_info":
                del out[k]
        return ""

    # 믿는 전화번호가 없거나, 그 번호로는 이미 조회했다 → 이번에 들은 번호를 본다
    fresh = [(k, d) for k, d in heard_phones if valid_phone(d) and d not in used] + \
            [(k, d) for k, d in heard_orders if valid_order(d) and d not in used and name == "get_order_info"]
    if not fresh and name == "get_order_info" and len(trusted_orders) == 1 and not heard_phones:
        only = trusted_orders[0]                        # 이전 대화에 주문이 하나뿐일 때만 그 주문번호로.
        if only not in used:                            # 여러 개면 고르지 않는다(다른 주문을 안내하는 것이 더 나쁘다)
            for k, _ in heard_orders:
                del out[k]
            out["order_id"] = format_order_id(only)
            return ""
    if fresh:
        turn = [r for r in (ctx.results if isinstance(ctx.results, (list, tuple)) else [])
                if isinstance(r, dict) and r.get("name") == name and isinstance(r.get("arguments"), dict)]
        retried = any(not ({_digits(v) for v in r["arguments"].values()} & set(trusted)) for r in turn)
        if tried and retried:
            return ("들은 번호로는 이미 한 번 다시 조회했습니다. 더 조회하지 말고, 확인되는 내역이 없다고 알린 뒤 "
                    "주문하실 때 쓴 연락처나 주문번호를 천천히 다시 말씀해 달라고 하세요.")
        keep = {k for k, _ in fresh}
        for k, _ in heard_phones + heard_orders:
            if k not in keep:
                del out[k]
        return ""
    if heard_phones or heard_orders:
        if tried:                                       # 믿는 번호로는 이미 조회했고, 들린 번호는 형식이 안 맞는다
            return ("방금 들린 번호는 형식이 맞지 않아 조회하지 않았습니다(음성 인식이 숫자를 틀리게 받아 적었을 수 있습니다). "
                    "이미 조회한 결과로 답할 수 있으면 그 결과로 답하고, 아니면 번호를 한 자리씩 천천히 다시 말씀해 달라고 하세요.")
        return ""                                       # 믿는 번호가 전혀 없으면 들은 그대로 조회한다
    return ""


def _fix_action_order(name: str, out: dict, kn: _Known) -> None:
    """접수 함수의 주문번호가 형식에 안 맞으면, 아는 주문이 하나뿐일 때만 그 주문으로 맞춘다."""
    orders = [o for o in dict.fromkeys(kn.trusted["orders"]) if valid_order(o)]
    for key in ORDER_KEYS:
        if key in out and not valid_order(_digits(out[key])) and len(orders) == 1:
            out[key] = format_order_id(orders[0])


# ── 검사 ─────────────────────────────────────────────
def _label(name: str, key: str, params: dict | None) -> str:
    if key == "reason":
        return _REASON_LABELS.get(name, "사유")
    if key in _LABELS:
        return _LABELS[key]
    props = params.get("properties") if isinstance(params, dict) else None
    spec = props.get(key) if isinstance(props, dict) else None
    desc = spec.get("description") if isinstance(spec, dict) else None
    return desc.strip() if isinstance(desc, str) and desc.strip() else key


def _missing(name: str, out: dict, params: dict | None) -> list[str]:
    """빠진 필수 인수를 고객에게 물을 말로."""
    labels: list[str] = []
    for key in _required(name, params):
        if _empty(out.get(key)):
            labels.append(_label(name, key, params))
    items = out.get(ITEMS_KEY)
    if not _empty(items):
        rows = items if isinstance(items, list) else [items]
        for key in _item_required(params):
            if any(not isinstance(it, dict) or _empty(it.get(key)) for it in rows):
                labels.append(_label(name, key, params))
    if name == "get_order_info" and not any(not _empty(out.get(k)) for k in (*ORDER_KEYS, *PHONE_KEYS)):
        labels.append("주문번호 또는 전화번호")
    return list(dict.fromkeys(labels))


def _ungrounded(name: str, out: dict, kn: _Known) -> list[tuple[str, str]]:
    """근거 없는 번호 인수 [(키, 고객에게 물을 말)]."""
    bad: list[tuple[str, str]] = []
    soups = [*kn.trusted_soups, *kn.heard_soups]
    for keys, kind in ((ORDER_KEYS, "orders"), (PHONE_KEYS, "phones")):
        known = kn.all(kind)
        for key in keys:
            if key in out:
                d = _digits(out[key])
                if d not in known and not _in_soup(d, soups):
                    bad.append((key, _LABELS[key]))
    items = out.get(ITEMS_KEY)
    if isinstance(items, list):
        known = kn.all("items")
        for it in items:
            d = _digits(it.get(ITEM_CODE_KEY)) if isinstance(it, dict) else ""
            if d not in known and not _in_soup(d, soups):
                bad.append((ITEMS_KEY, _LABELS[ITEM_CODE_KEY]))
                break
    return bad


def _grounding_on() -> bool:
    return os.environ.get(GROUNDING_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def _called_names(called: Any) -> list[str]:
    out = []
    for c in called if isinstance(called, (list, tuple, set, frozenset)) else []:
        n = c if isinstance(c, str) else c.get("name") if isinstance(c, dict) else getattr(c, "name", None)
        if isinstance(n, str):
            out.append(n)
    return out


def _block(args: dict, reason: str, message: str) -> Verdict:
    return Verdict(args=args, blocked={"status": "blocked", "message": message}, reason=reason)


FIRST_TURN_ENV = "DIRECTON_GUARD_FIRST_TURN"     # 0이면 첫 마디 접수 차단을 끈다
NAME_ENV = "DIRECTON_GUARD_NAME"                 # 0이면 전화 요청 이름 근거 확인을 끈다


def _on(env: str) -> bool:
    return os.environ.get(env, "1").strip().lower() not in ("0", "false", "no", "off")


def _has_history(ctx: GuardContext) -> bool:
    """이번 말 이전에 주고받은 대화(고객·상담원 말, 함수 기록)가 있는가."""
    hist = ctx.history if isinstance(ctx.history, (list, tuple)) else []
    return any(isinstance(h, dict) and h for h in hist)


_NO_NAME = re.compile(r"미상|모름|불명|없음|고객님?$|unknown|none|null|n/?a", re.I)


def _name_grounded(name: Any, ctx: GuardContext) -> bool:
    """이름이 고객 말·이전 대화·발신자 정보에 나오는가(공백 무시). 자리 채움 말(미상 등)은 근거 없음."""
    n = re.sub(r"\s+", "", str(name or ""))
    if len(n) < 2 or _NO_NAME.fullmatch(n):
        return False
    texts = [ctx.user_text or ""]
    hist = ctx.history if isinstance(ctx.history, (list, tuple)) else []
    texts += [str(h.get("text") or h.get("content") or "") for h in hist if isinstance(h, dict)]
    if isinstance(ctx.caller, dict):
        texts.append(str(ctx.caller.get("name") or ""))
    soup = re.sub(r"\s+", "", " ".join(texts))
    return n in soup


def check(name: str, args: dict, ctx: GuardContext) -> Verdict:
    """함수 호출 하나를 검사한다. 통과면 Verdict.args로 호출하고, 막았으면 Verdict.blocked를 LLM에 결과로 돌려준다.

    입력 args는 바꾸지 않는다. 형식이 잘못된 입력에도 예외를 내지 않는다(검사 중 오류가 나면 통과시킨다).
    """
    try:
        return _check(name, args, ctx)
    except Exception as e:  # noqa: BLE001
        log.warning("가드 검사 실패(통과시킴) %s: %r", name, e)
        return Verdict(args=_copy_args(args))


def _check(name: str, args: Any, ctx: GuardContext) -> Verdict:
    out = _copy_args(args)
    name = name if isinstance(name, str) else str(name or "")
    defs = _tool_defs(getattr(ctx, "tools", None))
    known_funcs = defs or _DEFAULT_REQUIRED

    # 1. 정의에 없는 함수
    if name not in known_funcs:
        return _block(out, "undefined",
                      f"'{name}' 함수는 없습니다. 쓸 수 있는 함수는 {', '.join(known_funcs)}뿐입니다. "
                      "상담원 연결이 필요하면 함수를 부르지 말고 말로 안내하세요. " + _NOT_DONE)
    params = defs.get(name) if defs else None

    # 2·3. 번호 인수 정리, 아는 번호에 맞춤
    kn = _collect(ctx)
    _normalize_ids(out, kn)
    caller = ctx.caller if isinstance(ctx.caller, dict) else {}
    if name in LOOKUP_FUNCS:
        stop = _lookup_numbers(name, out, kn, ctx, _id_digits(caller.get("phone"), phone=True))
        if stop:
            return _block(out, "unclear_number", stop)
    elif name in ACTION_FUNCS:
        _fix_action_order(name, out, kn)

    # 4. 필수 인수
    missing = _missing(name, out, params)
    if missing:
        what = ", ".join(missing)
        return _block(out, "missing",
                      f"{name} 호출에 필요한 {what} 값이 없습니다. 고객이 이미 말한 값이 있으면 그 값으로 다시 호출하고, "
                      f"없으면 고객에게 {what}만 여쭤보세요. 값을 지어내지 마세요. " + _NOT_DONE)

    if name in ACTION_FUNCS:
        # 4-1. 통화 첫 마디에서는 접수하지 않는다(연습 문항 80개에서 첫 턴 접수가 정답인 경우는 없다)
        if _on(FIRST_TURN_ENV) and not _has_history(ctx):
            return _block(out, "first_turn",
                          f"통화의 첫 마디에서 바로 {name}을 부르지 않습니다. 먼저 주문을 확인하고(주문번호나 전화번호), "
                          "사유처럼 접수에 필요한 내용을 고객에게 여쭤보세요. " + _NOT_DONE)
        # 4-2. 전화 요청의 이름은 고객이 말했거나 발신자 정보에 있는 이름이어야 한다
        if name == "request_callback" and _on(NAME_ENV) and not _name_grounded(out.get("customer_name"), ctx):
            return _block(out, "missing",
                          "request_callback에 넣을 고객 이름을 아직 받지 못했습니다. 이름을 지어내거나 '미상'으로 넣지 말고 "
                          "고객에게 성함을 여쭤보세요. 상담원 연결 안내만 필요하면 함수 없이 말로 안내하세요. " + _NOT_DONE)
        # 5. 접수 함수의 번호 근거
        if _grounding_on():
            required = set(_required(name, params))
            bad = _ungrounded(name, out, kn)
            for key, _ in bad:
                if key not in required and key != ITEMS_KEY:
                    del out[key]                    # 필수가 아닌 번호는 빼고 진행
            bad = [(k, label) for k, label in bad if k in out]
            if bad:
                what = ", ".join(dict.fromkeys(label for _, label in bad))
                return _block(out, "ungrounded",
                              f"{name}에 넣은 {what}는 고객이 말했거나 조회로 확인된 번호가 아닙니다. "
                              f"고객에게 {what}를 다시 확인하세요. 번호를 지어내지 마세요. " + _NOT_DONE)
        # 6. 한 턴에 접수 함수는 하나
        done = [c for c in _called_names(getattr(ctx, "called", None)) if c in ACTION_FUNCS]
        if done:
            if name in done:
                msg = (f"{name}은 이번 턴에 이미 실행했습니다. 다시 부르지 말고 앞서 받은 결과를 그대로 안내하세요.")
            else:
                msg = (f"이번 턴에 이미 {done[0]}을 실행했습니다. 한 번에 접수는 하나만 합니다. "
                       f"{name}은 실행되지 않았으니 처리됐다고 말하지 말고, 앞서 받은 결과만 안내하세요.")
            return _block(out, "second_action", msg)

    # 7. 통과
    return Verdict(args=out)
