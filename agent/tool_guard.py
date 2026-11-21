"""함수 호출 안전장치(AG-T45). 공식 측정의 세 가지 실패(docs/14_공식측정_결과보고서.md 6절 1~3)를 막는다.

1. 전체 반품을 일부 반품으로 호출: `fix_full_return` — return_order_item의 상품·수량이 이미 조회한 주문(get_order_info
   결과: 이번 턴 tool 메시지 또는 시스템 프롬프트의 이전 대화 '(조회 결과)' 줄) 전체와 같고, 최근 고객 말에 일부 반품
   표현이 없으면 return_order로 바꾼다. 주문 정보를 모르면 그대로 둔다.
2. 접수 함수 없이 "접수해 두었습니다": `SpeechGuard`가 처리 완료를 말하는 문장(`is_claim`)을 붙잡는다. 이번 턴에 접수
   함수가 성공했거나 이전 대화에 접수 기록이 있으면 그대로 내보낸다(지연 없음). 아니면 내보내지 않고 교정 지시로
   한 번 다시 생성한다. 그래도 같으면 그 문장만 버리고, 아무 말도 남지 않으면 FALLBACK_TEXT.
3. 함수 호출 문자열·JSON 낭독: `SpeechGuard`가 TTS로 보내기 전 문장 단위로 거른다(`is_leak`, 여러 문장에 걸친 괄호도
   추적). 알려진 함수 호출이면 실제 호출로 바꾸고(`parse_leaked_calls`), 아니고 남은 말도 없으면 한 번 다시 생성한다.

응답기(agent/responder.py)는 문장이 TTS로 가기 직전(`filter`·`finish`), 툴 실행 직전(`fix_full_returns`),
툴 결과 직후(`note_results`)에만 이 모듈을 부른다.
"""
from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any, Iterable

from agent.core.types import Message, ToolCall

log = logging.getLogger(__name__)

READ_ONLY_PREFIXES = ("get_", "search_", "track_", "check_", "find_", "lookup_", "list_")
SUCCESS_STATUSES = ("ok", "already_requested")
FALLBACK_TEXT = "죄송해요, 지금은 처리 결과를 바로 확인하기 어려워서 상담원이 확인한 뒤 안내해 드릴게요."
CLAIM_NOTE = ("[시스템 확인] 방금 답에서 '{claim}'처럼 접수·처리가 끝난 것처럼 말하려 했지만, 이번 통화에서 그 처리를 하는 "
              "함수가 호출되어 성공한 기록이 없다. 고객이 처리를 분명히 요청했고 필요한 정보가 모두 있으면 지금 말없이 해당 함수를 "
              "호출한다. 정보가 부족하면 필요한 것만 짧게 묻는다. 함수 결과를 받기 전에는 접수·처리됐다고 말하지 않는다.")
LEAK_NOTE = ("[시스템 확인] 방금 답에 함수 호출 형식이나 JSON을 글자로 썼다. 함수는 글로 쓰지 말고 함수 호출 기능으로만 부르고, "
             "고객에게는 한국어 문장만 말한다.")
SAID_NOTE = " 이번 답에서 고객에게 이미 '{said}'까지 말했으니 반복하지 말고 그 뒤를 이어서 말한다."


def registration_tools(names: Iterable[str]) -> frozenset[str]:
    """접수(쓰기) 함수: 조회 접두사(get_·search_ 등)가 아닌 함수. 공식 8개면 취소·반품 2·교환·전화 요청."""
    return frozenset(n for n in names if n and not str(n).startswith(READ_ONLY_PREFIXES))


def _norm_id(v: Any) -> str:
    return re.sub(r"[^0-9A-Za-z]", "", str(v or ""))


def _ws(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


# ── 1. 전체 반품 ────────────────────────────────────────
_HIST_RESULT = re.compile(r"\(조회 결과\) (\w+) → (.+)$", re.M)
_HIST_CUSTOMER = re.compile(r"^\s*- 고객: (.*)$", re.M)
# 일부 반품 표현. '4개만 와서'처럼 수량 불만은 반품 동사와 붙을 때만 본다.
_PARTIAL = re.compile(r"일부|부분반품|부분환불|(하나|한개|한박스|한통|\d+(개|박스|통|포|병))만(반품|환불|보내|돌려|회수)"
                      r"|만(반품|환불)|빼고|제외하고")


def _orders_in(result: Any) -> list[dict]:
    data = result.get("data") if isinstance(result, dict) else None
    orders = data.get("orders") if isinstance(data, dict) else None
    return [o for o in orders or [] if isinstance(o, dict)]


def _load(text: str) -> Any:
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def order_items(system: str, msgs: list[Message]) -> dict[str, dict[str, int]]:
    """조회한 주문 → {주문번호(원래 표기): {상세주문번호(숫자·영문만): 수량}}. 이전 대화 '(조회 결과)' 줄 + tool 메시지."""
    results: list[Any] = []
    for name, body in _HIST_RESULT.findall(system or ""):
        if name == "get_order_info":
            results.append(_load(body.strip()))
    for m in msgs or []:
        if m.role == "tool" and m.name == "get_order_info":
            results.append(_load(m.content))
    out: dict[str, dict[str, int]] = {}
    for res in results:
        for o in _orders_in(res):
            order = str(o.get("주문번호", o.get("order_id")) or "").strip()
            code = _norm_id(o.get("상세주문번호", o.get("order_item_code")))
            qty = o.get("수량", o.get("quantity"))
            if not order or not code:
                continue
            try:
                q = int(qty)
            except (TypeError, ValueError):
                q = 1
            out.setdefault(order, {})[code] = q
    return out


def _recent_customer(system: str, msgs: list[Message], n: int = 3) -> list[str]:
    users = [m.content for m in msgs or [] if m.role == "user"]
    return users[-n:] + _HIST_CUSTOMER.findall(system or "")[-n:]


def fix_full_return(tc: ToolCall, system: str, msgs: list[Message], schemas: dict) -> ToolCall | None:
    """전체 반품인데 return_order_item이면 같은 id의 return_order 호출. 바꿀 근거가 없으면 None."""
    if tc.name != "return_order_item" or "return_order" not in (schemas or {}):
        return None
    args = tc.arguments or {}
    items = args.get("return_items")
    if not isinstance(items, list) or not items:
        return None
    req: dict[str, int] = {}
    for it in items:
        if not isinstance(it, dict):
            return None
        code = _norm_id(it.get("order_item_code"))
        try:
            q = int(it.get("quantity"))
        except (TypeError, ValueError):
            return None
        if not code:
            return None
        req[code] = req.get(code, 0) + q
    owners = [order for order, codes in order_items(system, msgs).items() if set(req) & set(codes)]
    if len(owners) != 1:
        return None                                       # 모르는 상품이거나 여러 주문에 걸침 → 그대로
    codes = order_items(system, msgs)[owners[0]]
    if set(req) != set(codes) or any(req[c] < codes[c] for c in codes):
        return None                                       # 주문 일부(상품 일부·수량 일부) → 진짜 일부 반품
    if any(_PARTIAL.search(_ws(t)) for t in _recent_customer(system, msgs)):
        return None                                       # 고객이 일부 반품이라고 말함(공식 P0147-07)
    props = (schemas.get("return_order") or {}).get("properties") or {}
    new = {"order_number": owners[0]}
    for k, v in args.items():
        if k in props and k not in ("order_number", "return_items") and v not in (None, "", {}, []):
            new[k] = v
    return ToolCall(tc.id, "return_order", new)


def fix_full_returns(calls: list[ToolCall], system: str, msgs: list[Message], schemas: dict, tlog: Any = None) -> list[ToolCall]:
    out = []
    for tc in calls:
        fixed = fix_full_return(tc, system, msgs, schemas)
        if fixed is not None:
            log.info("full return: return_order_item -> return_order")
            if tlog:
                tlog.event("full_return_fixed", tool=tc.name, order=fixed.arguments.get("order_number"))
        out.append(fixed or tc)
    return out


# ── 2. 접수 주장 ────────────────────────────────────────
_ACTION = r"(접수|신청|요청|취소|반품|교환|환불|처리|전달)"
_DONE = (r"(완료(됐|되었|되셨|되어|돼|했|하였|해드렸|입니다|예요|이에요|였)|됐|되었|되셨|돼있|되어있|해드렸|해두었|해뒀|해놓았|해놨"
         r"|했|하였|드렸|남겨두었|남겨뒀|남겨드렸|남겨놓았|남겨놨)")
_CLAIM = re.compile(_ACTION + r"(이|가|을|를|은|는|도|건|까지)?" + _DONE)
_LEFT = re.compile(r"남겨(두었|뒀|드렸|놓았|놨)")
_NOT_CLAIM_AFTER = ("는지", "으면", "다면", "나요", "는데도", "을까", "는가", "던가", "을지", "는지요")


def is_claim(sentence: str) -> bool:
    """접수·처리가 끝났다고 말하는 문장인가("접수해 두었습니다", "반품 접수가 완료되었습니다"). 질문·조건·미래("접수해 드릴까요?",
    "접수되면", "접수해 드릴게요")는 아니다."""
    s = (sentence or "").strip()
    if not s or s.endswith("?"):
        return False
    t = _ws(s)
    for pat in (_CLAIM, _LEFT):
        for m in pat.finditer(t):
            if not t[m.end():].startswith(_NOT_CLAIM_AFTER):
                return True
    return False


# ── 3. 호출 문자열 ──────────────────────────────────────
_WORD = re.compile(r"[A-Za-z_][\w.]*")
_JSON = re.compile(r"\{\s*\"?[\w가-힣]+\"?\s*:")
_CALL_SHAPE = re.compile(r"[A-Za-z_][\w.]*\s*\(\s*\{")
_HANGUL = re.compile(r"[가-힣]")


def _norm_name(word: str) -> str:
    w = word.lower().split(".")[-1]
    return w.replace("_", "")


def _known(names: Iterable[str]) -> dict[str, str]:
    return {_norm_name(n): n for n in names if n}


def _leak_start(text: str, names: Iterable[str]) -> int:
    """호출·JSON이 시작되는 위치. 없으면 -1."""
    known = _known(names)
    hits = [m.start() for m in _WORD.finditer(text) if _norm_name(m.group()) in known]
    for pat in (_CALL_SHAPE, _JSON):
        m = pat.search(text)
        if m:
            hits.append(m.start())
    return min(hits) if hits else -1


def is_leak(sentence: str, names: Iterable[str]) -> bool:
    """함수 이름(searchproducts·search_products·functions.x)이나 호출 형태·JSON이 들어 있나."""
    return _leak_start(sentence or "", names) >= 0


def _balanced(text: str, start: int) -> int:
    """text[start]의 '{'와 짝이 맞는 '}' 다음 위치. 닫히지 않으면 -1."""
    depth = 0
    quote = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if quote:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                quote = False
            continue
        if c == '"':
            quote = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def _as_dict(body: str) -> dict | None:
    for loader in (json.loads, ast.literal_eval):
        try:
            v = loader(body)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(v, dict):
            return v
    return None


def parse_leaked_calls(text: str, names: Iterable[str]) -> list[ToolCall]:
    """글로 쓴 호출("search_products({...})", "functions.searchproducts({...})") → ToolCall. 알려진 함수 + 인자 JSON만."""
    known = _known(names)
    out: list[ToolCall] = []
    pos = 0
    while True:
        m = _WORD.search(text or "", pos)
        if not m:
            break
        pos = m.end()
        name = known.get(_norm_name(m.group()))
        if not name:
            continue
        rest = text[m.end():]
        lead = re.match(r"\s*\(?\s*", rest)
        brace = m.end() + (lead.end() if lead else 0)
        if brace >= len(text) or text[brace] != "{":
            continue
        end = _balanced(text, brace)
        if end < 0:
            continue
        args = _as_dict(text[brace:end])
        if args is None:
            continue
        out.append(ToolCall(f"leak_{len(out)}", name, args))
        pos = end
    return out


# ── 송신 직전 거르기 ─────────────────────────────────────
class SpeechGuard:
    """한 응답(턴) 동안 TTS로 가는 문장을 거른다. 회차마다 `filter`(스트리밍 중) → `finish`(회차 끝) 순서로 부른다.
    finish 뒤 `extra_calls`(글로 쓴 호출을 바꾼 것)·`retry_note`(다시 생성할 때 시스템 프롬프트에 붙일 지시)를 본다."""

    def __init__(self, schemas: dict, system: str, msgs: list[Message]):
        self.names = [n for n in (schemas or {}) if n]
        self.reg = registration_tools(self.names)
        self.retries = 0
        self.blocked = False
        self.registered = self._prior_registration(system, msgs)
        self.extra_calls: list[ToolCall] = []
        self.retry_note: str | None = None
        self._reset()

    def _reset(self) -> None:
        self._held: list[tuple[str, bool]] = []           # (문장, 접수 주장인가)
        self._depth = 0                                    # 여러 문장에 걸친 호출·JSON의 열린 괄호 수
        self._leaked = False
        self._passed: list[str] = []

    def _prior_registration(self, system: str, msgs: list[Message]) -> bool:
        """이전 대화에 접수 기록(함수 호출 줄 또는 상담원의 접수 완료 말)이 있거나, 이번 통화 앞선 답이 접수를 말했으면 True."""
        for name in self.reg:
            if f"(상담원이 함수 호출) {name} " in (system or ""):
                return True
        if any(is_claim(line) for line in re.findall(r"^\s*- 상담원: (.*)$", system or "", re.M)):
            return True
        for m in msgs or []:
            if m.role == "assistant" and m.content and any(is_claim(s) for s in re.split(r"(?<=[.?!])\s+", m.content)):
                return True
            if m.role == "tool" and m.name in self.reg and self._ok(_load(m.content)):
                return True
        return False

    @staticmethod
    def _ok(res: Any) -> bool:
        return isinstance(res, dict) and res.get("status") in SUCCESS_STATUSES

    def note_results(self, calls: list[ToolCall], results: list[Any]) -> None:
        """툴 결과: 접수 함수 성공, 또는 상태 조회(get_*) 결과에 접수·취소·반품 같은 처리 상태가 있으면 그 뒤 처리 완료 말을
        허용한다("이미 반품 접수되어 있어요"). search_products의 정책 문구는 근거로 보지 않는다."""
        for tc, res in zip(calls, results):
            if not self._ok(res):
                continue
            if tc.name in self.reg:
                self.registered = True
            elif tc.name.startswith("get_") and re.search(_ACTION, json.dumps(res, ensure_ascii=False)):
                self.registered = True

    def _strip_leak(self, s: str) -> str:
        """호출·JSON 부분을 뺀 나머지 말. 괄호가 닫히지 않으면 다음 문장까지 계속 뺀다."""
        rest = s
        kept = ""
        while rest:
            if self._depth > 0:
                d = 0
                cut = -1
                for i, c in enumerate(rest):
                    d += (c == "{") - (c == "}")
                    if self._depth + d <= 0:
                        cut = i + 1
                        break
                if cut < 0:
                    self._depth += d
                    return kept
                self._depth = 0
                rest = rest[cut:].lstrip(" )\"'`;,")
                continue
            start = _leak_start(rest, self.names)
            if start < 0:
                return kept + rest
            self._leaked = True
            kept += rest[:start]
            if rest[start] == "{":
                brace = start
            else:
                word = _WORD.match(rest, start)
                lead = re.match(r"\s*\(?\s*\{", rest[word.end():]) if word else None
                if lead is None:
                    return kept                              # 함수 이름만 쓴 말("search_products 호출할게요") → 뒤를 통째로 뺀다
                brace = word.end() + lead.end() - 1
            end = _balanced(rest, brace)
            if end < 0:
                self._depth = rest[brace:].count("{") - rest[brace:].count("}")
                return kept
            rest = rest[end:].lstrip(" )\"'`;,")
        return kept

    def _speakable(self, s: str) -> str | None:
        if self._depth == 0 and not is_leak(s, self.names):
            return s
        left = self._strip_leak(s).strip(" ,")
        if left and len(_HANGUL.findall(left)) >= 2 and not is_leak(left, self.names):
            return left
        if left or s:
            log.info("leaked call text dropped")
        return None

    def filter(self, sentences: Iterable[str]) -> list[str]:
        out: list[str] = []
        for raw in sentences:
            s = self._speakable(raw)
            if s is None:
                self.blocked = True
                continue
            claim = not self.registered and is_claim(s)
            if claim or self._held:                          # 주장 뒤 문장은 순서를 지키려고 함께 붙잡는다
                self._held.append((s, claim))
                continue
            self._passed.append(s)
            out.append(s)
        return out

    def finish(self, sentences: Iterable[str], calls: list[ToolCall], raw_text: str) -> list[str]:
        """회차 끝. 남은 문장을 거르고, 붙잡은 문장을 정리한다. extra_calls·retry_note를 새로 정한다."""
        out = self.filter(sentences)
        held, leaked, passed = self._held, self._leaked, self._passed + out
        self.extra_calls, self.retry_note = [], None
        self._reset()
        if leaked and not calls:
            self.extra_calls = parse_leaked_calls(raw_text, self.names)
        claims = [s for s, c in held if c]
        if calls or self.extra_calls:
            if claims:
                self.blocked = True                          # 함수 결과 전의 처리 완료 말은 버린다(결과 뒤 회차가 말한다)
            return out
        if claims and self.retries == 0:
            self.retries += 1
            self.blocked = True
            self.retry_note = CLAIM_NOTE.format(claim=claims[0]) + (SAID_NOTE.format(said=" ".join(passed)) if passed else "")
            return out
        if leaked and not passed and not held and self.retries == 0:
            self.retries += 1
            self.retry_note = LEAK_NOTE
            return out
        if claims:
            self.blocked = True
        return out + [s for s, c in held if not c]

    def fallback(self, said_chars: int) -> str | None:
        """거른 문장 때문에 아무 말도 못 했으면 내보낼 문장."""
        return FALLBACK_TEXT if self.blocked and said_chars == 0 else None
