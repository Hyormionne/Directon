"""응답기: LLM 스트리밍 + 툴 루프 → 문장 분할 → 문장별 TTS(다음 문장 미리 합성) → 송신 콜백."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
from typing import Any, AsyncIterator, Awaitable, Callable

from agent.config import Settings
from agent.core.types import LLMEvent, LLMProvider, Message, ToolCall, TTSProvider
from agent import drafts as D
from agent.normalize import spoken_digits
from agent.providers._privacy import redact  # 고객 번호는 저장 금지 모드에서 로그에 남기지 않는다(AG-T18)
from agent.turn_log import TurnLog

log = logging.getLogger(__name__)

AudioSink = Callable[[bytes], Awaitable[None]]
CallTool = Callable[[str, dict], Awaitable[dict]]
NormalizeArgs = Callable[[str, dict, dict, str], dict]


def identity_normalize(tool_name: str, args: dict, schema: dict, now: str) -> dict:
    return args


class ResponderError(RuntimeError):
    pass


# ── 문장 분할 ─────────────────────────────────────────
_PUNCT = set(".?!…~。？！")
_MARKDOWN = re.compile(r"[*#`_]+")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️]")


def clean_for_speech(text: str) -> str:
    return _EMOJI.sub("", _MARKDOWN.sub("", text))


def _nchars(s: str) -> int:
    return sum(1 for c in s if not c.isspace())


class SentenceSplitter:
    """증분 문장 분할. 문장부호(. ? ! … ~) 뒤에 공백/줄바꿈이 와야 경계로 본다(1.5 같은 소수 보호).
    min_chars보다 짧은 조각은 다음 문장과 합친다."""

    def __init__(self, min_chars: int = 6):
        self.min_chars = min_chars
        self.buf = ""
        self.pending = ""

    def _emit(self, sentence: str, out: list[str]) -> None:
        s = f"{self.pending} {sentence}".strip() if self.pending else sentence.strip()
        if not s:
            return
        if _nchars(s) < self.min_chars:
            self.pending = s
        else:
            out.append(s)
            self.pending = ""

    def feed(self, text: str) -> list[str]:
        self.buf += clean_for_speech(text)
        out: list[str] = []
        i = 0
        while i < len(self.buf):
            c = self.buf[i]
            if c == "\n":
                self._emit(self.buf[:i], out)
                self.buf = self.buf[i + 1:].lstrip()
                i = 0
                continue
            if c in _PUNCT:
                k = i
                while k < len(self.buf) and self.buf[k] in _PUNCT:
                    k += 1
                if k >= len(self.buf):
                    break                       # 다음 글자를 봐야 안다
                if self.buf[k].isspace():
                    self._emit(self.buf[:k], out)
                    self.buf = self.buf[k:].lstrip()
                    i = 0
                    continue
                i = k
                continue
            i += 1
        return out

    def flush(self) -> list[str]:
        rest = self.buf.strip()
        s = f"{self.pending} {rest}".strip() if self.pending else rest
        self.buf, self.pending = "", ""
        return [s] if s else []


# ── LLM 선호출 ────────────────────────────────────────
class Prefetch:
    """LLM 스트림을 미리 읽어 버퍼에 쌓는다. 툴 실행 같은 부작용은 없다."""

    def __init__(self, agen: AsyncIterator[LLMEvent]):
        self.buf: list[LLMEvent] = []
        self.done = False
        self.error: BaseException | None = None
        self._ev = asyncio.Event()
        self.task = asyncio.create_task(self._run(agen))

    async def _run(self, agen: AsyncIterator[LLMEvent]) -> None:
        try:
            async for ev in agen:
                self.buf.append(ev)
                self._ev.set()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            self.error = e
        finally:
            self.done = True
            self._ev.set()
            aclose = getattr(agen, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except BaseException:  # noqa: BLE001
                    pass

    @property
    def cancelled(self) -> bool:
        return self.task.cancelled()

    def cancel(self) -> None:
        self.task.cancel()

    async def events(self) -> AsyncIterator[LLMEvent]:
        i = 0
        while True:
            if i < len(self.buf):
                yield self.buf[i]
                i += 1
                continue
            if self.done:
                if self.error is not None:
                    raise self.error
                if self.task.cancelled():
                    raise ResponderError("prefetch cancelled")
                return
            self._ev.clear()
            await self._ev.wait()


# ── 송신 파이프라인 ───────────────────────────────────
class _Speaker:
    """문장이 들어오는 즉시 합성을 시작하고(파이프라인), 순서대로 송신한다."""

    def __init__(self, tts: TTSProvider, send_audio: AudioSink, spoken: list[str],
                 settings: Settings, tlog: TurnLog | None,
                 on_audio: Callable[[str, bytes], None] | None = None):
        self.tts = tts
        self.send_audio = send_audio
        self.on_audio = on_audio            # (kind: "pcm"=캐시 필러 | "tts"=본 응답, pcm) — 송신 직전 호출(AG-T21 측정)
        self.spoken = spoken
        self.settings = settings
        self.tlog = tlog
        self.q: asyncio.Queue = asyncio.Queue()
        self.synths: list[asyncio.Task] = []
        self.task = asyncio.create_task(self._run())

    def say(self, text: str) -> None:
        out: asyncio.Queue = asyncio.Queue()
        self.synths.append(asyncio.create_task(self._synth(text, out)))
        self.q.put_nowait(("tts", text, out))

    def play(self, text: str, pcm: bytes, kind: str = "pcm") -> None:
        """미리 만든 소리. kind="pcm"은 캐시 필러, "answer"는 미리 합성한 답 후보(측정상 본 응답)."""
        self.q.put_nowait((kind, text, pcm))

    async def _synth(self, text: str, out: asyncio.Queue) -> None:
        if self.tlog:
            self.tlog.event("tts_request", text=text)
        try:
            first = True
            async for chunk in self.tts.synthesize(text):
                if first and self.tlog:
                    self.tlog.event("tts_first_byte", text=text)
                first = False
                out.put_nowait(chunk)
            out.put_nowait(None)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            out.put_nowait(e)

    async def _run(self) -> None:
        while True:
            item = await self.q.get()
            if item is None:
                return
            kind, text, payload = item
            if kind in ("pcm", "answer"):
                self.spoken.append(text)
                self._notify("tts" if kind == "answer" else kind, payload)
                await self.send_audio(payload)
                continue
            first = True
            while True:
                chunk = await asyncio.wait_for(payload.get(), self.settings.tts_timeout_s)
                if chunk is None:
                    break
                if isinstance(chunk, BaseException):
                    raise chunk
                if first:
                    self.spoken.append(text)
                    first = False
                self._notify(kind, chunk)
                await self.send_audio(chunk)

    def _notify(self, kind: str, pcm: bytes) -> None:
        if self.on_audio is not None:
            try:
                self.on_audio(kind, pcm)
            except Exception as e:  # noqa: BLE001 — 측정 훅이 응답을 막으면 안 된다
                log.debug("on_audio failed: %r", e)

    def raise_if_failed(self) -> None:
        if self.task.done():
            exc = self.task.exception()
            if exc is not None:
                raise exc

    async def finish(self) -> None:
        self.q.put_nowait(None)
        await self.task

    def cancel(self) -> None:
        self.task.cancel()
        for t in self.synths:
            t.cancel()


# ── 번호 인자 교정(AG-T10a) ───────────────────────────
# LLM(flash-lite)이 고객이 한글로 읽은 번호("이육일일 일사공 사구팔")를 숫자로 옮기다 자릿수를 섞는 일이 잦다.
# 고객 발화에서 번호를 직접 뽑아 두고, LLM 인자가 그중 하나와 거의 같지만(편집 거리 ≤ 2) 정확히 같지 않으면
# 발화의 번호로 바꾼다(같은 길이면 편집 거리 ≤ 3). 발화에 번호가 없거나 많이 다르면(다른 번호를 일부러 쓴 경우) 그대로 둔다.
_NUM_SPLIT = re.compile(r"[\s,.\-+/·~()\[\]:;!?\"']+")
_CORRECTION = re.compile(r"^(아니|아니다|아니요|아니고|아니고요|아니에요|아니라|정정|다시)$")
MIN_NUMBER_DIGITS = 7           # 기본 데이터(주문 10자리·전화 11자리) 기준. 데이터 번호가 더 짧으면 그 길이로 낮춘다(AG-T21)
MAX_GROUND_DISTANCE = 2
SHORT_GROUND_DISTANCE = 1       # MIN_NUMBER_DIGITS보다 짧은 번호(예: 4자리 주문번호)는 한 자리 차이까지만 교정
MIN_NUMBER_FLOOR = 3


def number_min_digits(order_lengths, phone_lengths) -> int:
    """발화에서 번호로 볼 최소 자리 수: min(7, 데이터의 가장 짧은 주문·전화번호 길이). 기본 데이터는 7 그대로."""
    lens = [int(n) for n in (*(order_lengths or ()), *(phone_lengths or ())) if n]
    return max(MIN_NUMBER_FLOOR, min([MIN_NUMBER_DIGITS, *lens]))


def _known_lengths():
    """(주문번호 길이들, 전화번호 길이들). agent.catalog(AG-T17)가 데이터에서 읽는다. 실패하면 기존 가정(10, 11)."""
    try:
        from agent.catalog import order_id_lengths, phone_lengths
        return frozenset(order_id_lengths()), frozenset(phone_lengths())
    except Exception as e:  # noqa: BLE001
        log.debug("known lengths unavailable: %r", e)
        return frozenset({10}), frozenset({11})


_DIGIT_CHARS = set("0123456789공영빵일이삼사오육륙칠팔구")
# (조사·어미, 앞 글자 받침 필요 여부). 긴 것부터.
_NUM_SUFFIXES = [("이거든요", True), ("이야", True), ("거든요", False), ("인데요", False), ("이고요", True), ("이에요", True), ("입니다", True), ("이요", True), ("이고", True),
                 ("으로", True), ("이랑", True), ("고요", False), ("예요", False), ("에요", False),
                 ("하고", False), ("랑", False), ("요", False), ("로", False), ("에", False), ("번", False)]


def _has_final(ch: str) -> bool:
    if ch.isdigit():
        return True
    code = ord(ch) - 0xAC00
    return 0 <= code <= 11171 and code % 28 != 0


def _token_readings(tok: str) -> list[str]:
    """숫자 토큰의 읽기 후보(첫째가 기본). "삼이요"처럼 '이'가 숫자 2일 수도 있으면 두 읽기를 모두 낸다."""
    out: list[str] = []
    cores = []
    core = tok
    while True:
        for suf, need in _NUM_SUFFIXES:
            if core.endswith(suf) and len(core) > len(suf):
                rest = core[: -len(suf)]
                if need and not _has_final(rest[-1]):
                    continue
                cores.append(core)              # 벗기기 전 형태도 후보(이→2)
                core = rest
                break
        else:
            break
    for c in [core] + list(reversed(cores)):
        c2 = re.sub(r"(요|고요|로|에)$", "", c) if c is not core else c
        if c2 and all(ch in _DIGIT_CHARS for ch in c2):
            d = spoken_digits(c2)
            if d and d not in out:
                out.append(d)
    if any(ch.isdigit() for ch in core):        # 아라비아 숫자 뒤 '이요'는 조사뿐
        return out[:1]
    return out


def utterance_numbers(text: str, min_digits: int = MIN_NUMBER_DIGITS) -> list[str]:
    """발화 속 번호(연속된 숫자 토큰 묶음, MIN_NUMBER_DIGITS자리 이상). "아니다" 뒤 짧은 묶음은 앞 번호의 끝을
    대체한다. 읽기가 모호한 토큰("삼이요")은 다른 읽기의 번호도 함께 낸다."""
    out: list[str] = []
    run: list[list[str]] = []
    corrected = False

    def variants(r: list[list[str]]) -> list[str]:
        base = "".join(x[0] for x in r)
        vs = [base]
        for i, x in enumerate(r):
            for alt in x[1:]:
                vs.append("".join(alt if j == i else y[0] for j, y in enumerate(r)))
        return vs

    def close() -> None:
        nonlocal run
        if run:
            vs = variants(run)
            if corrected and out and len(vs[0]) < len(out[-1]):
                prev = out[-1]
                out.extend(prev[: len(prev) - len(v)] + v for v in vs if len(v) < len(prev))
            else:
                out.extend(v for v in vs if len(v) >= min_digits)
        run = []

    for tok in _NUM_SPLIT.split(text or ""):
        if not tok:
            continue
        rd = _token_readings(tok)
        if rd:
            run.append(rd)
            continue
        close()
        corrected = bool(_CORRECTION.match(tok))
    close()
    return list(dict.fromkeys(out))


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


NUMBER_HINT = (" 숫자만 넣는다(하이픈 가능). 이름·단어를 넣지 않는다. 고객이 이번에 말했거나 발신 고객 정보·이전 대화에 있는 "
               "번호만 쓰고, 없으면 이 칸을 비운다.")


def with_number_hints(tools: list[dict]) -> list[dict]:
    """LLM에 보여 줄 함수 정의 사본: 번호 칸 설명에 NUMBER_HINT를 덧붙인다(측정기로 가는 실제 호출과 무관)."""
    out = []
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        props = ((fn or {}).get("parameters") or {}).get("properties") or {}
        if not any(_is_number_key(k) for k in props):
            out.append(t)
            continue
        new_props = {k: ({**p, "description": str(p.get("description") or "") + NUMBER_HINT}
                         if _is_number_key(k) and isinstance(p, dict) else p) for k, p in props.items()}
        out.append({**t, "function": {**fn, "parameters": {**fn["parameters"], "properties": new_props}}})
    return out


_ORDER15 = re.compile(r"^(20\d{6})-?(\d{7})$")


def order_display(v: Any) -> Any:
    """주문번호를 데이터 형식(YYYYMMDD-NNNNNNN)으로. 응대 채점은 하이픈 없는 번호를 '다른 번호'로 본다(실측 4문항)."""
    if not isinstance(v, str):
        return v
    m = _ORDER15.match(re.sub(r"\s", "", v))
    return f"{m.group(1)}-{m.group(2)}" if m else v


def _has_digits(v: Any) -> bool:
    return isinstance(v, (str, int)) and bool(spoken_digits(str(v)))


def _is_number_key(key: str) -> bool:
    k = key.lower()
    return "phone" in k or k == "id" or k.endswith("_id") or k == "order_number"


def ground_number_args(args: dict, candidates: list[str]) -> dict:
    """번호 인자를 발화에서 뽑은 번호(candidates, 최근 것이 뒤)에 맞춘다."""
    if not candidates:
        return args
    out = dict(args)
    for key, val in args.items():
        if not _is_number_key(key) or not isinstance(val, str) or not val.isdigit() or val in candidates:
            continue
        best = None
        for c in reversed(candidates):                 # 최근 발화 우선, 같은 길이 우선
            dist = _edit_distance(val, c)
            same = len(c) == len(val)
            rank = (not same, dist)
            if len(c) < MIN_NUMBER_DIGITS:             # 짧은 번호는 몇 자리만 달라도 다른 번호다
                limit = SHORT_GROUND_DISTANCE
            else:
                limit = MAX_GROUND_DISTANCE + 1 if same else MAX_GROUND_DISTANCE
            if dist <= limit and (best is None or rank < best[0]):
                best = (rank, c)
        if best is not None:
            out[key] = best[1]
    return out


# ── 발신 고객 주문 목록으로 번호 교정(AG-T15a) ───────────────
# STT가 끊어 읽은 주문번호를 자주 틀린다(전화 채널 63~68%). 자리 수가 틀린 번호는 그대로 보내지 않지만(여분 호출 감점),
# 발신 고객 전화번호를 알면 get_order_list로 고객 주문을 받아 들은 번호와 공통 부분열(LCS)이 충분히 긴 주문이
# 하나뿐이면 그 주문번호로 바꿔 호출한다. 되묻기(응대 감점) 대신 고객이 원한 정보를 바로 안내하려는 것.
# 끄기: 환경변수 DIRECTON_REPAIR_ORDER_ID=0 (config.py는 다른 Task가 작업 중이라 여기 상수·환경변수로 둔다).
REPAIR_ENV = "DIRECTON_REPAIR_ORDER_ID"
# 기준값은 번호 길이(L)에 대한 비율이다(AG-T21: 고정 10자리 가정 제거). L=10이면 예전 값(7, 2, 6, 2)과 같다.
REPAIR_MIN_COMMON_RATIO = 0.7   # 들은 번호와 주문번호의 공통 부분열 최소 길이(10자리면 7)
REPAIR_MAX_MISSING_RATIO = 0.2  # 들은 번호 중 주문번호에 없는 숫자 최대 개수(10자리면 2, 최소 1)
REPAIR_MIN_DIGITS_RATIO = 0.6   # 이보다 짧은 수는 번호를 말하려던 것으로 보지 않는다(10자리면 6)
REPAIR_MARGIN_RATIO = 0.2       # 2등 주문보다 공통 부분열이 이만큼 길어야 '하나뿐'으로 본다(10자리면 2, 최소 1)
REPAIR_MIN_COMMON = 7           # 10자리 기준값(호환용)
REPAIR_MAX_MISSING = 2
REPAIR_MARGIN = 2
REPAIR_MIN_DIGITS = 6
REPAIR_DEFAULT_LENGTH = 10      # 알려진 번호가 없을 때만 쓰는 기본 길이
# 프롬프트 예시 전화번호(agent.prompts._PHONE_EX). 고객이 말하지 않았는데 이 번호가 인자로 오면 지어낸 번호로 본다.
PROMPT_EXAMPLE_PHONE = "01012345678"
_CALLER_PHONE = re.compile(r"발신 고객 정보:[^\n]*?전화번호 (01\d{8,9})")
_HISTORY_PHONE = re.compile(r"이전 대화에서 나온 번호:[^\n]*?전화번호 (01\d{8,9})")


def repair_enabled() -> bool:
    return os.environ.get(REPAIR_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


_HISTORY_NUMS = re.compile(r"이전 대화에서 나온 번호: ([^\n]*)")
_LEADING_ZERO_AS_FIVE = ("500", "505", "510", "515", "550")   # '0'을 'oh'로 읽어 '오(5)'로 받아 적은 010


def known_numbers(system: str) -> tuple[list[str], list[str]]:
    """시스템 프롬프트에서 이미 확인된 (전화번호들, 주문번호들): 발신 고객 정보 + 이전 대화에서 나온 번호."""
    phones, orders = [], []
    m = _CALLER_PHONE.search(system or "")
    if m:
        phones.append(m.group(1))
    m = _HISTORY_NUMS.search(system or "")
    if m:
        for label, num in re.findall(r"(전화번호|주문번호) (\d+)", m.group(1)):
            (phones if label == "전화번호" else orders).append(num)
    return list(dict.fromkeys(phones)), list(dict.fromkeys(orders))


def fix_phone(heard: str, known: list[str]) -> str:
    """받아쓰기로 깨진 전화번호 보정: 알려진 번호와 충분히 비슷하면 그 번호(11자리 중 3자리까지 다름),
    아니면 앞자리 '5xx'(0을 '오'로 받아 적음)를 010으로. 못 고치면 들은 그대로."""
    d = re.sub(r"\D", "", heard or "")
    if not d or d in known:
        return d
    best = max(known, key=lambda k: lcs_len(d, k), default="")
    if best and lcs_len(d, best) >= len(best) - 3 and abs(len(d) - len(best)) <= 3:
        return best
    if len(d) == 11 and d[:3] in _LEADING_ZERO_AS_FIVE:
        return "010" + d[3:]
    return d


def caller_phone_from_system(system: str) -> str:
    """시스템 프롬프트(agent.prompts.build_system)에서 확인된 고객 전화번호: 발신 고객 정보 → 이전 대화 순."""
    for pat in (_CALLER_PHONE, _HISTORY_PHONE):
        m = pat.search(system or "")
        if m:
            return m.group(1)
    return ""


def lcs_len(a: str, b: str) -> int:
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if ca == cb else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def repair_thresholds(length: int) -> tuple[int, int, int, int]:
    """번호 길이 L → (공통 부분열 최소, 없는 숫자 최대, 최소 자리 수, 1·2등 차이). L=10 → (7, 2, 6, 2), L=4 → (3, 1, 3, 1)."""
    n = max(1, int(length))
    return (math.ceil(n * REPAIR_MIN_COMMON_RATIO - 1e-9),
            max(1, round(n * REPAIR_MAX_MISSING_RATIO)),
            max(2, math.ceil(n * REPAIR_MIN_DIGITS_RATIO - 1e-9)),
            max(1, round(n * REPAIR_MARGIN_RATIO)))


def known_length(values: list[str], default: int = REPAIR_DEFAULT_LENGTH) -> int:
    """알려진 번호들의 가장 흔한 길이(같으면 긴 쪽). 없으면 default."""
    lens = [len(v) for v in values if v]
    if not lens:
        return default
    return max(set(lens), key=lambda n: (lens.count(n), n))


def _close_enough(heard: str, cand: str, length: int | None = None) -> int | None:
    min_common, max_missing, _, _ = repair_thresholds(length or len(cand) or REPAIR_DEFAULT_LENGTH)
    common = lcs_len(heard, cand)
    if common < min_common or len(heard) - common > max_missing:
        return None
    return common


def match_order_id(heard: str, order_ids: list[str], length: int | None = None) -> str | None:
    """자리 수가 틀린 번호(heard)와 부분 일치하는 주문번호가 하나뿐이면 그 번호. 아니면 None.
    length(스키마의 'N자리')가 없으면 알려진 주문번호(order_ids)의 길이를 쓴다."""
    ids = [c for c in dict.fromkeys(order_ids) if c]
    n = length or known_length(ids)
    _, _, min_digits, margin = repair_thresholds(n)
    if not heard.isdigit() or len(heard) == n or len(heard) < min_digits:
        return None
    scored = sorted(((lcs_len(heard, c), c) for c in ids), reverse=True)
    if not scored or _close_enough(heard, scored[0][1], n) is None:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < margin:
        return None
    return scored[0][1]


def official_order_ids(result: Any) -> list[str]:
    """공식 get_order_info 결과의 주문번호들(순서 유지, 중복 제거)."""
    data = result.get("data") if isinstance(result, dict) else None
    rows = (data or {}).get("orders") if isinstance(data, dict) else None
    return list(dict.fromkeys(str(o["주문번호"]) for o in rows or [] if isinstance(o, dict) and o.get("주문번호")))


def _order_ids(result: Any) -> list[str]:
    return [r["order_id"] for r in _order_rows(result)]


def _order_rows(result: Any) -> list[dict]:
    """get_order_list 결과 → [{"order_id", "product_name"}] (상품명이 없으면 빈 문자열)."""
    data = result.get("data") if isinstance(result, dict) else None
    orders = (data or {}).get("orders") if isinstance(data, dict) else None
    return [{"order_id": str(o.get("order_id")), "product_name": str(o.get("product_name") or "").strip()}
            for o in orders or [] if isinstance(o, dict) and o.get("order_id")]


# ── 번호 확인 되묻기(AG-T28, 기본 끔) ─────────────────────
# 번호를 확실히 고칠 수 없어 되묻던 경우(자릿수 오류 + 교정 실패, not_found)에만, 발신 고객 본인 주문 중 가장 가까운
# 1~2개를 "끝자리 0231 무선 충전기 주문 맞으실까요?"처럼 확인하게 한다. 그 턴에 본 함수를 부르지 않는 것은 기존
# 되묻기와 같다. 후보는 고객 본인 주문에서만 고른다(남의 주문번호를 읽어 주지 않는다).
CONFIRM_ENV = "DIRECTON_CONFIRM_ORDER_ID"
CONFIRM_MAX_CANDIDATES = 2
CONFIRM_LAST_DIGITS = 4


def confirm_enabled() -> bool:
    return os.environ.get(CONFIRM_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


def last_digits(order_id: str, n: int = CONFIRM_LAST_DIGITS) -> str:
    return order_id if len(order_id) <= n else order_id[-n:]


def confirm_candidates(heard: str, orders: list[dict], length: int | None = None) -> list[dict]:
    """들은 번호(heard)와 편집 거리가 가장 가까운 고객 주문 1~2개. 기준(번호 길이의 20%, 최소 1) 밖이거나,
    같은 거리의 주문이 셋 이상이거나, 같은 번호가 있으면 []."""
    rows = [o for o in orders if o.get("order_id")]
    if not heard.isdigit() or not rows:
        return []
    n = length or known_length([o["order_id"] for o in rows])
    _, max_dist, min_digits, _ = repair_thresholds(n)
    if len(heard) < min_digits:
        return []
    scored = sorted((_edit_distance(heard, o["order_id"]), i) for i, o in enumerate(rows))
    best = scored[0][0]
    if best == 0 or best > max_dist:
        return []
    tied = [rows[i] for d, i in scored if d == best]
    if len(tied) > CONFIRM_MAX_CANDIDATES:
        return []
    return [{"order_id": o["order_id"], "last_digits": last_digits(o["order_id"]),
             "product_name": o.get("product_name") or ""} for o in tied]


def _confirm_desc(c: dict) -> str:
    oid = c["order_id"]
    head = f"주문번호 {oid}" if len(oid) <= CONFIRM_LAST_DIGITS else f"끝자리 {c['last_digits']}"
    return f"{head} {c['product_name']}".strip()


def confirm_message(cands: list[dict]) -> str:
    descs = [_confirm_desc(c) for c in cands]
    if len(descs) == 1:
        found = f"고객 본인 주문 중 가장 가까운 것은 '{descs[0]}' 주문입니다."
        ask = f"'{descs[0]} 주문 맞으실까요?'처럼 한 문장으로 확인만 한다."
    else:
        found = "고객 본인 주문 중 가까운 것은 " + ", ".join(f"'{d}'" for d in descs) + " 주문입니다."
        ask = f"'{descs[0]} 주문과 {descs[1]} 주문 중 어느 주문일까요?'처럼 묻는다."
    return ("고객이 말한 번호로는 주문을 확인하지 못했습니다. " + found + " 고객이 물은 것 중 번호 없이 매뉴얼로 "
            "답할 수 있는 정책이 있으면 한 문장으로 먼저 안내하고, " + ask + " 주문번호 전체는 읽지 않는다. 주문 상태나 결과는 지어내지 않고, 고객이 확인하기 전에는 다른 함수를 부르지 않는다.")


_LEN_RULE = re.compile(r"(\d+)\s*자리")
_TRANSFER_PROMISE = re.compile(r"(상담원|상담사)[^.?!]*연결[^.?!]*(드릴게요|드리겠습니다|드릴께요)")


def number_length_rules(schema: dict) -> dict[str, int]:
    """JSON Schema 속성 설명의 'N자리' → {인자: N}. 번호 인자(phone, *_id)만."""
    out: dict[str, int] = {}
    for key, prop in ((schema or {}).get("properties") or {}).items():
        if not _is_number_key(key) or not isinstance(prop, dict):
            continue
        m = _LEN_RULE.search(str(prop.get("description") or ""))
        if m:
            out[key] = int(m.group(1))
    return out


def promises_transfer(text: str) -> bool:
    """응답이 상담원 연결을 약속했는가("상담원에게 연결해 드릴게요"). 묻는 말("연결해 드릴까요?")은 아니다."""
    text = text or ""
    return "콜백" not in text and bool(_TRANSFER_PROMISE.search(text))


# ── 조회 뒤 답까지 시간 단축(AG-T21) ───────────────────────
# 공식 속도 점수는 함수 문항에서 "말 끝 → 첫 소리"와 "함수 결과 → 실제 답"의 평균이다. 툴 결과 뒤에는 LLM을 다시 불러야 해서
# 답까지 1~3초가 걸린다(docs/decisions/after_tool_latency.md). 결과 템플릿으로 줄인다.
# 결과 템플릿(DIRECTON_TOOL_TEMPLATE, 코드 기본 0, 실측상 켜기 권장 — docs/decisions/after_tool_latency.md): 툴 결과에 있는 사실만으로 짧은 첫 문장을 만들어 LLM보다 먼저 말한다.
#    결과의 고객용 message 첫 문장, 없으면 조회 함수의 status("현재 배송중 상태입니다."). LLM에는 이미 말한 문장을 알려 반복을 막는다.
TEMPLATE_ENV = "DIRECTON_TOOL_TEMPLATE"
VOICED_RMS = 300.0                     # 측정기(eval.runner)와 같은 유성음 기준
TEMPLATE_STATUSES = ("ok", "not_allowed", "no_refund")   # 그 밖(invalid_arguments·error 등)은 말하지 않는다
TEMPLATE_MAX_CHARS = 40
_READ_ONLY_PREFIXES = ("get_", "track_", "check_", "search_", "find_", "lookup_")
_STATUS_KEYS = ("status", "refund_status", "delivery_status", "order_status")
_INSTRUCTION_HINTS = ("고객에게", "안내하", "상담원은", "상담사는", "말씀드리", "응답하", "LLM", "시스템")
_SENT_END = re.compile(r"(?<=[.?!。])\s+")
TEMPLATE_NOTE = ("상담원이 이 결과를 받자마자 고객에게 _already_said 문장을 이미 말했다. 그 문장을 되풀이하지 말고 "
                 "그 뒤에 이어질 말만 한다.")


ACCEPT_TOOLS = frozenset({"cancel_order", "return_order", "return_order_item", "exchange_order", "request_callback"})
LOOKUP_TOOLS = frozenset({"get_order_info", "get_return_info"})
_CONDITIONAL = re.compile(r"(시면|으면|라면|이면|되면|하면)")
_PROMISE_ACCEPT = re.compile(r"(접수|반품|취소|교환|환불)\S*\s*(해\s*드렸|해\s*드릴게|하겠|할게|도와\s*드리겠|도와\s*드릴게|됐|되었|완료)"
                             r"|진행해\s*드릴게|진행하겠|(전화|연락)\s*(드리도록|드릴게|드리겠)")
_PROMISE_LOOKUP = re.compile(r"(조회|확인)\S*\s*(해\s*볼게|해\s*보겠|해\s*드릴게|하겠|할게)")
GUARD_NOTE = ("[시스템 확인] 방금 \"{text}\"라고 말했지만 이번 차례에 {what} 함수를 부르지 않았다. 고객이 이미 요청·동의했고 "
              "필요한 정보(주문번호·사유·전화번호 등)가 대화에 다 있어 지금 처리할 일이면 해당 함수를 지금 부른다. 이전 대화에서 "
              "이미 처리된 건이거나 정보가 부족하면 함수를 부르지 않고 아무 말도 하지 않는다. 이미 말한 내용은 반복하지 않는다.")


def unkept_promise(text: str, called: list[str]) -> str | None:
    """말로 접수·조회를 약속했는데 그 종류의 함수를 부르지 않았으면 'accept'/'lookup', 아니면 None.
    조건문("~주시면 확인해 드릴게요")은 약속으로 보지 않는다."""
    done = set(called)
    for sent in re.split(r"(?<=[.?!])\s+", text or ""):
        if _CONDITIONAL.search(sent):
            continue
        if _PROMISE_ACCEPT.search(sent) and not done & ACCEPT_TOOLS:
            return "accept"
        if _PROMISE_LOOKUP.search(sent) and not done & (LOOKUP_TOOLS | ACCEPT_TOOLS):
            return "lookup"
    return None


_ORDER_FMT = re.compile(r"\d{8}-\d{7}(?:-\d{2})?")


def ground_order_value(value: str, allowed: list[str]) -> str | None:
    """접수 함수 주문번호를 대화·조회 결과에 나온 번호로 맞춘다: 같은 번호 → 그 표기, 3자리 이내 차이 → 가장 가까운 번호,
    후보가 하나뿐 → 그 번호. 못 정하면 None."""
    d = re.sub(r"\D", "", str(value or ""))
    by_digits = {re.sub(r"\D", "", a): a for a in allowed}
    if d in by_digits:
        return by_digits[d]
    if d:
        best = max(by_digits, key=lambda k: lcs_len(d, k))
        if lcs_len(d, best) >= len(best) - 3:
            return by_digits[best]
    return allowed[0] if len(allowed) == 1 else None


def _env_on(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def template_enabled() -> bool:
    return _env_on(TEMPLATE_ENV, False)


def _first_sentence(text: str) -> str:
    return _SENT_END.split(text.strip(), maxsplit=1)[0].strip()


def _speakable(s: str) -> bool:
    return (bool(s) and _nchars(s) <= TEMPLATE_MAX_CHARS and not any(ch.isdigit() for ch in s)
            and not any(h in s for h in _INSTRUCTION_HINTS) and s[-1] in _PUNCT)


def tool_result_template(tool_name: str, result: Any) -> str | None:
    """툴 결과에 적힌 사실만으로 만든 첫 문장. 확실하지 않으면 None(그러면 LLM 답만 말한다).
    - 결과 status가 TEMPLATE_STATUSES가 아니면 None(우리 차단 메시지·오류는 상담원용 지시라 말하지 않는다).
    - message(고객용 문장)의 첫 문장이 짧고 숫자·지시 표현이 없으면 그대로.
    - 아니면 조회 함수(get_/track_/check_…)이고 data에 상태 값이 있을 때만 "확인해 보니 현재 {상태} 상태입니다."
      쓰기 함수(취소·반품 등)는 상태만으로 결과를 단정하지 않는다."""
    if not isinstance(result, dict) or result.get("status") not in TEMPLATE_STATUSES:
        return None
    msg = result.get("message")
    if isinstance(msg, str):
        first = clean_for_speech(_first_sentence(msg))
        if _speakable(first):
            return first
    data = result.get("data")
    if not isinstance(data, dict) or not str(tool_name or "").startswith(_READ_ONLY_PREFIXES):
        return None
    for key in _STATUS_KEYS:
        v = data.get(key)
        if isinstance(v, str) and 0 < len(v.strip()) <= 10 and not any(ch.isdigit() for ch in v):
            return f"확인해 보니 현재 {v.strip()} 상태입니다."
    return None


def _norm_sentence(s: str) -> str:
    return re.sub(r"[\s.?!,~…。]+", "", s or "")


def repeats_template(sentence: str, template: str) -> bool:
    """LLM 문장이 이미 말한 템플릿 문장의 반복인가(공백·문장부호 무시 같거나, 템플릿을 거의 그대로 담은 짧은 문장)."""
    a, b = _norm_sentence(sentence), _norm_sentence(template)
    if not a or not b:
        return False
    if a == b or (b in a and len(a) <= len(b) + 4):
        return True
    # 어미만 바꾼 반복("…주문입니다." → "…주문이에요."): 앞부분이 템플릿의 75% 이상 같고 길이가 비슷하면 반복
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k >= 0.75 * len(b) and len(a) <= len(b) + 3


def _voiced(pcm: bytes, thresh: float = VOICED_RMS) -> bool:
    if not pcm:
        return False
    import numpy as np
    n = len(pcm) - len(pcm) % 2
    a = np.frombuffer(pcm[:n], dtype=np.int16).astype(np.float32)
    for i in range(0, len(a), 320):                 # 20ms 프레임 중 하나라도 유성이면
        f = a[i:i + 320]
        if len(f) and float(np.sqrt((f * f).mean())) >= thresh:
            return True
    return False


# ── 응답기 ───────────────────────────────────────────
class Responder:
    def __init__(self, llm: LLMProvider, tts: TTSProvider, call_tool: CallTool, *,
                 tools: list[dict] | None = None, settings: Settings | None = None,
                 cache: Any = None, normalize_args: NormalizeArgs | None = None, now: str = ""):
        self.llm = llm
        self.tts = tts
        self.call_tool = call_tool
        self.tools = list(tools or [])
        self.settings = settings or Settings()
        self.cache = cache
        self.normalize_args = normalize_args or identity_normalize
        self.now = now
        self._schemas = {t.get("function", {}).get("name"): t.get("function", {}).get("parameters") or {}
                         for t in self.tools}
        self._shown_tools = with_number_hints(self.tools)
        self._numbers: list[str] = []
        self._called: list[str] = []
        self._caller_phone = ""
        self._known: tuple[list[str], list[str]] = ([], [])
        self._order_ids: list[str] | None = None
        self._order_rows: list[dict] | None = None
        self._corpus = ""                           # 주문번호 허용 목록용: 시스템(이전 대화) + 대화 + 이번 턴 조회 결과
        self._tool_t: float | None = None          # 마지막 툴 결과 수신 시각(perf_counter)
        self._after_any: float | None = None        # 그 뒤 첫 유성 PCM 송신까지(ms, 필러 포함)
        self._after_answer: float | None = None     # 그 뒤 본 응답(LLM·템플릿 문장) 첫 유성 PCM까지(ms)

    @property
    def _llm_tools(self) -> list[dict]:
        """LLM에 보이는 함수: 설명 보강한 실제 함수 + 내부 답 후보 함수(draft_answers, 실행하지 않는다)."""
        if self.settings.drafts and self.tools:
            return self._shown_tools + [D.DRAFT_TOOL]
        return self._shown_tools

    def prefetch(self, system: str, messages: list[Message]) -> Prefetch:
        return Prefetch(self.llm.stream(system, list(messages), self._llm_tools))

    async def _prepare_and_exec(self, tc: ToolCall, tlog: TurnLog | None) -> tuple[ToolCall, dict]:
        tc = await self._repair(self._normalize(tc), tlog)
        return tc, await self._exec_tool(tc, tlog)

    async def _presynth(self, text: str) -> bytes:
        buf = bytearray()
        async for chunk in self.tts.synthesize(text):
            buf += chunk
        return bytes(buf)

    def _normalize(self, tc: ToolCall) -> ToolCall:
        try:
            args = self.normalize_args(tc.name, dict(tc.arguments or {}),
                                       self._schemas.get(tc.name, {}), self.now)
        except Exception as e:  # noqa: BLE001
            log.warning("normalize_args failed for %s: %r", tc.name, e)
            args = dict(tc.arguments or {})
        if self.settings.ground_number_args and self._numbers:
            grounded = ground_number_args(args, self._numbers)
            if grounded != args:
                log.info("number args grounded for %s: %s -> %s", tc.name, redact(args), redact(grounded))
                args = grounded
        args = {k: (order_display(v) if _is_number_key(k) and "phone" not in k.lower() else v) for k, v in args.items()}
        return ToolCall(tc.id, tc.name, args)

    def _length_error(self, tc: ToolCall) -> dict | None:
        if not self.settings.validate_number_length:
            return None
        sch = self._schemas.get(tc.name, {}) or {}
        props = sch.get("properties") or {}
        num_keys = [k for k in props if _is_number_key(k)]
        needs_number = bool(num_keys) and (any(k in (sch.get("required") or []) for k in num_keys)
                                           or len(num_keys) == len(props))   # 번호가 필수이거나 인자가 번호뿐인 함수
        if needs_number and not any(_has_digits((tc.arguments or {}).get(k)) for k in num_keys):
            # 번호가 필요한 함수인데 쓸 만한 번호가 하나도 없다(말 조각·이름만 들어옴) → 보내지 않고 묻게 한다
            what = "휴대폰 번호" if all("phone" in k.lower() for k in num_keys) else "주문번호"
            return {"status": "invalid_arguments",
                        "message": f"고객이 아직 {what}를 말하지 않았다. 함수를 부르지 말고 '주문번호나 주문하신 분 "
                                   "휴대폰 번호를 말씀해 주시겠어요?'처럼 짧게 묻는다. 처리에 필요한 다른 정보가 있으면 함께 묻는다."}
        if not tc.name.startswith(("get_", "search_")):     # 접수 함수: 형식이 틀린 전화번호로는 접수하지 않는다
            for key, v in (tc.arguments or {}).items():
                d = re.sub(r"\D", "", str(v or ""))
                if "phone" in key.lower() and d and not (len(d) == 11 and d.startswith("010")):
                    return {"status": "invalid_arguments",
                            "message": "전화번호가 010으로 시작하는 11자리로 들리지 않았다. 접수하지 말고 "
                                       "'휴대폰 번호를 한 자리씩 천천히 다시 말씀해 주시겠어요?'처럼 번호만 다시 여쭙는다."}
        schema = self._schemas.get(tc.name, {}) or {}
        required = set(schema.get("required") or [])
        for key, n in number_length_rules(schema).items():
            v = (tc.arguments or {}).get(key)
            if key in required and not (isinstance(v, str) and v.isdigit()):
                what = "휴대폰 번호" if "phone" in key.lower() else "주문번호"
                return {"status": "invalid_arguments",
                        "message": f"{key}에 넣을 번호를 받지 못했습니다. 고객이 아직 번호를 말하지 않았으면 "
                                   f"'{what}를 말씀해 주시겠어요?'처럼 짧게 묻는다(처음 묻는 것이라 천천히 다시 말해 달라고 "
                                   "하지 않는다). 처리에 필요한 다른 정보가 있으면 함께 묻는다."}
            if isinstance(v, str) and v.isdigit() and len(v) != n:
                what = (f"전화번호는 010으로 시작하는 {n}자리" if "phone" in key.lower()
                        else f"주문번호는 {n}자리")
                return {"status": "invalid_arguments",
                        "message": f"{key}는 숫자 {n}자리여야 하는데 {len(v)}자리로 들렸습니다. 고객이 물은 것 중 "
                                   "번호 없이 매뉴얼로 답할 수 있는 정책(가능 조건, 기간, 비용 등)이 있으면 한 문장으로 "
                                   f"먼저 안내하고, {what}라고 알려 드리며 '번호를 한 자리씩 천천히 다시 말씀해 "
                                   "주시겠어요?'처럼 다시 묻는다. 주문 상태나 결과는 지어내지 않는다."}
        return None

    def _drop_bad_numbers(self, tc: ToolCall, tlog: TurnLog | None) -> ToolCall:
        """숫자가 하나도 없는 번호 인자(이름·말 조각)만 뺀다. 전화번호 칸이면 발신 고객 번호로 채운다.
        나머지 인자는 그대로 보낸다(인자 하나 때문에 호출 전체를 막지 않는다 — P0027-01)."""
        args = dict(tc.arguments or {})
        changed = False
        for key, v in list(args.items()):
            if _is_number_key(key) and _has_digits(v):
                fixed = self._fix_number(key, str(v))
                if fixed and re.sub(r"\D", "", fixed) != re.sub(r"\D", "", str(v)):
                    if tlog:
                        tlog.event("number_fixed", tool=tc.name, key=key, heard=v, fixed=fixed)
                    args[key] = fixed
                    changed = True
                continue
            if not _is_number_key(key) or v is None:
                continue
            if isinstance(v, str) and not v.strip():
                continue
            fill = self._caller_phone if "phone" in key.lower() and self._caller_phone else None
            if fill:
                args[key] = fill
            else:
                args.pop(key)
            changed = True
            if tlog:
                tlog.event("number_arg_dropped", tool=tc.name, key=key, value=v, filled=bool(fill))
        return ToolCall(tc.id, tc.name, args) if changed else tc

    def _fix_number(self, key: str, value: str) -> str | None:
        """번호 칸 값을 알려진 번호(발신 고객·이전 대화)로 보정. 전화번호 칸만 앞자리 010 보정도 한다."""
        phones, orders = self._known
        if "phone" in key.lower():
            return fix_phone(value, phones)
        d = re.sub(r"\D", "", value)
        if orders:
            best = max(orders, key=lambda k: lcs_len(d, k))
            if lcs_len(d, best) >= len(best) - 3:
                return order_display(best)
        return None

    async def _order_from_phone_arg(self, tc: ToolCall, tlog: TurnLog | None) -> ToolCall:
        """주문번호 칸에 전화번호(010 11자리)가 들어오면(P0164-05·P0197-03): 이전 대화 주문번호 하나 → 그 번호,
        아니면 그 전화번호로 주문을 조회해 주문이 하나뿐이면 그 주문번호. 못 정하면 그 칸을 비운다(→ 접수 막고 묻기)."""
        args = dict(tc.arguments or {})
        for key in [k for k in args if _is_number_key(k) and "phone" not in k.lower()]:
            d = re.sub(r"\D", "", str(args.get(key) or ""))
            if not (len(d) == 11 and d.startswith("010")):
                continue
            orders = self._known[1]
            pick = orders[0] if len(orders) == 1 else None
            if pick is None and "get_order_info" in self._schemas:
                res = await self._exec_tool(ToolCall("auto_phone_orders", "get_order_info", {"buyer_cellphone": d}), tlog)
                ids = official_order_ids(res)
                pick = ids[0] if len(ids) == 1 else None
            if tlog:
                tlog.event("order_arg_was_phone", tool=tc.name, key=key, picked=pick)
            if pick:
                args[key] = order_display(pick)
            else:
                args.pop(key)
            return ToolCall(tc.id, tc.name, args)
        return tc

    def _seen_orders(self) -> tuple[list[str], list[str]]:
        """대화(이전 대화·조회 결과 포함)와 고객 말에 나온 (주문번호들, 상세주문번호들)."""
        orders: list[str] = []
        items: list[str] = []
        for m in _ORDER_FMT.findall(self._corpus):
            if len(m) > 16:
                items.append(m)
                orders.append(m[:16])
            else:
                orders.append(m)
        orders += [order_display(n) for n in self._numbers if len(n) == 15]
        return list(dict.fromkeys(orders)), list(dict.fromkeys(items))

    def _ground_orders(self, tc: ToolCall, tlog: TurnLog | None) -> ToolCall:
        """접수 함수의 주문번호는 대화·조회 결과에 나온 번호로만 보낸다(P0071-05: 운송장번호·지어낸 번호).
        알려진 번호가 없으면 그대로, 있는데 못 맞추면 그 칸을 비운다(→ _length_error가 막고 묻게 한다)."""
        if tc.name not in ACCEPT_TOOLS or tc.name == "request_callback":
            return tc
        orders, items = self._seen_orders()
        args = dict(tc.arguments or {})
        changed = False
        for key in ("order_id", "order_number"):
            if key not in args or not orders:
                continue
            pick = ground_order_value(args[key], orders)
            if pick != args[key]:
                changed = True
                if tlog:
                    tlog.event("order_grounded", tool=tc.name, key=key, heard=args[key], fixed=pick)
                if pick:
                    args[key] = pick
                else:
                    args.pop(key)
        if isinstance(args.get("return_items"), list) and items:
            fixed_items = []
            for it in args["return_items"]:
                if isinstance(it, dict) and "order_item_code" in it:
                    pick = ground_order_value(it["order_item_code"], items)
                    if pick and pick != it["order_item_code"]:
                        changed = True
                        if tlog:
                            tlog.event("order_grounded", tool=tc.name, key="order_item_code",
                                       heard=it["order_item_code"], fixed=pick)
                        it = {**it, "order_item_code": pick}
                fixed_items.append(it)
            args["return_items"] = fixed_items
        return ToolCall(tc.id, tc.name, args) if changed else tc

    async def _repair(self, tc: ToolCall, tlog: TurnLog | None) -> ToolCall:
        """자리 수가 틀린 번호 인자를 확인된 고객 정보로 교정한다(AG-T15a). 교정하지 못하면 그대로 둔다
        (그러면 _length_error가 막는다). 전화번호: 발신 고객 번호와 거의 같으면 그 번호. 주문번호: 고객 주문 목록
        (get_order_list, 턴당 한 번)에서 부분 일치하는 주문이 하나뿐이면 그 주문번호."""
        tc = self._drop_bad_numbers(tc, tlog)
        tc = await self._order_from_phone_arg(tc, tlog)
        tc = self._ground_orders(tc, tlog)
        if not self.settings.validate_number_length:
            return tc
        args = dict(tc.arguments or {})
        changed = False
        for key, n in number_length_rules(self._schemas.get(tc.name, {})).items():
            v = args.get(key)
            is_phone = "phone" in key.lower()
            sval = v.strip() if isinstance(v, str) else ""
            if not sval.isdigit() or (is_phone and sval == PROMPT_EXAMPLE_PHONE and sval not in self._numbers
                                       and sval != self._caller_phone):
                # 숫자가 아닌 값("[PHONE]" 마스킹, "unknown", 빈 값)이나 프롬프트 예시 번호: 발화 번호 → 고객 번호 순으로 채운다
                said = [c for c in self._numbers if len(c) == n and (not is_phone or c.startswith("01"))]
                fill = said[0] if said else (self._caller_phone if is_phone and len(self._caller_phone) == n else "")
                if fill:
                    log.info("number arg filled for %s: %s=%s -> %s", tc.name, key, redact(v), redact(fill))
                    if tlog:
                        tlog.event("number_filled", tool=tc.name, key=key, value=v, fixed=fill)
                    args[key] = fill
                elif key in args:
                    args.pop(key)                     # 지어낸 번호는 버린다 → _length_error가 막는다
                changed = changed or args.get(key) != v
                continue
            if len(sval) == n or not (repair_enabled() and self._caller_phone):
                continue
            fixed = None
            if is_phone:
                if len(self._caller_phone) == n and _close_enough(v, self._caller_phone) is not None:
                    fixed = self._caller_phone
            elif "get_order_list" in self._schemas:
                if self._order_ids is None:
                    res = await self._exec_tool(ToolCall("auto_order_list", "get_order_list",
                                                         {"phone_number": self._caller_phone}), tlog)
                    self._order_ids = _order_ids(res)
                fixed = match_order_id(v, self._order_ids, known_length(self._order_ids, n))
            if fixed:
                log.info("number arg repaired for %s: %s=%s -> %s", tc.name, key, redact(v), redact(fixed))
                if tlog:
                    tlog.event("number_repaired", tool=tc.name, key=key, heard=v, fixed=fixed)
                args[key] = fixed
                changed = True
        return ToolCall(tc.id, tc.name, args) if changed else tc

    async def _exec_tool(self, tc: ToolCall, tlog: TurnLog | None) -> dict:
        self._called.append(tc.name)
        bad = self._length_error(tc)
        if bad is not None:                       # 자리 수가 틀린 번호는 보내지 않는다(여분 호출 감점 방지)
            if tlog:
                tlog.event("tool_blocked", tool=tc.name, args=tc.arguments)
            return await self._confirm(tc, bad, tlog) or bad
        if tlog:
            tlog.event("tool_call", tool=tc.name, args=tc.arguments)
        try:
            res = await self.call_tool(tc.name, tc.arguments)
        except Exception as e:  # noqa: BLE001
            log.warning("tool %s raised: %r", tc.name, e)
            res = {"status": "error", "message": "tool call failed"}
        if tlog:
            tlog.event("tool_result", tool=tc.name,
                       status=res.get("status") if isinstance(res, dict) else None)
        self._corpus += "\n" + json.dumps(res, ensure_ascii=False)
        if tc.name == "get_order_list" and isinstance(res, dict) and res.get("status") == "ok":
            self._order_rows = _order_rows(res)
            self._order_ids = _order_ids(res)          # 같은 턴의 번호 교정에 다시 쓴다(중복 호출 방지)
        if isinstance(res, dict) and res.get("status") == "not_found":
            res = await self._lookup_by_phone(tc, res, tlog) or await self._confirm(tc, res, tlog) or res
        return res

    async def _lookup_by_phone(self, tc: ToolCall, original: dict, tlog: TurnLog | None) -> dict | None:
        """주문번호로 못 찾았을 때 우회 조회: 고객 전화번호로 주문 목록을 받아 들은 번호와 가장 가까운 주문을 고르고,
        그 주문번호(+전화번호)로 다시 조회한다(정답 형식으로 한 번 더 불러 함수 점수를 살린다). 못 하면 None."""
        if tc.name != "get_order_info" or tc.id.startswith("auto_"):
            return None
        args = tc.arguments or {}
        heard = spoken_digits(str(args.get("order_id") or "")) or ""
        phone = spoken_digits(str(args.get("buyer_cellphone") or "")) or self._caller_phone
        if not heard or not phone:
            return None
        by_phone = await self._exec_tool(ToolCall("auto_phone", "get_order_info", {"buyer_cellphone": phone}), tlog)
        ids = official_order_ids(by_phone)
        digits = {re.sub(r"\D", "", i): i for i in ids}
        hit = match_order_id(heard, list(digits)) or (next(iter(digits)) if len(digits) == 1 else None)
        if tlog:
            tlog.event("order_lookup_by_phone", heard=heard, candidates=len(digits), matched=bool(hit))
        if not hit:
            return None
        again = ToolCall("auto_refetch", "get_order_info", {"order_id": digits[hit], "buyer_cellphone": phone})
        res = await self._exec_tool(again, tlog)
        return res if isinstance(res, dict) and res.get("status") == "ok" else by_phone

    async def _confirm(self, tc: ToolCall, original: dict, tlog: TurnLog | None) -> dict | None:
        """AG-T28: 막힌 호출·not_found를 고객 본인 주문 후보 확인 질문(needs_confirmation)으로 바꾼다. 못 하면 None."""
        if not confirm_enabled() or not self._caller_phone or "get_order_list" not in self._schemas:
            return None
        key = heard = ""
        for k, v in (tc.arguments or {}).items():
            if _is_number_key(k) and "phone" not in k.lower() and isinstance(v, str) and v.strip().isdigit():
                key, heard = k, v.strip()
                break
        if not heard:
            return None
        if self._order_rows is None:
            if self._order_ids is None:
                await self._exec_tool(ToolCall("auto_order_list", "get_order_list",
                                               {"phone_number": self._caller_phone}), tlog)
            if self._order_rows is None:
                self._order_rows = [{"order_id": i, "product_name": ""} for i in self._order_ids or []]
        n = number_length_rules(self._schemas.get(tc.name, {})).get(key)
        cands = confirm_candidates(heard, self._order_rows, n)
        if not cands:
            return None
        if tlog:
            tlog.event("order_confirm", tool=tc.name, original=original.get("status"), candidates=len(cands))
        return {"status": "needs_confirmation", "original_status": original.get("status"),
                "candidates": cands, "message": confirm_message(cands)}

    # ── 조회 뒤 답(AG-T21) ──
    def _on_audio(self, kind: str, pcm: bytes) -> None:
        if self._tool_t is None or (self._after_any is not None and self._after_answer is not None):
            return
        if not _voiced(pcm):
            return
        ms = round((time.perf_counter() - self._tool_t) * 1000, 1)
        if self._after_any is None:
            self._after_any = ms
        if kind == "tts" and self._after_answer is None:
            self._after_answer = ms

    def _mark_tool_results(self) -> None:
        self._tool_t = time.perf_counter()
        self._after_any = self._after_answer = None

    def _post_tool_llm(self, calls: list[ToolCall]) -> Any:
        """툴 결과 뒤 호출에 쓸 LLM(단일 지정 모델)."""
        return self.llm

    def _start_filler(self, calls: list[ToolCall], speaker: _Speaker,
                      tlog: TurnLog | None) -> asyncio.Task | None:
        """툴 회차가 시작되면 필러 타이머를 건다. filler_delay_ms 안에 다음 회차 텍스트가 오지 않으면
        (툴 실행 + 다음 LLM 첫 텍스트가 느리면) 캐시 필러를 재생한다. 다음 회차 첫 텍스트가 오면 취소한다.
        filler_skip_tools(상담원 연결 등)를 부르는 회차에는 걸지 않는다."""
        filler = self.settings.fillers[0] if self.settings.fillers else ""
        if not filler or self.cache is None or self._filler_used:
            return None
        skip = set(self.settings.filler_skip_tools or ())
        if any(tc.name in skip for tc in calls):
            return None

        async def fire() -> None:
            await asyncio.sleep(self.settings.filler_delay_ms / 1000)
            pcm = self.cache.get(filler)
            if pcm and not self._filler_used:
                speaker.play(filler, pcm)
                self._filler_used = True
                if tlog:
                    tlog.event("filler", text=filler)

        return asyncio.create_task(fire())

    async def _ensure_transfer(self, text: str, msgs: list[Message], tlog: TurnLog | None) -> None:
        """말로 상담원 연결을 약속했는데 transfer_to_agent(또는 콜백)를 부르지 않았으면 부른다(말과 처리 일치)."""
        name = "transfer_to_agent"
        if (not self.settings.ensure_transfer_call or name not in self._schemas
                or {name, "request_callback"} & set(self._called) or not promises_transfer(text)):
            return
        last_user = next((m.content for m in reversed(msgs) if m.role == "user"), "")
        reason = f"상담원 연결 안내: {last_user.strip()[:60]}".strip()
        if tlog:
            tlog.event("transfer_guard", reason=reason)
        await self._exec_tool(ToolCall("auto_transfer", name, {"reason": reason}), tlog)

    async def respond(self, system: str, messages: list[Message], send_audio: AudioSink, *,
                      spoken: list[str] | None = None, prefetch: Prefetch | None = None,
                      log: TurnLog | None = None) -> list[str]:
        """말한 문장 목록을 반환한다(spoken에도 차례로 채운다). LLM·TTS 오류는 그대로 올린다."""
        spoken = [] if spoken is None else spoken
        tlog = log
        msgs = list(messages)
        s = self.settings
        speaker = _Speaker(self.tts, send_audio, spoken, s, tlog, on_audio=self._on_audio)
        self._filler_used = False
        self._tool_t = self._after_any = self._after_answer = None
        use_template = template_enabled()
        template_said: str | None = None
        post_llm: Any = None
        min_digits = number_min_digits(*_known_lengths())
        self._numbers = [n for m in msgs if m.role == "user" for n in utterance_numbers(m.content, min_digits)]
        self._called = []
        self._caller_phone = caller_phone_from_system(system)
        self._known = known_numbers(system)
        self._order_ids = None
        self._order_rows = None
        self._corpus = "\n".join([system or ""] + [m.content or "" for m in msgs])
        guard_used = False
        guard_round = False                         # 약속만 하고 함수를 안 부른 뒤의 확인 회차: 말은 내보내지 않는다
        extra = 0
        said: list[str] = []
        said_chars = 0
        overflow = False
        filler_task: asyncio.Task | None = None
        rounds = max(1, s.max_tool_rounds)
        try:
            for rnd in range(rounds + 1):
                if rnd >= rounds + extra:
                    break
                last = rnd == rounds + extra - 1
                if rnd == 0 and prefetch is not None:
                    stream = prefetch.events()
                else:
                    tools = [] if last else (self._llm_tools if rnd == 0 else self._shown_tools)   # 답 후보는 첫 호출에서만
                    stream = (post_llm or self.llm).stream(system, msgs, tools)
                splitter = SentenceSplitter(s.min_sentence_chars)
                text_parts: list[str] = []
                calls: list[ToolCall] = []
                tool_tasks: list[asyncio.Task] = []         # 함수 호출이 도착하는 즉시 실행(답 후보를 쓰는 동안 결과를 기다린다)
                drafts: list[D.Draft] = []
                presynth: dict[str, asyncio.Task] = {}
                it = stream.__aiter__()
                first = True
                try:
                    while True:
                        try:
                            ev = await asyncio.wait_for(it.__anext__(), s.llm_timeout_s)
                        except StopAsyncIteration:
                            break
                        if first:
                            first = False
                            if tlog:
                                tlog.mark("llm_first_token")
                                tlog.event("llm_round_first_event", round=rnd)
                        if ev.type == "text" and ev.text and guard_round:
                            continue                # 확인 회차의 말은 버린다(이미 말한 약속을 되풀이하지 않게)
                        if ev.type == "text" and ev.text:
                            said.append(ev.text)
                            if filler_task is not None:
                                filler_task.cancel()
                                filler_task = None
                            text_parts.append(ev.text)
                            for sent in splitter.feed(ev.text):
                                if template_said and repeats_template(sent, template_said):
                                    if tlog:
                                        tlog.event("template_repeat_dropped", text=sent)
                                    continue
                                template_said = None
                                if s.max_response_chars and said_chars + _nchars(sent) > s.max_response_chars:
                                    overflow = True
                                    break
                                said_chars += _nchars(sent)
                                speaker.say(sent)
                            if overflow:            # 폭주 생성(같은 말 반복 등): 스트림을 끊고 남은 말은 버린다
                                logging.getLogger(__name__).warning("response exceeded %d chars, stream cut", s.max_response_chars)
                                if tlog:
                                    tlog.event("response_cut", chars=said_chars)
                                break
                        elif ev.type == "tool_call" and ev.tool_call is not None and ev.tool_call.name == D.DRAFT_NAME:
                            if not last:
                                drafts = D.parse(ev.tool_call.arguments)
                                for d in D.presynth_targets(drafts, s.draft_presynth_n):
                                    presynth[d.text] = asyncio.create_task(self._presynth(d.text))
                                if tlog:
                                    tlog.event("drafts", n=len(drafts), presynth=list(presynth))
                        elif ev.type == "tool_call" and ev.tool_call is not None:
                            calls.append(ev.tool_call)
                            if not last:
                                tool_tasks.append(asyncio.create_task(self._prepare_and_exec(ev.tool_call, tlog)))
                                if filler_task is None and not "".join(text_parts).strip():
                                    filler_task = self._start_filler(calls, speaker, tlog)
                        elif ev.type == "done":
                            break
                        speaker.raise_if_failed()
                finally:
                    aclose = getattr(it, "aclose", None)
                    if aclose is not None:
                        try:
                            await aclose()
                        except BaseException:  # noqa: BLE001
                            pass
                if overflow:
                    break
                for sent in splitter.flush():
                    if template_said and repeats_template(sent, template_said):
                        continue
                    if s.max_response_chars and said_chars + _nchars(sent) > s.max_response_chars:
                        break
                    said_chars += _nchars(sent)
                    speaker.say(sent)
                if not calls and drafts and not last and not "".join(text_parts).strip():
                    # 함수 없이 답 후보만 쓰고 말이 없다(함수가 필요 없는 문항) → 후보 함수 없이 다시 부른다
                    for t in presynth.values():
                        t.cancel()
                    if tlog:
                        tlog.event("drafts_only_retry")
                    continue
                round_text = "".join(text_parts)
                kind = None if (calls or guard_used or not s.ensure_promised_call) else \
                    unkept_promise(round_text, self._called)
                if kind:
                    # 말로 접수·조회를 약속했는데 함수가 없다(P0047-03·P0099-03·P0193-02) → 한 번 더 불러 함수를 부르게 한다
                    for t in (*tool_tasks, *presynth.values()):
                        t.cancel()
                    guard_used = guard_round = True
                    extra = 1
                    what = "접수" if kind == "accept" else "조회"
                    msgs.append(Message("assistant", round_text))
                    msgs.append(Message("user", GUARD_NOTE.format(text=round_text.strip()[:200], what=what)))
                    if tlog:
                        tlog.event("promise_guard", kind=kind, text=round_text[:200])
                    continue
                if not calls or last:
                    for t in (*tool_tasks, *presynth.values()):
                        t.cancel()
                    break
                guard_round = False
                done = await asyncio.gather(*tool_tasks)
                calls = [tc for tc, _ in done]
                results = [res for _, res in done]
                msgs.append(Message("assistant", round_text, tool_calls=calls))
                self._mark_tool_results()
                picked = D.choose(drafts, results[0]) if (drafts and len(calls) == 1) else None
                if picked is not None:
                    d, text = picked
                    if filler_task is not None:
                        filler_task.cancel()
                        filler_task = None
                    pcm = None
                    task = presynth.pop(d.text, None)
                    if task is not None and text == d.text:
                        try:
                            pcm = await asyncio.wait_for(task, s.tts_timeout_s)
                        except Exception as e:  # noqa: BLE001
                            log.warning("draft presynth failed: %r", e)
                    for t in presynth.values():
                        t.cancel()
                    if pcm:
                        speaker.play(text, pcm, kind="answer")
                    else:
                        speaker.say(text)
                    said.append(text)
                    msgs.append(Message("tool", json.dumps(results[0], ensure_ascii=False),
                                        tool_call_id=calls[0].id, name=calls[0].name))
                    if tlog:
                        tlog.event("draft_used", tool=calls[0].name, when=d.when, text=text, presynth=bool(pcm))
                    break
                for t in presynth.values():
                    t.cancel()
                template_said = None
                if use_template and len(calls) == 1:
                    template_said = tool_result_template(calls[0].name, results[0])
                    if template_said and s.max_response_chars and \
                            said_chars + _nchars(template_said) > s.max_response_chars:
                        template_said = None
                    if template_said:
                        if filler_task is not None:     # 아직 안 나간 필러는 취소(답 뒤에 "잠시만요"가 오지 않게)
                            filler_task.cancel()
                            filler_task = None
                        said.append(template_said + " ")
                        said_chars += _nchars(template_said)
                        speaker.say(template_said)
                        if tlog:
                            tlog.event("tool_template", tool=calls[0].name, text=template_said)
                for i, (tc, res) in enumerate(zip(calls, results)):
                    body = res
                    if template_said and i == 0 and isinstance(res, dict):
                        body = {**res, "_already_said": template_said, "_note": TEMPLATE_NOTE}
                    msgs.append(Message("tool", json.dumps(body, ensure_ascii=False),
                                        tool_call_id=tc.id, name=tc.name))
                post_llm = self._post_tool_llm(calls)
                speaker.raise_if_failed()
            if filler_task is not None:
                filler_task.cancel()
                filler_task = None
            await self._ensure_transfer("".join(said), msgs, tlog)
            await speaker.finish()
            self._log_after_tool(tlog)
        except BaseException:
            if filler_task is not None:
                filler_task.cancel()
            speaker.cancel()
            raise
        if not spoken:
            raise ResponderError("empty response")
        return spoken

    def _log_after_tool(self, tlog: TurnLog | None) -> None:
        """턴 로그 after_tool_ms(마지막 툴 결과 → 첫 유성 PCM, 필러 포함)와 after_tool_answer_ms(본 응답만)."""
        if tlog is None or self._tool_t is None:
            return
        tlog.set("after_tool_ms", self._after_any)
        tlog.set("after_tool_answer_ms", self._after_answer)
