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
from agent.normalize import spoken_digits
from agent.providers._privacy import redact  # 고객 번호는 저장 금지 모드에서 로그에 남기지 않는다(AG-T18)
from agent.tool_guard import SpeechGuard, fix_full_returns  # AG-T45 함수 호출 안전장치
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

    def play(self, text: str, pcm: bytes) -> None:
        self.q.put_nowait(("pcm", text, pcm))

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
            if kind == "pcm":
                self.spoken.append(text)
                self._notify(kind, payload)
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


def _is_number_key(key: str) -> bool:
    k = key.lower()
    return "phone" in k or k == "id" or k.endswith("_id") or k == "order_number"


def _is_phone_key(key: str) -> bool:
    return "phone" in key.lower()


def _num(v: Any) -> str:
    """번호 값의 숫자열. 숫자·공백·하이픈만으로 된 값이 아니면 ""(마스킹·"unknown"·None)."""
    if isinstance(v, int) and not isinstance(v, bool):
        v = str(v)
    if not isinstance(v, str) or not re.fullmatch(r"[\d\s\-]*\d[\d\s\-]*", v.strip()):
        return ""
    return re.sub(r"\D", "", v)


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


def _order_ids(result: Any) -> list[str]:
    return [r["order_id"] for r in _order_rows(result)]


def _order_rows(result: Any) -> list[dict]:
    """고객 주문 목록 결과 → [{"order_id"(숫자열), "raw"(원래 표기), "product_name"}].
    공식 get_order_info는 한글 키(주문번호·상품명), 옛 get_order_list는 order_id·product_name."""
    data = result.get("data") if isinstance(result, dict) else None
    orders = (data or {}).get("orders") if isinstance(data, dict) else None
    out = []
    for o in orders or []:
        if not isinstance(o, dict):
            continue
        raw = o.get("주문번호", o.get("order_id"))
        oid = _num(raw) or (str(raw).strip() if raw else "")
        if not oid or any(r["order_id"] == oid for r in out):
            continue
        name = o.get("상품명", o.get("product_name")) or ""
        out.append({"order_id": oid, "raw": str(raw).strip(), "product_name": str(name).strip()})
    return out


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


# ── 번호 후보 확신도(AG-T43) ─────────────────────────────
# 시스템 프롬프트의 '번호 후보' 줄(agent.prompts.candidates_line: 발신번호, 이전 대화의 번호·조회 결과 주문번호)과
# 이번 턴에 받은 고객 주문을 후보로, 들은 번호를 강함(후보로 교체)·약함(그 후보 끝자리로 확인)·없음으로 나눈다.
# 강함 = 기존 _close_enough(공통 부분열) + 1·2등 차이. 약함 = 형식이 같은 앞자리(010·주문 날짜)를 빼려고 끝 8자리만 비교해
# 공통 부분열이 전화 5개·주문 6개 이상(무작위 번호 오탐 전화 4.5%·주문 8.7%, 실제 번호 검출 99%·97%, 시뮬레이션).
# 끄기: DIRECTON_NUMBER_CANDIDATES=0 (프롬프트 줄·이전 대화 후보·약한 확인. 발신번호 교정은 DIRECTON_REPAIR_ORDER_ID).
NUMBER_CANDIDATES_ENV = "DIRECTON_NUMBER_CANDIDATES"
WEAK_TAIL_DIGITS = 8
WEAK_TAIL_RATIO = {"phone": 0.6, "order": 0.75}     # 끝 8자리 기준 전화 5개, 주문 6개
WEAK_MAX_CANDIDATES = 2
_CAND_LINE = re.compile(r"^- 번호 후보: (.*)$", re.M)
_CAND_ITEM = re.compile(r"(전화번호|주문번호) (\d[\d\-]*\d)(?:\(([^()]*)\))?")


def number_candidates_enabled() -> bool:
    return _env_on(NUMBER_CANDIDATES_ENV, True)


def number_candidates_from_system(system: str) -> list[dict]:
    """시스템 프롬프트의 '번호 후보' 줄 → [{"kind", "number"(숫자열), "raw", "note"}]."""
    m = _CAND_LINE.search(system or "")
    if not m:
        return []
    body = m.group(1).split(". 음성 인식", 1)[0]
    out: list[dict] = []
    for kind, raw, note in _CAND_ITEM.findall(body):
        d = re.sub(r"\D", "", raw)
        if d and all(c["number"] != d for c in out):
            out.append({"kind": "phone" if kind == "전화번호" else "order", "number": d, "raw": raw, "note": note or ""})
    return out


def _tail_score(heard: str, number: str) -> int:
    t = min(WEAK_TAIL_DIGITS, len(number))
    return lcs_len(heard[-(t + 3):], number[-t:])


def classify_number(heard: str, cands: list[dict]) -> tuple[str, list[dict]]:
    """들은 번호의 확신도: ("exact"|"strong", [후보]), ("weak", [후보 1~2개]), ("none", [])."""
    rows = [c for c in cands if c.get("number")]
    if not heard or not heard.isdigit() or not rows:
        return "none", []
    for c in rows:
        if c["number"] == heard:
            return "exact", [c]
    margin = repair_thresholds(known_length([c["number"] for c in rows]))[3]
    scored = sorted(((lcs_len(heard, c["number"]), i) for i, c in enumerate(rows)), key=lambda x: (-x[0], x[1]))
    top, bi = scored[0]
    if _close_enough(heard, rows[bi]["number"]) is not None and (len(scored) == 1 or top - scored[1][0] >= margin):
        return "strong", [rows[bi]]
    ws = sorted(((_tail_score(heard, c["number"]), i) for i, c in enumerate(rows)), key=lambda x: (-x[0], x[1]))
    best, wi = ws[0]
    lead = rows[wi]
    need = math.ceil(min(WEAK_TAIL_DIGITS, len(lead["number"])) * WEAK_TAIL_RATIO.get(lead.get("kind"), 0.75) - 1e-9)
    tied = [rows[i] for s, i in ws if s == best]
    if best < need or len(tied) > WEAK_MAX_CANDIDATES:
        return "none", []
    return "weak", tied


def _cand_desc(c: dict) -> str:
    head = f"끝자리 {last_digits(c['number'])}"
    if c.get("kind") == "phone":
        return f"{head} 번호" + (f"({c['note']})" if c.get("note") else "")
    return f"{head} {c.get('note') or ''} 주문".replace("  ", " ")


def candidate_confirm_message(cands: list[dict]) -> str:
    descs = [_cand_desc(c) for c in cands]
    if len(descs) == 1:
        found = f"이미 아는 번호 중 가장 가까운 것은 '{descs[0]}'입니다."
        ask = f"'{descs[0]} 맞으실까요?'처럼 한 문장으로 확인만 한다."
    else:
        found = "이미 아는 번호 중 가까운 것은 " + ", ".join(f"'{d}'" for d in descs) + "입니다."
        ask = f"'{descs[0]}과 {descs[1]} 중 어느 쪽일까요?'처럼 묻는다."
    return ("고객이 말한 번호가 정확히 들리지 않았습니다. " + found + " " + ask + " 번호 전체는 읽지 않는다. 고객이 맞다고 하면 "
            "candidates의 number를 그대로 쓴다. 주문 상태나 결과는 지어내지 않고, 고객이 확인하기 전에는 다른 함수를 부르지 않는다.")


_LEN_RULE = re.compile(r"(\d+)\s*자리")


def number_formats(schema: dict) -> dict[str, frozenset[int]]:
    """번호 인자 → 허용 자리 수. 속성 설명의 'N자리'가 있으면 그 값, 없으면 알려진 형식(AG-T38: 공식 함수 설명에는
    자리 수가 없다) — 전화번호 키는 전화번호 길이, 주문번호 키(order_id·order_number)는 주문번호 길이(공식 15·16)."""
    out: dict[str, frozenset[int]] = {}
    order_lens, phone_lens = _known_lengths()
    for key, prop in ((schema or {}).get("properties") or {}).items():
        if not _is_number_key(key) or not isinstance(prop, dict):
            continue
        m = _LEN_RULE.search(str(prop.get("description") or ""))
        if m:
            out[key] = frozenset({int(m.group(1))})
        elif _is_phone_key(key):
            out[key] = phone_lens
        elif key.lower() in ("order_id", "order_number"):
            out[key] = order_lens
    return out


def _all_numbers_empty(args: dict | None, schema: dict) -> bool:
    """AG-T49: 번호 인자가 모두 선택 인자(get_order_info)이고 하나도 채워지지 않았다."""
    formats = number_formats(schema)
    required = set((schema or {}).get("required") or [])
    return bool(formats) and not (required & set(formats)) and not any(_num((args or {}).get(k)) for k in formats)


def _drop_wrong_optional(args: dict, schema: dict) -> list[str]:
    """AG-T49: 자리 수가 맞는 번호 인자가 하나라도 있으면 자리 수가 틀린 선택 번호 인자는 None으로 바꾼다(제자리). 바꾼 키."""
    formats = number_formats(schema)
    required = set((schema or {}).get("required") or [])
    if not any(_num(args.get(k)) and len(_num(args.get(k))) in lens for k, lens in formats.items()):
        return []
    bad = [k for k, lens in formats.items()
           if k not in required and _num(args.get(k)) and len(_num(args.get(k))) not in lens]
    for k in bad:
        args[k] = None
    return bad


def _has_rows(res: Any) -> bool:
    """조회 결과에 내용이 있다(ok + data의 목록 중 비지 않은 것, 목록이 없으면 data가 비지 않음)."""
    if not isinstance(res, dict) or res.get("status") != "ok":
        return False
    data = res.get("data")
    if not isinstance(data, dict):
        return bool(data)
    lists = [v for v in data.values() if isinstance(v, list)]
    return any(lists) if lists else bool(data)


_ASKS = re.compile(r"[?？]|시겠어요|실까요|을까요|할까요")
SKIPPED_RESULT = {"status": "skipped",
                  "message": "같은 회차에 다른 번호로 같은 조회를 했으니 이 호출은 보내지 않았다. 그 조회 결과로 답하고 "
                             "번호를 다시 묻지 않는다."}


def _lengths_ko(lens) -> str:
    return "나 ".join(f"{n}자리" for n in sorted(lens))


# ── 조회 뒤 답까지 시간 단축(AG-T21) ───────────────────────
# 공식 속도 점수는 함수 문항에서 "말 끝 → 첫 소리"와 "함수 결과 → 실제 답"의 평균이다. 툴 결과 뒤에는 LLM을 다시 불러야 해서
# 답까지 1~3초가 걸린다(docs/decisions/after_tool_latency.md). 두 가지로 줄인다. config.py는 다른 Task가 작업 중이라 환경변수로 둔다.
# 1) 후속 hedge(DIRECTON_POST_TOOL_HEDGE, 기본 1): 툴 결과 뒤 호출도 첫 호출처럼 동시 요청한다. HedgedLLM은 툴 후속 라운드를
#    툴을 낸 모델에 고정하고 2.5초 첫 이벤트 제한 뒤에야 넘기므로, 응답기에서 같은 체인으로 새 HedgedLLM을 만들어(원래 모델을
#    앞에) 지연 뒤 대체 모델에도 보낸다. 대체 모델은 서명이 없는 호출 id에 공식 우회 서명을 붙인다(GeminiLLM).
# 2) 결과 템플릿(DIRECTON_TOOL_TEMPLATE, 코드 기본 0, 실측상 켜기 권장 — docs/decisions/after_tool_latency.md): 툴 결과에 있는 사실만으로 짧은 첫 문장을 만들어 LLM보다 먼저 말한다.
#    결과의 고객용 message 첫 문장, 없으면 조회 함수의 status("현재 배송중 상태입니다."). LLM에는 이미 말한 문장을 알려 반복을 막는다.
POST_HEDGE_ENV = "DIRECTON_POST_TOOL_HEDGE"
TEMPLATE_ENV = "DIRECTON_TOOL_TEMPLATE"
VOICED_RMS = 300.0                     # 측정기(eval.runner)와 같은 유성음 기준
TEMPLATE_STATUSES = ("ok", "not_allowed", "no_refund")   # 그 밖(invalid_arguments·error 등)은 말하지 않는다
TEMPLATE_MAX_CHARS = 40
_READ_ONLY_PREFIXES = ("get_", "track_", "check_", "search_", "find_", "lookup_")
_STATUS_KEYS = ("status", "refund_status", "delivery_status", "order_status", "배송상태", "주문상태", "상태")
_INSTRUCTION_HINTS = ("고객에게", "안내하", "상담원은", "상담사는", "말씀드리", "응답하", "LLM", "시스템")
_SENT_END = re.compile(r"(?<=[.?!。])\s+")
TEMPLATE_NOTE = ("상담원이 이 결과를 받자마자 고객에게 _already_said 문장을 이미 말했다. 그 문장을 되풀이하지 말고 "
                 "그 뒤에 이어질 말만 한다.")


def _env_on(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def post_hedge_enabled() -> bool:
    return _env_on(POST_HEDGE_ENV, True)


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
    orders = data.get("orders")
    if isinstance(orders, list):                       # 공식 get_order_info: 주문 하나일 때만 그 주문의 상태
        if len(orders) != 1 or not isinstance(orders[0], dict):
            return None
        data = orders[0]
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


def _model_name(llm: Any) -> str:
    return str(getattr(llm, "model", None) or type(llm).__name__)     # fallback_llm._name과 같은 규칙(고정 기록 키)


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
        self._numbers: list[str] = []
        self._called: list[str] = []
        self._caller_phone = ""
        self._sys_cands: list[dict] = []
        self._order_ids: list[str] | None = None
        self._order_rows: list[dict] | None = None
        self._caller_orders_res: Any = None
        self._post_hedger: Any = None
        self._tool_t: float | None = None           # 마지막 툴 결과 수신 시각(perf_counter)
        self._after_any: float | None = None        # 그 뒤 첫 유성 PCM 송신까지(ms, 필러 포함)
        self._after_answer: float | None = None     # 그 뒤 본 응답(LLM·템플릿 문장) 첫 유성 PCM까지(ms)

    def prefetch(self, system: str, messages: list[Message]) -> Prefetch:
        return Prefetch(self.llm.stream(system, list(messages), self.tools))

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
        return ToolCall(tc.id, tc.name, args)

    def _length_error(self, tc: ToolCall) -> dict | None:
        """자리 수가 틀린 번호 인자면 막는 결과(invalid_arguments). 하이픈·공백은 무시하고 숫자만 센다."""
        if not self.settings.validate_number_length:
            return None
        schema = self._schemas.get(tc.name, {}) or {}
        required = set(schema.get("required") or [])
        if _all_numbers_empty(tc.arguments, schema):  # AG-T49: 번호 인자가 모두 빈 조회(get_order_info {})는 보내지 않는다
            return {"status": "invalid_arguments",
                    "message": "조회에 넣을 번호를 받지 못했습니다. 고객이 아직 번호를 말하지 않았으면 '주문번호나 주문하신 분 "
                               "연락처를 말씀해 주시겠어요?'처럼 한 번만 짧게 묻는다(처음 묻는 것이라 천천히 다시 말해 달라고 "
                               "하지 않고, 이미 물었으면 같은 질문을 되풀이하지 않는다). 처리에 필요한 다른 정보가 있으면 함께 묻는다."}
        for key, lens in number_formats(schema).items():
            v = (tc.arguments or {}).get(key)
            d = _num(v)
            if key in required and not d:
                what = "휴대폰 번호" if _is_phone_key(key) else "주문번호"
                return {"status": "invalid_arguments",
                        "message": f"{key}에 넣을 번호를 받지 못했습니다. 고객이 아직 번호를 말하지 않았으면 "
                                   f"'{what}를 말씀해 주시겠어요?'처럼 짧게 묻는다(처음 묻는 것이라 천천히 다시 말해 달라고 "
                                   "하지 않는다). 처리에 필요한 다른 정보가 있으면 함께 묻는다."}
            if d and len(d) not in lens:
                what = (f"전화번호는 010으로 시작하는 {_lengths_ko(lens)}" if _is_phone_key(key)
                        else f"주문번호는 {_lengths_ko(lens)}")
                return {"status": "invalid_arguments",
                        "message": f"{key}는 숫자 {_lengths_ko(lens)}여야 하는데 {len(d)}자리로 들렸습니다(참고: {what}). "
                                   "음성 인식이 번호를 잘못 받아 적은 것이니 자리 수 규칙은 고객에게 설명하지 않는다. "
                                   "'번호가 정확히 들리지 않았어요. 번호를 한 자리씩 천천히 다시 말씀해 주시겠어요?'처럼 "
                                   "짧게 다시 묻는다. 고객이 이번 말에서 번호 없이 브랜드 정보로 답할 수 있는 정책(가능 조건, "
                                   "기간, 비용 등)을 직접 물었을 때만 그 답을 한 문장으로 먼저 하고, 묻지 않은 안내는 "
                                   "덧붙이지 않는다. 주문 상태나 결과는 지어내지 않는다."}
        return None

    def _orders_tool(self) -> tuple[str, str] | None:
        """발신 고객 주문 목록을 받는 함수와 전화번호 인자. 공식: get_order_info(buyer_cellphone), 옛: get_order_list."""
        props = (self._schemas.get("get_order_info") or {}).get("properties") or {}
        if "buyer_cellphone" in props:
            return "get_order_info", "buyer_cellphone"
        if "get_order_list" in self._schemas:
            return "get_order_list", "phone_number"
        return None

    async def _load_caller_orders(self, tlog: TurnLog | None) -> None:
        """발신 고객 주문 목록(턴당 한 번). 조회를 더 부르는 것은 감점이 없다(SCORING.md)."""
        if self._order_ids is not None:
            return
        spec = self._orders_tool()
        if spec is None or not self._caller_phone:
            self._order_ids = []
            return
        name, key = spec
        await self._exec_tool(ToolCall("auto_order_list", name, {key: self._caller_phone}), tlog)
        if self._order_ids is None:
            self._order_ids = []

    async def _repair(self, tc: ToolCall, tlog: TurnLog | None) -> ToolCall:
        """자리 수가 틀린 번호 인자를 확인된 고객 정보로 교정한다(AG-T15a). 교정하지 못하면 그대로 둔다
        (그러면 _length_error가 막는다). 전화번호: 발신 고객 번호와 거의 같으면 그 번호. 주문번호: 발신 고객 주문 목록
        (공식 get_order_info(buyer_cellphone), 턴당 한 번)에서 부분 일치하는 주문이 하나뿐이면 그 주문번호."""
        if not self.settings.validate_number_length:
            return tc
        args = dict(tc.arguments or {})
        changed = False
        schema = self._schemas.get(tc.name, {}) or {}
        required = set(schema.get("required") or [])
        if _all_numbers_empty(args, schema) and tc.name.startswith(_READ_ONLY_PREFIXES):
            # AG-T49: 번호 없는 조회(get_order_info {})는 발신번호로 채운다. 없으면 _length_error가 막는다
            key = next((k for k, lens in number_formats(schema).items()
                        if _is_phone_key(k) and len(self._caller_phone) in lens), None)
            if key:
                if tlog:
                    tlog.event("number_filled", tool=tc.name, key=key, value=args.get(key), fixed=self._caller_phone)
                args[key] = self._caller_phone
                return ToolCall(tc.id, tc.name, args)
        for key, lens in number_formats(schema).items():
            v = args.get(key)
            if (v is None or (isinstance(v, str) and not v.strip())) and key not in required:
                continue                              # 선택 인자(get_order_info의 null)는 그대로 둔다
            is_phone = _is_phone_key(key)
            d = _num(v)
            if not d or (is_phone and d == PROMPT_EXAMPLE_PHONE and d not in self._numbers
                         and d != self._caller_phone):
                # 숫자가 아닌 값("[PHONE]" 마스킹, "unknown", 빈 값)이나 프롬프트 예시 번호: 발화 번호 → 고객 번호 순으로 채운다
                said = [c for c in self._numbers if len(c) in lens and (not is_phone or c.startswith("01"))]
                fill = said[0] if said else (self._caller_phone if is_phone and len(self._caller_phone) in lens else "")
                if fill:
                    log.info("number arg filled for %s: %s=%s -> %s", tc.name, key, redact(v), redact(fill))
                    if tlog:
                        tlog.event("number_filled", tool=tc.name, key=key, value=v, fixed=fill)
                    args[key] = fill
                elif key in args:
                    args.pop(key)                     # 지어낸 번호는 버린다 → _length_error가 막는다
                changed = changed or args.get(key) != v
                continue
            if not repair_enabled():
                continue
            fixed = None
            if len(d) in lens:
                # AG-T41·T43: 자리 수는 맞아도 이미 아는 번호(발신번호·번호 후보)와 강하게 맞으면 STT 오인식으로 본다
                level, hit = classify_number(d, self._pool(is_phone, lens))
                if level == "strong":
                    fixed = hit[0]["raw"]
            else:
                if is_phone:
                    if len(self._caller_phone) in lens and _close_enough(d, self._caller_phone) is not None:
                        fixed = self._caller_phone
                elif self._orders_tool() is not None and self._caller_phone:
                    await self._load_caller_orders(tlog)
                    ids = self._order_ids or []
                    hit = match_order_id(d, ids, known_length(ids, max(lens)))
                    if hit:
                        fixed = next((r.get("raw") or hit for r in self._order_rows or [] if r["order_id"] == hit), hit)
                if not fixed:                         # AG-T43: 번호 후보와 강하게 맞으면 그 후보
                    level, hit = classify_number(d, self._pool(is_phone, lens))
                    if level == "strong":
                        fixed = hit[0]["raw"]
            if fixed:
                log.info("number arg repaired for %s: %s=%s -> %s", tc.name, key, redact(v), redact(fixed))
                if tlog:
                    tlog.event("number_repaired", tool=tc.name, key=key, heard=v, fixed=fixed)
                args[key] = fixed
                changed = True
        dropped = _drop_wrong_optional(args, schema)      # AG-T49: 다른 번호가 맞으면 틀린 선택 인자는 버린다
        if dropped:
            log.info("wrong-length optional number dropped for %s: %s", tc.name, dropped)
            if tlog:
                tlog.event("number_dropped", tool=tc.name, keys=dropped)
            changed = True
        return ToolCall(tc.id, tc.name, args) if changed else tc

    def _pool(self, is_phone: bool, lens=None) -> list[dict]:
        """AG-T43 번호 후보: 전화 = 발신번호 + 번호 후보 줄, 주문 = 번호 후보 줄 + 이번 턴에 받은 고객 주문. lens면 그 자리 수만."""
        kind = "phone" if is_phone else "order"
        out: list[dict] = []
        if is_phone and self._caller_phone:
            out.append({"kind": "phone", "number": self._caller_phone, "raw": self._caller_phone, "note": "발신번호"})
        if number_candidates_enabled():
            out += [c for c in self._sys_cands if c["kind"] == kind]
            if not is_phone:
                out += [{"kind": "order", "number": r["order_id"], "raw": r.get("raw") or r["order_id"],
                         "note": r.get("product_name") or ""} for r in self._order_rows or []]
        seen: set[str] = set()
        uniq = []
        for c in out:
            if c["number"] not in seen and (lens is None or len(c["number"]) in lens):
                seen.add(c["number"])
                uniq.append(c)
        return uniq

    async def _candidate_confirm(self, tc: ToolCall, original: dict, tlog: TurnLog | None) -> dict | None:
        """AG-T43: 막힌 호출(자리 수 틀림)·not_found의 번호가 이미 아는 번호와 약하게 맞으면 그 후보로 확인하는 결과
        (needs_confirmation)로 바꾼다. 못 하면 None."""
        if not (number_candidates_enabled() and repair_enabled()) or tc.id == "auto_order_list":
            return None
        args = tc.arguments or {}
        for key, lens in number_formats(self._schemas.get(tc.name, {}) or {}).items():
            d = _num(args.get(key))
            if not d or (original.get("status") == "invalid_arguments" and len(d) in lens):
                continue
            is_phone = _is_phone_key(key)
            if not is_phone and self._caller_phone and self._orders_tool() is not None:
                await self._load_caller_orders(tlog)
            level, cands = classify_number(d, self._pool(is_phone, lens))
            if level not in ("weak", "strong"):
                continue
            if tlog:
                tlog.event("number_confirm", tool=tc.name, key=key, original=original.get("status"), candidates=len(cands))
            return {"status": "needs_confirmation", "original_status": original.get("status"),
                    "candidates": [{"kind": c["kind"], "number": c["raw"], "last_digits": last_digits(c["number"]),
                                    "note": c.get("note") or ""} for c in cands],
                    "message": candidate_confirm_message(cands)}
        return None

    def _is_orders_list_call(self, tc: ToolCall) -> bool:
        spec = self._orders_tool()
        if spec is None or tc.name != spec[0]:
            return False
        args = tc.arguments or {}
        return bool(_num(args.get(spec[1]))) and not _num(args.get("order_id"))

    async def _exec_tool(self, tc: ToolCall, tlog: TurnLog | None) -> dict:
        self._called.append(tc.name)
        bad = self._length_error(tc)
        if bad is not None:                       # 자리 수가 틀린 번호는 보내지 않는다(잘못된 접수 방지, 빠른 되묻기)
            if tlog:
                tlog.event("tool_blocked", tool=tc.name, args=tc.arguments)
            return (await self._confirm(tc, bad, tlog) or await self._candidate_confirm(tc, bad, tlog)
                    or await self._caller_fallback(tc, bad, tlog) or bad)
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
        if self._is_orders_list_call(tc) and isinstance(res, dict) and res.get("status") == "ok":
            self._order_rows = _order_rows(res)
            self._order_ids = [r["order_id"] for r in self._order_rows]   # 같은 턴의 번호 교정에 다시 쓴다(중복 호출 방지)
            if self._caller_phone and _num((tc.arguments or {}).get(self._orders_tool()[1])) == self._caller_phone:
                self._caller_orders_res = res                                  # AG-T49 발신번호 대신 조회에 다시 쓴다
        if isinstance(res, dict) and res.get("status") == "not_found":
            res = await self._confirm(tc, res, tlog) or await self._candidate_confirm(tc, res, tlog) or res
        return res

    async def _confirm(self, tc: ToolCall, original: dict, tlog: TurnLog | None) -> dict | None:
        """AG-T28: 막힌 호출·not_found를 고객 본인 주문 후보 확인 질문(needs_confirmation)으로 바꾼다. 못 하면 None."""
        if not confirm_enabled() or not self._caller_phone or self._orders_tool() is None:
            return None
        if self._is_orders_list_call(tc):
            return None
        key = heard = ""
        for k, v in (tc.arguments or {}).items():
            if _is_number_key(k) and not _is_phone_key(k) and _num(v):
                key, heard = k, _num(v)
                break
        if not heard:
            return None
        if self._order_rows is None:
            await self._load_caller_orders(tlog)
            if self._order_rows is None:
                self._order_rows = [{"order_id": i, "raw": i, "product_name": ""} for i in self._order_ids or []]
        lens = number_formats(self._schemas.get(tc.name, {})).get(key)
        n = known_length([r["order_id"] for r in self._order_rows], max(lens)) if lens else None
        cands = confirm_candidates(heard, self._order_rows, n)
        if not cands:
            return None
        if tlog:
            tlog.event("order_confirm", tool=tc.name, original=original.get("status"), candidates=len(cands))
        return {"status": "needs_confirmation", "original_status": original.get("status"),
                "candidates": cands, "message": confirm_message(cands)}

    async def _caller_fallback(self, tc: ToolCall, original: dict, tlog: TurnLog | None) -> dict | None:
        """AG-T49: 번호가 막힌 조회(get_*)를 발신번호로 대신 조회한다. 결과가 있으면 그 결과(+ 다시 묻지 말라는 _note),
        없으면 원래 막힌 결과에 '발신번호로도 없었다'를 붙인다. 발신번호·전화번호 인자가 없으면 None."""
        if not self._caller_phone or tc.id == "auto_order_list" or not tc.name.startswith(_READ_ONLY_PREFIXES):
            return None
        schema = self._schemas.get(tc.name, {}) or {}
        formats = number_formats(schema)
        key = next((k for k, lens in formats.items() if _is_phone_key(k) and len(self._caller_phone) in lens), None)
        if key is None or _num((tc.arguments or {}).get(key)) == self._caller_phone:
            return None
        required = set(schema.get("required") or [])
        if any(k in required for k in formats if k != key):
            return None
        args = {**(tc.arguments or {}), **{k: None for k in formats if k != key}, key: self._caller_phone}
        if tlog:
            tlog.event("number_fallback", tool=tc.name, key=key, heard=(tc.arguments or {}).get(key))
        spec = self._orders_tool()
        res = self._caller_orders_res if spec and spec == (tc.name, key) else None
        if res is None:
            if tlog:
                tlog.event("tool_call", tool=tc.name, args=args)
            try:
                res = await self.call_tool(tc.name, args)
            except Exception as e:  # noqa: BLE001
                log.warning("tool %s raised: %r", tc.name, e)
                res = {"status": "error", "message": "tool call failed"}
            if tlog:
                tlog.event("tool_result", tool=tc.name, status=res.get("status") if isinstance(res, dict) else None)
        if _has_rows(res):
            return {**res, "_note": "고객이 말한 번호가 정확히 들리지 않아 발신번호로 조회한 결과다. 이 결과로 바로 답하고 "
                                    "번호를 다시 묻지 않는다. 고객이 다른 번호로 주문했다고 하면 그때 그 번호를 묻는다."}
        return {**original, "message": f"{original.get('message', '')} 발신번호로도 조회했지만 결과가 없었다."}

    async def _run_tools(self, calls: list[ToolCall], tlog: TurnLog | None) -> list[dict]:
        # AG-T49: 막힌 호출과 같은 함수의 다른 호출이 이번 회차에 통과하면 막힌 호출은 보내지 않는다(P0173-02:
        # 발신번호 조회는 성공했는데 막힌 번호 때문에 결과를 말하지 않고 다시 물었다)
        ok = [self._length_error(tc) is None for tc in calls]
        skip = {i for i, tc in enumerate(calls)
                if not ok[i] and any(ok[j] and calls[j].name == tc.name for j in range(len(calls)) if j != i)}
        for i in skip:
            if tlog:
                tlog.event("tool_skipped", tool=calls[i].name, args=calls[i].arguments)
        done = iter(await asyncio.gather(*(self._exec_tool(tc, tlog) for i, tc in enumerate(calls) if i not in skip)))
        return [dict(SKIPPED_RESULT) if i in skip else next(done) for i in range(len(calls))]

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
        """툴 결과 뒤 호출에 쓸 LLM. 기본 LLM이 HedgedLLM 계열(llms 2개 이상, hedge_delay_s)이고 스위치가 켜져 있으면
        같은 모델들로 새 HedgedLLM을 만든다(툴을 낸 모델을 앞에). 아니면 기본 LLM(툴 후속 고정 동작)."""
        base = self.llm
        llms = list(getattr(base, "llms", None) or [])
        if not post_hedge_enabled() or len(llms) < 2 or not hasattr(base, "hedge_delay_s"):
            return base
        try:
            from agent.providers.hedged_llm import HedgedLLM
        except Exception:  # noqa: BLE001
            return base
        origin = None
        for src in (self._post_hedger, base):            # 툴을 낸 모델(HedgedLLM의 고정 기록)
            pins = getattr(src, "_pins", None)
            if not isinstance(pins, dict):
                continue
            origin = next((pins[tc.id] for tc in calls if tc.id in pins), None)
            if origin:
                break
        order = sorted(llms, key=lambda m: _model_name(m) != origin) if origin else llms
        try:
            self._post_hedger = HedgedLLM(order, hedge_delay_s=float(base.hedge_delay_s),
                                          mode=getattr(base, "mode", "hedge"),
                                          first_event_timeout_s=float(getattr(base, "first_event_timeout_s", 2.5)))
        except Exception as e:  # noqa: BLE001
            log.warning("post-tool hedge unavailable: %r", e)
            return base
        return self._post_hedger

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
        if calls and all(self._length_error(tc) is not None for tc in calls):
            return None                        # AG-T42: 모두 막힐 회차("잠시만요" 뒤 바로 되묻기)엔 필러 없음

        async def fire() -> None:
            await asyncio.sleep(self.settings.filler_delay_ms / 1000)
            pcm = self.cache.get(filler)
            if pcm and not self._filler_used:
                speaker.play(filler, pcm)
                self._filler_used = True
                if tlog:
                    tlog.event("filler", text=filler)

        return asyncio.create_task(fire())

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
        self._sys_cands = number_candidates_from_system(system) if number_candidates_enabled() else []
        self._order_ids = None
        self._order_rows = None
        self._caller_orders_res = None
        said: list[str] = []
        said_chars = 0
        overflow = False
        filler_task: asyncio.Task | None = None
        guard = SpeechGuard(self._schemas, system, msgs)   # AG-T45: 접수 주장·호출 문자열은 TTS 전에 거른다
        rounds = max(1, s.max_tool_rounds)
        try:
            for rnd in range(rounds + 1):                   # +1: guard가 한 번 다시 생성할 때(retries)만 쓴다
                if rnd >= rounds + guard.retries:
                    break
                last = rnd == rounds + guard.retries - 1
                if rnd == 0 and prefetch is not None:
                    stream = prefetch.events()
                else:
                    stream = (post_llm or self.llm).stream(system, msgs, [] if last else self.tools)
                splitter = SentenceSplitter(s.min_sentence_chars)
                text_parts: list[str] = []
                calls: list[ToolCall] = []
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
                        if ev.type == "text" and ev.text:
                            said.append(ev.text)
                            if filler_task is not None:
                                filler_task.cancel()
                                filler_task = None
                            text_parts.append(ev.text)
                            for sent in guard.filter(splitter.feed(ev.text)):
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
                        elif ev.type == "tool_call" and ev.tool_call is not None:
                            calls.append(ev.tool_call)
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
                for sent in guard.finish(splitter.flush(), calls, "".join(text_parts)):
                    if template_said and repeats_template(sent, template_said):
                        continue
                    if s.max_response_chars and said_chars + _nchars(sent) > s.max_response_chars:
                        break
                    said_chars += _nchars(sent)
                    speaker.say(sent)
                calls = calls or guard.extra_calls            # AG-T45: 글로 쓴 호출 → 실제 호출
                if guard.retry_note:                          # AG-T45: 거른 말 대신 교정 지시로 한 번 다시 생성
                    if tlog:
                        tlog.event("guard_retry", note=guard.retry_note[:40])
                    system = f"{system}\n\n{guard.retry_note}"
                    continue
                if not calls or last:
                    break
                calls = [self._normalize(tc) for tc in calls]
                calls = [await self._repair(tc, tlog) for tc in calls]
                calls = fix_full_returns(calls, system, msgs, self._schemas, tlog)   # AG-T45: 전체 반품 → return_order
                round_text = "" if guard.extra_calls else "".join(text_parts)
                if _ASKS.search(round_text) and all(_all_numbers_empty(tc.arguments, self._schemas.get(tc.name, {}))
                                                     and self._length_error(tc) is not None for tc in calls):
                    if tlog:                                  # AG-T49: 이미 번호를 물었으면 번호 없는 조회는 버리고 끝낸다
                        tlog.event("empty_lookup_dropped", tools=[tc.name for tc in calls])
                    break
                msgs.append(Message("assistant", round_text, tool_calls=calls))
                if filler_task is None and not round_text.strip():
                    filler_task = self._start_filler(calls, speaker, tlog)
                results = await self._run_tools(calls, tlog)
                guard.note_results(calls, results)
                self._mark_tool_results()
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
            fallback = guard.fallback(said_chars)           # AG-T45: 거른 문장 때문에 아무 말도 못 했으면
            if fallback:
                speaker.say(fallback)
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
