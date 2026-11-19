"""응답기: LLM 스트리밍 + 툴 루프 → 문장 분할 → 문장별 TTS(다음 문장 미리 합성) → 송신 콜백."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, AsyncIterator, Awaitable, Callable

from agent.config import Settings
from agent.core.types import LLMEvent, LLMProvider, Message, ToolCall, TTSProvider
from agent.normalize import spoken_digits
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
                 settings: Settings, tlog: TurnLog | None):
        self.tts = tts
        self.send_audio = send_audio
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
                await self.send_audio(chunk)

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
MIN_NUMBER_DIGITS = 7
MAX_GROUND_DISTANCE = 2


_DIGIT_CHARS = set("0123456789공영빵일이삼사오육륙칠팔구")
# (조사·어미, 앞 글자 받침 필요 여부). 긴 것부터.
_NUM_SUFFIXES = [("이거든요", True), ("거든요", False), ("인데요", False), ("이고요", True), ("이에요", True), ("입니다", True), ("이요", True), ("이고", True),
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


def utterance_numbers(text: str) -> list[str]:
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
                out.extend(v for v in vs if len(v) >= MIN_NUMBER_DIGITS)
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
    return "phone" in k or k == "id" or k.endswith("_id")


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
            limit = MAX_GROUND_DISTANCE + 1 if same else MAX_GROUND_DISTANCE
            if dist <= limit and (best is None or rank < best[0]):
                best = (rank, c)
        if best is not None:
            out[key] = best[1]
    return out


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
                log.info("number args grounded for %s: %s -> %s", tc.name, args, grounded)
                args = grounded
        return ToolCall(tc.id, tc.name, args)

    def _length_error(self, tc: ToolCall) -> dict | None:
        if not self.settings.validate_number_length:
            return None
        for key, n in number_length_rules(self._schemas.get(tc.name, {})).items():
            v = (tc.arguments or {}).get(key)
            if isinstance(v, str) and v.isdigit() and len(v) != n:
                return {"status": "invalid_arguments",
                        "message": f"{key}는 숫자 {n}자리여야 하는데 {len(v)}자리입니다. 고객에게 "
                                   "'번호를 한 자리씩 천천히 다시 말씀해 주시겠어요?'처럼 다시 묻는다."}
        return None

    async def _exec_tool(self, tc: ToolCall, tlog: TurnLog | None) -> dict:
        self._called.append(tc.name)
        bad = self._length_error(tc)
        if bad is not None:                       # 자리 수가 틀린 번호는 보내지 않는다(여분 호출 감점 방지)
            if tlog:
                tlog.event("tool_blocked", tool=tc.name, args=tc.arguments)
            return bad
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
        return res

    async def _run_tools(self, calls: list[ToolCall], tlog: TurnLog | None) -> list[dict]:
        return list(await asyncio.gather(*(self._exec_tool(tc, tlog) for tc in calls)))

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
        speaker = _Speaker(self.tts, send_audio, spoken, s, tlog)
        self._filler_used = False
        self._numbers = [n for m in msgs if m.role == "user" for n in utterance_numbers(m.content)]
        self._called = []
        said: list[str] = []
        said_chars = 0
        overflow = False
        filler_task: asyncio.Task | None = None
        rounds = max(1, s.max_tool_rounds)
        try:
            for rnd in range(rounds):
                last = rnd == rounds - 1
                if rnd == 0 and prefetch is not None:
                    stream = prefetch.events()
                else:
                    stream = self.llm.stream(system, msgs, [] if last else self.tools)
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
                            for sent in splitter.feed(ev.text):
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
                for sent in splitter.flush():
                    if s.max_response_chars and said_chars + _nchars(sent) > s.max_response_chars:
                        break
                    said_chars += _nchars(sent)
                    speaker.say(sent)
                if not calls or last:
                    break
                calls = [self._normalize(tc) for tc in calls]
                round_text = "".join(text_parts)
                msgs.append(Message("assistant", round_text, tool_calls=calls))
                if filler_task is None and not round_text.strip():
                    filler_task = self._start_filler(calls, speaker, tlog)
                results = await self._run_tools(calls, tlog)
                for tc, res in zip(calls, results):
                    msgs.append(Message("tool", json.dumps(res, ensure_ascii=False),
                                        tool_call_id=tc.id, name=tc.name))
                speaker.raise_if_failed()
            if filler_task is not None:
                filler_task.cancel()
                filler_task = None
            await self._ensure_transfer("".join(said), msgs, tlog)
            await speaker.finish()
        except BaseException:
            if filler_task is not None:
                filler_task.cancel()
            speaker.cancel()
            raise
        if not spoken:
            raise ResponderError("empty response")
        return spoken
