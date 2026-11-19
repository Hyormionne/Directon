"""측정기 세션 하나. 프레임 수신 → 턴 감지 → 첫 마디 → 본 응답 → transcript → turn.end.

타이밍 규칙(프레임 수 기준, 20ms/프레임):
- 확정 전에는 아무 오디오도 보내지 않는다.
- 침묵 precall_silence_ms(200)에 STT finalize, Smart Turn, LLM 선호출을 시작한다.
- finalize가 확정 전에 끝나면 그 텍스트로 선호출을 다시 건다(이전 선호출과 정규화 텍스트가 다를 때, AG-T10c).
  결정기에도 그때부터 finalize 텍스트를 준다(짧은 발화는 부분 결과가 비어 있는 일이 잦다).
- 선호출 LLM의 첫 문장은 확정 전에 미리 합성해 버퍼에만 둔다(송신하지 않는다). 재사용되면 그 버퍼를 이어 보낸다.
- 확정 전에 음성이 다시 들리면 선호출·finalize·선합성을 취소하고 처음 상태로 돌아간다.
- 확정되면 같은 프레임 처리 안에서 캐시된 첫 마디 PCM을 즉시 보낸다. 첫 마디는 확정 시점 텍스트로 고른다
  (선호출이 있으면 그 선호출을 만들 때 고른 첫 마디 — 시스템 프롬프트의 "이미 '…'라고 말했다"와 맞춘다).
- finalize 텍스트가 선호출 텍스트와 정규화 비교(match_key: 공백·문장부호 제거, 한글 숫자→아라비아 숫자)로 같으면
  선호출 결과를 그대로 쓰고, 다르면 LLM을 다시 부른다.
- 확정부터 safety_timeout_s 안에 turn.end를 보낸다. 실패·시간 초과면 캐시 사과 문구.
- turn.end 뒤에도 연결이 살아 있으면 프레임을 계속 읽는다(다음 발화를 다시 받는다).

디버그(AG-T14, 데모 전용): session.start에 "debug": true가 있을 때만 debug.* 텍스트 이벤트를 더 보낸다
(debug.turn, debug.stt, debug.tool, debug.timing). 없으면 송신 메시지는 기존과 바이트 단위로 같다.
"""
from __future__ import annotations

import asyncio
import collections
import dataclasses
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from agent.config import Settings
from agent.core.types import (FRAME_BYTES, FRAME_MS, Message, SessionInfo, SmartTurn, STTProvider,
                              STTStream, TTSProvider, LLMProvider, TurnDetector, TurnSignals, VAD)
from agent.normalize import spoken_digits
from agent.prompts import build_system
from agent.responder import NormalizeArgs, Prefetch, Responder, SentenceSplitter
from agent.tools_client import ToolsClient
from agent.tts_cache import TTSCache, select_opener, select_opener_for_text
from agent.turn_log import TurnLog, TurnLogWriter

try:  # 실제 websockets 연결이 끊긴 경우
    from websockets.exceptions import ConnectionClosed
except Exception:  # pragma: no cover
    class ConnectionClosed(Exception):  # type: ignore[no-redef]
        pass

log = logging.getLogger(__name__)


@dataclass
class SessionDeps:
    stt: STTProvider
    tts: TTSProvider
    llm: LLMProvider
    vad_factory: Callable[[], VAD]                  # 세션마다 새 VAD(상태 있음)
    smart_turn: SmartTurn | None
    detector_factory: Callable[[], TurnDetector]    # 세션마다 새 결정기
    tools: ToolsClient
    cache: TTSCache
    settings: Settings = field(default_factory=Settings)
    normalize_args: NormalizeArgs | None = None
    log_writer: TurnLogWriter | None = None
    clock: Callable[[], float] = time.perf_counter
    manual: str | None = None                       # None이면 settings.manual_path에서 읽는다


def _expect_short_from_history(history: list[dict]) -> bool:
    if not history:
        return False
    last = history[-1]
    if not isinstance(last, dict) or last.get("role") not in ("agent", "assistant", "bot"):
        return False
    text = (last.get("text") or last.get("content") or "").strip()
    return text.endswith("?") or text.endswith("？")


# ── 선호출 재사용 비교(AG-T10c) ───────────────────────
_TOKEN_SPLIT = re.compile(r"[\s,.!?~…·]+")
_NON_WORD = re.compile(r"[^\w]+")


def match_key(text: str) -> str:
    """선호출 재사용 비교용 정규화: 어절마다 한글 숫자 낭독을 아라비아 숫자로 바꾸고(agent.normalize.spoken_digits),
    공백·문장부호를 지우고 소문자로. 부분 결과("이육일일 일오공")와 최종 결과("2611150")의 표기 차이를 없앤다."""
    out = []
    for tok in _TOKEN_SPLIT.split(text or ""):
        if not tok:
            continue
        try:
            d = spoken_digits(tok)
        except Exception:  # noqa: BLE001
            d = None
        out.append(d if d else tok)
    return _NON_WORD.sub("", "".join(out)).lower()


def _nchars(s: str) -> int:
    return len(re.sub(r"\s", "", s))


class _PreSynth:
    """선호출 첫 문장의 TTS를 미리 받아 버퍼에 쌓는다(송신하지 않음). stream()은 버퍼 + 이어지는 덩어리."""

    def __init__(self, text: str, tts: TTSProvider):
        self.text = text
        self.chunks: list[bytes] = []
        self.done = False
        self.error: BaseException | None = None
        self.claimed = False
        self._ev = asyncio.Event()
        self.task = asyncio.create_task(self._run(tts))

    async def _run(self, tts: TTSProvider) -> None:
        try:
            async for c in tts.synthesize(self.text):
                self.chunks.append(c)
                self._ev.set()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            self.error = e
        finally:
            self.done = True
            self._ev.set()

    async def stream(self) -> AsyncIterator[bytes]:
        i = 0
        while True:
            if i < len(self.chunks):
                yield self.chunks[i]
                i += 1
                continue
            if self.done:
                if self.error is not None:
                    raise self.error
                if self.task.cancelled():
                    raise RuntimeError("presynth cancelled")
                return
            self._ev.clear()
            await self._ev.wait()

    def cancel(self) -> None:
        self.task.cancel()


@dataclass
class _Precall:
    text: str
    key: str
    opener: str
    prefetch: Prefetch
    presynth: _PreSynth | None = None
    watcher: asyncio.Task | None = None
    closed: bool = False            # True면 더는 선합성을 시작하지 않는다

    def cancel(self) -> None:
        self.closed = True
        self.prefetch.cancel()
        if self.watcher is not None and not self.watcher.done():
            self.watcher.cancel()
        if self.presynth is not None and not self.presynth.claimed:
            self.presynth.cancel()


class _PresynthTTS:
    """TTSProvider 래퍼: 재사용된 선호출의 선합성 문장이면 그 버퍼를 이어 주고, 아니면 원래 TTS."""

    def __init__(self, inner: TTSProvider, precall: _Precall | None):
        self.inner = inner
        self.precall = precall
        self.used = False

    def __getattr__(self, name: str) -> Any:
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        pc = self.precall
        pre = pc.presynth if pc is not None else None
        if pc is not None and pre is None:
            pc.closed = True                                # 이제 시작해 봐야 늦다
        if pre is not None and not pre.claimed and pre.text == text and not pre.task.cancelled():
            pre.claimed = True
            self.used = True
            sent = False
            try:
                async for c in pre.stream():
                    sent = True
                    yield c
                return
            except Exception:  # noqa: BLE001
                if sent:
                    raise
                log.warning("presynth failed, synthesizing again: %r", text[:40])
        async for c in self.inner.synthesize(text):
            yield c


class _DebugTurnLog(TurnLog):
    """event()마다 listener를 부른다(디버그 세션에서 응답기의 툴 이벤트를 받기 위해)."""

    listener: Callable[[str, dict], None] | None = None

    def event(self, name: str, **extra: Any) -> None:
        super().event(name, **extra)
        if self.listener is not None:
            try:
                self.listener(name, extra)
            except Exception:  # noqa: BLE001
                pass


def _ms_between(marks: dict[str, float], a: str, b: str) -> float | None:
    if a in marks and b in marks:
        return round(marks[b] - marks[a], 1)
    return None


class Session:
    def __init__(self, ws: Any, deps: SessionDeps):
        self.ws = ws
        self.d = deps
        self.s = deps.settings
        self.closed = False
        self.debug = False              # session.start의 "debug": true(데모 전용)
        self._dbg_pending: list[dict] = []
        self._dbg_tool_args: dict[str, list] = {}
        self.frame_idx = 0
        self.info: SessionInfo | None = None
        self.vad = deps.vad_factory()
        self.detector = deps.detector_factory()
        self.stt: STTStream | None = None
        self._stt_open: asyncio.Task | None = None
        self._tools_task: asyncio.Task | None = None
        self.opener = ""                # 세션 기본 첫 마디(select_opener)
        self.system = ""
        self._systems: dict[str, str] = {}
        self.conversation: list[Message] = []
        self.expect_short = False
        self.responding = False
        self._response_task: asyncio.Task | None = None
        self._ring: collections.deque[bytes] = collections.deque(
            maxlen=max(1, int(self.s.smart_turn_window_s * 1000 / FRAME_MS)))
        self._reset_turn()

    # ── 상태 ─────────────────────────────────────────
    def _reset_turn(self) -> None:
        self.speech_ms = 0
        self.silence_ms = 0
        self.prepared = False
        self.smart_p: float | None = None
        self._finalize_task: asyncio.Task | None = None
        self._smart_task: asyncio.Task | None = None
        self._precall: _Precall | None = None
        self._final_seen = False
        self._early_final: str | None = None            # 확정 전에 끝난 finalize 텍스트
        self.turn_opener = ""
        self._opener_ms: int | None = None
        self.tlog = self._new_tlog(self.info.session_id if self.info else "")
        self.precall_cancels = 0
        self.precall_count = 0

    def _new_tlog(self, session_id: str, t0: float | None = None) -> TurnLog:
        if not self.debug:
            return TurnLog(session_id, clock=self.d.clock, t0=t0)
        tl = _DebugTurnLog(session_id, clock=self.d.clock, t0=t0)
        tl.listener = self._on_tlog_event
        return tl

    # ── 디버그(데모 전용) ───────────────────────────
    def _on_tlog_event(self, name: str, extra: dict) -> None:
        """응답기의 툴 이벤트 → debug.tool(다음 송신 직전에 같이 보낸다)."""
        if name == "tool_call":
            self._dbg_tool_args.setdefault(extra.get("tool") or "", []).append(extra.get("args"))
        elif name in ("tool_result", "tool_blocked"):
            tool = extra.get("tool") or ""
            if name == "tool_result":
                q = self._dbg_tool_args.get(tool) or []
                args = q.pop(0) if q else None
                status = extra.get("status")
            else:
                args, status = extra.get("args"), "blocked"
            self._dbg_pending.append({"type": "debug.tool", "name": tool, "arguments": args,
                                      "status": status})

    async def _debug(self, obj: dict) -> None:
        if self.debug:
            await self._send_json(obj)

    def _timing(self) -> dict:
        m = self.tlog.marks
        f = self.tlog.fields
        return {"type": "debug.timing",
                "decision_reason": f.get("decision_reason"),
                "speech_ms": f.get("speech_ms"),
                "silence_to_commit_ms": _ms_between(m, "silence_start", "commit"),
                "commit_to_opener_ms": _ms_between(m, "commit", "opener_start"),
                "commit_to_finalize_ms": _ms_between(m, "commit", "finalize_done"),
                "commit_to_llm_first_ms": _ms_between(m, "commit", "llm_first_token"),
                "commit_to_response_audio_ms": _ms_between(m, "commit", "response_first_audio"),
                "commit_to_turn_end_ms": _ms_between(m, "commit", "turn_end"),
                "gap_ms": f.get("gap_ms"),
                "opener": f.get("opener"),
                "precall_reused": f.get("precall_reused"),
                "presynth_reused": f.get("presynth_reused"),
                "failed": f.get("failed")}

    def _cancel_prepared(self) -> None:
        for t in (self._finalize_task, self._smart_task):
            if t is not None and not t.done():
                t.cancel()
        if self._precall is not None:
            self._precall.cancel()
        self._finalize_task = self._smart_task = None
        self._precall = None
        self._final_seen = False
        self._early_final = None
        self.smart_p = None
        self.prepared = False

    # ── 송신 ─────────────────────────────────────────
    async def _send(self, msg: bytes | str) -> None:
        if self.closed:
            return
        try:
            while self._dbg_pending:                    # 디버그 세션에서만 쌓인다
                await self.ws.send(json.dumps(self._dbg_pending.pop(0), ensure_ascii=False))
            await self.ws.send(msg)
        except ConnectionClosed:
            self.closed = True

    async def send_audio(self, pcm: bytes) -> None:
        n = self.s.out_chunk_bytes
        for i in range(0, len(pcm), n):
            await self._send(pcm[i:i + n])

    async def _send_json(self, obj: dict) -> None:
        await self._send(json.dumps(obj, ensure_ascii=False))

    # ── 메인 루프 ────────────────────────────────────
    async def run(self) -> None:
        try:
            start = await self._recv_start()
            if start is None:
                return
            self.info = SessionInfo.from_start(start)
            if start.get("debug") is True:
                self.debug = True
                self.tlog = self._new_tlog(self.info.session_id, t0=self.tlog.t0)
            if self.info.caller is None and start.get("customer_profile"):
                self.info.caller = start["customer_profile"]
            self.tlog.session_id = self.info.session_id
            self.expect_short = _expect_short_from_history(self.info.history)
            self.opener = select_opener(self.info, self.s)
            self.system = self._system_for(self.opener)
            self._tools_task = asyncio.create_task(self.d.tools.list_tools(self.info.tools_url))
            self._stt_open = asyncio.create_task(self.d.stt.open_stream(self.s.stt_context))
            try:
                async for msg in self.ws:
                    if isinstance(msg, (bytes, bytearray)):
                        await self._on_frame(bytes(msg))
            except ConnectionClosed:
                pass
            self.closed = True
            if self._response_task is not None:
                await self._response_task
        finally:
            await self._cleanup()

    async def _recv_start(self) -> dict | None:
        while True:
            try:
                raw = await self.ws.recv()
            except (ConnectionClosed, StopAsyncIteration):
                return None
            if isinstance(raw, (bytes, bytearray)):
                continue
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            if msg.get("type") == "session.start":
                return msg

    async def _cleanup(self) -> None:
        self._cancel_prepared()
        if self._response_task is not None and not self._response_task.done():
            self._response_task.cancel()
        if self._tools_task is not None and not self._tools_task.done():
            self._tools_task.cancel()
        stt = self.stt
        if stt is None and self._stt_open is not None:
            if self._stt_open.done() and not self._stt_open.cancelled() and self._stt_open.exception() is None:
                stt = self._stt_open.result()
            else:
                self._stt_open.cancel()
        if stt is not None:
            try:
                await stt.close()
            except Exception:  # noqa: BLE001
                pass

    async def _ensure_stt(self) -> STTStream | None:
        if self.stt is None and self._stt_open is not None:
            try:
                self.stt = await self._stt_open
            except Exception as e:  # noqa: BLE001
                log.error("stt open failed: %r", e)
                self._stt_open = None
        return self.stt

    def _system_for(self, opener: str) -> str:
        """첫 마디마다 시스템 프롬프트("이미 '{opener}'라고 말했다")를 만든다(세션 안에서 캐시)."""
        sys_ = self._systems.get(opener)
        if sys_ is None:
            assert self.info is not None
            sys_ = build_system(self.info, opener, manual_path=self.s.manual_path, manual=self.d.manual)
            self._systems[opener] = sys_
        return sys_

    def _opener_fixed(self) -> bool:
        name = ((self.info.caller or {}).get("name") or "").strip() if self.info else ""
        return bool(self.s.opener_use_name and name)

    def _opener_for(self, text: str) -> str:
        """텍스트 기반 첫 마디. 캐시에 없는 문구면 세션 기본 첫 마디(확정 즉시 내보낼 수 있어야 한다)."""
        base = self.opener
        if not self.s.opener_by_text or self._opener_fixed():
            return base
        cand = select_opener_for_text(text, self.s, default=base)
        if cand != base and self.d.cache.get(cand) is None:
            return base
        return cand

    def _tools(self) -> list[dict] | None:
        t = self._tools_task
        if t is not None and t.done() and not t.cancelled() and t.exception() is None:
            return t.result()
        return None

    # ── 프레임 처리 ──────────────────────────────────
    async def _on_frame(self, frame: bytes) -> None:
        self.frame_idx += 1
        if self.responding:
            return                                      # 응답 중 프레임은 버린다(drain)
        stt = await self._ensure_stt()
        if stt is not None:
            try:
                await stt.send_audio(frame)
            except Exception as e:  # noqa: BLE001
                log.warning("stt send failed: %r", e)
        self._ring.append(frame)
        p = self.vad.process(frame)
        ms = len(frame) * 1000 // (2 * 16000) or FRAME_MS
        if p >= self.s.vad_threshold:
            if self.speech_ms == 0:
                self.tlog.mark("first_speech")
                self.tlog.mark("first_speech_frame", self.frame_idx)
                await self._debug({"type": "debug.turn", "event": "speech_start", "silence_ms": 0})
            elif self.silence_ms > 0 and self.debug:
                await self._debug({"type": "debug.turn", "event": "resume", "silence_ms": self.silence_ms,
                                   "prepared": self.prepared})
            if self.silence_ms > 0 and self.prepared:
                self._cancel_prepared()                 # 확정 전 음성 재개 → 선호출 취소
                self.precall_cancels += 1
                self.tlog.event("precall_cancelled")
            self.speech_ms += ms
            self.silence_ms = 0
            return
        if self.speech_ms < self.s.min_speech_ms:
            if self.speech_ms and self.silence_ms + ms >= self.s.precall_silence_ms:
                self.speech_ms = 0                      # 짧은 잡음은 잊는다
                self.silence_ms = 0
            elif self.speech_ms:
                self.silence_ms += ms
            return
        if self.silence_ms == 0:
            self.tlog.mark("silence_start", overwrite=True)
            await self._debug({"type": "debug.turn", "event": "silence", "silence_ms": 0,
                               "speech_ms": self.speech_ms})
        self.silence_ms += ms
        if not self.prepared and self.silence_ms >= self.s.precall_silence_ms:
            self._prepare(stt)
        self._check_finalize()
        self._collect_smart()
        # 텍스트가 비어도(짧은 발화는 부분 결과가 늦다) 결정기 규칙 그대로. finalize가 끝났으면 그 텍스트.
        early = self._early_final if self.s.precall_refresh else None
        text = early or (stt.latest_text() if stt is not None else "")
        sig = TurnSignals(silence_ms=self.silence_ms, speech_ms=self.speech_ms, text=text,
                          smart_turn_p=self.smart_p, expect_short=self.expect_short)
        dec = self.detector.decide(sig)
        if dec.end:
            await self._commit(dec.reason, text)

    def _prepare(self, stt: STTStream | None) -> None:
        self.prepared = True
        if stt is not None:
            self.tlog.mark("finalize_request", overwrite=True)
            self._finalize_task = asyncio.create_task(stt.finalize(self.s.finalize_timeout_s))
        if self.d.smart_turn is not None:
            pcm = b"".join(self._ring)
            self._smart_task = asyncio.create_task(asyncio.to_thread(self.d.smart_turn.predict, pcm))
        self._start_precall(stt.latest_text() if stt is not None else "", "partial")

    def _key(self, text: str) -> str:
        return match_key(text) if self.s.precall_refresh else (text or "").strip()

    def _check_finalize(self) -> None:
        """확정 전에 finalize가 끝났으면 그 텍스트를 기억하고, 선호출 텍스트와 다르면 선호출을 다시 건다."""
        t = self._finalize_task
        if t is None or self._final_seen or not t.done():
            return
        self._final_seen = True
        if t.cancelled() or t.exception() is not None:
            return
        self._early_final = (t.result() or "").strip()
        self.tlog.event("finalize_early", text=self._early_final)
        if self._early_final and self.s.precall_refresh:
            self._start_precall(self._early_final, "finalize")

    def _start_precall(self, text: str, source: str) -> None:
        text = (text or "").strip()
        tools = self._tools()
        if not text or tools is None:
            return
        key = self._key(text)
        if self._precall is not None:
            if self._precall.key == key:
                return
            self._precall.cancel()
            self.tlog.event("precall_replaced", old=self._precall.text)
        opener = self._opener_for(text)
        r = self._responder(tools)
        pf = r.prefetch(self._system_for(opener), self.conversation + [Message("user", text)])
        pc = _Precall(text=text, key=key, opener=opener, prefetch=pf)
        if self.s.precall_refresh:
            pc.watcher = asyncio.create_task(self._presynth_first(pc))
        self._precall = pc
        self.precall_count += 1
        self.tlog.event("precall", text=text, source=source, opener=opener)

    async def _presynth_first(self, pc: _Precall) -> None:
        """선호출 스트림에서 첫 문장을 응답기와 같은 규칙(SentenceSplitter)으로 찾아 미리 합성한다(송신 안 함)."""
        sp = SentenceSplitter(self.s.min_sentence_chars)
        first = None
        try:
            async for ev in pc.prefetch.events():
                if ev.type == "text" and ev.text:
                    out = sp.feed(ev.text)
                    if out:
                        first = out[0]
                        break
                elif ev.type == "done":
                    break
            if first is None:
                rest = sp.flush()
                first = rest[0] if rest else None
        except Exception:  # noqa: BLE001
            return
        if not first or pc.closed or (self.s.max_response_chars and _nchars(first) > self.s.max_response_chars):
            return
        pc.presynth = _PreSynth(first, self.d.tts)
        self.tlog.event("presynth", text=first)

    def _collect_smart(self) -> None:
        t = self._smart_task
        if t is not None and self.smart_p is None and t.done():
            if not t.cancelled() and t.exception() is None:
                self.smart_p = float(t.result())
                self.tlog.mark("smart_turn", overwrite=True)
                self.tlog.set("smart_turn_p", self.smart_p)

    def _responder(self, tools: list[dict], tts: Any = None, settings: Settings | None = None) -> Responder:
        info = self.info
        assert info is not None

        async def call_tool(name: str, args: dict) -> dict:
            return await self.d.tools.call(info.tools_url, info.session_id, name, args)

        return Responder(self.d.llm, tts or self.d.tts, call_tool, tools=tools, settings=settings or self.s,
                         cache=self.d.cache, normalize_args=self.d.normalize_args, now=info.now)

    # ── 확정 → 응답 ──────────────────────────────────
    async def _commit(self, reason: str, text: str) -> None:
        self.responding = True
        self.tlog.mark("commit")
        self.tlog.mark("commit_frame", self.frame_idx)
        self.tlog.set("decision_reason", reason)
        self.tlog.set("speech_ms", self.speech_ms)
        self.tlog.set("silence_ms", self.silence_ms)
        self.tlog.set("precall_cancels", self.precall_cancels)
        self._check_finalize()
        if self._precall is not None:                   # 선호출 프롬프트와 같은 첫 마디
            self.turn_opener = self._precall.opener
        else:
            self.turn_opener = self._opener_for((self._early_final if self.s.precall_refresh else None) or text)
        self.tlog.set("opener", self.turn_opener)
        spoken: list[str] = []
        opener_pcm = self.d.cache.get(self.turn_opener)
        if opener_pcm is not None:
            self.tlog.mark("opener_start")
            await self.send_audio(opener_pcm)           # 확정 즉시(같은 프레임 처리 안)
            spoken.append(self.turn_opener)
            self.tlog.mark("opener_sent")
            self._opener_ms = len(opener_pcm) // 32
        await self._debug({"type": "debug.turn", "event": "decided", "silence_ms": self.silence_ms,
                           "speech_ms": self.speech_ms, "reason": reason, "text": text,
                           "opener": self.turn_opener})
        self._response_task = asyncio.create_task(
            self._respond_turn(spoken, opener_sent=opener_pcm is not None))

    async def _final_text(self) -> str:
        stt = self.stt
        if self._finalize_task is None and stt is not None:
            self.tlog.mark("finalize_request", overwrite=True)
            self._finalize_task = asyncio.create_task(stt.finalize(self.s.finalize_timeout_s))
        text = ""
        if self._finalize_task is not None:
            try:
                text = await asyncio.wait_for(self._finalize_task, self.s.finalize_timeout_s + 0.5)
            except Exception as e:  # noqa: BLE001
                log.warning("finalize failed: %r", e)
                text = stt.latest_text() if stt is not None else ""
        elif stt is not None:
            text = stt.latest_text()
        self.tlog.mark("finalize_done")
        return (text or "").strip()

    async def _play_cached(self, text: str, spoken: list[str]) -> bytes | None:
        pcm = self.d.cache.get(text)
        if pcm is None:
            try:
                pcm = await asyncio.wait_for(self.d.cache.get_or_synth(text, self.d.tts), 3.0)
            except Exception:  # noqa: BLE001
                pcm = None
        if pcm:
            await self.send_audio(pcm)
        spoken.append(text)
        return pcm

    async def _respond_turn(self, spoken: list[str], opener_sent: bool) -> None:
        user_text = ""
        failed = False
        first_audio = False

        async def send_response(pcm: bytes) -> None:
            nonlocal first_audio
            if not first_audio:
                first_audio = True
                self.tlog.mark("response_first_audio")
            await self.send_audio(pcm)

        try:
            async with asyncio.timeout(self.s.safety_timeout_s):
                if not opener_sent:                     # 캐시에 없으면 실시간 합성
                    self.tlog.mark("opener_start")
                    pcm = await self._play_cached(self.turn_opener, spoken)
                    self.tlog.mark("opener_sent")
                    if pcm:
                        self._opener_ms = len(pcm) // 32
                user_text = await self._final_text()
                self.tlog.set("user_text", user_text)
                await self._debug({"type": "debug.stt", "text": user_text})
                if not user_text:
                    if self._precall is not None:
                        self._precall.cancel()
                    await self._play_cached(self.s.reprompt, spoken)
                else:
                    tools = self._tools()
                    if tools is None and self._tools_task is not None:
                        try:
                            tools = await asyncio.wait_for(asyncio.shield(self._tools_task),
                                                           self.s.tools_timeout_s)
                        except Exception:  # noqa: BLE001
                            tools = []
                    tools = tools or []
                    pc = self._precall
                    if pc is not None and pc.key != self._key(user_text):
                        pc.cancel()
                        pc = None
                        self.tlog.event("precall_discarded")
                    self.tlog.set("precall_reused", pc is not None)
                    self.tlog.set("precall_count", self.precall_count)
                    settings = self.s
                    if self.turn_opener == self.s.opener_lookup and self.s.opener_lookup_filler:
                        settings = dataclasses.replace(self.s, fillers=(self.s.opener_lookup_filler,))
                    tts = _PresynthTTS(self.d.tts, pc)
                    messages = self.conversation + [Message("user", user_text)]
                    try:
                        await self._responder(tools, tts=tts, settings=settings).respond(
                            self._system_for(self.turn_opener), messages, send_response, spoken=spoken,
                            prefetch=pc.prefetch if pc is not None else None, log=self.tlog)
                    finally:
                        self.tlog.set("presynth_reused", tts.used)
                        if pc is not None:
                            pc.cancel()                 # 남은 선합성 정리(이미 쓴 것은 claimed라 그대로)
        except (Exception, TimeoutError) as e:  # noqa: BLE001
            failed = True
            log.warning("response failed (%s): %r", self.info.session_id if self.info else "", e)
            self.tlog.set("error", repr(e)[:300])
            if self._precall is not None:
                self._precall.cancel()
            await self._play_cached(self.s.apology, spoken)
        self._log_gap()
        agent_text = " ".join(t for t in spoken if t)
        await self._send_json({"type": "agent.transcript", "text": agent_text})
        await self._send_json({"type": "turn.end"})
        self.tlog.mark("turn_end")
        self.tlog.set("agent_text", agent_text)
        self.tlog.set("failed", failed)
        await self._debug(self._timing())
        if self._smart_task is not None and not self._smart_task.done():
            await asyncio.wait({self._smart_task}, timeout=0.2)   # 로그용(응답과 무관)
        self._collect_smart()
        self._write_log()
        if user_text:
            self.conversation.append(Message("user", user_text))
        self.conversation.append(Message("assistant", agent_text))
        self.expect_short = agent_text.rstrip().endswith("?")
        await self._next_turn()

    def _log_gap(self) -> None:
        """공백 = 본 응답 첫 오디오 송신 - (첫 마디 송신 시작 + 첫 마디 길이), ms.
        음수면 첫 마디가 끝나기 전에 본 응답이 도착(들리는 공백 없음)."""
        m = self.tlog.marks
        if self._opener_ms is not None:
            self.tlog.set("opener_ms", self._opener_ms)
        if "opener_start" in m and "response_first_audio" in m and self._opener_ms is not None:
            self.tlog.set("gap_ms", round(m["response_first_audio"] - (m["opener_start"] + self._opener_ms), 1))

    def _write_log(self) -> None:
        if self.d.log_writer is None:
            return
        try:
            self.d.log_writer.write(self.tlog.to_dict())
        except Exception as e:  # noqa: BLE001
            log.warning("turn log write failed: %r", e)

    async def _next_turn(self) -> None:
        """turn.end 뒤 연결이 유지되면 다음 발화를 받을 준비(새 STT 스트림, VAD 초기화)."""
        if self.closed:
            return
        old = self.stt
        self.stt = None
        self._stt_open = asyncio.create_task(self.d.stt.open_stream(self.s.stt_context))
        if old is not None:
            try:
                await old.close()
            except Exception:  # noqa: BLE001
                pass
        self.vad.reset()
        self._ring.clear()
        self.detector = self.d.detector_factory()
        self._dbg_tool_args = {}
        self._reset_turn()
        self.responding = False
