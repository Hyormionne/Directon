"""측정기 세션 하나. 프레임 수신 → 턴 감지 → 첫 마디 → 본 응답 → transcript → turn.end.

타이밍 규칙(프레임 수 기준, 20ms/프레임):
- 확정 전에는 아무 오디오도 보내지 않는다.
- 침묵 precall_silence_ms(200)에 STT finalize, Smart Turn, LLM 선호출을 시작한다.
- 확정 전에 음성이 다시 들리면 선호출·finalize를 취소하고 처음 상태로 돌아간다.
- 확정되면 같은 프레임 처리 안에서 캐시된 첫 마디 PCM을 즉시 보낸다.
- finalize 텍스트가 선호출 텍스트와 같으면 선호출 결과를 그대로 쓰고, 다르면 LLM을 다시 부른다.
- 확정부터 safety_timeout_s 안에 turn.end를 보낸다. 실패·시간 초과면 캐시 사과 문구.
- turn.end 뒤에도 연결이 살아 있으면 프레임을 계속 읽는다(다음 발화를 다시 받는다).
"""
from __future__ import annotations

import asyncio
import collections
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.config import Settings
from agent.core.types import (FRAME_BYTES, FRAME_MS, Message, SessionInfo, SmartTurn, STTProvider,
                              STTStream, TTSProvider, LLMProvider, TurnDetector, TurnSignals, VAD)
from agent.prompts import build_system
from agent.responder import NormalizeArgs, Prefetch, Responder
from agent.tools_client import ToolsClient
from agent.tts_cache import TTSCache, select_opener
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


class Session:
    def __init__(self, ws: Any, deps: SessionDeps):
        self.ws = ws
        self.d = deps
        self.s = deps.settings
        self.closed = False
        self.frame_idx = 0
        self.info: SessionInfo | None = None
        self.vad = deps.vad_factory()
        self.detector = deps.detector_factory()
        self.stt: STTStream | None = None
        self._stt_open: asyncio.Task | None = None
        self._tools_task: asyncio.Task | None = None
        self.opener = ""
        self.system = ""
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
        self._prefetch: Prefetch | None = None
        self._prefetch_text = ""
        self.tlog = TurnLog(self.info.session_id if self.info else "", clock=self.d.clock)
        self.precall_cancels = 0

    def _cancel_prepared(self) -> None:
        for t in (self._finalize_task, self._smart_task):
            if t is not None and not t.done():
                t.cancel()
        if self._prefetch is not None:
            self._prefetch.cancel()
        self._finalize_task = self._smart_task = None
        self._prefetch = None
        self._prefetch_text = ""
        self.smart_p = None
        self.prepared = False

    # ── 송신 ─────────────────────────────────────────
    async def _send(self, msg: bytes | str) -> None:
        if self.closed:
            return
        try:
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
            if self.info.caller is None and start.get("customer_profile"):
                self.info.caller = start["customer_profile"]
            self.tlog.session_id = self.info.session_id
            self.expect_short = _expect_short_from_history(self.info.history)
            self.opener = select_opener(self.info, self.s)
            self.system = build_system(self.info, self.opener, manual_path=self.s.manual_path,
                                       manual=self.d.manual)
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
        self.silence_ms += ms
        if not self.prepared and self.silence_ms >= self.s.precall_silence_ms:
            self._prepare(stt)
        self._collect_smart()
        text = stt.latest_text() if stt is not None else ""
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
        text = stt.latest_text() if stt is not None else ""
        tools = self._tools()
        if text.strip() and tools is not None:
            r = self._responder(tools)
            self._prefetch = r.prefetch(self.system, self.conversation + [Message("user", text)])
            self._prefetch_text = text
            self.tlog.event("precall", text=text)

    def _collect_smart(self) -> None:
        t = self._smart_task
        if t is not None and self.smart_p is None and t.done():
            if not t.cancelled() and t.exception() is None:
                self.smart_p = float(t.result())
                self.tlog.mark("smart_turn", overwrite=True)
                self.tlog.set("smart_turn_p", self.smart_p)

    def _responder(self, tools: list[dict]) -> Responder:
        info = self.info
        assert info is not None

        async def call_tool(name: str, args: dict) -> dict:
            return await self.d.tools.call(info.tools_url, info.session_id, name, args)

        return Responder(self.d.llm, self.d.tts, call_tool, tools=tools, settings=self.s,
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
        spoken: list[str] = []
        opener_pcm = self.d.cache.get(self.opener)
        if opener_pcm is not None:
            await self.send_audio(opener_pcm)           # 확정 즉시(같은 프레임 처리 안)
            spoken.append(self.opener)
            self.tlog.mark("opener_sent")
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

    async def _play_cached(self, text: str, spoken: list[str]) -> None:
        pcm = self.d.cache.get(text)
        if pcm is None:
            try:
                pcm = await asyncio.wait_for(self.d.cache.get_or_synth(text, self.d.tts), 3.0)
            except Exception:  # noqa: BLE001
                pcm = None
        if pcm:
            await self.send_audio(pcm)
        spoken.append(text)

    async def _respond_turn(self, spoken: list[str], opener_sent: bool) -> None:
        user_text = ""
        failed = False
        try:
            async with asyncio.timeout(self.s.safety_timeout_s):
                if not opener_sent:                     # 캐시에 없으면 실시간 합성
                    await self._play_cached(self.opener, spoken)
                    self.tlog.mark("opener_sent")
                user_text = await self._final_text()
                self.tlog.set("user_text", user_text)
                if not user_text:
                    if self._prefetch is not None:
                        self._prefetch.cancel()
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
                    prefetch = self._prefetch
                    if prefetch is not None and self._prefetch_text.strip() != user_text:
                        prefetch.cancel()
                        prefetch = None
                        self.tlog.event("precall_discarded")
                    self.tlog.set("precall_reused", prefetch is not None)
                    messages = self.conversation + [Message("user", user_text)]
                    await self._responder(tools).respond(self.system, messages, self.send_audio,
                                                         spoken=spoken, prefetch=prefetch,
                                                         log=self.tlog)
        except (Exception, TimeoutError) as e:  # noqa: BLE001
            failed = True
            log.warning("response failed (%s): %r", self.info.session_id if self.info else "", e)
            self.tlog.set("error", repr(e)[:300])
            if self._prefetch is not None:
                self._prefetch.cancel()
            await self._play_cached(self.s.apology, spoken)
        agent_text = " ".join(t for t in spoken if t)
        await self._send_json({"type": "agent.transcript", "text": agent_text})
        await self._send_json({"type": "turn.end"})
        self.tlog.mark("turn_end")
        self.tlog.set("agent_text", agent_text)
        self.tlog.set("failed", failed)
        if self._smart_task is not None and not self._smart_task.done():
            await asyncio.wait({self._smart_task}, timeout=0.2)   # 로그용(응답과 무관)
        self._collect_smart()
        self._write_log()
        if user_text:
            self.conversation.append(Message("user", user_text))
        self.conversation.append(Message("assistant", agent_text))
        self.expect_short = agent_text.rstrip().endswith("?")
        await self._next_turn()

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
        self._reset_turn()
        self.responding = False
