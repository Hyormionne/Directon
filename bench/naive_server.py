"""단순(naive) STT+LLM+TTS 기준선 서버 (AG-T13). 실행: uv run python -m bench.naive_server --port 9000

우리 최적화(보정 턴 결정기, 캐시 첫 마디, 선호출, 문장 단위 TTS, 인자 정규화·번호 교정, 튜닝 프롬프트,
LLM 폴백, 숫자 복원 STT 래퍼, TTS 레벨 정규화)의 효과를 보기 위한 '업계에서 흔한 최소 구현'.
같은 Provider(factory)를 쓰되 우리 보정은 끈다.

흐름(세션당 1턴):
  session.start → 20ms PCM을 STT에 그대로 흘리고 Silero VAD로 음성 여부 판정
  → 음성 시작(연속 start_ms 이상) 뒤 침묵 end_silence_ms(800ms, Starter Kit 예시 서버 값)면 말 끝
  → STT finalize → LLM(짧은 역할 + 매뉴얼 + now/caller/history 원문) + GET /tools 그대로
  → 툴 호출이면 X-Session-Id로 호출(인자 그대로) → 결과 넣고 다시 LLM(최대 max_rounds 회, 마지막 회차는 툴 없이)
  → 전체 응답 텍스트로 TTS 1회(문장 분할 없음, --tts-mode sentence면 N1 변형: 문장 단위 스트리밍)
  → PCM 송신(받는 대로, 앞 무음 제거·레벨 정규화 없음) → agent.transcript → turn.end → drain.
에러가 나도 turn.end는 보낸다.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agent.core.types import FRAME_MS, Message, SessionInfo, ToolCall

log = logging.getLogger("bench.naive")

ROOT = Path(__file__).resolve().parents[1]
MANUAL_PATH = ROOT / "cases" / "manual.md"

ROLE = ("너는 온라인 쇼핑몰 다이렉트몰 고객센터의 전화 상담원이다. 아래 응대 매뉴얼에 따라 고객의 문의에 답하고, "
        "필요하면 제공된 함수를 호출한다. 너의 답은 음성으로 읽힌다.")

_SENT_END = re.compile(r"[.!?。？！]\s|[.!?。？！]$|\n")


@dataclass
class NaiveConfig:
    vad_threshold: float = 0.5
    start_ms: int = 200            # 연속 음성이 이만큼이면 발화 시작(흔한 VAD 기본값: pipecat start_secs 0.2)
    end_silence_ms: int = 800      # Starter Kit example_server END_SILENCE_MS
    finalize_timeout_s: float = 1.0
    max_rounds: int = 3            # LLM 호출 최대 횟수
    out_chunk_bytes: int = 3200    # 송신 청크(100ms)
    response_timeout_s: float = 22.0   # 말 끝 → turn.end 안전 상한(측정기 꼬리 25초 안)
    tts_mode: str = "full"         # full(N0: 전체 텍스트 TTS 1회) | sentence(N1: 문장 단위 스트리밍)


@dataclass
class NaiveDeps:
    stt: Any
    llm: Any
    tts: Any
    tools: Any                     # agent.tools_client.ToolsClient
    vad_factory: Callable[[], Any]
    manual: str = ""
    cfg: NaiveConfig = field(default_factory=NaiveConfig)


def build_system(info: SessionInfo, manual: str) -> str:
    caller = json.dumps(info.caller, ensure_ascii=False) if info.caller else "없음"
    history = json.dumps(info.history or [], ensure_ascii=False)
    return (f"{ROLE}\n\n## 응대 매뉴얼\n{manual.strip()}\n\n"
            f"## 현재 시각\n{info.now}\n\n## 발신 고객 정보\n{caller}\n\n## 이전 대화\n{history}")


class NaiveSession:
    def __init__(self, ws: Any, deps: NaiveDeps):
        self.ws = ws
        self.d = deps
        self.c = deps.cfg
        self.info: SessionInfo | None = None
        self.stt: Any = None
        self.log: dict[str, Any] = {}

    async def _send(self, msg: Any) -> None:
        try:
            await self.ws.send(msg)
        except Exception:  # noqa: BLE001  (연결 끊김 등)
            pass

    async def _send_pcm(self, pcm: bytes) -> None:
        n = self.c.out_chunk_bytes
        for i in range(0, len(pcm), n):
            await self._send(pcm[i:i + n])

    async def _recv_start(self) -> dict | None:
        while True:
            try:
                raw = await self.ws.recv()
            except Exception:  # noqa: BLE001
                return None
            if isinstance(raw, (bytes, bytearray)):
                continue
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            if msg.get("type") == "session.start":
                return msg

    async def run(self) -> None:
        start = await self._recv_start()
        if start is None:
            return
        self.info = SessionInfo.from_start(start)
        if self.info.caller is None and start.get("customer_profile"):
            self.info.caller = start["customer_profile"]
        self.log["session_id"] = self.info.session_id
        tools_task = asyncio.create_task(self.d.tools.list_tools(self.info.tools_url))
        stt_task = asyncio.create_task(self.d.stt.open_stream(""))
        drain: asyncio.Task | None = None
        try:
            ended = await self._listen(stt_task)
            if not ended:
                return                                   # 말 끝 전에 연결이 닫힘
            drain = asyncio.create_task(self._drain())   # 측정기는 끝까지 무음을 보낸다
            try:
                async with asyncio.timeout(self.c.response_timeout_s):
                    await self._respond(tools_task)
            except Exception as e:  # noqa: BLE001
                log.warning("respond failed %s: %r", self.info.session_id, e)
                self.log["error"] = repr(e)[:200]
            await self._send(json.dumps({"type": "turn.end"}))
        finally:
            if drain is not None:
                drain.cancel()
            if not tools_task.done():
                tools_task.cancel()
            await self._close_stt(stt_task)
            log.info("turn %s", json.dumps(self.log, ensure_ascii=False))

    async def _close_stt(self, stt_task: asyncio.Task) -> None:
        stt = self.stt
        if stt is None:
            if stt_task.done() and not stt_task.cancelled() and stt_task.exception() is None:
                stt = stt_task.result()
            else:
                stt_task.cancel()
        if stt is not None:
            try:
                await stt.close()
            except Exception:  # noqa: BLE001
                pass

    async def _drain(self) -> None:
        try:
            async for _ in self.ws:
                pass
        except Exception:  # noqa: BLE001
            pass

    async def _listen(self, stt_task: asyncio.Task) -> bool:
        """말 끝(음성 시작 후 고정 침묵)까지 프레임을 받는다. 말 끝이면 True."""
        vad = self.d.vad_factory()
        speech_run = silence = 0
        heard = False
        stt_failed = False
        try:
            async for msg in self.ws:
                if not isinstance(msg, (bytes, bytearray)):
                    continue
                frame = bytes(msg)
                if self.stt is None and not stt_failed:
                    try:
                        self.stt = await stt_task
                    except Exception as e:  # noqa: BLE001
                        log.error("stt open failed: %r", e)
                        stt_failed = True
                if self.stt is not None:
                    try:
                        await self.stt.send_audio(frame)
                    except Exception as e:  # noqa: BLE001
                        log.warning("stt send failed: %r", e)
                ms = len(frame) * 1000 // (2 * 16000) or FRAME_MS
                if vad.process(frame) >= self.c.vad_threshold:
                    speech_run += ms
                    silence = 0
                    if speech_run >= self.c.start_ms:
                        heard = True
                    continue
                speech_run = 0
                if heard:
                    silence += ms
                    if silence >= self.c.end_silence_ms:
                        return True
        except Exception:  # noqa: BLE001  (연결 끊김)
            return False
        return False

    async def _final_text(self) -> str:
        if self.stt is None:
            return ""
        try:
            return (await asyncio.wait_for(self.stt.finalize(self.c.finalize_timeout_s),
                                           self.c.finalize_timeout_s + 0.5) or "").strip()
        except Exception as e:  # noqa: BLE001
            log.warning("finalize failed: %r", e)
            try:
                return (self.stt.latest_text() or "").strip()
            except Exception:  # noqa: BLE001
                return ""

    async def _respond(self, tools_task: asyncio.Task) -> None:
        assert self.info is not None
        t0 = time.perf_counter()
        text = await self._final_text()
        self.log["stt"] = text
        self.log["t_stt"] = round(time.perf_counter() - t0, 3)
        try:
            tools = await tools_task
        except Exception:  # noqa: BLE001
            tools = []
        system = build_system(self.info, self.d.manual)
        msgs: list[Message] = [Message("user", text or "(음성 인식 결과 없음)")]
        tts_q: asyncio.Queue[str | None] | None = None
        tts_task: asyncio.Task | None = None
        if self.c.tts_mode == "sentence":
            tts_q = asyncio.Queue()
            tts_task = asyncio.create_task(self._tts_worker(tts_q))
        parts: list[str] = []
        calls_log: list[dict] = []
        for rnd in range(self.c.max_rounds):
            last = rnd == self.c.max_rounds - 1
            round_text, calls = await self._llm_round(system, msgs, [] if last else tools, tts_q)
            if round_text.strip():
                parts.append(round_text.strip())
            if not calls:
                break
            msgs.append(Message("assistant", round_text, tool_calls=calls))
            for tc in calls:
                res = await self.d.tools.call(self.info.tools_url, self.info.session_id, tc.name, tc.arguments)
                calls_log.append({"name": tc.name, "arguments": tc.arguments})
                msgs.append(Message("tool", json.dumps(res, ensure_ascii=False), tool_call_id=tc.id, name=tc.name))
        reply = " ".join(parts).strip()
        self.log["calls"] = calls_log
        self.log["reply"] = reply
        self.log["t_llm"] = round(time.perf_counter() - t0, 3)
        if tts_task is not None and tts_q is not None:
            await tts_q.put(None)
            await tts_task
        elif reply:
            await self._speak(reply)
        self.log["t_audio_done"] = round(time.perf_counter() - t0, 3)
        await self._send(json.dumps({"type": "agent.transcript", "text": reply}, ensure_ascii=False))

    async def _llm_round(self, system: str, msgs: list[Message], tools: list[dict],
                         tts_q: asyncio.Queue | None) -> tuple[str, list[ToolCall]]:
        text = ""
        pending = ""                                     # N1: 아직 TTS로 보내지 않은 조각
        calls: list[ToolCall] = []
        async for ev in self.d.llm.stream(system, msgs, tools):
            if ev.type == "text" and ev.text:
                text += ev.text
                if tts_q is not None:
                    pending += ev.text
                    pending = await self._flush_sentences(pending, tts_q)
            elif ev.type == "tool_call" and ev.tool_call is not None:
                calls.append(ev.tool_call)
            elif ev.type == "done":
                break
        if tts_q is not None and pending.strip():
            await tts_q.put(pending.strip())
        return text, calls

    @staticmethod
    async def _flush_sentences(buf: str, q: asyncio.Queue) -> str:
        while True:
            m = _SENT_END.search(buf)
            if not m:
                return buf
            sent, buf = buf[:m.end()].strip(), buf[m.end():]
            if sent:
                await q.put(sent)

    async def _tts_worker(self, q: asyncio.Queue) -> None:
        while True:
            s = await q.get()
            if s is None:
                return
            try:
                await self._speak(s)
            except Exception as e:  # noqa: BLE001
                log.warning("tts failed: %r", e)

    async def _speak(self, text: str) -> None:
        async for pcm in self.d.tts.synthesize(text):
            if pcm:
                await self._send_pcm(pcm)


async def handle(ws: Any, deps: NaiveDeps) -> None:
    await NaiveSession(ws, deps).run()


def build_deps(cfg: NaiveConfig | None = None) -> NaiveDeps:
    """같은 Provider, 우리 보정은 끈다: 숫자 복원·재인식 래퍼 끔, LLM 폴백 없음(make_llm), TTS 레벨 정규화 없음."""
    from agent.providers.factory import ProviderConfig, make_llm, make_stt, make_tts
    from agent.tools_client import ToolsClient
    from agent.turn.vad import SileroVAD

    pc = ProviderConfig.from_env(stt_digit_interim_fix=False, stt_digit_recheck=False,
                                 llm_fallback_models="none", stt_vocabulary="")
    manual = MANUAL_PATH.read_text(encoding="utf-8") if MANUAL_PATH.exists() else ""
    log.info("providers: stt=%s llm=%s tts=%s", pc.stt_live_model, pc.gemini_llm_model, pc.edge_voice)
    return NaiveDeps(stt=make_stt(pc), llm=make_llm(pc), tts=make_tts(pc), tools=ToolsClient(),
                     vad_factory=SileroVAD, manual=manual, cfg=cfg or NaiveConfig())


async def serve(port: int, host: str = "0.0.0.0", deps: NaiveDeps | None = None) -> None:
    import websockets

    deps = deps or build_deps()
    async with websockets.serve(lambda ws: handle(ws, deps), host, port, max_size=None):
        log.info("naive server on :%d (tts_mode=%s)", port, deps.cfg.tts_mode)
        await asyncio.Future()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--tts-mode", choices=["full", "sentence"], default="full",
                    help="full=N0(전체 텍스트 TTS 1회), sentence=N1(문장 단위 스트리밍 TTS)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:  # noqa: BLE001
        pass
    asyncio.run(serve(a.port, a.host, build_deps(NaiveConfig(tts_mode=a.tts_mode))))


if __name__ == "__main__":
    main()
