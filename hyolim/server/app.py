"""상담 서버 (WebSocket, 9000번).

한 연결 = 손님 말 한 번:
  session.start → 손님 음성(실시간) → [STT + 말 끝 판단] → 채움말 즉시 → fncall(조회·검증·접수)
  → 템플릿 답 또는 LLM 답 → TTS 스트리밍 → turn.end

속도 설계
- 첫 소리: 말 끝 직후 미리 합성해 둔 채움말 PCM 을 바로 보낸다 (TTS 왕복 0).
- 조회 뒤 답: fncall 템플릿이 있으면 LLM 없이 TTS 만. 없으면 LLM 스트리밍 → 문장 단위로 TTS 를 겹쳐 돌린다.
- 조회 후보 여러 개는 병렬로 보내 지연은 한 번만 든다.

실행: python3 -m server.app --port 9000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback

import httpx
import websockets

from . import config as C
from . import llm, tts
from .answer import guarded, plan_answer
from . import usage
from .stt import make_listener
from fncall.pipeline import run_turn
from fncall.planner import plan_lookups, suggest_commit, topic
from fncall.slots import resolve
from fncall.verify import ToolClient

ASSETS = os.path.join(os.path.dirname(__file__), "assets") if C.TTS_BACKEND == "gemini" \
    else os.path.join(os.path.dirname(__file__), "assets", C.TTS_BACKEND)      # 음성이 백엔드마다 달라 폴더를 나눈다
FILLER_TEXT = {
    "ack": "네, 고객님.",
    "lookup_order": "주문 내역 확인해 볼게요.",          # '네, 고객님.' 뒤에 이어지므로 '네'를 빼 둔다
    "lookup_return": "반품 진행 상황 확인해 볼게요.",
    "commit": "바로 접수해 드릴게요.",
    "callback": "연락처 남겨 드릴게요.",
}
FALLBACK_TEXT = {
    "not_heard": "죄송해요, 잘 못 들었어요. 한 번만 다시 말씀해 주시겠어요?",
    "error": "죄송해요, 지금 확인이 어려워요. 상담원에게 연결해 드릴게요.",
}
FILLERS: dict[str, bytes] = {}
TURN_BUDGET = 20.0                     # 25초 안에 turn.end (여유 5초)


def trim_silence(pcm: bytes, thr: float = 300.0, keep_ms: int = 20) -> bytes:
    """채움말 앞뒤 무음을 잘라 첫 소리를 당긴다 (측정기는 소리가 담긴 첫 조각부터 잰다)."""
    import numpy as np
    a = np.frombuffer(pcm, dtype=np.int16)
    if not len(a):
        return pcm
    win = 160                                            # 10ms
    n = len(a) // win
    if n == 0:
        return pcm
    e = np.sqrt((a[: n * win].astype(np.float32).reshape(n, win) ** 2).mean(1))
    loud = np.where(e >= thr)[0]
    if not len(loud):
        return pcm
    s = max(0, loud[0] * win - keep_ms * 16)
    t = min(len(a), (loud[-1] + 1) * win + 60 * 16)
    return a[s:t].tobytes()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


async def load_assets(http: httpx.AsyncClient):
    """채움말 PCM 을 디스크에서 읽고, 없으면 지금 합성해서 저장한다."""
    os.makedirs(ASSETS, exist_ok=True)
    for k, text in {**FILLER_TEXT, **FALLBACK_TEXT}.items():
        p = os.path.join(ASSETS, f"{k}.pcm")
        if os.path.exists(p) and not C.FAKE:
            FILLERS[k] = trim_silence(open(p, "rb").read())
            continue
        try:
            FILLERS[k] = trim_silence(await tts.synth(http, text))
            if not C.FAKE and FILLERS[k]:
                open(p, "wb").write(FILLERS[k])
        except Exception as e:
            log("filler synth failed", k, repr(e)[:120])


class Turn:
    def __init__(self, ws, http, start):
        self.ws, self.http, self.start = ws, http, start
        self.sent_audio = False
        self.said: list[str] = []

    async def send_pcm(self, pcm: bytes, step: int = 16000):   # 0.5초 단위로 나눠 보낸다
        for i in range(0, len(pcm), step):
            await self.ws.send(pcm[i:i + step])
        self.sent_audio = self.sent_audio or bool(pcm)

    async def say_text(self, text: str):
        await self.ws.send(json.dumps({"type": "agent.transcript", "text": text}, ensure_ascii=False))
        self.said.append(text)

    async def speak(self, text: str):
        await self.say_text(text)
        async for chunk in tts.synth_stream(self.http, text):
            await self.send_pcm(chunk)

    async def speak_stream(self, sentence_iter):
        """문장이 나오는 대로 TTS 를 미리 돌리고, 순서대로 보낸다."""
        q: asyncio.Queue = asyncio.Queue()

        async def produce():
            async for s in sentence_iter:
                buf: asyncio.Queue = asyncio.Queue()

                async def run(s=s, buf=buf):
                    try:
                        async for c in tts.synth_stream(self.http, s):
                            await buf.put(c)
                    finally:
                        await buf.put(None)
                await q.put((s, buf, asyncio.create_task(run())))
            await q.put(None)

        prod = asyncio.create_task(produce())
        try:
            while (item := await q.get()) is not None:
                s, buf, _task = item
                await self.say_text(s)
                while (c := await buf.get()) is not None:
                    await self.send_pcm(c)
        finally:
            prod.cancel()

    async def filler(self, key: str):
        pcm = FILLERS.get(key)
        if pcm:
            await self.say_text(FILLER_TEXT.get(key, FALLBACK_TEXT.get(key, "")))
            await self.send_pcm(pcm)
        else:
            await self.speak(FILLER_TEXT.get(key) or FALLBACK_TEXT[key])


def pick_filler(session, utt) -> str | None:
    slots = resolve(session, utt)
    looks = plan_lookups(session, utt, slots)
    if any(c.name == "get_return_info" for c in looks):
        return "lookup_return"
    if looks:
        return "lookup_order"
    commit, why = suggest_commit(session, utt, slots)
    if commit:
        return "callback" if commit.name == "request_callback" else "commit"
    if why == "lookup_first":
        return "lookup_order"
    return "ack" if C.FILLER_ALWAYS else None


async def respond(turn: Turn, session: dict, utt: str, ack_played: bool = False, t_end: float | None = None):
    t_end = t_end or time.perf_counter()
    t_utt = time.perf_counter()
    if not utt.strip():
        await turn.filler("not_heard")
        return
    async with ToolClient(session["tools_url"], session["session_id"]) as tc:
        work = asyncio.create_task(run_turn(session, utt, tc.call))     # 조회는 지금 바로 출발
        fk = pick_filler(session, utt)
        if ack_played and fk == "ack":
            fk = None                                    # 맞장구는 말 끝에서 이미 냈다
        if fk:
            await turn.filler(fk)
        fk = fk or ("ack" if ack_played else None)
        out = await work
    t_tool = time.perf_counter()
    plan = plan_answer(session, utt, out)
    if plan.mode in ("safe", "template"):
        await turn.speak(plan.text)
    else:
        await turn.speak_stream(guarded(plan, llm.stream_answer(turn.http, plan.messages), ack_played=bool(fk)))
    log(session.get("case_id"), f"utt={utt[:40]!r}", f"calls={[c.name for c, _ in out.calls]}",
        f"{plan.mode} risk={plan.risk.level}{plan.risk.reasons or ''}{' guard=' + str(plan.guarded) if plan.guarded else ''}",
        f"stt_wait={t_utt - t_end:.2f}s tool={t_tool - t_utt:.2f}s total={time.perf_counter() - t_end:.2f}s")


_FAKE_TX: dict | None = None


def _fake_transcript(case_id):
    global _FAKE_TX
    if _FAKE_TX is None:
        _FAKE_TX = {}
        for line in open(os.environ["FAKE_TRANSCRIPTS"], encoding="utf-8"):
            c = json.loads(line)
            _FAKE_TX[c["case_id"]] = c["customer_utterance"]
    return _FAKE_TX.get(case_id)


async def handler(ws, http):
    try:
        start = json.loads(await ws.recv())
    except Exception:
        return
    if start.get("type") != "session.start":
        return
    session = start
    listener = make_listener()
    turn_stats = {"chars": 0, "cached_chars": 0}
    tts.TURN.set(turn_stats)                 # 이 연결(턴)에서 만든 작업들은 이 dict 에 센다
    stt_task = asyncio.create_task(listener.start())
    turn = Turn(ws, http, start)
    drain = None
    try:
        async for m in ws:                                   # 20ms 음성 조각
            if isinstance(m, bytes):
                await listener.feed(m)
                if listener.endpoint.is_set():
                    break

        async def _drain():                                  # 측정기는 끝까지 무음을 보낸다
            async for _ in ws:
                pass
        drain = asyncio.create_task(_drain())
        t_end = time.perf_counter()
        ack_played = False
        if C.FILLER_ALWAYS and "ack" in FILLERS:
            await turn.filler("ack")                         # 말 끝 즉시 '네, 고객님.' — 받아쓰기를 기다리지 않는다 (첫 소리)
            ack_played = True
        utt = await listener.wait_transcript()
        try:
            await ws.send(json.dumps({"type": "debug.heard", "text": utt}, ensure_ascii=False))
        except Exception:
            pass
        if C.FAKE and os.environ.get("FAKE_TRANSCRIPTS"):
            utt = _fake_transcript(start.get("case_id")) or utt   # 로컬 배관 시험 전용 (완벽한 STT 가정)
        if listener.error:
            log(start.get("case_id"), "STT:", listener.error)
        asyncio.create_task(listener.close())
        try:
            await asyncio.wait_for(respond(turn, session, utt, ack_played, t_end), TURN_BUDGET)
        except Exception as e:
            log(start.get("case_id"), "respond failed:", repr(e)[:200])
            traceback.print_exc()
            if not turn.sent_audio and "error" in FILLERS:
                await turn.send_pcm(FILLERS["error"])
        await ws.send(json.dumps({"type": "turn.end"}))
        usage.add_turn(start.get("case_id") or "", listener.sent_bytes / (C.RATE * 2),
                       turn_stats["chars"], turn_stats["cached_chars"])
    except websockets.ConnectionClosed:
        pass
    finally:
        stt_task.cancel()
        if drain:
            drain.cancel()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9000)
    a = ap.parse_args()
    limits = httpx.Limits(max_connections=64, max_keepalive_connections=32, keepalive_expiry=120)
    async with httpx.AsyncClient(limits=limits, timeout=15) as http:
        missing = [k for k, v in {"SONIOX_API_KEY": C.SONIOX_API_KEY, "OPENAI_API_KEY": C.OPENAI_API_KEY,
                                  "GEMINI_API_KEY": C.GEMINI_API_KEY}.items() if not v]
        if missing and not C.FAKE:
            log("경고: API 키 없음", missing)
        async def _assets():
            await load_assets(http)
            log(f"fillers ready: {sorted(FILLERS)}  fake={C.FAKE}")
        # 서버는 바로 연다. 채움말이 아직 없으면 그 자리에서 합성한다(조금 느릴 뿐).
        # 제출 전 `python3 -m server.make_assets` 로 server/assets/*.pcm 을 만들어 함께 넣어 두면 즉시 쓴다.
        asyncio.create_task(_assets())
        async with websockets.serve(lambda ws: handler(ws, http), "0.0.0.0", a.port, max_size=None):
            log(f"voice cs server on :{a.port}")
            await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
