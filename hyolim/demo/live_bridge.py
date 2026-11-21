"""직접 말해 보는 통화 시험 도구 (로컬 전용, 제출물 아님).

브라우저 마이크 → 이 브리지 → 상담 서버(WebSocket)  /  상담 서버 → 함수 중계(측정기와 같은 지연) → 모의 함수 서버
턴마다 들은 말·AI 답·함수 호출·첫 소리·조회 뒤 답 시간을 demo/sessions/*.jsonl 에 남기고,
python3 demo/score_live.py 로 채점기와 같은 기준(응대·함수·속도)으로 평가한다.

  # 1) 모의 함수 서버와 상담 서버를 먼저 띄운다 (예: OpenAI 임시 백엔드)
  python3 mock_tools/mock_server.py --port 8101
  STT_BACKEND=openai TTS_BACKEND=openai python3 -m server.app --port 9001
  # 2) 브리지
  python3 demo/live_bridge.py --server ws://localhost:9001 --mock http://localhost:8101
  # 3) 브라우저(Chrome)에서 http://localhost:8300 열기 — 이어폰을 쓰면 AI 목소리가 마이크로 다시 들어가지 않는다
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import time
import uuid
from http import HTTPStatus

import numpy as np
import websockets
from websockets.datastructures import Headers
from websockets.http11 import Response

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
_spec = importlib.util.spec_from_file_location("run_eval", os.path.join(ROOT, "eval", "run_eval.py"))
run_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_eval)          # 측정기의 함수 중계(Relay)를 그대로 쓴다 — 함수별 지연·호출 기록 동일

RATE = 16000


def rms(b: bytes) -> float:
    a = np.frombuffer(b, dtype=np.int16).astype(np.float32)
    return float(np.sqrt((a * a).mean())) if len(a) else 0.0


class Call:
    """브라우저 연결 하나 = 통화 하나. 턴마다 상담 서버에 새로 연결한다(측정기 규격과 같음)."""

    def __init__(self, bws, args, relay):
        self.bws, self.args, self.relay = bws, args, relay
        self.history: list[dict] = []
        self.turn = None
        os.makedirs(os.path.join(HERE, "sessions"), exist_ok=True)
        self.path = os.path.join(HERE, "sessions", time.strftime("live_%m%d_%H%M%S.jsonl"))
        self.n = 0

    async def start_turn(self, msg):
        if self.turn:
            return
        self.n += 1
        sid = f"LIVE{self.n:02d}-{uuid.uuid4().hex[:6]}"
        caller = None
        if msg.get("caller_phone"):
            caller = {"name": msg.get("caller_name") or None, "phone": msg["caller_phone"], "email": None}
        start = {"type": "session.start", "session_id": sid, "case_id": sid, "now": msg.get("now") or "2026-10-08 (목) 14:00",
                 "history": list(self.history), "caller": caller, "tools_url": self.relay_url,
                 "audio": {"encoding": "pcm_s16le", "sample_rate": RATE, "channels": 1}}
        sws = await websockets.connect(self.args.server, max_size=None, open_timeout=10)
        await sws.send(json.dumps(start, ensure_ascii=False))
        self.turn = {"sid": sid, "start": start, "sws": sws, "user": [], "agent": [], "events": [], "noise": []}
        self.turn["reader"] = asyncio.create_task(self._read_server(self.turn))
        await self.bws.send(json.dumps({"type": "turn_started", "turn": self.n}))

    async def feed(self, pcm: bytes):
        t = self.turn
        if not t:
            return
        now = time.perf_counter()
        r = rms(pcm)
        if len(t["noise"]) < 15:
            t["noise"].append(r)                     # 처음 0.3초로 배경 소음 수준을 잰다
        t["user"].append((now, r))
        try:
            await t["sws"].send(pcm)
        except Exception:
            pass

    async def _read_server(self, t):
        try:
            async for m in t["sws"]:
                now = time.perf_counter()
                if isinstance(m, bytes):
                    t["agent"].append((now, m))
                    await self.bws.send(m)
                else:
                    e = json.loads(m)
                    t["events"].append({"t": now, **e})
                    await self.bws.send(json.dumps({"type": "server", "event": e}, ensure_ascii=False))
                    if e.get("type") == "turn.end":
                        break
        except Exception as ex:
            t["events"].append({"t": time.perf_counter(), "type": "error", "text": repr(ex)[:200]})
        await self._finish(t)

    async def _finish(self, t):
        await asyncio.sleep(0.2)
        try:
            await t["sws"].close()
        except Exception:
            pass
        calls = self.relay.take(t["sid"])
        noise = float(np.median(t["noise"])) if t["noise"] else 0.0
        thr = max(500.0, noise * 3)
        voiced_agent = [ts for ts, b in t["agent"] if rms(b) >= run_eval.SILENCE_RMS]
        t_first = voiced_agent[0] if voiced_agent else None
        user_voiced = [ts for ts, r in t["user"] if r >= thr]
        before = [ts for ts in user_voiced if t_first is None or ts < t_first]
        speech_end = before[-1] + 0.02 if before else None
        first_sound = (t_first - speech_end) if (t_first and speech_end) else None
        # 첫 소리 직전 0.15초 안에 손님 목소리가 있었으면 말 도중 끼어든 것으로 본다
        interrupt = bool(t_first and any(t_first - 0.15 <= ts < t_first for ts in user_voiced))
        wall0 = time.time() - time.perf_counter()
        after_tool = None
        if calls:
            last = max(c["t_response"] for c in calls) - wall0
            after = [ts for ts in voiced_agent if ts > last]
            after_tool = (after[0] - last) if after else None
        heard = next((e.get("text") for e in t["events"] if e.get("type") == "debug.heard"), "")
        said = " ".join(e.get("text", "") for e in t["events"] if e.get("type") == "agent.transcript")
        rec = {"turn": self.n, "session_id": t["sid"], "now": t["start"]["now"], "caller": t["start"]["caller"],
               "history": t["start"]["history"], "heard": heard, "truth": None, "said": said,
               "tool_calls": [{"name": c["name"], "arguments": c["arguments"], "status": c["status"],
                               "matched": c["matched"]} for c in calls],
               "first_sound": first_sound, "after_tool_sound": after_tool, "spoke_before_end": interrupt,
               "errors": [e for e in t["events"] if e.get("type") == "error"]}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # 다음 턴의 대화 이력 (측정기 history 형식)
        if heard:
            self.history.append({"role": "customer", "text": heard})
        for c in calls:
            self.history.append({"role": "tool_call", "name": c["name"], "arguments": c["arguments"]})
        if said:
            self.history.append({"role": "agent", "text": said})
        self.turn = None
        await self.bws.send(json.dumps({"type": "turn_done", "record": rec, "log": os.path.relpath(self.path, ROOT)},
                                       ensure_ascii=False))

    async def set_truth(self, turn: int, text: str):
        """손님이 실제로 한 말(정답 글)을 기록해 둔다 — 번호를 제대로 알아들었는지 채점할 때 쓴다."""
        if not os.path.exists(self.path):
            return
        rows = [json.loads(l) for l in open(self.path, encoding="utf-8")]
        for r in rows:
            if r["turn"] == turn:
                r["truth"] = text
        with open(self.path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def page() -> bytes:
    return open(os.path.join(HERE, "live.html"), "rb").read()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="ws://localhost:9001")
    ap.add_argument("--mock", default="http://localhost:8101")
    ap.add_argument("--port", type=int, default=8300)
    ap.add_argument("--relay-port", type=int, default=8301)
    args = ap.parse_args()
    relay = run_eval.Relay(args.mock, args.relay_port)
    relay_url = f"http://localhost:{args.relay_port}"

    def process_request(connection, request):
        if request.path in ("/", "/index.html"):
            return Response(HTTPStatus.OK.value, "OK", Headers([("Content-Type", "text/html; charset=utf-8")]), page())
        return None

    async def handler(bws):
        call = Call(bws, args, relay)
        call.relay_url = relay_url
        async for m in bws:
            if isinstance(m, bytes):
                await call.feed(m)
                continue
            msg = json.loads(m)
            if msg.get("type") == "start_turn":
                await call.start_turn(msg)
            elif msg.get("type") == "truth":
                await call.set_truth(int(msg["turn"]), msg.get("text", ""))

    async with websockets.serve(handler, "127.0.0.1", args.port, max_size=None, process_request=process_request):
        print(f"브라우저에서 http://localhost:{args.port} 를 여세요  (상담 서버 {args.server}, 모의 함수 {args.mock})", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
