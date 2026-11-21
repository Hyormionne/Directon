"""Soniox 실시간 STT (docs/research/01_STT_턴감지.md 1절).

- `wss://stt-rt.soniox.com/transcribe-websocket`, `Authorization: Bearer` 헤더(시작 메시지의 api_key는 쓰지 않는다)
- 모델 `stt-rt-v6`, pcm_s16le 16k mono, language_hints ["ko"]
- final 토큰은 누적, non-final은 응답마다 교체. `<end>`/`<fin>`은 텍스트에서 뺀다.
- finalize: `{"type":"finalize"}` → `<fin>` 수신까지 대기
- 오디오가 keepalive_s 동안 없으면 `{"type":"keepalive"}`
- 종료: 빈 텍스트 프레임 ""
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Awaitable, Callable

from agent.core.types import SAMPLE_RATE

log = logging.getLogger(__name__)

SONIOX_URL = "wss://stt-rt.soniox.com/transcribe-websocket"
SPECIAL = {"<end>", "<fin>"}

Connect = Callable[[str, dict], Awaitable[Any]]


async def _ws_connect(url: str, headers: dict) -> Any:
    import websockets
    return await websockets.connect(url, additional_headers=headers, max_size=2 ** 22,
                                    ping_interval=20)


class SonioxStream:
    def __init__(self, ws: Any, keepalive_s: float):
        self._ws = ws
        self._keepalive_s = keepalive_s
        self._final = ""
        self._nonfinal = ""
        self._fin = asyncio.Event()
        self._dead = False
        self._closed = False
        self._error: str = ""
        self._lock = asyncio.Lock()
        loop = asyncio.get_running_loop()
        self._last_send = loop.time()
        # 로그용 집계(동작에는 쓰지 않는다): 글이 빈 채 끝난 턴이 게이트웨이 탓인지 본 측정 로그에서 구분한다
        self._opened = loop.time()
        self._audio_bytes = 0
        self._tokens = 0                # 받은 글 토큰 수(<end>/<fin> 제외)
        self._finalizes = 0             # 끝맺음 요청 횟수
        self._fin_timeouts = 0
        self._rx = asyncio.create_task(self._recv_loop())
        self._ka = asyncio.create_task(self._keepalive_loop())

    async def _send(self, data: Any) -> None:
        if self._dead:
            return
        try:
            async with self._lock:
                await self._ws.send(data)
            self._last_send = asyncio.get_running_loop().time()
        except Exception as e:
            log.warning("soniox send error: %s", e)
            self._dead = True
            self._fin.set()

    async def _recv_loop(self) -> None:
        try:
            while True:
                raw = await self._ws.recv()
                res = json.loads(raw)
                if res.get("error_code"):
                    self._error = f"{res.get('error_code')} {res.get('error_type')}: {res.get('error_message')}"
                    log.warning("soniox error %s", self._error)
                    break
                nonfinal = []
                for t in res.get("tokens", []):
                    txt = t.get("text", "")
                    if txt == "<fin>":
                        self._fin.set()
                        continue
                    if txt in SPECIAL:
                        continue
                    self._tokens += 1
                    if t.get("is_final"):
                        self._final += txt
                    else:
                        nonfinal.append(txt)
                self._nonfinal = "".join(nonfinal)
                if res.get("finished"):
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not self._closed:
                log.warning("soniox recv error: %s", e)
        finally:
            self._dead = True
            self._fin.set()

    async def _keepalive_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._dead:
            await asyncio.sleep(self._keepalive_s / 2)
            if loop.time() - self._last_send >= self._keepalive_s:
                await self._send(json.dumps({"type": "keepalive"}))

    async def send_audio(self, pcm: bytes) -> None:
        if pcm and not self._closed:
            self._audio_bytes += len(pcm)
            await self._send(pcm)

    def latest_text(self) -> str:
        return (self._final + self._nonfinal).strip()

    def _stats(self) -> str:
        """경고 로그에 붙이는 집계. 고객 말 내용은 넣지 않는다(글자 수·시간만)."""
        now = asyncio.get_running_loop().time()
        return (f"chars={len(self.latest_text())} tokens={self._tokens} "
                f"audio={self._audio_bytes / (2 * SAMPLE_RATE):.2f}s open={now - self._opened:.2f}s "
                f"finalize={self._finalizes} fin_timeout={self._fin_timeouts} dead={self._dead}")

    async def finalize(self, timeout: float = 1.0) -> str:
        if not self._dead:
            self._finalizes += 1
            self._fin.clear()
            await self._send(json.dumps({"type": "finalize"}))
            try:
                await asyncio.wait_for(self._fin.wait(), timeout)
            except asyncio.TimeoutError:
                self._fin_timeouts += 1
                log.warning("soniox finalize timeout after %.2fs: %s", timeout, self._stats())
        # <fin> 이후에는 non-final이 비어 있다. 시간 초과·끊김이면 지금까지의 최선.
        return self.latest_text()

    async def close(self) -> None:
        if self._closed:
            return
        if self._finalizes and self._tokens == 0:       # 끝맺음까지 요청했는데(말소리가 있었다) 글 토큰이 하나도 안 왔다
            log.warning("soniox stream ended without tokens: %s", self._stats())
        self._closed = True
        await self._send("")
        for t in (self._ka, self._rx):
            t.cancel()
        for t in (self._ka, self._rx):
            try:
                await t
            except BaseException:
                pass
        try:
            await self._ws.close()
        except Exception:
            pass


class SonioxSTT:
    def __init__(self, api_key: str = "", connect: Connect | None = None, model: str = "stt-rt-v6",
                 language_hints: list[str] | None = None, keepalive_s: float = 8.0,
                 url: str = SONIOX_URL, endpoint_detection: bool = False):
        self.api_key = api_key or os.environ.get("SONIOX_API_KEY", "")
        self.connect = connect or _ws_connect
        self.model = model
        self.language_hints = language_hints or ["ko"]
        self.keepalive_s = keepalive_s
        self.url = url
        self.endpoint_detection = endpoint_detection

    def start_message(self, context: str = "") -> dict:
        msg: dict = {
            "model": self.model,
            "audio_format": "pcm_s16le",
            "sample_rate": SAMPLE_RATE,
            "num_channels": 1,
            "language_hints": self.language_hints,
            "enable_endpoint_detection": self.endpoint_detection,
        }
        ctx: dict = {"general": [{"key": "domain", "value": "쇼핑몰 고객센터 상담"}]}
        terms = [t.strip() for t in context.replace("\n", ",").split(",") if t.strip()]
        if terms:
            ctx["terms"] = terms
        msg["context"] = ctx
        return msg

    async def open_stream(self, context: str = "") -> SonioxStream:
        ws = await self.connect(self.url, {"Authorization": f"Bearer {self.api_key}"})
        await ws.send(json.dumps(self.start_message(context), ensure_ascii=False))
        return SonioxStream(ws, self.keepalive_s)
