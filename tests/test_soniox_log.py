"""Soniox 스트림 경고 로그: 끝맺음 시간 초과, 토큰 없이 끝난 스트림(2026-11-21 공식 측정 P0029-01·P0120-01을
본 측정에서 구분하기 위한 것). 동작은 그대로이고, 고객 말 내용은 로그에 넣지 않는다(글자 수·시간만)."""
import asyncio
import json
import logging

from agent.providers.soniox_stt import SonioxSTT

LOGGER = "agent.providers.soniox_stt"
SECRET = "환불하고싶어요"


class QuietWS:
    """끝맺음 요청에 <fin>을 주지 않는 웹소켓. tokens를 주면 오디오 첫 조각 뒤에 그 토큰만 준다."""

    def __init__(self, tokens=None, fin=False):
        self.sent = []
        self.q: asyncio.Queue = asyncio.Queue()
        self.tokens = tokens
        self.fin = fin

    async def send(self, data):
        self.sent.append(data)
        if isinstance(data, bytes) and self.tokens and len([d for d in self.sent if isinstance(d, bytes)]) == 1:
            self.q.put_nowait(json.dumps({"tokens": self.tokens}, ensure_ascii=False))
        elif self.fin and isinstance(data, str) and data and json.loads(data).get("type") == "finalize":
            self.q.put_nowait(json.dumps({"tokens": [{"text": "<fin>", "is_final": True}]}))

    async def recv(self):
        m = await self.q.get()
        if m is None:
            raise ConnectionError("closed")
        return m

    async def close(self):
        self.q.put_nowait(None)


async def open_stream(ws):
    async def connect(url, headers):
        return ws

    return await SonioxSTT(api_key="k", connect=connect).open_stream()


def warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.name == LOGGER and r.levelno >= logging.WARNING]


async def test_finalize_timeout_without_tokens_is_logged(caplog):
    caplog.set_level(logging.WARNING, logger=LOGGER)
    stream = await open_stream(QuietWS())
    for _ in range(50):                                  # 1초 분량
        await stream.send_audio(b"\x10\x27" * 320)
    assert await stream.finalize(timeout=0.05) == ""     # 동작은 그대로: 시간 초과면 지금까지의 글
    await stream.close()
    msgs = warnings(caplog)
    timeout = [m for m in msgs if "finalize timeout" in m]
    assert len(timeout) == 1
    assert "chars=0" in timeout[0] and "tokens=0" in timeout[0] and "audio=1.00s" in timeout[0]
    ended = [m for m in msgs if "without tokens" in m]
    assert len(ended) == 1 and "audio=1.00s" in ended[0]


async def test_finalize_timeout_logs_length_not_customer_text(caplog):
    caplog.set_level(logging.WARNING, logger=LOGGER)
    stream = await open_stream(QuietWS(tokens=[{"text": SECRET, "is_final": False}]))
    await stream.send_audio(b"\x10\x27" * 320)
    for _ in range(20):
        if stream.latest_text():
            break
        await asyncio.sleep(0.005)
    assert await stream.finalize(timeout=0.05) == SECRET
    await stream.close()
    msgs = warnings(caplog)
    assert len(msgs) == 1 and "finalize timeout" in msgs[0]              # 토큰이 왔으니 '토큰 없이 끝남'은 아니다
    assert f"chars={len(SECRET)}" in msgs[0] and "tokens=1" in msgs[0]
    assert SECRET not in caplog.text


async def test_no_warning_on_normal_finalize_or_unused_stream(caplog):
    caplog.set_level(logging.WARNING, logger=LOGGER)
    ok = await open_stream(QuietWS(tokens=[{"text": "네", "is_final": True}], fin=True))
    await ok.send_audio(b"\x10\x27" * 320)
    assert await ok.finalize(timeout=1.0) == "네"
    await ok.close()
    idle = await open_stream(QuietWS())                  # 예열·대기용으로 열었다 닫은 스트림(끝맺음 요청 없음)
    await idle.send_audio(b"\x00\x00" * 320)
    await idle.close()
    assert warnings(caplog) == []


async def test_fin_without_any_token_is_logged_at_close(caplog):
    """<fin>은 제때 왔지만 토큰이 하나도 없던 스트림도 닫을 때 한 줄 남긴다."""
    caplog.set_level(logging.WARNING, logger=LOGGER)
    stream = await open_stream(QuietWS(fin=True))
    await stream.send_audio(b"\x10\x27" * 320 * 25)      # 0.5초 분량
    assert await stream.finalize(timeout=1.0) == ""
    await stream.close()
    msgs = warnings(caplog)
    assert len(msgs) == 1 and "without tokens" in msgs[0] and "audio=0.50s" in msgs[0] and "finalize=1" in msgs[0]
