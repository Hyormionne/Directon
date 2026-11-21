"""AG-T29: Soniox 엔드포인트 평가기 (fake websocket, 네트워크 없음)."""
import asyncio
import json

import pytest

from agent.providers.soniox_stt import SonioxSTT
from eval.soniox_endpoint_eval import FRAME_BYTES, Pause, classify_endpoint, stream_endpoints


def test_classify_ok_latency():
    r = classify_endpoint("u", [3.6, 5.0], end_t=3.0, pauses=[])
    assert r.outcome == "ok" and r.latency_ms == pytest.approx(600.0)


def test_classify_false_cut_in_pause_uses_cut_type():
    r = classify_endpoint("u", [2.0], end_t=4.0, pauses=[Pause(1.5, 800, "connective")])
    assert r.outcome == "false_cut" and r.where == "connective"


def test_classify_false_cut_outside_pause_is_intra():
    r = classify_endpoint("u", [1.0], end_t=4.0, pauses=[Pause(2.0, 500, "digits")])
    assert r.outcome == "false_cut" and r.where == "intra"


def test_classify_late_end_is_miss_with_late_ms():
    r = classify_endpoint("u", [5.0], end_t=3.0, pauses=[], window_s=1.5)
    assert r.outcome == "miss" and r.late_ms == pytest.approx(2000.0)


def test_classify_no_end_is_miss():
    assert classify_endpoint("u", [], end_t=3.0, pauses=[]).outcome == "miss"


class FakeWS:
    """오디오 프레임 n개를 받으면 <end>를 보낸다. 빈 텍스트 프레임이면 finished."""

    def __init__(self, end_after_frames: int):
        self.sent: list = []
        self.q: asyncio.Queue = asyncio.Queue()
        self.frames = 0
        self.end_after = end_after_frames

    async def send(self, data):
        self.sent.append(data)
        if isinstance(data, bytes):
            self.frames += 1
            if self.frames == self.end_after:
                await self.q.put(json.dumps({"tokens": [{"text": "안녕", "is_final": True},
                                                        {"text": "<end>", "is_final": True}]}))
        elif data == "":
            await self.q.put(json.dumps({"tokens": [], "finished": True}))

    async def recv(self):
        return await self.q.get()

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_stream_endpoints_enables_detection_and_records_end_time():
    ws = FakeWS(end_after_frames=10)

    async def connect(url, headers):
        assert headers["Authorization"] == "Bearer k"
        return ws

    clock_t = [0.0]

    async def fake_sleep(s):
        clock_t[0] += s
        await asyncio.sleep(0)

    stt = SonioxSTT(api_key="k", connect=connect)
    pcm = b"\x01\x00" * (FRAME_BYTES // 2) * 20
    ends, text = await stream_endpoints(stt, pcm, until_s=0.6, extra={"max_endpoint_delay_ms": 500},
                                        clock=lambda: clock_t[0], sleep=fake_sleep)
    start = json.loads(ws.sent[0])
    assert start["enable_endpoint_detection"] is True
    assert start["max_endpoint_delay_ms"] == 500
    assert text == "안녕"
    assert len(ends) == 1 and ends[0] == pytest.approx(0.18, abs=0.03)
    # 오디오 20프레임 후 무음으로 until_s(30프레임)까지, 마지막은 빈 텍스트 프레임
    assert sum(isinstance(d, bytes) for d in ws.sent) == 30 and ws.sent[-1] == ""


@pytest.mark.asyncio
async def test_stream_endpoints_raises_on_error_message():
    class ErrWS(FakeWS):
        async def send(self, data):
            self.sent.append(data)
            if len(self.sent) == 2:
                await self.q.put(json.dumps({"error_code": 402, "error_message": "balance"}))

    ws = ErrWS(0)

    async def connect(url, headers):
        return ws

    with pytest.raises(RuntimeError, match="402"):
        await stream_endpoints(SonioxSTT(api_key="k", connect=connect), b"\x00" * FRAME_BYTES * 5,
                               until_s=0.1, speed=0)
