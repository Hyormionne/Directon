"""AG-T14 데모용 디버그 이벤트: session.start에 "debug": true가 있을 때만 debug.* 텍스트 이벤트를 보낸다.

debug가 없거나 false면 측정기 경로와 바이트 단위로 같아야 한다(메시지 순서·종류·내용·송신 시점 프레임).
fake는 tests/test_session.py의 것을 그대로 쓴다(그 파일은 수정하지 않는다).
"""
import json

import httpx

from agent.core.types import LLMEvent, ToolCall
from test_session import (OPENER_PCM, TOOLS, FakeDetector, FakeLLM, FakeSTT, FakeWS, make_deps,
                          pcm_for, run, start_msg, text_events, utterance)


def _texts(ws):
    return [json.loads(m) for _, m in ws.sent if isinstance(m, str)]


def _non_debug(ws):
    out = []
    for i, m in ws.sent:
        if isinstance(m, str) and json.loads(m).get("type", "").startswith("debug."):
            continue
        out.append(m)
    return out


async def _run_once(tmp_path, sub, start, *, llm_script=None, tool_handler=None):
    ws = FakeWS(start, utterance(50))
    llm = FakeLLM(*(llm_script or [text_events("주문하신 상품은 내일 도착합니다.")]))
    deps, _ = make_deps(tmp_path / sub, ws=ws, llm=llm, tool_handler=tool_handler)
    await run(ws, deps)
    return ws


async def test_debug_absent_or_false_is_byte_identical(tmp_path):
    base = await _run_once(tmp_path, "a", start_msg())
    off = await _run_once(tmp_path, "b", start_msg(debug=False))
    assert base.sent == off.sent                      # 송신 시점 프레임까지 동일
    assert not any(t["type"].startswith("debug.") for t in _texts(base))
    assert [t["type"] for t in _texts(base)] == ["agent.transcript", "turn.end"]


async def test_debug_on_adds_events_without_changing_protocol_messages(tmp_path):
    base = await _run_once(tmp_path, "a", start_msg())
    dbg = await _run_once(tmp_path, "b", start_msg(debug=True))
    # 측정기 메시지(오디오·transcript·turn.end)는 순서·내용 그대로
    assert _non_debug(dbg) == [m for _, m in base.sent]
    # 오디오 송신 시점(프레임)도 같다
    assert [(i, m) for i, m in dbg.sent if isinstance(m, bytes)] == base.binaries()

    texts = _texts(dbg)
    types = [t["type"] for t in texts]
    turn = [t for t in texts if t["type"] == "debug.turn"]
    events = [t["event"] for t in turn]
    assert events[0] == "speech_start"
    assert "silence" in events
    assert events[-1] == "decided"
    decided = turn[-1]
    assert decided["reason"] == "silence"
    assert decided["silence_ms"] == 600
    assert decided["text"] == "배송 언제 와요?"

    stt = [t for t in texts if t["type"] == "debug.stt"]
    assert stt == [{"type": "debug.stt", "text": "배송 언제 와요?"}]

    # 순서: speech_start … decided → debug.stt → agent.transcript → turn.end → debug.timing
    assert types.index("debug.stt") < types.index("agent.transcript") < types.index("turn.end")
    assert types[-1] == "debug.timing"
    timing = texts[-1]
    for k in ("commit_to_opener_ms", "commit_to_llm_first_ms", "commit_to_response_audio_ms",
              "commit_to_turn_end_ms", "silence_to_commit_ms", "decision_reason"):
        assert k in timing, k
    assert timing["commit_to_opener_ms"] is not None and timing["commit_to_opener_ms"] >= 0
    assert timing["commit_to_turn_end_ms"] >= timing["commit_to_opener_ms"]

    # 확정 전에는 오디오 없음(디버그 이벤트는 텍스트라 허용)
    first_audio_idx = min(i for i, m in dbg.sent if isinstance(m, bytes))
    decided_idx = next(i for i, m in dbg.sent if isinstance(m, str)
                       and json.loads(m).get("event") == "decided")
    assert decided_idx == first_audio_idx == 80
    assert dbg.binaries()[0] == (80, OPENER_PCM)


async def test_debug_tool_event(tmp_path):
    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json=TOOLS)
        return httpx.Response(200, json={"status": "ok", "data": {"status": "배송중"}})

    script = [[LLMEvent("tool_call", tool_call=ToolCall("c1", "get_order_info", {"order_id": "20260101-0000001"})),
               LLMEvent("done")], text_events("배송 중입니다.")]
    dbg = await _run_once(tmp_path, "a", start_msg(debug=True), llm_script=script, tool_handler=handler)
    texts = _texts(dbg)
    tools = [t for t in texts if t["type"] == "debug.tool"]
    assert tools == [{"type": "debug.tool", "name": "get_order_info", "arguments": {"order_id": "20260101-0000001"},
                      "status": "ok"}]
    types = [t["type"] for t in texts]
    assert types.index("debug.tool") < types.index("agent.transcript")

    base = await _run_once(tmp_path, "b", start_msg(), llm_script=script, tool_handler=handler)
    assert _non_debug(dbg) == [m for _, m in base.sent]
    assert pcm_for("배송 중입니다.") in [m for _, m in base.binaries()]


async def test_debug_multi_turn(tmp_path):
    ws = FakeWS(start_msg(debug=True), utterance(30), utterance(30), turns=2)
    llm = FakeLLM(text_events("첫 답변입니다."), text_events("두 번째 답변입니다."))
    stt = FakeSTT(texts=[("첫 질문", "첫 질문"), ("두번째 질문", "두번째 질문")])
    deps, _ = make_deps(tmp_path, ws=ws, llm=llm, stt=stt, detector=FakeDetector(None))
    await run(ws, deps)
    texts = _texts(ws)
    assert [t["text"] for t in texts if t["type"] == "debug.stt"] == ["첫 질문", "두번째 질문"]
    proto = [t["type"] for t in texts if not t["type"].startswith("debug.")]
    assert proto == ["agent.transcript", "turn.end", "agent.transcript", "turn.end"]
    assert sum(1 for t in texts if t["type"] == "debug.timing") == 2
    decided = [t for t in texts if t.get("event") == "decided"]
    assert len(decided) == 2
