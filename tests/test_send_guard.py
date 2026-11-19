"""세션 송신 견고성(AG-T18 요청, session.py): ConnectionClosed가 아닌 송신 오류가 나도 turn.end를 보내고
다음 턴을 받을 수 있어야 한다. 프레임 루프가 예외로 끝나면 응답 상태를 정리한다."""
import json

from agent.session import Session
from test_session import FakeLLM, FakeSTT, FakeWS, make_deps, run, start_msg, text_events, utterance


class FlakyWS(FakeWS):
    """처음 fail_n번의 바이너리 송신에서 RuntimeError(연결은 살아 있음)."""

    def __init__(self, *a, fail_n=1, fail_text=None, **kw):
        super().__init__(*a, **kw)
        self.fail_n = fail_n
        self.fail_text = fail_text
        self.errors = 0

    async def send(self, msg):
        if isinstance(msg, bytes) and self.fail_n > 0:
            self.fail_n -= 1
            self.errors += 1
            raise RuntimeError("send glitch")
        if isinstance(msg, str) and self.fail_text and json.loads(msg).get("type") == self.fail_text:
            self.fail_text = None
            self.errors += 1
            raise RuntimeError("send glitch")
        await super().send(msg)


async def test_audio_send_error_still_ends_turn_and_next_turn_works(tmp_path):
    ws = FlakyWS(start_msg(), utterance(30), utterance(30), turns=2, fail_n=1)
    llm = FakeLLM(text_events("첫 답변입니다."), text_events("두 번째 답변입니다."))
    stt = FakeSTT(texts=[("첫 질문", "첫 질문"), ("두번째 질문", "두번째 질문")])
    from test_session import FakeDetector
    deps, _ = make_deps(tmp_path, ws=ws, llm=llm, stt=stt, detector=FakeDetector(None))
    await run(ws, deps)
    assert ws.errors == 1
    types = [t["type"] for t in ws.texts()]
    assert types == ["agent.transcript", "turn.end", "agent.transcript", "turn.end"]


async def test_transcript_send_error_still_sends_turn_end(tmp_path):
    ws = FlakyWS(start_msg(), utterance(50), fail_n=0, fail_text="agent.transcript")
    deps, _ = make_deps(tmp_path, ws=ws, llm=FakeLLM(text_events("안내해 드리겠습니다.")))
    await run(ws, deps)
    assert ws.errors == 1
    assert ws.texts()[-1] == {"type": "turn.end"}


class BreakingWS(FakeWS):
    """확정 직후(첫 오디오를 받은 뒤) 프레임 수신이 예외로 끊긴다."""

    async def __anext__(self):
        if any(isinstance(m, bytes) for _, m in self.sent):
            raise RuntimeError("socket broke")
        return await super().__anext__()


async def test_frame_loop_error_resets_turn_state(tmp_path):
    ws = BreakingWS(start_msg(), utterance(50))
    deps, _ = make_deps(tmp_path, ws=ws, llm=FakeLLM("hang"))
    s = Session(ws, deps)
    try:
        await s.run()
    except RuntimeError:
        pass
    assert s.responding is False
    assert s._precall is None and s._replay is None and s._barged is False
    assert s._response_task is None or s._response_task.done()
