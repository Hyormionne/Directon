"""AG-T07 responder: LLM 툴 루프, 문장 분할, TTS 파이프라인, 필러, 정규화 훅
+ tts_cache / prompts / turn_log 단위 테스트."""
import asyncio
import json

import httpx
import pytest

from agent.config import Settings
from agent.core.types import LLMEvent, Message, SessionInfo, ToolCall
from agent.prompts import build_system
from agent.responder import Responder, SentenceSplitter
from agent.tools_client import ToolsClient
from agent.tts_cache import TTSCache, select_opener
from agent.turn_log import TurnLog, TurnLogWriter

TOOLS = [{"type": "function", "function": {
    "name": "get_order_info", "description": "주문 조회",
    "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                   "required": ["order_id"]}}}]


def pcm_for(text: str) -> bytes:
    return b"\x10\x27" * 4 + text.encode("utf-8")


# ── fakes ────────────────────────────────────────────
class FakeLLM:
    """scripts: 호출마다 이벤트 목록(또는 예외/'hang'). 호출 기록을 남긴다."""

    def __init__(self, *scripts):
        self.scripts = list(scripts)
        self.calls: list[dict] = []

    async def stream(self, system, messages, tools):
        self.calls.append({"system": system, "messages": [Message(**vars(m)) for m in messages],
                           "tools": tools})
        script = self.scripts[min(len(self.calls) - 1, len(self.scripts) - 1)]
        if isinstance(script, Exception):
            raise script
        if script == "hang":
            await asyncio.Event().wait()
        for ev in script:
            await asyncio.sleep(0)
            yield ev


def text_events(*chunks):
    return [LLMEvent("text", text=c) for c in chunks] + [LLMEvent("done")]


def tool_events(name, args, cid="c1", pre_text=""):
    evs = [LLMEvent("text", text=pre_text)] if pre_text else []
    return evs + [LLMEvent("tool_call", tool_call=ToolCall(cid, name, args)), LLMEvent("done")]


class FakeTTS:
    def __init__(self, fail=False):
        self.requests: list[str] = []
        self.fail = fail

    async def synthesize(self, text):
        self.requests.append(text)
        if self.fail:
            raise RuntimeError("tts down")
        data = pcm_for(text)
        await asyncio.sleep(0)
        yield data[:4]
        await asyncio.sleep(0)
        yield data[4:]


class Sink:
    def __init__(self):
        self.chunks: list[bytes] = []

    async def __call__(self, pcm: bytes):
        self.chunks.append(pcm)

    @property
    def data(self):
        return b"".join(self.chunks)


def settings(**kw):
    base = dict(filler_delay_ms=700, llm_timeout_s=2.0, tts_timeout_s=2.0, min_sentence_chars=6)
    base.update(kw)
    return Settings(**base)


async def no_tool(name, args):  # pragma: no cover - 호출되면 실패
    raise AssertionError("tool should not be called")


# ── 문장 분할 ─────────────────────────────────────────
def test_splitter_basic_boundaries_and_merge_short():
    sp = SentenceSplitter(min_chars=6)
    out = []
    for ch in ["네. 주문하신 상품은 ", "내일 도착합니다. 다른 문의", " 있으신가요? 감사합", "니다!"]:
        out += sp.feed(ch)
    out += sp.flush()
    assert out == ["네. 주문하신 상품은 내일 도착합니다.", "다른 문의 있으신가요?", "감사합니다!"]


def test_splitter_does_not_split_decimal_and_flushes_tail():
    sp = SentenceSplitter(min_chars=6)
    out = sp.feed("금액은 1.5만 원이에요. 배송비는 무료")
    assert out == ["금액은 1.5만 원이에요."]
    assert sp.flush() == ["배송비는 무료"]


def test_splitter_strips_markdown():
    sp = SentenceSplitter(min_chars=1)
    out = sp.feed("**배송** 중입니다. ") + sp.flush()
    assert out == ["배송 중입니다."]


# ── 응답기 ───────────────────────────────────────────
async def test_respond_text_sentences_go_to_tts_in_order():
    llm = FakeLLM(text_events("주문하신 상품은 내일 ", "도착합니다. 다른 문의 있으신가요?"))
    tts = FakeTTS()
    r = Responder(llm, tts, no_tool, tools=TOOLS, settings=settings())
    sink = Sink()
    spoken = await r.respond("SYS", [Message("user", "언제 와요?")], sink)
    assert spoken == ["주문하신 상품은 내일 도착합니다.", "다른 문의 있으신가요?"]
    assert tts.requests == spoken
    assert sink.data == pcm_for(spoken[0]) + pcm_for(spoken[1])
    assert [t["function"]["name"] for t in llm.calls[0]["tools"][:len(TOOLS)]] == [t["function"]["name"] for t in TOOLS] and llm.calls[0]["system"] == "SYS"


async def test_tool_loop_calls_mock_with_session_header_and_feeds_result_back():
    seen = []

    def handler(req: httpx.Request):
        seen.append((req.url.path, req.headers.get("X-Session-Id"), json.loads(req.content)))
        return httpx.Response(200, json={"status": "ok", "delivery": "배송중"})

    client = ToolsClient(http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    async def call_tool(name, args):
        return await client.call("http://mock:8100", "S-42", name, args)

    llm = FakeLLM(tool_events("get_order_info", {"order_id": "2611150231"}),
                  text_events("현재 배송 중입니다. 내일 도착 예정이에요."))
    tts = FakeTTS()
    r = Responder(llm, tts, call_tool, tools=TOOLS, settings=settings())
    sink = Sink()
    spoken = await r.respond("SYS", [Message("user", "배송 어디쯤이에요?")], sink)

    assert seen == [("/tools/get_order_info", "S-42", {"order_id": "2611150231"})]
    assert len(llm.calls) == 2
    msgs = llm.calls[1]["messages"]
    assert msgs[0].role == "user"
    assert msgs[1].role == "assistant" and msgs[1].tool_calls[0].name == "get_order_info"
    assert msgs[2].role == "tool" and msgs[2].tool_call_id == "c1" and msgs[2].name == "get_order_info"
    assert json.loads(msgs[2].content) == {"status": "ok", "delivery": "배송중"}
    assert spoken == ["현재 배송 중입니다.", "내일 도착 예정이에요."]


async def test_normalize_hook_applied_before_call():
    got = {}

    def normalize(name, args, schema, now):
        got.update(name=name, schema=schema, now=now)
        return {"order_id": "2611150231"}

    calls = []

    async def call_tool(name, args):
        calls.append((name, args))
        return {"status": "ok"}

    llm = FakeLLM(tool_events("get_order_info", {"order_id": "이육일일 일오공 이삼일"}),
                  text_events("확인했습니다. 곧 도착해요."))
    r = Responder(llm, FakeTTS(), call_tool, tools=TOOLS, settings=settings(),
                  normalize_args=normalize, now="2026-11-21 11:00")
    await r.respond("SYS", [Message("user", "x")], Sink())
    assert calls == [("get_order_info", {"order_id": "2611150231"})]
    assert got["name"] == "get_order_info" and got["now"] == "2026-11-21 11:00"
    assert got["schema"] == TOOLS[0]["function"]["parameters"]
    # LLM에 넘긴 assistant tool_call 인자도 정규화된 값
    assert llm.calls[1]["messages"][1].tool_calls[0].arguments == {"order_id": "2611150231"}


async def test_default_normalize_is_identity():
    calls = []

    async def call_tool(name, args):
        calls.append(args)
        return {"status": "ok"}

    llm = FakeLLM(tool_events("get_order_info", {"order_id": "공일"}), text_events("네 확인했습니다."))
    await Responder(llm, FakeTTS(), call_tool, tools=TOOLS, settings=settings()).respond(
        "S", [Message("user", "x")], Sink())
    assert calls == [{"order_id": "공일"}]


async def test_max_three_rounds_last_round_without_tools():
    async def call_tool(name, args):
        return {"status": "ok"}

    loop = tool_events("get_order_info", {"order_id": "1"})
    llm = FakeLLM(loop, loop, text_events("확인 결과를 안내드릴게요."))
    r = Responder(llm, FakeTTS(), call_tool, tools=TOOLS, settings=settings(max_tool_rounds=3))
    spoken = await r.respond("S", [Message("user", "x")], Sink())
    assert len(llm.calls) == 3
    assert llm.calls[2]["tools"] == []
    assert spoken == ["확인 결과를 안내드릴게요."]


async def test_filler_played_when_tool_slow(tmp_path):
    cache = TTSCache(tmp_path)
    filler = "확인해 드릴게요, 잠시만요."
    cache.put(filler, b"\xaa\x55" * 8)

    async def slow_tool(name, args):
        await asyncio.sleep(0.3)
        return {"status": "ok"}

    llm = FakeLLM(tool_events("get_order_info", {"order_id": "1"}), text_events("배송 중입니다. 감사합니다."))
    r = Responder(llm, FakeTTS(), slow_tool, tools=TOOLS, cache=cache,
                  settings=settings(filler_delay_ms=50, fillers=(filler,)))
    sink = Sink()
    spoken = await r.respond("S", [Message("user", "x")], sink)
    assert spoken[0] == filler
    assert sink.chunks[0] == b"\xaa\x55" * 8


async def test_no_filler_when_tool_fast(tmp_path):
    cache = TTSCache(tmp_path)
    filler = "확인해 드릴게요, 잠시만요."
    cache.put(filler, b"\xaa\x55" * 8)

    async def fast_tool(name, args):
        return {"status": "ok"}

    llm = FakeLLM(tool_events("get_order_info", {"order_id": "1"}), text_events("배송 중입니다."))
    r = Responder(llm, FakeTTS(), fast_tool, tools=TOOLS, cache=cache,
                  settings=settings(filler_delay_ms=500, fillers=(filler,)))
    spoken = await r.respond("S", [Message("user", "x")], Sink())
    assert filler not in spoken


async def test_tts_pipeline_starts_next_sentence_before_previous_finishes():
    started = {}

    class GatedTTS:
        async def synthesize(self, text):
            started[text] = asyncio.get_running_loop().time()
            if text.startswith("첫"):
                # 다음 문장 합성이 미리 시작되지 않으면 여기서 타임아웃
                for _ in range(200):
                    if any(t.startswith("둘") for t in started):
                        break
                    await asyncio.sleep(0.005)
                else:
                    raise AssertionError("next sentence synthesis not started early")
            yield pcm_for(text)

    llm = FakeLLM(text_events("첫 번째 문장입니다. ", "둘째 문장도 있어요."))
    r = Responder(llm, GatedTTS(), no_tool, tools=TOOLS, settings=settings())
    sink = Sink()
    spoken = await r.respond("S", [Message("user", "x")], sink)
    assert spoken == ["첫 번째 문장입니다.", "둘째 문장도 있어요."]
    assert sink.data == pcm_for(spoken[0]) + pcm_for(spoken[1])


async def test_llm_error_propagates():
    r = Responder(FakeLLM(RuntimeError("llm down")), FakeTTS(), no_tool, tools=TOOLS, settings=settings())
    with pytest.raises(Exception):
        await r.respond("S", [Message("user", "x")], Sink())


async def test_llm_hang_times_out():
    r = Responder(FakeLLM("hang"), FakeTTS(), no_tool, tools=TOOLS, settings=settings(llm_timeout_s=0.05))
    with pytest.raises((asyncio.TimeoutError, TimeoutError)):
        await asyncio.wait_for(r.respond("S", [Message("user", "x")], Sink()), 2.0)


async def test_tts_error_propagates_and_spoken_keeps_partial():
    r = Responder(FakeLLM(text_events("안내 드리겠습니다.")), FakeTTS(fail=True), no_tool,
                  tools=TOOLS, settings=settings())
    spoken: list[str] = []
    with pytest.raises(Exception):
        await r.respond("S", [Message("user", "x")], Sink(), spoken=spoken)
    assert spoken == []


async def test_prefetch_is_reused_without_second_llm_call():
    llm = FakeLLM(text_events("네 바로 확인해 드리겠습니다."))
    r = Responder(llm, FakeTTS(), no_tool, tools=TOOLS, settings=settings())
    msgs = [Message("user", "배송 언제 와요?")]
    pf = r.prefetch("S", msgs)
    await asyncio.sleep(0.01)          # 선호출이 버퍼에 쌓인다
    spoken = await r.respond("S", msgs, Sink(), prefetch=pf)
    assert len(llm.calls) == 1
    assert spoken == ["네 바로 확인해 드리겠습니다."]


async def test_prefetch_cancel_stops_stream():
    cancelled = asyncio.Event()

    class HangLLM:
        async def stream(self, system, messages, tools):
            try:
                await asyncio.Event().wait()
                yield LLMEvent("done")
            finally:
                cancelled.set()

    r = Responder(HangLLM(), FakeTTS(), no_tool, tools=TOOLS, settings=settings())
    pf = r.prefetch("S", [Message("user", "x")])
    await asyncio.sleep(0.01)
    pf.cancel()
    await asyncio.wait_for(cancelled.wait(), 1.0)


async def test_turn_log_records_llm_and_tts_events():
    async def call_tool(name, args):
        return {"status": "ok"}

    t = iter(range(1000))
    log = TurnLog("S", clock=lambda: next(t) / 1000)
    llm = FakeLLM(tool_events("get_order_info", {"order_id": "1"}), text_events("배송 중입니다."))
    await Responder(llm, FakeTTS(), call_tool, tools=TOOLS, settings=settings()).respond(
        "S", [Message("user", "x")], Sink(), log=log)
    d = log.to_dict()
    names = [e["name"] for e in d["events"]]
    assert "llm_first_token" in d["marks"]
    assert "tool_call" in names and "tool_result" in names
    assert "tts_request" in names and "tts_first_byte" in names


# ── TTS 캐시 ──────────────────────────────────────────
async def test_tts_cache_warm_synthesizes_only_missing_and_persists(tmp_path):
    c = TTSCache(tmp_path)
    c.put("네, 고객님.", b"\x01\x02" * 10)
    tts = FakeTTS()
    n = await c.warm(["네, 고객님.", "잠시만요."], tts)
    assert n == 1 and tts.requests == ["잠시만요."]
    assert len(list(tmp_path.glob("*.pcm"))) == 2
    c2 = TTSCache(tmp_path)                       # 새 인스턴스가 디스크에서 읽는다
    assert c2.get("네, 고객님.") == b"\x01\x02" * 10
    assert c2.get("잠시만요.") is not None
    assert c2.get("없는 문구") is None


async def test_tts_cache_trims_leading_silence(tmp_path):
    c = TTSCache(tmp_path)
    loud = (b"\x00\x10" * 320)          # 4096 진폭
    c.put("x", b"\x00\x00" * 320 * 5 + loud)
    assert c.get("x") == loud


def test_select_opener_default_without_name():
    s = Settings()
    info = SessionInfo("S", caller={"name": "김민수", "phone": "01012345678"})
    assert select_opener(info, s) == s.openers[0]
    assert "김민수" not in select_opener(info, s)
    s2 = Settings(opener_use_name=True)
    assert select_opener(info, s2) == "네, 김민수 고객님."


# ── 프롬프트 ─────────────────────────────────────────
def test_build_system_contains_rules_session_and_opener(tmp_path):
    manual = tmp_path / "manual.md"
    manual.write_text("환불은 7일 이내 가능", encoding="utf-8")
    info = SessionInfo("S", now="2026-11-21 11:00",
                       caller={"name": "김민수", "phone": "01012345678", "email": "a@b.c"},
                       history=[{"role": "agent", "text": "주문번호를 알려 주시겠어요?"}])
    sys_ = build_system(info, "네, 고객님.", manual_path=manual)
    assert "환불은 7일 이내 가능" in sys_
    assert "2026-11-21 11:00" in sys_
    assert "01012345678" in sys_ and "김민수" in sys_
    assert "주문번호를 알려 주시겠어요?" in sys_
    assert "이미 '네, 고객님.'라고 말했다" in sys_
    assert "마크다운" in sys_
    # 고정부가 앞, 가변부가 뒤(프롬프트 캐시)
    assert sys_.index("마크다운") < sys_.index("2026-11-21 11:00")


def test_build_system_without_manual(tmp_path):
    info = SessionInfo("S", now="2026-11-21 11:00")
    sys_ = build_system(info, "네, 고객님.", manual_path=tmp_path / "missing.md")
    assert "응대 매뉴얼" not in sys_      # 매뉴얼 절 자체를 생략
    assert "2026-11-21 11:00" in sys_


def test_turn_log_writer_appends_jsonl(tmp_path):
    w = TurnLogWriter(tmp_path, date_fn=lambda: "20261121")
    w.write({"session_id": "S", "a": 1})
    w.write({"session_id": "S", "a": 2})
    lines = (tmp_path / "turns-20261121.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(l)["a"] for l in lines] == [1, 2]


def test_unkept_promise_detects_accept_and_lookup_but_not_conditional():
    from agent.responder import unkept_promise
    assert unkept_promise("성함과 연락처로 전화 요청 접수됐어요.", []) == "accept"
    assert unkept_promise("전화번호로 바로 확인해 볼게요.", []) == "lookup"
    assert unkept_promise("번호를 말씀해 주시면 확인해 드릴게요.", []) is None
    assert unkept_promise("전체 반품 접수 도와드리겠습니다.", ["get_order_info"]) is None    # 제안은 약속으로 안 봄
    assert unkept_promise("전체 반품으로 접수됐습니다.", ["get_order_info"]) == "accept"
    assert unkept_promise("반품 접수됐습니다.", ["return_order"]) is None


def test_ground_order_value_uses_seen_numbers():
    from agent.responder import ground_order_value
    seen = ["20261001-0060343"]
    assert ground_order_value("9900659938305410", seen) == "20261001-0060343"   # 후보가 하나뿐
    assert ground_order_value("202610010060343", ["20261001-0060343", "20261002-0011111"]) == "20261001-0060343"
    assert ground_order_value("111", ["20261001-0060343", "20261002-0011111"]) is None
