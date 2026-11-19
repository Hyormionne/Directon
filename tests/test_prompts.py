"""AG-T10a 프롬프트 규칙 단언: 함수 사용 규칙, 인계·콜백, 호칭·인사 반복 금지, caller 재질문 금지,
history 요약, 숫자 되읽기 형식, 영업시간 판단. (기존 tests/test_responder.py 프롬프트 단언은 그대로 유지)"""
from agent.core.types import SessionInfo
from agent.normalize import digits_for_speech
from agent.prompts import FIXED, build_system, business_hours_note, history_summary

CALLER = {"name": "김민준", "phone": "01023456781", "email": None}


def sys_for(**kw):
    info = SessionInfo("S", now=kw.pop("now", "2026-11-20 14:00"), caller=kw.pop("caller", None),
                       history=kw.pop("history", []))
    return build_system(info, kw.pop("opener", "네, 고객님."), manual="매뉴얼 본문")


# ── 고정부 규칙 ───────────────────────────────────────
def test_handoff_rule_requires_transfer_call():
    assert "transfer_to_agent" in FIXED
    assert "말로만" in FIXED                      # 말로만 안내하지 말 것
    assert "반드시" in FIXED
    assert "request_callback" in FIXED            # 영업시간 외 인계는 콜백


def test_no_honorific_or_greeting_repeat_rule():
    assert "호칭" in FIXED and "인사" in FIXED
    assert "반복하지 않는다" in FIXED
    assert "고객님" in FIXED


def test_caller_no_reask_rule_in_fixed_and_session():
    assert "다시 묻지 않는다" in FIXED
    s = sys_for(caller=CALLER)
    var = s[s.index("## 이번 통화"):]
    assert "01023456781" in var and "김민준" in var
    assert "다시 묻지" in var
    assert "get_order_list" in var                # 그 번호로 바로 조회


def test_no_caller_execution_request_asks_only():
    s = sys_for()
    var = s[s.index("## 이번 통화"):]
    assert "없음" in var
    assert "되묻" in FIXED                        # 번호 없이 실행 요청이면 되묻기만


def test_history_reuse_rule():
    assert "이전 대화" in FIXED and "다시 조회하지 않는다" in FIXED


def test_digit_readback_format_uses_digits_for_speech():
    assert digits_for_speech("2611150231", "order") in FIXED
    assert digits_for_speech("01012345678", "phone") in FIXED


def test_status_phrases_and_brevity():
    for st in ("not_found", "not_allowed", "no_refund"):
        assert st in FIXED
    assert "2~3문장" in FIXED
    assert "첫 문장은 짧게" in FIXED


def test_faq_without_tools_rule():
    assert "함수 없이" in FIXED


def test_fixed_part_has_no_test_utterances():
    # 과적합 방지: 테스트셋 문장이 프롬프트에 들어가지 않는다
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    with open(root / "data/testset/cases.jsonl", encoding="utf-8") as f:
        utts = [json.loads(l)["customer_utterance"] for l in f if l.strip()]
    assert not [u for u in utts if len(u) >= 8 and u in FIXED]


# ── 영업시간 ─────────────────────────────────────────
def test_business_hours_open_weekday():
    note = business_hours_note("2026-11-20 14:00")       # 금요일 오후
    assert "연결 가능" in note and "transfer_to_agent" in note


def test_business_hours_closed_evening_weekend_lunch():
    for now in ("2026-11-20 19:40", "2026-11-21 10:30", "2026-11-20 12:20", "2026-11-20 08:30"):
        note = business_hours_note(now)
        assert "연결 시간이 아니" in note and "request_callback" in note, now


def test_business_hours_unknown_now():
    assert business_hours_note("") == ""
    assert business_hours_note("이상한 값") == ""


def test_business_hours_in_session_part():
    s = sys_for(now="2026-11-20 19:40", caller=CALLER)
    var = s[s.index("## 이번 통화"):]
    assert "연결 시간이 아니" in var


# ── history 요약 ─────────────────────────────────────
HIST = [{"role": "customer", "text": "주문 2611150231 배송 언제 와요?"},
        {"role": "agent", "text": "내일 도착 예정이에요. 더 궁금하신 점 있으세요?"},
        {"role": "customer", "text": "취소할까 고민 중이에요."},
        {"role": "agent", "text": "배송 중이라 취소는 어렵고 받으신 뒤 반품하실 수 있어요. 반품 접수해 드릴까요?"}]


def test_history_summary_contents():
    lines = history_summary(HIST)
    joined = "\n".join(lines)
    assert "2611150231" in joined                     # 이전 대화에 나온 번호
    assert "반품 접수해 드릴까요?" in joined           # 직전 상담원 질문
    assert "다시 조회" in joined


def test_history_summary_empty():
    assert history_summary([]) == []


def test_history_summary_in_system_with_full_history():
    s = sys_for(history=HIST)
    var = s[s.index("## 이번 통화"):]
    assert "이전 대화:" in var
    assert "취소할까 고민 중이에요." in var            # 원문 그대로
    assert "직전 상담원 질문" in var
    assert "2611150231" in var


def test_opener_continuation_rule():
    s = sys_for(opener="네, 고객님.")
    assert "이미 '네, 고객님.'라고 말했다" in s
    assert "이어" in s


def test_fixed_before_session_part():
    s = sys_for(caller=CALLER)
    assert s.index("transfer_to_agent") < s.index("## 이번 통화")


# ── 응답기: 툴 회차 필러(AG-T10a, 첫 마디 뒤 공백 줄이기) ─────────────
import asyncio  # noqa: E402

from agent.config import Settings  # noqa: E402
from agent.core.types import LLMEvent, Message, ToolCall  # noqa: E402
from agent.responder import Responder  # noqa: E402

FILLER = "확인해 드릴게요, 잠시만요."
TOOLS = [{"type": "function", "function": {"name": n, "description": "", "parameters": {
    "type": "object", "properties": {"order_id": {"type": "string"}, "reason": {"type": "string"}}}}}
    for n in ("get_order_info", "transfer_to_agent")]


class _Cache:
    def get(self, text):
        return b"\x11\x11" if text == FILLER else None


class _TTS:
    async def synthesize(self, text):
        yield text.encode("utf-8")


class _LLM:
    """1회차: 툴 호출, 2회차: second_delay 뒤 텍스트."""

    def __init__(self, tool="get_order_info", second_delay=0.0):
        self.tool, self.second_delay, self.n = tool, second_delay, 0

    async def stream(self, system, messages, tools):
        self.n += 1
        if self.n == 1:
            yield LLMEvent("tool_call", tool_call=ToolCall("c1", self.tool, {"order_id": "1"}))
        else:
            await asyncio.sleep(self.second_delay)
            yield LLMEvent("text", text="배송 중이에요. ")
        yield LLMEvent("done")


async def _ok(name, args):
    return {"status": "ok"}


async def _sink(pcm):
    return None


def test_default_filler_delay_is_short():
    assert Settings().filler_delay_ms <= 300


async def test_filler_covers_slow_next_llm_round_even_when_tool_fast():
    # 툴은 빠르지만 툴 결과 뒤 LLM 회차가 느리면 필러로 빈틈을 메운다
    r = Responder(_LLM(second_delay=0.2), _TTS(), _ok, tools=TOOLS, cache=_Cache(),
                  settings=Settings(filler_delay_ms=50, fillers=(FILLER,)))
    spoken = await r.respond("S", [Message("user", "x")], _sink)
    assert spoken == [FILLER, "배송 중이에요."]


async def test_no_filler_when_next_round_text_arrives_quickly():
    r = Responder(_LLM(second_delay=0.0), _TTS(), _ok, tools=TOOLS, cache=_Cache(),
                  settings=Settings(filler_delay_ms=200, fillers=(FILLER,)))
    spoken = await r.respond("S", [Message("user", "x")], _sink)
    assert FILLER not in spoken


async def test_no_filler_for_transfer_to_agent():
    r = Responder(_LLM(tool="transfer_to_agent", second_delay=0.2), _TTS(), _ok, tools=TOOLS, cache=_Cache(),
                  settings=Settings(filler_delay_ms=20, fillers=(FILLER,)))
    spoken = await r.respond("S", [Message("user", "x")], _sink)
    assert FILLER not in spoken


# ── 응답기: 번호 인자 교정(LLM이 한글 숫자를 잘못 옮긴 경우) ─────────────
from agent.responder import utterance_numbers  # noqa: E402


def test_utterance_numbers_extracts_spoken_and_arabic_runs():
    assert utterance_numbers("이육일일 일사공 사구팔 취소해 줘.") == ["2611140498"]
    assert utterance_numbers("주문번호 2611 150 231이요") == ["2611150231"]
    # "삼사오육이요"는 '이'가 숫자 2일 수도 있어 다른 읽기도 낸다(기본 읽기가 먼저)
    assert utterance_numbers("제 번호는 공일공에 칠팔일이에 삼사오육이요.")[0] == "01078123456"
    assert "01098765432" in utterance_numbers("공일공에 구팔칠육에 오사삼이요.")
    assert utterance_numbers("이 주문 취소해 주세요.") == []                 # 짧은 수 단어는 무시
    # 정정: 뒤에 온 짧은 묶음이 앞 번호의 끝자리를 대체한다
    assert utterance_numbers("이육일일 일팔공 공사삼, 아니다, 공사이요.")[-1] == "2611180042"


class _ArgLLM:
    def __init__(self, args):
        self.args, self.n = args, 0

    async def stream(self, system, messages, tools):
        self.n += 1
        if self.n == 1:
            yield LLMEvent("tool_call", tool_call=ToolCall("c1", "get_order_info", dict(self.args)))
        else:
            yield LLMEvent("text", text="확인했어요. ")
        yield LLMEvent("done")


async def _run_with(utt, args, **kw):
    seen = []

    async def tool(name, a):
        seen.append(a)
        return {"status": "ok"}

    from agent.normalize import normalize_args
    r = Responder(_ArgLLM(args), _TTS(), tool, tools=TOOLS, settings=Settings(**kw), normalize_args=normalize_args)
    await r.respond("S", [Message("user", utt)], _sink)
    return seen[0]


async def test_order_id_grounded_to_utterance_when_llm_mangles_digits():
    got = await _run_with("이육일일 일사공 사구팔 취소해 줘.", {"order_id": "2611144948"})
    assert got == {"order_id": "2611140498"}


async def test_order_id_kept_when_matches_or_far_or_no_number():
    assert await _run_with("이육일일 일사공 사구팔이요", {"order_id": "2611140498"}) == {"order_id": "2611140498"}
    assert await _run_with("이육일일 일사공 사구팔이요", {"order_id": "2610300266"}) == {"order_id": "2610300266"}
    assert await _run_with("배송 언제 와요?", {"order_id": "2611150231"}) == {"order_id": "2611150231"}


async def test_grounding_can_be_disabled():
    got = await _run_with("이육일일 일사공 사구팔 취소해 줘.", {"order_id": "2611144948"}, ground_number_args=False)
    assert got == {"order_id": "2611144948"}


async def test_grounding_prefers_same_length_candidate():
    # "공이육이요"는 ...0026(기본)과 ...00262(다른 읽기) 둘 다 후보. LLM 값과 길이가 같은 쪽을 고른다
    got = await _run_with("이육일일 일칠공 공이육이요.", {"order_id": "2611170262"})
    assert got == {"order_id": "2611170026"}


def test_history_summary_converts_spoken_numbers():
    hist = [{"role": "customer", "text": "이육일일 일구공 사일이 주문 배송지 바꾸고 싶어요."},
            {"role": "agent", "text": "새 주소를 말씀해 주시겠어요?"}]
    joined = "\n".join(history_summary(hist))
    assert "주문번호 2611190412" in joined


# ── 응답기: 자리 수가 틀린 번호는 Mock에 보내지 않는다 ─────────────
LEN_TOOLS = [{"type": "function", "function": {"name": "get_order_info", "description": "", "parameters": {
    "type": "object", "properties": {"order_id": {"type": "string", "description": "주문번호. 숫자 10자리(예: 2611150231)."}},
    "required": ["order_id"]}}}]


async def test_wrong_length_number_not_sent_to_tool():
    seen = []

    async def tool(name, a):
        seen.append(a)
        return {"status": "ok"}

    llm = _ArgLLM({"order_id": "261115023"})
    r = Responder(llm, _TTS(), tool, tools=LEN_TOOLS, settings=Settings())
    spoken = await r.respond("S", [Message("user", "x")], _sink)
    assert seen == []                                  # 호출하지 않았다
    assert spoken == ["확인했어요."]


async def test_right_length_number_sent():
    seen = []

    async def tool(name, a):
        seen.append(a)
        return {"status": "ok"}

    r = Responder(_ArgLLM({"order_id": "2611150231"}), _TTS(), tool, tools=LEN_TOOLS, settings=Settings())
    await r.respond("S", [Message("user", "x")], _sink)
    assert seen == [{"order_id": "2611150231"}]


# ── 응답기: 상담원 연결을 말했는데 transfer_to_agent를 안 불렀으면 부른다 ─────────────
class _TalkLLM:
    def __init__(self, text):
        self.text = text

    async def stream(self, system, messages, tools):
        yield LLMEvent("text", text=self.text)
        yield LLMEvent("done")


async def _talk(text, **kw):
    seen = []

    async def tool(name, a):
        seen.append((name, a))
        return {"status": "ok"}

    r = Responder(_TalkLLM(text), _TTS(), tool, tools=TOOLS, settings=Settings(**kw))
    await r.respond("S", [Message("user", "결제가 두 번 됐어요.")], _sink)
    return seen


async def test_transfer_called_when_response_promises_connection():
    seen = await _talk("불편을 드려 죄송합니다. 담당 상담원에게 바로 연결해 드릴게요.")
    assert [n for n, _ in seen] == ["transfer_to_agent"]
    assert seen[0][1]["reason"]


async def test_no_transfer_for_offer_question_or_disabled():
    assert await _talk("담당 상담원에게 연결해 드릴까요?") == []
    assert await _talk("배송은 내일 도착해요.") == []
    assert await _talk("담당 상담원에게 바로 연결해 드릴게요.", ensure_transfer_call=False) == []


async def test_no_transfer_guard_when_callback_mentioned():
    assert await _talk("지금은 상담 시간이 아니라서 상담원 연결 대신 콜백 접수해 드릴게요.") == []


async def test_grounding_same_length_distance_three():
    got = await _run_with("이육일일, 이공공, 삼사이.", {"order_id": "2611203434"})
    assert got == {"order_id": "2611200342"}


# ── 응답기: 폭주 생성(같은 말 무한 반복) 상한 ─────────────
class _RunawayLLM:
    def __init__(self):
        self.stopped = False

    async def stream(self, system, messages, tools):
        try:
            yield LLMEvent("text", text="불편을 드려 죄송합니다. ")
            for _ in range(100000):
                await asyncio.sleep(0)
                yield LLMEvent("text", text="새 문장입니다. ")
            yield LLMEvent("done")
        finally:
            self.stopped = True


async def test_runaway_generation_is_cut_at_max_chars():
    llm = _RunawayLLM()
    r = Responder(llm, _TTS(), _ok, tools=TOOLS, settings=Settings(max_response_chars=60))
    spoken = await asyncio.wait_for(r.respond("S", [Message("user", "x")], _sink), 5)
    assert spoken[0] == "불편을 드려 죄송합니다."
    assert sum(len(s.replace(" ", "")) for s in spoken) <= 60 + 10
    assert llm.stopped


def test_default_max_response_chars_reasonable():
    assert 200 <= Settings().max_response_chars <= 600


# ── 번호 자릿수 되묻기 규칙(AG-T10b 결과 반영: STT 숫자 누락 대비) ─────────────
def test_digit_count_reask_rule():
    assert "10자리" in FIXED and "010으로 시작하는 11자리" in FIXED
    assert "한 자리씩 천천히 다시 말씀해 주시겠어요?" in FIXED
    assert "자릿수가 맞으면 되묻지 말고 바로 호출" in FIXED


async def test_length_block_message_suggests_reask():
    seen = []

    async def tool(name, a):
        seen.append(a)
        return {"status": "ok"}

    class _Cap(_ArgLLM):
        async def stream(self, system, messages, tools):
            self.msgs = list(messages)
            async for ev in super().stream(system, messages, tools):
                yield ev

    llm = _Cap({"order_id": "261115023"})
    r = Responder(llm, _TTS(), tool, tools=LEN_TOOLS, settings=Settings())
    await r.respond("S", [Message("user", "x")], _sink)
    tool_msg = [m for m in llm.msgs if m.role == "tool"][0].content
    assert seen == [] and "한 자리씩" in tool_msg
