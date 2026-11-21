"""AG-T10a 오프라인 LLM 평가기(eval/llm_eval.py): fake LLM으로 함수 일치·응답 기록·집계·재시도·심사 검증.

실제 프롬프트(agent.prompts.build_system)와 실제 Responder 툴 루프를 돌리고, 툴은 mock_api의
load_shop().lookup을 직접 부른다(정규화 훅 agent.normalize.normalize_args 적용).
"""
import asyncio
import json

import pytest

from agent.core.types import LLMEvent, ToolCall
from eval import llm_eval
from mock_api.server import load_shop


# ── fakes ────────────────────────────────────────────
class ScriptLLM:
    """고객 발화(첫 user 메시지) → 회차별 이벤트 목록. 시스템 프롬프트·도구를 기록한다."""

    def __init__(self, scripts: dict, errors: list | None = None):
        self.scripts = scripts
        self.errors = list(errors or [])   # 앞에서부터 한 번씩 raise
        self.calls: list[dict] = []

    async def stream(self, system, messages, tools):
        utt = next(m.content for m in messages if m.role == "user")
        rnd = sum(1 for m in messages if m.role == "assistant")
        self.calls.append({"system": system, "utt": utt, "round": rnd, "tools": tools,
                           "messages": list(messages)})
        if self.errors:
            raise self.errors.pop(0)
        script = self.scripts[utt]
        evs = script[min(rnd, len(script) - 1)]
        for ev in evs:
            await asyncio.sleep(0)
            yield ev


def text(*chunks):
    return [LLMEvent("text", text=c) for c in chunks] + [LLMEvent("done")]


def call(name, args, cid="c1"):
    return [LLMEvent("tool_call", tool_call=ToolCall(cid, name, args)), LLMEvent("done")]


def case(cid, utt, expected, category="tool_lookup", caller=None, history=None, points=None, must_not=None):
    return {"case_id": cid, "now": "2026-11-20 14:00", "history": history or [],
            "customer_profile": caller, "customer_utterance": utt, "category": category,
            "tags": [], "expected_tools": expected, "expected_points": points or ["p1", "p2"],
            "must_not": must_not or []}


@pytest.fixture
def shop():
    return load_shop()


# ── 단일 문항 ─────────────────────────────────────────
async def test_exact_match_with_normalized_args_and_tool_result_fed_back(shop):
    # LLM이 하이픈 섞인 번호를 줘도 normalize_args가 고쳐서 Mock이 matched
    c = case("T1", "주문 이육일일 일오공 이삼일 배송 언제 와요?",
             [{"name": "track_delivery", "arguments": {"order_id": "2611150231"}}])
    llm = ScriptLLM({c["customer_utterance"]: [call("track_delivery", {"order_id": "2611-150-231"}),
                                               text("내일 도착 예정이에요. ", "더 궁금하신 점 있으세요?")]})
    rows = await llm_eval.evaluate([c], lambda: llm, shop=shop, concurrency=1)
    r = rows[0]
    assert r["tool_calls"] == [{"name": "track_delivery", "arguments": {"order_id": "2611150231"},
                                "status": "ok", "matched": True}]
    assert r["scores"]["tool"] == 100.0 and r["scores"]["tool_exact"] is True
    assert r["response"] == "내일 도착 예정이에요. 더 궁금하신 점 있으세요?"
    assert r["agent_text"].startswith("네, 고객님. ")          # 첫 마디 포함(측정기 transcript와 같은 모양)
    assert r["rounds"] == 2
    # 2회차 LLM 입력에 툴 결과가 들어갔다
    tool_msgs = [m for m in llm.calls[1]["messages"] if m.role == "tool"]
    assert tool_msgs and json.loads(tool_msgs[0].content)["status"] == "ok"
    assert "error" not in r


async def test_unexpected_call_when_expected_empty_scores_zero(shop):
    c = case("T2", "영업시간 알려 주세요.", [], category="faq")
    llm = ScriptLLM({c["customer_utterance"]: [call("get_order_list", {"phone_number": "01023456781"}),
                                               text("평일 아홉 시부터 여섯 시까지예요.")]})
    r = (await llm_eval.evaluate([c], lambda: llm, shop=shop))[0]
    assert r["scores"]["tool"] == 0.0 and r["scores"]["tool_exact"] is False
    assert r["scores"]["tool_extra"][0]["name"] == "get_order_list"


async def test_no_call_when_expected_empty_scores_full(shop):
    c = case("T3", "영업시간 알려 주세요.", [], category="faq")
    llm = ScriptLLM({c["customer_utterance"]: [text("평일 아홉 시부터 여섯 시까지예요.")]})
    r = (await llm_eval.evaluate([c], lambda: llm, shop=shop))[0]
    assert r["scores"]["tool"] == 100.0 and r["scores"]["tool_exact"] is True
    assert r["tool_calls"] == []


async def test_missing_and_partial_two_step_match(shop):
    exp = [{"name": "get_order_list", "arguments": {"phone_number": "01087654321"}},
           {"name": "cancel_order", "arguments": {"order_id": "2611190412"}}]
    c = case("T4", "요가 매트 취소해 주세요.", exp, category="tool_action",
             caller={"name": "임도현", "phone": "01087654321", "email": None})
    llm = ScriptLLM({c["customer_utterance"]: [call("get_order_list", {"phone_number": "01087654321"}),
                                               text("요가 매트 주문 취소해 드릴까요?")]})
    r = (await llm_eval.evaluate([c], lambda: llm, shop=shop))[0]
    assert r["scores"]["tool"] == 50.0 and r["scores"]["tool_exact"] is False
    assert r["scores"]["tool_missing"][0]["name"] == "cancel_order"


async def test_free_text_args_ignored_for_match(shop):
    exp = [{"name": "transfer_to_agent", "arguments": {"reason": "직접 요청"}}]
    c = case("T5", "상담원 연결해 주세요.", exp, category="complaint_handoff")
    llm = ScriptLLM({c["customer_utterance"]: [call("transfer_to_agent", {"reason": "고객이 상담원 요청"}),
                                               text("담당 상담원에게 연결해 드릴게요.")]})
    r = (await llm_eval.evaluate([c], lambda: llm, shop=shop))[0]
    assert r["scores"]["tool_exact"] is True


async def test_uses_real_prompt_with_caller_history_and_opener(shop):
    hist = [{"role": "customer", "text": "주문 취소하려고요."},
            {"role": "agent", "text": "주문번호를 말씀해 주시겠어요?"}]
    c = case("T6", "이육일일 이공공 삼사이요.", [{"name": "cancel_order", "arguments": {"order_id": "2611200342"}}],
             category="followup", caller={"name": "정우진", "phone": "01067812345", "email": None},
             history=hist)
    llm = ScriptLLM({c["customer_utterance"]: [call("cancel_order", {"order_id": "2611200342"}),
                                               text("취소됐어요.")]})
    await llm_eval.evaluate([c], lambda: llm, shop=shop, opener="네, 고객님.")
    sys_ = llm.calls[0]["system"]
    assert "01067812345" in sys_ and "정우진" in sys_
    assert "주문번호를 말씀해 주시겠어요?" in sys_
    assert "네, 고객님." in sys_
    assert "2026-11-20 14:00" in sys_
    # 도구는 Mock 정의(OpenAI 형식)
    names = {t["function"]["name"] for t in llm.calls[0]["tools"]}
    assert {"track_delivery", "transfer_to_agent", "request_callback"} <= names


async def test_unknown_tool_name_recorded_not_crash(shop):
    c = case("T7", "뭐 좀 해 주세요.", [])
    llm = ScriptLLM({c["customer_utterance"]: [call("no_such_tool", {"x": "1"}), text("확인이 어려워요.")]})
    r = (await llm_eval.evaluate([c], lambda: llm, shop=shop))[0]
    assert r["tool_calls"][0]["name"] == "no_such_tool"
    assert r["tool_calls"][0]["status"] == "unknown_tool"
    assert r["scores"]["tool"] == 0.0


async def test_latency_and_style_metrics(shop):
    c = case("T8", "쿠폰 있어요?", [], category="faq")
    llm = ScriptLLM({c["customer_utterance"]: [text("고객님, 확인해 보니 쿠폰이 없어요. ", "더 궁금하신 점 있으세요?")]})
    r = (await llm_eval.evaluate([c], lambda: llm, shop=shop))[0]
    assert r["llm_first_event_ms"] is not None and r["llm_first_event_ms"] >= 0
    assert r["first_sentence_ms"] is not None and r["first_sentence_ms"] >= r["llm_first_event_ms"]
    assert r["total_ms"] >= r["first_sentence_ms"]
    assert r["sentences"] == 2
    assert r["response_chars"] == len("고객님,확인해보니쿠폰이없어요.더궁금하신점있으세요?")
    assert r["honorific_repeat"] is True        # 첫 마디 "네, 고객님." 뒤 본 응답이 또 "고객님"으로 시작


def test_style_flags():
    assert llm_eval.honorific_repeat("고객님, 확인해 볼게요.") is True
    assert llm_eval.honorific_repeat("김민준 고객님, 주문은 배송 중이에요.") is True
    assert llm_eval.honorific_repeat("안녕하세요, 다이렉트몰 고객센터입니다.") is True   # 인사 반복도 같은 묶음
    assert llm_eval.honorific_repeat("주문하신 이어폰은 내일 도착해요.") is False
    assert llm_eval.honorific_repeat("") is False


# ── 재시도·오류 ───────────────────────────────────────
async def test_retry_on_503_then_success(shop):
    c = case("T9", "영업시간이요?", [], category="faq")
    llm = ScriptLLM({c["customer_utterance"]: [text("평일 아홉 시부터예요.")]},
                    errors=[RuntimeError("503 UNAVAILABLE model overloaded")])
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    r = (await llm_eval.evaluate([c], lambda: llm, shop=shop, retries=2, sleep=fake_sleep))[0]
    assert "error" not in r and r["attempts"] == 2
    assert sleeps and sleeps[0] > 0


async def test_non_retryable_error_recorded(shop):
    c = case("T10", "영업시간이요?", [], category="faq")
    llm = ScriptLLM({}, errors=[RuntimeError("400 INVALID_ARGUMENT bad"), RuntimeError("x")])

    async def fake_sleep(s):
        pass

    r = (await llm_eval.evaluate([c], lambda: llm, shop=shop, retries=3, sleep=fake_sleep))[0]
    assert "400" in r["error"] and r["attempts"] == 1
    assert r["scores"]["tool"] is None          # 오류 문항은 함수 점수 집계에서 뺀다


async def test_concurrency_limit(shop):
    active, peak = 0, 0

    class SlowLLM:
        async def stream(self, system, messages, tools):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            yield LLMEvent("text", text="네, 안내해 드릴게요.")
            yield LLMEvent("done")

    cases = [case(f"C{i}", f"질문 {i}", [], category="faq") for i in range(6)]
    rows = await llm_eval.evaluate(cases, SlowLLM, shop=shop, concurrency=2)
    assert peak <= 2 and len(rows) == 6
    assert [r["case_id"] for r in rows] == [c["case_id"] for c in cases]   # 입력 순서 유지


# ── 심사 ─────────────────────────────────────────────
async def test_judge_scores_points_and_must_not(shop):
    c1 = case("J1", "질문1", [], category="faq", points=["a", "b"])
    c2 = case("J2", "질문2", [], category="faq", points=["a"], must_not=["반말"])
    llm = ScriptLLM({"질문1": [text("답 하나예요.")], "질문2": [text("답 둘이에요.")]})
    seen = []

    def judge(row):
        seen.append(row)
        if row["case_id"] == "J1":
            return {"points": [True, False], "violations": [], "comment": "반만"}
        return {"points": [True], "violations": [True]}

    rows = await llm_eval.evaluate([c1, c2], lambda: llm, shop=shop, judge_fn=judge)
    assert rows[0]["scores"]["judge"] == 50.0 and rows[0]["scores"]["judge_comment"] == "반만"
    assert rows[1]["scores"]["judge"] == 0.0
    # 심사 입력은 eval.score 심사 함수가 읽는 필드
    assert {"customer_utterance", "agent_text", "tool_calls", "expected_points", "must_not", "history"} <= set(seen[0])


async def test_judge_error_does_not_break_row(shop):
    c = case("J3", "질문3", [], category="faq")
    llm = ScriptLLM({"질문3": [text("답이에요.")]})

    def judge(row):
        raise RuntimeError("429 quota")

    r = (await llm_eval.evaluate([c], lambda: llm, shop=shop, judge_fn=judge))[0]
    assert r["scores"]["judge"] is None and "judge_error" in r["scores"]


# ── 집계·저장 ─────────────────────────────────────────
def _row(cat, tool, exact, judge=None, chars=10, fe=500.0, fs=700.0, hon=False, err=False):
    r = {"case_id": "x", "category": cat, "response_chars": chars, "llm_first_event_ms": fe,
         "first_sentence_ms": fs, "honorific_repeat": hon, "sentences": 2, "rounds": 1,
         "scores": {"tool": tool, "tool_exact": exact, "judge": judge}}
    if err:
        r["error"] = "boom"
        r["scores"] = {"tool": None, "tool_exact": None, "judge": None}
    return r


def test_summarize_by_category():
    rows = [_row("faq", 100.0, True, judge=80.0), _row("faq", 0.0, False, judge=40.0, hon=True),
            _row("tool_lookup", 50.0, False, chars=30, fe=900.0), _row("tool_lookup", 0, None, err=True)]
    s = llm_eval.summarize(rows)
    assert s["overall"]["n"] == 4 and s["overall"]["errors"] == 1
    assert s["by_category"]["faq"]["tool_exact"] == 0.5
    assert s["by_category"]["faq"]["tool"] == 50.0
    assert s["by_category"]["faq"]["judge"] == 60.0
    assert s["by_category"]["faq"]["honorific_repeat"] == 1
    assert s["by_category"]["tool_lookup"]["tool_exact"] == 0.0     # 오류 문항 제외
    assert s["by_category"]["tool_lookup"]["judge"] is None
    assert s["overall"]["llm_first_event_p50"] is not None
    text_ = llm_eval.render_summary(s)
    assert "faq" in text_ and "tool_lookup" in text_ and "함수" in text_


async def test_write_run_outputs(tmp_path, shop):
    c = case("W1", "영업시간이요?", [], category="faq")
    llm = ScriptLLM({c["customer_utterance"]: [text("평일 아홉 시부터예요.")]})
    rows = await llm_eval.evaluate([c], lambda: llm, shop=shop)
    out = llm_eval.write_run(tmp_path / "run", rows, {"model": "fake"})
    lines = (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["case_id"] == "W1"
    summ = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summ["overall"]["n"] == 1 and summ["meta"]["model"] == "fake"
    assert "faq" in (out / "report.md").read_text(encoding="utf-8")


def test_load_cases_filters(tmp_path):
    p = tmp_path / "cases.jsonl"
    rows = [case("A1", "u1", [], category="faq"), case("A2", "u2", [], category="followup"),
            case("A3", "u3", [], category="faq")]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    assert [c["case_id"] for c in llm_eval.load_cases(p)] == ["A1", "A2", "A3"]
    assert [c["case_id"] for c in llm_eval.load_cases(p, categories=["faq"])] == ["A1", "A3"]
    assert [c["case_id"] for c in llm_eval.load_cases(p, ids=["A2"])] == ["A2"]
    assert [c["case_id"] for c in llm_eval.load_cases(p, limit=2)] == ["A1", "A2"]


def test_real_testset_loads():
    cases = llm_eval.load_cases("legacy/testset/cases.jsonl")
    assert len(cases) >= 172
    assert all("customer_utterance" in c and "expected_tools" in c for c in cases)


async def test_filler_recorded_in_tool_turn(shop):
    # 실제 서버처럼 필러 캐시가 있다: 툴 회차 뒤 LLM이 느리면 필러가 transcript에 들어간다
    c = case("F1", "주문 2611150231 어디쯤이에요?", [{"name": "track_delivery", "arguments": {"order_id": "2611150231"}}])

    class SlowSecond:
        def __init__(self):
            self.n = 0

        async def stream(self, system, messages, tools):
            self.n += 1
            if self.n == 1:
                yield LLMEvent("tool_call", tool_call=ToolCall("c1", "track_delivery", {"order_id": "2611150231"}))
            else:
                await asyncio.sleep(0.3)
                yield LLMEvent("text", text="내일 도착해요. ")
            yield LLMEvent("done")

    from agent.config import Settings
    r = (await llm_eval.evaluate([c], SlowSecond, shop=shop,
                                 settings=Settings(filler_delay_ms=50)))[0]
    assert r["filler"] is True
    assert r["response"].startswith(Settings().fillers[0])
    assert r["first_audio_ms"] < r["first_sentence_ms"]
