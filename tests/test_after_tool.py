"""AG-T21 조회 뒤 답까지 시간(after_tool): 측정(runner·score), 응답기 단축(후속 hedge·결과 템플릿), 자릿수 상대화."""
import asyncio
import json
import time

import numpy as np
import pytest

from agent.config import Settings
from agent.core.types import LLMEvent, Message, ToolCall
from agent.providers.hedged_llm import HedgedLLM
from agent.responder import (Responder, match_order_id, repair_thresholds, tool_result_template)
from agent.turn_log import TurnLog
from eval import runner as R
from eval import score as S

pytestmark = pytest.mark.usefixtures("legacy_world")   # AG-T38: 옛 체계 전제(10자리 주문번호)

TOOLS = [{"type": "function", "function": {
    "name": "track_delivery", "description": "배송 조회",
    "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                   "required": ["order_id"]}}}]
DELIVERY = {"status": "ok", "data": {"order_id": "1", "status": "배송중", "current_location": "옥천HUB"},
            "message": "배송중입니다. 현재 위치는 옥천HUB이고 11월 21일 도착 예정입니다."}


@pytest.fixture(autouse=True)
def _switches_off(monkeypatch):
    """기본값과 무관하게 각 테스트가 켤 스위치만 켠다."""
    monkeypatch.setenv("DIRECTON_POST_TOOL_HEDGE", "0")
    monkeypatch.setenv("DIRECTON_TOOL_TEMPLATE", "0")


def frame(amp):
    return (np.ones(320, np.int16) * amp).tobytes()


def pcm_for(text: str) -> bytes:
    return b"\x10\x27" * 4 + text.encode("utf-8")


class FakeTTS:
    def __init__(self):
        self.requests: list[str] = []

    async def synthesize(self, text):
        self.requests.append(text)
        await asyncio.sleep(0)
        yield pcm_for(text)


class Sink:
    def __init__(self):
        self.chunks: list[bytes] = []

    async def __call__(self, pcm):
        self.chunks.append(pcm)


class ScriptLLM:
    """호출마다 스크립트(이벤트 목록). 'hang' 항목을 만나면 그 자리에서 멈춘다. ('sleep', s)는 지연."""

    def __init__(self, *scripts, name="m"):
        self.scripts = list(scripts)
        self.calls: list[list[Message]] = []
        self.model = name

    async def stream(self, system, messages, tools):
        self.calls.append([Message(**vars(m)) for m in messages])
        script = self.scripts[min(len(self.calls) - 1, len(self.scripts) - 1)]
        for ev in script:
            if ev == "hang":
                await asyncio.Event().wait()
            if isinstance(ev, tuple) and ev[0] == "sleep":
                await asyncio.sleep(ev[1])
                continue
            await asyncio.sleep(0)
            yield ev


def tool_call(name="track_delivery", args=None, cid="c1"):
    return [LLMEvent("tool_call", tool_call=ToolCall(cid, name, args or {"order_id": "2611150231"})), LLMEvent("done")]


def text(*chunks):
    return [LLMEvent("text", text=c) for c in chunks] + [LLMEvent("done")]


def settings(**kw):
    base = dict(filler_delay_ms=700, llm_timeout_s=2.0, tts_timeout_s=2.0, min_sentence_chars=6, fillers=())
    base.update(kw)
    return Settings(**base)


async def ok_tool(name, args):
    return dict(DELIVERY)


# ───────────────────────── runner ─────────────────────────
def test_runner_records_after_tool_ms():
    frames = [(0.9, frame(2000)), (2.0, frame(0)), (2.45, frame(3000))]
    calls = [{"name": "track_delivery", "arguments": {}, "matched": True, "t_response": 1002.0}]
    out = R.summarize(frames, [{"t": 3.0, "type": "turn.end"}], calls=calls, wall_speech_end=1000.0)
    assert out["after_tool_sound"] == pytest.approx(0.45)
    assert out["after_tool_ms"] == pytest.approx(450, abs=1)


def test_runner_after_tool_ms_none_when_silent_after_tool_and_absent_without_tools():
    calls = [{"name": "x", "arguments": {}, "matched": True, "t_response": 1002.0}]
    out = R.summarize([(0.9, frame(2000))], [], calls=calls, wall_speech_end=1000.0)
    assert out["after_tool_ms"] is None
    out = R.summarize([(0.9, frame(2000))], [], calls=[], wall_speech_end=1000.0)
    assert "after_tool_ms" not in out


# ───────────────────────── score ─────────────────────────
def row(**kw):
    base = {"case_id": "A", "first_sound": 2.2, "turn_end": 6.0, "spoke_before_end": False,
            "expected_tools": [], "tool_calls": []}
    base.update(kw)
    return base


TOOLROW = dict(tool_calls=[{"name": "x", "arguments": {}, "matched": True, "t_response": 1.0}])


def test_official_speed_averages_first_sound_and_after_tool():
    assert S.speed_official(row(**TOOLROW, after_tool_sound=0.5)) == pytest.approx((80 + 100) / 2)
    assert S.speed_official(row(**TOOLROW, after_tool_sound=2.6)) == pytest.approx((80 + 70) / 2)
    assert S.speed_official(row(**TOOLROW, after_tool_sound=None)) == pytest.approx(40)   # 답 없음 → 0점
    assert S.speed_official(row()) == 80                                                # 함수 없는 문항은 첫 소리만
    assert S.speed_official(row(**TOOLROW, after_tool_sound=0.5, spoke_before_end=True)) == 0


def test_score_rows_keeps_old_speed_and_adds_official():
    sc = S.score_rows([row(**TOOLROW, after_tool_sound=2.6, after_tool_ms=2600,
                           expected_tools=[{"name": "x", "arguments": {}}])])[0]["scores"]
    assert sc["speed"] == 80                                   # 기존 지표 그대로
    assert sc["speed_after_tool"] == 70
    assert sc["speed_official"] == pytest.approx(75)
    assert sc["total_no_judge_official"] == pytest.approx((0.2 * 100 + 0.3 * 75) / 0.5)
    assert sc["total_no_judge"] == pytest.approx((0.2 * 100 + 0.3 * 80) / 0.5)


def test_summary_has_after_tool_percentiles_and_official_speed():
    rows = [row(case_id="A", **TOOLROW, after_tool_sound=0.2, after_tool_ms=200),
            row(case_id="B", **TOOLROW, after_tool_sound=1.8, after_tool_ms=1800),
            row(case_id="C")]
    s = S.summarize(S.score_rows(rows))
    assert s["after_tool_n"] == 2
    assert s["after_tool_p50"] == pytest.approx(0.2) and s["after_tool_p95"] == pytest.approx(1.8)
    assert s["speed_official"] == pytest.approx(((80 + 100) / 2 + (80 + 90) / 2 + 80) / 3)


def test_score_run_merges_turn_log_answer_latency(tmp_path):
    run = tmp_path / "run"
    (run / "logs").mkdir(parents=True)
    r = row(case_id="T1", session_id="T1-abc", **TOOLROW, after_tool_sound=0.15, after_tool_ms=150)
    (run / "results.jsonl").write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
    (run / "logs" / "turns-20261120.jsonl").write_text(json.dumps(
        {"session_id": "T1-abc", "after_tool_ms": 140.0, "after_tool_answer_ms": 1900.0,
         "marks": {}, "events": []}) + "\n", encoding="utf-8")
    sc, report = S.score_run(run)
    assert sc[0]["after_tool_answer_ms"] == 1900.0
    assert sc[0]["scores"]["speed_official_answer"] == pytest.approx((80 + 90) / 2)
    assert sc[0]["scores"]["speed_official"] == pytest.approx((80 + 100) / 2)
    assert "조회 뒤" in report


# ───────────────────────── responder: 측정 ─────────────────────────
async def test_turn_log_after_tool_ms_counts_filler_but_answer_excludes_it(tmp_path):
    from agent.tts_cache import TTSCache
    cache = TTSCache(tmp_path)
    filler = "확인해 드릴게요, 잠시만요."
    cache.put(filler, b"\xaa\x55" * 8)
    llm = ScriptLLM(tool_call(), [("sleep", 0.15)] + text("현재 배송중입니다. 감사합니다."))
    tlog = TurnLog("s1")
    r = Responder(llm, FakeTTS(), ok_tool, tools=TOOLS, cache=cache,
                  settings=settings(filler_delay_ms=20, fillers=(filler,)))
    spoken = await r.respond("S", [Message("user", "배송 어디쯤이에요?")], Sink(), log=tlog)
    assert spoken[0] == filler
    f = tlog.fields
    assert f["after_tool_ms"] is not None and f["after_tool_answer_ms"] is not None
    assert f["after_tool_ms"] < 100 <= f["after_tool_answer_ms"]


async def test_first_post_tool_sentence_goes_to_tts_before_stream_ends():
    tts = FakeTTS()
    llm = ScriptLLM(tool_call(), [LLMEvent("text", text="현재 배송중입니다. "), "hang"])
    r = Responder(llm, tts, ok_tool, tools=TOOLS, settings=settings(llm_timeout_s=0.3))
    task = asyncio.create_task(r.respond("S", [Message("user", "배송 어디쯤?")], Sink()))
    for _ in range(100):
        if tts.requests:
            break
        await asyncio.sleep(0.005)
    assert tts.requests == ["현재 배송중입니다."]
    with pytest.raises(Exception):
        await task                                    # 멈춘 스트림은 llm_timeout으로 끝난다


# ───────────────────────── responder: 후속 hedge ─────────────────────────
def hedged_pair(slow_followup_s: float):
    a = ScriptLLM(tool_call(), [("sleep", slow_followup_s)] + text("A 모델 답입니다."), name="model-a")
    b = ScriptLLM(text("B 모델 답입니다."), name="model-b")
    return a, b, HedgedLLM([a, b], hedge_delay_s=0.05, first_event_timeout_s=5.0)


async def test_post_tool_round_is_hedged_when_origin_model_slow(monkeypatch):
    monkeypatch.setenv("DIRECTON_POST_TOOL_HEDGE", "1")
    a, b, h = hedged_pair(1.0)
    r = Responder(h, FakeTTS(), ok_tool, tools=TOOLS, settings=settings())
    t0 = time.perf_counter()
    spoken = await r.respond("S", [Message("user", "배송 어디쯤?")], Sink())
    assert time.perf_counter() - t0 < 0.6
    assert spoken == ["B 모델 답입니다."]
    assert len(a.calls) == 2 and len(b.calls) == 1        # 후속: 원래 모델 먼저, 지연 뒤 대체 모델
    assert b.calls[0][-1].role == "tool"


async def test_post_tool_hedge_off_keeps_pinned_model(monkeypatch):
    monkeypatch.setenv("DIRECTON_POST_TOOL_HEDGE", "0")
    a, b, h = hedged_pair(0.2)
    r = Responder(h, FakeTTS(), ok_tool, tools=TOOLS, settings=settings())
    spoken = await r.respond("S", [Message("user", "배송 어디쯤?")], Sink())
    assert spoken == ["A 모델 답입니다."] and b.calls == []


# ───────────────────────── responder: 결과 템플릿 ─────────────────────────
def test_template_uses_only_tool_result_facts():
    assert tool_result_template("track_delivery", DELIVERY) == "배송중입니다."
    t = tool_result_template("get_order_info", {"status": "ok", "data": {"order_id": "1", "status": "상품준비중"}})
    assert t == "확인해 보니 현재 상품준비중 상태입니다."
    # 숫자가 섞인 메시지는 읽기가 불안하므로 상태 문장으로 대신한다
    t = tool_result_template("track_delivery", {"status": "ok", "data": {"status": "배송완료"},
                                                "message": "2026-11-18에 배송 완료되었습니다."})
    assert t == "확인해 보니 현재 배송완료 상태입니다."
    # 내부 차단·오류·정보 없음 → 템플릿 없음
    assert tool_result_template("track_delivery", {"status": "invalid_arguments", "message": "다시 묻는다."}) is None
    assert tool_result_template("track_delivery", {"status": "error", "message": "tool call failed"}) is None
    assert tool_result_template("get_coupons", {"status": "ok", "data": {"coupons": []}}) is None
    # 쓰기 함수(취소 등)는 상태만으로 지어내지 않는다 — 메시지가 있어야 한다
    assert tool_result_template("cancel_order", {"status": "ok", "data": {"status": "취소완료"}}) is None
    assert tool_result_template("cancel_order", {"status": "not_allowed", "data": {"status": "배송중"},
                                                 "message": "이미 배송이 시작되어 취소할 수 없습니다. 반품을 신청해 주세요."}) \
        == "이미 배송이 시작되어 취소할 수 없습니다."
    # 상담원에게 주는 지시처럼 보이는 메시지는 말하지 않는다
    assert tool_result_template("x", {"status": "ok", "message": "고객에게 반품 절차를 안내하세요."}) is None


async def test_template_spoken_first_and_llm_told_not_to_repeat(monkeypatch):
    monkeypatch.setenv("DIRECTON_TOOL_TEMPLATE", "1")
    tts = FakeTTS()
    llm = ScriptLLM(tool_call(), text("배송중입니다. 내일 도착 예정입니다."))
    tlog = TurnLog("s1")
    r = Responder(llm, tts, ok_tool, tools=TOOLS, settings=settings())
    spoken = await r.respond("S", [Message("user", "배송 어디쯤?")], Sink(), log=tlog)
    assert spoken == ["배송중입니다.", "내일 도착 예정입니다."]       # LLM이 같은 문장을 되풀이하면 뺀다
    body = json.loads([m for m in llm.calls[1] if m.role == "tool"][0].content)
    assert body["_already_said"] == "배송중입니다."
    assert any(e["name"] == "tool_template" for e in tlog.events)


async def test_template_off_by_env(monkeypatch):
    monkeypatch.setenv("DIRECTON_TOOL_TEMPLATE", "0")
    llm = ScriptLLM(tool_call(), text("현재 배송중이에요."))
    r = Responder(llm, FakeTTS(), ok_tool, tools=TOOLS, settings=settings())
    spoken = await r.respond("S", [Message("user", "배송 어디쯤?")], Sink())
    assert spoken == ["현재 배송중이에요."]
    body = json.loads([m for m in llm.calls[1] if m.role == "tool"][0].content)
    assert "_already_said" not in body


async def test_template_cancels_pending_filler(monkeypatch, tmp_path):
    from agent.tts_cache import TTSCache
    monkeypatch.setenv("DIRECTON_TOOL_TEMPLATE", "1")
    cache = TTSCache(tmp_path)
    filler = "확인해 드릴게요, 잠시만요."
    cache.put(filler, b"\xaa\x55" * 8)
    llm = ScriptLLM(tool_call(), [("sleep", 0.2)] + text("내일 도착 예정입니다."))
    r = Responder(llm, FakeTTS(), ok_tool, tools=TOOLS, cache=cache,
                  settings=settings(filler_delay_ms=100, fillers=(filler,)))
    spoken = await r.respond("S", [Message("user", "배송 어디쯤?")], Sink())
    assert filler not in spoken and spoken[0] == "배송중입니다."


# ───────────────────────── 자릿수 상대화 ─────────────────────────
def test_repair_thresholds_scale_with_order_id_length():
    assert repair_thresholds(10) == (7, 2, 6, 2)          # 기존 10자리 값 그대로
    min_common, max_missing, min_digits, margin = repair_thresholds(4)
    assert min_common == 3 and max_missing == 1 and min_digits == 3 and margin == 1


def test_match_order_id_four_digit_ids():
    ids = ["1234", "5678", "9012"]
    assert match_order_id("123", ids, 4) == "1234"         # 끝자리 빠짐
    assert match_order_id("12344", ids, 4) == "1234"       # 한 자리 더 들림
    assert match_order_id("12", ids, 4) is None            # 너무 짧다
    assert match_order_id("1234", ids, 4) is None          # 자릿수가 맞으면 교정 대상 아님


def test_match_order_id_length_follows_known_ids_when_rule_missing():
    ids = ["1234", "5678"]
    assert match_order_id("123", ids) == "1234"            # 길이 인자 없이도 알려진 주문번호 길이(4) 기준


# ───────────────────────── 짧은 주문번호(공식 예시 "주문번호 1234") ─────────────────────────
def test_utterance_numbers_keeps_short_ids_when_min_digits_given():
    from agent.responder import utterance_numbers
    assert utterance_numbers("주문번호 일이삼사 취소해 주세요.") == []              # 기본 데이터(10자리)는 그대로
    assert utterance_numbers("주문번호 일이삼사 취소해 주세요.", min_digits=4) == ["1234"]
    assert utterance_numbers("이 주문 취소해 주세요.", min_digits=4) == []


def test_number_min_digits_follows_known_order_lengths():
    from agent.responder import number_min_digits
    assert number_min_digits(frozenset({10}), frozenset({11})) == 7       # 기존 값
    assert number_min_digits(frozenset({4}), frozenset({11})) == 4


def test_ground_short_numbers_only_one_edit():
    from agent.responder import ground_number_args
    assert ground_number_args({"order_id": "1235"}, ["1234"]) == {"order_id": "1234"}
    assert ground_number_args({"order_id": "5634"}, ["1234"]) == {"order_id": "5634"}   # 두 자리 다르면 다른 번호
    assert ground_number_args({"order_id": "2611140499"}, ["2611140498"]) == {"order_id": "2611140498"}


async def test_responder_grounds_four_digit_order_from_utterance(monkeypatch):
    import agent.responder as RS
    monkeypatch.setattr(RS, "_known_lengths", lambda: (frozenset({4}), frozenset({11})))
    seen = []

    async def tool(name, args):
        seen.append(args)
        return {"status": "ok", "data": {"order_id": args["order_id"], "status": "배송중"}}

    llm = ScriptLLM(tool_call(args={"order_id": "1235"}), text("배송중이에요."))
    r = Responder(llm, FakeTTS(), tool, tools=TOOLS, settings=settings())
    await r.respond("S", [Message("user", "주문번호 일이삼사 어디쯤이에요?")], Sink())
    assert seen == [{"order_id": "1234"}]


# ───────────────────────── 로그에 고객 번호를 남기지 않음(AG-T18 요청) ─────────────────────────
async def test_number_logs_redacted_in_no_audio_log_mode(monkeypatch, caplog):
    import logging
    monkeypatch.setenv("DIRECTON_NO_AUDIO_LOG", "1")

    async def tool(name, args):
        return {"status": "ok", "data": {"order_id": args["order_id"], "status": "배송중"}}

    llm = ScriptLLM(tool_call(args={"order_id": "2611140499"}), text("배송중이에요."))
    r = Responder(llm, FakeTTS(), tool, tools=TOOLS, settings=settings())
    with caplog.at_level(logging.INFO, logger="agent.responder"):
        await r.respond("S", [Message("user", "이육일일 일사공 사구팔 어디쯤이에요?")], Sink())
    assert "grounded" in caplog.text
    assert "2611140498" not in caplog.text and "2611140499" not in caplog.text


def test_repeats_template_catches_ending_variation_only():
    from agent.responder import repeats_template
    assert repeats_template("이미 반품이 접수된 주문이에요.", "이미 반품이 접수된 주문입니다.")
    assert repeats_template("배송중입니다.", "배송중입니다.")
    assert not repeats_template("배송이 완료된 주문은 취소할 수 없어서 수령 후 칠일 이내라면 반품을 신청해 주세요.",
                                "배송이 완료된 주문은 취소할 수 없습니다.")
    assert not repeats_template("결제하신 수단으로 칠만 원이 환불될 예정이에요.", "주문이 취소되었습니다.")


def test_switch_defaults(monkeypatch):
    """후속 hedge는 기본 켬. 템플릿은 코드 기본 끔(기존 llm_eval 필러 테스트 보존) — 서버는 환경변수로 켠다."""
    from agent.responder import template_enabled, post_hedge_enabled
    monkeypatch.delenv("DIRECTON_TOOL_TEMPLATE", raising=False)
    monkeypatch.delenv("DIRECTON_POST_TOOL_HEDGE", raising=False)
    assert post_hedge_enabled() and not template_enabled()
    monkeypatch.setenv("DIRECTON_TOOL_TEMPLATE", "1")
    assert template_enabled()
