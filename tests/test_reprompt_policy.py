"""AG-T15a 번호 되묻기 정책 재설계 + 응답 내용 보강 + llm_eval STT 텍스트 모드.

- 프롬프트: 자릿수가 틀린 번호도 스스로 되묻지 말고 들은 그대로 함수를 호출(조회 우선). 되묻기는
  결과가 invalid_arguments / not_found일 때만. 응답은 상황별 3~4문장까지, 함수 결과의 핵심 필드를 빠뜨리지 않는다.
- 응답기: 자리 수가 틀린 주문번호는 그대로 보내지 않는다(기존 단언 유지). 대신 발신 고객 전화번호가 있으면
  get_order_list로 고객 주문을 받아 번호가 부분 일치(공통 부분열)하는 주문이 하나뿐이면 그 주문번호로 바꿔 호출한다.
  막힌 호출의 invalid_arguments 안내에는 정책 안내를 함께 하라는 지시가 들어간다.
- llm_eval: --stt-text로 문항별 실제 STT 최종 텍스트를 LLM 입력으로 쓴다(심사·기록의 고객 발화는 정답 그대로).
"""
import asyncio
import json

import pytest

from agent.config import Settings
from agent.core.types import LLMEvent, Message, SessionInfo, ToolCall
from agent.prompts import FIXED, build_fixed, build_system
from agent.responder import Responder, caller_phone_from_system, lcs_len, match_order_id
from eval import llm_eval
from mock_api.server import load_shop

CALLER = {"name": "한소율", "phone": "01076543210", "email": None}
ORDERS = [{"order_id": "2611160457", "status": "배송중"}, {"order_id": "2611140498", "status": "배송완료"},
          {"order_id": "2610300266", "status": "배송완료"}]


@pytest.fixture(scope="module")
def shop():
    return load_shop()


@pytest.fixture(scope="module")
def tools(shop):
    return llm_eval.shop_tools(shop)


def system_for(caller=None, history=None):
    info = SessionInfo("S", now="2026-11-20 14:00", caller=caller, history=history or [])
    return build_system(info, "네, 고객님.", manual="매뉴얼 본문")


# ── fakes ────────────────────────────────────────────
class _TTS:
    async def synthesize(self, text):
        yield text.encode("utf-8")


async def _sink(pcm):
    return None


class ScriptLLM:
    """회차별 이벤트. 받은 메시지를 기록한다."""

    def __init__(self, rounds):
        self.rounds = rounds
        self.seen: list[list[Message]] = []

    async def stream(self, system, messages, tools):
        self.seen.append(list(messages))
        for ev in self.rounds[min(len(self.seen) - 1, len(self.rounds) - 1)]:
            await asyncio.sleep(0)
            yield ev


def calls(*tcs):
    return [LLMEvent("tool_call", tool_call=ToolCall(f"c{i}", n, a)) for i, (n, a) in enumerate(tcs)] + [LLMEvent("done")]


def say(t):
    return [LLMEvent("text", text=t), LLMEvent("done")]


class FakeTools:
    def __init__(self, orders=ORDERS):
        self.orders = orders
        self.seen: list[tuple[str, dict]] = []

    async def __call__(self, name, args):
        self.seen.append((name, dict(args)))
        if name == "get_order_list":
            return {"status": "ok", "data": {"name": "한소율", "orders": list(self.orders)}}
        return {"status": "ok", "data": {"order_id": args.get("order_id")}}


async def run(llm, tool, tools, system, utt="환불 상태 봐 주세요."):
    r = Responder(llm, _TTS(), tool, tools=tools, settings=Settings())
    spoken = await r.respond(system, [Message("user", utt)], _sink)
    return spoken


# ── 프롬프트: 조회 우선 + 조건부 되묻기 ─────────────────
def test_prompt_lookup_first_then_reask_only_on_failure():
    assert "들은 번호 그대로 함수를 호출" in FIXED
    assert "invalid_arguments" in FIXED and "not_found" in FIXED
    assert "일 때만" in FIXED                                  # 되묻기는 결과가 실패일 때만
    # 기존 규칙 문구도 유지(tests/test_prompts.py와 충돌하지 않음)
    assert "자릿수가 맞으면 되묻지 말고 바로 호출" in FIXED


def test_prompt_first_ask_is_plain_and_asks_all_needed_info():
    # 처음 묻는 번호는 '한 자리씩 천천히'를 붙이지 않고, 실행에 필요한 다른 정보(새 주소 등)도 함께 묻는다
    assert "처음 물을 때" in FIXED
    assert "새 주소" in FIXED


def test_prompt_reask_includes_policy_info():
    # 되물을 때도 매뉴얼로 답할 수 있는 정책 안내를 함께 한다
    assert "정책" in FIXED and "함께" in FIXED


# ── 프롬프트: 응답 내용(심사) ──────────────────────────
def test_prompt_sentence_limit_relaxed_by_situation():
    assert "2~3문장" in FIXED and "3~4문장" in FIXED


def test_prompt_key_fields_and_absolute_dates():
    for word in ("상태", "날짜", "금액", "다음 절차"):
        assert word in FIXED, word
    assert "내일" in FIXED and "월·일" in FIXED               # '내일'만 말하지 말고 월·일로


def test_detail_rules_can_be_turned_off():
    base = build_fixed(detail=False)
    assert "3~4문장" not in base and "3~4문장" in build_fixed(detail=True)
    assert "자릿수가 맞으면 되묻지 말고 바로 호출" in base       # 되묻기 정책은 두 변형 모두


def test_prompt_has_no_test_utterances():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    with open(root / "data/testset/cases.jsonl", encoding="utf-8") as f:
        utts = [json.loads(l)["customer_utterance"] for l in f if l.strip()]
    for fixed in (FIXED, build_fixed(detail=False)):
        assert not [u for u in utts if len(u) >= 8 and u in fixed]


# ── 부분 일치 주문 선택 ────────────────────────────────
def test_lcs_len():
    assert lcs_len("26140498", "2611140498") == 8
    assert lcs_len("", "123") == 0


def test_match_order_id_unique_partial_match():
    ids = [o["order_id"] for o in ORDERS]
    assert match_order_id("26140498", ids) == "2611140498"          # 두 자리 빠짐
    assert match_order_id("261114049", ids) == "2611140498"         # 끝자리 빠짐
    assert match_order_id("26111404988", ids) == "2611140498"       # 한 자리 더 들림


def test_match_order_id_rejects_far_or_ambiguous():
    ids = [o["order_id"] for o in ORDERS]
    assert match_order_id("26112992", ids) is None                  # 6자리만 겹침
    assert match_order_id("261115023", ["2611150231", "2611150232"]) is None   # 둘 다 같은 만큼 일치
    assert match_order_id("123", ids) is None                       # 번호라 보기 어려운 짧은 수
    assert match_order_id("2611140498", ids) is None                # 자릿수가 맞으면 교정 대상 아님


def test_caller_phone_from_system():
    assert caller_phone_from_system(system_for(CALLER)) == "01076543210"
    assert caller_phone_from_system(system_for(None)) == ""
    hist = [{"role": "customer", "text": "제 번호는 010-2345-6781이에요."},
            {"role": "agent", "text": "확인했어요. 어떤 주문을 도와드릴까요?"}]
    assert caller_phone_from_system(system_for(None, hist)) == "01023456781"   # 이전 대화에서 나온 번호


# ── 응답기: 발신 고객 주문 목록으로 번호 교정 ─────────────
async def test_wrong_length_order_repaired_from_caller_order_list(tools):
    llm = ScriptLLM([calls(("get_refund_status", {"order_id": "26140498"})), say("환불 내역이 없어요. ")])
    tool = FakeTools()
    await run(llm, tool, tools, system_for(CALLER))
    assert tool.seen == [("get_order_list", {"phone_number": "01076543210"}),
                         ("get_refund_status", {"order_id": "2611140498"})]
    # LLM이 보는 대화에도 교정된 번호가 들어간다
    asst = [m for m in llm.seen[1] if m.role == "assistant"][0]
    assert asst.tool_calls[0].arguments == {"order_id": "2611140498"}
    tool_msg = [m for m in llm.seen[1] if m.role == "tool"][0]
    assert json.loads(tool_msg.content)["status"] == "ok"


async def test_no_repair_without_caller_blocks_as_before(tools):
    llm = ScriptLLM([calls(("get_refund_status", {"order_id": "26140498"})), say("번호를 다시 말씀해 주세요. ")])
    tool = FakeTools()
    await run(llm, tool, tools, system_for(None))
    assert tool.seen == []                                     # 주문 목록도, 틀린 번호도 보내지 않는다
    tool_msg = [m for m in llm.seen[1] if m.role == "tool"][0]
    body = json.loads(tool_msg.content)
    assert body["status"] == "invalid_arguments" and "한 자리씩" in body["message"]


async def test_no_repair_when_ambiguous_blocks(tools):
    llm = ScriptLLM([calls(("get_order_info", {"order_id": "261115023"})), say("다시 말씀해 주세요. ")])
    tool = FakeTools(orders=[{"order_id": "2611150231"}, {"order_id": "2611150232"}])
    await run(llm, tool, tools, system_for(CALLER))
    assert [n for n, _ in tool.seen] == ["get_order_list"]      # 목록만 보고 본 호출은 막는다
    body = json.loads([m for m in llm.seen[1] if m.role == "tool"][0].content)
    assert body["status"] == "invalid_arguments"


async def test_repair_reuses_order_list_called_earlier_in_turn(tools):
    llm = ScriptLLM([calls(("get_order_list", {"phone_number": "01076543210"})),
                     calls(("get_refund_status", {"order_id": "26140498"})), say("환불 내역이 없어요. ")])
    tool = FakeTools()
    await run(llm, tool, tools, system_for(CALLER))
    assert [n for n, _ in tool.seen] == ["get_order_list", "get_refund_status"]
    assert tool.seen[1][1] == {"order_id": "2611140498"}


async def test_repair_can_be_disabled_by_env(tools, monkeypatch):
    monkeypatch.setenv("DIRECTON_REPAIR_ORDER_ID", "0")
    llm = ScriptLLM([calls(("get_refund_status", {"order_id": "26140498"})), say("다시 말씀해 주세요. ")])
    tool = FakeTools()
    await run(llm, tool, tools, system_for(CALLER))
    assert tool.seen == []


async def test_wrong_length_phone_close_to_caller_uses_caller_phone(tools):
    llm = ScriptLLM([calls(("get_coupons", {"phone_number": "0107654321"})), say("쿠폰이 있어요. ")])
    tool = FakeTools()
    await run(llm, tool, tools, system_for(CALLER), utt="쿠폰 있어요?")
    assert tool.seen == [("get_coupons", {"phone_number": "01076543210"})]


async def test_blocked_message_asks_policy_info_with_reask(tools):
    llm = ScriptLLM([calls(("cancel_order", {"order_id": "261115023"})), say("다시 말씀해 주세요. ")])
    await run(llm, FakeTools(), tools, system_for(None), utt="취소해 주세요.")
    msg = json.loads([m for m in llm.seen[1] if m.role == "tool"][0].content)["message"]
    assert "한 자리씩" in msg and "열 자리" in msg and "정책" in msg


async def test_right_length_order_not_touched(tools):
    llm = ScriptLLM([calls(("get_refund_status", {"order_id": "2611140498"})), say("환불 내역이 없어요. ")])
    tool = FakeTools()
    await run(llm, tool, tools, system_for(CALLER))
    assert tool.seen == [("get_refund_status", {"order_id": "2611140498"})]


# ── llm_eval: STT 텍스트 모드 ─────────────────────────
def test_load_stt_texts_from_stt_eval_rows(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [
        {"case_id": "T1", "ref": "정답", "final": "2611 150 231 언제 와요?"},
        {"case_id": "T2", "ref": "정답2", "final": "", "error": "boom"}]), encoding="utf-8")
    assert llm_eval.load_stt_texts(p) == {"T1": "2611 150 231 언제 와요?"}


def test_load_stt_texts_from_turn_log_filtered_by_run_sessions(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "results.jsonl").write_text(json.dumps({"case_id": "T1", "session_id": "T1-aaaa"}) + "\n", encoding="utf-8")
    log = tmp_path / "turns.jsonl"
    log.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [
        {"session_id": "T1-zzzz", "user_text": "다른 실행"},
        {"session_id": "T1-aaaa", "user_text": "이번 실행 텍스트"},
        {"session_id": "T9-bbbb", "user_text": "실행 밖"}]), encoding="utf-8")
    sids = llm_eval.run_session_ids(run)
    assert llm_eval.load_stt_texts(log, sessions=sids) == {"T1": "이번 실행 텍스트"}


def test_load_stt_texts_from_json_mapping(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"T1": "가", "T2": "나"}, ensure_ascii=False), encoding="utf-8")
    assert llm_eval.load_stt_texts(p) == {"T1": "가", "T2": "나"}


def _case(cid, utt, expected, caller=None):
    return {"case_id": cid, "now": "2026-11-20 14:00", "history": [], "customer_profile": caller,
            "customer_utterance": utt, "category": "tool_lookup", "tags": [], "expected_tools": expected,
            "expected_points": ["p"], "must_not": []}


class UttLLM:
    def __init__(self):
        self.utts = []

    async def stream(self, system, messages, tools):
        self.utts.append(next(m.content for m in messages if m.role == "user"))
        yield LLMEvent("text", text="주문번호를 한 자리씩 천천히 다시 말씀해 주시겠어요? ")
        yield LLMEvent("done")


async def test_evaluate_uses_stt_text_as_llm_input_but_keeps_reference(shop):
    llm = UttLLM()
    cases = [_case("T1", "정답 발화", []), _case("T2", "두번째 정답", [])]
    seen = []

    def judge(row):
        seen.append(row)
        return {"points": [True], "violations": []}

    rows = await llm_eval.evaluate(cases, lambda: llm, shop=shop, stt_texts={"T1": "STT 발화"}, judge_fn=judge)
    assert llm.utts == ["STT 발화", "두번째 정답"] or sorted(llm.utts) == sorted(["STT 발화", "두번째 정답"])
    r1, r2 = rows
    assert r1["customer_utterance"] == "정답 발화" and r1["input_utterance"] == "STT 발화"
    assert r1.get("stt_missing") is not True
    assert r2["input_utterance"] == "두번째 정답" and r2["stt_missing"] is True   # STT 없으면 정답으로 대체·표시
    assert {s["customer_utterance"] for s in seen} == {"정답 발화", "두번째 정답"}   # 심사는 정답 발화 기준


async def test_reprompt_counted(shop):
    rows = await llm_eval.evaluate([_case("T1", "질문", [])], UttLLM, shop=shop)
    assert rows[0]["reprompt"] is True
    s = llm_eval.summarize(rows)
    assert s["overall"]["reprompts"] == 1


def test_is_reprompt():
    assert llm_eval.is_reprompt("번호를 한 자리씩 천천히 다시 말씀해 주시겠어요?")
    assert llm_eval.is_reprompt("주문번호가 열 자리인데, 한 번 더 말씀해 주시겠어요?")
    assert not llm_eval.is_reprompt("주문번호나 휴대폰 번호를 말씀해 주시겠어요?")   # 처음 묻기는 되묻기 아님
    assert not llm_eval.is_reprompt("내일 도착해요.")


# ── 응답기: 숫자가 아닌 번호 인자(게이트웨이 마스킹 "[PHONE]", "unknown", 빈 값) ─────────────
# BAZE 경유 LLM은 전화번호가 가려진 채 보고 "[PHONE]"을 그대로 인자로 넣는다. 응답기는 가려지지 않은 원문을
# 갖고 있으므로 발화의 번호 → 확인된 고객 번호 순으로 채우고, 채울 수 없으면 호출하지 않는다(여분 호출 감점 방지).
async def test_masked_phone_filled_from_utterance(tools):
    llm = ScriptLLM([calls(("get_coupons", {"phone_number": "[PHONE]"})), say("쿠폰이 있어요. ")])
    tool = FakeTools()
    await run(llm, tool, tools, system_for(None), utt="01054321098 번호로 쿠폰 조회해 주세요.")
    assert tool.seen == [("get_coupons", {"phone_number": "01054321098"})]


async def test_masked_phone_filled_from_caller(tools):
    llm = ScriptLLM([calls(("get_coupons", {"phone_number": "[PHONE]"})), say("쿠폰이 있어요. ")])
    tool = FakeTools()
    await run(llm, tool, tools, system_for(CALLER), utt="쿠폰 있어요?")
    assert tool.seen == [("get_coupons", {"phone_number": "01076543210"})]


@pytest.mark.parametrize("args", [{"phone_number": "unknown"}, {}, {"phone_number": ""},
                                  {"phone_number": "주문번호나 휴대폰 번호를 말씀해 주시겠어요?"}])
async def test_number_arg_without_digits_not_sent(tools, args):
    llm = ScriptLLM([calls(("request_callback", args)), say("휴대폰 번호를 말씀해 주시겠어요? ")])
    tool = FakeTools()
    await run(llm, tool, tools, system_for(None), utt="콜백 해 주세요.")
    assert tool.seen == []
    body = json.loads([m for m in llm.seen[1] if m.role == "tool"][0].content)
    assert body["status"] == "invalid_arguments" and "한 자리씩" not in body["message"]   # 처음 묻기


async def test_prompt_example_phone_not_used_when_customer_did_not_say_it(tools):
    llm = ScriptLLM([calls(("get_coupons", {"phone_number": "01012345678"})), say("번호를 말씀해 주세요. ")])
    tool = FakeTools()
    await run(llm, tool, tools, system_for(None), utt="쿠폰 있어요?")
    assert tool.seen == []


async def test_prompt_example_phone_used_when_customer_said_it(tools):
    llm = ScriptLLM([calls(("get_coupons", {"phone_number": "01012345678"})), say("쿠폰이 있어요. ")])
    tool = FakeTools()
    await run(llm, tool, tools, system_for(None), utt="공일공 일이삼사 오육칠팔이요.")
    assert tool.seen == [("get_coupons", {"phone_number": "01012345678"})]


# ── LLM이 덜 들린 번호에 숫자를 지어 붙여 자릿수를 맞춘 경우 ─────────────
def test_utterance_numbers_handles_iya_ending():
    from agent.responder import utterance_numbers
    assert "010345781" in utterance_numbers("제 번호는 010-3457-81이야.")


async def test_padded_phone_grounded_back_to_heard_and_blocked(tools):
    llm = ScriptLLM([calls(("get_member_info", {"phone_number": "01034578191"})), say("다시 말씀해 주세요. ")])
    tool = FakeTools()
    await run(llm, tool, tools, system_for(None), utt="제 번호는 010-3457-81이야.")
    assert tool.seen == []                        # 9자리로 되돌려져 자릿수 검사에 막힌다
