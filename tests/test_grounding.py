"""AG-T48 근거 없는 사실 금지: 프롬프트 근거 규칙·결제 수단별 환불 정리, 조회 없이 조회 결과를 말하면 다시 생성.

공식 측정(main 17380d6) 즉시 0점 5건 중 4건이 지어낸 사실(조회 없이 '입금 확인이 안 된다', 브랜드 정보에 없는 계산 방식,
'미확인'인 박스당 일수, 선불금 환불 기간을 카드 기간으로). 외부 호출 없음(fake LLM·TTS·툴).
"""
import json

import pytest

from agent import catalog as catalog_mod
from agent.core.types import Message, SessionInfo
from agent.grounding import GROUNDING_HEADER, REFUND_HEADER
from agent.prompts import VAR_MARKER, build_system
from agent.tool_guard import FALLBACK_TEXT, is_lookup_claim
from tests.test_tool_guard import ScriptLLM, call, make, run, system_with, text


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("DIRECTON_CATALOG_PATH", "DIRECTON_MOCK_DATA", "DIRECTON_CONFIRM_ORDER_ID", "DIRECTON_REPAIR_ORDER_ID",
              "DIRECTON_TOOL_TEMPLATE"):
        monkeypatch.delenv(k, raising=False)
    catalog_mod.reset_default_catalog()
    yield
    catalog_mod.reset_default_catalog()


def _brand(tmp_path, entries):
    d = tmp_path / "brand"
    d.mkdir()
    (d / "information.json").write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    return d


def _sys(brand_dir):
    info = SessionInfo("S", now="2026-11-04 (수) 11:00", caller=None, history=[])
    return build_system(info, "네, 고객님.", manual="", brand_dir=brand_dir)


# ── 프롬프트 ─────────────────────────────────────────────
def test_grounding_rules_in_fixed_part(tmp_path):
    s = _sys(_brand(tmp_path, [{"tag": "배송비", "content": "3만원 이상 무료"}]))
    fixed = s[:s.index(VAR_MARKER)]
    assert GROUNDING_HEADER in fixed
    rules = fixed[fixed.index(GROUNDING_HEADER):]
    assert "조회하지 않았으면" in rules                  # 조회 없이 조회 결과를 말하지 않는다
    assert "며칠분" in rules and "계산하지" in rules      # 정 수·며칠분을 짐작·계산하지 않는다
    assert "적용 순서" in rules                           # 할인·배송비 계산 방식 추측 금지
    assert "다른 제품" in rules                           # 검색 결과의 다른 제품 정보를 옮기지 않는다
    assert "식전·식후" in rules                           # 섭취 방법은 결과 문구 그대로


def test_refund_table_only_with_prepaid_brand(tmp_path):
    haruon = _sys(_brand(tmp_path, [{"tag": "[결제수단 별 환불기간 일정]",
                                     "content": "- 선불금: 환불완료 후 즉시 환불 처리된다."}]))
    fixed = haruon[:haruon.index(VAR_MARKER)]
    assert REFUND_HEADER in fixed
    table = fixed[fixed.index(REFUND_HEADER):]
    assert "선불금" in table and "1~2일" in table and "계좌를 묻지 않는다" in table
    assert "카드" in table and "3~4일" in table
    assert "결제수단" in table                            # 주문 조회 결과의 결제수단 줄만 쓴다
    assert "환불보장" in table


def test_refund_table_absent_for_other_brand(tmp_path):
    other = _sys(_brand(tmp_path, [{"tag": "배송비", "content": "3만원 이상 무료"}]))
    assert REFUND_HEADER not in other


# ── 조회 결과 주장 판별 ────────────────────────────────────
@pytest.mark.parametrize("s", [
    "입금 확인이 되지 않았어요.", "입력하신 주문번호로는 조회가 어려워서, 연락처를 알려 주세요.",
    "주문 내역이 없어요.", "그 번호로는 주문을 찾을 수 없어요.", "결제가 확인되지 않습니다.",
    "반품 내역이 조회되지 않아요."])
def test_lookup_claim_positive(s):
    assert is_lookup_claim(s)


@pytest.mark.parametrize("s", [
    "입금 확인까지 20분 정도 걸릴 수 있어요.", "20분이 지나도 입금 확인이 안 되면 다시 말씀해 주세요.",
    "정확한 정 수는 확인이 어려워요.", "할인 행사 기간은 미리 확인이 어려워요.",
    "입금이 확인되지 않으셨나요?", "주문 내역은 마이페이지에서 확인하실 수 있어요.",
    "배송 일정은 택배사 사정에 따라 확인이 어려울 수 있어요.", "입금 확인이 안 되셨다면 입금자명을 확인해 주세요.",
    "다른 쇼핑몰 주문은 저희 주문 조회로 확인이 어려워요.", "쿠팡에서 주문하신 건은 여기서 조회가 안 돼요.", "주문번호가 정확히 들리지 않아서 조회가 어려워요."])
def test_lookup_claim_negative(s):
    assert not is_lookup_claim(s)


# ── 응답기 배선 ──────────────────────────────────────────
ORDER = {"status": "ok", "data": {"orders": [{"주문번호": "20260924-0035429", "결제여부": "T", "결제수단": ["무통장입금"]}]},
         "message": "주문 조회 완료"}


def test_no_lookup_claim_regenerates_with_lookup():
    llm = ScriptLLM(text("입금 확인이 되지 않았어요. ", "주문번호를 다시 확인해 주시겠어요?"),
                    call("get_order_info", {"order_id": "20260924-0035429"}),
                    text("결제 완료된 주문이에요."))
    r, tts, seen = make(llm, {"get_order_info": ORDER})
    spoken = run(r, system_with(caller={"phone": "01047775671"}),
                 [Message("user", "주문번호 20260924-0035429 입금했는데 확인 안 되고 있어요")])
    assert not any("입금 확인이 되지 않았어요" in t for t in tts.requests)
    assert not any("다시 확인해 주시겠어요" in t for t in tts.requests)     # 주장 뒤 문장도 함께 버린다
    assert [n for n, _ in seen] == ["get_order_info"]
    assert "조회" in llm.calls[1]["system"] and llm.calls[1]["system"] != llm.calls[0]["system"]
    assert llm.calls[1]["tools"]
    assert spoken == ["결제 완료된 주문이에요."]


def test_claim_after_lookup_this_turn_passes():
    llm = ScriptLLM(call("get_order_info", {"order_id": "20260924-0035429"}),
                    text("그 번호로는 주문을 찾을 수 없어요. 번호를 한 번 더 확인해 주시겠어요?"))
    r, tts, seen = make(llm, {"get_order_info": {"status": "not_found", "message": "주문 없음"}})
    spoken = run(r, system_with(), [Message("user", "20260924-0035429 주문 확인해 주세요")])
    assert spoken == ["그 번호로는 주문을 찾을 수 없어요.", "번호를 한 번 더 확인해 주시겠어요?"]
    assert len(llm.calls) == 2


def test_claim_with_lookup_in_history_passes():
    hist = [{"role": "customer", "text": "주문 확인해 주세요"},
            {"role": "tool_call", "name": "get_order_info", "arguments": {"order_id": "20260924-0035429"}},
            {"role": "tool_result", "name": "get_order_info", "result": {"status": "not_found", "message": "주문 없음"}},
            {"role": "agent", "text": "번호를 한 번 더 확인해 주시겠어요?"}]
    llm = ScriptLLM(text("말씀하신 번호로는 조회가 어려워요."))
    r, tts, seen = make(llm)
    spoken = run(r, system_with(hist), [Message("user", "그 번호 맞아요")])
    assert spoken == ["말씀하신 번호로는 조회가 어려워요."]
    assert len(llm.calls) == 1


def test_claim_after_failed_registration_lookup_passes():
    """접수 함수가 not_found를 돌려주면 '주문을 찾을 수 없어요'는 결과 그대로다."""
    llm = ScriptLLM(call("cancel_order", {"order_id": "20260924-0035429", "reason": "단순 변심"}),
                    text("그 번호로는 주문을 찾을 수 없어요."))
    r, tts, seen = make(llm, {"cancel_order": {"status": "not_found", "message": "주문 없음"}})
    spoken = run(r, system_with(), [Message("user", "20260924-0035429 취소해 주세요 단순 변심이에요")])
    assert spoken == ["그 번호로는 주문을 찾을 수 없어요."]


def test_successful_registration_in_history_is_not_a_lookup():
    """공식 P0010-11: 이전 대화엔 반품 접수 성공만 있고 주문 조회는 없다 → '입금 확인이 안 된다'는 다시 생성."""
    hist = [{"role": "customer", "text": "반품해 주세요"},
            {"role": "tool_call", "name": "return_order", "arguments": {"order_number": "20260924-0035429", "reason": "수량 부족"}},
            {"role": "tool_result", "name": "return_order", "result": {"status": "ok", "message": "주문 반품 접수 성공"}},
            {"role": "agent", "text": "반품 접수가 완료되었습니다."}]
    llm = ScriptLLM(text("입금 확인이 되지 않았어요."), call("get_order_info", {"order_id": "20260924-0035429"}),
                    text("결제 완료된 주문이에요."))
    r, tts, seen = make(llm, {"get_order_info": ORDER})
    spoken = run(r, system_with(hist, caller={"phone": "01047775671"}),
                 [Message("user", "주문번호 20260924-0035429 입금 확인 안 되고 있습니다")])
    assert [n for n, _ in seen] == ["get_order_info"]
    assert spoken == ["결제 완료된 주문이에요."]


def test_normal_policy_text_untouched():
    llm = ScriptLLM(text("무통장 입금은 연동까지 20분 정도 걸려요. ", "20분이 지나도 입금 확인이 안 되면 다시 말씀해 주세요."))
    r, tts, seen = make(llm)
    spoken = run(r, system_with(), [Message("user", "방금 입금했어요")])
    assert spoken == ["무통장 입금은 연동까지 20분 정도 걸려요.", "20분이 지나도 입금 확인이 안 되면 다시 말씀해 주세요."]
    assert len(llm.calls) == 1 and seen == []


def test_repeated_lookup_claim_dropped():
    llm = ScriptLLM(text("주문 내역이 없어요."))
    r, tts, seen = make(llm)
    spoken = run(r, system_with(), [Message("user", "주문 확인해 주세요")])
    assert not any("내역이 없어요" in t for t in tts.requests)
    assert spoken == [FALLBACK_TEXT]
