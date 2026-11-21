"""scripts/intent_eval.py의 행동 분류(네트워크 없음)."""
import importlib.util
from pathlib import Path

_p = Path(__file__).resolve().parents[1] / "scripts" / "intent_eval.py"
_spec = importlib.util.spec_from_file_location("intent_eval", _p)
ie = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ie)


def test_text_classes():
    c = ie.classify_text
    assert c("주문번호나 주문하신 분 연락처를 말씀해 주시겠어요?") == "ASK_ID"
    assert c("반품·환불 현황 확인을 위해 성함과 전화번호를 알려 주세요.") == "ASK_ID"
    assert c("어떤 제품을 드시는지 알려 주시겠어요?") == "ASK_PRODUCT"
    assert c("혈관 순환 세트 제품으로 안내해 드릴까요?") == "ASK_PRODUCT"
    assert c("어떤 도움이 필요하신가요?") == "CLARIFY"
    assert c("반품 사유를 말씀해 주시겠어요?") == "ASK_DETAIL"
    assert c("상담원에게 연결해 드리겠습니다. 고객센터 번호는 1600-0000이에요.") == "HANDOFF"
    assert c("5만 원 이상 구매하시면 무료 배송이에요. 더 궁금하신 점 있으실까요?") == "ANSWER"


def test_calls_take_priority():
    calls = [{"name": "get_order_info", "arguments": {}}, {"name": "return_order", "arguments": {}}]
    assert ie.classify(calls, "", ["LOOKUP_ORDER"]) == "ACT:return_order"
    calls = [{"name": "get_order_info", "arguments": {}}, {"name": "get_return_info", "arguments": {}}]
    assert ie.classify(calls, "", ["LOOKUP_RETURN"]) == "LOOKUP_RETURN"
    assert ie.classify([], "주문번호를 알려 주세요.", ["ASK_ID"]) == "ASK_ID"
