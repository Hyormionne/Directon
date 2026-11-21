"""AG-T39 공식 브랜드(하루온) 프롬프트: 함수 8개 규칙, 브랜드 자료(scenarios·information·style), 영업시간, now 형식.

브랜드 자료는 합성 소형 폴더(공식 자료는 저장소에 넣지 않는다).
"""
import json

import pytest

from agent import handoff_jev
from agent.config import Settings
from agent.core.types import SessionInfo
from agent.normalize import group_digits
from agent.prompts import (INFO_HEADER, SCENARIO_HEADER, VAR_MARKER, build_fixed, build_system, business_hours_note,
                           load_brand)
from agent.responder import caller_phone_from_system

OFFICIAL_TOOLS = ("get_order_info", "get_return_info", "search_products", "cancel_order", "return_order",
                  "return_order_item", "exchange_order", "request_callback")
OLD_TOOLS = ("transfer_to_agent", "get_order_list", "track_delivery", "get_member_info", "get_coupons",
             "request_return", "request_exchange", "get_refund_status", "check_stock", "get_product_info",
             "change_shipping_address")


@pytest.fixture
def brand_dir(tmp_path):
    d = tmp_path / "brand"
    d.mkdir()
    (d / "scenarios.md").write_text("# 가짜몰 시나리오\n## 1. 배송\n- 주문을 조회한다.", encoding="utf-8")
    (d / "information.json").write_text(json.dumps([
        {"tag": "[전화 문의 방법]", "content": "고객센터(☎ 02-000-0000) 평일 10:00-17:00 https://example.com/x 참고"},
        {"tag": "배송비", "content": "3만원 이상 무료"},
    ], ensure_ascii=False), encoding="utf-8")
    (d / "style.json").write_text(json.dumps({"greeting": "안녕하세요, 가짜몰입니다.", "style": "- 존댓말을 쓴다."},
                                             ensure_ascii=False), encoding="utf-8")
    return d


def sys_for(brand_dir, **kw):
    info = SessionInfo("S", now=kw.pop("now", "2026-11-04 (수) 11:00"), caller=kw.pop("caller", None),
                       history=kw.pop("history", []))
    return build_system(info, "네, 고객님.", manual="", brand_dir=brand_dir)


def test_fixed_rules_name_official_tools_only():
    f = build_fixed()
    assert "하루온" in f and "다이렉트몰" not in f
    for name in OFFICIAL_TOOLS:
        assert name in f, name
    for name in OLD_TOOLS:
        assert name not in f, name
    assert "전화 문의 방법" in f                      # 인계는 말로 + 고객센터 번호·영업 시간


def test_brand_material_goes_into_fixed_part(brand_dir):
    s = sys_for(brand_dir)
    fixed = s[:s.index(VAR_MARKER)]
    assert SCENARIO_HEADER in fixed and "주문을 조회한다" in fixed
    assert INFO_HEADER in fixed and "### 전화 문의 방법" in fixed and "3만원 이상 무료" in fixed
    assert "존댓말을 쓴다" in fixed
    assert "https://" not in fixed                       # 링크는 읽지 않는다


def test_load_brand_missing_dir_is_empty(tmp_path):
    b = load_brand(tmp_path / "none")
    assert b == {"scenarios": "", "info": "", "style": "", "greeting": ""}


def test_load_brand_greeting(brand_dir):
    assert load_brand(brand_dir)["greeting"] == "안녕하세요, 가짜몰입니다."


@pytest.mark.parametrize("now,open_", [
    ("2026-11-04 (수) 10:30", True), ("2026-11-04 (수) 09:30", False), ("2026-11-04 (수) 17:10", False),
    ("2026-11-04 (수) 12:30", False), ("2026-11-07 (토) 11:00", False), ("2026-11-04 16:59", True),
])
def test_business_hours_haruon_with_weekday_format(now, open_):
    note = business_hours_note(now)
    assert note, now                                     # "(수)"가 든 공식 형식도 읽는다
    assert ("영업 시간이다" in note) == open_
    assert handoff_jev.in_business_hours(now) is open_


def test_now_line_shows_weekday_for_official_format(brand_dir):
    var = sys_for(brand_dir, now="2026-11-04 (수) 12:39")
    var = var[var.index(VAR_MARKER):]
    assert "수요일" in var


def test_caller_line_names_official_phone_args_and_is_parsed(brand_dir):
    s = sys_for(brand_dir, caller={"name": "김가짜", "phone": "01055502714", "email": None})
    var = s[s.index(VAR_MARKER):]
    assert "buyer_cellphone" in var and "get_return_info" in var and "01055502714" in var
    assert "get_order_list" not in var
    assert caller_phone_from_system(s) == "01055502714"


def test_official_order_id_grouping():
    assert group_digits("20260925-0041152", "order") == "2026, 0925, 004, 1152"


def test_manual_default_is_empty():
    assert Settings().manual_path == ""
