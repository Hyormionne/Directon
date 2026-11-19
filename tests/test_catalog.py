"""AG-T10d 카탈로그 기준 상품명·옵션 정규화(agent/catalog.py + normalize_args 확장)."""
import json

import pytest

from agent import catalog as catalog_mod
from agent.catalog import Catalog, load_catalog
from agent.normalize import normalize_args

PRODUCTS = {
    "무선 이어폰 에어핏": ["화이트", "블랙"],
    "보조배터리 10000": ["기본"],
    "스마트워치 밴드": ["블랙", "네이비", "핑크"],
    "기모 후드티": ["S", "M", "L", "XL"],
    "데님 와이드 팬츠": ["26", "28", "30", "32"],
    "러닝화 에어런": ["240", "250", "260", "270", "280"],
    "무선 충전기 패드": ["기본"],
    "캠핑 의자": ["카키", "베이지"],
    "극세사 이불": ["싱글", "퀸"],
    "요가 매트": ["퍼플", "그레이"],
    "노트북 파우치": ["13인치", "15인치"],
    "스테인리스 텀블러 500": ["실버", "화이트"],
}

CHECK_STOCK = {"type": "object", "properties": {"product_name": {"type": "string"},
                                                 "option": {"type": "string"}}}


@pytest.fixture
def cat():
    return Catalog(PRODUCTS)


# ── 상품명 ──────────────────────────────────────────
@pytest.mark.parametrize("text, expected", [
    ("기모 후드티", "기모 후드티"),                 # 정확 일치
    ("스마트 워치 밴드", "스마트워치 밴드"),        # 공백 무시
    ("  러닝화에어런 ", "러닝화 에어런"),
    ("후드티", "기모 후드티"),                      # 부분 일치(고객 표현 ⊂ 카탈로그)
    ("이어폰", "무선 이어폰 에어핏"),
    ("무선 이어폰", "무선 이어폰 에어핏"),
    ("요가 매트를", "요가 매트"),                   # 카탈로그 ⊂ 고객 표현
    ("러닝화 에어런 이백육십", "러닝화 에어런"),
    ("보조배터리 만짜리", "보조배터리 10000"),       # 낱말 일치
    ("스테인레스 텀블러", "스테인리스 텀블러 500"),
])
def test_resolve_product(cat, text, expected):
    assert cat.resolve_product(text) == expected


@pytest.mark.parametrize("text", [
    "무선",               # 이어폰·충전기 둘 다 → 애매
    "블루투스 스피커",     # 없는 상품
    "",
    "   ",
    "가",
])
def test_resolve_product_ambiguous_or_unknown_is_none(cat, text):
    assert cat.resolve_product(text) is None


def test_resolve_product_case_insensitive():
    c = Catalog({"Air Pods Pro": ["White"]})
    assert c.resolve_product("air pods pro") == "Air Pods Pro"
    assert c.resolve_option("Air Pods Pro", "white") == "White"


# ── 옵션 ────────────────────────────────────────────
@pytest.mark.parametrize("product, text, expected", [
    ("기모 후드티", "XL", "XL"),
    ("기모 후드티", "xl", "XL"),
    ("기모 후드티", "엑스라지", "XL"),
    ("기모 후드티", "엑스엘", "XL"),
    ("기모 후드티", "엠", "M"),
    ("기모 후드티", "미디엄", "M"),
    ("기모 후드티", "M 사이즈로", "M"),
    ("기모 후드티", "엘", "L"),
    ("기모 후드티", "라지", "L"),
    ("기모 후드티", "스몰", "S"),
    ("기모 후드티", "에스", "S"),
    ("요가 매트", "회색", "그레이"),
    ("요가 매트", "그레이요", "그레이"),
    ("스마트워치 밴드", "검정", "블랙"),
    ("스마트워치 밴드", "검은색", "블랙"),
    ("스마트워치 밴드", "남색", "네이비"),
    ("스마트워치 밴드", "분홍색", "핑크"),
    ("무선 이어폰 에어핏", "흰색", "화이트"),
    ("무선 이어폰 에어핏", "하얀색", "화이트"),
    ("캠핑 의자", "베이지 색", "베이지"),
    ("스테인리스 텀블러 500", "은색", "실버"),
    ("러닝화 에어런", "이백칠십", "270"),
    ("러닝화 에어런", "이백육십 사이즈", "260"),
    ("러닝화 에어런", "270", "270"),
    ("데님 와이드 팬츠", "삼십이", "32"),
    ("노트북 파우치", "십오 인치", "15인치"),
    ("노트북 파우치", "15인치", "15인치"),
    ("노트북 파우치", "15", "15인치"),
    ("극세사 이불", "퀸 사이즈", "퀸"),
])
def test_resolve_option(cat, product, text, expected):
    assert cat.resolve_option(product, text) == expected


def test_resolve_option_reverse_alias():
    # 카탈로그가 한글 색 이름을 쓰면 반대 방향도 된다
    c = Catalog({"티셔츠": ["회색", "검정"]})
    assert c.resolve_option("티셔츠", "그레이") == "회색"
    assert c.resolve_option("티셔츠", "black") == "검정"


@pytest.mark.parametrize("product, text", [
    ("기모 후드티", "XXL"),          # 상품에 없는 옵션(별칭은 알지만 카탈로그에 없음)
    ("기모 후드티", "레드"),
    ("러닝화 에어런", "28"),         # 숫자 부분 일치 금지(280으로 바꾸지 않음)
    ("데님 와이드 팬츠", "이백칠십"),
    ("요가 매트", ""),
    ("없는 상품", "블랙"),           # 상품이 카탈로그에 없으면 옵션도 손대지 않음
])
def test_resolve_option_unknown_is_none(cat, product, text):
    assert cat.resolve_option(product, text) is None


def test_resolve_option_ambiguous_partial_is_none():
    c = Catalog({"셔츠": ["라이트 블루", "다크 블루"]})
    assert c.resolve_option("셔츠", "블루") is None
    assert c.resolve_option("셔츠", "다크블루") == "다크 블루"


def test_xxl_aliases():
    c = Catalog({"셔츠": ["L", "XL", "XXL"]})
    assert c.resolve_option("셔츠", "투엑스라지") == "XXL"
    assert c.resolve_option("셔츠", "2XL") == "XXL"
    assert c.resolve_option("셔츠", "엑스라지") == "XL"


# ── 카탈로그 로드 ───────────────────────────────────
def test_load_catalog_from_mock_data():
    c = load_catalog()  # 기본 경로: mock_api/data/mock_data.json
    assert c is not None
    assert c.resolve_product("후드티") == "기모 후드티"
    assert c.resolve_option("기모 후드티", "엑스라지") == "XL"


def test_load_catalog_formats(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"products": [{"name": "A 상품", "options": ["블랙"]}]}, ensure_ascii=False),
                 encoding="utf-8")
    assert load_catalog(p).resolve_option("A 상품", "검정") == "블랙"
    p2 = tmp_path / "b.json"
    p2.write_text(json.dumps([{"name": "B 상품", "options": [{"name": "L"}]}], ensure_ascii=False),
                  encoding="utf-8")
    assert load_catalog(p2).resolve_option("B 상품", "라지") == "L"


@pytest.mark.parametrize("content", [None, "not json", json.dumps({"x": 1}), json.dumps([1, 2]),
                                     json.dumps({"products": "abc"})])
def test_load_catalog_missing_or_bad_is_none(tmp_path, content):
    p = tmp_path / "c.json"
    if content is not None:
        p.write_text(content, encoding="utf-8")
    assert load_catalog(p) is None


def test_env_catalog_path(tmp_path, monkeypatch):
    p = tmp_path / "official.json"
    p.write_text(json.dumps({"products": [{"name": "공식 상품", "options": [{"name": "XL"}]}]},
                            ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("DIRECTON_CATALOG_PATH", str(p))
    catalog_mod.reset_default_catalog()
    try:
        out = normalize_args("check_stock", {"product_name": "공식", "option": "엑스라지"}, CHECK_STOCK)
        assert out == {"product_name": "공식 상품", "option": "XL"}
    finally:
        monkeypatch.delenv("DIRECTON_CATALOG_PATH")
        catalog_mod.reset_default_catalog()


def test_env_catalog_path_missing_is_safe(tmp_path, monkeypatch):
    monkeypatch.setenv("DIRECTON_CATALOG_PATH", str(tmp_path / "nope.json"))
    catalog_mod.reset_default_catalog()
    try:
        out = normalize_args("check_stock", {"product_name": " 후드티 ", "option": "엑스라지"}, CHECK_STOCK)
        assert out == {"product_name": "후드티", "option": "엑스라지"}
    finally:
        monkeypatch.delenv("DIRECTON_CATALOG_PATH")
        catalog_mod.reset_default_catalog()


# ── normalize_args 통합 ─────────────────────────────
def test_normalize_args_with_injected_catalog(cat):
    out = normalize_args("check_stock", {"product_name": "후드티", "option": "엑스라지"}, CHECK_STOCK,
                         "2026-11-20 10:00", catalog=cat)
    assert out == {"product_name": "기모 후드티", "option": "XL"}


def test_normalize_args_option_uses_resolved_product(cat):
    out = normalize_args("check_stock", {"product_name": "요가매트", "option": "회색"}, None, catalog=cat)
    assert out == {"product_name": "요가 매트", "option": "그레이"}


def test_normalize_args_unresolved_keeps_stripped_original(cat):
    out = normalize_args("check_stock", {"product_name": " 무선 ", "option": " 블랙 "}, CHECK_STOCK, catalog=cat)
    assert out == {"product_name": "무선", "option": "블랙"}


def test_normalize_args_other_args_untouched_by_catalog(cat):
    out = normalize_args("request_exchange", {"order_id": "이육일일 일이공 삼팔사", "reason": "엠 사이즈로 교환"},
                         None, catalog=cat)
    assert out == {"order_id": "2611120384", "reason": "엠 사이즈로 교환"}


def test_normalize_args_default_catalog_for_product_tools():
    # 응답기처럼 4인자로 부르면 기본 카탈로그(mock_data.json)를 쓴다
    out = normalize_args("check_stock", {"product_name": "스마트 워치 밴드", "option": "검정"}, CHECK_STOCK,
                         "2026-11-20 10:00")
    assert out == {"product_name": "스마트워치 밴드", "option": "블랙"}
    out = normalize_args("get_product_info", {"product_name": "후드티"}, None, "2026-11-20 10:00")
    assert out == {"product_name": "기모 후드티"}


def test_normalize_args_default_catalog_not_applied_to_unrelated_tool():
    # 기존 동작 보존: 스키마에 상품명 인자가 없는 임의 도구는 strip만
    out = normalize_args("x", {"product_name": "  무선 이어폰 "}, None, "2026-11-20 10:00")
    assert out == {"product_name": "무선 이어폰"}


def test_normalize_args_env_off(monkeypatch):
    monkeypatch.setenv("DIRECTON_CATALOG_NORMALIZE", "0")
    out = normalize_args("check_stock", {"product_name": "후드티", "option": "엑스라지"}, CHECK_STOCK)
    assert out == {"product_name": "후드티", "option": "엑스라지"}


def test_normalize_args_catalog_false_disables():
    out = normalize_args("check_stock", {"product_name": "후드티"}, CHECK_STOCK, catalog=False)
    assert out == {"product_name": "후드티"}


def test_normalize_args_does_not_mutate_input(cat):
    args = {"product_name": "후드티", "option": "엠"}
    normalize_args("check_stock", args, CHECK_STOCK, catalog=cat)
    assert args == {"product_name": "후드티", "option": "엠"}


def test_catalog_error_is_safe(monkeypatch):
    class Boom:
        def resolve_product(self, text):
            raise RuntimeError("x")

        def resolve_option(self, product, text):
            raise RuntimeError("x")

    out = normalize_args("check_stock", {"product_name": " 후드티 ", "option": "엠"}, CHECK_STOCK, catalog=Boom())
    assert out == {"product_name": "후드티", "option": "엠"}
