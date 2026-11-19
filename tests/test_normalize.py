"""AG-T08 인자 정규화 테스트.

기준 시각 now = 2026-11-21(토) 11:00.
"""
import copy
from datetime import datetime

import pytest

from agent.normalize import (
    digits_for_speech,
    normalize_args,
    normalize_date,
    normalize_datetime,
    normalize_order_id,
    normalize_phone,
    normalize_time,
    spoken_digits,
)

NOW = "2026-11-21 11:00"  # 토요일


# ---------------------------------------------------------------- spoken_digits

@pytest.mark.parametrize(
    "text, expected",
    [
        ("공일공 일이삼사 오육칠팔", "01012345678"),
        ("영일영 일이삼사 오육칠팔", "01012345678"),
        ("공일공일이삼사오육칠팔", "01012345678"),
        ("010-1234-5678", "01012345678"),
        ("010 1234 5678", "01012345678"),
        ("공일공, 일이삼사, 오육칠팔", "01012345678"),
        ("공일공 일이삼사 다시 오육칠팔", "01012345678"),
        ("구팔칠육 륙", "98766"),
        ("빵일빵", "010"),
        ("이육일일 일오공 이삼일", "2611150231"),
        ("천이백삼십사", "1234"),
        ("오천육백칠십팔", "5678"),
        ("영일영 천이백삼십사 오천육백칠십팔", "01012345678"),
        ("영일영 천이백삼십사에 오천육백칠십팔", "01012345678"),
        ("010에 1234 오육칠팔", "01012345678"),
        ("제 번호는 공일공 일이삼사 오육칠팔이에요", "01012345678"),
        ("오육칠팔번이요", "5678"),
        ("십", "10"),
        ("이천이십육", "2026"),
        ("만이천", "12000"),
    ],
)
def test_spoken_digits(text, expected):
    assert spoken_digits(text) == expected


def test_spoken_digits_none_when_no_number():
    assert spoken_digits("배송 언제 와요") is None
    assert spoken_digits("") is None


def test_place_value_group_is_not_zero_padded():
    # 문서화된 규칙: 자릿값 읽기 그룹이 4자리 미만이어도 앞을 0으로 채우지 않는다.
    assert spoken_digits("공일공 이백삼십사 오천육백칠십팔") == "0102345678"


# ---------------------------------------------------------------- normalize_phone

@pytest.mark.parametrize(
    "text, expected",
    [
        ("공일공 일이삼사 오육칠팔", "01012345678"),
        ("010-1234-5678", "01012345678"),
        ("010.1234.5678", "01012345678"),
        ("영일영 천이백삼십사에 오천육백칠십팔", "01012345678"),
        ("+82 10-1234-5678", "01012345678"),
        ("공일공 일이삼", "010123"),  # 11자리가 아니어도 숫자만 남긴다
    ],
)
def test_normalize_phone(text, expected):
    assert normalize_phone(text) == expected


def test_normalize_phone_without_digits_keeps_text():
    assert normalize_phone("  모르겠어요 ") == "모르겠어요"


# ---------------------------------------------------------------- normalize_order_id

@pytest.mark.parametrize(
    "text, expected",
    [
        ("이육일일 일오공 이삼일", "2611150231"),
        ("2611-150-231", "2611150231"),
        ("디 이 엠 오", "DEMO"),
        ("디이엠오", "DEMO"),
        ("에이 비 씨", "ABC"),
        ("더블유", "W"),
        ("제트", "Z"),
        ("지", "G"),
        ("디 이 엠 오 일공공일", "DEMO1001"),
        ("에이 비 씨 일 이 삼", "ABC123"),
        ("demo-1001", "DEMO1001"),
        ("에이치 아이 제이 케이 엘 엠 엔", "HIJKLMN"),
        ("엑스 와이 브이 큐 알 에스 티 유 피 에프", "XYVQRSTUPF"),
    ],
)
def test_normalize_order_id(text, expected):
    assert normalize_order_id(text) == expected


def test_ambiguous_single_syllable_alone_is_digit():
    # 문자 문맥이 없으면 "이"·"오"는 숫자로 읽는다.
    assert normalize_order_id("이 오") == "25"


# ---------------------------------------------------------------- dates

@pytest.mark.parametrize(
    "text, expected",
    [
        ("오늘", "2026-11-21"),
        ("내일", "2026-11-22"),
        ("모레", "2026-11-23"),
        ("내일모레", "2026-11-23"),
        ("글피", "2026-11-24"),
        ("다음 주 월요일", "2026-11-23"),
        ("다음주 금요일", "2026-11-27"),
        ("이번 주 금요일", "2026-11-20"),  # 달력상 이번 주(월~일), 지난 날이어도 그대로
        ("이번 주 일요일", "2026-11-22"),
        ("월요일", "2026-11-23"),  # 요일만: 오늘 이후 가장 가까운 날
        ("토요일", "2026-11-28"),  # 오늘과 같은 요일이면 다음 주
        ("11월 25일", "2026-11-25"),
        ("십일월 이십오일", "2026-11-25"),
        ("25일", "2026-11-25"),
        ("20일", "2026-12-20"),  # 이번 달 지난 날이면 다음 달
        ("1월 3일", "2027-01-03"),  # 올해 지난 날이면 내년
        ("2026-12-01", "2026-12-01"),
        ("다음 달 5일", "2026-12-05"),
        ("3일 뒤", "2026-11-24"),
    ],
)
def test_normalize_date(text, expected):
    assert normalize_date(text, NOW) == expected


def test_normalize_date_accepts_datetime_now():
    assert normalize_date("내일", datetime(2026, 11, 21, 11, 0)) == "2026-11-22"


def test_normalize_date_unknown_returns_none():
    assert normalize_date("언제든지요", NOW) is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("오후 세 시", "15:00"),
        ("오후세시", "15:00"),
        ("세시 반", "15:30"),  # 오전/오후 없고 1~6시면 오후로 본다
        ("오전 열 시", "10:00"),
        ("열한시 삼십분", "11:30"),
        ("저녁 일곱 시", "19:00"),
        ("오후 2시 15분", "14:15"),
        ("15시", "15:00"),
        ("정오", "12:00"),
        ("오전 9시", "09:00"),
    ],
)
def test_normalize_time(text, expected):
    assert normalize_time(text) == expected


def test_normalize_time_ignores_duration():
    assert normalize_time("네 시간 걸려요") is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("내일 오후 세 시", "2026-11-22 15:00"),
        ("다음 주 월요일 오전 열 시 반", "2026-11-23 10:30"),
        ("내일", "2026-11-22"),
        ("오후 세 시", "15:00"),
    ],
)
def test_normalize_datetime(text, expected):
    assert normalize_datetime(text, NOW) == expected


# ---------------------------------------------------------------- normalize_args

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "request_callback",
        "parameters": {
            "type": "object",
            "properties": {
                "phone_number": {"type": "string"},
                "preferred_time": {"type": "string"},
            },
        },
    },
}


def test_normalize_args_phone_and_time():
    args = {"phone_number": "공일공 일이삼사 오육칠팔", "preferred_time": "내일 오후 세 시"}
    out = normalize_args("request_callback", args, TOOL_SCHEMA, NOW)
    assert out == {"phone_number": "01012345678", "preferred_time": "2026-11-22 15:00"}


def test_normalize_args_does_not_mutate_input():
    args = {"phone_number": "010-1234-5678", "extra": {"a": [1]}}
    before = copy.deepcopy(args)
    out = normalize_args("request_callback", args, TOOL_SCHEMA, NOW)
    assert args == before
    assert out["phone_number"] == "01012345678"
    assert out["extra"] == {"a": [1]}


def test_normalize_args_order_id_and_reason_strip():
    schema = {"type": "object", "properties": {"order_id": {"type": "string"}, "reason": {"type": "string"}}}
    out = normalize_args("cancel_order", {"order_id": "이육일일 일오공 이삼일", "reason": "  단순 변심 "}, schema, NOW)
    assert out == {"order_id": "2611150231", "reason": "단순 변심"}


def test_normalize_args_generic_id_suffix_and_code():
    out = normalize_args("x", {"coupon_id": "디 이 엠 오"}, None, NOW)
    assert out == {"coupon_id": "DEMO"}


def test_normalize_args_integer_schema():
    schema = {"type": "object", "properties": {"order_id": {"type": "integer"}, "quantity": {"type": "integer"}}}
    out = normalize_args("x", {"order_id": "2611-150-231", "quantity": "3"}, schema, NOW)
    assert out == {"order_id": 2611150231, "quantity": 3}


def test_normalize_args_int_phone_restores_leading_zero():
    out = normalize_args("get_member_info", {"phone_number": 1012345678}, None, NOW)
    assert out == {"phone_number": "01012345678"}


def test_normalize_args_unknown_and_non_string_kept():
    args = {"product_name": "  무선 이어폰 ", "count": 2, "flag": True, "address": "서울시 강남구 테헤란로 123"}
    out = normalize_args("x", args, None, NOW)
    assert out == {"product_name": "무선 이어폰", "count": 2, "flag": True, "address": "서울시 강남구 테헤란로 123"}


def test_normalize_args_unparsable_date_kept_stripped():
    out = normalize_args("x", {"preferred_time": " 아무 때나 "}, None, NOW)
    assert out == {"preferred_time": "아무 때나"}


def test_normalize_args_plain_parameters_schema_and_now_default():
    schema = {"name": "get_order_list", "parameters": {"type": "object", "properties": {"phone_number": {"type": "string"}}}}
    out = normalize_args("get_order_list", {"phone_number": "010 1234 5678"}, schema)
    assert out == {"phone_number": "01012345678"}


# ---------------------------------------------------------------- digits_for_speech

@pytest.mark.parametrize(
    "s, kind, expected",
    [
        ("01012345678", "phone", "공일공, 일이삼사, 오육칠팔"),
        ("0101234567", "phone", "공일공, 일이삼, 사오육칠"),
        ("2611150231", "order", "이육일일, 일오공, 이삼일"),
        ("DEMO1001", "order", "디 이 엠 오, 일공공일"),
        ("12345678", "digits", "일이삼사, 오육칠팔"),
    ],
)
def test_digits_for_speech(s, kind, expected):
    assert digits_for_speech(s, kind) == expected


def test_roundtrip_speech_to_digits():
    for s, kind in [("01098765432", "phone"), ("2611150231", "order")]:
        assert spoken_digits(digits_for_speech(s, kind)) == s
