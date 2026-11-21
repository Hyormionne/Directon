"""AG-T22 건강기능식품 연습판 생성기: tools.json, source_data.json, mock_data.json, cases.jsonl을 다시 만든다.

- source_data.json: 사람이 읽는 쇼핑몰 형식 원본(회원·상품·주문·결제·환불).
- mock_data.json: Mock 서버용 Starter Kit 매핑 형식 {함수명: [{"arguments": {...}, "result": {...}}]}.
  기존 함수의 응답은 mock_api/server.py의 쇼핑몰 규칙(Shop._fn_*)으로 계산하고, 결제 함수와 '입금대기'
  상태는 이 파일의 규칙으로 만든다. 그래서 서버 코드를 바꾸지 않고 `--data`로 그대로 띄울 수 있다.

  uv run python data/practice_health/generate.py

같은 원본 → 같은 출력(seed 고정). 문항을 고칠 때는 이 파일을 고치고 다시 실행한다.
가상 브랜드 '온결헬스'(실제 회사 아님). 데이터 기준일 2026-11-20(금).
"""
from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
ROOT = DIR.parents[1]
SEED = 20261121


def S(desc):
    return {"type": "string", "description": desc}


def tool(name, desc, props, required):
    return {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}


PHONE = S("가입한 휴대폰 번호. 010으로 시작하는 숫자 11자리, 하이픈 없이. 예: 01024681357")
ORDER = S("주문번호. 숫자 4자리. 예: 1234")
PRODUCT = S("온결헬스 상품명(정확한 이름). 예: 온결 오메가3")

TOOLS = [
    tool("get_member_info", "휴대폰 번호로 회원 정보(이름, 등급, 적립금, 기본 배송지)를 조회한다.",
         {"phone_number": PHONE}, ["phone_number"]),
    tool("get_order_list", "휴대폰 번호로 그 회원의 주문 목록(주문번호, 주문일, 상태, 상품, 금액)을 최신순으로 조회한다. "
         "고객이 주문번호를 모를 때 쓴다.", {"phone_number": PHONE}, ["phone_number"]),
    tool("get_order_info", "주문번호로 주문 상세(상태, 상품, 금액, 결제 수단, 배송지, 출고 예정일)를 조회한다.",
         {"order_id": ORDER}, ["order_id"]),
    tool("track_delivery", "주문번호로 배송 현황(택배사, 현재 위치, 도착 예정일 또는 출고 예정일)을 조회한다. "
         "'언제 와요?' 같은 배송 문의에 우선 쓴다.", {"order_id": ORDER}, ["order_id"]),
    tool("change_shipping_address", "주문의 배송지를 바꾼다. 결제완료·상품준비중 상태에서만 가능하다.",
         {"order_id": ORDER, "new_address": S("새 배송지 주소(고객이 말한 그대로).")}, ["order_id", "new_address"]),
    tool("cancel_order", "주문을 취소한다. 고객이 취소를 분명히 요청했을 때만 부른다. 출고 전(결제완료·상품준비중)은 즉시 취소, "
         "배송준비는 취소 요청 접수, 배송중·배송완료는 불가.",
         {"order_id": ORDER, "reason": S("취소 사유(자유 텍스트, 선택).")}, ["order_id"]),
    tool("request_return", "배송완료 주문의 반품(환불)을 접수한다. 수령 후 7일 이내만 가능하다. 고객이 반품을 분명히 요청했을 때만 부른다.",
         {"order_id": ORDER, "reason": S("반품 사유(예: 단순변심 미개봉, 파손, 오배송).")}, ["order_id"]),
    tool("request_exchange", "배송완료 주문의 교환을 접수한다(파손·오배송·맛 변경 등). 수령 후 7일 이내만 가능하다.",
         {"order_id": ORDER, "reason": S("교환 사유(예: 다른 맛이 옴, 용기 파손).")}, ["order_id"]),
    tool("get_refund_status", "주문번호로 환불 진행 상태(환불 금액, 환불 수단, 예정일 또는 완료일)를 조회한다.",
         {"order_id": ORDER}, ["order_id"]),
    tool("get_payment_info", "주문번호로 결제 정보를 조회한다: 결제 수단, 결제 금액, 결제 상태(결제완료/입금대기/환불완료), "
         "카드사·할부, 무통장입금이면 입금 은행·계좌번호·예금주·입금자명·입금 기한.",
         {"order_id": ORDER}, ["order_id"]),
    tool("check_deposit", "무통장입금 주문의 입금 확인 여부를 조회한다(입금 확인 시각 또는 미확인·입금 기한). "
         "무통장입금 주문이 아니면 not_allowed.", {"order_id": ORDER}, ["order_id"]),
    tool("get_product_info", "상품명으로 상품 정보를 조회한다: 가격, 용량, 주요 성분, 섭취 방법, 섭취 시 주의사항, 보관 방법, 옵션(맛·구성).",
         {"product_name": PRODUCT}, ["product_name"]),
    tool("check_stock", "상품(과 옵션)의 재고를 조회한다. 품절이면 재입고 예정일을 준다.",
         {"product_name": PRODUCT, "option": S("옵션명(맛·구성). 예: 복숭아. 생략하면 전체 옵션.")}, ["product_name"]),
    tool("request_callback", "상담원 콜백(전화 회신)을 접수한다. 영업시간 외에 상담원을 원하거나 고객이 나중에 전화를 원할 때.",
         {"phone_number": PHONE, "preferred_time": S("고객이 원하는 연락 시간(자유 텍스트, 선택).")}, ["phone_number"]),
    tool("transfer_to_agent", "상담원에게 연결한다. 고객의 직접 요청, 제품 섭취 후 몸이 불편하다는 문의(이상 반응)·질환·약 복용 등 "
         "의학적 판단이 필요한 질문, 강한 불만·욕설, 반복 이해 실패, 매뉴얼로 처리할 수 없는 예외일 때.",
         {"reason": S("인계 사유 요약. 예: 제품 섭취 후 두드러기 이상 반응 문의.")}, ["reason"]),
]

# ───────────────────────── 상품 ─────────────────────────

def product(name, price, category, volume, ingredients, how, cautions, storage, options, summary):
    return {"name": name, "price": price, "category": category, "volume": volume,
            "description": f"{summary} 섭취 방법: {how} 주의: {cautions}",
            "ingredients": ingredients, "how_to_take": how, "cautions": cautions, "storage": storage,
            "options": options}


def opt(name, stock, restock=None):
    d = {"name": name, "stock": stock}
    if restock:
        d["restock_date"] = restock
    return d


COMMON_CAUTION = "특이체질·알레르기가 있거나 임산부·수유부, 질환이 있거나 의약품을 복용 중이면 섭취 전 의사·약사와 상담하세요."
PRODUCTS = [
    product("온결 멀티비타민", 19800, "비타민", "60정(1개월분)", "비타민 A·B군·C·D·E, 아연, 셀레늄",
            "하루 1회 2정을 물과 함께 식후에 섭취하세요.", COMMON_CAUTION, "직사광선을 피해 실온 보관",
            [opt("기본", 150)], "하루 한 번 챙기는 종합 비타민."),
    product("온결 오메가3", 32000, "오메가3", "60캡슐(2개월분)", "rTG 오메가3(EPA·DHA 합 600mg)",
            "하루 1회 1캡슐을 식후에 섭취하세요.",
            "혈액응고 억제제(항응고제)를 복용 중이면 섭취 전 의사와 상담하세요. " + COMMON_CAUTION,
            "개봉 후 서늘한 곳 보관", [opt("기본", 88)], "rTG 형태 오메가3."),
    product("온결 유산균", 29000, "프로바이오틱스", "30포(1개월분)", "프로바이오틱스 19종 100억 CFU, 프리바이오틱스",
            "하루 1회 1포를 아침 공복 또는 식후에 물 없이 또는 물과 함께 섭취하세요.",
            "섭취 초기에 일시적으로 복부 팽만감이 있을 수 있습니다. " + COMMON_CAUTION,
            "실온 보관 가능, 냉장 보관 권장", [opt("기본", 64)], "장 건강 유산균."),
    product("온결 비타민디", 12900, "비타민", "90정(3개월분)", "비타민D3 2000IU",
            "하루 1회 1정을 식후에 섭취하세요.", COMMON_CAUTION, "실온 보관", [opt("기본", 210)],
            "뼈 건강에 필요한 비타민D."),
    product("온결 루테인", 27500, "눈 건강", "30캡슐(1개월분)", "마리골드꽃추출물(루테인 20mg), 지아잔틴",
            "하루 1회 1캡슐을 식후에 섭취하세요.",
            "흡연자는 고함량 베타카로틴 섭취에 주의하세요. " + COMMON_CAUTION, "실온 보관", [opt("기본", 41)],
            "눈 건강 루테인."),
    product("온결 밀크씨슬", 24000, "간 건강", "60정(2개월분)", "밀크씨슬추출물(실리마린 130mg)",
            "하루 1회 1정을 식후에 섭취하세요.", COMMON_CAUTION, "실온 보관", [opt("기본", 73)], "간 건강 밀크씨슬."),
    product("온결 홍삼스틱", 45000, "홍삼", "10ml×30포(1개월분)", "6년근 홍삼농축액(진세노사이드 Rg1+Rb1+Rg3 합 15mg)",
            "하루 1회 1포를 그대로 섭취하세요.",
            "고혈압·당뇨 약이나 혈액응고 억제제를 복용 중이면 섭취 전 의사와 상담하세요. " + COMMON_CAUTION,
            "실온 보관", [opt("기본", 55)], "6년근 홍삼 스틱."),
    product("온결 콜라겐 젤리", 34000, "이너뷰티", "20g×30포(1개월분)", "저분자 피쉬콜라겐 펩타이드 1000mg, 비타민C",
            "하루 1회 1포를 그대로 섭취하세요.",
            "생선(어류) 알레르기가 있으면 섭취하지 마세요. " + COMMON_CAUTION, "실온 보관",
            [opt("복숭아", 37), opt("청포도", 0, "2026-11-27")], "짜 먹는 콜라겐 젤리."),
    product("온결 마그네슘", 15900, "미네랄", "60정(2개월분)", "산화마그네슘(마그네슘 250mg)",
            "하루 1회 1정을 식후에 섭취하세요.",
            "과다 섭취 시 묽은 변이 나올 수 있습니다. " + COMMON_CAUTION, "실온 보관", [opt("기본", 96)],
            "근육·신경 기능에 필요한 마그네슘."),
    product("온결 식물성 프로틴", 39000, "단백질", "750g(약 25회분)", "완두·현미 단백질(1회 단백질 20g)",
            "1회 1스쿱(30g)을 물이나 두유 200ml에 타서 하루 1~2회 섭취하세요.",
            "콩·대두 알레르기 성분과 같은 시설에서 제조합니다. " + COMMON_CAUTION, "개봉 후 밀봉해 서늘한 곳 보관",
            [opt("초코", 22), opt("곡물", 3)], "식물성 단백질 쉐이크."),
    product("온결 칼마디", 16500, "미네랄", "90정(1개월분)", "칼슘 700mg, 마그네슘 250mg, 비타민D 800IU",
            "하루 1회 3정을 식후에 섭취하세요.", COMMON_CAUTION, "실온 보관", [opt("기본", 120)],
            "칼슘·마그네슘·비타민D 복합."),
    product("온결 비타민씨", 13900, "비타민", "120정(4개월분)", "비타민C 1000mg",
            "하루 1회 1정을 식후에 섭취하세요.",
            "위가 약하면 공복 섭취를 피하세요. " + COMMON_CAUTION, "실온 보관", [opt("기본", 180)], "고함량 비타민C."),
    product("온결 코큐텐", 28000, "항산화", "60캡슐(2개월분)", "코엔자임Q10 100mg",
            "하루 1회 1캡슐을 식후에 섭취하세요.",
            "혈압약·혈액응고 억제제를 복용 중이면 섭취 전 의사와 상담하세요. " + COMMON_CAUTION, "실온 보관",
            [opt("기본", 0, "2026-11-25")], "항산화 코엔자임Q10."),
    product("온결 키즈 멀티비타민", 22000, "어린이", "60구미(1개월분)", "비타민 A·C·D·E, 아연(어린이 함량)",
            "만 3세 이상 어린이는 하루 2개를 씹어서 드세요.",
            "만 3세 미만은 섭취하지 마세요. 어린이 손이 닿지 않는 곳에 보관하고 정해진 양만 드세요. " + COMMON_CAUTION,
            "서늘한 곳 보관", [opt("딸기", 45), opt("포도", 12)], "어린이용 젤리형 비타민."),
    product("온결 철분 엽산", 18500, "여성 건강", "60정(2개월분)", "철분 12mg, 엽산 400μg, 비타민C",
            "하루 1회 1정을 식후에 섭취하세요. 공복에 섭취하면 속이 불편할 수 있습니다.",
            "변비·속쓰림이 생길 수 있습니다. 임산부는 섭취 전 의사와 상담하세요. " + COMMON_CAUTION, "실온 보관",
            [opt("기본", 67)], "임신 준비·여성용 철분 엽산."),
]
PRICE = {p["name"]: p["price"] for p in PRODUCTS}

# ───────────────────────── 회원 ─────────────────────────
MEMBERS = [
    ("01024681357", "김서윤", "골드", 8200, "서울특별시 강서구 공항대로 247, 1102호", "2022-04-11"),
    ("01035792468", "이도윤", "일반", 500, "경기도 고양시 덕양구 화중로 104, 503호", "2025-09-02"),
    ("01046813579", "박하린", "실버", 3100, "인천광역시 남동구 예술로 152, 804호", "2023-12-19"),
    ("01057924680", "최지호", "VIP", 27400, "부산광역시 해운대구 마린시티2로 38, 2203호", "2019-06-27"),
    ("01068035791", "정민서", "일반", 0, "대전광역시 서구 둔산로 100, 1407호", "2026-10-15"),
    ("01079146802", "강태오", "골드", 6600, "경기도 성남시 수정구 산성대로 301, 205호", "2021-01-30"),
    ("01080257913", "윤채원", "실버", 2300, "광주광역시 북구 첨단과기로 123, 901호", "2024-05-08"),
    ("01091368024", "한유준", "일반", 1200, "서울특별시 동작구 상도로 55, 302호", "2025-03-21"),
]
ADDR = {m[0]: m[4] for m in MEMBERS}
NAME = {m[0]: m[1] for m in MEMBERS}

# ───────────────────────── 주문 ─────────────────────────
# (order_id, phone, ordered_at, status, [(product, option, qty)], method, extra)
ORDERS_SRC = [
    ("1234", "01024681357", "2026-11-17", "배송중", [("온결 멀티비타민", "기본", 1)], "신용카드",
     dict(courier="한빛택배", tracking_number="640012340017", shipped_at="2026-11-19",
          current_location="대전 허브터미널", estimated_delivery="2026-11-21")),
    ("1187", "01024681357", "2026-11-14", "배송완료", [("온결 오메가3", "기본", 2)], "신용카드",
     dict(courier="한빛택배", tracking_number="640011870014", shipped_at="2026-11-15",
          delivered_at="2026-11-17", return_deadline="2026-11-24")),
    ("1302", "01024681357", "2026-11-20", "결제완료", [("온결 홍삼스틱", "기본", 1)], "카카오페이",
     dict(expected_ship_date="2026-11-23")),
    ("1095", "01024681357", "2026-10-29", "배송완료", [("온결 유산균", "기본", 1)], "신용카드",
     dict(courier="다온로지스", tracking_number="640010950029", shipped_at="2026-10-30",
          delivered_at="2026-11-01", return_deadline="2026-11-08", return_window_expired=True)),
    ("1311", "01035792468", "2026-11-19", "입금대기", [("온결 식물성 프로틴", "초코", 1)], "무통장입금",
     dict(expected_ship_date="입금 확인 후 1~3영업일")),
    ("1256", "01035792468", "2026-11-15", "배송완료", [("온결 콜라겐 젤리", "복숭아", 2)], "무통장입금",
     dict(courier="한빛택배", tracking_number="640012560015", shipped_at="2026-11-17",
          delivered_at="2026-11-18", return_deadline="2026-11-25")),
    ("1150", "01035792468", "2026-11-09", "취소완료", [("온결 비타민씨", "기본", 1)], "신용카드",
     dict(cancelled_at="2026-11-09")),
    ("1320", "01046813579", "2026-11-19", "상품준비중", [("온결 루테인", "기본", 1)], "신용카드",
     dict(expected_ship_date="2026-11-21")),
    ("1278", "01046813579", "2026-11-18", "배송준비", [("온결 밀크씨슬", "기본", 2)], "네이버페이",
     dict(expected_ship_date="2026-11-20")),
    ("1203", "01046813579", "2026-11-10", "반품접수", [("온결 코큐텐", "기본", 1)], "신용카드",
     dict(courier="다온로지스", tracking_number="640012030010", shipped_at="2026-11-11",
          delivered_at="2026-11-13", return_deadline="2026-11-20", requested_at="2026-11-18")),
    ("1329", "01057924680", "2026-11-20", "입금대기", [("온결 홍삼스틱", "기본", 2)], "무통장입금",
     dict(expected_ship_date="입금 확인 후 1~3영업일")),
    ("1245", "01057924680", "2026-11-16", "배송중", [("온결 키즈 멀티비타민", "딸기", 1)], "신용카드",
     dict(courier="다온로지스", tracking_number="640012450016", shipped_at="2026-11-19",
          current_location="부산 해운대 대리점", estimated_delivery="2026-11-20")),
    ("1168", "01057924680", "2026-11-10", "교환접수", [("온결 콜라겐 젤리", "청포도", 1)], "신용카드",
     dict(courier="한빛택배", tracking_number="640011680010", shipped_at="2026-11-11",
          delivered_at="2026-11-12", return_deadline="2026-11-19", requested_at="2026-11-16")),
    ("1112", "01057924680", "2026-11-03", "배송완료", [("온결 철분 엽산", "기본", 1)], "무통장입금",
     dict(courier="한빛택배", tracking_number="640011120003", shipped_at="2026-11-04",
          delivered_at="2026-11-05", return_deadline="2026-11-12", return_window_expired=True)),
    ("1335", "01068035791", "2026-11-19", "결제완료", [("온결 비타민디", "기본", 2)], "무통장입금",
     dict(expected_ship_date="2026-11-23")),
    ("1219", "01068035791", "2026-11-13", "배송완료", [("온결 마그네슘", "기본", 1)], "카카오페이",
     dict(courier="다온로지스", tracking_number="640012190013", shipped_at="2026-11-14",
          delivered_at="2026-11-16", return_deadline="2026-11-23")),
    ("1288", "01079146802", "2026-11-17", "배송중", [("온결 식물성 프로틴", "곡물", 2)], "신용카드",
     dict(courier="한빛택배", tracking_number="640012880017", shipped_at="2026-11-19",
          current_location="경기 광주 허브터미널", estimated_delivery="2026-11-21")),
    ("1176", "01079146802", "2026-11-12", "배송완료", [("온결 칼마디", "기본", 1)], "신용카드",
     dict(courier="다온로지스", tracking_number="640011760012", shipped_at="2026-11-13",
          delivered_at="2026-11-15", return_deadline="2026-11-22")),
    ("1340", "01079146802", "2026-11-17", "입금대기", [("온결 오메가3", "기본", 1)], "무통장입금",
     dict(expected_ship_date="입금 확인 후 1~3영업일")),
    ("1262", "01080257913", "2026-11-15", "배송완료", [("온결 콜라겐 젤리", "복숭아", 1)], "신용카드",
     dict(courier="한빛택배", tracking_number="640012620015", shipped_at="2026-11-17",
          delivered_at="2026-11-18", return_deadline="2026-11-25")),
    ("1137", "01080257913", "2026-11-08", "취소완료", [("온결 루테인", "기본", 1)], "무통장입금",
     dict(cancelled_at="2026-11-10")),
    ("1299", "01091368024", "2026-11-19", "상품준비중", [("온결 멀티비타민", "기본", 3)], "신용카드",
     dict(expected_ship_date="2026-11-21")),
    ("1224", "01091368024", "2026-11-14", "배송완료", [("온결 유산균", "기본", 1)], "네이버페이",
     dict(courier="다온로지스", tracking_number="640012240014", shipped_at="2026-11-15",
          delivered_at="2026-11-17", return_deadline="2026-11-24")),
]

BANK = {"bank": "새온은행", "account_number": "301-0715-2026-11", "account_holder": "(주)온결헬스"}
# 무통장입금: order_id -> (deposit_deadline, deposited_at|None)
DEPOSITS = {"1311": ("2026-11-22 23:59", None), "1256": ("2026-11-18 23:59", "2026-11-15 14:12"),
            "1329": ("2026-11-23 23:59", None), "1112": ("2026-11-06 23:59", "2026-11-03 19:40"),
            "1335": ("2026-11-22 23:59", "2026-11-20 10:05"), "1340": ("2026-11-20 23:59", None),
            "1137": ("2026-11-11 23:59", "2026-11-08 21:30")}
CARD = {"1234": "한빛카드", "1187": "새온카드", "1095": "한빛카드", "1150": "누리카드", "1320": "새온카드",
        "1203": "누리카드", "1245": "한빛카드", "1168": "새온카드", "1288": "누리카드", "1176": "한빛카드",
        "1262": "새온카드", "1299": "누리카드"}
INSTALLMENT = {"1288": "3개월 무이자"}


def build_orders():
    orders, payments = [], []
    for oid, phone, at, status, items, method, extra in ORDERS_SRC:
        its = [{"product_name": p, "option": o, "quantity": q, "unit_price": PRICE[p]} for p, o, q in items]
        amount = sum(i["unit_price"] * i["quantity"] for i in its)
        fee = 0 if amount >= 50000 else 3000
        o = {"order_id": oid, "phone_number": phone, "ordered_at": at, "status": status, "items": its,
             "product_amount": amount, "shipping_fee": fee, "total_amount": amount + fee,
             "payment_method": method, "shipping_address": ADDR[phone], **extra}
        orders.append(o)
        p = {"order_id": oid, "method": method, "amount": o["total_amount"]}
        if status == "취소완료":
            p["status"] = "환불완료"
        elif status == "입금대기":
            p["status"] = "입금대기"
        else:
            p["status"] = "결제완료"
        if method == "무통장입금":
            deadline, dep = DEPOSITS[oid]
            p.update(BANK, depositor_name=NAME[phone], deposit_deadline=deadline,
                     deposit_confirmed=dep is not None)
            if dep:
                p["deposited_at"] = dep
                p["paid_at"] = dep
        elif method == "신용카드":
            p.update(card_company=CARD[oid], installment=INSTALLMENT.get(oid, "일시불"), paid_at=f"{at} 10:00")
        else:
            p["paid_at"] = f"{at} 10:00"
        payments.append(p)
    return orders, payments


REFUNDS = [
    {"order_id": "1150", "refund_status": "환불완료", "refund_amount": 16900, "refund_method": "신용카드",
     "expected_or_completed_date": "2026-11-12", "note": "주문 취소 전액 환불"},
    {"order_id": "1137", "refund_status": "환불완료", "refund_amount": 30500, "refund_method": "계좌 환불(무통장입금)",
     "expected_or_completed_date": "2026-11-11", "note": "주문 시 등록한 환불 계좌로 입금"},
    {"order_id": "1203", "refund_status": "환불예정", "refund_amount": 22000, "refund_method": "신용카드",
     "expected_or_completed_date": "2026-11-25",
     "note": "단순변심 반품: 상품 금액 28,000원에서 왕복 배송비 6,000원 차감. 상품 회수 확인 후 카드 취소"},
]


def build_data():
    orders, payments = build_orders()
    return {
        "shop": {"name": "온결헬스", "kind": "건강기능식품 쇼핑몰(가상)", "reference_date": "2026-11-20",
                 "free_shipping_threshold": 50000, "base_shipping_fee": 3000,
                 "return_shipping_fee_change_of_mind": 6000, "deposit_window_days": 3,
                 "payment_methods": ["신용카드", "무통장입금", "카카오페이", "네이버페이"],
                 "business_hours": "평일 09:00~18:00(점심 12:00~13:00), 주말·공휴일 휴무"},
        "members": [{"phone_number": p, "name": n, "grade": g, "points": pt, "address": a, "joined_at": j}
                    for p, n, g, pt, a, j in MEMBERS],
        "products": PRODUCTS,
        "orders": orders,
        "payments": payments,
        "refunds": REFUNDS,
        "coupons": {},
    }


# ───────────────────────── 시나리오 ─────────────────────────
# s: 발화(str 또는 [(text, pause_ms), ...]), who: caller 번호, h: history, x: expected_tools
def C(task, cat, s, x=(), p=(), n=(), t=(), who=None, h=(), now=None, style=None):
    return dict(task=task, cat=cat, s=s, x=list(x), p=list(p), n=list(n), t=list(t), who=who, h=list(h),
                now=now, style=style)


def call(name, **args):
    return {"name": name, "arguments": args}


SEOYUN, DOYUN, JIHO, MINSEO, TAEO, CHAEWON = ("01024681357", "01035792468", "01057924680", "01068035791",
                                               "01079146802", "01080257913")

SCENARIOS = [
    # ── 배송 (6)
    C("배송", "tool_lookup", "주문번호 천이백삼십사번인데요, 배송 언제 와요?", [call("track_delivery", order_id="1234")],
      ["배송중이고 대전 허브터미널에 있다고 안내한다", "11월 21일 도착 예정이라고 안내한다"], t=["digits"]),
    C("배송", "tool_lookup", "애들 먹는 비타민 젤리 시킨 거 오늘 오나요?",
      [call("get_order_list", phone_number=JIHO), call("track_delivery", order_id="1245")],
      ["다온로지스로 배송중이라고 안내한다", "오늘(11월 20일) 도착 예정이라고 안내한다"],
      ["전화번호를 다시 묻는다"], who=JIHO),
    C("배송", "tool_lookup", [("주문한 홍삼스틱이", 700), ("아직 출발 안 했나요? 주문번호는 일삼공이예요.", 0)],
      [call("track_delivery", order_id="1302")],
      ["아직 출고 전이라고 안내한다", "11월 23일 출고 예정이라고 안내한다", "출고되면 송장번호를 문자로 보낸다고 안내한다"],
      t=["digits", "mid_pause", "connective_end"]),
    C("배송", "tool_action", "주문번호 일삼이공 배송지를 회사로 바꾸고 싶어요. 서울 중구 세종대로 백십이요.",
      [call("change_shipping_address", order_id="1320", new_address="서울 중구 세종대로 110")],
      ["배송지가 서울 중구 세종대로 110으로 변경되었다고 안내한다"], t=["digits"]),
    C("배송", "tool_action", "일이칠팔 주문 주소 좀 바꿔 주세요. 경기도 수원시 팔달구 효원로 이백구십구로요.",
      [call("change_shipping_address", order_id="1278", new_address="경기도 수원시 팔달구 효원로 299")],
      ["이미 출고 준비가 끝나 배송지를 바꿀 수 없다고 안내한다", "택배사에 직접 요청하도록 안내한다"],
      ["배송지가 변경되었다고 말한다"], t=["digits"]),
    C("배송", "faq", "배송비 얼마야? 얼마 이상 사면 공짜야?", [],
      ["5만 원 이상 주문하면 무료배송이라고 안내한다", "5만 원 미만은 배송비 3,000원이라고 안내한다"],
      ["반말로 응대한다"], t=["banmal", "short"]),
    # ── 주문 취소 (6)
    C("주문취소", "tool_action", "주문번호 일삼공이 취소해 주세요. 마음이 바뀌었어요.",
      [call("cancel_order", order_id="1302")],
      ["주문이 취소되었다고 안내한다", "48,000원이 카카오페이로 전액 환불된다고 안내한다"], t=["digits"]),
    C("주문취소", "tool_action", "멀티비타민 주문한 거 취소하고 싶어요.",
      [call("get_order_list", phone_number=SEOYUN), call("cancel_order", order_id="1234")],
      ["이미 배송이 시작되어 취소할 수 없다고 안내한다", "받은 뒤 미개봉 상태로 7일 이내 반품할 수 있다고 안내한다"],
      ["주문이 취소되었다고 말한다"], who=SEOYUN),
    C("주문취소", "tool_action", "밀크씨슬 주문 일이칠팔, 그거 취소해 주세요.",
      [call("cancel_order", order_id="1278")],
      ["출고 직전이라 취소 요청으로 접수되었다고 안내한다", "1영업일 안에 취소 확정 문자를 보낸다고 안내한다"],
      t=["digits"]),
    C("주문취소", "faq", [("취소할까 고민 중인데요,", 600), ("아직 결정은 안 했어요. 취소하면 바로 환불돼요?", 0)], [],
      ["출고 전이면 바로 취소되고 결제 수단으로 전액 환불된다고 안내한다", "카드는 3~5영업일, 그 외는 1~3영업일 걸린다고 안내한다"],
      ["고객 확인 없이 취소를 실행한다"], t=["negation", "mid_pause", "connective_end"]),
    C("주문취소", "need_info", "어… 주문 취소하려고 전화했어요.", [],
      ["주문번호나 가입한 휴대폰 번호를 요청한다"], ["주문번호 확인 없이 취소되었다고 말한다"], t=["filler_words", "short"]),
    C("주문취소", "followup", "일이구구요.", [call("cancel_order", order_id="1299")],
      ["주문이 취소되었다고 안내한다", "59,400원이 신용카드로 환불되고 3~5영업일 걸린다고 안내한다"],
      t=["digits", "short"],
      h=[{"role": "customer", "text": "주문 취소 좀 하려고요."},
         {"role": "agent", "text": "네, 고객님. 주문번호를 말씀해 주시겠어요?"}]),
    # ── 반품·환불 (6)
    C("반품환불", "tool_action", "오메가쓰리 받았는데 아직 안 뜯었거든요. 단순 변심인데 반품할게요.",
      [call("get_order_list", phone_number=SEOYUN), call("request_return", order_id="1187")],
      ["반품이 접수되었다고 안내한다", "단순변심이라 왕복 배송비 6,000원이 환불 금액에서 차감된다고 안내한다",
       "1~2영업일 안에 회수 기사가 방문한다고 안내한다"], who=SEOYUN),
    C("반품환불", "tool_action", "일공구오 주문한 유산균 반품하고 싶어요.",
      [call("request_return", order_id="1095")],
      ["수령 후 7일이 지나 반품 접수가 안 된다고 안내한다", "반품 기한이 11월 8일이었다고 안내한다"],
      ["반품이 접수되었다고 말한다"], t=["digits"]),
    C("반품환불", "tool_lookup", "주문번호 일일오공 환불 들어왔는지 궁금해서요.",
      [call("get_refund_status", order_id="1150")],
      ["16,900원이 11월 12일 신용카드로 환불 완료되었다고 안내한다"], t=["digits"]),
    C("반품환불", "tool_lookup", "일이공삼 반품한 거요, 환불은 언제 돼요?",
      [call("get_refund_status", order_id="1203")],
      ["22,000원이 11월 25일 환불 예정이라고 안내한다", "왕복 배송비 6,000원이 차감되었다고 안내한다",
       "상품 회수 확인 후 환불된다고 안내한다"], t=["digits"]),
    C("반품환불", "tool_lookup", "취소한 루테인 환불금 계좌로 들어왔나요?",
      [call("get_order_list", phone_number=CHAEWON), call("get_refund_status", order_id="1137")],
      ["30,500원이 11월 11일 환불 계좌로 환불 완료되었다고 안내한다"], who=CHAEWON),
    C("반품환불", "faq", "뜯어서 몇 번 먹은 것도 반품 돼요? 그냥 입에 안 맞아서요.", [],
      ["단순변심 반품은 미개봉 상품만 가능하다고 안내한다", "개봉 후에는 제품 불량·변질·오배송일 때만 가능하다고 안내한다"],
      ["개봉한 상품도 단순변심으로 반품된다고 말한다"]),
    # ── 교환 (4)
    C("교환", "tool_action", "주문번호 일이육이요. 콜라겐 젤리 복숭아 맛 시켰는데 청포도 맛이 왔어요. 교환해 주세요.",
      [call("request_exchange", order_id="1262")],
      ["교환이 접수되었다고 안내한다", "오배송이라 배송비는 무료라고 안내한다", "1~2영업일 안에 회수 기사가 방문한다고 안내한다"],
      ["교환 배송비를 고객이 낸다고 말한다"], t=["digits", "long"]),
    C("교환", "tool_action", [("마그네슘 받았는데요,", 700), ("통이 깨져서 왔어요. 새 걸로 바꿔 주세요.", 0)],
      [call("get_order_list", phone_number=MINSEO), call("request_exchange", order_id="1219")],
      ["교환이 접수되었다고 안내한다", "파손이라 배송비는 무료라고 안내한다"], who=MINSEO,
      t=["mid_pause", "connective_end"]),
    C("교환", "tool_action", "주문번호 일일칠… 아니 일일칠육이요. 칼마디 교환하려고요. 포장이 뜯겨서 왔어요.",
      [call("request_exchange", order_id="1176")],
      ["교환이 접수되었다고 안내한다", "회수 확인 후 새 상품을 보내 준다고 안내한다"], t=["digits", "self_correction"]),
    C("교환", "tool_action", "일일일이 주문 철분제 다른 걸로 교환돼요? 아직 안 먹었어요.",
      [call("request_exchange", order_id="1112")],
      ["수령 후 7일이 지나 교환 접수가 안 된다고 안내한다", "교환 기한이 11월 12일이었다고 안내한다"],
      ["교환이 접수되었다고 말한다"], t=["digits"]),
    # ── 결제 (7)
    C("결제", "tool_lookup", "주문번호 일삼일일 입금 계좌 좀 다시 알려 주세요.",
      [call("get_payment_info", order_id="1311")],
      ["새온은행 계좌와 예금주 (주)온결헬스를 안내한다", "입금할 금액이 42,000원이라고 안내한다",
       "11월 22일까지 입금해야 한다고 안내한다"], t=["digits"]),
    C("결제", "tool_lookup", "아까 무통장으로 입금했는데 확인됐어요?",
      [call("get_order_list", phone_number=MINSEO), call("check_deposit", order_id="1335")],
      ["오늘 오전 10시 5분에 입금이 확인되었다고 안내한다", "11월 23일 출고 예정이라고 안내한다"],
      ["전화번호를 다시 묻는다"], who=MINSEO, t=["short"], now="2026-11-20 14:05"),
    C("결제", "tool_lookup", [("일삼이구 주문이요,", 600), ("입금했는데 아직 확인이 안 됐다고 나와요.", 0)],
      [call("check_deposit", order_id="1329")],
      ["아직 입금이 확인되지 않았다고 안내한다", "입금자명이 주문자명(최지호)과 같은지 확인을 요청한다",
       "입금 기한이 11월 23일이라고 안내한다"],
      ["입금이 확인되었다고 말한다"], t=["digits", "mid_pause"]),
    C("결제", "tool_lookup", "일이오육 주문 결제를 뭘로 했는지 기억이 안 나요.",
      [call("get_payment_info", order_id="1256")],
      ["무통장입금으로 68,000원 결제되었다고 안내한다"], t=["digits"]),
    C("결제", "faq", "결제는 뭐로 할 수 있어요? 무통장입금도 돼요?", [],
      ["신용카드, 무통장입금, 카카오페이, 네이버페이를 쓸 수 있다고 안내한다",
       "무통장입금은 주문 후 3일 안에 입금해야 하고 기한이 지나면 자동 취소된다고 안내한다"]),
    C("결제", "tool_lookup", "오메가쓰리 무통장으로 주문했는데 언제까지 넣어야 돼요?",
      [call("get_order_list", phone_number=TAEO), call("get_payment_info", order_id="1340")],
      ["오늘(11월 20일) 밤 11시 59분까지 입금해야 한다고 안내한다", "입금할 금액이 35,000원이라고 안내한다"],
      who=TAEO),
    C("결제", "followup", "일삼삼오요.", [call("check_deposit", order_id="1335")],
      ["입금이 확인되었다고 안내한다", "11월 23일 출고 예정이라고 안내한다"], t=["digits", "short"],
      now="2026-11-20 15:10",
      h=[{"role": "customer", "text": "입금 확인 좀 해 주세요."},
         {"role": "agent", "text": "네, 고객님. 확인을 위해 주문번호를 말씀해 주시겠어요?"}]),
    # ── 상품 (6)
    C("상품", "tool_lookup", "온결 오메가쓰리는 하루에 몇 알 먹어요?",
      [call("get_product_info", product_name="온결 오메가3")],
      ["하루 한 번 1캡슐을 식후에 섭취한다고 안내한다"], ["상품 정보에 없는 효능을 말한다"], t=["short"]),
    C("상품", "tool_lookup", "홍삼스틱이요, 고혈압 약 먹는 사람도 먹어도 돼요?",
      [call("get_product_info", product_name="온결 홍삼스틱")],
      ["고혈압 약을 복용 중이면 섭취 전 의사와 상담하라는 주의사항을 안내한다"],
      ["먹어도 괜찮다고 단정한다", "의학적 판단을 내린다"]),
    C("상품", "tool_lookup", "콜라겐 젤리 청포도 맛 지금 살 수 있어요?",
      [call("check_stock", product_name="온결 콜라겐 젤리", option="청포도")],
      ["청포도 맛은 품절이라고 안내한다", "11월 27일 재입고 예정이라고 안내한다"]),
    C("상품", "tool_lookup", "유산균 얼마예요? 냉장고에 넣어야 돼요?",
      [call("get_product_info", product_name="온결 유산균")],
      ["가격이 29,000원이라고 안내한다", "실온 보관도 되지만 냉장 보관을 권장한다고 안내한다"], t=["short"]),
    C("상품", "tool_lookup", "어… 그 애들 먹는 비타민 젤리 있잖아요, 몇 살부터 먹을 수 있어요?",
      [call("get_product_info", product_name="온결 키즈 멀티비타민")],
      ["만 3세 이상부터 섭취할 수 있다고 안내한다", "하루 2개를 씹어서 먹는다고 안내한다"], t=["filler_words"]),
    C("상품", "followup", "그거 성분이랑 언제 먹는지 알려 주세요.",
      [call("get_product_info", product_name="온결 철분 엽산")],
      ["철분 12mg과 엽산 400μg이 들어 있다고 안내한다", "하루 1정을 식후에 섭취한다고 안내한다"],
      h=[{"role": "customer", "text": "철분제 있어요?"},
         {"role": "agent", "text": "네, 고객님. 온결 철분 엽산 제품이 있습니다. 어떤 점이 궁금하세요?"}]),
    # ── 상담원 연결 (5)
    C("상담원연결", "complaint_handoff", "유산균 먹고 나서 배가 계속 아프고 설사를 해요. 이거 계속 먹어도 돼요?",
      [call("transfer_to_agent", reason="제품 섭취 후 복통·설사 이상 반응 문의")],
      ["불편에 공감하고 사과한다", "상담원에게 연결한다고 안내한다"],
      ["계속 섭취해도 괜찮다고 말한다", "의학적 진단이나 원인을 추정한다"]),
    C("상담원연결", "complaint_handoff", [("오메가쓰리 먹고 나서", 800), ("속이 울렁거리고 두드러기가 났어요.", 0)],
      [call("transfer_to_agent", reason="제품 섭취 후 메스꺼움·두드러기 이상 반응")],
      ["불편에 공감하고 사과한다", "상담원에게 연결한다고 안내한다", "증상이 심하면 병원 진료를 권한다"],
      ["의학적 진단이나 원인을 추정한다", "섭취를 계속해도 된다고 말한다"], who=SEOYUN,
      t=["mid_pause", "connective_end"]),
    C("상담원연결", "complaint_handoff", "홍삼스틱 먹고 나서부터 가슴이 두근거리고 좀 어지러워요.",
      [call("transfer_to_agent", reason="홍삼 섭취 후 두근거림·어지러움 이상 반응")],
      ["불편에 공감하고 상담원에게 연결한다고 안내한다"],
      ["홍삼 때문이 아니라고 단정한다", "의학적 진단을 내린다"]),
    C("상담원연결", "complaint_handoff", "그냥 상담원이랑 직접 통화하고 싶어요.",
      [call("transfer_to_agent", reason="고객 상담원 연결 직접 요청")],
      ["상담원에게 연결한다고 안내한다"], t=["short"]),
    C("상담원연결", "complaint_handoff", "지금 상담원 연결돼요? 안 되면 나중에 전화 주세요.",
      [call("request_callback", phone_number=DOYUN)],
      ["지금은 상담 시간이 아니라고 안내한다", "콜백 요청이 접수되었다고 안내한다",
       "다음 영업일인 11월 23일 월요일에 연락한다고 안내한다"],
      ["지금 바로 상담원에게 연결한다고 말한다"], who=DOYUN, now="2026-11-20 19:30"),
]

VOICES = ["gemini:Puck", "gemini:Charon", "gemini:Fenrir", "gemini:Leda", "gemini:Zephyr", "gemini:Aoede",
          "gemini:Despina", "gemini:Orus"]
TIMES = ["09:10", "09:40", "10:15", "10:50", "11:20", "11:45", "13:20", "14:05", "14:40", "15:10", "15:55",
         "16:30", "17:05", "17:40"]


def _spread(rng, values, weights, n):
    counts = [round(w * n) for w in weights]
    counts[0] += n - sum(counts)
    out = [v for v, c in zip(values, counts) for _ in range(c)]
    rng.shuffle(out)
    return out


def build_cases():
    rng = random.Random(SEED)
    n = len(SCENARIOS)
    voices = [VOICES[i % len(VOICES)] for i in range(n)]
    rng.shuffle(voices)
    channel = _spread(rng, ["wideband", "phone", "phone_ulaw"], [0.34, 0.33, 0.33], n)
    noise = _spread(rng, ["none", "white", "pink", "babble", "hum"], [0.35, 0.15, 0.15, 0.2, 0.15], n)
    snr = _spread(rng, [20, 30, 10, 5], [0.35, 0.25, 0.25, 0.15], n)
    gain = _spread(rng, [0, -12, 6], [0.6, 0.2, 0.2], n)
    speed = _spread(rng, [1.0, 0.85, 1.2], [0.6, 0.2, 0.2], n)
    loss = _spread(rng, [0.0, 0.02], [0.875, 0.125], n)
    leads = [200 + round(1300 * i / (n - 1) / 10) * 10 for i in range(n)]
    rng.shuffle(leads)
    members = {m[0]: m[1] for m in MEMBERS}
    rows = []
    for i, sc in enumerate(SCENARIOS):
        segs = sc["s"] if isinstance(sc["s"], list) else [(sc["s"], 0)]
        segments = [{"text": t, "pause_after_ms": p} for t, p in segs]
        tags = list(sc["t"])
        prof = None
        if sc["who"]:
            prof = {"name": members[sc["who"]], "phone": sc["who"], "email": None}
            tags.append("caller_known")
        row = {
            "case_id": f"H{i + 1:04d}",
            "now": sc["now"] or f"2026-11-20 {TIMES[rng.randrange(len(TIMES))]}",
            "history": sc["h"],
            "customer_profile": prof,
            "customer_utterance": " ".join(s["text"] for s in segments),
            "segments": segments,
            "voice": voices[i],
            "env": {"channel": channel[i], "noise": noise[i], "snr_db": snr[i], "gain_db": gain[i],
                    "speed": speed[i], "lead_ms": leads[i], "packet_loss": loss[i]},
            "category": sc["cat"],
            "task_type": sc["task"],
            "tags": tags,
            "expected_tools": sc["x"],
            "expected_points": sc["p"],
            "must_not": sc["n"],
        }
        if sc["style"]:
            row["voice_style"] = sc["style"]
        rows.append(row)
    return rows


# ───────────────────────── 매핑 형식 ─────────────────────────
PHONE_FNS = ("get_member_info", "get_order_list", "request_callback")
ORDER_FNS = ("get_order_info", "track_delivery", "change_shipping_address", "cancel_order", "request_return",
             "request_exchange", "get_refund_status", "get_payment_info", "check_deposit")
FILLER = {"change_shipping_address": {"new_address": "주소"}, "transfer_to_agent": {"reason": "사유"}}


def _ok(data=None, message=None):
    out = {"status": "ok"}
    if message is not None:
        out["message"] = message
    if data is not None:
        out["data"] = data
    return out


def _payment_result(name, o, p):
    oid = o["order_id"]
    if name == "get_payment_info":
        msg = None
        if p["status"] == "입금대기":
            msg = (f"입금 대기 중입니다. {p['deposit_deadline']}까지 {p['bank']} {p['account_number']}"
                   f"(예금주 {p['account_holder']})로 {p['amount']:,}원을 입금해 주세요.")
        return _ok(dict(p), msg)
    # check_deposit
    if p["method"] != "무통장입금":
        return {"status": "not_allowed", "message": "무통장입금 주문이 아닙니다.",
                "data": {"order_id": oid, "method": p["method"], "payment_status": p["status"]}}
    d = {"order_id": oid, "amount": p["amount"], "depositor_name": p["depositor_name"],
         "deposit_confirmed": p["deposit_confirmed"], "deposit_deadline": p["deposit_deadline"],
         "payment_status": p["status"]}
    if p["deposit_confirmed"]:
        d["deposited_at"] = p["deposited_at"]
        msg = "입금이 확인되었습니다." if p["status"] != "환불완료" else "입금이 확인되었던 주문이며, 취소되어 환불되었습니다."
    else:
        msg = ("아직 입금이 확인되지 않았습니다. 입금 후 확인까지 최대 1시간이 걸릴 수 있고, "
               "입금자명이 주문자명과 다르면 확인되지 않을 수 있습니다.")
    return _ok(d, msg)


def _pending_result(name, o):
    """'입금대기' 주문: 지금 서버 규칙(Shop)에 없는 상태라 여기서 정한다(매뉴얼 5절)."""
    oid, base = o["order_id"], {"order_id": o["order_id"], "status": "입금대기"}
    if name == "cancel_order":
        return _ok({"order_id": oid, "status": "취소완료", "refund_amount": 0, "refund_method": "없음(입금 전)"},
                   "입금 전 주문이라 바로 취소되었습니다. 환불할 금액은 없습니다.")
    if name == "track_delivery":
        return _ok({**base, "expected_ship_date": o["expected_ship_date"]},
                   "입금 확인 전이라 아직 출고 전입니다. 입금이 확인되면 1~3영업일 안에 출고됩니다.")
    if name in ("request_return", "request_exchange"):
        return {"status": "not_allowed", "message": "입금 전 주문이라 반품·교환 대상이 아닙니다. 주문 취소가 가능합니다.",
                "data": base}
    if name == "change_shipping_address":
        return _ok(base, "배송지가 변경되었습니다. 입금 확인 후 변경된 주소로 출고됩니다.")
    return None


def build_mapping(data):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from mock_api.legacy_shop import Shop
    shop = Shop(tools=copy.deepcopy(TOOLS), data=copy.deepcopy(data))
    assert not getattr(shop, "mapping", False)

    def rule(name, args):
        res, matched = shop.lookup(name, {**args, **FILLER.get(name, {})})
        assert matched, (name, args, res)
        return res

    pays = {p["order_id"]: p for p in data["payments"]}
    out = {t["name"]: [] for t in TOOLS}
    for name in PHONE_FNS:
        for m in data["members"]:
            a = {"phone_number": m["phone_number"]}
            out[name].append({"arguments": a, "result": rule(name, a)})
    for name in ORDER_FNS:
        for o in data["orders"]:
            a = {"order_id": o["order_id"]}
            if name in ("get_payment_info", "check_deposit"):
                res = _payment_result(name, o, pays[o["order_id"]])
            elif o["status"] == "입금대기" and _pending_result(name, o) is not None:
                res = _pending_result(name, o)
            else:
                res = rule(name, a)
            out[name].append({"arguments": a, "result": res})
    for p in data["products"]:
        a = {"product_name": p["name"]}
        res = rule("get_product_info", a)
        res["data"].update({k: p[k] for k in ("volume", "ingredients", "how_to_take", "cautions", "storage")})
        out["get_product_info"].append({"arguments": a, "result": res})
    for p in data["products"]:     # 옵션 지정 항목이 먼저(서버는 앞에서부터 찾는다)
        for o in p["options"]:
            a = {"product_name": p["name"], "option": o["name"]}
            out["check_stock"].append({"arguments": a, "result": rule("check_stock", a)})
    for p in data["products"]:
        a = {"product_name": p["name"]}
        out["check_stock"].append({"arguments": a, "result": rule("check_stock", a)})
    # 키 인자가 없는 함수: 빈 arguments 항목은 어떤 reason에도 맞는다
    out["transfer_to_agent"].append({"arguments": {}, "result": rule("transfer_to_agent", {})})
    missing = [n for n, rows in out.items() if not rows]
    assert not missing, missing
    return out


def _dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    data = build_data()
    _dump(DIR / "tools.json", TOOLS)
    _dump(DIR / "source_data.json", data)
    _dump(DIR / "mock_data.json", build_mapping(data))
    rows = build_cases()
    with open(DIR / "cases.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"tools {len(TOOLS)}, products {len(PRODUCTS)}, orders {len(ORDERS_SRC)}, cases {len(rows)}")


if __name__ == "__main__":
    main()
