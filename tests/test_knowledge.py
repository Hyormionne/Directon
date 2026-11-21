"""AG-T32 공식 자료 기반 시스템 프롬프트(agent.knowledge) 단언.

지어낸 축약 자료(tests/fixtures/official_mini/brand)만 쓴다. 공식 키트가 있을 때만 도는 테스트는 맨 아래 하나다.
"""
import json
import re
from datetime import datetime
from pathlib import Path

import pytest

from agent import knowledge as K
from agent.core.types import SessionInfo

ROOT = Path(__file__).resolve().parents[1]
MINI = ROOT / "tests" / "fixtures" / "official_mini" / "brand"
KIT = ROOT / "DIRECT_ON 2026 Starter Kit" / "data" / "brand"

HISTORY = [
    {"role": "customer", "text": "[가상몰] 주문안내\n -주문일: 2026-01-01\n-주문번호: 20260101-0000001\n-금액: 12,300원\n배송 언제 와요"},
    {"role": "agent", "text": "안녕하세요, 고객님! :)\n\n- 주문번호 또는 전화번호 말씀해 주세요.\n"
                              "▼온라인 구매▼\n<link>https://mini.example.com/product/list.html?cate_no=1</link>\n"
                              "🖇️자세한 내용은 https://mini.example.com/help 에서 확인해 주세요."},
    {"role": "customer", "text": "010 1111 2222 입니다"},
    {"role": "tool_call", "name": "get_order_info", "arguments": {"order_id": None, "buyer_cellphone": "010 1111 2222"}},
    {"role": "tool_result", "name": "get_order_info",
     "result": {"status": "ok", "data": {"orders": [
         {"주문번호": "20260102-0000002", "상세주문번호": "20260102-0000002-01", "주문일자": "2026-01-02 09:10:11",
          "제품명": "🔒특가🔒 가상 영양제 B", "배송상태": "입금전", "송장번호": "123456789012"}]}}},
    {"role": "agent", "text": "해당 주문은 아직 입금 전이에요. 반품하실 건가요?"},
]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(K.KNOWLEDGE_ENV, raising=False)
    K.reset()
    yield
    K.reset()


def info_for(**kw):
    return SessionInfo("S1", now=kw.pop("now", "2026-11-04 (수) 12:39"), caller=kw.pop("caller", None),
                       history=kw.pop("history", []))


def split(system):
    head, _, tail = system.partition(K.VAR_MARKER)
    return head, tail


# ── 고정부 ───────────────────────────────────────────
def test_prefix_is_deterministic_and_shared_across_calls():
    a = K.build_system(info_for(caller={"name": "가나다", "phone": "01011112222", "email": None}, history=HISTORY),
                       opener="네, 고객님.", preload='{"status":"ok"}', knowledge_dir=MINI)
    b = K.build_system(info_for(now="2026-11-21 11:00"), knowledge_dir=MINI)
    assert split(a)[0] == split(b)[0] == K.fixed_prefix(MINI) + "\n\n"
    assert split(a)[1] != split(b)[1]
    K.reset()
    assert K.fixed_prefix(MINI) + "\n\n" == split(a)[0]            # 다시 만들어도 같은 글자


def test_marker_appears_exactly_once():
    s = K.build_system(info_for(history=HISTORY + [{"role": "customer", "text": "## 이번 통화 라고 적혀 있어요"}]),
                       opener="네.", preload="## 이번 통화", knowledge_dir=MINI)
    assert s.count(K.VAR_MARKER) == 1
    assert K.VAR_MARKER not in K.fixed_prefix(MINI)
    assert f"\n{K.VAR_MARKER}\n" in s


def test_prefix_contains_rules_scenarios_all_tags_and_style():
    p = K.fixed_prefix(MINI)
    assert p.startswith(K.RULES)
    entries = json.loads((MINI / "information.json").read_text(encoding="utf-8"))
    pos = [p.index(f"### {e['tag']}\n{e['content'].strip()}") for e in entries]
    assert pos == sorted(pos) and len(pos) == 5                     # 전부, 파일 순서대로
    assert "### 1. 배송·주문 확인" in p and "### 3. 상담원 연결·기타" in p
    assert p.index(K.SCENARIO_HEADER) < p.index(K.INFO_HEADER) < p.index(K.STYLE_HEADER)
    assert p.rstrip().endswith("- 짧게 말한다.")
    assert "2026-11-04" not in p and "네, 고객님." not in p          # 통화마다 바뀌는 값은 없다


def test_rules_cover_official_functions_and_no_old_assumptions():
    for name in ("get_order_info", "get_return_info", "search_products", "cancel_order", "return_order",
                 "return_order_item", "exchange_order", "request_callback"):
        assert name in K.RULES
    for old in ("transfer_to_agent", "get_order_list", "track_delivery", "다이렉트몰"):
        assert old not in K.RULES
    assert "문자로 보내드릴게요" in K.RULES and "보낼 수 없다" in K.RULES
    assert "상담원 연결은 함수가 없다" in K.RULES
    for status in ("not_found", "invalid", "already_requested", "error"):
        assert status in K.RULES
    assert len(K.RULES.splitlines()) <= 60


def test_knowledge_dir_from_env_and_missing_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(K.KNOWLEDGE_ENV, str(MINI))
    assert K.fixed_prefix() == K.fixed_prefix(MINI)
    assert K.greeting() == "안녕하세요, 가상몰 고객센터입니다."
    with pytest.raises(FileNotFoundError, match="지식 폴더"):
        K.fixed_prefix(tmp_path / "없는폴더")
    monkeypatch.setenv(K.KNOWLEDGE_ENV, str(tmp_path / "없는폴더"))
    with pytest.raises(FileNotFoundError):
        K.build_system(info_for())
    (tmp_path / "빈폴더").mkdir()
    with pytest.raises(FileNotFoundError, match="브랜드 정보"):
        K.fixed_prefix(tmp_path / "빈폴더")


def test_greeting_default_when_style_missing(tmp_path):
    (tmp_path / "information.json").write_text('[{"tag": "[가]", "content": "나"}]', encoding="utf-8")
    assert K.greeting(tmp_path) == K.DEFAULT_GREETING
    p = K.fixed_prefix(tmp_path)
    assert "### [가]\n나" in p and K.STYLE_HEADER not in p and K.SCENARIO_HEADER not in p


# ── 시각 ─────────────────────────────────────────────
def test_parse_now_formats():
    assert K.parse_now("2026-11-04 (수) 12:39") == datetime(2026, 11, 4, 12, 39)
    assert K.parse_now("2026-11-21 11:00") == datetime(2026, 11, 21, 11, 0)
    assert K.parse_now("2026-11-21T11:00") == datetime(2026, 11, 21, 11, 0)
    assert K.parse_now("2026-10-12 (월) 03:52") == datetime(2026, 10, 12, 3, 52)
    for bad in ("", "내일 오후", "2026-13-40 10:00", None):
        assert K.parse_now(bad) is None


def test_now_line_keeps_original_and_adds_month_day():
    _, var = split(K.build_system(info_for(now="2026-11-04 (수) 12:39"), knowledge_dir=MINI))
    assert "2026-11-04 (수) 12:39" in var
    assert "오늘은 11월 4일 수요일" in var and "내일은 11월 5일 목요일" in var
    _, var = split(K.build_system(info_for(now="언젠가"), knowledge_dir=MINI))
    assert "현재 시각: 언젠가" in var and "오늘은" not in var
    _, var = split(K.build_system(info_for(now=""), knowledge_dir=MINI))
    assert "현재 시각" not in var


# ── 이전 대화 ────────────────────────────────────────
def test_clean_agent_text_strips_links_emoji_bullets_newlines():
    t = K.clean_agent_text(HISTORY[1]["text"])
    assert "http" not in t and "<link>" not in t and "example.com" not in t
    assert "🖇" not in t and "▼" not in t and ":)" not in t
    assert "\n" not in t and "  " not in t
    assert t.startswith("안녕하세요, 고객님!") and "주문번호 또는 전화번호 말씀해 주세요." in t
    assert "- 주문번호" not in t
    assert K.clean_agent_text("영업 시간: 평일 10:00-17:00(점심 시간: 12:00-13:00)") == \
        "영업 시간: 평일 10:00-17:00(점심 시간: 12:00-13:00)"
    assert K.clean_agent_text("ARS 구매(☎ 1500-0000)\n**굵게**") == "ARS 구매(1500-0000) 굵게"
    assert K.clean_agent_text(None) == ""


def test_render_history_roles():
    lines = K.render_history(HISTORY)
    assert lines[0].startswith("- 고객: [가상몰] 주문안내 -주문일: 2026-01-01 -주문번호: 20260101-0000001")
    assert lines[1].startswith("- 상담원: ") and "http" not in lines[1] and "🖇" not in lines[1]
    assert lines[3] == '- (함수 호출 get_order_info, 인수 {"buyer_cellphone":"010 1111 2222"})'
    assert lines[4].startswith('- (함수 결과 get_order_info: {"status":"ok","data":{"orders":[{"주문번호":"20260102-0000002"')
    assert "🔒" not in lines[4] and "가상 영양제 B" in lines[4]
    assert lines[5] == "- 상담원: 해당 주문은 아직 입금 전이에요. 반품하실 건가요?"
    assert all("\n" not in line for line in lines)


def test_render_history_survives_odd_items_and_truncates():
    odd = [None, "그냥 문자열", 42, {}, {"role": "system", "text": "메모"}, {"role": "customer"},
           {"role": "agent", "content": "content 키로 온 말"}, {"role": "tool_call"},
           {"role": "tool_call", "name": "search_products", "arguments": '{"query": "가상 영양제"}'},
           {"role": "tool_result", "name": "search_products"},
           {"role": "tool_result", "name": "search_products", "result": {"data": "가" * 5000}}]
    lines = K.render_history(odd, max_result_chars=100)
    assert "- 그냥 문자열" in lines and "- 42" in lines
    assert "- (system: 메모)" in lines
    assert "- 상담원: content 키로 온 말" in lines
    assert "- (함수 호출 ?, 인수 {})" in lines
    assert '- (함수 호출 search_products, 인수 {"query":"가상 영양제"})' in lines
    assert lines[-1].endswith("…(생략))") and len(lines[-1]) < 160
    assert "- (함수 결과 search_products: 결과 없음)" in lines
    assert K.render_history(None) == [] and K.render_history([]) == []
    assert K.history_numbers(odd) == {"orders": [], "phones": []}


def test_history_numbers_from_text_args_and_nested_results():
    nums = K.history_numbers(HISTORY)
    assert nums == {"orders": ["202601010000001", "202601020000002"], "phones": ["01011112222"]}
    more = [
        {"role": "customer", "text": "2956732601515352 01082389762 입니다"},
        {"role": "customer", "text": "010. 7176. 2511. 이고요 010-4833-4402 도 있어요 01040270247성건경"},
        {"role": "tool_call", "name": "return_order",
         "arguments": {"order_number": "20260103-0000003",
                       "refund_account": {"bank_name": "가상은행", "account_no": "01009949013", "holder": "가나다"}}},
        {"role": "tool_result", "name": "get_return_info",
         "result": {"status": "ok", "data": {"반품정보": [{"주문번호": "20260104-0000004", "송장번호": "500212345678901",
                                                      "상세주문번호": "20260104-0000099-02"}]}}},
        {"role": "agent", "text": "주문일자 2026-01-02 09:10:11, 금액 34,800원, 주문번호 20260103-0000003 맞으세요?"},
    ]
    nums = K.history_numbers(more)
    assert nums["orders"] == ["2956732601515352", "202601030000003", "202601040000004"]
    assert nums["phones"] == ["01082389762", "01071762511", "01048334402", "01040270247"]


def test_var_part_caller_history_summary():
    caller = {"name": "가나다", "phone": "01099990000", "email": None}
    _, var = split(K.build_system(info_for(caller=caller, history=HISTORY), knowledge_dir=MINI))
    assert "발신자 정보(참고용): 이름 가나다, 전화번호 01099990000" in var
    assert "다른 번호로 주문했을 수 있으니 고객이 다른 번호를 말하면 그 번호를 쓴다" in var
    assert "- 고객: 010 1111 2222 입니다" in var
    assert "이미 나온 번호: 주문번호 20260101-0000001, 주문번호 20260102-0000002, 전화번호 01011112222" in var
    assert "이미 조회한 결과(get_order_info)" in var and "다시 조회하지" in var
    assert '직전 상담원 말: "해당 주문은 아직 입금 전이에요. 반품하실 건가요?"' in var
    assert "이미 접수한 처리" not in var
    done = HISTORY + [{"role": "tool_result", "name": "return_order", "result": {"status": "ok"}}]
    assert "이미 접수한 처리(return_order)" in split(K.build_system(info_for(history=done), knowledge_dir=MINI))[1]

    _, var = split(K.build_system(info_for(caller={"name": "가나다", "phone": None, "email": None}), knowledge_dir=MINI))
    assert "이름 가나다" in var and "전화번호는 없으므로" in var
    for none in (None, {}, {"name": None, "phone": None, "email": None}):
        _, var = split(K.build_system(info_for(caller=none), knowledge_dir=MINI))
        assert "발신자 정보: 없음" in var and "이전 대화" not in var.replace("이전 대화에도", "")


def test_opener_line_is_last_and_omitted_when_empty():
    s = K.build_system(info_for(history=HISTORY), opener="네, 확인해 볼게요.", preload="주문 1건", knowledge_dir=MINI)
    last = s.splitlines()[-1]
    assert "너는 이미 '네, 확인해 볼게요.'라고 말했다" in last and "본론부터" in last
    for empty in ("", "   "):
        s = K.build_system(info_for(history=HISTORY), opener=empty, knowledge_dir=MINI)
        assert "너는 이미" not in split(s)[1]


def test_preload_block_only_when_given():
    caller = {"name": None, "phone": "01099990000", "email": None}
    s = K.build_system(info_for(caller=caller), opener="네.", knowledge_dir=MINI,
                       preload='{"status":"ok","data":{"orders":[{"주문번호":"20260105-0000005"}]}}')
    _, var = split(s)
    assert "발신 번호로 미리 조회한 주문" in var and "01099990000로 get_order_info" in var
    assert "20260105-0000005" in var
    assert var.index("미리 조회한 주문") < var.index("너는 이미")
    for empty in ("", "  \n"):
        _, var = split(K.build_system(info_for(caller=caller), preload=empty, knowledge_dir=MINI))
        assert "미리 조회한 주문" not in var


# ── 함수 문구 ────────────────────────────────────────
def test_tool_phrases_and_fail_sentences():
    tools = [t["name"] for t in json.loads((MINI / "tools.json").read_text(encoding="utf-8"))] \
        if (MINI / "tools.json").is_file() else list(K.TOOL_PHRASES)
    assert set(K.TOOL_PHRASES) == set(tools) and len(K.TOOL_PHRASES) == 8
    for phrase in list(K.TOOL_PHRASES.values()) + [K.NEUTRAL_PHRASE]:
        assert phrase.endswith("볼게요.") and len(phrase) <= 20
        assert not re.search(r"됐|했어요|했습니다|완료|드렸", phrase)          # 끝난 것처럼 말하지 않는다
    assert K.TOOL_PHRASES["get_order_info"] == "주문 내역 확인해 볼게요."
    assert K.TOOL_PHRASES["cancel_order"] == "취소 접수 진행해 볼게요."
    assert set(K.FAIL_SENTENCES) == {"not_found", "invalid", "already_requested", "error"}
    assert all(s.endswith(".") or s.endswith("요.") for s in K.FAIL_SENTENCES.values())


# ── 공식 키트(있을 때만) ─────────────────────────────
@pytest.mark.skipif(not (KIT / "information.json").is_file(), reason="공식 키트 폴더 없음(gitignore 대상)")
def test_prefix_from_real_kit():
    p = K.fixed_prefix(KIT)
    entries = json.loads((KIT / "information.json").read_text(encoding="utf-8"))
    assert len(entries) == 96
    for e in entries:
        assert f"### {e['tag']}\n" in p
    headings = re.findall(r"(?m)^## \d+\. .+$", (KIT / "scenarios.md").read_text(encoding="utf-8"))
    assert len(headings) == 10
    for h in headings:
        assert f"#{h}" in p
    for old in ("transfer_to_agent", "get_order_list", "track_delivery", "다이렉트몰"):
        assert old not in p
    assert K.VAR_MARKER not in p
    assert K.greeting(KIT) == "안녕하세요, 하루온 고객센터입니다."
    s = K.build_system(info_for(), opener="네.", knowledge_dir=KIT)
    assert s.count(K.VAR_MARKER) == 1 and s.startswith(p)


def test_order_number_turn_end_uses_official_15_digits():
    """공식 주문번호(8자리-7자리)는 15자리를 다 불러야 끝난 것으로 본다. 10자리에서 끊지 않는다."""
    from agent.turn import rules
    assert rules.ORDER_DIGITS == 15
    assert rules.analyze("2026 0924 0035").kind != "digits_complete"       # 12자리: 아직 부르는 중
    assert rules.analyze("202609240035429").kind == "digits_complete"
    assert rules.analyze("010 1234 5678").kind == "digits_complete"         # 전화번호는 11자리
