"""AG-T05a 테스트셋(legacy/testset/cases.jsonl) 스키마·Mock 일치·분포 검사.

형식: docs/AGENT_PLAN.md 3.2(스키마), 3.3(통화 환경), 3.4(범주·태그).
생성: scripts/gen_scenarios.py (원본 legacy/testset/scenarios_src.yaml).
"""
import importlib.util
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import pytest

from mock_api.server import load_shop

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "legacy" / "testset" / "cases.jsonl"
GEN = ROOT / "scripts" / "gen_scenarios.py"

CATEGORIES = {"faq", "tool_lookup", "tool_action", "need_info", "out_of_scope", "complaint_handoff", "followup"}
TAGS = {"mid_pause", "filler_words", "connective_end", "digits", "self_correction", "negation",
        "banmal", "long", "short", "caller_known"}
CHANNELS = {"wideband", "phone", "phone_ulaw"}
NOISES = {"none", "white", "pink", "babble", "hum"}
SNRS = {30, 20, 10, 5}
GAINS = {0, -12, 6}
LOSSES = {0.0, 0.02}
SPEEDS = {0.85, 1.0, 1.2}
REQUIRED = {"case_id": str, "now": str, "history": list, "customer_utterance": str, "segments": list,
            "voice": str, "env": dict, "category": str, "tags": list, "expected_tools": list,
            "expected_points": list, "must_not": list}
ENV_KEYS = {"channel", "noise", "snr_db", "gain_db", "speed", "lead_ms", "packet_loss"}
VOICE_RE = re.compile(r"^(gemini:[A-Z][a-z]+|edge:ko-KR-[A-Za-z]+Neural)$")
NOW_RE = re.compile(r"^2026-11-(\d\d) ([01]\d|2[0-3]):[0-5]\d$")
PHONE_TOOLS = {"get_member_info", "get_order_list", "get_coupons", "request_callback"}


def _load():
    assert CASES.exists(), f"{CASES} 없음 — scripts/gen_scenarios.py로 생성"
    with open(CASES, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


@pytest.fixture(scope="module")
def cases():
    return _load()


@pytest.fixture(scope="module")
def shop():
    return load_shop()


def _norm(s):
    return re.sub(r"\s+", "", s)


def _ids(cases):
    return [c.get("case_id") for c in cases]


# ---------- 스키마 ----------

def test_file_parses_and_ids_unique(cases):
    assert cases, "빈 테스트셋"
    ids = _ids(cases)
    assert len(ids) == len(set(ids)), [i for i, n in Counter(ids).items() if n > 1]
    for i in ids:
        assert isinstance(i, str) and re.fullmatch(r"T\d{4}", i), i


def test_required_fields_and_types(cases):
    for c in cases:
        for k, t in REQUIRED.items():
            assert k in c, (c.get("case_id"), k)
            assert isinstance(c[k], t), (c["case_id"], k, type(c[k]))
        assert "customer_profile" in c, c["case_id"]
        assert c["customer_profile"] is None or isinstance(c["customer_profile"], dict), c["case_id"]
        assert c["customer_utterance"].strip(), c["case_id"]
        if "voice_style" in c:
            assert c["voice_style"] is None or isinstance(c["voice_style"], str), c["case_id"]


def test_now_format_and_range(cases):
    for c in cases:
        m = NOW_RE.match(c["now"])
        assert m, (c["case_id"], c["now"])
        assert 19 <= int(m.group(1)) <= 23, (c["case_id"], c["now"])
        datetime.strptime(c["now"], "%Y-%m-%d %H:%M")


def test_segments_join_to_utterance(cases):
    for c in cases:
        segs = c["segments"]
        assert segs, c["case_id"]
        for s in segs:
            assert set(s) >= {"text", "pause_after_ms"}, c["case_id"]
            assert isinstance(s["text"], str) and s["text"].strip(), c["case_id"]
            assert isinstance(s["pause_after_ms"], int), c["case_id"]
        assert segs[-1]["pause_after_ms"] == 0, c["case_id"]
        for s in segs[:-1]:
            assert 300 <= s["pause_after_ms"] <= 1200, (c["case_id"], s)
        assert _norm("".join(s["text"] for s in segs)) == _norm(c["customer_utterance"]), c["case_id"]


def test_no_arabic_digits_in_speech(cases):
    """TTS가 읽을 원문이므로 숫자는 한글 낭독으로 적는다."""
    for c in cases:
        assert not re.search(r"[0-9]", c["customer_utterance"]), (c["case_id"], c["customer_utterance"])


def test_voice_format(cases):
    for c in cases:
        assert VOICE_RE.match(c["voice"]), (c["case_id"], c["voice"])
        assert c["voice"] != "gemini:Kore", c["case_id"]  # 상담원 음성 — 과적합 방지


def test_env_values(cases):
    for c in cases:
        e = c["env"]
        assert set(e) >= ENV_KEYS, (c["case_id"], set(e))
        assert e["channel"] in CHANNELS, c["case_id"]
        assert e["noise"] in NOISES, c["case_id"]
        assert e["snr_db"] in SNRS, c["case_id"]
        assert e["gain_db"] in GAINS, c["case_id"]
        assert e["packet_loss"] in LOSSES, c["case_id"]
        assert e["speed"] in SPEEDS, c["case_id"]
        assert isinstance(e["lead_ms"], int) and 200 <= e["lead_ms"] <= 1500, c["case_id"]


def test_category_and_tags_known(cases):
    for c in cases:
        assert c["category"] in CATEGORIES, c["case_id"]
        assert set(c["tags"]) <= TAGS, (c["case_id"], set(c["tags"]) - TAGS)
        assert len(c["tags"]) == len(set(c["tags"])), c["case_id"]


def test_history_shape(cases):
    for c in cases:
        for h in c["history"]:
            assert h.get("role") in ("agent", "customer"), c["case_id"]
            assert isinstance(h.get("text"), str) and h["text"].strip(), c["case_id"]
        if c["history"]:
            assert c["history"][-1]["role"] == "agent", c["case_id"]  # 이번 고객 발화 직전은 상담원
        if c["category"] == "followup":
            assert c["history"], c["case_id"]


def test_points_and_must_not(cases):
    for c in cases:
        assert 1 <= len(c["expected_points"]) <= 3, c["case_id"]
        assert 0 <= len(c["must_not"]) <= 2, c["case_id"]
        for p in c["expected_points"] + c["must_not"]:
            assert isinstance(p, str) and len(p.strip()) >= 6, (c["case_id"], p)


# ---------- 태그 ↔ 내용 일관성 ----------

def test_pause_tags_have_pause(cases):
    for c in cases:
        if {"mid_pause", "connective_end"} & set(c["tags"]):
            assert len(c["segments"]) >= 2, c["case_id"]


def test_caller_known_matches_profile(cases, shop):
    members = {m["phone_number"]: m for m in shop.data["members"]}
    for c in cases:
        p = c["customer_profile"]
        assert ("caller_known" in c["tags"]) == (p is not None), c["case_id"]
        if p is None:
            continue
        assert set(p) == {"name", "phone", "email"}, c["case_id"]
        assert p["phone"] in members, (c["case_id"], p["phone"])
        assert p["name"] == members[p["phone"]]["name"], c["case_id"]


# ---------- expected_tools ↔ Mock API ----------

def test_expected_tools_match_mock(cases, shop):
    for c in cases:
        for call in c["expected_tools"]:
            assert set(call) <= {"name", "arguments", "expect_not_found"}, (c["case_id"], call)
            assert call["name"] in shop.tool_names, (c["case_id"], call["name"])
            assert isinstance(call["arguments"], dict), c["case_id"]
            res, matched = shop.lookup(call["name"], call["arguments"])
            if call.get("expect_not_found"):
                assert not matched and res["status"] == "not_found", (c["case_id"], call, res)
            else:
                assert matched, (c["case_id"], call, res)


def test_caller_owns_orders_and_phone(cases, shop):
    """caller가 있으면 전화번호 인자는 그 번호, 주문번호는 그 회원 주문이어야 한다(본인 확인 규칙)."""
    orders = {o["order_id"]: o for o in shop.data["orders"]}
    for c in cases:
        p = c["customer_profile"]
        if p is None:
            continue
        for call in c["expected_tools"]:
            if call.get("expect_not_found"):
                continue
            a = call["arguments"]
            if call["name"] in PHONE_TOOLS:
                assert a["phone_number"] == p["phone"], c["case_id"]
            if "order_id" in a:
                assert orders[a["order_id"]]["phone_number"] == p["phone"], c["case_id"]


def test_category_tool_rules(cases):
    for c in cases:
        names = {t["name"] for t in c["expected_tools"]}
        cat = c["category"]
        if cat in ("faq", "out_of_scope"):
            assert not names, c["case_id"]
        elif cat == "complaint_handoff":
            assert names & {"transfer_to_agent", "request_callback"}, c["case_id"]
        elif cat == "need_info":
            assert names <= {"get_order_list", "get_member_info"}, c["case_id"]
            if names:
                assert c["customer_profile"] is not None, c["case_id"]
        elif cat in ("tool_lookup", "tool_action"):
            assert names, c["case_id"]


# ---------- 분포 ----------

def test_size_and_categories(cases):
    assert len(cases) >= 160, len(cases)
    cnt = Counter(c["category"] for c in cases)
    for cat in CATEGORIES:
        assert cnt[cat] >= 12, (cat, cnt[cat])


def test_tag_distribution(cases):
    cnt = Counter(t for c in cases for t in c["tags"])
    need = {"mid_pause": 30, "connective_end": 15, "digits": 30, "filler_words": 15, "self_correction": 8,
            "negation": 8, "banmal": 15, "caller_known": 30}
    for t, n in need.items():
        assert cnt[t] >= n, (t, cnt[t], n)
    assert sum(1 for c in cases if c["history"]) >= 20


def test_env_distribution(cases):
    n = len(cases)
    ch = Counter(c["env"]["channel"] for c in cases)
    for v in CHANNELS:
        assert abs(ch[v] / n - 1 / 3) <= 0.10, (v, ch[v])
    noise = Counter(c["env"]["noise"] for c in cases)
    assert set(noise) == NOISES, noise
    assert noise["none"] / n >= 0.30, noise
    assert {c["env"]["snr_db"] for c in cases} == SNRS
    assert {c["env"]["gain_db"] for c in cases} == GAINS
    assert {c["env"]["speed"] for c in cases} == SPEEDS
    assert sum(1 for c in cases if c["env"]["packet_loss"] > 0) >= 10
    leads = [c["env"]["lead_ms"] for c in cases]
    assert min(leads) >= 200 and max(leads) <= 1500
    assert max(leads) - min(leads) >= 800, "lead_ms가 범위를 고르게 쓰지 않는다"


def test_voice_distribution(cases):
    voices = Counter(c["voice"] for c in cases)
    gem = {v for v in voices if v.startswith("gemini:")}
    edge = {v for v in voices if v.startswith("edge:")}
    assert len(voices) >= 10 and len(gem) >= 6 and len(edge) >= 3, voices
    assert "gemini:Kore" not in voices


def test_off_hours_cases_exist(cases):
    def off(now):
        d = datetime.strptime(now, "%Y-%m-%d %H:%M")
        return d.weekday() >= 5 or not (9 <= d.hour < 18) or d.hour == 12
    assert sum(1 for c in cases if off(c["now"])) >= 5


# ---------- 생성기 동기화 ----------

def test_generator_reproduces_file(cases):
    spec = importlib.util.spec_from_file_location("gen_scenarios", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = mod.build_cases()
    assert rows == cases, "cases.jsonl이 생성기 출력과 다르다 — gen_scenarios.py를 다시 실행"
