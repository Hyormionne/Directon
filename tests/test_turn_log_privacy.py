"""시험 음성 저장 금지(DIRECTON_NO_AUDIO_LOG): 턴 로그에서 고객 발화를 지운다."""
from __future__ import annotations

import json

from agent.turn_log import TurnLogWriter, no_audio_log



def _record():
    return {"session_id": "s", "user_text": "주문 배송 언제 와요 공일공", "agent_text": "내일 도착해요.",
            "marks": {"decided": 10.0},
            "events": [{"name": "precall", "t": 1.0, "text": "주문 배송 언제", "source": "partial"},
                       {"name": "finalize_early", "t": 2.0, "text": "주문 배송 언제 와요"},
                       {"name": "number_repaired", "t": 3.0, "tool": "x", "key": "order_id",
                        "heard": "공일공", "fixed": "1234"},
                       {"name": "tool_call", "t": 4.0, "tool": "track_delivery", "args": {"order_id": "1234"}}]}


def test_no_audio_log_flag(monkeypatch):
    assert no_audio_log() is False
    monkeypatch.setenv("DIRECTON_NO_AUDIO_LOG", "1")
    assert no_audio_log() is True


def test_turn_log_keeps_text_by_default(tmp_path):
    TurnLogWriter(tmp_path, date_fn=lambda: "d").write(_record())
    assert "배송 언제" in (tmp_path / "turns-d.jsonl").read_text(encoding="utf-8")


def test_turn_log_scrubs_customer_text_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("DIRECTON_NO_AUDIO_LOG", "1")
    TurnLogWriter(tmp_path, date_fn=lambda: "d").write(_record())
    files = list(tmp_path.rglob("*"))
    assert not [f for f in files if f.suffix.lower() in (".wav", ".pcm", ".raw")]
    raw = (tmp_path / "turns-d.jsonl").read_text(encoding="utf-8")
    assert "배송 언제" not in raw and "공일공" not in raw
    rec = json.loads(raw)
    assert rec["marks"] == {"decided": 10.0}            # 시간 지표는 남긴다
    assert rec["agent_text"] == "내일 도착해요."
    assert [e["name"] for e in rec["events"]] == ["precall", "finalize_early", "number_repaired", "tool_call"]
