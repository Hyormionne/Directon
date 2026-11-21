"""AG-T52 데모 통화 묶음(bundle) 저장: demo/feedback_store.py + POST /api/sessions.

네트워크 금지: 데모 서버는 127.0.0.1 임의 포트, 오디오는 테스트에서 만든 사인파·무음.
"""
import array
import base64
import io
import json
import math
import threading
import urllib.error
import urllib.request
import wave

import pytest

from demo import feedback_store as fs
from demo import server as demo_server

RATE = 16000


def _tone(ms: int, amp: int = 8000, freq: float = 440.0) -> bytes:
    n = ms * RATE // 1000
    return array.array("h", (int(amp * math.sin(2 * math.pi * freq * i / RATE)) for i in range(n))).tobytes()


def _silence(ms: int) -> bytes:
    return b"\x00\x00" * (ms * RATE // 1000)


def _wav_b64(pcm: bytes, rate: int = RATE, channels: int = 1) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return base64.b64encode(buf.getvalue()).decode()


def _read_wav(path):
    with wave.open(str(path), "rb") as w:
        return w.getframerate(), w.getnchannels(), w.readframes(w.getnframes())


def _bundle(sid="DEMO-20261121-101500-ab12", customer=None, agent=None):
    # 고객: 0~1s 무음, 1~3s 말, 3~6s 무음(상담원 말하는 동안 막힘), 6~8s 말, 8~9s 무음
    customer = customer if customer is not None else (
        _silence(1000) + _tone(2000) + _silence(3000) + _tone(2000) + _silence(1000))
    # 상담원: 3.5~5.5s, 8.5~9.5s
    agent = agent if agent is not None else (
        _silence(3500) + _tone(2000, freq=220) + _silence(3000) + _tone(1000, freq=220))
    return {
        "schema": fs.SCHEMA,
        "session_id": sid,
        "started_at": "2026-11-21T10:15:00.000Z",
        "ended_at": "2026-11-21T10:15:10.000Z",
        "setup": {"caller": {"name": "한서윤", "phone": "01072641185", "email": None},
                  "now": "2026-11-20 14:00", "history_preset": None,
                  "history": [{"role": "customer", "text": "배송 문의요."},
                              {"role": "agent", "text": "네, 말씀하세요."}],
                  "half_duplex": True, "adaptive_opener": True},
        "feedback": {"rating": 4, "comment": "두 번째 답이 느렸다"},
        "turns": [
            {"index": 1, "user_stt": "주문 어디쯤 왔어요", "user_corrected": "",
             "agent": "확인해 보니 배송 중입니다.", "tools": [{"name": "get_order_info", "arguments": {"buyer_cellphone": "01072641185"}}],
             "ts": {"speech_start_ms": 1000, "decided_ms": 3300, "turn_end_ms": 5400, "agent_audio_end_ms": 5500},
             "decided": {"silence_ms": 300},
             "audio": {"customer": {"start_ms": 500, "end_ms": 3500, "speech_end_ms": 2500},
                       "agent": {"start_ms": 3300, "end_ms": 5700}},
             "feedback": {"vote": "up", "tags": [], "note": ""}},
            {"index": 2, "user_stt": "언제 와요 그럼", "user_corrected": "언제 와요, 그럼?",
             "agent": "내일 도착 예정입니다.", "tools": [],
             "ts": {"speech_start_ms": 6000, "decided_ms": 8400, "turn_end_ms": 9400, "agent_audio_end_ms": 9500},
             "decided": {"silence_ms": 400},
             "audio": {"customer": {"start_ms": 5500, "end_ms": 8600, "speech_end_ms": 2500},
                       "agent": {"start_ms": 8400, "end_ms": 9700}},
             "feedback": {"vote": "down", "tags": ["slow"], "note": "공백이 길다"}},
        ],
        "audio": {"encoding": "pcm_s16le", "sample_rate": RATE, "channels": 1,
                  "tracks": {"customer": {"wav_base64": _wav_b64(customer)},
                             "agent": {"wav_base64": _wav_b64(agent)}}},
    }


# ── write_session ─────────────────────────────────────
def test_write_session_lays_out_files(tmp_path):
    res = fs.write_session(_bundle(), tmp_path)
    d = tmp_path / "DEMO-20261121-101500-ab12"
    assert res["dir"] == str(d)
    for name in ["session.json", "customer.wav", "agent.wav", "mix.wav", "cases.jsonl",
                 "turns/01_customer.wav", "turns/01_agent.wav", "turns/02_customer.wav", "turns/02_agent.wav"]:
        assert (d / name).is_file(), name
    rate, ch, pcm = _read_wav(d / "customer.wav")
    assert (rate, ch, len(pcm)) == (RATE, 1, 9000 * 32)


def test_session_json_has_paths_not_base64(tmp_path):
    fs.write_session(_bundle(), tmp_path)
    d = tmp_path / "DEMO-20261121-101500-ab12"
    raw = (d / "session.json").read_text(encoding="utf-8")
    assert "wav_base64" not in raw
    rec = json.loads(raw)
    assert rec["schema"] == fs.SCHEMA
    assert rec["audio"]["tracks"]["customer"] == {"file": "customer.wav", "duration_ms": 9000}
    assert rec["audio"]["tracks"]["agent"]["file"] == "agent.wav"
    assert rec["audio"]["mix"] == {"file": "mix.wav", "channels": 2, "left": "customer", "right": "agent"}
    t2 = rec["turns"][1]
    assert t2["audio"]["customer"]["file"] == "turns/02_customer.wav"
    assert t2["audio"]["agent"]["file"] == "turns/02_agent.wav"
    assert t2["feedback"] == {"vote": "down", "tags": ["slow"], "note": "공백이 길다"}
    assert rec["feedback"]["rating"] == 4
    assert "saved_at" in rec


def test_turn_clips_are_sliced_by_ms(tmp_path):
    fs.write_session(_bundle(), tmp_path)
    d = tmp_path / "DEMO-20261121-101500-ab12"
    _, _, c1 = _read_wav(d / "turns/01_customer.wav")
    assert len(c1) == (3500 - 500) * 32
    _, _, full = _read_wav(d / "customer.wav")
    assert c1 == full[500 * 32:3500 * 32]
    _, _, a2 = _read_wav(d / "turns/02_agent.wav")
    assert len(a2) == (9500 - 8400) * 32          # 끝은 트랙 길이(9.5s)로 자른다


def test_mix_is_stereo_customer_left_agent_right(tmp_path):
    b = _bundle(customer=_tone(1000), agent=_silence(500) + _tone(1000, freq=220))
    fs.write_session(b | {"turns": []}, tmp_path)
    d = tmp_path / b["session_id"]
    rate, ch, mix = _read_wav(d / "mix.wav")
    assert (rate, ch) == (RATE, 2)
    s = array.array("h", mix)
    assert len(s) == 2 * 1500 * 16                 # 긴 트랙 길이, 짧은 쪽은 무음으로 채움
    left, right = s[0::2], s[1::2]
    assert list(left[:16000]) == list(array.array("h", _tone(1000)))
    assert all(v == 0 for v in left[16000:])
    assert all(v == 0 for v in right[:8000])


def test_cases_jsonl_testset_shape(tmp_path):
    fs.write_session(_bundle(), tmp_path)
    d = tmp_path / "DEMO-20261121-101500-ab12"
    cases = [json.loads(x) for x in (d / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [c["case_id"] for c in cases] == ["DEMO-20261121-101500-ab12-t01", "DEMO-20261121-101500-ab12-t02"]
    c1, c2 = cases
    assert c1["now"] == "2026-11-20 14:00"
    assert c1["customer_profile"]["phone"] == "01072641185"
    assert c1["history"] == [{"role": "customer", "text": "배송 문의요."}, {"role": "agent", "text": "네, 말씀하세요."}]
    assert c1["customer_utterance"] == "주문 어디쯤 왔어요" and c1["utterance_source"] == "stt"
    assert c1["recording"] == {"file": "turns/01_customer.wav", "speech_end": 2.5, "duration": 3.0}
    # 2턴: 이전 턴까지 누적, 정정한 말이 있으면 그것을 쓴다
    assert c2["history"][-2:] == [{"role": "customer", "text": "주문 어디쯤 왔어요"},
                                  {"role": "agent", "text": "확인해 보니 배송 중입니다."}]
    assert c2["customer_utterance"] == "언제 와요, 그럼?" and c2["utterance_source"] == "corrected"
    assert c2["agent_reply"] == "내일 도착 예정입니다."
    assert c2["feedback"]["tags"] == ["slow"]
    assert c2["session_rating"] == 4


def test_cases_history_uses_corrected_text_of_previous_turns(tmp_path):
    b = _bundle()
    b["turns"][0]["user_corrected"] = "주문 어디쯤 왔어요?"
    fs.write_session(b, tmp_path)
    lines = (tmp_path / b["session_id"] / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[1])["history"][-2]["text"] == "주문 어디쯤 왔어요?"


def test_turn_without_clip_has_no_file(tmp_path):
    b = _bundle()
    b["turns"][1]["audio"] = {"customer": None, "agent": None}
    fs.write_session(b, tmp_path)
    d = tmp_path / b["session_id"]
    rec = json.loads((d / "session.json").read_text(encoding="utf-8"))
    assert rec["turns"][1]["audio"] == {"customer": None, "agent": None}
    assert not (d / "turns/02_customer.wav").exists()
    case2 = json.loads((d / "cases.jsonl").read_text(encoding="utf-8").splitlines()[1])
    assert case2["recording"] is None


def test_resave_overwrites_and_removes_stale_clips(tmp_path):
    fs.write_session(_bundle(), tmp_path)
    b = _bundle()
    b["turns"] = b["turns"][:1]
    b["feedback"]["rating"] = 2
    fs.write_session(b, tmp_path)
    d = tmp_path / b["session_id"]
    assert not (d / "turns/02_customer.wav").exists()
    assert json.loads((d / "session.json").read_text(encoding="utf-8"))["feedback"]["rating"] == 2


def test_audio_optional(tmp_path):
    b = _bundle()
    del b["audio"]
    fs.write_session(b, tmp_path)
    d = tmp_path / b["session_id"]
    rec = json.loads((d / "session.json").read_text(encoding="utf-8"))
    assert rec["audio"] is None
    assert not (d / "mix.wav").exists()
    assert rec["turns"][0]["audio"] == {"customer": None, "agent": None}


@pytest.mark.parametrize("sid", ["", "../x", "..", "a/b", "a\\b", ".hidden", "x" * 101, None, 3])
def test_rejects_bad_session_id(tmp_path, sid):
    b = _bundle()
    b["session_id"] = sid
    with pytest.raises(fs.BundleError):
        fs.write_session(b, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_rejects_wrong_schema(tmp_path):
    b = _bundle()
    b["schema"] = "other/v9"
    with pytest.raises(fs.BundleError):
        fs.write_session(b, tmp_path)


def test_rejects_wrong_sample_rate(tmp_path):
    b = _bundle()
    b["audio"]["tracks"]["agent"]["wav_base64"] = _wav_b64(_silence(100), rate=24000)
    with pytest.raises(fs.BundleError, match="16000"):
        fs.write_session(b, tmp_path)


def test_rejects_broken_base64(tmp_path):
    b = _bundle()
    b["audio"]["tracks"]["customer"]["wav_base64"] = "!!!not base64"
    with pytest.raises(fs.BundleError):
        fs.write_session(b, tmp_path)


def test_rejects_unknown_track(tmp_path):
    b = _bundle()
    b["audio"]["tracks"]["robot"] = b["audio"]["tracks"]["agent"]
    with pytest.raises(fs.BundleError):
        fs.write_session(b, tmp_path)


def test_index_line_appended(tmp_path):
    fs.write_session(_bundle(), tmp_path)
    fs.write_session(_bundle(sid="DEMO-2"), tmp_path)
    rows = [json.loads(x) for x in (tmp_path / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["session_id"] for r in rows] == ["DEMO-20261121-101500-ab12", "DEMO-2"]
    assert rows[0]["rating"] == 4 and rows[0]["turns"] == 2 and rows[0]["down_votes"] == 1


def test_cli_unpacks_downloaded_bundle(tmp_path):
    src = tmp_path / "dl" / "DEMO-x.json"
    src.parent.mkdir()
    src.write_text(json.dumps(_bundle(sid="DEMO-x"), ensure_ascii=False), encoding="utf-8")
    fs.main([str(src), "--out", str(tmp_path / "out")])
    assert (tmp_path / "out" / "DEMO-x" / "mix.wav").is_file()


# ── 서버 POST /api/sessions ────────────────────────────
@pytest.fixture
def demo(tmp_path):
    srv = demo_server.create_server(port=0, mock_url="http://127.0.0.1:9", agent_ws="ws://127.0.0.1:9999",
                                    feedback_dir=tmp_path / "fb", sessions_dir=tmp_path / "sessions",
                                    date_fn=lambda: "20261121")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", tmp_path / "sessions"
    srv.shutdown()
    srv.server_close()


def _post(url, body: bytes):
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_post_sessions_saves_bundle(demo):
    base, sdir = demo
    code, res = _post(base + "/api/sessions", json.dumps(_bundle()).encode())
    assert code == 200 and res["ok"] is True
    d = sdir / "DEMO-20261121-101500-ab12"
    assert res["dir"] == str(d)
    assert "turns/01_customer.wav" in res["files"]
    assert (d / "mix.wav").is_file()


def test_post_sessions_rejects_invalid(demo):
    base, sdir = demo
    b = _bundle()
    b["session_id"] = "../evil"
    code, res = _post(base + "/api/sessions", json.dumps(b).encode())
    assert code == 400 and "session_id" in res["error"]
    code, _ = _post(base + "/api/sessions", b"not json")
    assert code == 400
    assert not sdir.exists() or list(sdir.iterdir()) == []


def test_config_reports_sessions_dir(demo):
    base, sdir = demo
    with urllib.request.urlopen(base + "/api/config", timeout=5) as r:
        cfg = json.loads(r.read())
    assert cfg["sessions_dir"] == str(sdir)
