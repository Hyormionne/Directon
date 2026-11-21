"""AG-T52·T53 브라우저 녹음 모듈(feedback_kit/static/session-record.js)을 node로 시험한다. node가 없으면 건너뛴다."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "feedback_kit" / "static" / "session-record.js"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node 없음")

SCRIPT = r"""
const R = require(process.argv[1]);
const out = {};
const frame = (v, n = 320) => { const a = new Int16Array(n); a.fill(v); return a; };

// 1) 고객 트랙: 보낸 프레임을 이어 붙인다
const rec = new R.Recorder();
rec.addMic(frame(100)); rec.addMic(frame(0)); rec.addMic(frame(-5));
const mic = rec.micTrack();
out.micLen = mic.length; out.micFirst = mic[0]; out.micLast = mic[mic.length - 1];
out.micMs = rec.micMs();

// 2) 상담원 트랙: 재생 위치에 놓고, 끼어들기 지점 뒤는 지운다
rec.addAgent(800, frame(7, 400));      // 50ms ~ 75ms
rec.addAgent(1600, frame(9, 800));     // 100ms ~ 150ms
rec.cut(1800);                         // 112.5ms에서 끊김
const ag = rec.agentTrack();
out.agLen = ag.length;
out.ag = [ag[799], ag[800], ag[1199], ag[1200], ag[1600], ag[1799]];
out.durationMs = rec.durationMs();

// 3) 음수 위치(시작 전 예약)는 앞을 잘라 0부터
const r2 = new R.Recorder();
r2.addAgent(-10, frame(3, 20));
out.negLen = r2.agentTrack().length; out.neg0 = r2.agentTrack()[0];

// 4) WAV 헤더
const wav = R.wavBytes(frame(1, 16000), 1);
const dv = new DataView(wav.buffer);
out.wav = { len: wav.length, riff: String.fromCharCode(...wav.slice(0, 4)), rate: dv.getUint32(24, true),
            ch: dv.getUint16(22, true), bits: dv.getUint16(34, true), data: dv.getUint32(40, true) };
const st = R.wavBytes(R.interleave(frame(1, 3), frame(2, 5)), 2);
out.stereoData = new DataView(st.buffer).getUint32(40, true);
out.inter = Array.from(R.interleave(frame(1, 2), frame(2, 3)));

// 5) 구간 자르기(ms)
const tr = new Int16Array(16000); for (let i = 0; i < tr.length; i++) tr[i] = i % 1000;
const s = R.sliceMs(tr, 100, 200);
out.slice = [s.length, s[0], s[s.length - 1]];
out.sliceClamp = R.sliceMs(tr, 900, 5000).length;

// 6) 턴 구간 계산
out.clips = R.turnClips([
  { ts: { speech_start_ms: 1000, decided_ms: 3300, turn_end_ms: 5400, agent_audio_end_ms: 5500 }, decided: { silence_ms: 300 } },
  { ts: { speech_start_ms: 3400, decided_ms: 8400, turn_end_ms: 9400, agent_audio_end_ms: null }, decided: { silence_ms: 400 } },
  { ts: { speech_start_ms: null, decided_ms: null, turn_end_ms: null, agent_audio_end_ms: null }, decided: null },
  { ts: { decided_ms: null, first_sound_ms: 200, turn_end_ms: 1500, agent_audio_end_ms: 1800 }, decided: null },
], 9600);

// 9) 말 뒤 공백이 긴 턴(클라이언트 판정: 판정 시각 = 상담원 응답 시작)은 고객 구간 끝을 말 끝 + 800ms로
out.trail = R.turnClips([
  { ts: { speech_start_ms: 1000, decided_ms: 5000, turn_end_ms: 6000, agent_audio_end_ms: 6000 }, decided: { silence_ms: 2500 } },
], 9000)[0];

// 10) 클라이언트 말 구간 추적: 유성 프레임의 첫 시작 ~ 마지막 끝
const sp = new R.SpeechTracker();
out.spEmpty = sp.take();
sp.frame(0, false); sp.frame(20, true); sp.frame(40, true); sp.frame(60, false); sp.frame(500, true); sp.frame(520, false);
out.sp = sp.take();
out.spAfter = sp.take();
sp.frame(1000, false);
out.spSilent = sp.take();

// 8) WAV 왕복, 다른 형식 거부
const back = R.wavToInt16(R.wavBytes(Int16Array.from([1, -2, 32767, -32768]), 1));
out.roundtrip = Array.from(back);
const b64back = R.wavToInt16(R.fromBase64(R.toBase64(R.wavBytes(Int16Array.from([5, 6]), 1))));
out.b64roundtrip = Array.from(b64back);
try { R.wavToInt16(R.wavBytes(Int16Array.from([1, 2]), 2)); out.stereoErr = null; } catch (e) { out.stereoErr = e.message; }
try { R.wavToInt16(new Uint8Array(10)); out.junkErr = null; } catch (e) { out.junkErr = e.message; }

// 7) base64
out.b64 = R.toBase64(new Uint8Array([0, 1, 2, 250, 255]));
out.b64big = R.toBase64(new Uint8Array(100000)).length;
console.log(JSON.stringify(out));
"""


def _run():
    r = subprocess.run([NODE, "-e", SCRIPT, str(MODULE)], capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_recorder_tracks():
    o = _run()
    assert (o["micLen"], o["micFirst"], o["micLast"]) == (960, 100, -5)
    assert o["micMs"] == 60
    assert o["agLen"] == 1800                      # 끊긴 지점까지만
    assert o["ag"] == [0, 7, 7, 0, 9, 9]
    assert o["durationMs"] == 113                  # 긴 트랙 기준(1800샘플 = 112.5ms 올림)
    assert (o["negLen"], o["neg0"]) == (10, 3)


def test_wav_and_interleave():
    o = _run()
    assert o["wav"] == {"len": 44 + 32000, "riff": "RIFF", "rate": 16000, "ch": 1, "bits": 16, "data": 32000}
    assert o["stereoData"] == 5 * 2 * 2           # 긴 쪽 5샘플 × 2채널 × 2바이트
    assert o["inter"] == [1, 2, 1, 2, 0, 2]


def test_slice_ms():
    o = _run()
    assert o["slice"] == [1600, 600, 199]
    assert o["sliceClamp"] == 1600


def test_turn_clips():
    c1, c2, c3, c4 = _run()["clips"]
    # 고객: 말 시작 500ms 앞(앞 턴 판정 시각보다 앞으로는 안 감) ~ 판정 + 200ms, 말 끝 = 판정 - 침묵
    assert c1["customer"] == {"start_ms": 500, "end_ms": 3500, "speech_end_ms": 2500}
    # 상담원: 판정 ~ 마지막 재생 끝 + 200ms
    assert c1["agent"] == {"start_ms": 3300, "end_ms": 5700}
    assert c2["customer"] == {"start_ms": 3300, "end_ms": 8600, "speech_end_ms": 4700}
    # 재생 끝을 모르면 turn.end + 200, 전체 길이로 자른다
    assert c2["agent"] == {"start_ms": 8400, "end_ms": 9600}
    assert c3 == {"customer": None, "agent": None}
    # 고객 말 없는 상담원 선발화: 첫 소리부터
    assert c4 == {"customer": None, "agent": {"start_ms": 200, "end_ms": 2000}}


def test_turn_clip_caps_long_trailing_gap():
    t = _run()["trail"]
    # 말 끝 2500ms → 고객 구간 끝 = 2500 + 800 + 200
    assert t["customer"] == {"start_ms": 500, "end_ms": 3500, "speech_end_ms": 2000}
    assert t["agent"] == {"start_ms": 5000, "end_ms": 6200}


def test_speech_tracker():
    o = _run()
    assert o["spEmpty"] is None
    assert o["sp"] == {"speech_start_ms": 20, "speech_end_ms": 520}
    assert o["spAfter"] is None
    assert o["spSilent"] is None


def test_wav_to_int16():
    o = _run()
    assert o["roundtrip"] == [1, -2, 32767, -32768]
    assert o["b64roundtrip"] == [5, 6]
    assert o["stereoErr"] and "16kHz" in o["stereoErr"]
    assert o["junkErr"]


def test_base64():
    o = _run()
    assert o["b64"] == "AAEC+v8="
    assert o["b64big"] == 133336
