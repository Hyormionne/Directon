"""통화 묶음(bundle) 저장 — feedback_kit의 원본(AG-T52, AG-T53). 표준 라이브러리만 쓴다.

bundle은 체험 페이지(feedback_kit 또는 Directon demo)가 통화 하나를 담아 보내거나 내려받는 JSON 한 개다(schema "directon.demo_session/v1").

    {
      "schema": "directon.demo_session/v1",
      "session_id": "DEMO-...", "started_at": ISO, "ended_at": ISO,
      "setup": {"caller", "now", "history_preset", "history", "half_duplex", "adaptive_opener", ...},
      "feedback": {"rating": 1~5|null, "comment": str},
      "turns": [{"index", "user_stt", "user_corrected", "agent", "tools", "decided", "timing",
                 "client_latency_ms", "interrupted",
                 "ts": {"speech_start_ms", "decided_ms", "first_sound_ms", "turn_end_ms", "agent_audio_end_ms"},
                 "audio": {"customer": {"start_ms", "end_ms", "speech_end_ms"} | null,
                           "agent": {"start_ms", "end_ms"} | null},
                 "feedback": {"vote": "up"|"down"|null, "tags": [...], "note": str}}],
      "audio": {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1,
                "tracks": {"customer": {"wav_base64"}, "agent": {"wav_base64"}}} | 없음
    }

시각(ms)은 모두 통화 연결 순간(오디오 시계)부터 잰다. customer 트랙은 상담원에게 실제로 보낸 음성
(반이중으로 막힌 구간은 무음), agent 트랙은 고객이 실제로 들은 상담원 음성(끼어들기로 끊긴 뒤는 무음)이다.

write_session(bundle, root) → root/<session_id>/
    session.json        bundle에서 base64를 파일 경로로 바꾼 것(+ saved_at)
    customer.wav        16kHz mono
    agent.wav           16kHz mono
    mix.wav             16kHz 스테레오, 왼쪽 고객·오른쪽 상담원
    turns/NN_customer.wav, turns/NN_agent.wav
    cases.jsonl         턴마다 테스트셋(haruon_cs_testset) 모양 문항 한 줄
  그리고 root/index.jsonl에 요약 한 줄을 더한다.

CLI(내려받은 bundle 풀기): python -m feedback_kit.store <bundle.json>... [--out DIR]  (기본 ./feedback_sessions)
"""
from __future__ import annotations

import argparse
import array
import base64
import binascii
import io
import json
import re
import sys
import threading
import wave
from datetime import datetime
from pathlib import Path

SCHEMA = "directon.demo_session/v1"
RATE = 16000
TRACKS = ("customer", "agent")
DEFAULT_ROOT = Path("feedback_sessions")   # 실행한 폴더 기준
_SID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_LOCK = threading.Lock()


class BundleError(ValueError):
    """bundle 형식이 올바르지 않다(서버는 400으로 돌려준다)."""


# ── 오디오 도우미 ─────────────────────────────────────
def decode_wav(b64: str, track: str) -> bytes:
    """WAV base64 → PCM s16le. 16kHz mono 16bit만 받는다."""
    if not isinstance(b64, str):
        raise BundleError(f"audio.tracks.{track}.wav_base64가 문자열이 아닙니다")
    try:
        raw = base64.b64decode(b64, validate=True)
        with wave.open(io.BytesIO(raw), "rb") as w:
            rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            pcm = w.readframes(w.getnframes())
    except (binascii.Error, ValueError, wave.Error, EOFError) as e:
        raise BundleError(f"audio.tracks.{track}: WAV를 읽을 수 없습니다({e})") from None
    if (rate, ch, width) != (RATE, 1, 2):
        raise BundleError(f"audio.tracks.{track}: 16000Hz mono 16bit여야 합니다(지금 {rate}Hz {ch}ch {width * 8}bit)")
    return pcm


def wav_bytes(pcm: bytes, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def slice_ms(pcm: bytes, start_ms: float, end_ms: float) -> bytes:
    n = len(pcm) // 2
    a = max(0, min(n, round(start_ms * RATE / 1000)))
    b = max(a, min(n, round(end_ms * RATE / 1000)))
    return pcm[a * 2:b * 2]


def interleave(left: bytes, right: bytes) -> bytes:
    """mono 둘 → 스테레오. 짧은 쪽은 무음으로 채운다."""
    n = max(len(left), len(right)) // 2
    out = array.array("h", bytes(n * 4))
    out[0::2] = array.array("h", left[: n * 2].ljust(n * 2, b"\0"))
    out[1::2] = array.array("h", right[: n * 2].ljust(n * 2, b"\0"))
    return out.tobytes()


def _ms(pcm: bytes) -> int:
    return round(len(pcm) / 2 * 1000 / RATE)


# ── 검사 ──────────────────────────────────────────────
def _validate(bundle) -> str:
    if not isinstance(bundle, dict):
        raise BundleError("bundle은 JSON 객체여야 합니다")
    if bundle.get("schema") != SCHEMA:
        raise BundleError(f"schema가 {SCHEMA}가 아닙니다: {bundle.get('schema')!r}")
    sid = bundle.get("session_id")
    if not isinstance(sid, str) or not _SID_RE.match(sid):
        raise BundleError(f"session_id가 올바르지 않습니다(영문·숫자·._-, 100자 이하): {sid!r}")
    if not isinstance(bundle.get("turns", []), list):
        raise BundleError("turns는 배열이어야 합니다")
    audio = bundle.get("audio")
    if audio is not None:
        tracks = audio.get("tracks") if isinstance(audio, dict) else None
        if not isinstance(tracks, dict):
            raise BundleError("audio.tracks가 없습니다")
        unknown = set(tracks) - set(TRACKS)
        if unknown:
            raise BundleError(f"모르는 오디오 트랙: {sorted(unknown)}")
    return sid


def _clip(spec) -> dict | None:
    if not isinstance(spec, dict):
        return None
    try:
        s, e = float(spec["start_ms"]), float(spec["end_ms"])
    except (KeyError, TypeError, ValueError):
        return None
    return dict(spec) if e > s else None


# ── 테스트셋 모양 문항 ─────────────────────────────────
def _said(turn: dict) -> tuple[str, str]:
    fixed = str(turn.get("user_corrected") or "").strip()
    if fixed:
        return fixed, "corrected"
    return str(turn.get("user_stt") or "").strip(), "stt"


def build_cases(record: dict) -> list[dict]:
    """턴마다 haruon_cs_testset metadata 모양 문항. history는 주입한 이전 대화 + 앞 턴들(정정한 말 우선)."""
    setup = record.get("setup") or {}
    history = [dict(h) for h in setup.get("history") or []]
    sid = record["session_id"]
    rating = (record.get("feedback") or {}).get("rating")
    out = []
    for i, t in enumerate(record.get("turns") or [], 1):
        said, source = _said(t)
        cust = (t.get("audio") or {}).get("customer")
        recording = None
        if cust and cust.get("file"):
            recording = {"file": cust["file"],
                         "speech_end": round(float(cust.get("speech_end_ms") or 0) / 1000, 3),
                         "duration": round((float(cust["end_ms"]) - float(cust["start_ms"])) / 1000, 3)}
        out.append({
            "case_id": f"{sid}-t{i:02d}",
            "session_id": sid,
            "turn_index": t.get("index", i),
            "now": setup.get("now"),
            "history": [dict(h) for h in history],
            "customer_utterance": said,
            "utterance_source": source,
            "customer_profile": setup.get("caller"),
            "recording": recording,
            "agent_reply": t.get("agent", ""),
            "tools": t.get("tools") or [],
            "feedback": t.get("feedback") or {},
            "session_rating": rating,
        })
        if said:
            history.append({"role": "customer", "text": said})
        if t.get("agent"):
            history.append({"role": "agent", "text": t["agent"]})
    return out


# ── 저장 ──────────────────────────────────────────────
def write_session(bundle: dict, root: Path | str = DEFAULT_ROOT, *, saved_at: str | None = None) -> dict:
    """bundle을 root/<session_id>/로 푼다. 같은 session_id면 덮어쓴다(평가를 고쳐 다시 저장)."""
    sid = _validate(bundle)
    pcm: dict[str, bytes] = {}
    audio = bundle.get("audio")
    if audio is not None:
        for name, tr in audio["tracks"].items():
            pcm[name] = decode_wav((tr or {}).get("wav_base64"), name)

    record = {k: v for k, v in bundle.items() if k != "audio"}
    record["saved_at"] = saved_at or datetime.now().isoformat(timespec="seconds")
    files: dict[str, bytes] = {}
    if pcm:
        tracks_out = {}
        for name in TRACKS:
            if name in pcm:
                files[f"{name}.wav"] = wav_bytes(pcm[name])
                tracks_out[name] = {"file": f"{name}.wav", "duration_ms": _ms(pcm[name])}
        record["audio"] = {"encoding": "pcm_s16le", "sample_rate": RATE, "channels": 1, "tracks": tracks_out}
        if "customer" in pcm and "agent" in pcm:
            files["mix.wav"] = wav_bytes(interleave(pcm["customer"], pcm["agent"]), channels=2)
            record["audio"]["mix"] = {"file": "mix.wav", "channels": 2, "left": "customer", "right": "agent"}
    else:
        record["audio"] = None

    turns = []
    for i, t in enumerate(bundle.get("turns") or [], 1):
        t = dict(t) if isinstance(t, dict) else {}
        clips = t.get("audio") if isinstance(t.get("audio"), dict) else {}
        new_audio = {}
        for name in TRACKS:
            spec = _clip(clips.get(name))
            if spec is None or name not in pcm:
                new_audio[name] = None
                continue
            rel = f"turns/{i:02d}_{name}.wav"
            files[rel] = wav_bytes(slice_ms(pcm[name], spec["start_ms"], spec["end_ms"]))
            spec["file"] = rel
            new_audio[name] = spec
        t["audio"] = new_audio
        turns.append(t)
    record["turns"] = turns

    cases = build_cases(record)
    root = Path(root)
    d = root / sid
    with _LOCK:
        d.mkdir(parents=True, exist_ok=True)
        tdir = d / "turns"
        if tdir.is_dir():
            for old in tdir.glob("*.wav"):
                old.unlink()
        for name in ("mix.wav", *(f"{n}.wav" for n in TRACKS)):
            if name not in files and (d / name).exists():
                (d / name).unlink()
        for rel, data in files.items():
            p = d / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        (d / "session.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        (d / "cases.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases),
                                       encoding="utf-8")
        fb = record.get("feedback") or {}
        votes = [(t.get("feedback") or {}).get("vote") for t in turns]
        idx = {"session_id": sid, "dir": str(d), "saved_at": record["saved_at"],
               "rating": fb.get("rating"), "turns": len(turns),
               "up_votes": votes.count("up"), "down_votes": votes.count("down"),
               "duration_ms": max((_ms(p) for p in pcm.values()), default=None)}
        with (root / "index.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(idx, ensure_ascii=False) + "\n")
    listed = sorted(files) + ["session.json", "cases.jsonl"]
    return {"session_id": sid, "dir": str(d), "files": listed}


def main(argv: list[str] | None = None, *, default_out: Path | str = DEFAULT_ROOT) -> int:
    ap = argparse.ArgumentParser(description="내려받은 데모 통화 묶음(JSON)을 세션 폴더로 풀기")
    ap.add_argument("bundles", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path(default_out))
    a = ap.parse_args(argv)
    bad = 0
    for p in a.bundles:
        try:
            res = write_session(json.loads(p.read_text(encoding="utf-8")), a.out)
        except (OSError, ValueError) as e:
            print(f"{p}: 실패 — {e}", file=sys.stderr)
            bad += 1
            continue
        print(f"{p} → {res['dir']} ({len(res['files'])}개 파일)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
