"""오프라인 턴 감지 시뮬레이터 (AG-T12).

오디오를 20ms 프레임으로 재생하며 agent/session.py `_on_frame`과 같은 로직으로 TurnSignals를 만들고
결정기를 매 프레임 부른다.
  - VAD 확률 ≥ vad_threshold(0.5)면 음성 프레임: speech_ms += 20, silence_ms = 0(준비 상태 취소)
  - 누적 음성 < min_speech_ms(60)인 상태의 무음: 침묵이 precall(200ms)에 닿으면 잡음으로 보고 잊는다
  - 침묵 precall(200ms)이 되는 프레임에 Smart Turn을 최근 8초(그 프레임 포함 400프레임)로 1회 계산
    (smart_delay_frames 뒤부터 결정기에 보인다. 세션에서는 스레드 추론 25ms 안팎)
  - 텍스트: 그 침묵이 시작된 위치 직전의 라벨 지점(쉼 시작·최종 끝)의 STT 텍스트
    (text_lag_ms>0이면 침묵이 그만큼 지나기 전에는 한 지점 앞 텍스트 → 스트리밍 STT 지연 근사)
판정: 첫 end 시각(프레임 끝)이 최종 음성 끝보다 앞이면 false cut(쉼 구간이면 그 쉼 유형, 아니면 intra),
      뒤면 지연 = 판정 시각 − 음성 끝. 끝까지 판정이 없으면 miss.

비싼 계산(VAD 프레임 확률, Smart Turn 확률)은 extract_features로 오디오별 한 번 계산해 캐시한다
(data/turnset/audio/feats/<uid>.npz). simulate_fast는 침묵 구간마다 결정기의 wait_ms를 두세 번만
불러 같은 결과를 낸다(파라미터 탐색용, 테스트로 simulate와 일치 확인).

CLI:
  uv run python -m eval.turn_sim features            # 특징 캐시 생성(VAD·Smart Turn)
  uv run python -m eval.turn_sim eval [--split test] # 기본 결정기(make_detector) 지표
"""
from __future__ import annotations

import argparse
import json
import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from agent.core.types import FRAME_MS, SAMPLE_RATE, TurnSignals

ROOT = Path(__file__).resolve().parents[1]
TURNSET = ROOT / "data" / "turnset"
LABELS = TURNSET / "labels.jsonl"
AUDIO_DIR = TURNSET / "audio"
FEATS_DIR = AUDIO_DIR / "feats"
FRAME_S = FRAME_MS / 1000
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
SMART_WINDOW_FRAMES = 8000 // FRAME_MS          # Settings.smart_turn_window_s = 8.0
POINT_TOL_S = 0.15                              # 침묵 시작과 라벨 지점의 허용 차(VAD 지연)
PAUSE_TOL_S = 0.10                              # 쉼 구간 판정 여유
FEATS_VERSION = 1


@dataclass(frozen=True)
class SessionParams:
    vad_threshold: float = 0.5
    min_speech_ms: int = 60
    precall_ms: int = 200
    smart_delay_frames: int = 0
    text_lag_ms: int = 0
    expect_short: bool = False
    # 확정 시점 STT 부분 텍스트가 비는 비율(실서버 스모크 7/20). 침묵 구간(run)마다 결정적 해시로 고른다
    empty_text_rate: float = 0.0


@dataclass
class SimPoint:
    t: float                 # 초. 쉼 시작 또는 최종 음성 끝(깨끗한 트랙 기준)
    kind: str                # "pause" | "end"
    cut_type: str = ""       # 쉼 유형(connective/particle/digits/filler/trap)
    dur_ms: int = 0          # 쉼 길이
    text: str = ""           # 이 지점까지의 STT 텍스트


@dataclass
class SimItem:
    uid: str
    probs: np.ndarray                    # 프레임별 VAD 확률
    points: list[SimPoint]               # 시각순, 마지막은 kind="end"
    smart: dict[int, float] = field(default_factory=dict)       # 준비 프레임 → Smart Turn p
    smart_fn: Callable[[int], float] | None = None              # 캐시에 없을 때(준비 프레임 → p)
    meta: dict = field(default_factory=dict)
    _runs: dict = field(default_factory=dict, repr=False)

    @property
    def end_t(self) -> float:
        return self.points[-1].t

    def smart_p(self, frame: int) -> float | None:
        if frame in self.smart:
            return self.smart[frame]
        if self.smart_fn is None:
            return None
        p = float(self.smart_fn(frame))
        self.smart[frame] = p
        return p


@dataclass
class SimResult:
    uid: str
    end_frame: int | None
    decision_t: float | None
    outcome: str                         # ok | false_cut | miss
    latency_ms: float | None = None
    where: str = ""                      # false cut 위치: 쉼 유형 | intra | early_end
    hard_max: bool = False
    reason: str = ""
    silence_ms: int = 0


# ───────────────────────── 텍스트 ─────────────────────────
def _point_index(item: SimItem, silence_start_t: float) -> int:
    idx = -1
    for k, p in enumerate(item.points):
        if p.t <= silence_start_t + POINT_TOL_S:
            idx = k
        else:
            break
    return idx


def text_empty(item: SimItem, start: int, params: SessionParams) -> bool:
    if params.empty_text_rate <= 0:
        return False
    if params.empty_text_rate >= 1:
        return True
    return (zlib.crc32(f"{item.uid}:{start}".encode()) % 10000) / 10000 < params.empty_text_rate


def text_at(item: SimItem, frame: int, silence_ms: int, params: SessionParams = SessionParams()) -> str:
    """프레임 frame(침묵 silence_ms째)에 세션의 stt.latest_text()가 줄 텍스트 근사."""
    start = frame + 1 - silence_ms // FRAME_MS          # 첫 침묵 프레임
    if text_empty(item, start, params):
        return ""
    k = _point_index(item, start * FRAME_S)
    if params.text_lag_ms and silence_ms < params.text_lag_ms:
        k -= 1
    return item.points[k].text if k >= 0 else ""


# ───────────────────────── 판정 결과 분류 ─────────────────────────
def classify(item: SimItem, frame: int | None, reason: str = "", hard_max: bool = False,
             silence_ms: int = 0) -> SimResult:
    if frame is None:
        return SimResult(item.uid, None, None, "miss")
    t = (frame + 1) * FRAME_S
    end_t = item.end_t
    if t + 1e-9 >= end_t:
        return SimResult(item.uid, frame, t, "ok", latency_ms=round((t - end_t) * 1000, 3),
                         hard_max=hard_max, reason=reason, silence_ms=silence_ms)
    where = "intra"
    for p in item.points:
        if p.kind == "pause" and p.t - PAUSE_TOL_S <= t <= p.t + p.dur_ms / 1000 + PAUSE_TOL_S:
            where = p.cut_type or "pause"
            break
    else:
        if end_t - t <= 0.3 and all(p.t < t for p in item.points[:-1]):
            where = "early_end"
    return SimResult(item.uid, frame, t, "false_cut", where=where, hard_max=hard_max, reason=reason,
                     silence_ms=silence_ms)


def _hard_cfg(detector: Any) -> int | None:
    c = getattr(detector, "config", None)
    return getattr(c, "hard_max_ms", None)


# ───────────────────────── 프레임 단위 시뮬레이션 ─────────────────────────
def simulate(item: SimItem, detector: Any, params: SessionParams = SessionParams()) -> SimResult:
    """세션과 같은 순서로 매 프레임 detector.decide를 부른다."""
    speech = silence = 0
    prepared = False
    smart: float | None = None
    prep_frame = -1
    for i, p in enumerate(item.probs):
        if p >= params.vad_threshold:
            if silence > 0 and prepared:
                prepared, smart = False, None
            speech += FRAME_MS
            silence = 0
            continue
        if speech < params.min_speech_ms:
            if speech and silence + FRAME_MS >= params.precall_ms:
                speech = silence = 0
            elif speech:
                silence += FRAME_MS
            continue
        silence += FRAME_MS
        if not prepared and silence >= params.precall_ms:
            prepared, prep_frame = True, i
        if prepared and smart is None and i >= prep_frame + params.smart_delay_frames:
            smart = item.smart_p(prep_frame)
        text = text_at(item, i, silence, params)
        d = detector.decide(TurnSignals(silence_ms=silence, speech_ms=speech, text=text,
                                        smart_turn_p=smart, expect_short=params.expect_short))
        if d.end:
            return classify(item, i, d.reason, hard_max="hard_max" in d.reason, silence_ms=silence)
    return classify(item, None)


# ───────────────────────── 침묵 구간(run) 단위 빠른 시뮬레이션 ─────────────────────────
@dataclass
class Run:
    start: int              # 첫 침묵 프레임(결정기가 불리는 프레임)
    n: int                  # 이어지는 침묵 프레임 수
    speech_ms: int          # 이 침묵 동안 고정된 누적 음성 길이
    prep: int | None        # 준비(Smart Turn) 프레임, 없으면 None


def runs(item: SimItem, params: SessionParams = SessionParams()) -> list[Run]:
    key = (params.vad_threshold, params.min_speech_ms, params.precall_ms)
    if key in item._runs:
        return item._runs[key]
    out: list[Run] = []
    speech = silence = 0
    cur: Run | None = None
    for i, p in enumerate(item.probs):
        if p >= params.vad_threshold:
            speech += FRAME_MS
            silence = 0
            cur = None
            continue
        if speech < params.min_speech_ms:
            if speech and silence + FRAME_MS >= params.precall_ms:
                speech = silence = 0
            elif speech:
                silence += FRAME_MS
            continue
        silence += FRAME_MS
        if cur is None:
            cur = Run(start=i, n=0, speech_ms=speech, prep=None)
            out.append(cur)
        cur.n += 1
        if cur.prep is None and silence >= params.precall_ms:
            cur.prep = i
    item._runs[key] = out
    return out


def _ceil_frame(ms: float) -> int:
    return int(math.ceil(ms / FRAME_MS - 1e-9)) * FRAME_MS


def simulate_fast(item: SimItem, detector: Any, params: SessionParams = SessionParams()) -> SimResult:
    """wait_ms(signals)가 침묵 길이에 의존하지 않는 결정기용. simulate와 같은 결과."""
    if not hasattr(detector, "wait_ms") or not hasattr(detector, "config"):
        return simulate(item, detector, params)
    c = detector.config
    min_sil, hard, min_speech = c.min_silence_ms, c.hard_max_ms, c.min_speech_ms
    for r in runs(item, params):
        start_t = r.start * FRAME_S              # 침묵 시작 시각(첫 침묵 프레임 시작)
        k0 = _point_index(item, start_t)
        # 상태가 바뀌는 침묵 길이(ms): smart 도착, 텍스트 지연 해소
        bps = {FRAME_MS}
        p_from = None
        if r.prep is not None:
            p_from = (r.prep - r.start + 1 + params.smart_delay_frames) * FRAME_MS
            bps.add(p_from)
        if params.text_lag_ms:
            bps.add(_ceil_frame(params.text_lag_ms))
        bps = sorted(b for b in bps if b <= r.n * FRAME_MS)
        limit = r.n * FRAME_MS
        for j, a in enumerate(bps):
            b = bps[j + 1] if j + 1 < len(bps) else limit + FRAME_MS
            k = k0 - 1 if (params.text_lag_ms and a < params.text_lag_ms) else k0
            text = "" if text_empty(item, r.start, params) else (item.points[k].text if k >= 0 else "")
            smart = item.smart_p(r.prep) if (p_from is not None and a >= p_from) else None
            if r.speech_ms < min_speech and not text.strip():
                continue
            sig = TurnSignals(silence_ms=a, speech_ms=r.speech_ms, text=text, smart_turn_p=smart,
                              expect_short=params.expect_short)
            w, why = detector.wait_ms(sig)
            cap = detector.cap_ms(sig) if hasattr(detector, "cap_ms") else hard
            th = max(min_sil, min(cap, w))
            s = max(a, _ceil_frame(th))
            if s < b and s <= limit:
                frame = r.start + s // FRAME_MS - 1
                return classify(item, frame, why, hard_max=s >= cap, silence_ms=s)
    return classify(item, None)


# ───────────────────────── 지표 ─────────────────────────
def _lat_stats(lat: list[float]) -> dict:
    if not lat:
        return {"latency_mean": None, "latency_p50": None, "latency_p95": None}
    a = np.asarray(lat, float)
    return {"latency_mean": float(a.mean()), "latency_p50": float(np.percentile(a, 50)),
            "latency_p95": float(np.percentile(a, 95))}


def _summary(rs: list[SimResult]) -> dict:
    n = len(rs)
    ok = [r for r in rs if r.outcome == "ok"]
    fc = [r for r in rs if r.outcome == "false_cut"]
    d = {"n": n,
         "false_cut": len(fc),
         "false_cut_rate": len(fc) / n if n else 0.0,
         "miss_rate": sum(r.outcome == "miss" for r in rs) / n if n else 0.0,
         "hard_max_rate": (sum(r.hard_max for r in ok) / len(ok)) if ok else 0.0,
         "over_1100_rate": (sum((r.latency_ms or 0) > 1100 for r in ok) / len(ok)) if ok else 0.0,
         # 결정 침묵이 1100ms를 넘은 비율(상한 예외로 더 기다린 경우). 지연에는 VAD 늦음(수십 ms)이 더해진다
         "over_cap_rate": (sum(r.silence_ms > 1100 for r in ok) / len(ok)) if ok else 0.0}
    d.update(_lat_stats([r.latency_ms for r in ok]))
    return d


def summarize(results: Iterable[SimResult], meta: dict[str, dict],
              group_keys: tuple[str, ...] = ("type", "channel", "noise")) -> dict:
    rs = list(results)
    out = _summary(rs)
    where: dict[str, int] = {}
    for r in rs:
        if r.outcome == "false_cut":
            where[r.where] = where.get(r.where, 0) + 1
    out["false_cut_where"] = where
    out["by"] = {}
    for g in group_keys:
        groups: dict[str, list[SimResult]] = {}
        for r in rs:
            groups.setdefault(str(meta.get(r.uid, {}).get(g, "?")), []).append(r)
        out["by"][g] = {k: _summary(v) for k, v in sorted(groups.items())}
    return out


def run_all(items: list[SimItem], detector_factory: Callable[[], Any], params: SessionParams = SessionParams(),
            fast: bool = True) -> list[SimResult]:
    sim = simulate_fast if fast else simulate
    return [sim(it, detector_factory(), params) for it in items]


# ───────────────────────── 특징 추출(VAD·Smart Turn) ─────────────────────────
def extract_features(audio: np.ndarray, vad: Any, smart_turn: Any,
                     params: SessionParams = SessionParams()) -> tuple[np.ndarray, dict[int, float]]:
    """오디오 → (프레임별 VAD 확률, 준비 프레임별 Smart Turn p). 세션과 같은 프레임 분할(640B)."""
    a = np.asarray(audio, np.int16)
    n = int(math.ceil(len(a) / FRAME_SAMPLES))
    pad = np.zeros(n * FRAME_SAMPLES, np.int16)
    pad[: len(a)] = a
    if hasattr(vad, "reset"):
        vad.reset()
    probs = np.array([vad.process(pad[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES].tobytes()) for i in range(n)],
                     dtype=np.float32)
    item = SimItem(uid="_", probs=probs, points=[SimPoint(t=0.0, kind="end")])
    smart: dict[int, float] = {}
    for r in runs(item, params):
        if r.prep is None:
            continue
        lo = max(0, r.prep + 1 - SMART_WINDOW_FRAMES) * FRAME_SAMPLES
        hi = (r.prep + 1) * FRAME_SAMPLES
        smart[r.prep] = float(smart_turn.predict(pad[lo:hi].tobytes()))
    return probs, smart


def feats_path(uid: str, feats_dir: Path = FEATS_DIR) -> Path:
    return feats_dir / f"{uid}.npz"


def load_labels(path: Path = LABELS) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def row_meta(row: dict) -> dict:
    env = row.get("env", {})
    return {"type": row.get("type"), "split": row.get("split"), "channel": env.get("channel"),
            "noise": env.get("noise"), "snr_db": env.get("snr_db"), "gain_db": env.get("gain_db"),
            "speed": env.get("speed"), "packet_loss": env.get("packet_loss"), "voice": row.get("voice"),
            "base_voice": row.get("base_voice"), "n_pauses": len(row.get("segments", [])) - 1,
            "noise_snr": "none" if env.get("noise") == "none" else f"snr{env.get('snr_db')}"}


def row_points(row: dict, text_source: str = "stt") -> list[SimPoint]:
    pts = []
    for p in row["points"]:
        text = p.get("text_stt") if text_source == "stt" else None
        if text is None:
            text = p.get("text_gt", "")
        pts.append(SimPoint(t=float(p["t"]), kind=p["kind"], cut_type=p.get("cut_type", ""),
                            dur_ms=int(p.get("dur_ms", 0)), text=text))
    return pts


def load_items(labels: Path = LABELS, feats_dir: Path = FEATS_DIR, split: str | None = None,
               text_source: str = "stt") -> list[SimItem]:
    items = []
    for row in load_labels(labels):
        if split and row.get("split") != split:
            continue
        fp = feats_path(row["uid"], feats_dir)
        if not fp.exists():
            continue
        z = np.load(fp)
        smart = {int(k): float(v) for k, v in zip(z["smart_frames"], z["smart_p"])}
        items.append(SimItem(uid=row["uid"], probs=z["probs"], points=row_points(row, text_source),
                             smart=smart, meta=row_meta(row)))
    return items


def build_features(labels: Path = LABELS, audio_dir: Path = AUDIO_DIR, feats_dir: Path = FEATS_DIR,
                   force: bool = False, workers: int = 4) -> int:
    from concurrent.futures import ThreadPoolExecutor

    from agent import audio as au
    from agent.turn.smart_turn import SmartTurnV3
    from agent.turn.vad import SileroVAD

    feats_dir.mkdir(parents=True, exist_ok=True)
    rows = load_labels(labels)
    todo = []
    for row in rows:
        fp = feats_path(row["uid"], feats_dir)
        wav = audio_dir / f"{row['uid']}.wav"
        if not wav.exists():
            continue
        if fp.exists() and not force:
            z = np.load(fp)
            if int(z["version"]) == FEATS_VERSION and str(z["render_key"]) == row.get("render_key", ""):
                continue
        todo.append(row)
    smart = SmartTurnV3(threads=2)

    def one(row):
        pcm, sr = au.wav_read(audio_dir / f"{row['uid']}.wav")
        assert sr == SAMPLE_RATE
        a = np.frombuffer(pcm, np.int16)
        probs, sm = extract_features(a, SileroVAD(), smart)
        fr = np.array(sorted(sm), dtype=np.int64)
        np.savez(feats_path(row["uid"], feats_dir), probs=probs, smart_frames=fr,
                 smart_p=np.array([sm[int(k)] for k in fr], dtype=np.float32),
                 version=FEATS_VERSION, render_key=row.get("render_key", ""))

    with ThreadPoolExecutor(workers) as ex:
        for k, _ in enumerate(ex.map(one, todo)):
            if (k + 1) % 100 == 0:
                print(f"features {k + 1}/{len(todo)}", flush=True)
    return len(todo)


# ───────────────────────── 172문항 테스트셋(AG-T15b) ─────────────────────────
# 실서버 측정과 같은 오디오(legacy/testset/audio)에 시뮬레이터를 돌린다. 쉼 라벨이 없으므로
#  - 지점 = Silero VAD 침묵 구간(≥ precall 200ms) 중 말 끝(meta speech_end) − 0.2초 전에 시작한 것(말 중간 침묵) + 말 끝
#  - 지점 텍스트 = 그 침묵 시작 + 200ms까지 오디오의 실제 STT(gemini-3.1-flash-lite 받아쓰기, 캐시)
#  - 말 끝 텍스트 = round1 턴 로그의 확정 텍스트(정상 확정 문항) 또는 STT 평가(base_full)의 최종 텍스트(나머지)
#  - expect_short = 문항 history 마지막 상담원 말이 물음표로 끝남(세션과 같음)
TESTSET_DIR = ROOT / "legacy" / "testset"
TS172_DIR = TURNSET / "testset172"
TS172_PROBS = TS172_DIR / "probs"
TS172_STT = TS172_DIR / "stt_points.jsonl"
TS172_END = TS172_DIR / "end_texts.json"
ROUND1_RUN = ROOT / "runs" / "round1_172_2003"
TURN_LOG = ROOT / "runs" / "logs" / "turns-20261119.jsonl"
STT_EVAL_ROWS = ROOT / "runs" / "stt_eval" / "base_full" / "rows.jsonl"


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def ts172_cases() -> tuple[dict, dict]:
    cases = {c["case_id"]: c for c in _jsonl(TESTSET_DIR / "cases.jsonl")}
    meta = json.loads((TESTSET_DIR / "audio" / "meta.json").read_text(encoding="utf-8"))
    return cases, meta


def ts172_probs(cid: str, vad: Any = None) -> np.ndarray:
    fp = TS172_PROBS / f"{cid}.npy"
    if fp.exists():
        return np.load(fp)
    from agent import audio as au
    from agent.turn.vad import SileroVAD
    vad = vad or SileroVAD()
    pcm, sr = au.wav_read(TESTSET_DIR / "audio" / f"{cid}.wav")
    assert sr == SAMPLE_RATE
    a = np.frombuffer(pcm, np.int16)
    if hasattr(vad, "reset"):
        vad.reset()
    n = len(a) // FRAME_SAMPLES
    p = np.array([vad.process(a[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES].tobytes()) for i in range(n)], np.float32)
    TS172_PROBS.mkdir(parents=True, exist_ok=True)
    np.save(fp, p)
    return p


def ts172_mid_runs(probs: np.ndarray, speech_end: float, params: SessionParams = SessionParams()) -> list[Run]:
    it = SimItem(uid="_", probs=probs, points=[SimPoint(t=speech_end, kind="end")])
    return [r for r in runs(it, params) if r.prep is not None and r.start * FRAME_S + 0.2 < speech_end]


def ts172_end_texts() -> dict[str, str]:
    """말 끝 텍스트: round1 정상 확정 문항은 턴 로그 확정 텍스트, 그 밖은 STT 평가 최종 텍스트."""
    out: dict[str, str] = {}
    res = {r["session_id"]: r for r in _jsonl(ROUND1_RUN / "results.jsonl")}
    for l in _jsonl(TURN_LOG):
        r = res.get(l["session_id"])
        if r is not None and not r["spoke_before_end"]:
            out[r["case_id"]] = l.get("user_text") or ""
    for row in _jsonl(STT_EVAL_ROWS):
        out.setdefault(row["case_id"], row.get("final") or "")
    return out


def ts172_build(stt_model: str = "gemini-3.1-flash-lite", rpm: float = 15.0, limit: int | None = None) -> dict:
    """VAD 확률 캐시 + 말 중간 침묵 지점 STT(캐시) + 말 끝 텍스트 파일."""
    import asyncio
    import time

    from agent import audio as au
    from agent.turn.vad import SileroVAD
    cases, meta = ts172_cases()
    vad = SileroVAD()
    TS172_DIR.mkdir(parents=True, exist_ok=True)
    cache = {d["key"]: d for d in _jsonl(TS172_STT)}
    jobs = []
    for cid, m in sorted(meta.items()):
        if cid not in cases:
            continue
        p = ts172_probs(cid, vad)
        for r in ts172_mid_runs(p, m["speech_end"]):
            key = f"{cid}:{r.start}"
            if key not in cache:
                jobs.append((key, cid, r.start * FRAME_S + 0.2))
    if limit is not None:
        jobs = jobs[:limit]
    TS172_END.write_text(json.dumps(ts172_end_texts(), ensure_ascii=False, indent=0), encoding="utf-8")
    stats = {"jobs": len(jobs), "ok": 0, "err": 0, "calls": 0}
    if not jobs:
        return stats

    async def run():
        from agent.providers.gemini_stt import GeminiBatchSTT
        stt = GeminiBatchSTT(model=stt_model)
        await stt.open_stream()
        gap, next_t = 60.0 / rpm, 0.0
        with TS172_STT.open("a", encoding="utf-8") as f:
            queue = list(jobs)
            tries: dict[str, int] = {}
            while queue:
                key, cid, t = queue.pop(0)
                now = time.monotonic()
                if now < next_t:
                    await asyncio.sleep(next_t - now)
                next_t = time.monotonic() + gap
                pcm, _ = au.wav_read(TESTSET_DIR / "audio" / f"{cid}.wav")
                stats["calls"] += 1
                try:
                    text = await asyncio.wait_for(stt.transcribe(pcm[: int(round(t * SAMPLE_RATE)) * 2]), 60)
                except Exception as e:  # noqa: BLE001
                    stats["err"] += 1
                    tries[key] = tries.get(key, 0) + 1
                    if tries[key] < 3:
                        queue.append((key, cid, t))
                    await asyncio.sleep(20 if "429" in str(e) else 3)
                    continue
                f.write(json.dumps({"key": key, "text": text, "model": stt_model}, ensure_ascii=False) + "\n")
                f.flush()
                stats["ok"] += 1

    asyncio.run(run())
    return stats


def ts172_items(params: SessionParams = SessionParams()) -> list[SimItem]:
    cases, meta = ts172_cases()
    stt = {d["key"]: d["text"] for d in _jsonl(TS172_STT)}
    ends = json.loads(TS172_END.read_text(encoding="utf-8")) if TS172_END.exists() else {}
    items = []
    for cid, m in sorted(meta.items()):
        c = cases.get(cid)
        if c is None or not (TS172_PROBS / f"{cid}.npy").exists():
            continue
        p = np.load(TS172_PROBS / f"{cid}.npy")
        pts = []
        for r in ts172_mid_runs(p, m["speech_end"], params):
            pts.append(SimPoint(t=r.start * FRAME_S, kind="pause", cut_type="mid", dur_ms=r.n * FRAME_MS,
                                text=stt.get(f"{cid}:{r.start}") or ""))
        pts.append(SimPoint(t=float(m["speech_end"]), kind="end", text=ends.get(cid, "")))
        env = c.get("env", {})
        items.append(SimItem(uid=cid, probs=p, points=pts, meta={
            "type": c.get("category"), "channel": env.get("channel"), "noise": env.get("noise"),
            "tags": c.get("tags", []), "expect_short": _expect_short(c.get("history") or []),
            "segments": c.get("segments", [])}))
    return items


def _expect_short(history: list) -> bool:
    if not history or not isinstance(history[-1], dict):
        return False
    last = history[-1]
    if last.get("role") not in ("agent", "assistant", "bot"):
        return False
    return (last.get("text") or last.get("content") or "").strip().endswith(("?", "？"))


def run_ts172(items: list[SimItem], detector_factory: Callable[[], Any],
              params: SessionParams = SessionParams()) -> list[SimResult]:
    from dataclasses import replace
    return [simulate_fast(it, detector_factory(), replace(params, expect_short=bool(it.meta.get("expect_short"))))
            for it in items]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features")
    f.add_argument("--force", action="store_true")
    tb = sub.add_parser("testset-build", help="172문항 VAD·지점 STT 캐시(data/turnset/testset172)")
    tb.add_argument("--rpm", type=float, default=15.0)
    tb.add_argument("--limit", type=int, default=None)
    sub.add_parser("testset-eval", help="172문항 시뮬레이션(make_detector vs T12)")
    e = sub.add_parser("eval")
    e.add_argument("--split", default=None)
    e.add_argument("--text", default="stt", choices=["stt", "gt"])
    e.add_argument("--default", action="store_true", help="DEFAULT_CONFIG(보정 전)로 측정")
    e.add_argument("--empty-text", type=float, default=0.35, help="확정 시점 STT 텍스트가 빈 비율")
    e.add_argument("--smart-delay", type=int, default=2)
    a = ap.parse_args(argv)
    if a.cmd == "features":
        print("built", build_features(force=a.force))
        return
    if a.cmd == "testset-build":
        print(ts172_build(rpm=a.rpm, limit=a.limit))
        return
    if a.cmd == "testset-eval":
        from agent.turn.detector import CALIBRATED_CONFIG_T12, TableTurnDetector, make_detector
        items = ts172_items()
        for name, fac in (("T12", lambda: TableTurnDetector(CALIBRATED_CONFIG_T12)), ("make_detector", make_detector)):
            s = summarize(run_ts172(items, fac), {it.uid: it.meta for it in items}, group_keys=("type",))
            print(name, json.dumps({k: v for k, v in s.items() if k != "by"}, ensure_ascii=False))
        return
    from agent.turn.detector import RuleTurnDetector, make_detector
    items = load_items(split=a.split, text_source=a.text)
    fac = RuleTurnDetector if a.default else make_detector
    res = run_all(items, fac, SessionParams(smart_delay_frames=a.smart_delay, empty_text_rate=a.empty_text))
    s = summarize(res, {it.uid: it.meta for it in items})
    print(json.dumps({k: v for k, v in s.items() if k != "by"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
