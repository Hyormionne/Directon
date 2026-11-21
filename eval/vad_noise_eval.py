"""잡음 강건 VAD 오프라인 측정(AG-T10c). 실제 API 호출 없음.

테스트셋 오디오(legacy/testset/audio/<id>.wav)를 20ms 프레임으로 Silero에 흘려 확률·에너지를 구하고,
여러 판정 방식(원래 Silero p≥0.5, p≥0.7, NoiseGate 설정들)을 같은 프레임열에 적용해 비교한다.

정답(GT) 음성 구간: 합성 캐시(legacy/testset/cache)에서 깨끗한 트랙을 다시 조립(속도·lead·tail 반영)해
20ms 프레임 RMS로 음성/쉼을 정한다(eval.augment와 같은 임계). 캐시가 없으면 그 문항은 쉼 지표에서 뺀다.

지표(잡음 종류별)
- post200: 말 끝(meta.speech_end) 뒤 '비음성 연속 200ms'(세션 선호출·결정 시작점)에 도달한 시각 - 말 끝(ms).
  파일 끝까지 도달 못 하면 never로 세고 값은 남은 길이로 둔다.
- 말 중간 긴 침묵: 첫 GT 음성 ~ 말 끝-100ms 사이 비음성 연속 구간 중 길이 ≥ W(400/600/900ms).
  spurious = GT 쉼(같은 위치, 길이 ≥ W-200ms)이 없는데 생긴 구간 → 말 중간을 말 끝으로 오인할 위험.
- recall: GT 음성 프레임 중 음성 판정 비율.

사용:
  uv run python -m eval.vad_noise_eval [--cases legacy/testset/cases.jsonl] [--audio legacy/testset/audio]
      [--out runs/vad_noise_eval] [--sweep]
  uv run python -m eval.vad_noise_eval --turnset [--split test]
      → AG-T12 턴 감지 보정셋(data/turnset)에서 원래 Silero 확률과 NoiseGate 적용 확률로 eval.turn_sim을 돌려
        make_detector()의 false cut·지연을 비교한다(잡음 래퍼가 쉼을 더 '침묵'으로 만들어 조기 확정이 느는지).
"""
from __future__ import annotations

import argparse
import json
import wave
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from agent.turn.vad import NoiseGate, NoiseGateConfig, SileroVAD

ROOT = Path(__file__).resolve().parents[1]
FR = 320                        # 20ms
FRAME_S = 0.02
NOISES = ("none", "white", "pink", "hum", "babble")
WINDOWS_MS = (400, 600, 900)


# ── 입력 ─────────────────────────────────────────────
def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16)


def silero_trace(x: np.ndarray, vad: SileroVAD) -> tuple[np.ndarray, np.ndarray]:
    """프레임별 (Silero 확률, 에너지 dBFS)."""
    vad.reset()
    n = len(x) // FR
    p = np.zeros(n)
    e = np.zeros(n)
    for i in range(n):
        f = x[i * FR:(i + 1) * FR]
        p[i] = vad.process(f.tobytes())
        a = f.astype(np.float64)
        e[i] = 10 * np.log10(np.mean(a * a) / 32768.0 ** 2 + 1e-12)
    return p, e


def clean_track(case: dict, m: dict, synth) -> np.ndarray | None:
    """build_testset과 같은 방식으로 깨끗한 트랙(속도·lead·tail 반영)을 캐시에서 다시 만든다."""
    from eval import augment as A
    from eval.build_testset import TARGET_RMS, case_segments

    voice, style = case["voice"], case.get("voice_style")
    for s in case_segments(case):
        if not synth.cache_path(s["text"], voice, style).exists():
            return None
    clean = A.assemble_segments(case_segments(case), lambda t: synth.synthesize(t, voice, style))
    x = np.round(A.normalize_level(clean, TARGET_RMS)).astype(np.float32)
    env = A.Env.from_dict(case.get("env") or {})
    if env.speed != 1.0:
        x = A.time_stretch(x, float(env.speed), method=m.get("speed_method") or "auto")
    lead = int(round(env.lead_ms * A.SR / 1000))
    tail = int(round(env.tail_ms * A.SR / 1000))
    return np.concatenate([np.zeros(lead, np.float32), np.asarray(x, np.float32), np.zeros(tail, np.float32)])


def gt_voiced(clean: np.ndarray, n: int) -> np.ndarray:
    from eval import augment as A

    voiced, _ = A._voiced_frames(clean, FR)
    out = np.zeros(n, bool)
    k = min(n, len(voiced))
    out[:k] = voiced[:k]
    return out


# ── 판정 방식 ─────────────────────────────────────────
def decide_threshold(p: np.ndarray, e: np.ndarray, thr: float) -> np.ndarray:
    return p >= thr


def decide_gate(p: np.ndarray, e: np.ndarray, cfg: NoiseGateConfig) -> np.ndarray:
    g = NoiseGate(cfg)
    return np.array([g.update_db(float(ei), float(pi)) for pi, ei in zip(p, e)])


# ── 지표 ─────────────────────────────────────────────
def silence_runs(sp: np.ndarray, a: int, b: int) -> list[tuple[int, int]]:
    """[a,b) 안의 비음성 연속 구간 [s,e) 목록(구간 경계에서 잘린 것 포함)."""
    runs, s = [], None
    for i in range(a, b):
        if not sp[i]:
            if s is None:
                s = i
        elif s is not None:
            runs.append((s, i))
            s = None
    if s is not None:
        runs.append((s, b))
    return runs


def post_end_reach(sp: np.ndarray, end_f: int, need: int = 10) -> tuple[int, bool]:
    """말 끝 프레임부터 비음성 need개 연속에 도달한 시각(ms, 말 끝 기준). (값, 도달 여부)."""
    run = 0
    for i in range(max(0, end_f - need), len(sp)):    # 말 끝 직전부터 이어진 침묵도 인정
        run = 0 if sp[i] else run + 1
        if run >= need and i >= end_f:
            return int(round((i + 1 - end_f) * FRAME_S * 1000)), True
    return int(round((len(sp) - end_f) * FRAME_S * 1000)), False


def case_metrics(sp: np.ndarray, end_f: int, gt: np.ndarray | None) -> dict:
    reach, ok = post_end_reach(sp, end_f)
    out: dict = {"post200_ms": reach, "post200_ok": ok}
    if gt is None or not gt.any():
        return out
    start_f = int(np.flatnonzero(gt)[0])
    stop = max(start_f, end_f - 5)
    runs = silence_runs(sp, start_f, stop)
    gt_runs = silence_runs(gt, start_f, stop)
    for w in WINDOWS_MS:
        nw = w // 20
        longs = [(s, e) for s, e in runs if e - s >= nw]
        spur = 0
        for s, e in longs:
            ok_gt = any(min(e, ge) - max(s, gs) >= nw - 10 for gs, ge in gt_runs)
            spur += 0 if ok_gt else 1
        out[f"long{w}"] = len(longs)
        out[f"spur{w}"] = spur
    voiced = gt[:end_f]
    out["recall"] = float(sp[:end_f][voiced].mean()) if voiced.any() else None
    return out


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    post = np.array([r["post200_ms"] for r in rows])
    s = {"n": len(rows), "post200_p50": float(np.median(post)), "post200_p90": float(np.percentile(post, 90)),
         "post200_max": float(post.max()), "never": int(sum(not r["post200_ok"] for r in rows))}
    g = [r for r in rows if "long600" in r]
    s["n_gt"] = len(g)
    for w in WINDOWS_MS:
        s[f"cases_long{w}"] = int(sum(r[f"long{w}"] > 0 for r in g))
        s[f"spur{w}"] = int(sum(r[f"spur{w}"] for r in g))
        s[f"cases_spur{w}"] = int(sum(r[f"spur{w}"] > 0 for r in g))
    rec = [r["recall"] for r in g if r.get("recall") is not None]
    s["recall"] = float(np.mean(rec)) if rec else None
    return s


# ── 실행 ─────────────────────────────────────────────
def load(cases_path: Path, audio_dir: Path, use_gt: bool = True) -> list[dict]:
    cases = {json.loads(l)["case_id"]: json.loads(l) for l in cases_path.read_text(encoding="utf-8").splitlines()
             if l.strip()}
    meta = json.loads((audio_dir / "meta.json").read_text(encoding="utf-8"))
    synth = None
    if use_gt:
        from eval.synth import Synthesizer
        synth = Synthesizer()
    vad = SileroVAD()
    items = []
    for cid, m in sorted(meta.items()):
        if cid not in cases or not (audio_dir / f"{cid}.wav").exists():
            continue
        x = read_wav(audio_dir / f"{cid}.wav")
        p, e = silero_trace(x, vad)
        gt = None
        if synth is not None:
            try:
                clean = clean_track(cases[cid], m, synth)
            except Exception:  # noqa: BLE001
                clean = None
            if clean is not None:
                gt = gt_voiced(clean, len(p))
        items.append({"case_id": cid, "noise": m["env"].get("noise", "none"), "snr_db": m["env"].get("snr_db"),
                      "end_f": int(round(m["speech_end"] / FRAME_S)), "p": p, "e": e, "gt": gt})
    return items


def evaluate(items: list[dict], decide) -> dict:
    by: dict[str, list[dict]] = {k: [] for k in NOISES}
    per_case = {}
    for it in items:
        sp = decide(it["p"], it["e"])
        r = case_metrics(sp, it["end_f"], it["gt"])
        by.setdefault(it["noise"], []).append(r)
        per_case[it["case_id"]] = r
    out = {k: summarize(v) for k, v in by.items() if v}
    out["all"] = summarize([r for v in by.values() for r in v])
    return {"summary": out, "per_case": per_case}


def variants(sweep: bool) -> dict:
    base = NoiseGateConfig()
    v = {"silero>=0.5": lambda p, e: decide_threshold(p, e, 0.5),
         "silero>=0.7": lambda p, e: decide_threshold(p, e, 0.7),
         "gate(default)": lambda p, e, c=base: decide_gate(p, e, c)}
    if sweep:
        grid = []
        for margin in (4.5, 6.0):
            for hold in (1.5, 3.0):
                for onset in (2, 3):
                    for pn in (0.6, 0.7):
                        for fl in (1.2, 1.5, 2.0):
                            grid.append(replace(base, margin_db=margin, hold_margin_db=hold, onset_frames=onset,
                                                p_noisy=pn, fluct_on_db=fl))
        for c in grid:
            name = f"gate(m{c.margin_db:g},h{c.hold_margin_db:g},o{c.onset_frames},p{c.p_noisy:g},f{c.fluct_on_db:g})"
            v[name] = lambda p, e, c=c: decide_gate(p, e, c)
    return v


def fmt_table(results: dict[str, dict], noises=NOISES + ("all",)) -> str:
    lines = ["| 방식 | 잡음 | n | 말 끝→침묵200 p50 / p90 / 최대(ms) | 도달 못 함 | 중간 긴 침묵 ≥600 (문항) | 오인 ≥400 / ≥600 / ≥900 | recall |",
             "|---|---|---|---|---|---|---|---|"]
    for name, res in results.items():
        for nz in noises:
            s = res["summary"].get(nz)
            if not s:
                continue
            rec = "-" if s.get("recall") is None else f"{s['recall']:.2f}"
            lines.append(f"| {name} | {nz} | {s['n']} | {s['post200_p50']:.0f} / {s['post200_p90']:.0f} / "
                         f"{s['post200_max']:.0f} | {s['never']} | {s.get('cases_long600', '-')} | "
                         f"{s.get('spur400', '-')} / {s.get('spur600', '-')} / {s.get('spur900', '-')} | {rec} |")
    return "\n".join(lines)


def gate_probs(probs: np.ndarray, energies: np.ndarray, cfg: NoiseGateConfig | None = None) -> np.ndarray:
    """세션이 보는 확률(NoiseRobustSileroVAD.process 출력)을 원래 Silero 확률 + 프레임 에너지로 재현."""
    g = NoiseGate(cfg)
    return np.array([g.output(float(p), g.update_db(float(e), float(p))) for p, e in zip(probs, energies)])


def turnset_compare(split: str | None = None, empty_text: float = 0.35, smart_delay: int = 2) -> dict:
    from agent import audio as au
    from agent.turn.detector import make_detector
    from eval import turn_sim as TS

    items = TS.load_items(split=split)
    gated = []
    for it in items:
        pcm, _ = au.wav_read(TS.AUDIO_DIR / f"{it.uid}.wav")
        a = np.frombuffer(pcm, np.int16).astype(np.float64)
        n = len(it.probs)
        pad = np.zeros(n * FR)
        pad[: min(len(a), n * FR)] = a[: n * FR]
        e = 10 * np.log10(np.mean(pad.reshape(n, FR) ** 2, axis=1) / 32768.0 ** 2 + 1e-12)
        gated.append(TS.SimItem(uid=it.uid, probs=gate_probs(it.probs, e), points=it.points, smart=dict(it.smart),
                                meta=it.meta))
    params = TS.SessionParams(smart_delay_frames=smart_delay, empty_text_rate=empty_text)
    meta = {it.uid: it.meta for it in items}
    out = {}
    for name, its in (("silero", items), ("gate", gated)):
        out[name] = TS.summarize(TS.run_all(its, make_detector, params), meta, group_keys=("noise",))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(ROOT / "legacy/testset/cases.jsonl"))
    ap.add_argument("--audio", default=str(ROOT / "legacy/testset/audio"))
    ap.add_argument("--out", default=str(ROOT / "runs/vad_noise_eval"))
    ap.add_argument("--sweep", action="store_true", help="NoiseGate 설정 격자 탐색")
    ap.add_argument("--no-gt", action="store_true", help="GT 쉼(합성 캐시) 없이")
    ap.add_argument("--turnset", action="store_true", help="턴 감지 보정셋에서 false cut 비교(eval.turn_sim)")
    ap.add_argument("--split", default=None)
    a = ap.parse_args(argv)
    if a.turnset:
        res = turnset_compare(a.split)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"turnset_{a.split or 'all'}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                                              encoding="utf-8")
        print("| VAD | 잡음 | n | false cut | 비율 | 지연 p50 / p95(ms) | miss |")
        print("|---|---|---|---|---|---|---|")
        for name, s in res.items():
            rows = [("all", s)] + list(s["by"]["noise"].items())
            for nz, r in rows:
                p50 = "-" if r["latency_p50"] is None else f"{r['latency_p50']:.0f}"
                p95 = "-" if r["latency_p95"] is None else f"{r['latency_p95']:.0f}"
                print(f"| {name} | {nz} | {r['n']} | {r['false_cut']} | {r['false_cut_rate']:.3f} | "
                      f"{p50} / {p95} | {r['miss_rate']:.3f} |")
        return
    items = load(Path(a.cases), Path(a.audio), use_gt=not a.no_gt)
    n_gt = sum(it["gt"] is not None for it in items)
    print(f"문항 {len(items)}개, GT 복원 {n_gt}개")
    results = {name: evaluate(items, fn) for name, fn in variants(a.sweep).items()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({k: v["summary"] for k, v in results.items()},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "per_case.json").write_text(json.dumps({k: v["per_case"] for k, v in results.items()
                                                   if not k.startswith("gate(m")}, ensure_ascii=False),
                                       encoding="utf-8")
    (out / "config.json").write_text(json.dumps(asdict(NoiseGateConfig()), indent=1), encoding="utf-8")
    keep = {k: v for k, v in results.items() if not k.startswith("gate(m")}
    print(fmt_table(keep))
    if a.sweep:
        rows = []
        for k, v in results.items():
            b, al = v["summary"].get("babble", {}), v["summary"]["all"]
            rows.append((k, b.get("post200_p50"), b.get("post200_p90"), b.get("never"), al.get("spur600"),
                         al.get("spur400"), al.get("recall")))
        rows.sort(key=lambda r: (r[4] or 0, r[2] or 0))
        print("\n| 설정 | babble p50 | babble p90 | babble never | 전체 오인≥600 | 전체 오인≥400 | recall |")
        print("|---|---|---|---|---|---|---|")
        for r in rows[:40]:
            print("| " + " | ".join("-" if x is None else (f"{x:.2f}" if isinstance(x, float) and x < 2 else
                                                             f"{x:.0f}" if isinstance(x, float) else str(x))
                                    for x in r) + " |")


if __name__ == "__main__":
    main()
