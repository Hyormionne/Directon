"""AG-T30 엔드포인팅 비교: 기존 결정기 / Soniox `<end>` 단독 / Soniox + 규칙 / Soniox + Jev / Soniox + 규칙 + Jev.

입력: `eval.soniox_endpoint_eval`이 녹화한 rows.jsonl(토큰 이벤트 포함). Soniox를 다시 부르지 않고
agent/turn/soniox_gate.simulate_gate로 재생한다. Jev 응답은 data/turnset/jev_cache.jsonl(eval.jev_turn_eval과 같은
캐시·키)에서 읽고, 없는 텍스트만 부른다. Jev 판정 시각 = `<end>` 도착 + 그 요청의 실측 왕복(ms).
기존 결정기는 eval.turn_sim(같은 오디오, 오프라인)으로 같은 판정 규칙을 쓴다.

사용:
  uv run python -m eval.soniox_gate_eval [--run runs/endpoint/ev_turnset --run runs/onedam/ev_audio ...]
      --out runs/endpoint/gate_report.md [--no-jev]
기존 결정기는 두 가지로 본다: turn_sim(Gemini 받아쓰기 지점 텍스트, turnset·ts172만)과
Soniox 이벤트 재생(simulate_detector: 세션과 같은 VAD·finalize 순서, Soniox 토큰 지연 포함).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from agent.turn import jev_turn as jt
from agent.turn.soniox_gate import GatePolicy, GateVerdict, combine, is_speech_token, simulate_gate
from eval.soniox_endpoint_eval import ROOT, EndpointResult, classify_endpoint, load_dataset

CACHE = ROOT / "data/turnset/jev_cache.jsonl"


def end_texts(events: list[dict]) -> list[str]:
    """각 `<end>` 시점의 누적 확정 텍스트(빈 것 제외)."""
    final, out = "", []
    for ev in events:
        for x in ev.get("toks", []):
            if x.get("is_final") and is_speech_token(x):
                final += x["text"]
        if any(x.get("text") == "<end>" for x in ev.get("toks", [])) and final.strip():
            out.append(final.strip())
    return out


class JevCache:
    def __init__(self, path: Path = CACHE, call: bool = True):
        from eval import jev_turn_eval as je
        self.je, self.path, self.call = je, path, call
        self.d: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.d[r["k"]] = r
        self.calls = 0

    def key(self, text: str, history: list, known: bool) -> str:
        return self.je.body_key(jt.build_body(text, history, known, jt.MODEL))

    def get(self, text: str, history: list, known: bool) -> tuple[float | None, float]:
        r = self.d.get(self.key(text, history, known))
        if not r:
            return None, 0.0
        res = jt.parse_answers(r)
        return (res.finished if res else None), float(r.get("ms") or 0.0)

    def fill(self, reqs: list[tuple[str, list, bool]], workers: int = 4) -> None:
        todo = {}
        for t, h, k in reqs:
            kk = self.key(t, h, k)
            if kk not in self.d:
                todo[kk] = (t, h, k)
        if not todo or not self.call:
            return
        self.je.load_dotenv()
        jev = self.je.Jev()

        def one(item):
            kk, (t, h, k) = item
            body = jt.build_body(t, h, k, jt.MODEL)
            resp, ms = jev.ask(body)
            ans = {q: (resp.get("answers") or {}).get(q) for q in jt.QUESTIONS}
            ans = {q: (v.get("noul") if isinstance(v, dict) else v) for q, v in ans.items()}
            return {"k": kk, "model": jt.MODEL, "answers": ans, "ms": round(ms, 1), "usage": resp.get("usage")}

        with ThreadPoolExecutor(workers) as ex, self.path.open("a", encoding="utf-8") as f:
            for r in ex.map(one, todo.items()):
                self.d[r["k"]] = r
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                self.calls += 1


def load_rows(run_dir: Path, items: list[dict]) -> list[tuple[dict, dict]]:
    by = {it["uid"]: it for it in items}
    out = []
    for line in (run_dir / "rows.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["uid"] in by and not r.get("error"):
            out.append((by[r["uid"]], r))
    return out


VAD_CACHE = ROOT / "runs/endpoint/vad"


def vad_probs(wav: Path, vad: Any = None) -> np.ndarray:
    """wav → 20ms 프레임별 Silero 확률(세션과 같은 분할). runs/endpoint/vad/<경로 해시>.npy에 캐시."""
    key = hashlib.sha1(str(Path(wav).resolve()).encode()).hexdigest()[:16]
    fp = VAD_CACHE / f"{key}.npy"
    if fp.exists() and fp.stat().st_mtime >= Path(wav).stat().st_mtime:
        return np.load(fp)
    from agent import audio as au
    from agent.turn.vad import SileroVAD
    from eval.turn_sim import FRAME_SAMPLES
    vad = vad or SileroVAD()
    vad.reset()
    pcm, _ = au.wav_read(wav)
    a = np.frombuffer(pcm, np.int16)
    p = np.array([vad.process(a[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES].tobytes())
                  for i in range(len(a) // FRAME_SAMPLES)], np.float32)
    VAD_CACHE.mkdir(parents=True, exist_ok=True)
    np.save(fp, p)
    return p


def attach_vad(name: str, pairs: list[tuple[dict, dict]], threshold: float = 0.5) -> None:
    """프레임별 VAD 음성 여부 → it["speech"]. turnset·ts172는 turn_sim 캐시(같은 Silero), 그 밖은 직접 계산."""
    if name in ("turnset", "ts172"):
        from eval import turn_sim as ts
        items = ts.load_items(split="test") if name == "turnset" else ts.ts172_items()
        probs = {i.uid: i.probs for i in items}
    else:
        from agent.turn.vad import SileroVAD
        v = SileroVAD()
        probs = {it["uid"]: vad_probs(it["wav"], v) for it, _ in pairs}
    for it, _ in pairs:
        it["speech"] = [bool(x >= threshold) for x in probs[it["uid"]]]


def simulate_detector(events: list[dict], speech: list[bool], until_s: float, detector: Any,
                      expect_short: bool = False, finalize_ms: float = 240.0, precall_ms: int = 200,
                      min_speech_ms: int = 60, jev: Any = None, combo: "Combo | None" = None,
                      collect: list | None = None) -> tuple[float | None, str]:
    """기존 결정기를 Soniox 이벤트로 재생한다(agent/session.py _process_frame과 같은 순서).
    - 텍스트: Soniox 최신(확정 + 마지막 응답의 미확정). 침묵 precall_ms에 finalize를 요청했다고 보고
      finalize_ms 뒤부터는 '요청 시각 전에 시작한 확정 토큰 전부'(finalize 결과 근사)를 쓴다(precall_refresh).
    - 음성이 다시 나오면 finalize 결과를 버린다(_cancel_prepared). → (확정 시각, 이유)
    Jev 결합(combo, jev(text) → (p, ms)):
      trigger="silence": agent/turn/jev_turn.JevTurnDetector와 같다. 침묵 200ms부터 텍스트가 바뀔 때마다(침묵 하나에 최대 2회)
        요청, 응답은 ms 뒤 도착, 그 텍스트와 지금 텍스트가 같을 때만 decide_with_p(늘리기 lo·줄이기 hi·fast_ms).
      trigger="end": Soniox `<end>`가 오면(VAD 침묵일 때) 그 확정 텍스트로 요청. 응답이 오면 p ≥ end_hi이면 즉시 확정,
        p ≤ lo이면 이번 침묵 동안 대기를 상한까지 늘린다. 그 밖은 결정기 그대로.
    collect: 리스트를 주면 Jev 요청 텍스트를 모은다(캐시 채우기용)."""
    from agent.core.types import TurnDecision, TurnSignals
    from agent.turn.jev_turn import decide_with_p
    finals = [(x.get("start_ms", 0), x["text"]) for ev in events for x in ev.get("toks", [])
              if x.get("is_final") and is_speech_token(x)]
    final, nonfinal, ei = "", "", 0
    speech_ms = sil = 0
    fin_req = None
    pending: list[tuple[float, str, float | None]] = []   # (도착 시각, 텍스트 키, p)
    got: tuple[str, float] | None = None                   # silence 트리거: (텍스트 키, p)
    asked: list[str] = []
    end_p: float | None = None                             # end 트리거: 이번 침묵의 <end> Jev p

    def ask(text: str, now: float) -> None:
        k = " ".join(text.split())
        if collect is not None:
            collect.append(text)
        if jev is None:
            return
        pv, ms = jev(text)
        if pv is not None:
            pending.append((now + ms / 1000, k, pv))

    for f in range(int(round(until_s / 0.02))):
        t = (f + 1) * 0.02
        end_now = False
        while ei < len(events) and events[ei]["t"] <= t:
            toks = events[ei]["toks"]
            final += "".join(x["text"] for x in toks if x.get("is_final") and is_speech_token(x))
            nonfinal = "".join(x["text"] for x in toks if not x.get("is_final") and is_speech_token(x))
            end_now = end_now or any(x.get("text") == "<end>" for x in toks)
            ei += 1
        if f < len(speech) and speech[f]:
            if sil > 0:
                fin_req = None
                pending, got, asked, end_p = [], None, [], None
            speech_ms += 20
            sil = 0
            continue
        if speech_ms < min_speech_ms:
            if speech_ms and sil + 20 >= precall_ms:
                speech_ms = sil = 0
            elif speech_ms:
                sil += 20
            continue
        sil += 20
        if fin_req is None and sil >= precall_ms:
            fin_req = t
        early = None
        if fin_req is not None and t >= fin_req + finalize_ms / 1000:
            early = "".join(x for st, x in finals if st < fin_req * 1000).strip() or None
        text = early or (final + nonfinal).strip()
        sig = TurnSignals(silence_ms=sil, speech_ms=speech_ms, text=text, smart_turn_p=None, expect_short=expect_short)
        if combo is None:
            dec = detector.decide(sig)
        else:
            if combo.trigger == "silence" and sil >= precall_ms and text and len(asked) < combo.max_requests:
                k = " ".join(text.split())
                if k not in asked:
                    asked.append(k)
                    ask(text, t)
            if combo.trigger == "end" and end_now and final.strip():
                ask(final.strip(), t)
            for item in [x for x in pending if x[0] <= t]:
                pending.remove(item)
                if combo.trigger == "silence":
                    got = (item[1], item[2])
                else:
                    end_p = item[2]
            if combo.trigger == "silence":
                if combo.match == "latest":     # 이번 침묵에서 가장 최근에 온 p(텍스트가 바뀌어도)
                    pv = got[1] if got is not None else None
                else:                           # JevTurnDetector와 같음: 지금 텍스트와 같은 텍스트의 p만
                    pv = got[1] if (got is not None and got[0] == " ".join(text.split())) else None
                dec = decide_with_p(sig, detector, pv, combo.policy)
            else:
                if end_p is not None and end_p >= combo.end_hi:
                    dec = TurnDecision(True, f"end_jev(p={end_p:.2f})")
                else:
                    pv = end_p if (end_p is not None and end_p <= combo.policy.lo) else None
                    dec = decide_with_p(sig, detector, pv, combo.policy)
        if dec.end:
            return t, dec.reason
    return None, "no_end"


class _NeverEnd:
    """Jev 요청 텍스트를 모을 때 쓰는 결정기(끝내지 않는다)."""
    config = None

    def decide(self, s: Any) -> Any:
        from agent.core.types import TurnDecision
        return TurnDecision(False, "")


@dataclass(frozen=True)
class Combo:
    trigger: str                  # "silence" | "end"
    policy: Any                   # agent.turn.jev_turn.JevTurnPolicy
    end_hi: float = 1.01          # trigger="end": p ≥ end_hi이면 즉시 확정
    max_requests: int = 2         # trigger="silence": 침묵 하나에서 텍스트가 바뀔 때 요청 상한(JevTurnDetector 기본 2)
    match: str = "exact"          # "exact"(지금 텍스트와 같을 때만) | "latest"(최근 응답)


def collect_jev_texts(pairs: list[tuple[dict, dict]], tail: float = 2.5) -> list[tuple[str, list, bool]]:
    """두 트리거가 요청할 수 있는 텍스트 전부(끝내지 않는 결정기로 끝까지 돌려 모은 상위 집합)."""
    from agent.turn.jev_turn import JevTurnPolicy
    out = []
    for it, row in pairs:
        h, k = it.get("history", []), it.get("caller_known", False)
        for trig in ("silence", "end"):
            got: list = []
            simulate_detector(row["events"], it["speech"], it["end_t"] + tail, _NeverEnd(),
                              combo=Combo(trig, JevTurnPolicy(), max_requests=6), collect=got)
            out += [(x, h, k) for x in got]
    return out


def run_detector(pairs: list[tuple[dict, dict]], window: float = 1.5, tail: float = 2.5,
                 combo: Combo | None = None, jev: "JevCache | None" = None) -> list[EndpointResult]:
    from agent.turn.detector import make_detector
    from eval.turn_sim import _expect_short
    res = []
    for it, row in pairs:
        h, k = it.get("history", []), it.get("caller_known", False)
        fn = (lambda text, h=h, k=k: jev.get(text, h, k)) if (combo is not None and jev is not None) else None
        t, _ = simulate_detector(row["events"], it["speech"], it["end_t"] + tail, make_detector(),
                                 _expect_short(it.get("history") or []), jev=fn, combo=combo)
        r = classify_endpoint(it["uid"], [t] if t is not None else [], it["end_t"], it["pauses"], window, it["mid"])
        r.meta = it["meta"]
        res.append(r)
    return res


def run_gate(pairs: list[tuple[dict, dict]], pol: GatePolicy | None, jev: JevCache | None,
             window: float = 1.5, tail: float = 2.5) -> list[EndpointResult]:
    res = []
    for it, row in pairs:
        if pol is None:   # Soniox 단독: 첫 <end>
            ends = row["ends"][:1]
        else:
            h, k = it.get("history", []), it.get("caller_known", False)

            def verdict(text: str) -> GateVerdict:
                p, ms = (None, 0.0)
                if pol.mode != "rule" and jev is not None:
                    v0 = combine(text, None, 0.0, replace(pol, mode="rule"))
                    if pol.mode == "jev" or v0.accept:
                        p, ms = jev.get(text, h, k)
                return combine(text, p, ms if p is not None else 0.0, pol)

            g = simulate_gate(row["events"], verdict, pol, it["speech"], it["end_t"] + tail)
            ends = [g.decision_t] if g.decision_t is not None else []
        r = classify_endpoint(it["uid"], ends, it["end_t"], it["pauses"], window, it["mid"])
        r.meta = it["meta"]
        res.append(r)
    return res


def stats(rs: list[EndpointResult]) -> dict:
    n = len(rs)
    lat = [r.latency_ms for r in rs if r.outcome == "ok"]
    fc: dict[str, int] = {}
    for r in rs:
        if r.outcome == "false_cut":
            fc[r.where] = fc.get(r.where, 0) + 1
    d = {"n": n, "fc": sum(fc.values()), "fc_rate": sum(fc.values()) / n if n else 0,
         "miss_rate": sum(r.outcome == "miss" for r in rs) / n if n else 0, "where": fc}
    if lat:
        d["p50"], d["p95"] = float(np.percentile(lat, 50)), float(np.percentile(lat, 95))
    return d


def baseline(dataset: str, uids: set[str]) -> dict:
    from agent.turn.detector import make_detector
    from eval import turn_sim as ts
    if dataset == "turnset":
        items = [i for i in ts.load_items(split="test") if i.uid in uids]
        rs = ts.run_all(items, make_detector, ts.SessionParams(empty_text_rate=0.10))
    else:
        items = [i for i in ts.ts172_items() if i.uid in uids]
        rs = ts.run_ts172(items, make_detector)
    fc: dict[str, int] = {}
    for r in rs:
        if r.outcome == "false_cut":
            fc[r.where] = fc.get(r.where, 0) + 1
    lat = [r.latency_ms for r in rs if r.outcome == "ok"]
    n = len(rs)
    return {"n": n, "fc": sum(fc.values()), "fc_rate": sum(fc.values()) / n, "where": fc,
            "miss_rate": sum(r.outcome == "miss" for r in rs) / n,
            "p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))}


MAIN = [("Soniox + VAD만(텍스트 판정 없음)", GatePolicy("rule", hold_kinds=frozenset())),
        ("Soniox + 규칙", GatePolicy("rule", hold_ms=1000)),
        ("Soniox + Jev", GatePolicy("jev", jev_threshold=0.5, hold_ms=1000)),
        ("Soniox + 규칙 + Jev", GatePolicy("rule_jev", jev_threshold=0.5, hold_ms=1000))]
SWEEP = ([(f"규칙 hold={h}", GatePolicy("rule", hold_ms=h)) for h in (700, 1000, 1300, 1600)]
         + [(f"Jev thr={t} hold=1000", GatePolicy("jev", jev_threshold=t, hold_ms=1000)) for t in (0.3, 0.5, 0.7, 0.85)]
         + [(f"규칙+Jev thr={t} hold=1000", GatePolicy("rule_jev", jev_threshold=t, hold_ms=1000)) for t in (0.5, 0.7)])


def combos() -> list[tuple[str, Combo]]:
    from agent.turn.jev_turn import JevTurnPolicy as P
    return [
        ("기존 + Jev 늘리기(침묵 200ms, lo 0.3)", Combo("silence", P(hi=1.01, lo=0.3))),
        ("기존 + Jev 늘리기 lo 0.5", Combo("silence", P(hi=1.01, lo=0.5))),
        ("기존 + Jev 늘리기+줄이기(hi 0.9 → 500ms)", Combo("silence", P(hi=0.9, lo=0.3, fast_ms=500))),
        ("기존 + Jev 늘리기+줄이기(hi 0.95 → 600ms)", Combo("silence", P(hi=0.95, lo=0.3, fast_ms=600))),
        ("기존 + Jev 늘리기 lo 0.3 [최근 응답·요청 6]", Combo("silence", P(hi=1.01, lo=0.3), max_requests=6, match="latest")),
        ("기존 + Jev 늘리기 lo 0.5 [최근 응답·요청 6]", Combo("silence", P(hi=1.01, lo=0.5), max_requests=6, match="latest")),
        ("기존 + Jev 늘리기+줄이기 hi 0.95 [최근 응답·요청 6]",
         Combo("silence", P(hi=0.95, lo=0.3, fast_ms=600), max_requests=6, match="latest")),
        ("기존 + <end>·Jev(≥0.9 즉시, ≤0.3 늘리기)", Combo("end", P(hi=1.01, lo=0.3), end_hi=0.9)),
        ("기존 + <end>·Jev(≥0.95 즉시, ≤0.3 늘리기)", Combo("end", P(hi=1.01, lo=0.3), end_hi=0.95)),
        ("기존 + <end>·Jev 늘리기만(≤0.3)", Combo("end", P(hi=1.01, lo=0.3), end_hi=1.01)),
    ]


def _fmt(d: dict) -> str:
    p = f"{d['p50']:.0f} / {d['p95']:.0f}" if "p50" in d else "-"
    return f"{d['fc_rate']*100:.1f}% ({d['fc']}) | {d['miss_rate']*100:.1f}% | {p}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[],
                    help="녹화 폴더(eval.soniox_endpoint_eval --out). 여러 번. 기본: runs/endpoint/ev_turnset, ev_ts172")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-jev", action="store_true", help="Jev 새 호출 금지(캐시만)")
    ap.add_argument("--combos", action="store_true", help="기존 결정기 + Jev 결합 방안(COMBOS)도 비교")
    a = ap.parse_args(argv)
    sets = {}
    for run in a.run or ["runs/endpoint/ev_turnset", "runs/endpoint/ev_ts172"]:
        if not Path(run, "rows.jsonl").exists():
            continue
        cfg = json.loads(Path(run, "summary.json").read_text(encoding="utf-8"))["config"]
        name = cfg["dataset"]
        sets[name] = load_rows(Path(run), load_dataset(name, cfg.get("cases", ""), cfg.get("audio", "")))
        attach_vad(name, sets[name])
    jev = JevCache(call=not a.no_jev)
    reqs = [(t, it.get("history", []), it.get("caller_known", False))
            for pairs in sets.values() for it, row in pairs for t in end_texts(row["events"])]
    if a.combos:
        for pairs in sets.values():
            reqs += collect_jev_texts(pairs)
    jev.fill(reqs)
    lines = ["| 데이터 | 방식 | 말 끝 전 확정 | miss | 지연 p50 / p95 ms |", "|---|---|---|---|---|"]
    detail: dict = {}
    for name, pairs in sets.items():
        uids = {it["uid"] for it, _ in pairs}
        rows = [("기존 결정기 (turn_sim, Gemini 지점 텍스트)", baseline(name, uids))] if name in ("turnset", "ts172") else []
        rows += [("기존 결정기 (Soniox 텍스트 재생)", stats(run_detector(pairs))),
                 ("Soniox 단독", stats(run_gate(pairs, None, jev)))]
        rows += [(lbl, stats(run_gate(pairs, pol, jev))) for lbl, pol in MAIN]
        if a.combos:
            rows += [(lbl, stats(run_detector(pairs, combo=c, jev=jev))) for lbl, c in combos()]
        for lbl, d in rows:
            lines.append(f"| {name} (n={d['n']}) | {lbl} | {_fmt(d)} |")
        detail[name] = {lbl: d for lbl, d in rows}
        detail[name]["_sweep"] = {lbl: stats(run_gate(pairs, pol, jev)) for lbl, pol in SWEEP}
    sweep = ["| 데이터 | 설정 | 말 끝 전 확정 | miss | 지연 p50 / p95 ms |", "|---|---|---|---|---|"]
    for name in detail:
        for lbl, d in detail[name]["_sweep"].items():
            sweep.append(f"| {name} | {lbl} | {_fmt(d)} |")
    where = ["| 데이터 | 방식 | 말 끝 전 확정 위치 |", "|---|---|---|"]
    for name in detail:
        for lbl, d in detail[name].items():
            if lbl != "_sweep":
                where.append(f"| {name} | {lbl} | {', '.join(f'{k} {v}' for k, v in sorted(d['where'].items(), key=lambda x: -x[1]))} |")
    text = "\n".join(lines) + "\n\n" + "\n".join(sweep) + "\n\n" + "\n".join(where) + f"\n\nJev 새 호출 {jev.calls}회\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        Path(a.out).with_suffix(".json").write_text(json.dumps(detail, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
