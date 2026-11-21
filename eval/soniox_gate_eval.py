"""AG-T30 엔드포인팅 비교: 기존 결정기 / Soniox `<end>` 단독 / Soniox + 규칙 / Soniox + Jev / Soniox + 규칙 + Jev.

입력: `eval.soniox_endpoint_eval`이 녹화한 rows.jsonl(토큰 이벤트 포함). Soniox를 다시 부르지 않고
agent/turn/soniox_gate.simulate_gate로 재생한다. Jev 응답은 data/turnset/jev_cache.jsonl(eval.jev_turn_eval과 같은
캐시·키)에서 읽고, 없는 텍스트만 부른다. Jev 판정 시각 = `<end>` 도착 + 그 요청의 실측 왕복(ms).
기존 결정기는 eval.turn_sim(같은 오디오, 오프라인)으로 같은 판정 규칙을 쓴다.

사용:
  uv run python -m eval.soniox_gate_eval --turnset runs/endpoint/ev_turnset --ts172 runs/endpoint/ev_ts172
      --out docs/decisions/soniox_gate_report.md [--no-jev]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from agent.turn import jev_turn as jt
from agent.turn.soniox_gate import GatePolicy, GateVerdict, combine, is_speech_token, simulate_gate
from eval.soniox_endpoint_eval import ROOT, EndpointResult, Pause, classify_endpoint, load_ts172, load_turnset

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


def attach_vad(name: str, pairs: list[tuple[dict, dict]], threshold: float = 0.5) -> None:
    """turn_sim 캐시의 프레임별 VAD 확률(세션과 같은 Silero) → it["speech"]."""
    from eval import turn_sim as ts
    items = ts.load_items(split="test") if name == "turnset" else ts.ts172_items()
    probs = {i.uid: i.probs for i in items}
    for it, _ in pairs:
        it["speech"] = [bool(x >= threshold) for x in probs[it["uid"]]]


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


def _ctx_ts172(items: list[dict]) -> None:
    cases = {}
    for line in (ROOT / "data/testset/cases.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            c = json.loads(line)
            cases[c["case_id"]] = c
    for it in items:
        c = cases.get(it["uid"], {})
        it["history"] = list(c.get("history") or [])
        it["caller_known"] = bool(c.get("customer_profile"))


MAIN = [("Soniox + VAD만(텍스트 판정 없음)", GatePolicy("rule", hold_kinds=frozenset())),
        ("Soniox + 규칙", GatePolicy("rule", hold_ms=1000)),
        ("Soniox + Jev", GatePolicy("jev", jev_threshold=0.5, hold_ms=1000)),
        ("Soniox + 규칙 + Jev", GatePolicy("rule_jev", jev_threshold=0.5, hold_ms=1000))]
SWEEP = ([(f"규칙 hold={h}", GatePolicy("rule", hold_ms=h)) for h in (700, 1000, 1300, 1600)]
         + [(f"Jev thr={t} hold=1000", GatePolicy("jev", jev_threshold=t, hold_ms=1000)) for t in (0.3, 0.5, 0.7, 0.85)]
         + [(f"규칙+Jev thr={t} hold=1000", GatePolicy("rule_jev", jev_threshold=t, hold_ms=1000)) for t in (0.5, 0.7)])


def _fmt(d: dict) -> str:
    p = f"{d['p50']:.0f} / {d['p95']:.0f}" if "p50" in d else "-"
    return f"{d['fc_rate']*100:.1f}% ({d['fc']}) | {d['miss_rate']*100:.1f}% | {p}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--turnset", default="runs/endpoint/ev_turnset")
    ap.add_argument("--ts172", default="runs/endpoint/ev_ts172")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-jev", action="store_true", help="Jev 새 호출 금지(캐시만)")
    a = ap.parse_args(argv)
    sets = {}
    for name, run, loader in (("turnset", a.turnset, load_turnset), ("ts172", a.ts172, load_ts172)):
        if not Path(run, "rows.jsonl").exists():
            continue
        items = loader()
        if name == "ts172":
            _ctx_ts172(items)
        sets[name] = load_rows(Path(run), items)
        attach_vad(name, sets[name])
    jev = JevCache(call=not a.no_jev)
    reqs = [(t, it.get("history", []), it.get("caller_known", False))
            for pairs in sets.values() for it, row in pairs for t in end_texts(row["events"])]
    jev.fill(reqs)
    lines = ["| 데이터 | 방식 | 말 끝 전 확정 | miss | 지연 p50 / p95 ms |", "|---|---|---|---|---|"]
    detail: dict = {}
    for name, pairs in sets.items():
        uids = {it["uid"] for it, _ in pairs}
        rows = [("기존 결정기", baseline(name, uids)), ("Soniox 단독", stats(run_gate(pairs, None, jev)))]
        rows += [(lbl, stats(run_gate(pairs, pol, jev))) for lbl, pol in MAIN]
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
