"""Jev(TypeSafe System One) 턴 종료 판정 오프라인 평가(go/no-go 실험).

질문: 침묵 지점의 STT 텍스트(+대화 맥락)만으로 Jev Noul이 "말 중간 쉼"과 "말 끝"을 가르는가?
비교 대상: Smart Turn(쉼 vs 말 끝 AUC 0.57, 텍스트가 애매한 지점 0.48, docs/decisions/turn_calibration.md).

데이터
  - turnset: labels.jsonl의 text_stt가 있는 지점(test 430 전부, train 383). pause → 0, end → 1. 대화 맥락 없음.
  - ts172: testset172/stt_points.jsonl(말 중간 침묵 지점, 0) + end_texts.json(말 끝, 1). cases.jsonl의 history·고객 정보 포함.
  빈 텍스트 지점은 Jev가 판단할 근거가 없으므로 뺀다(실서버에서는 표 결정기 그대로).

명령
  uv run python -m eval.jev_turn_eval latency [--n 20]
  uv run python -m eval.jev_turn_eval run [--workers 4] [--limit N] [--model jev-1.14.0]
  uv run python -m eval.jev_turn_eval report
응답은 data/turnset/jev_cache.jsonl에 캐시(요청 본문 해시 키)하므로 다시 돌려도 새 지점만 호출한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from agent.turn import jev_turn as jt

ROOT = Path(__file__).resolve().parents[1]
TURNSET = ROOT / "data" / "turnset"
CACHE = TURNSET / "jev_cache.jsonl"
REPORT = TURNSET / "jev_eval_report.md"
URL = "https://api.typesafe.ai/v1/systemone"
MODEL = jt.MODEL

SITUATION, QUESTIONS = jt.SITUATION, jt.QUESTIONS   # 질문·상태는 agent/turn/jev_turn.py와 같다(캐시 키 공유)


@dataclass
class Point:
    src: str            # turnset | ts172
    key: str
    split: str          # train | test | ts172
    label: int          # 1 = 말 끝, 0 = 말 중간 쉼
    text: str
    cut_type: str = ""
    history: list = field(default_factory=list)
    caller_known: bool = False
    answers: dict | None = None
    ms: float | None = None


def load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _jsonl(p: Path):
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)


def load_points(with_gt: bool = False) -> list[Point]:
    pts: list[Point] = []
    for d in _jsonl(TURNSET / "labels.jsonl"):
        for i, p in enumerate(d["points"]):
            t = p.get("text_stt")
            src = "turnset"
            if t is None:                      # STT 없는 train 지점: 시뮬레이터(turn_sim.row_points)처럼 정답 텍스트
                t, src = p.get("text_gt") or "", "turnset_gt"
                if not with_gt:
                    continue
            if not t.strip():
                continue
            pts.append(Point(src, f"{d['uid']}:{i}", d["split"], int(p["kind"] == "end"), t.strip(),
                             p.get("cut_type", "end") if p["kind"] == "pause" else "end"))
    cases = {d["case_id"]: d for d in _jsonl(ROOT / "legacy" / "testset" / "cases.jsonl")}

    def ctx(cid: str) -> tuple[list, bool]:
        c = cases.get(cid, {})
        return list(c.get("history") or []), bool(c.get("customer_profile"))

    d172 = TURNSET / "testset172"
    for r in _jsonl(d172 / "stt_points.jsonl"):
        t = (r.get("text") or "").strip()
        if not t:
            continue
        cid = r["key"].split(":")[0]
        h, known = ctx(cid)
        pts.append(Point("ts172", r["key"], "ts172", 0, t, "mid", h, known))
    for cid, t in json.loads((d172 / "end_texts.json").read_text(encoding="utf-8")).items():
        t = (t or "").strip()
        if not t:
            continue
        h, known = ctx(cid)
        pts.append(Point("ts172", f"{cid}:end", "ts172", 1, t, "end", h, known))
    return pts


def body_for(p: Point, model: str) -> dict:
    return jt.build_body(p.text, p.history, p.caller_known, model)


def body_key(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:20]


class Jev:
    def __init__(self, timeout: float = 20.0):
        key = os.environ.get("TYPESAFE_API_KEY")
        if not key:
            sys.exit("TYPESAFE_API_KEY 없음(.env 또는 환경변수)")
        self._c = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {key}"},
                               limits=httpx.Limits(max_keepalive_connections=8, max_connections=8))

    def ask(self, body: dict, retries: int = 4) -> tuple[dict, float]:
        for a in range(retries + 1):
            t0 = time.perf_counter()
            r = self._c.post(URL, json=body)
            ms = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                return r.json(), ms
            if r.status_code in (429, 500, 502, 503, 504) and a < retries:
                time.sleep(min(8.0, 0.5 * 2 ** a))
                continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        raise RuntimeError("unreachable")


# ── 지표 ──
def auc(scores: list[float], labels: list[int]) -> float | None:
    """Mann-Whitney U / (n_pos·n_neg). 동점은 0.5."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    allv = sorted((s, i) for i, s in enumerate(pos + neg))
    ranks = [0.0] * len(allv)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[allv[k][1]] = (i + j) / 2 + 1
        i = j + 1
    rp = sum(ranks[: len(pos)])
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


# ── 명령 ──
def cmd_latency(n: int, model: str, concurrency: int = 1) -> None:
    """실서버와 같은 경로(typesafe_sdk AsyncTypeSafeClient, 연결 재사용, 재시도 없음)로 왕복 시간을 잰다.
    턴 판정(질문 3개)과 인계 판정(질문 6개)을 번갈아 n번씩. 첫 호출은 연결 수립(cold)이라 따로 적는다."""
    import asyncio

    from agent import handoff_jev as hj

    pts = [p for p in load_points() if p.src == "ts172"][:n]

    async def run():
        from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy
        out: dict[str, list[float]] = {"turn": [], "handoff": []}
        async with AsyncTypeSafeClient(retry=RetryPolicy(max_retries=0), timeout=10.0) as c:
            sem = asyncio.Semaphore(concurrency)

            async def one(kind, state, qs):
                async with sem:
                    t0 = time.perf_counter()
                    await c.system_one(state=state, questions=qs, model=model)
                    out[kind].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            await c.system_one(state=jt.build_state("네"), questions=jt.QUESTIONS, model=model)
            cold = (time.perf_counter() - t0) * 1000
            jobs = []
            for p in pts:
                jobs.append(lambda p=p: one("turn", jt.build_state(p.text, p.history, p.caller_known), jt.QUESTIONS))
                jobs.append(lambda p=p: one("handoff", hj.build_state(p.text, p.history), hj.CRITERIA))
            if concurrency == 1:
                for j in jobs:
                    await j()
            else:
                await asyncio.gather(*(j() for j in jobs))
        return cold, out

    cold, out = asyncio.run(run())
    print(f"cold {cold:.0f}ms (연결 수립 포함), 동시 {concurrency}")
    for k, v in out.items():
        print(f"{k:8s} n={len(v)} p50 {statistics.median(v):.0f} p95 {pct(v, 0.95):.0f} max {max(v):.0f} ms")
    print(f"calls {1 + sum(len(v) for v in out.values())}")


def load_cache() -> dict[str, dict]:
    return {r["k"]: r for r in _jsonl(CACHE)}


def cmd_run(workers: int, limit: int | None, model: str) -> None:
    pts = load_points(with_gt=True)
    cache = load_cache()
    bodies = {body_key(b): b for b in (body_for(p, model) for p in pts)}   # 같은 본문은 한 번만
    todo = [k for k in bodies if k not in cache]
    if limit is not None:
        todo = todo[:limit]
    print(f"points {len(pts)}, unique bodies {len(bodies)}, calls {len(todo)}")
    if not todo:
        return
    jev = Jev()
    lock = threading.Lock()
    done = err = 0
    with CACHE.open("a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(jev.ask, bodies[k]): k for k in todo}
        for fu in as_completed(futs):
            k = futs[fu]
            try:
                res, ms = fu.result()
            except Exception as e:  # noqa: BLE001
                err += 1
                print(f"err {k}: {e}", file=sys.stderr)
                continue
            ans = {q: a.get("noul") for q, a in res.get("answers", {}).items()}
            with lock:
                f.write(json.dumps({"k": k, "model": res.get("model"), "answers": ans, "ms": round(ms, 1),
                                    "usage": res.get("usage")}, ensure_ascii=False) + "\n")
                f.flush()
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{len(todo)}")
    print(f"done {done}, err {err}")


def cmd_report(model: str) -> None:
    from agent.turn.detector import CALIBRATED_CONFIG, table_kind
    from agent.turn.rules import analyze

    cache = load_cache()
    pts = load_points()
    for p in pts:
        r = cache.get(body_key(body_for(p, model)))
        if r:
            p.answers, p.ms = r["answers"], r["ms"]
    have = [p for p in pts if p.answers]
    lines = [f"# Jev 턴 종료 판정 오프라인 평가 ({model})", "",
             f"지점 {len(pts)}개 중 응답 {len(have)}개. 빈 텍스트 지점 제외. 라벨 1 = 말 끝, 0 = 말 중간 쉼.",
             "점수: `finished`(높을수록 끝), `1-cut_off`, `1-more_expected`, `avg` = 세 값 평균. "
             "기준선: 규칙 `end_score`(rules.analyze). Smart Turn은 전체 0.57, 애매 구간 0.48(T12 문서).", ""]

    def scores(p: Point) -> dict[str, float]:
        a = p.answers
        s = {"finished": a["finished"], "1-cut_off": 1 - a["cut_off"], "1-more_expected": 1 - a["more_expected"]}
        s["avg"] = sum(s.values()) / 3
        s["rule end_score"] = analyze(p.text).score
        return s

    names = ["finished", "1-cut_off", "1-more_expected", "avg", "rule end_score"]

    def row(title: str, ps: list[Point]) -> str:
        if not ps:
            return ""
        ys = [p.label for p in ps]
        cells = []
        for n in names:
            v = auc([scores(p)[n] for p in ps], ys)
            cells.append("-" if v is None else f"{v:.3f}")
        return f"| {title} | {len(ps)} | {sum(ys)}/{len(ys) - sum(ys)} | " + " | ".join(cells) + " |"

    hdr = "| 묶음 | n | 끝/쉼 | " + " | ".join(names) + " |\n|" + "---|" * (3 + len(names))
    lines += ["## AUC (쉼 vs 말 끝)", "", hdr]
    for t, sel in [("전체", lambda p: True), ("turnset test", lambda p: p.split == "test"),
                   ("turnset train", lambda p: p.split == "train"), ("ts172", lambda p: p.src == "ts172")]:
        lines.append(row(t, [p for p in have if sel(p)]))

    amb = {"short", "question", "dcomp", "final", "soft", "weak", "neutral", "comma"}
    for p in have:
        p.cut_type = p.cut_type or ""
    grp = {id(p): CALIBRATED_CONFIG.group(table_kind(p.text, CALIBRATED_CONFIG).kind) for p in have}
    lines += ["", "## 규칙 묶음별 AUC (r2 `CALIBRATED_CONFIG` 묶음)", "",
              "애매 구간 = short·question·dcomp·final·soft·weak·neutral·comma(규칙이 '끝 쪽'으로 보는 묶음, 잘못 끊기가 몰린 곳)", "",
              hdr]
    lines.append(row("**애매 구간 전체**", [p for p in have if grp[id(p)] in amb]))
    lines.append(row("애매 구간 · turnset test", [p for p in have if grp[id(p)] in amb and p.split == "test"]))
    lines.append(row("애매 구간 · ts172", [p for p in have if grp[id(p)] in amb and p.src == "ts172"]))
    for g in sorted(set(grp.values())):
        lines.append(row(g, [p for p in have if grp[id(p)] == g]))

    lines += ["", "## 쉼 유형별 `finished` 평균 (말 끝은 높고 쉼은 낮아야 함)", "", "| 유형 | n | finished 평균 | p≥0.7 비율 |",
              "|---|---|---|---|"]
    by: dict[str, list[float]] = {}
    for p in have:
        by.setdefault(f"{p.src}:{p.cut_type}", []).append(p.answers["finished"])
    for k in sorted(by):
        v = by[k]
        lines.append(f"| {k} | {len(v)} | {statistics.mean(v):.2f} | {sum(x >= 0.7 for x in v) / len(v):.0%} |")

    ms = [p.ms for p in have if p.ms]
    if ms:
        lines += ["", f"응답 시간(평가 실행 중, 동시 요청 포함): p50 {statistics.median(ms):.0f}ms, p95 {pct(ms, 0.95):.0f}ms"]

    # 애매 구간에서 Jev가 가장 크게 틀린 예
    wrong = sorted((p for p in have if grp[id(p)] in amb),
                   key=lambda p: -abs(p.answers["finished"] - p.label))[:20]
    lines += ["", "## 애매 구간 오답 상위 20 (finished와 라벨 차이 순)", "", "| key | 라벨 | finished | cut_off | more | 묶음 | 텍스트 |",
              "|---|---|---|---|---|---|---|"]
    for p in wrong:
        a = p.answers
        lines.append(f"| {p.key} | {p.label} | {a['finished']:.2f} | {a['cut_off']:.2f} | {a['more_expected']:.2f} | "
                     f"{grp[id(p)]} | {p.text.replace('|', '/')} |")
    REPORT.write_text("\n".join(l for l in lines if l is not None) + "\n", encoding="utf-8")
    print(REPORT.read_text(encoding="utf-8"))


# ── 표 결정기 + Jev 결합 시뮬레이션 ──
SIM_JSON = TURNSET / "jev_sim.json"
SIM_MD = TURNSET / "jev_sim_report.md"


class SimJevDetector:
    """시뮬레이터용 어댑터: smart_turn_p 자리에 Jev finished 확률을 실어 보낸다(eval.turn_sim.simulate_fast가
    침묵 200ms + 지연 프레임 뒤부터 p를 보여 준다). 판정은 jev_turn.decide_with_p와 같다."""

    def __init__(self, base, pol):
        self.base, self.pol, self.config = base, pol, base.config

    def wait_ms(self, s):
        w0, why = self.base.wait_ms(s)
        w, tag = jt.combine_wait(w0, self.base.cap_ms(s), s.smart_turn_p, self.pol)
        return w, why + tag

    def cap_ms(self, s):
        return self.base.cap_ms(s)


def attach_jev(items, params, lookup, hist):
    """침묵 구간마다 그 텍스트의 Jev p를 준비 프레임(침묵 200ms)에 붙인 사본. 텍스트가 비거나 캐시에 없으면 p 없음."""
    from dataclasses import replace as _r

    from eval import turn_sim as ts
    out, n_runs, n_p, n_missing = [], 0, 0, 0
    for it in items:
        sm = {}
        for r in ts.runs(it, params):
            if r.prep is None:
                continue
            n_runs += 1
            k = ts._point_index(it, r.start * ts.FRAME_S)
            text = "" if ts.text_empty(it, r.start, params) else (it.points[k].text if k >= 0 else "")
            if not text.strip():
                continue
            h, known = hist(it.uid)
            p = lookup(text, h, known)
            if p is None:
                n_missing += 1
                continue
            sm[r.prep] = p
            n_p += 1
        out.append(_r(it, smart=sm, smart_fn=None, _runs={}))
    return out, {"runs": n_runs, "with_p": n_p, "missing": n_missing}


def _summ(res) -> dict:
    import numpy as np
    a = np.asarray([r.latency_ms for r in res if r.outcome == "ok"], float)
    miss = sum(r.outcome == "miss" for r in res)
    return {"n": len(res), "false_cut": sum(r.outcome == "false_cut" for r in res), "miss": miss,
            "miss_rate": miss / len(res), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "mean": float(a.mean()), "fc_cases": sorted(r.uid for r in res if r.outcome == "false_cut")}


POLICY_GRID = [(hi, lo, f) for hi in (0.8, 0.85, 0.9, 0.93, 0.95, 1.01) for lo in (None, 0.1, 0.15, 0.2, 0.3)
               for f in (300, 400, 500, 600, 700) if not (hi > 1 and f != 300)]


def _pol(hi, lo, f) -> jt.JevTurnPolicy:
    return jt.JevTurnPolicy(hi=hi, lo=0.0 if lo is None else lo, fast_ms=f, extend=lo is not None)


def cmd_sim(model: str, latency_ms: list[int]) -> None:
    from dataclasses import replace as _r

    from agent.turn.detector import CALIBRATED_CONFIG, TableTurnDetector
    from eval import turn_sim as ts

    cache = load_cache()
    cases = {d["case_id"]: d for d in _jsonl(ROOT / "legacy" / "testset" / "cases.jsonl")}

    def lookup(text, h, known):
        r = cache.get(body_key(jt.build_body(text.strip(), h, known, model)))
        return None if r is None else r["answers"]["finished"]

    def hist_ts(uid):
        return [], False

    def hist_172(uid):
        c = cases.get(uid, {})
        return list(c.get("history") or []), bool(c.get("customer_profile"))

    params = ts.SessionParams(smart_delay_frames=2, empty_text_rate=0.10)      # r2 보정과 같은 조건
    p172 = _r(params, empty_text_rate=0.0)
    data = {"train": (ts.load_items(split="train"), params, False, hist_ts),
            "test": (ts.load_items(split="test"), params, False, hist_ts),
            "ts172": (ts.ts172_items(), p172, True, hist_172)}

    def run(items, fac, prm, is172):
        return ts.run_ts172(items, fac, prm) if is172 else ts.run_all(items, fac, prm)

    base = lambda: TableTurnDetector(CALIBRATED_CONFIG)  # noqa: E731
    out = {"model": model, "baseline": {k: _summ(run(v[0], base, v[1], v[2])) for k, v in data.items()},
           "latency_variants": {}}
    print("baseline", json.dumps({k: {x: v[x] for x in ("false_cut", "miss", "p50", "p95")}
                                  for k, v in out["baseline"].items()}), flush=True)
    for lat in latency_ms:
        frames = -(-lat // 20)
        sets, cover = {}, {}
        for name, (its, prm, is172, hist) in data.items():
            prm_l = _r(prm, smart_delay_frames=frames)
            its2, cover[name] = attach_jev(its, prm_l, lookup, hist)
            sets[name] = (its2, prm_l, is172)
        rows = []
        for hi, lo, f in POLICY_GRID:
            fac = lambda pol=_pol(hi, lo, f): SimJevDetector(TableTurnDetector(CALIBRATED_CONFIG), pol)  # noqa: E731
            rows.append({"policy": {"hi": hi, "lo": lo, "fast_ms": f},
                         **{name: _summ(run(v[0], fac, v[1], v[2])) for name, v in sets.items()}})
        out["latency_variants"][str(lat)] = {"coverage": cover, "rows": rows}
        print(f"latency {lat}ms coverage {cover}", flush=True)
    SIM_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    sim_report(out)


def pick(rows, base, key="train"):
    """train 기준 선택. 안전형 = miss·p95를 기준선보다 나쁘게 하지 않고 false cut 최소.
    속도형 = false cut·miss를 기준선 이하로 두고 p95 최소."""
    b = base[key]
    safe = [r for r in rows if r[key]["miss"] <= b["miss"] and r[key]["p95"] <= b["p95"] + 1e-6]
    fast = [r for r in rows if r[key]["false_cut"] <= b["false_cut"] and r[key]["miss"] <= b["miss"]]
    a = min(safe, key=lambda r: (r[key]["false_cut"], r[key]["p95"])) if safe else None
    f = min(fast, key=lambda r: (r[key]["p95"], r[key]["false_cut"])) if fast else None
    return a, f


def sim_report(out: dict | None = None) -> None:
    out = out or json.loads(SIM_JSON.read_text(encoding="utf-8"))
    base = out["baseline"]
    lines = [f"# 표 결정기(CALIBRATED_CONFIG) + Jev 결합 시뮬레이션 ({out['model']})", "",
             "조건: r2 보정과 같음(turnset 빈 텍스트 10%, 172는 0%). Jev p는 침묵 200ms에 그 순간 텍스트로 요청해 "
             "200+지연 ms에 도착. 정책: finished ≥ hi → 대기 min(표, fast_ms), finished ≤ lo → 대기를 상한까지"
             "(상한은 그대로). 선택은 train에서. 칸 = false cut / 지연 p50 / p95 / miss.", ""]

    def cells(r):
        return " | ".join(f"{r[k]['false_cut']} / {r[k]['p50']:.0f} / {r[k]['p95']:.0f} / {r[k]['miss_rate']:.1%}"
                          for k in ("train", "test", "ts172"))

    hdr = "| 결정기 | train (n {}) | test (n {}) | 172 (n {}) |\n|---|---|---|---|".format(
        base["train"]["n"], base["test"]["n"], base["ts172"]["n"])
    for lat, v in out["latency_variants"].items():
        rows = v["rows"]
        a, f = pick(rows, base)
        named = [("**안전형**(train 선택)", a), ("**속도형**(train 선택)", f)]
        for nm, pol in (("줄이기만 hi .9 fast 500", {"hi": 0.9, "lo": None, "fast_ms": 500}),
                        ("늘리기만 lo .2", {"hi": 1.01, "lo": 0.2, "fast_ms": 300}),
                        ("둘 다 hi .9 lo .2 fast 500", {"hi": 0.9, "lo": 0.2, "fast_ms": 500})):
            named.append((nm, next((r for r in rows if r["policy"] == pol), None)))
        lines += [f"## Jev 지연 {lat}ms (p 도착 = 침묵 {200 + int(lat)}ms)", "",
                  f"p 붙은 침묵 구간: {v['coverage']}", "", hdr, f"| 기준선(표만) | {cells(base)} |"]
        lines += [f"| {nm} `{r['policy']}` | {cells(r)} |" for nm, r in named if r]
        lines.append("")
        if a:
            lines += [f"안전형 172 false cut: {', '.join(a['ts172']['fc_cases'])}  ",
                      f"기준선 172 false cut: {', '.join(base['ts172']['fc_cases'])}", ""]
    SIM_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(SIM_MD.read_text(encoding="utf-8"))


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("latency")
    a.add_argument("--n", type=int, default=20)
    a.add_argument("--concurrency", type=int, default=1)
    a.add_argument("--model", default=MODEL)
    b = sub.add_parser("run")
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--limit", type=int)
    b.add_argument("--model", default=MODEL)
    c = sub.add_parser("report")
    c.add_argument("--model", default=MODEL)
    sm = sub.add_parser("sim")
    sm.add_argument("--model", default=MODEL)
    sm.add_argument("--latency", type=int, nargs="+", default=[0, 240, 400])
    sub.add_parser("sim-report")
    args = ap.parse_args()
    if args.cmd == "latency":
        cmd_latency(args.n, args.model, args.concurrency)
    elif args.cmd == "sim":
        cmd_sim(args.model, args.latency)
    elif args.cmd == "sim-report":
        sim_report()
    elif args.cmd == "run":
        cmd_run(args.workers, args.limit, args.model)
    else:
        cmd_report(args.model)


if __name__ == "__main__":
    main()
