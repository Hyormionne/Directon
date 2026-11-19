"""AG-T12 턴 결정기 보정. data/turnset(labels.jsonl + 특징 캐시)으로 train에서 고르고 test에서 보고한다.

  uv run python -m eval.turn_calibrate [--text stt|gt] [--target 0.01]

하는 일
 1. 베이스라인: DEFAULT_CONFIG(RuleTurnDetector)
 2. TurnConfig 좌표 탐색(fast/mid/slow/hold/p_high/p_low)
 3. 표 방식(TableTurnConfig): 텍스트 묶음 × Smart Turn 구간별 대기 시간을 라그랑주 완화로 고른다
      셀마다 Σ(진짜 끝 지연) + λ·Σ(쉼에서 끊김)을 최소화하는 대기 → λ를 올려 train false cut ≤ 목표를 맞춘다
    상한 예외(연결어미·간투사·숫자 미완 텍스트는 hard_max 대신 exempt_max까지) 켠/끈 변형을 같이 본다
 4. 신호 유용성: 쉼 지점(미완료) vs 최종 끝(완료)에서 Smart Turn p·end_score·결합 로지스틱의 AUC
결과: data/turnset/calibration.json, data/turnset/calibration_report.md(문서용 표)
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from agent.core.types import TurnSignals
from agent.turn.detector import (DEFAULT_CONFIG, KIND_GROUPS_DEFAULT, RuleTurnDetector, TableTurnConfig, table_kind,
                                 TableTurnDetector, TurnConfig)
from agent.turn.rules import analyze
from eval import turn_sim as ts

OUT_JSON = ts.TURNSET / "calibration.json"
OUT_MD = ts.TURNSET / "calibration_report.md"
FRAME = 20
WAIT_GRID = list(range(200, 1101, 20))
EXEMPT_GRID = list(range(1100, 2101, 100))
GROUPS = ("strong", "final", "soft", "weak", "neutral", "particle", "conn", "filler", "dinc")
# 보정용 묶음: 연결어미·간투사·숫자 미완을 따로 둔다(숫자 미완은 규칙 오분류가 있어 상한 예외에서 뺀다)
KIND_GROUPS_CAL = (
    ("short_answer", "strong"), ("question", "strong"), ("digits_complete", "strong"),
    ("final", "final"), ("soft_final", "soft"), ("weak_final", "weak"), ("neutral", "neutral"),
    ("particle", "particle"), ("connective", "conn"), ("filler", "filler"), ("digits_incomplete", "dinc"),
    ("digits_partial", "dinc"),
)
HOLD_GROUPS = ("conn", "filler")
REFINE_GRID = list(range(200, 1101, 50))
# 과적합 방지 하한: 보류 성격 텍스트·빈/중립 텍스트는 빠른 확정 금지(실서버 스모크: 빈 텍스트·neutral에서 400ms 확정 사고)
# 사전 하한(데이터 밖 근거): 완결 텍스트도 문장 사이 쉼(edge 400~600ms)에서 끊긴 스모크 결과 → final ≥ 700,
# "-는데요/-거든요"(soft)는 데이터에 쉼 사례가 없어 700, weak("주문번호가요", ≤4자리+요) 600, 단답·물음표 300.
# 첫 소리가 확정 후 ~10ms라 대기 ~1000ms까지는 속도 점수 손실이 없다.
FLOORS = {"conn": 900, "filler": 900, "dinc": 900, "particle": 800, "neutral": 800,
          "final": 700, "soft": 700, "weak": 600, "strong": 300}
REFINE_TOP = 3


# ───────────────────────── 평가 ─────────────────────────
def evaluate(items, det_factory, params=ts.SessionParams(), groups=True):
    res = ts.run_all(items, det_factory, params)
    s = ts.summarize(res, {it.uid: it.meta for it in items},
                     group_keys=("type", "channel", "noise", "noise_snr", "speed", "gain_db", "voice") if groups else ())
    s["false_cut_avoidable_rate"] = avoidable(res, items)
    return res, s


def avoidable(res, items, cap_ms=1100):
    """쉼 길이가 상한 이하인 곳에서의 false cut(상한 1.1초에서는 구조적으로 못 피하는 긴 쉼 제외)."""
    by = {it.uid: it for it in items}
    n = 0
    for r in res:
        if r.outcome != "false_cut":
            continue
        it = by[r.uid]
        pause = next((p for p in it.points if p.kind == "pause" and p.t - 0.1 <= r.decision_t <= p.t + p.dur_ms / 1000 + 0.1), None)
        if pause is None or pause.dur_ms <= cap_ms:
            n += 1
    return n / len(res) if res else 0.0


def brief(s, res=None, items=None):
    d = {k: s[k] for k in ("n", "false_cut", "false_cut_rate", "false_cut_avoidable_rate", "miss_rate",
                           "latency_mean", "latency_p50", "latency_p95", "hard_max_rate", "over_1100_rate",
                           "over_cap_rate")}
    d["false_cut_where"] = s.get("false_cut_where", {})
    return d


P95_MAX = 1100.0          # train 제약(완료 지점 지연 p95)
MISS_SLACK = 0.015


def objective(s, miss_max, p95_max=P95_MAX):
    """제약(완료 지점 지연 p95 ≤ p95_max, miss ≤ miss_max, 상한 넘는 대기 ≤ 2%) 하 false cut 최소 → 평균 지연."""
    if (s["latency_p95"] or 9e9) > p95_max or s["miss_rate"] > miss_max + 1e-9 or s["over_cap_rate"] > 0.02:
        return (1, round(s["latency_p95"] or 9e9, 1), s["false_cut_rate"])
    return (0, round(s["false_cut_rate"], 5), round(s["latency_mean"] or 9e9, 1))


# ───────────────────────── 2. TurnConfig 좌표 탐색 ─────────────────────────
COORD_GRID = {
    "fast_ms": list(range(200, 701, 40)),
    "mid_ms": list(range(240, 1101, 40)),
    "slow_ms": list(range(400, 1101, 50)),
    "hold_ms": [900, 1000, 1100],
    "p_high": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    "p_low": [0.05, 0.1, 0.2, 0.3, 0.4],
}


def coord_search(train, miss_max, start=DEFAULT_CONFIG, passes=3, params=ts.SessionParams()):
    best = start
    _, s = evaluate(train, lambda: RuleTurnDetector(best), params, groups=False)
    best_sc = objective(s, miss_max)
    for _ in range(passes):
        improved = False
        for k, grid in COORD_GRID.items():
            for v in grid:
                c = replace(best, **{k: v})
                if c.p_low > c.p_high:
                    continue
                _, s = evaluate(train, lambda c=c: RuleTurnDetector(c), params, groups=False)
                sc = objective(s, miss_max)
                if sc < best_sc:
                    best, best_sc, improved = c, sc, True
        if not improved:
            break
    return best


# ───────────────────────── 3. 표 방식(라그랑주) ─────────────────────────
def run_records(items, cfg: TableTurnConfig, params=ts.SessionParams()):
    """침묵 구간마다 (종류, 셀, 이용 가능 침묵 길이, uid, 끝 대비 시작 ms). 종류: cut(끝 전) | final(끝 뒤 첫 구간)."""
    recs = []
    for it in items:
        final_seen = False
        for r in ts.runs(it, params):
            if r.prep is None:
                continue      # 침묵 200ms 미만 구간은 결정 불가
            start_t = r.start * ts.FRAME_S
            text = ts.text_at(it, r.prep, (r.prep - r.start + 1) * FRAME, params)   # 빈 텍스트 반영
            if r.speech_ms < cfg.min_speech_ms and not text.strip():
                continue
            p = it.smart_p(r.prep)
            kind = table_kind(text, cfg).kind
            cell = (cfg.group(kind), cfg.p_bin(p), cfg.s_bin(r.speech_ms))
            L = r.n * FRAME
            if start_t + 0.2 < it.end_t:      # 끝 전 구간(쉼·말 중간): 침묵 L ≥ wait면 끊김
                recs.append(("cut", cell, L, it.uid, 0.0))
            elif not final_seen:
                final_seen = True
                recs.append(("final", cell, L, it.uid, (start_t - it.end_t) * 1000))
    return recs


def lagrange_table(recs, lam, exempt_groups=(), exempt_max=1100, latency_cap=1100, long_cut_weight=0.0):
    """셀별 대기 선택: Σ지연 + λ·Σ끊김 최소. 1.1초 넘는 쉼의 끊김은 long_cut_weight 배로만 센다
    (상한 1.1초면 못 피하므로 0, 상한 예외 묶음은 피할 수 있어 1)."""
    cells = {}
    for kind, cell, L, uid, off in recs:
        cells.setdefault(cell, []).append((kind, L, off))
    waits = {}
    for cell, rs in cells.items():
        g = cell[0]
        ex = g in exempt_groups
        grid = WAIT_GRID + ([w for w in EXEMPT_GRID if 1100 < w <= exempt_max] if ex else [])
        grid = [w for w in grid if w >= FLOORS.get(g, 0)]
        cuts = np.array([L for k, L, _ in rs if k == "cut"], float)
        wts = np.where(cuts > 1100, 1.0 if ex else long_cut_weight, 1.0) if len(cuts) else cuts
        fin = [(L, off) for k, L, off in rs if k == "final"]
        best, best_c = None, None
        for w in grid:
            c = lam * float(wts[cuts >= w].sum()) if len(cuts) else 0.0
            for L, off in fin:
                c += off + (w if w <= L else max(w, latency_cap) + 500)
            if best_c is None or c < best_c - 1e-9:
                best, best_c = w, c
        waits[cell] = best
    return waits


def table_config(waits: dict, base: TableTurnConfig, exempt_groups=(), exempt_max=2000, default_wait=1100):
    np_ = len(base.p_edges) + 1
    ns = len(base.speech_edges) + 1
    rows = []
    for g in GROUPS:
        known = [w for (gg, _, _), w in waits.items() if gg == g]
        fill = max(known) if known else max(default_wait, FLOORS.get(g, 0))
        grid = [[waits.get((g, pb, sb), fill) for sb in range(ns)] for pb in range(1, np_ + 1)]
        # 같은 음성 길이 구간에서 p가 높을수록 대기가 길어지지 않게(단조)
        for sb in range(ns):
            for pb in range(1, np_):
                grid[pb][sb] = min(grid[pb][sb], grid[pb - 1][sb])
        none_row = [max(grid[pb][sb] for pb in range(np_)) for sb in range(ns)]   # p 없음 = 가장 보수적
        flat = none_row + [w for row in grid for w in row]
        rows.append((g, tuple(int(w) for w in flat)))
    return replace(base, waits=tuple(rows), exempt_groups=tuple(exempt_groups), exempt_max_ms=exempt_max)


LAMBDAS = np.geomspace(30, 3e5, 40)


def table_variants(train, edges_list, speech_list, exempt_options, params):
    """(p 구간, 음성 길이 구간, 상한 예외) 조합마다 λ를 훑어 train 지표 곡선을 만든다."""
    out = []
    for edges, sp, (ex_groups, ex_max) in itertools.product(edges_list, speech_list, exempt_options):
        base = TableTurnConfig(p_edges=edges, speech_edges=sp, kind_groups=KIND_GROUPS_CAL)
        recs = run_records(train, base, params)
        seen = set()
        for lam in LAMBDAS:
            cfg = table_config(lagrange_table(recs, lam, ex_groups, ex_max), base, ex_groups, ex_max)
            if cfg.waits in seen:
                continue
            seen.add(cfg.waits)
            _, s = evaluate(train, lambda cfg=cfg: TableTurnDetector(cfg), params, groups=False)
            out.append({"cfg": cfg, "lam": float(lam), "train": s})
    return out


def _with_cell(cfg: TableTurnConfig, g: str, idx: int, w: int) -> TableTurnConfig:
    """묶음 g의 (p 구간, 음성 구간) 칸 idx(p 없음 제외 평탄 인덱스)를 w로 바꾸고 p 없음 열을 다시 맞춘다."""
    ns = len(cfg.speech_edges) + 1
    rows = []
    for gg, flat in cfg.waits:
        flat = list(flat)
        if gg == g:
            flat[ns + idx] = w
            body = flat[ns:]
            np_ = len(body) // ns
            for sb in range(ns):
                flat[sb] = max(body[pb * ns + sb] for pb in range(np_))
        rows.append((gg, tuple(flat)))
    return replace(cfg, waits=tuple(rows))


def refine(train, cfg: TableTurnConfig, miss_max, params, passes=2):
    """칸별 좌표 탐색: 목적(objective)을 직접 최소화. 라그랑주 해에서 시작."""
    _, s = evaluate(train, lambda: TableTurnDetector(cfg), params, groups=False)
    best, best_o = cfg, objective(s, miss_max)
    ns = len(cfg.speech_edges) + 1
    n_cells = (len(cfg.p_edges) + 1) * ns
    for _ in range(passes):
        improved = False
        for g in GROUPS:
            grid = REFINE_GRID + ([1300, 1500, 1700, 2000] if g in cfg.exempt_groups else [])
            grid = [w for w in grid if FLOORS.get(g, 0) <= w <= max(cfg.exempt_max_ms, cfg.hard_max_ms)]
            for idx in range(n_cells):
                for w in grid:
                    c = _with_cell(best, g, idx, w)
                    if c.waits == best.waits:
                        continue
                    _, s = evaluate(train, lambda c=c: TableTurnDetector(c), params, groups=False)
                    o = objective(s, miss_max)
                    if o < best_o:
                        best, best_o, improved = c, o, True
        if not improved:
            break
    return best, best_o


def pick(variants, miss_max):
    ok = [v for v in variants if objective(v["train"], miss_max)[0] == 0]
    if not ok:
        return None
    return min(ok, key=lambda v: objective(v["train"], miss_max))


# ───────────────────────── 4. 신호 유용성(AUC) ─────────────────────────
def auc(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    # 동점 평균 순위
    i = 0
    r = np.arange(1, len(allv) + 1, dtype=float)
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        r[i:j + 1] = (i + j + 2) / 2
        i = j + 1
    ranks[order] = r
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def point_features(items, params=ts.SessionParams()):
    """준비(침묵 200ms)가 일어난 쉼·최종 끝 구간의 특징. y=1 완료(최종 끝), 0 미완료(쉼)."""
    X, y, meta = [], [], []
    for it in items:
        final_seen = False
        for r in ts.runs(it, params):
            if r.prep is None:
                continue
            start_t = r.start * ts.FRAME_S
            k = ts._point_index(it, start_t)
            if k < 0:
                continue
            pt = it.points[k]
            if pt.kind == "pause" and start_t <= pt.t + pt.dur_ms / 1000:
                lab = 0
            elif pt.kind == "end" and not final_seen:
                lab, final_seen = 1, True
            else:
                continue
            a = analyze(pt.text)
            X.append([it.smart_p(r.prep), a.score, np.log1p(r.speech_ms / 1000)])
            y.append(lab)
            meta.append({"kind": a.kind, "cut_type": pt.cut_type, "uid": it.uid})
    return np.array(X, float), np.array(y), meta


def logistic_fit(X, y, l2=1e-2, iters=3000, lr=0.5):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]]
        w -= lr * g
    return w, mu, sd


def logistic_predict(model, X):
    w, mu, sd = model
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    return 1 / (1 + np.exp(-Z @ w))


def signal_report(train, test):
    Xtr, ytr, _ = point_features(train)
    Xte, yte, mte = point_features(test)
    out = {"n_train": int(len(ytr)), "n_test": int(len(yte)), "pos_rate_test": float(yte.mean()) if len(yte) else None}
    out["auc_smart_turn"] = auc(yte, Xte[:, 0])
    out["auc_end_score"] = auc(yte, Xte[:, 1])
    m = logistic_fit(Xtr, ytr)
    out["auc_logistic(p,score,speech)"] = auc(yte, logistic_predict(m, Xte))
    m2 = logistic_fit(Xtr[:, 1:], ytr)
    out["auc_logistic(score,speech)"] = auc(yte, logistic_predict(m2, Xte[:, 1:]))
    out["logistic_weights(bias,p,score,log_speech)"] = [round(float(v), 3) for v in m[0]]
    # 텍스트가 애매한(final/soft/weak/neutral) 지점에서 Smart Turn만의 AUC
    amb = np.array([mm["kind"] in ("final", "soft_final", "weak_final", "neutral") for mm in mte])
    if amb.any():
        out["auc_smart_turn_ambiguous_text"] = auc(yte[amb], Xte[amb, 0])
        out["n_ambiguous_text"] = int(amb.sum())
    # 쉼 유형별 Smart Turn 평균 p
    pt = {}
    for (row, lab, mm) in zip(Xte, yte, mte):
        key = "END" if lab == 1 else mm["cut_type"]
        pt.setdefault(key, []).append(row[0])
    out["smart_p_mean_by_point"] = {k: round(float(np.mean(v)), 3) for k, v in sorted(pt.items())}
    return out


# ───────────────────────── 보고 ─────────────────────────
def _pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def _ms(x):
    return "-" if x is None else f"{x:.0f}"


def md_table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def group_table(name, sums: dict):
    keys = sorted(set().union(*[set(s["by"][name]) for s in sums.values()]))
    rows = []
    for k in keys:
        r = [k]
        for label, s in sums.items():
            g = s["by"][name].get(k)
            r += ["-", "-", "-"] if g is None else [f"{g['false_cut']}/{g['n']} ({_pct(g['false_cut_rate'])})",
                                                    _ms(g["latency_p50"]), _ms(g["latency_p95"])]
        rows.append(r)
    header = [name] + [f"{l} {c}" for l in sums for c in ("false cut", "p50", "p95")]
    return md_table(rows, header)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="stt", choices=["stt", "gt"])
    ap.add_argument("--target", type=float, default=0.01)
    ap.add_argument("--smart-delay", type=int, default=2, help="Smart Turn 결과가 보이기까지 프레임(세션 ~25-40ms)")
    ap.add_argument("--empty-text", type=float, default=0.35, help="확정 시점 STT 텍스트가 빈 비율(실서버 7/20)")
    a = ap.parse_args(argv)
    params = ts.SessionParams(smart_delay_frames=a.smart_delay, empty_text_rate=a.empty_text)
    train = ts.load_items(split="train", text_source=a.text)
    test = ts.load_items(split="test", text_source=a.text)
    print("items", len(train), len(test), flush=True)

    def ev(items, fac):
        return evaluate(items, fac, params)

    results = {"text_source": a.text, "target": a.target, "params": asdict(params)}
    # 1. 베이스라인
    base = {}
    for name, items in (("train", train), ("test", test)):
        res, s = ev(items, lambda: RuleTurnDetector(DEFAULT_CONFIG))
        base[name] = (res, s)
    results["baseline"] = {k: brief(v[1], v[0], train if k == "train" else test) for k, v in base.items()}
    print("baseline", json.dumps(results["baseline"]["test"], ensure_ascii=False), flush=True)

    # 2. 좌표 탐색
    miss_max = results["baseline"]["train"]["miss_rate"] + MISS_SLACK
    cc = coord_search(train, miss_max, params=params)
    coord = {k: ev(items, lambda: RuleTurnDetector(cc)) for k, items in (("train", train), ("test", test))}
    results["coord"] = {"config": asdict(cc), **{k: brief(v[1], v[0], train if k == "train" else test) for k, v in coord.items()}}
    print("coord", json.dumps(results["coord"]["test"], ensure_ascii=False), flush=True)

    # 3. 표 방식
    # Smart Turn은 AUC ~0.55(쓸모 없음) → p 축은 빼고 비교용으로 (0.5,) 하나만 둔다
    edges_list = [(), (0.5,)]
    speech_list = [(), (1000, 2000), (800, 1600, 2400)]
    strict_opts = [((), 1100)]
    # 숫자 미완(dinc)은 상한 예외에서 뺀다: STT가 숫자 하나를 빠뜨리거나("667173382요") '에'를 '의'로 들으면
    # 진짜 말 끝이 미완으로 보여 2초를 기다린다(2026-11-19 test에서 miss +5%p 확인).
    exempt_opts = [(HOLD_GROUPS, 2000), (HOLD_GROUPS + ("particle",), 2000)]
    strict_vars = table_variants(train, edges_list, speech_list, strict_opts, params)
    exempt_vars = table_variants(train, edges_list, speech_list, exempt_opts, params)
    chosen = {}
    for name, vars_ in (("table_strict", strict_vars), ("table", exempt_vars)):
        # 조합별 라그랑주 최선 → 칸별 좌표 탐색 → train 목적이 가장 좋은 것(동률이면 칸 수가 적은 것)
        combos = {}
        for v in vars_:
            c = v["cfg"]
            combos.setdefault((c.p_edges, c.speech_edges, c.exempt_groups, c.exempt_max_ms), []).append(v)
        refined = []
        starts = []
        for key, vs in combos.items():
            v0 = pick(vs, miss_max)
            if v0 is None:
                v0 = min(vs, key=lambda v: objective(v["train"], miss_max))
            starts.append((objective(v0["train"], miss_max), key, v0))
        for _, key, v0 in sorted(starts, key=lambda x: x[0])[:REFINE_TOP]:   # 라그랑주 상위만 다듬는다
            c, o = refine(train, v0["cfg"], miss_max, params)
            refined.append((o, (len(c.p_edges) + 1) * (len(c.speech_edges) + 1), c, v0["lam"]))
            print(name, key, o, flush=True)
        feas = [r for r in refined if r[0][0] == 0]
        if not feas:
            results[name] = None
            continue
        best_fc = min(r[0][1] for r in feas)
        o, _, cfg, lam = min((r for r in feas if r[0][1] <= best_fc + 0.005), key=lambda r: (r[1], r[0]))
        v = {"lam": lam}
        tab = {k: ev(items, lambda cfg=cfg: TableTurnDetector(cfg)) for k, items in (("train", train), ("test", test))}
        chosen[name] = (cfg, tab)
        results[name] = {"config": asdict(cfg), "lambda": v["lam"],
                         **{k: brief(t[1]) for k, t in tab.items()}}
        print(name, json.dumps(results[name]["test"], ensure_ascii=False), flush=True)
    # 파레토 곡선(train 기준 비지배 해 → test 지표)
    front = []
    for name, vars_ in (("strict", strict_vars), ("exempt", exempt_vars)):
        pts = sorted(vars_, key=lambda v: (v["train"]["latency_p95"] or 9e9, v["train"]["false_cut_rate"]))
        best_fc = 9.0
        for v in pts:
            bad = (v["train"]["over_cap_rate"] > 0.02 or v["train"]["miss_rate"] > miss_max
                   or v["train"]["false_cut_rate"] >= best_fc - 1e-9 or (v["train"]["latency_p95"] or 9e9) > 1300)
            if bad:
                continue
            best_fc = v["train"]["false_cut_rate"]
            _, st = ev(test, lambda cfg=v["cfg"]: TableTurnDetector(cfg))
            front.append({"mode": name, "p_edges": v["cfg"].p_edges, "speech_edges": v["cfg"].speech_edges,
                          "exempt": [v["cfg"].exempt_groups, v["cfg"].exempt_max_ms],
                          "train": brief(v["train"]), "test": brief(st)})
    results["pareto"] = front
    best = (None, chosen["table"][0]) if "table" in chosen else ((None, chosen["table_strict"][0]) if "table_strict" in chosen else None)
    if best is not None:
        tab = chosen.get("table", chosen.get("table_strict"))[1]
    # 4. 신호 유용성
    results["signals"] = signal_report(train, test)
    print("signals", json.dumps(results["signals"], ensure_ascii=False), flush=True)
    # 텍스트 지연 민감도(최종 설정, test)
    if best is not None:
        sens = {}
        for label, p2 in (("빈 텍스트 0%", replace(params, empty_text_rate=0.0)),
                          ("빈 텍스트 35%(기준)", params),
                          ("빈 텍스트 100%(오디오만)", replace(params, empty_text_rate=1.0)),
                          ("빈 35% + 텍스트 지연 300ms", replace(params, text_lag_ms=300)),
                          ("Smart Turn 지연 0프레임", replace(params, smart_delay_frames=0))):
            sb = {}
            for nm, fac in (("베이스라인", lambda: RuleTurnDetector(DEFAULT_CONFIG)),
                            ("채택", lambda: TableTurnDetector(best[1]))):
                _, st = evaluate(test, fac, p2, groups=False)
                sb[nm] = brief(st)
            sens[label] = sb
        results["text_lag_sensitivity_test"] = sens
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    # 보고서 표
    md = [f"# 턴 보정 결과 (text={a.text}, smart_delay={a.smart_delay}프레임, 빈 텍스트 {a.empty_text:.0%}, "
          f"목표: false cut 최소 s.t. 완료 지연 p95 ≤ {P95_MAX:.0f}ms, miss ≤ 베이스라인+{MISS_SLACK:.1%}p)\n"]
    rows = []
    for label, key in (("베이스라인(DEFAULT_CONFIG)", "baseline"), ("좌표 탐색(TurnConfig)", "coord"),
                       ("표 방식 엄격(상한 1.1초)", "table_strict"), ("표 방식 채택", "table")):
        r = results.get(key)
        if not r:
            continue
        for split in ("train", "test"):
            s = r[split]
            rows.append([label, split, s["n"], f"{s['false_cut']} ({_pct(s['false_cut_rate'])})",
                         _pct(s.get("false_cut_avoidable_rate")), _ms(s["latency_mean"]), _ms(s["latency_p50"]),
                         _ms(s["latency_p95"]), _pct(s["hard_max_rate"]), _pct(s["over_cap_rate"]), _pct(s["miss_rate"])])
    md.append(md_table(rows, ["결정기", "split", "n", "false cut", "피할 수 있는 false cut", "지연 평균", "p50", "p95",
                              "상한 도달", "침묵>1.1s 대기", "miss"]))
    if best is not None:
        sums = {"베이스라인": base["test"][1], "채택": tab["test"][1]}
        for g, title in (("type", "유형별"), ("channel", "채널별"), ("noise", "잡음별"), ("noise_snr", "SNR별"),
                         ("speed", "속도별"), ("gain_db", "이득별"), ("voice", "음성별")):
            md.append(f"\n## test {title}\n")
            md.append(group_table(g, sums))
        for nm in ("table_strict", "table"):
            if nm not in chosen:
                continue
            c = chosen[nm][0]
            ns = len(c.speech_edges) + 1
            md.append(f"\n## {nm} 표(대기 ms)\n")
            md.append(f"p_edges={c.p_edges}, speech_edges={c.speech_edges}, exempt={c.exempt_groups}->{c.exempt_max_ms}ms. "
                      f"열 pXsY: X=0 p 없음, X>=1 p 구간 / Y = 음성 길이 구간\n")
            md.append(md_table([[g, *w] for g, w in c.waits], ["묶음"] +
                               [f"p{i // ns}s{i % ns}" for i in range(len(c.waits[0][1]))]))
        if results.get("pareto"):
            md.append("\n## 파레토(train 비지배 -> test)\n")
            md.append(md_table([[f["mode"], f["p_edges"], f["speech_edges"], f"{f['exempt'][0]}->{f['exempt'][1]}",
                                 _pct(f["train"]["false_cut_rate"]), _ms(f["train"]["latency_p95"]),
                                 f"{f['test']['false_cut']} ({_pct(f['test']['false_cut_rate'])})",
                                 _pct(f["test"]["false_cut_avoidable_rate"]), _ms(f["test"]["latency_mean"]),
                                 _ms(f["test"]["latency_p50"]), _ms(f["test"]["latency_p95"]),
                                 _pct(f["test"]["miss_rate"])] for f in results["pareto"]],
                               ["모드", "p_edges", "speech_edges", "예외", "train fc", "train p95",
                                "test fc", "test 피할수있는 fc", "test 평균", "test p50", "test p95", "test miss"]))
        md.append("\n## false cut 위치(test)\n")
        md.append(f"베이스라인 {base['test'][1]['false_cut_where']} / 채택 {tab['test'][1]['false_cut_where']}")
    md.append("\n## 신호 유용성(test, 쉼=0 / 최종 끝=1)\n")
    md.append("```\n" + json.dumps(results["signals"], ensure_ascii=False, indent=1) + "\n```")
    if "text_lag_sensitivity_test" in results:
        md.append("\n## 민감도(test)\n")
        md.append(md_table([[k, nm, f"{v['false_cut']} ({_pct(v['false_cut_rate'])})", _ms(v["latency_mean"]),
                             _ms(v["latency_p50"]), _ms(v["latency_p95"]), _pct(v["miss_rate"])]
                            for k, d in results["text_lag_sensitivity_test"].items() for nm, v in d.items()],
                           ["조건", "결정기", "false cut", "평균", "p50", "p95", "miss"]))
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote", OUT_JSON, OUT_MD)


if __name__ == "__main__":
    main()
