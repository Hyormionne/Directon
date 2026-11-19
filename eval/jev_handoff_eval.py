"""AG-T24 Jev 사람 인계 판정 오프라인 평가.

정답: 문항 expected_tools에 transfer_to_agent가 있으면 인계(1), 없으면 0.
  - testset 172문항: 고객 발화 = 실서버 STT 최종 텍스트(data/turnset/testset172/end_texts.json, 없으면 원문) + history
  - practice_health 40문항(있으면): 원문 발화 + history (STT 없음)
비교: round1 실서버 LLM(runs/round1_172_2003/results.jsonl의 tool_calls에 transfer_to_agent).
결합 효과: "LLM이 부르거나 Jev가 인계라 하면 인계"(연결안 = Jev가 인계라 하면 transfer_to_agent 강제)의 정밀도·재현율.

명령
  uv run python -m eval.jev_handoff_eval run [--workers 4]
  uv run python -m eval.jev_handoff_eval report
응답 캐시: runs/jev_handoff_cache.jsonl(본문 해시 키).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agent import handoff_jev as hj
from eval.jev_turn_eval import Jev, _jsonl, auc, load_dotenv

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs" / "jev_handoff_cache.jsonl"
REPORT = ROOT / "data" / "turnset" / "jev_handoff_report.md"
ROUND1 = ROOT / "runs" / "round1_172_2003" / "results.jsonl"


def _has_tool(tools, name: str) -> bool:
    return any((t or {}).get("name") == name for t in (tools or []))


def _has_transfer(tools) -> bool:
    return _has_tool(tools, "transfer_to_agent")


def jev_route(c: dict, th: dict | None = None) -> str | None:
    """코드 정책까지 적용한 Jev 결과: None | transfer | callback(영업시간 외)."""
    ok, why = hj.decide(c["probs"], th)
    return hj.route(hj.HandoffJudgment(c["probs"], ok, why), c["now"])


def load_cases() -> list[dict]:
    out = []
    ends_p = ROOT / "data" / "turnset" / "testset172" / "end_texts.json"
    ends = json.loads(ends_p.read_text(encoding="utf-8")) if ends_p.exists() else {}
    r1 = {r["case_id"]: r for r in _jsonl(ROUND1)}
    for c in _jsonl(ROOT / "data" / "testset" / "cases.jsonl"):
        cid = c["case_id"]
        text = (ends.get(cid) or "").strip() or c["customer_utterance"]
        llm = _has_transfer(r1[cid]["tool_calls"]) if cid in r1 else None
        out.append({"set": "testset", "id": cid, "category": c.get("category"), "text": text,
                    "gt": c["customer_utterance"], "history": c.get("history") or [],
                    "known": bool(c.get("customer_profile")), "label": int(_has_transfer(c.get("expected_tools"))),
                    "llm": llm, "now": c.get("now", ""), "callback": _has_tool(c.get("expected_tools"), "request_callback")})
    for c in _jsonl(ROOT / "data" / "practice_health" / "cases.jsonl"):
        out.append({"set": "health", "id": c["case_id"], "category": c.get("category"), "text": c["customer_utterance"],
                    "gt": c["customer_utterance"], "history": c.get("history") or [],
                    "known": bool(c.get("customer_profile")), "label": int(_has_transfer(c.get("expected_tools"))),
                    "llm": None, "now": c.get("now", ""), "callback": _has_tool(c.get("expected_tools"), "request_callback")})
    return out


def body(c: dict) -> dict:
    return hj.build_body(c["text"], c["history"], 0, c["known"])


def key(b: dict) -> str:
    return hashlib.sha256(json.dumps(b, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:20]


def load_cache() -> dict:
    return {r["k"]: r for r in _jsonl(CACHE)}


def cmd_run(workers: int) -> None:
    cache = load_cache()
    bodies = {key(b): b for b in (body(c) for c in load_cases())}
    todo = [k for k in bodies if k not in cache]
    print(f"bodies {len(bodies)}, calls {len(todo)}")
    if not todo:
        return
    jev = Jev()
    lock = threading.Lock()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
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
    print(f"done {done}, err {err}")


def prf(pred: list[int], y: list[int]) -> tuple[int, int, int, float, float]:
    tp = sum(p and t for p, t in zip(pred, y))
    fp = sum(p and not t for p, t in zip(pred, y))
    fn = sum((not p) and t for p, t in zip(pred, y))
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    return tp, fp, fn, prec, rec


def cmd_report() -> None:
    cache = load_cache()
    cs = []
    for c in load_cases():
        r = cache.get(key(body(c)))
        if r:
            c["probs"] = {k: float(r["answers"].get(k) or 0.0) for k in hj.CRITERIA}
            c["ms"] = r["ms"]
            cs.append(c)
    L = [f"# Jev 사람 인계 판정 오프라인 평가 ({hj.MODEL})", "",
         f"문항 {len(cs)}개(testset {sum(c['set'] == 'testset' for c in cs)}, practice_health "
         f"{sum(c['set'] == 'health' for c in cs)}). 정답 = expected_tools에 transfer_to_agent. "
         f"임계값 {hj.THRESHOLDS}.", ""]

    def block(title, sub):
        y = [c["label"] for c in sub]
        pred = [int(jev_route(c) == "transfer") for c in sub]
        mx = [max(c["probs"].values()) for c in sub]
        tp, fp, fn, p, r = prf(pred, y)
        a = auc(mx, y)
        row = [f"| {title} Jev | {len(sub)} | {sum(y)} | {tp} | {fp} | {fn} | {p:.0%} | {r:.0%} | "
               f"{'-' if a is None else f'{a:.3f}'} |"]
        if all(c["llm"] is not None for c in sub):
            llm = [int(c["llm"]) for c in sub]
            tp, fp, fn, p, r = prf(llm, y)
            row.append(f"| {title} LLM(round1) | {len(sub)} | {sum(y)} | {tp} | {fp} | {fn} | {p:.0%} | {r:.0%} | - |")
            uni = [int(a or b) for a, b in zip(llm, pred)]
            tp, fp, fn, p, r = prf(uni, y)
            row.append(f"| {title} LLM ∪ Jev | {len(sub)} | {sum(y)} | {tp} | {fp} | {fn} | {p:.0%} | {r:.0%} | - |")
        return row

    L += ["## 정밀도·재현율", "", "| 묶음 | n | 인계 | TP | FP | FN | 정밀도 | 재현율 | AUC(max p) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for title, sub in (("testset", [c for c in cs if c["set"] == "testset"]),
                       ("health", [c for c in cs if c["set"] == "health"]), ("전체", cs)):
        if sub:
            L += block(title, sub)

    L += ["", "## 기준별 AUC(인계 vs 비인계)와 인계 문항 평균 p", "", "| 기준 | 임계값 | AUC | 인계 평균 | 비인계 평균 | 비인계 중 p≥임계 |",
          "|---|---|---|---|---|---|"]
    y = [c["label"] for c in cs]
    for k in hj.CRITERIA:
        v = [c["probs"][k] for c in cs]
        a = auc(v, y)
        pos = [x for x, t in zip(v, y) if t]
        neg = [x for x, t in zip(v, y) if not t]
        L.append(f"| {k} | {hj.THRESHOLDS[k]} | {'-' if a is None else f'{a:.3f}'} | {sum(pos) / len(pos):.2f} | "
                 f"{sum(neg) / len(neg):.2f} | {sum(x >= hj.THRESHOLDS[k] for x in neg)} |")

    L += ["", "## 임계값 일괄 변경 민감도(모든 기준 같은 값, testset+health)", "", "| 임계값 | TP | FP | FN | 정밀도 | 재현율 |",
          "|---|---|---|---|---|---|"]
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        th = {k: t for k in hj.CRITERIA}
        pred = [int(jev_route(c, th) == "transfer") for c in cs]
        tp, fp, fn, p, r = prf(pred, y)
        L.append(f"| {t} | {tp} | {fp} | {fn} | {p:.0%} | {r:.0%} |")

    cb = [c for c in cs if c["callback"]]
    L += ["", f"영업시간 외 상담원 요청(정답 request_callback) {len(cb)}건 중 Jev 경로 callback: "
          f"{sum(jev_route(c) == 'callback' for c in cb)}건, transfer: {sum(jev_route(c) == 'transfer' for c in cb)}건"]
    L += ["", "## 문항별(인계 정답 또는 Jev 인계 판정)", "", "| id | 분류 | 정답 | LLM | Jev 경로 | 근거 | 주요 p | 텍스트 |",
          "|---|---|---|---|---|---|---|---|"]
    for c in cs:
        ok, why = hj.decide(c["probs"])
        if not (c["label"] or ok):
            continue
        top = ", ".join(f"{k} {v:.2f}" for k, v in sorted(c["probs"].items(), key=lambda kv: -kv[1])[:3])
        llm = "-" if c["llm"] is None else ("O" if c["llm"] else "X")
        L.append(f"| {c['id']} | {c['category']} | {'O' if c['label'] else 'X'} | {llm} | {jev_route(c) or '-'} | "
                 f"{'·'.join(why)} | {top} | {c['text'].replace('|', '/')[:70]} |")
    ms = sorted(c["ms"] for c in cs)
    L += ["", f"응답 시간(평가 실행, 동시 4): p50 {ms[len(ms) // 2]:.0f}ms, p95 {ms[int(0.95 * (len(ms) - 1))]:.0f}ms"]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(REPORT.read_text(encoding="utf-8"))


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=4)
    sub.add_parser("report")
    a = ap.parse_args()
    cmd_run(a.workers) if a.cmd == "run" else cmd_report()


if __name__ == "__main__":
    main()
