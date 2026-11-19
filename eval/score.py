"""측정 결과 채점 → runs/X/report.md + runs/X/scored.jsonl + 콘솔 요약.

사용:
  uv run python -m eval.score --run runs/X [--judge] [--cases data/testset/cases.jsonl] [--strict-free-text]

- 속도: Starter Kit score_demo.py와 같은 표. spoke_before_end / error / turn.end 없음 / turn_end>25초 → 0.
- 함수: expected_tools vs 실제 호출. 이름 + 키 인자 정확 일치. 자유 텍스트 인자(FREE_TEXT_KEYS)는
  기본 제외(--strict-free-text로 포함). 같은 호출 중복은 한 번으로 본다.
  점수 = 100 × 일치 수 / (기대 수 + 기대 밖 호출 수). 기대가 []이고 호출도 없으면 100, 호출하면 0.
- 응대(옵션 --judge): LLM(Gemini flash)이 expected_points 충족 / must_not 위반을 판정.
  점수 = 100 × 충족 비율, must_not 하나라도 위반하면 0. 심사 함수는 주입 가능(judge_fn).
- 총점 추정 = 응대 50% + 함수 20% + 속도 30% (공식 배점, docs/00_대회규격_분석.md). 심사가 없으면
  total_no_judge = (함수 20 + 속도 30)/50 로 환산한 값만 낸다.
- 공식 속도(AG-T21, docs/04_목표_점검.md): 함수를 부른 문항은 첫 소리 점수와 "함수 결과 → 답"(after_tool_sound) 점수의
  평균(speed_official). 같은 표를 쓴다. 기존 speed(첫 소리만)·total*은 그대로 두고 *_official을 따로 낸다.
  run 폴더의 logs/*.jsonl(서버 턴 로그, DIRECTON_LOG_DIR)이 있으면 after_tool_answer_ms(채움말을 뺀 본 응답까지)를 붙이고
  그것으로 계산한 speed_official_answer도 낸다(채움말을 답으로 치지 않는 보수적 해석).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Callable

TABLE = [(1.5, 100), (2.0, 90), (2.5, 80), (3.0, 70), (3.5, 60), (4.0, 50), (5.0, 30), (6.0, 10)]
FREE_TEXT_KEYS = {"reason", "new_address", "preferred_time"}
GROUP_KEYS = ("category", "tags", "channel", "noise")
JUDGE_MODEL = "gemini-3.8-flash"


# ───────────────────────── 속도 ─────────────────────────
def speed_points(t) -> int:
    if t is None or not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
        return 0
    return next((v for bound, v in TABLE if t <= bound), 0)


def _speed_bad(row: dict) -> bool:
    return bool(row.get("spoke_before_end") or "error" in row or row.get("turn_end") is None
                or row.get("turn_end", 999) > 25)


def speed_score(row: dict) -> int:
    return 0 if _speed_bad(row) else speed_points(row.get("first_sound"))


def called_tool(row: dict) -> bool:
    """함수 응답 시각이 기록된 호출이 있는 문항(공식 규칙의 '함수 문항')."""
    return any(c.get("t_response") is not None for c in row.get("tool_calls") or [])


def speed_after_tool(row: dict) -> int | None:
    """함수 결과 → 그 뒤 첫 유성음(after_tool_sound) 점수. 함수 문항이 아니면 None, 답 소리가 없으면 0."""
    if not called_tool(row):
        return None
    return 0 if _speed_bad(row) else speed_points(row.get("after_tool_sound"))


def speed_official(row: dict) -> float:
    """공식 속도: 함수 문항은 (첫 소리 점수 + 함수 결과 뒤 답 점수) / 2, 아니면 첫 소리 점수."""
    if _speed_bad(row):
        return 0
    first, after = speed_score(row), speed_after_tool(row)
    return first if after is None else (first + after) / 2


def speed_official_answer(row: dict) -> float | None:
    """speed_official과 같되 '답'을 서버 턴 로그의 after_tool_answer_ms(채움말 제외)로 잰다. 기록이 없으면 None."""
    if not called_tool(row) or "after_tool_answer_ms" not in row:
        return None
    if _speed_bad(row):
        return 0
    ms = row.get("after_tool_answer_ms")
    return (speed_score(row) + speed_points(None if ms is None else ms / 1000)) / 2


# ───────────────────────── 함수 일치 ─────────────────────────
def _key(call: dict, exclude_free_text: bool) -> tuple[str, str]:
    args = call.get("arguments") or {}
    if exclude_free_text:
        args = {k: v for k, v in args.items() if k not in FREE_TEXT_KEYS}
    return call.get("name"), json.dumps(args, ensure_ascii=False, sort_keys=True)


def tool_match(expected: list[dict], actual: list[dict], exclude_free_text: bool = True) -> dict:
    actual = actual or []
    distinct = list(dict.fromkeys(_key(c, exclude_free_text) for c in actual))
    unused = list(distinct)
    matched, missing = 0, []
    for e in expected:
        k = _key(e, exclude_free_text)
        if k in unused:
            unused.remove(k)
            matched += 1
        else:
            missing.append(e)
    extra = [{"name": n, "arguments": json.loads(a)} for n, a in unused]
    denom = len(expected) + len(extra)
    score = 100.0 if denom == 0 else 100.0 * matched / denom
    names_exp = sorted(e.get("name") for e in expected)
    names_act = sorted(dict.fromkeys(c.get("name") for c in actual))
    return {"score": score, "exact": matched == len(expected) and not extra, "matched": matched,
            "missing": missing, "extra": extra, "name_match": names_exp == names_act}


# ───────────────────────── 응대 심사 ─────────────────────────
def judge_points(verdict: dict, n_points: int) -> float:
    if "score" in verdict and verdict["score"] is not None:
        return float(max(0, min(100, verdict["score"])))
    if any(verdict.get("violations") or []):
        return 0.0
    pts = list(verdict.get("points") or [])[:n_points] if n_points else []
    if not n_points:
        return 100.0
    return 100.0 * sum(bool(p) for p in pts) / n_points


JUDGE_PROMPT = """당신은 한국어 쇼핑몰 고객센터 음성 상담 평가자입니다. 상담원 응답이 기준을 지키는지 판정하세요.

[대화 이력]
{history}

[고객 발화]
{utterance}

[상담원 응답 텍스트]
{agent_text}

[상담원이 호출한 함수]
{tools}

[충족해야 할 항목] (순서대로 true/false)
{points}

[하면 안 되는 행동] (순서대로, 위반했으면 true)
{must_not}

JSON만 출력: {{"points": [bool,...], "violations": [bool,...], "comment": "한 문장 근거"}}"""


def make_gemini_judge(model: str = JUDGE_MODEL, client=None, retries: int = 3,
                      sleep: Callable[[float], None] = time.sleep) -> Callable[[dict], dict]:
    from eval.synth import is_retryable

    def judge(row: dict) -> dict:
        nonlocal client
        if client is None:
            from google import genai
            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"), http_options={"api_version": "v1beta", "timeout": 90_000})
        from google.genai import types
        num = lambda xs: "\n".join(f"{i + 1}. {x}" for i, x in enumerate(xs)) or "(없음)"  # noqa: E731
        prompt = JUDGE_PROMPT.format(
            history=json.dumps(row.get("history") or [], ensure_ascii=False),
            utterance=row.get("customer_utterance", ""), agent_text=row.get("agent_text") or "(응답 없음)",
            tools=json.dumps([{"name": c.get("name"), "arguments": c.get("arguments")}
                              for c in row.get("tool_calls") or []], ensure_ascii=False),
            points=num(row.get("expected_points") or []), must_not=num(row.get("must_not") or []))
        config = types.GenerateContentConfig(response_mime_type="application/json", temperature=0)
        for attempt in range(retries + 1):              # 503(과부하)·짧은 429는 재시도
            try:
                resp = client.models.generate_content(model=model, contents=prompt, config=config)
                return json.loads(resp.text)
            except Exception as e:
                if attempt >= retries or not is_retryable(e):
                    raise
                sleep(min(30.0, 2.0 * 2 ** attempt))
    return judge


# ───────────────────────── 채점·집계 ─────────────────────────
def score_rows(rows: list[dict], judge_fn: Callable[[dict], dict] | None = None,
               exclude_free_text: bool = True) -> list[dict]:
    out = []
    for r in rows:
        sc: dict = {"speed": speed_score(r), "tool": None, "tool_exact": None, "judge": None,
                    "speed_after_tool": speed_after_tool(r), "speed_official": speed_official(r)}
        soa = speed_official_answer(r)
        if soa is not None:
            sc["speed_official_answer"] = soa
        if "expected_tools" in r and r["expected_tools"] is not None:
            m = tool_match(r["expected_tools"], r.get("tool_calls") or [], exclude_free_text)
            sc.update(tool=m["score"], tool_exact=m["exact"], tool_missing=m["missing"], tool_extra=m["extra"])
        if judge_fn is not None:
            try:
                v = judge_fn(r)
                sc["judge"] = judge_points(v, len(r.get("expected_points") or []))
                if v.get("comment"):
                    sc["judge_comment"] = v["comment"]
            except Exception as e:
                sc["judge_error"] = repr(e)[:300]
        if sc["tool"] is not None:
            sc["total_no_judge"] = (0.2 * sc["tool"] + 0.3 * sc["speed"]) / 0.5
            sc["total_no_judge_official"] = (0.2 * sc["tool"] + 0.3 * sc["speed_official"]) / 0.5
            if sc["judge"] is not None:
                sc["total"] = 0.5 * sc["judge"] + 0.2 * sc["tool"] + 0.3 * sc["speed"]
                sc["total_official"] = 0.5 * sc["judge"] + 0.2 * sc["tool"] + 0.3 * sc["speed_official"]
        out.append({**r, "scores": sc})
    return out


def _group_values(row: dict, key: str) -> list[str]:
    env = row.get("env") or {}
    if key == "tags":
        return list(row.get("tags") or []) or ["(없음)"]
    if key == "channel":
        return [env.get("channel", "wideband")]
    if key == "noise":
        return [env.get("noise", "none")]
    return [str(row.get(key) or "unknown")]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    i = min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))
    return xs[i]


def _stats(rows: list[dict]) -> dict:
    s = [r["scores"] for r in rows]
    fs = [r.get("first_sound") for r in rows if not r.get("spoke_before_end") and "error" not in r]
    tool_rows = [r for r in rows if called_tool(r) and "error" not in r]
    at = [r.get("after_tool_sound") for r in tool_rows]
    ans = [None if r.get("after_tool_answer_ms") is None else r["after_tool_answer_ms"] / 1000
           for r in tool_rows if "after_tool_answer_ms" in r]
    return {"n": len(rows),
            "speed_official": _mean([x.get("speed_official") for x in s]),
            "speed_official_answer": _mean([x.get("speed_official_answer") for x in s]),
            "after_tool_n": len(tool_rows),
            "after_tool_none": sum(1 for x in at if x is None),
            "after_tool_p50": _pct(at, 0.5), "after_tool_p95": _pct(at, 0.95),
            "after_tool_answer_p50": _pct(ans, 0.5), "after_tool_answer_p95": _pct(ans, 0.95),
            "speed_after_tool": _mean([x.get("speed_after_tool") for x in s]),
            "total_no_judge_official": _mean([x.get("total_no_judge_official") for x in s]),
            "total_official": _mean([x.get("total_official") for x in s]),
            "speed": _mean([x["speed"] for x in s]),
            "speed_zero": sum(1 for x in s if x["speed"] == 0),
            "spoke_before_end": sum(1 for r in rows if r.get("spoke_before_end")),
            "errors": sum(1 for r in rows if "error" in r),
            "first_sound_p50": _pct(fs, 0.5), "first_sound_p95": _pct(fs, 0.95),
            "tool": _mean([x["tool"] for x in s]),
            "tool_exact": _mean([None if x["tool_exact"] is None else float(x["tool_exact"]) for x in s]),
            "judge": _mean([x["judge"] for x in s]),
            "total": _mean([x.get("total") for x in s]),
            "total_no_judge": _mean([x.get("total_no_judge") for x in s])}


def aggregate(scored: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list] = {}
    for r in scored:
        for v in _group_values(r, key):
            groups.setdefault(v, []).append(r)
    return {k: _stats(v) for k, v in sorted(groups.items())}


def summarize(scored: list[dict]) -> dict:
    return _stats(scored)


# ───────────────────────── 보고서 ─────────────────────────
def _f(v, nd=1):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


COLS = [("n", "n", 0), ("speed", "속도", 1), ("speed_zero", "속도0", 0), ("first_sound_p50", "첫소리p50", 2),
        ("first_sound_p95", "첫소리p95", 2), ("tool", "함수", 1), ("tool_exact", "함수정확률", 2),
        ("judge", "응대", 1), ("total_no_judge", "추정(응대제외)", 1), ("total", "추정총점", 1),
        ("speed_official", "속도(공식)", 1), ("after_tool_p50", "조회뒤p50", 2), ("after_tool_p95", "조회뒤p95", 2),
        ("after_tool_answer_p50", "조회뒤답p50", 2), ("total_official", "추정총점(공식)", 1)]


def _table(stats: dict[str, dict], label: str) -> list[str]:
    lines = [f"| {label} | " + " | ".join(c[1] for c in COLS) + " |",
             "|---|" + "---|" * len(COLS)]
    for k, st in stats.items():
        lines.append(f"| {k} | " + " | ".join(_f(st[c[0]], c[2]) for c in COLS) + " |")
    return lines


def _failure_reason(r: dict) -> list[str]:
    sc, why = r["scores"], []
    if "error" in r:
        why.append(f"에러: {r['error'][:80]}")
    if r.get("spoke_before_end"):
        why.append("말 끝 전 발화")
    elif r.get("turn_end") is None and "error" not in r:
        why.append("turn.end 없음")
    elif sc["speed"] == 0 and "error" not in r:
        why.append(f"속도 0 (first_sound={_f(r.get('first_sound'), 2)})")
    if sc["tool_exact"] is False:
        miss = ", ".join(f"{m['name']}{json.dumps(m.get('arguments') or {}, ensure_ascii=False)}"
                         for m in sc.get("tool_missing", []))
        extra = ", ".join(f"{m['name']}{json.dumps(m.get('arguments') or {}, ensure_ascii=False)}"
                          for m in sc.get("tool_extra", []))
        why.append(f"함수 불일치(누락: {miss or '-'} / 여분: {extra or '-'})")
    if sc["judge"] is not None and sc["judge"] < 50:
        why.append(f"응대 {sc['judge']:.0f}" + (f" — {sc['judge_comment']}" if sc.get("judge_comment") else ""))
    if sc.get("judge_error"):
        why.append("심사 오류")
    return why


def render_report(scored: list[dict], title: str = "") -> str:
    s = summarize(scored)
    L = [f"# 측정 보고서 {title}".rstrip(), "",
         "배점(공식): 응대 50% / 함수 20% / 속도 30%. 속도 표는 Starter Kit score_demo.py와 같다.", "",
         "## 전체 요약", ""]
    L += _table({"전체": s}, "구분")
    L += ["", f"- 말 끝 전 발화: {s['spoke_before_end']}건, 에러: {s['errors']}건, 속도 0점: {s['speed_zero']}건"]
    L += [f"- 조회 뒤(함수 결과 → 첫 유성음): 함수 문항 {s['after_tool_n']}건, p50 {_f(s['after_tool_p50'], 2)}s "
          f"p95 {_f(s['after_tool_p95'], 2)}s, 소리 없음 {s['after_tool_none']}건, 점수 {_f(s['speed_after_tool'])} → "
          f"공식 속도 {_f(s['speed_official'])} (첫 소리만 {_f(s['speed'])})",
          f"- 조회 뒤 본 응답(채움말 제외, 서버 턴 로그): p50 {_f(s['after_tool_answer_p50'], 2)}s "
          f"p95 {_f(s['after_tool_answer_p95'], 2)}s → 공식 속도(보수적) {_f(s['speed_official_answer'])}"]
    for key, label in (("category", "범주(category)"), ("tags", "태그(tags)"),
                       ("channel", "채널(channel)"), ("noise", "잡음(noise)")):
        L += ["", f"## {label}별", ""] + _table(aggregate(scored, key), key)
    fails = [(r["case_id"], _failure_reason(r)) for r in scored]
    fails = [(c, w) for c, w in fails if w]
    L += ["", f"## 실패·감점 문항 ({len(fails)}건)", ""]
    if fails:
        L += ["| case_id | 사유 |", "|---|---|"] + [f"| {c} | {'; '.join(w)} |" for c, w in fails]
    else:
        L.append("없음")
    return "\n".join(L) + "\n"


def load_rows(run_dir: Path) -> list[dict]:
    with open(run_dir / "results.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


TURN_LOG_FIELDS = (("after_tool_answer_ms", "after_tool_answer_ms"), ("after_tool_ms", "server_after_tool_ms"))


def load_turn_logs(paths) -> dict[str, dict]:
    """서버 턴 로그(jsonl 파일 또는 폴더) → {session_id: 마지막 턴 기록}."""
    out: dict[str, dict] = {}
    for p in paths:
        p = Path(p)
        files = sorted(p.glob("*.jsonl")) if p.is_dir() else ([p] if p.exists() else [])
        for f in files:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(rec, dict) and rec.get("session_id"):
                        out[rec["session_id"]] = rec
    return out


def merge_turn_logs(rows: list[dict], logs: dict[str, dict]) -> None:
    for r in rows:
        rec = logs.get(r.get("session_id") or "")
        if not rec:
            continue
        for src, dst in TURN_LOG_FIELDS:
            if src in rec and dst not in r:
                r[dst] = rec[src]


def score_run(run_dir, judge_fn=None, cases_path=None, exclude_free_text: bool = True, turn_logs=None):
    run_dir = Path(run_dir)
    rows = load_rows(run_dir)
    merge_turn_logs(rows, load_turn_logs([run_dir / "logs"] + [Path(x) for x in (turn_logs or [])]))
    if cases_path and Path(cases_path).exists():
        with open(cases_path, encoding="utf-8") as f:
            cases = {c["case_id"]: c for c in (json.loads(l) for l in f if l.strip())}
        from eval.runner import attach_case
        for r in rows:
            if r["case_id"] in cases:
                attach_case(r, cases[r["case_id"]])
    scored = score_rows(rows, judge_fn=judge_fn, exclude_free_text=exclude_free_text)
    report = render_report(scored, f"— {run_dir.name}")
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    with open(run_dir / "scored.jsonl", "w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return scored, report


def main(argv=None, judge_fn=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--judge", action="store_true", help="LLM 응대 심사(Gemini)")
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--cases", default=None, help="결과 행에 기대값이 없을 때 병합할 cases.jsonl")
    ap.add_argument("--strict-free-text", action="store_true", help="자유 텍스트 인자도 정확 일치로 비교")
    ap.add_argument("--turn-logs", action="append", default=[],
                    help="서버 턴 로그(jsonl 또는 폴더). run/logs는 자동으로 읽는다")
    a = ap.parse_args(argv)
    if a.judge and judge_fn is None:
        judge_fn = make_gemini_judge(a.judge_model)
    scored, _ = score_run(a.run, judge_fn=judge_fn if a.judge or judge_fn else None, cases_path=a.cases,
                          exclude_free_text=not a.strict_free_text, turn_logs=a.turn_logs)
    s = summarize(scored)
    print(f"문항 {s['n']}개 | 속도 평균 {_f(s['speed'])} (0점 {s['speed_zero']}건, 말 끝 전 발화 "
          f"{s['spoke_before_end']}건) | 첫 소리 p50 {_f(s['first_sound_p50'], 2)}s p95 {_f(s['first_sound_p95'], 2)}s")
    print(f"함수 평균 {_f(s['tool'])} (정확률 {_f(s['tool_exact'], 2)}) | 응대 {_f(s['judge'])} | "
          f"추정총점 {_f(s['total'])} | 응대 제외 추정 {_f(s['total_no_judge'])}")
    print(f"조회 뒤 p50 {_f(s['after_tool_p50'], 2)}s p95 {_f(s['after_tool_p95'], 2)}s (함수 문항 {s['after_tool_n']}개) | "
          f"공식 속도 {_f(s['speed_official'])} | 본 응답 기준 {_f(s['speed_official_answer'])} "
          f"(답 p50 {_f(s['after_tool_answer_p50'], 2)}s) | 추정총점(공식) {_f(s['total_official'])}")
    print(f"보고서: {Path(a.run) / 'report.md'}")


if __name__ == "__main__":
    main()
