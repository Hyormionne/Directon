"""함수 호출 오프라인 평가 — 음성·LLM·API 키 없이 몇 초 안에 돈다.

연습 문항의 손님 말(글)을 그대로, 또는 '받아쓰기처럼 한글 숫자로 바꾼' 형태로 넣고
pipeline.run_turn(규칙 기반)이 실제로 보낸 호출을 채점기(eval/score.py)의 tool_match 로 채점한다.
모의 함수 서버(mock_tools/mock_server.py)의 Store 를 같은 프로세스에서 불러 쓰므로 서버를 띄울 필요가 없다.

  python3 -m fncall.eval_fncall                 # 글 그대로 + 한글 숫자 변형 2종
  python3 -m fncall.eval_fncall --show-fail     # 틀린 문항 출력
  python3 -m fncall.eval_fncall --stt-noise 0.3 # 숫자 한 자리 오인식을 30% 문항에 섞어 후보 검증 효과 측정
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import random
import re
from collections import Counter

from .numbers_ko import to_spoken_digits
from .pipeline import run_turn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


score = _load("score", os.path.join(ROOT, "eval", "score.py"))
mock_server = _load("mock_server", os.path.join(ROOT, "mock_tools", "mock_server.py"))

SINO = {"0": "", "1": "일", "2": "이", "3": "삼", "4": "사", "5": "오", "6": "육", "7": "칠", "8": "팔", "9": "구"}


def _sino(n: int) -> str:
    """2026 → 이천이십육 (받아쓰기가 자릿수로 읽는 경우 흉내)"""
    out = ""
    for v, u in [(1000, "천"), (100, "백"), (10, "십")]:
        q, n = divmod(n, v)
        if q:
            out += ("" if q == 1 else SINO[str(q)]) + u
    return out + (SINO[str(n)] if n else "")


def spoken(text: str, rng: random.Random, noise: float = 0.0) -> str:
    """'010 1234 5678' / '20260101-0001234' 같은 숫자를 받아쓰기가 한글로 줄 법한 형태로 바꾼다.
    noise 확률로 '같은 숫자 두 번'을 한 번으로 줄이는 오인식을 넣는다(실제 STT 에서 자주 나는 오류)."""
    def rep(m):
        raw = m.group(0)
        d = re.sub(r"\D", "", raw)
        if noise and rng.random() < noise:
            i = next((k for k in range(4, len(d)) if d[k] == d[k - 1]), None)
            if i is not None:
                d = d[:i] + d[i + 1:]
                raw = d
        zero = rng.choice(["공", "영"])
        read = lambda s: "".join({"0": zero}.get(c, SINO[c]) for c in s)
        if re.fullmatch(r"20\d{6}\d{7}", d) and rng.random() < 0.5:
            return f"{_sino(int(d[:4]))} {read(d[4:8])} 다시 {read(d[8:])}"
        if re.search(r"[\s\-.,]", raw.strip()):
            return " ".join(read(re.sub(r"\D", "", g)) for g in re.split(r"[\s\-.,]+", raw) if g)
        return to_spoken_digits(d).replace("공", zero).replace(",", rng.choice([",", "", " "]))
    return re.sub(r"\d[\d\s\-.,]{5,}\d", rep, text)


async def run_all(cases, variant, call):
    rows = []
    for case in cases:
        utt = variant(case)
        session = {"history": case["history"], "now": case.get("now"), "caller": case.get("customer_profile")}
        rows.append((case, utt, await run_turn(session, utt, call)))
    return rows


def report(vname, rows, show_fail):
    ok, fails, by, stat = 0, [], Counter(), Counter()
    for case, utt, out in rows:
        got = [{"name": c.name, "arguments": c.arguments} for c, _ in out.calls]
        hit, why = score.tool_match(case["reference"]["tool_calls"], got, case["history"])
        ok += hit
        ref = tuple(sorted({c["name"] for c in case["reference"]["tool_calls"]} - score.INFO_FUNCS)) or ("(없음)",)
        by[(ref, hit)] += 1
        if out.calls:
            stat["함수 부른 문항"] += 1
            stat["템플릿 답" if out.speech else "LLM 답 필요"] += 1
            stat[f"순차 왕복 {out.rounds}회"] += 1
        if not hit:
            fails.append((case["case_id"], why, utt[:60].replace("\n", " "),
                          [(c.name, {k: v for k, v in c.arguments.items() if k not in ("reason", "customer_name")}, r.get("status"))
                           for c, r in out.calls], out.commit_reason))
    print(f"\n[{vname}] 함수 점수 {ok / len(rows) * 100:.1f}  ({ok}/{len(rows)})")
    for (ref, hit), n in sorted(by.items(), key=lambda x: (x[0][0], not x[0][1])):
        print(f"   {'O' if hit else 'X'} {', '.join(ref):28s} {n}")
    print("  ", dict(sorted(stat.items())))
    if show_fail:
        for f in fails:
            print("   -", *f)
    return ok / len(rows) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=os.path.join(ROOT, "data", "practice_cases.jsonl"))
    ap.add_argument("--db", default=os.path.join(ROOT, "data", "mock_db.json"))
    ap.add_argument("--show-fail", action="store_true")
    ap.add_argument("--stt-noise", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cases = [json.loads(l) for l in open(a.cases, encoding="utf-8")]
    store = mock_server.Store(a.db, os.path.join(ROOT, "data", "brand", "tools.json"))

    async def call(name, args):           # HTTP 없이 같은 프로세스에서 모의 함수 호출
        return store.lookup(name, args)[0]

    variants = {"text": lambda c: c["customer_utterance"]}
    for seed in (a.seed, a.seed + 1):
        rng = random.Random(seed)
        variants[f"spoken#{seed}"] = (lambda r: lambda c: spoken(c["customer_utterance"], r))(rng)
    if a.stt_noise:
        rng = random.Random(a.seed + 99)
        variants[f"spoken+noise{a.stt_noise}"] = lambda c: spoken(c["customer_utterance"], rng, a.stt_noise)
    for vname, vf in variants.items():
        report(vname, asyncio.run(run_all(cases, vf, call)), a.show_fail)


if __name__ == "__main__":
    main()
