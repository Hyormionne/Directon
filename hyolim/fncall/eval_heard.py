"""실제 음성 측정 때 '받아쓴 글'을 다시 넣어 함수 점수를 잰다 (음성 재생·API 없이 몇 초).

음성 인식 오류에 대한 번호 보정·재조회가 효과가 있는지, 실제 음성을 다시 틀지 않고 확인한다.
받아쓴 글은 eval/heard_openai.json ({"run1": {case_id: 받아쓴 글}, ...}) 에서 읽는다.

  python3 -m fncall.eval_heard --show-fail
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os

from .eval_fncall import mock_server, score
from .pipeline import run_turn

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heard", default=os.path.join(ROOT, "eval", "heard_openai.json"))
    ap.add_argument("--show-fail", action="store_true")
    a = ap.parse_args()
    cases = {json.loads(l)["case_id"]: json.loads(l) for l in open(os.path.join(ROOT, "data", "practice_cases.jsonl"), encoding="utf-8")}
    heard = json.load(open(a.heard, encoding="utf-8"))
    store = mock_server.Store(os.path.join(ROOT, "data", "mock_db.json"), os.path.join(ROOT, "data", "brand", "tools.json"))

    log: list = []

    async def call(n, args):                 # 측정기 중계처럼 '보낸 모든 호출'을 기록한다 (채점은 이 기록으로)
        log.append({"name": n, "arguments": args})
        return store.lookup(n, args)[0]

    for run, texts in heard.items():
        ok, fails, rounds = 0, [], 0
        texts = {k: v for k, v in texts.items() if v}
        if not texts:
            print(f"[{run}] 받아쓴 글이 없습니다"); continue
        for cid, utt in texts.items():
            c = cases[cid]
            s = {"history": c["history"], "now": c.get("now"), "caller": c.get("customer_profile")}
            log.clear()
            out = await run_turn(s, utt, call)
            rounds = max(rounds, out.rounds)
            got = list(log)
            hit, why = score.tool_match(c["reference"]["tool_calls"], got, c["history"])
            ok += hit
            if not hit:
                fails.append((cid, why, utt[:50], [(x.name, x.arguments.get("buyer_cellphone") or x.arguments.get("customer_phone_number")
                                                    or x.arguments.get("order_id") or x.arguments.get("phone_number"), r.get("status"))
                                                   for x, r in out.calls][:4]))
        print(f"[{run}] 함수 점수 {ok / len(texts) * 100:.1f} ({ok}/{len(texts)})  최대 순차 왕복 {rounds}회")
        if a.show_fail:
            for f in fails:
                print("   -", *f)


if __name__ == "__main__":
    asyncio.run(main())
