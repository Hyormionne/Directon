"""글 기반 응대 평가 — OpenAI 키만으로 응대 점수를 미리 잰다 (Soniox·Gemini 불필요).

손님 말(원문 글) → fncall(조회·접수, 모의 함수 데이터) → 실제 GPT 답 → 채점기(eval/score.py)와 같은 심사 AI 로 판정.
음성 단계(STT 오인식, 말 끝 판단, TTS)는 빠지므로 '응대 품질의 상한'에 가깝다.

  python3 -m server.eval_text --limit 20            # 고정 순서 20문항 (크레딧 절약)
  python3 -m server.eval_text --limit 0             # 전체
  python3 -m server.eval_text --offset 20 --limit 0 # 튜닝에 쓴 앞 20문항을 뺀 나머지 (과적합 확인)
  python3 -m server.eval_text --only P0052-02,P0148-08

결과: eval/runs/text_<시각>/scored.jsonl (문항별 답·판정·이유), summary.json
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import statistics
import time

import httpx

from . import config as C           # .env 를 읽어 OPENAI_API_KEY 를 환경 변수로 올린다
from . import llm
from .answer import guarded, plan_answer
from .app import FILLER_TEXT, pick_filler
from fncall.pipeline import run_turn


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


score = _load("score", os.path.join(C.ROOT, "eval", "score.py"))
mock_server = _load("mock_server", os.path.join(C.ROOT, "mock_tools", "mock_server.py"))


async def answer_one(http, store, case):
    session = {"history": case["history"], "now": case.get("now"), "caller": case.get("customer_profile"),
               "case_id": case["case_id"]}
    utt = case["customer_utterance"]

    async def call(name, args):
        return store.lookup(name, args)[0]

    t0 = time.perf_counter()
    out = await run_turn(session, utt, call)
    fk = pick_filler(session, utt)
    said = [FILLER_TEXT[fk]] if fk else []
    ttft = None
    plan = plan_answer(session, utt, out)
    how = plan.mode
    if plan.mode in ("safe", "template"):
        said.append(plan.text)
    else:
        t1 = time.perf_counter()
        async for s in guarded(plan, llm.stream_answer(http, plan.messages), ack_played=bool(fk)):
            if ttft is None:
                ttft = time.perf_counter() - t1
            said.append(s)
    calls = [{"name": c.name, "arguments": c.arguments, "matched": r.get("status") == "ok"} for c, r in out.calls]
    return {"case_id": case["case_id"], "utterance": utt, "said": " ".join(said), "how": how,
            "llm_first_sentence_s": ttft, "elapsed_s": time.perf_counter() - t0, "tool_calls": calls,
            "commit_reason": out.commit_reason, "risk": plan.risk.level, "risk_reasons": plan.risk.reasons,
            "guarded": plan.guarded}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=os.path.join(C.ROOT, "data", "practice_cases.jsonl"))
    ap.add_argument("--limit", type=int, default=20, help="0 = 전체")
    ap.add_argument("--offset", type=int, default=0, help="고정 순서에서 앞의 N문항을 건너뛴다 (예: 튜닝에 쓴 20개 제외 = --offset 20)")
    ap.add_argument("--only", default="", help="쉼표로 구분한 case_id")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--all-cases", action="store_true", help="음성 유무로 거르지 않는다 (직접 만든 문항 파일용)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY 가 없습니다. 맨 위 폴더 .env 에 OPENAI_API_KEY=... 를 넣어 주세요.")

    cases = [json.loads(l) for l in open(a.cases, encoding="utf-8")]
    meta = json.load(open(os.path.join(C.ROOT, "data", "audio", "practice", "meta.json"), encoding="utf-8"))
    if not a.all_cases:
        cases = [c for c in cases if c["case_id"] in meta]                # 음성이 있는 문항 = 실제 측정 대상
    if a.only:
        want = set(a.only.split(","))
        cases = [c for c in cases if c["case_id"] in want]
    else:
        cases = sorted(cases, key=lambda c: hashlib.md5(c["case_id"].encode()).hexdigest())   # run_eval --sample 과 같은 순서
        cases = cases[a.offset:][: a.limit or None]

    out_dir = a.out or os.path.join(C.ROOT, "eval", "runs", time.strftime("text_%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    store = mock_server.Store(os.path.join(C.ROOT, "data", "mock_db.json"), os.path.join(C.ROOT, "data", "brand", "tools.json"))
    brand = os.path.join(C.ROOT, "data", "brand")
    info = json.load(open(os.path.join(brand, "information.json"), encoding="utf-8"))
    ctx = {"scenarios": open(os.path.join(brand, "scenarios.md"), encoding="utf-8").read(),
           "info": "\n".join(f"{i['tag']}\n{i['content']}" for i in info), "rubric": {}}
    sem = asyncio.Semaphore(a.concurrency)

    async with httpx.AsyncClient(timeout=120) as http:
        async def one(case):
            async with sem:
                try:
                    r = await answer_one(http, store, case)
                except Exception as e:
                    r = {"case_id": case["case_id"], "said": "", "error": repr(e)[:300], "tool_calls": []}
                v = await score.judge(http, case, r["said"], r["tool_calls"], ctx)
                ok, why = score.tool_match(case["reference"]["tool_calls"], r["tool_calls"], case["history"])
                r.update(v, tool_ok=ok, tool_why=why, reference=case["reference"].get("answer"))
                mark = {"맞음": "O", "괜찮음": "△", "틀림": "X"}.get(r["verdict"], "?")
                print(f"{mark} {case['case_id']}  {r.get('how', 'ERR'):8s} {r['said'][:70]!r}", flush=True)
                if r.get("error"):
                    print("   error:", r["error"], flush=True)
                return r
        rows = await asyncio.gather(*[one(c) for c in cases])

    with open(os.path.join(out_dir, "scored.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    resp = statistics.mean(score.PTS[r["verdict"]] for r in rows) * 100
    tool = statistics.mean(1.0 if r["tool_ok"] else 0.0 for r in rows) * 100
    tt = sorted(r["llm_first_sentence_s"] for r in rows if r.get("llm_first_sentence_s") is not None)
    summ = {"cases": len(rows), "response_score": round(resp, 1), "tool_score": round(tool, 1),
            "verdicts": {k: sum(r["verdict"] == k for r in rows) for k in score.PTS},
            "template_answers": sum(r.get("how") == "template" for r in rows),
            "llm_first_sentence_p50": tt[len(tt) // 2] if tt else None, "llm_first_sentence_max": tt[-1] if tt else None}
    json.dump(summ, open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n문항 {summ['cases']}개  응대 {summ['response_score']}  함수 {summ['tool_score']}  {summ['verdicts']}")
    print(f"템플릿 답 {summ['template_answers']}개 / LLM 첫 문장 p50 {summ['llm_first_sentence_p50']}s max {summ['llm_first_sentence_max']}s")
    print("틀림 문항:")
    for r in rows:
        if r["verdict"] == "틀림":
            print(f"  - {r['case_id']} 손님: {r.get('utterance', '')[:50]!r}\n      AI: {r['said'][:90]!r}\n      이유: {r['reason']}")
    print(f"\n저장: {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
