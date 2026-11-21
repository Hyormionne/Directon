"""STT+LLM 평가 채점: 무음 TTS 서버(DIRECTON_SILENT_TTS=1)를 공식 측정기로 돌린 결과를 상담원 글(agent.transcript)로 채점한다.
답 음성 받아쓰기(gpt-4o-transcribe) 대신 agent_text를 공식 judge에 넘긴다. 함수 판정은 공식 tool_match 그대로. 속도는 보지 않는다.

  python scripts/stt_score.py --run runs/<측정 폴더>
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx
from dotenv import load_dotenv

from text_eval import OFFICIAL, PTS, ROOT, _score_module


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--concurrency", type=int, default=6)
    a = ap.parse_args()
    load_dotenv(ROOT / ".env")
    sc = _score_module()
    cases = {json.loads(ln)["case_id"]: json.loads(ln)
             for ln in open(OFFICIAL / "data" / "practice_cases.jsonl", encoding="utf-8")}
    results = [json.loads(ln) for ln in open(Path(a.run) / "results.jsonl", encoding="utf-8")]
    brand = OFFICIAL / "data" / "brand"
    info = json.load(open(brand / "information.json", encoding="utf-8"))
    ctx = {"scenarios": open(brand / "scenarios.md", encoding="utf-8").read(),
           "info": "\n".join(f"{i['tag']}\n{i['content']}" for i in info), "rubric": {}}
    sem = asyncio.Semaphore(a.concurrency)

    async def one(hc, r):
        async with sem:
            case = cases[r["case_id"]]
            calls = r.get("tool_calls", [])
            ok, why = sc.tool_match(case["reference"]["tool_calls"], calls, case.get("history", []))
            v = await sc.judge(hc, case, r.get("agent_text", ""), calls, ctx)
            return {"case_id": r["case_id"], "agent_text": r.get("agent_text", ""), "tool_calls": calls,
                    "first_sound": r.get("first_sound"), "spoke_before_end": r.get("spoke_before_end"),
                    "tool_ok": ok, "tool_why": why, **v}

    async with httpx.AsyncClient(timeout=180) as hc:
        scored = await asyncio.gather(*(one(hc, r) for r in results))
    with open(Path(a.run) / "scored_text.jsonl", "w", encoding="utf-8") as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    n = len(scored)
    resp = 100 * sum(PTS.get(s["verdict"], 0) for s in scored) / n
    func = 100 * sum(bool(s["tool_ok"]) for s in scored) / n
    dist = {k: sum(s["verdict"] == k for s in scored) for k in PTS}
    print(f"문항 {n} | 응대 {resp:.1f} | 함수 {func:.1f} | {dist}")


if __name__ == "__main__":
    asyncio.run(main())
