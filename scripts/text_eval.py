"""글 전용 평가: 연습 문항을 음성 없이(손님 말 원문 → 우리 응답기 → Mock) 돌리고 공식 채점기(score.py의 judge·tool_match)로
응대·함수를 채점한다. 받아쓰기 오류를 뺀 응대·함수 점수의 상한을 본다. STT·TTS 크레딧 0, LLM·채점은 OPENAI_API_KEY.

  python scripts/text_eval.py --out runs/text1 [--ids P0001-02,P0010-11] [--concurrency 6]

출력: <out>/scored.jsonl(문항별 판정·이유·우리 답·함수), 화면에 응대·함수 점수와 판정 분포. Mock(8100)이 떠 있어야 한다.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OFFICIAL = ROOT.parent / "02_대회자료" / "directon-2026-voice-cs"
MOCK = "http://127.0.0.1:8100"
PTS = {"맞음": 1.0, "괜찮음": 0.7, "틀림": 0.0}


def _score_module():
    spec = importlib.util.spec_from_file_location("official_score", OFFICIAL / "eval" / "score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _SilentTTS:
    async def synthesize(self, text):
        yield bytes(640)


async def run_case(case: dict, tools: list) -> dict:
    from agent.config import Settings
    from agent.core.types import Message, SessionInfo
    from agent.normalize import normalize_args
    from agent.prompts import build_system
    from agent.providers.factory import make_llm
    from agent.responder import Responder

    s = Settings.from_env()
    calls = []

    async def call_tool(name, args):
        async with httpx.AsyncClient(timeout=30) as c:
            res = (await c.post(f"{MOCK}/tools/{name}", json=args, headers={"X-Session-Id": case["case_id"]})).json()
        calls.append({"name": name, "arguments": args,
                      "matched": isinstance(res, dict) and res.get("status") not in ("not_found", "invalid")})
        return res

    info = SessionInfo(session_id=case["case_id"], case_id=case["case_id"], now=case.get("now", ""),
                       history=case.get("history") or [], caller=case.get("customer_profile"), tools_url=MOCK)
    system = build_system(info, s.openers[0], manual_path=s.manual_path)
    r = Responder(make_llm(), _SilentTTS(), call_tool, tools=tools, settings=s, normalize_args=normalize_args)

    async def sink(pcm):
        pass

    try:
        spoken = await r.respond(system, [Message("user", case["customer_utterance"])], sink)
        text = s.openers[0] + " " + " ".join(spoken)
    except Exception as e:  # noqa: BLE001
        text = f"(오류: {e!r})"
    return {"case_id": case["case_id"], "agent_text": text, "tool_calls": calls}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default="")
    ap.add_argument("--concurrency", type=int, default=6)
    a = ap.parse_args()
    load_dotenv(ROOT / ".env")
    sc = _score_module()
    cases = [json.loads(ln) for ln in open(OFFICIAL / "data" / "practice_cases.jsonl", encoding="utf-8")]
    if a.ids:
        want = set(a.ids.split(","))
        cases = [c for c in cases if c["case_id"] in want]
    brand = OFFICIAL / "data" / "brand"
    info = json.load(open(brand / "information.json", encoding="utf-8"))
    ctx = {"scenarios": open(brand / "scenarios.md", encoding="utf-8").read(),
           "info": "\n".join(f"{i['tag']}\n{i['content']}" for i in info), "rubric": {}}
    tools = httpx.get(f"{MOCK}/tools").json()
    sem = asyncio.Semaphore(a.concurrency)

    async def one(hc, case):
        async with sem:
            r = await run_case(case, tools)
            ok, why = sc.tool_match(case["reference"]["tool_calls"], r["tool_calls"], case.get("history", []))
            v = await sc.judge(hc, case, r["agent_text"], r["tool_calls"], ctx)
            return {**r, "tool_ok": ok, "tool_why": why, **v}

    async with httpx.AsyncClient(timeout=180) as hc:
        scored = await asyncio.gather(*(one(hc, c) for c in cases))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "scored.jsonl", "w", encoding="utf-8") as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    n = len(scored)
    resp = 100 * sum(PTS.get(s["verdict"], 0) for s in scored) / n
    func = 100 * sum(bool(s["tool_ok"]) for s in scored) / n
    dist = {k: sum(s["verdict"] == k for s in scored) for k in PTS}
    print(f"문항 {n} | 응대 {resp:.1f} | 함수 {func:.1f} | {dist}")


if __name__ == "__main__":
    asyncio.run(main())
