"""연습 문항을 음성 없이 글로만 돌려 본다(STT·TTS 크레딧 0, LLM만 호출). Mock(8100)이 떠 있어야 한다.

  python scripts/text_check.py P0144-01 [P0001-01 ...]

출력: 손님 말, 우리 함수 호출(인자·결과 status), 우리 답, 실제 상담 답(참고).
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASES = ROOT.parent / "02_대회자료" / "directon-2026-voice-cs" / "data" / "practice_cases.jsonl"
MOCK = "http://127.0.0.1:8100"


class _SilentTTS:
    async def synthesize(self, text):
        yield bytes(640)      # 무음 20ms(응답기는 소리가 있어야 문장을 말한 것으로 센다)


async def run(case: dict) -> None:
    from agent.config import Settings
    from agent.core.types import Message, SessionInfo
    from agent.normalize import normalize_args
    from agent.prompts import build_system
    from agent.providers.factory import make_llm
    from agent.responder import Responder

    s = Settings.from_env()
    tools = httpx.get(f"{MOCK}/tools").json()
    calls = []

    async def call_tool(name, args):
        async with httpx.AsyncClient() as c:
            res = (await c.post(f"{MOCK}/tools/{name}", json=args, headers={"X-Session-Id": "text"})).json()
        calls.append((name, args, res.get("status")))
        return res

    info = SessionInfo(session_id="text", case_id=case["case_id"], now=case.get("now", ""),
                       history=case.get("history") or [], caller=case.get("customer_profile"), tools_url=MOCK)
    system = build_system(info, s.openers[0], manual_path=s.manual_path)
    r = Responder(make_llm(), _SilentTTS(), call_tool, tools=tools, settings=s,
                  normalize_args=normalize_args)

    async def sink(pcm):
        pass

    spoken = await r.respond(system, [Message("user", case["customer_utterance"])], sink)
    ref = case.get("reference") or {}
    print(f"== {case['case_id']}\n손님: {case['customer_utterance']}")
    print("함수:", [(n, a, st) for n, a, st in calls] or "없음")
    print("우리 답:", " ".join(spoken))
    print("참고 함수:", [t.get("name") for t in ref.get("tool_calls") or []] or "없음")
    print("참고 답:", str(ref.get("answer", "")).replace("\n", " ")[:200])


def main() -> None:
    load_dotenv(ROOT / ".env")
    want = set(sys.argv[1:])
    cases = [json.loads(ln) for ln in CASES.open(encoding="utf-8")]
    for c in cases:
        if c["case_id"] in want:
            asyncio.run(run(c))


if __name__ == "__main__":
    main()
