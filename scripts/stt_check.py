"""연습 음성을 Soniox(stt-rt-v6)로 받아써 문맥 힌트 유무를 비교한다(STT만, TTS 0).

  python scripts/stt_check.py P0158-01 P0049-01 ...   # 힌트 없음 vs agent.config 기본 문맥

출력: 원문 / 힌트 없음 / 힌트 있음.
"""
from __future__ import annotations

import asyncio
import json
import sys
import wave
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT.parent / "02_대회자료" / "directon-2026-voice-cs" / "data"


async def transcribe(stt, pcm: bytes, context: str) -> str:
    stream = await stt.open_stream(context)
    try:
        for i in range(0, len(pcm), 640):
            await stream.send_audio(pcm[i:i + 640])
            await asyncio.sleep(0.005)               # 실시간의 4배 속도
        for _ in range(50):                          # 말 끝 뒤 무음 1초
            await stream.send_audio(bytes(640))
        return await stream.finalize(3.0)
    finally:
        await stream.close()


async def main(ids: list[str]) -> None:
    load_dotenv(ROOT / ".env")
    from agent.config import Settings
    from agent.providers.factory import make_stt
    stt = make_stt()
    ctx = Settings().stt_context
    cases = {json.loads(l)["case_id"]: json.loads(l) for l in open(DATA / "practice_cases.jsonl", encoding="utf-8")}
    for cid in ids:
        with wave.open(str(DATA / "audio" / "practice" / f"{cid}.wav"), "rb") as w:
            pcm = w.readframes(w.getnframes())
        plain, hinted = await asyncio.gather(transcribe(stt, pcm, ""), transcribe(stt, pcm, ctx))
        print(f"== {cid}\n  원문: {cases[cid]['customer_utterance'][:120]!r}\n  없음: {plain!r}\n  힌트: {hinted!r}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
