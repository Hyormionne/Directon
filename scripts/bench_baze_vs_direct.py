"""BAZE 게이트웨이 vs 공식 직접 API 지연 비교 (LLM 첫 토큰, TTS 첫 오디오). 실호출."""
from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time

from dotenv import load_dotenv

load_dotenv()
from openai import AsyncOpenAI  # noqa: E402

from agent.providers.baze_tts import BazeTTS  # noqa: E402
from agent.providers.gemini_tts import GeminiTTS  # noqa: E402
from agent.providers._gemini import make_gemini_client  # noqa: E402

BAZE = os.environ.get("BAZE_BASE_URL", "https://api.baze.co/v1/gateway")
SYS = "너는 건강기능식품 쇼핑몰 전화 상담원이다. 한두 문장으로 짧게 답한다."
Q = ["배송 언제 와요?", "주문 취소하고 싶어요", "반품하려면 어떻게 해야 돼요?", "영업시간이 어떻게 돼요?"]


async def llm_ttft(client: AsyncOpenAI, q: str) -> float:
    t0 = time.perf_counter()
    stream = await client.responses.create(model="gpt-5.4-mini", instructions=SYS, input=q,
                                           reasoning={"effort": "none"}, stream=True, store=False)
    first = None
    async for ev in stream:
        if first is None and getattr(ev, "type", "") == "response.output_text.delta":
            first = time.perf_counter() - t0
    return first if first is not None else float("nan")


async def tts_first(tts, text: str) -> float:
    t0 = time.perf_counter()
    async for _ in tts.synthesize(text):
        return time.perf_counter() - t0
    return float("nan")


def show(name: str, xs: list[float]) -> None:
    xs = [x for x in xs if x == x]
    print(f"{name:14s} n={len(xs)} p50={statistics.median(xs):.3f}s min={min(xs):.3f} max={max(xs):.3f}")


async def main(n_llm: int, n_tts: int) -> None:
    direct = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    baze = AsyncOpenAI(api_key=os.environ["BAZE_API_KEY"], base_url=BAZE)
    for c in (direct, baze):  # 연결 예열
        await llm_ttft(c, "안녕")
    d, b = [], []
    for i in range(n_llm):
        q = Q[i % len(Q)]
        d.append(await llm_ttft(direct, q))
        b.append(await llm_ttft(baze, q))
    show("LLM direct", d)
    show("LLM baze", b)
    if n_tts:
        model = "gemini-3.8-flash-lite-tts"
        g = GeminiTTS(client=make_gemini_client(os.environ["GEMINI_API_KEY"]), model=model)
        z = BazeTTS(api_key=os.environ["BAZE_API_KEY"], base_url=BAZE, model=model)
        gd, zb = [], []
        for i in range(n_tts):
            text = "네, 주문 내역 확인해 드릴게요." if i % 2 == 0 else "고객님 주문은 내일 도착 예정입니다."
            gd.append(await tts_first(g, text))
            zb.append(await tts_first(z, text))
        show("TTS direct", gd)
        show("TTS baze", zb)


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 8, int(sys.argv[2]) if len(sys.argv) > 2 else 2))
