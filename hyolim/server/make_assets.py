"""채움말 음성을 미리 합성해 server/assets/*.pcm 으로 저장한다. 제출 전에 한 번 돌려 함께 넣는다.

  python3 -m server.make_assets          (GEMINI_API_KEY 필요)
"""
import asyncio
import os

import httpx

from . import app, config as C


async def main():
    async with httpx.AsyncClient(timeout=30) as http:
        for k, text in {**app.FILLER_TEXT, **app.FALLBACK_TEXT}.items():
            pcm = await app.tts.synth(http, text)
            p = os.path.join(app.ASSETS, f"{k}.pcm")
            os.makedirs(app.ASSETS, exist_ok=True)
            open(p, "wb").write(pcm)
            print(f"{k:14s} {len(pcm) / 2 / C.RATE:.2f}s  {text}")


if __name__ == "__main__":
    asyncio.run(main())
