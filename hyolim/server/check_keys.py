"""API 키가 되는지 확인 — 음성·글을 거의 보내지 않아 크레딧이 들지 않는다. 키 값은 출력하지 않는다.

  python3 -m server.check_keys
"""
import asyncio
import json

import httpx
import websockets

from . import config as C


def shape(k: str) -> str:
    return f"길이 {len(k)}, 앞부분 '{k[:5]}…'" if k else "비어 있음"


async def check_soniox():
    k = C.SONIOX_API_KEY
    print(f"[Soniox] 주소 {C.STT_URL}\n         키 {shape(k)} · 모델 {C.STT_MODEL}")
    if not k:
        return print("  → 키가 없습니다 (.env 의 SONIOX_API_KEY)")
    try:
        try:
            ws = await websockets.connect(C.STT_URL, additional_headers={"Authorization": f"Bearer {k}"}, open_timeout=8)
        except TypeError:
            ws = await websockets.connect(C.STT_URL, extra_headers={"Authorization": f"Bearer {k}"}, open_timeout=8)
        async with ws:
            cfg = {"model": C.STT_MODEL, "audio_format": "pcm_s16le", "sample_rate": 16000, "num_channels": 1, "language_hints": ["ko"]}
            if not k.startswith("baze"):
                cfg["api_key"] = k
            await ws.send(json.dumps(cfg))
            await ws.send("")                      # 음성 없이 바로 끝 → 인증만 확인
            async for m in ws:
                r = json.loads(m)
                if r.get("error_code"):
                    if "No audio" in str(r.get("error_message")):
                        return print("  → 성공! 키가 됩니다 (음성을 안 보내서 '음성 없음'이라고 답한 것 = 인증 통과)")
                    return print(f"  → 실패: {r.get('error_code')} {r.get('error_type')} — {r.get('error_message')}")
                if r.get("finished"):
                    return print("  → 성공! 키가 됩니다 (보낸 음성 0초)")
    except websockets.ConnectionClosed as e:
        reason = e.rcvd.reason if e.rcvd else ""
        print(f"  → 실패: 서버가 연결을 끊음 — {reason}")
        if "Invalid API key" in reason:
            print("     (주소와 보내는 방식은 맞고, 키 값 자체를 서버가 인정하지 않는다는 뜻)")
        if "not supplied" in reason:
            print("     (키를 보내는 방식이 서버가 원하는 것과 다르다는 뜻)")
    except Exception as e:
        print(f"  → 연결 실패: {type(e).__name__} {str(e)[:120]}")


async def check_openai():
    k = C.OPENAI_API_KEY
    print(f"[GPT] 키 {shape(k)} · 답 모델 {C.LLM_MODEL} · 채점 모델 {C.JUDGE_MODEL}")
    if not k:
        return print("  → 키가 없습니다 (.env 의 BAZE_API_KEY 또는 OPENAI_API_KEY)")
    print(f"         주소 {C.LLM_URL}" + ("  (베이즈 게이트웨이)" if C.USE_BAZE else ""))
    async with httpx.AsyncClient(timeout=20) as c:
        for model in (C.LLM_MODEL, C.JUDGE_MODEL):        # 답 생성 모델, 채점 모델
            body = {"model": model, "messages": [{"role": "user", "content": "1"}], "max_completion_tokens": 16}
            r = await c.post(C.LLM_URL, json=body, headers={"Authorization": f"Bearer {k}"})
            ok = r.status_code == 200
            print(f"  {model:14s} → " + ("성공!" if ok else f"실패: {r.status_code} {r.text[:120]}"))


async def check_gemini():
    k = C.GEMINI_API_KEY
    print(f"[Gemini] 키 {shape(k)} · 모델 {C.TTS_MODEL}")
    if not k:
        return print("  → 키가 없습니다 (.env 의 GEMINI_API_KEY) — 음성합성을 빼고 시험할 때는 없어도 됩니다")
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(f"https://generativelanguage.googleapis.com/v1beta/models/{C.TTS_MODEL}", headers={"x-goog-api-key": k})
    print("  → 성공! 키와 모델이 됩니다" if r.status_code == 200 else f"  → 실패: {r.status_code} {r.text[:120]}")


async def main():
    await check_soniox()
    print()
    await check_openai()
    print("\n[Gemini] 음성합성은 지금 빼고 시험하므로 확인하지 않습니다")


if __name__ == "__main__":
    asyncio.run(main())
