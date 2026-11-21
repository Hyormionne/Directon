"""Soniox 로 번호 받아쓰기 정확도 재기 — 크레딧 절약판 (SONIOX_API_KEY 필요).

번호를 말로 부르는 연습 문항(23개, 음성 약 3.2분)만 Soniox(stt-rt-v6)에 보내 받아쓰고,
받아쓴 글을 eval/heard_soniox.json 에 저장한다. 이후 번호 처리 코드를 고칠 때는
`python3 -m fncall.eval_heard --heard eval/heard_soniox.json` 로 저장한 글만 다시 써서 크레딧이 들지 않는다.

  python3 -m fncall.eval_stt_soniox --limit 1     # 연결 확인 (음성 약 5초)
  python3 -m fncall.eval_stt_soniox               # 번호 23문항 (음성 약 3.2분)
  python3 -m fncall.eval_stt_soniox --all         # 연습 80문항 전체 (약 7.9분) — 필요할 때만

음성은 실시간보다 빠르게 보낸다(요금은 보낸 음성 길이 기준이라 같고, 기다리는 시간만 줄어든다).
이미 저장된 문항은 다시 보내지 않는다(크레딧 중복 방지). 다시 재려면 --redo.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import wave

import websockets

from server import config as C
from server.stt import soniox_context
from .eval_stt_digits import hit, number_cases
from .numbers_ko import parse_now

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "eval", "heard_soniox.json")


async def transcribe(path: str) -> str:
    w = wave.open(path)
    pcm = w.readframes(w.getnframes())
    headers = {"Authorization": f"Bearer {C.SONIOX_API_KEY}"}
    try:
        ws = await websockets.connect(C.STT_URL, additional_headers=headers, max_size=None, open_timeout=10)
    except TypeError:                                         # websockets 12 이하
        ws = await websockets.connect(C.STT_URL, extra_headers=headers, max_size=None, open_timeout=10)
    async with ws:
        cfg = {"model": C.STT_MODEL, "audio_format": "pcm_s16le", "sample_rate": 16000, "num_channels": 1,
               "language_hints": ["ko"], "enable_endpoint_detection": False, "context": soniox_context()}
        if not C.SONIOX_API_KEY.startswith("baze"):           # 베이즈(행사 게이트웨이) 키는 헤더로만 보낸다
            cfg["api_key"] = C.SONIOX_API_KEY
        await ws.send(json.dumps(cfg, ensure_ascii=False))
        for i in range(0, len(pcm), 32000):                  # 1초씩 빠르게 보낸다
            await ws.send(pcm[i:i + 32000])
        await ws.send("")                                     # 빈 텍스트 프레임 = 끝
        text = ""
        async for m in ws:
            r = json.loads(m)
            if r.get("error_code"):
                raise RuntimeError(f"Soniox 오류 {r.get('error_code')} {r.get('error_type')}: {r.get('error_message')}")
            for t in r.get("tokens", []):
                if t.get("is_final") and t.get("text") not in ("<end>", "<fin>"):
                    text += t.get("text", "")
            if r.get("finished"):
                break
        return text.strip()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="번호 문항만이 아니라 연습 80문항 전체")
    ap.add_argument("--redo", action="store_true", help="저장된 문항도 다시 보낸다")
    a = ap.parse_args()
    if not C.SONIOX_API_KEY:
        raise SystemExit("SONIOX_API_KEY 가 없습니다. 맨 위 폴더 .env 에 SONIOX_API_KEY=... 를 넣어 주세요.")

    cases = number_cases()
    if a.all:
        meta = json.load(open(os.path.join(ROOT, "data", "audio", "practice", "meta.json"), encoding="utf-8"))
        want = {c["case_id"]: w for c, w in cases}
        allc = [json.loads(l) for l in open(os.path.join(ROOT, "data", "practice_cases.jsonl"), encoding="utf-8")]
        cases = [(c, want.get(c["case_id"], set())) for c in allc if c["case_id"] in meta]
    cases = cases[: a.limit or None]

    saved = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    texts = saved.setdefault("soniox", {})
    sec = 0.0
    for c, _ in cases:
        cid = c["case_id"]
        if cid in texts and not a.redo:
            continue
        path = os.path.join(ROOT, "data", "audio", "practice", cid + ".wav")
        w = wave.open(path)
        sec += w.getnframes() / w.getframerate()
        t = await transcribe(path)
        if not t:
            print(f"  {cid}: (받아쓴 글 없음 — 저장하지 않음)", flush=True)
            continue
        texts[cid] = t
        json.dump(saved, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)   # 한 문항마다 저장
        text = texts.get(cid) or ""
        print(f"  {cid}: {(texts.get(cid) or '')[:60]!r}", flush=True)
    print(f"\n이번에 보낸 음성 {sec / 60:.2f}분 · 저장: {os.path.relpath(OUT, ROOT)}")

    num = [(c, w) for c, w in cases if w]
    if num:
        n = sum(hit(texts.get(c["case_id"]), w, parse_now(c.get("now"))) for c, w in num)
        print(f"번호 맞힌 수: Soniox {n}/{len(num)}  (참고: OpenAI 실시간 4/23, 통째 받아쓰기 5/23)")
        for c, w in num:
            if not hit(texts.get(c["case_id"]), w, parse_now(c.get("now"))):
                print(f"    x {c['case_id']} 정답 {sorted(w)} ← {(texts.get(c['case_id']) or '')[:50]!r}")


if __name__ == "__main__":
    asyncio.run(main())
