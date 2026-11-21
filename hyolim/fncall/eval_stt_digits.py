"""음성 인식 설정별 '번호 받아쓰기' 정확도 비교 (OpenAI 키 필요, 측정용).

번호(전화번호·주문번호)를 말로 부르는 연습 문항만 골라, 음성 파일 전체를 설정별로 받아쓴 뒤
우리 번호 추출기로 뽑은 후보에 정답 번호가 있는지 센다. 실시간 측정 로그(쪼개서 받아쓴 글)도 같은 기준으로 비교한다.

  python3 -m fncall.eval_stt_digits
"""
from __future__ import annotations

import asyncio
import json
import os
import re

import httpx

from server import config as C
from .numbers_ko import extract_ids, parse_now

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
dg = lambda v: re.sub(r"\D", "", str(v or ""))
DIGIT_PROMPT = ("건강기능식품 쇼핑몰 고객센터 통화. 손님이 휴대폰 번호(010으로 시작하는 11자리)나 주문번호(2026으로 시작하는 8자리-7자리)를 "
                "한 자리씩 불러 줍니다. 숫자는 아라비아 숫자로 적으세요. 예: 공일공 일이삼사 오육칠팔 → 010-1234-5678")
CONFIGS = {
    "4o-통째": {"model": "gpt-4o-transcribe", "prompt": None},
    "4o-통째+숫자안내": {"model": "gpt-4o-transcribe", "prompt": DIGIT_PROMPT},
    "4o-mini-통째+숫자안내": {"model": "gpt-4o-mini-transcribe", "prompt": DIGIT_PROMPT},
    "gpt-transcribe-통째+숫자안내": {"model": "gpt-transcribe", "prompt": DIGIT_PROMPT},
}


def number_cases():
    out = []
    meta = json.load(open(os.path.join(ROOT, "data", "audio", "practice", "meta.json"), encoding="utf-8"))
    for l in open(os.path.join(ROOT, "data", "practice_cases.jsonl"), encoding="utf-8"):
        c = json.loads(l)
        if c["case_id"] not in meta:
            continue
        ud = dg(c["customer_utterance"])
        want = set()
        for t in c["reference"]["tool_calls"]:
            for k in ("buyer_cellphone", "customer_phone_number", "phone_number", "order_id", "order_number"):
                v = dg(t["arguments"].get(k))
                if v and len(v) >= 10 and v in ud:          # 손님이 이번 말에서 직접 부른 번호만
                    want.add(v)
        if want:
            out.append((c, want))
    return out


def hit(text, want, now):
    got = {dg(x.value) for x in extract_ids(text or "", now)}
    return bool(want & got)


async def transcribe(http, path, cfg):
    data = {"model": cfg["model"], "language": "ko"}
    if cfg["prompt"]:
        data["prompt"] = cfg["prompt"]
    for _ in range(3):
        r = await http.post("https://api.openai.com/v1/audio/transcriptions", data=data,
                            files={"file": (os.path.basename(path), open(path, "rb"), "audio/wav")},
                            headers={"Authorization": f"Bearer {C.OPENAI_API_KEY}"}, timeout=60)
        if r.status_code == 200:
            return r.json().get("text", "")
        if r.status_code == 400:
            return f"[오류 {r.text[:80]}]"
        await asyncio.sleep(2)
    return ""


async def main():
    cases = number_cases()
    print(f"번호를 말로 부른 문항 {len(cases)}개\n")
    heard = json.load(open(os.path.join(ROOT, "eval", "heard_openai.json"), encoding="utf-8"))
    for run, texts in heard.items():
        n = sum(hit(texts.get(c["case_id"]), w, parse_now(c.get("now"))) for c, w in cases)
        print(f"{'실시간(0.5초 쪼개기) ' + run:32s} {n}/{len(cases)}")
    sem = asyncio.Semaphore(6)
    async with httpx.AsyncClient() as http:
        for name, cfg in CONFIGS.items():
            async def one(c, w):
                async with sem:
                    t = await transcribe(http, os.path.join(ROOT, "data", "audio", "practice", c["case_id"] + ".wav"), cfg)
                    return c["case_id"], t, hit(t, w, parse_now(c.get("now")))
            res = await asyncio.gather(*[one(c, w) for c, w in cases])
            n = sum(h for _, _, h in res)
            print(f"{name:32s} {n}/{len(cases)}")
            for cid, t, h in res:
                if not h:
                    print(f"    x {cid}: {t[:60]!r}")


if __name__ == "__main__":
    asyncio.run(main())
