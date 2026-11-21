"""과적합 확인 — 연습 문항의 손님 말을 GPT 로 '다르게 말하게' 바꿔서 함수 점수가 유지되는지 본다.

규칙(의도 판단 정규식 등)을 연습 80문항을 보며 다듬었기 때문에, 같은 뜻을 다른 말로 하면 무너지는지 확인한다.
번호·이름·제품명은 그대로 두고 표현만 바꾼다. 바꾼 문장은 캐시 파일에 저장해 다시 돌릴 때 재사용한다.

  python3 -m fncall.eval_paraphrase                # 문항당 2가지 변형 (OPENAI_API_KEY 필요, 첫 실행만 GPT 호출)
  python3 -m fncall.eval_paraphrase --show-fail
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os

import httpx

from server import config as C          # .env 읽기
from .eval_fncall import mock_server, score
from .pipeline import run_turn

ROOT = C.ROOT
CACHE = os.path.join(ROOT, "eval", "paraphrase_cache.json")
PROMPT = """아래는 건강기능식품 쇼핑몰 고객센터에 전화한 손님의 말들이다. 각 문장을 같은 뜻, 같은 요청으로 '다르게 말한' 버전 2개씩 만들어라.
규칙:
- 전화로 말하듯 자연스러운 한국어 구어. 어순·단어·말투(존댓말/반말, 짧게/길게)를 바꿔라. 같은 표현을 되풀이하지 마라.
- 숫자(전화번호·주문번호·날짜·수량), 사람 이름, 제품명은 글자 그대로 둔다.
- 요청을 더하거나 빼지 않는다. 뜻이 애매한 문장은 애매한 채로 둔다.
- 붙여넣은 문자 안내문처럼 보이는 것은 손님이 그 내용을 말로 전하는 문장으로 바꾼다.
출력: JSON {"items": [{"id": "...", "variants": ["...", "..."]}]}

"""


STYLES = {
    "default": "",
    "elderly": "\n추가 지시: 60~70대 손님처럼 말하라. 군말(아, 그, 저기, 뭐냐)과 돌려 말하기, 짧게 끊어 말하기, 구어체 맞춤법을 섞어라.\n",
    "terse": "\n추가 지시: 바쁜 손님처럼 아주 짧게, 조사·어미를 생략하고 핵심 단어만 말하라(예: '반품요. 전부.').\n",
}


async def paraphrase(cases, http, cache_path=CACHE, style="default") -> dict:
    cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}
    todo = [c for c in cases if c["case_id"] not in cache]
    for i in range(0, len(todo), 10):
        batch = todo[i:i + 10]
        lines = "\n".join(json.dumps({"id": c["case_id"], "text": c["customer_utterance"][:400]}, ensure_ascii=False) for c in batch)
        body = {"model": C.LLM_MODEL, "messages": [{"role": "user", "content": PROMPT + STYLES[style] + lines}],
                "response_format": {"type": "json_object"}}
        r = await http.post(C.LLM_URL, json=body, headers={"Authorization": f"Bearer {C.OPENAI_API_KEY}"}, timeout=120)
        r.raise_for_status()
        for it in json.loads(r.json()["choices"][0]["message"]["content"]).get("items", []):
            if it.get("id") and it.get("variants"):
                cache[it["id"]] = it["variants"][:2]
        json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  바꿔 말하기 {min(i + 10, len(todo))}/{len(todo)}", flush=True)
    return cache


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-fail", action="store_true")
    ap.add_argument("--style", default="default", choices=list(STYLES), help="바꿔 말하기 말투 (새 말투 = 새 검증 세트)")
    a = ap.parse_args()
    cases = [json.loads(l) for l in open(os.path.join(ROOT, "data", "practice_cases.jsonl"), encoding="utf-8")]
    async with httpx.AsyncClient() as http:
        cache_path = CACHE if a.style == "default" else CACHE.replace(".json", f"_{a.style}.json")
        para = await paraphrase(cases, http, cache_path, a.style)
    store = mock_server.Store(os.path.join(ROOT, "data", "mock_db.json"), os.path.join(ROOT, "data", "brand", "tools.json"))

    async def call(n, args):
        return store.lookup(n, args)[0]

    for k in range(3):
        ok, fails = 0, []
        for c in cases:
            utt = c["customer_utterance"] if k == 0 else (para.get(c["case_id"]) or [c["customer_utterance"]] * 2)[min(k - 1, 1)]
            s = {"history": c["history"], "now": c.get("now"), "caller": c.get("customer_profile")}
            out = await run_turn(s, utt, call)
            got = [{"name": x.name, "arguments": x.arguments} for x, _ in out.calls]
            hit, why = score.tool_match(c["reference"]["tool_calls"], got, c["history"])
            ok += hit
            if not hit:
                fails.append((c["case_id"], why, utt[:60].replace("\n", " ")))
        name = ["원문", "바꿔 말하기 1", "바꿔 말하기 2"][k]
        print(f"[{name}] 함수 점수 {ok / len(cases) * 100:.1f} ({ok}/{len(cases)})")
        if a.show_fail and k:
            for f in fails:
                print("   -", *f)


if __name__ == "__main__":
    asyncio.run(main())
