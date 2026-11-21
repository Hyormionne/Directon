"""요청 종류 판정률 평가. 시스템 프롬프트(agent.knowledge)로 고객 요청을 맞게 가려 맞는 행동을 했는지 본다.

text_eval.run_case(실제 서버와 같은 프롬프트·응답기·검사·모의 함수, STT·TTS 생략)로 한 턴을 돌리고,
그 턴의 행동을 하나로 분류해 data/eval/request_types.json의 허용 행동과 비교한다. 응대 채점(OpenAI 채점 AI)은 하지 않는다.

행동 분류(우선순위 순)
  1. 접수 함수를 실행했으면 ACT:<함수>
  2. 조회·검색을 실행했으면 LOOKUP_ORDER / LOOKUP_RETURN / SEARCH (허용 행동과 겹치는 것이 있으면 그것)
  3. 함수 없이 말만 했으면 말의 내용으로 HANDOFF / ASK_ID / ASK_PRODUCT / CLARIFY / ASK_DETAIL / ANSWER

  uv run python scripts/intent_eval.py --out runs/intent/base [--repeat 3] [--source original|stt] [--only P0023-01,...]
  uv run python scripts/intent_eval.py --set haruon200 --out runs/intent/h200     # 녹음 대본 200문항(글 그대로)

--set practice(기본): 공식 연습 80문항 + data/eval/request_types.json
--set haruon200: data/eval/haruon200/cases.jsonl + labels.json (모의 DB에 주문이 없어 조회는 not_found)

필요: .env의 OPENAI_API_KEY. 공식 키트 위치는 --kit 또는 자동 탐색
(./DIRECT_ON 2026 Starter Kit, ../directon-2026-voice-cs, ../../directon-2026-voice-cs).
결과(runs/ 아래)는 연습 문항 내용을 담으므로 커밋하지 않는다.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
LABELS = ROOT / "data" / "eval" / "request_types.json"
STT_CACHE = ROOT / "runs" / "text" / "stt_cache.json"

ACT_FUNCS = ("cancel_order", "return_order", "return_order_item", "exchange_order", "request_callback")
LOOKUP_KIND = {"get_order_info": "LOOKUP_ORDER", "get_return_info": "LOOKUP_RETURN", "search_products": "SEARCH"}

# ── 말 분류 ──────────────────────────────────────────
# 말을 문장으로 나눠 마지막 질문·요청 문장으로 주된 행동을 정한다. 정책 등을 실제로 안내한 문장이 있으면
# (공감·사과·"확인해 볼게요" 같은 말 제외, 합쳐 25자 이상) ANSWER도 함께 맞은 것으로 본다(안내 뒤 덧붙인 질문).
_ASK_END = r"(주시겠어요|주실\s?수\s?있(을까요|나요|으실까요)|주세요|주시면|있으실까요|맞으실까요|괜찮으실까요|괜찮을까요|할까요|드릴까요|\?)"
_HANDOFF = re.compile(r"상담원(에게|께|과|으로|분께)?\s*(연결|넘겨|전달|바꿔)|상담원\s*연결")
_ASK_ID = re.compile(r"(주문\s?번호|(?<!입금자\s)(?<!입금자)연락처|전화\s?번호|휴대폰\s?번호|(?<!입금자\s)(?<!입금자)성함)")
_ASK_PRODUCT = re.compile(r"(어떤|어느)\s?(제품|상품)|제품명|상품명|제품으로\s?안내")
_CLARIFY = re.compile(r"어떤\s?(도움|문의|용건|내용)|무엇을\s?도와|용건을|문의\s?내용을")
_CLOSING = re.compile(r"(더\s?)?궁금하신\s?(점|게|부분)|다른\s?(문의|궁금)")
_QUESTION = re.compile(_ASK_END)
_FILLER = re.compile(r"셨을\s?것\s?같|죄송|불편|감사|확인해\s?(볼|드릴)게요|확인해\s?보겠습니다|잠시만|도와드릴게요")
_SPLIT = re.compile(r"(?<=[.?!])\s+")
_LINK = re.compile(r"<link>.*?</link>|https?://\S+", re.S)


def _sentences(text: str) -> list[str]:
    return [x.strip() for x in _SPLIT.split(" ".join(_LINK.sub("", text or "").split())) if x.strip()]


def _ask_kind(sent: str) -> str:
    if _ASK_PRODUCT.search(sent):
        return "ASK_PRODUCT"
    if _ASK_ID.search(sent):
        return "ASK_ID"
    if _CLARIFY.search(sent):
        return "CLARIFY"
    return "ASK_DETAIL"


def text_kinds(text: str) -> tuple[str, bool]:
    """(주된 행동, 실제 안내 문장이 있는가)."""
    sents = _sentences(text)
    if _HANDOFF.search(" ".join(sents)):
        return "HANDOFF", True
    asks = [x for x in sents if _QUESTION.search(x) and not _CLOSING.search(x)]
    plain = [x for x in sents if x not in asks and not _CLOSING.search(x) and not _FILLER.search(x)]
    substance = sum(len(x) for x in plain) >= 25
    if not asks:
        return "ANSWER", True
    return _ask_kind(asks[-1]), substance


def classify_text(text: str) -> str:
    return text_kinds(text)[0]


def classify(calls: list[dict], text: str, expect: list[str]) -> str:
    acts = [c["name"] for c in calls if c.get("name") in ACT_FUNCS]
    if acts:
        hit = [a for a in acts if "ACT:" + a in expect]
        return "ACT:" + (hit[0] if hit else acts[0])
    kinds = [LOOKUP_KIND[c["name"]] for c in calls if c.get("name") in LOOKUP_KIND]
    if kinds:
        hit = [k for k in kinds if k in expect]
        return hit[0] if hit else kinds[0]
    kind, substance = text_kinds(text)
    if kind not in expect and substance and "ANSWER" in expect:
        return "ANSWER"
    return kind


# ── 실행 ─────────────────────────────────────────────
def find_kit(arg: str) -> Path:
    cands = [Path(arg)] if arg else []
    cands += [ROOT / "DIRECT_ON 2026 Starter Kit", ROOT.parent / "directon-2026-voice-cs",
              ROOT.parent.parent / "directon-2026-voice-cs"]
    for c in cands:
        if (c / "data" / "practice_cases.jsonl").is_file():
            return c.resolve()
    raise SystemExit("공식 키트를 찾지 못했습니다. --kit <directon-2026-voice-cs 경로>를 주세요.")


def strip_opener(text: str, opener: str) -> str:
    text = (text or "").strip()
    return text[len(opener):].strip() if opener and text.startswith(opener) else text


def report(rows: list[dict], labels: dict) -> str:
    reps = sorted({r["rep"] for r in rows})
    out = []
    accs = [sum(r["ok"] for r in rows if r["rep"] == k) / max(1, sum(r["rep"] == k for r in rows)) for k in reps]
    mean = sum(accs) / len(accs) if accs else 0.0
    out.append(f"# 요청 종류 판정률\n\n전체 {mean * 100:.1f}%  (회차별: {', '.join(f'{a * 100:.1f}' for a in accs)})\n")
    by_type: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r["ok"])
    out.append("| 요청 종류 | 문항×회차 | 맞음 |\n|---|---|---|")
    for t, v in sorted(by_type.items(), key=lambda x: sum(x[1]) / len(x[1])):
        out.append(f"| {t} | {len(v)} | {sum(v) / len(v) * 100:.0f}% |")
    conf = Counter((r["expect"][0], r["observed"]) for r in rows if not r["ok"])
    out.append("\n## 틀린 행동 (기대 첫째 → 실제)\n\n| 기대 | 실제 | 건수 |\n|---|---|---|")
    for (e, o), n in conf.most_common():
        out.append(f"| {e} | {o} | {n} |")
    out.append("\n## 틀린 문항\n")
    miss = defaultdict(list)
    for r in rows:
        if not r["ok"]:
            miss[r["case_id"]].append(r)
    for cid, rs in sorted(miss.items(), key=lambda x: -len(x[1])):
        r = rs[0]
        out.append(f"### {cid} ({len(rs)}/{len(reps)}회 틀림) — {r['type']}, 기대 {r['expect']}")
        out.append(f"- 고객: {r['user_text']}")
        for x in rs:
            calls = ", ".join(f"{c['name']}({json.dumps(c['arguments'], ensure_ascii=False)})" for c in x["calls"]) or "없음"
            out.append(f"- [{x['rep']}] 실제 {x['observed']} / 호출 {calls}")
            out.append(f"  - 말: {x['text']}")
            if x.get("blocked"):
                out.append(f"  - 막힌 호출: {json.dumps(x['blocked'], ensure_ascii=False)}")
        if labels.get(cid, {}).get("note"):
            out.append(f"- 라벨 메모: {labels[cid]['note']}")
        out.append("")
    return "\n".join(out)


async def main_async(a) -> None:
    kit = find_kit(a.kit)
    os.environ.setdefault("DIRECTON_KNOWLEDGE_DIR", str(kit / "data" / "brand"))
    os.environ.setdefault("DIRECTON_MOCK_DATA", str(kit / "data" / "mock_db.json"))
    os.environ.setdefault("DIRECTON_MOCK_TOOLS", str(kit / "data" / "brand" / "tools.json"))
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY가 필요합니다(.env).")

    from agent.providers.factory import ProviderConfig, make_llm
    from mock_api.server import load_store
    from text_eval import run_case

    if a.set == "haruon200":
        set_dir = ROOT / "data" / "eval" / "haruon200"
        labels = json.loads((set_dir / "labels.json").read_text(encoding="utf-8"))
        cases_path = set_dir / "cases.jsonl"
        if a.source == "stt":
            raise SystemExit("haruon200은 아직 녹음이 없어 --source original만 됩니다.")
    else:
        labels = json.loads(LABELS.read_text(encoding="utf-8"))
        cases_path = kit / "data" / "practice_cases.jsonl"
    cases = {json.loads(l)["case_id"]: json.loads(l) for l in open(cases_path, encoding="utf-8")}
    ids = [c.strip() for c in a.only.split(",") if c.strip()] if a.only else [c for c in cases if c in labels]
    if a.limit:
        ids = ids[:a.limit]
    if a.source == "stt":
        if not STT_CACHE.is_file():
            raise SystemExit(f"{STT_CACHE} 없음 — 먼저 `scripts/text_eval.py stt`를 돌리세요")
        from agent.knowledge import fix_brand
        texts = {c: fix_brand(t) for c, t in json.loads(STT_CACHE.read_text(encoding="utf-8")).items()}
        ids = [c for c in ids if c in texts]
    else:
        texts = {c: " ".join(cases[c]["customer_utterance"].split()) for c in ids}

    llm = make_llm(ProviderConfig.from_env(baze_api_key=""))
    store = load_store()
    tools = [{"type": "function", "function": t} for t in store.tools]
    sem = asyncio.Semaphore(a.concurrency)

    async def one(cid: str, rep: int) -> dict:
        async with sem:
            r = await run_case(cases[cid], texts[cid], llm, store, tools)
        lab = labels[cid]
        text = strip_opener(r.get("agent_text", ""), r.get("opener", ""))
        obs = classify(r["tool_calls"], text, lab["expect"])
        return {"case_id": cid, "rep": rep, "type": lab["type"], "expect": lab["expect"], "observed": obs,
                "ok": int(obs in lab["expect"]), "user_text": texts[cid], "text": text, "calls": r["tool_calls"],
                "blocked": r.get("blocked") or [], "error": r.get("error", "")}

    rows = await asyncio.gather(*[one(c, k) for k in range(1, a.repeat + 1) for c in ids])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep = report(list(rows), labels)
    (out / "report.md").write_text(rep, encoding="utf-8")
    print(rep.split("\n## 틀린 문항")[0])
    print(f"\n자세한 결과: {out / 'report.md'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--kit", default="")
    p.add_argument("--set", choices=["practice", "haruon200"], default="practice")
    p.add_argument("--only", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--repeat", type=int, default=1, help="같은 설정으로 N번(변경 효과 판단은 3 이상)")
    p.add_argument("--source", choices=["original", "stt"], default="original")
    p.add_argument("--concurrency", type=int, default=8)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
