"""LLM 지연·정확도·비용 실측 2차(AG-T15c). 실제 Gemini API를 호출한다.

    uv run python scripts/bench_llm_r2.py                      # 기본: 질문 10개 × 3라운드 × 모드 8개
    uv run python scripts/bench_llm_r2.py --rounds 1 --modes single35,hedge600

- 프롬프트: agent.prompts.build_system(실제 규칙·매뉴얼) + legacy/shop_mock/tools.json(14개 함수 정의).
  호출마다 now를 바꿔 '이번 통화' 접미부를 다르게 한다(고정 접두부는 같음 — 실서버와 같은 조건).
- 모드는 같은 시간대에 번갈아 돈다(질문·라운드마다 순서 회전).
- 측정: 첫 이벤트(텍스트·툴 호출), 툴 결정(첫 tool_call), 툴 결과 후속 호출 첫 이벤트, 정확도(기대 함수 포함,
  툴 없음 질문은 미호출이 정답), usage_metadata(입력·캐시·출력 토큰) → 호출당 비용.
  동시 요청에서 취소된 쪽은 usage가 오지 않는다 → 같은 입력 토큰이 청구된다고 가정해 추정 비용에 더한다.
- 원시 결과: runs/llm_r2/<시각>.jsonl. 요약은 표준 출력(마크다운 표).
API 키는 환경변수(GEMINI_API_KEY)에서만 읽고 출력하지 않는다.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from agent.core.types import Message, SessionInfo  # noqa: E402
from agent.prompts import build_system  # noqa: E402
from agent.providers._gemini import make_gemini_client  # noqa: E402
from agent.providers.fallback_llm import FallbackLLM  # noqa: E402
from agent.providers.gemini_llm import GeminiLLM  # noqa: E402
from agent.providers.hedged_llm import HedgedLLM  # noqa: E402
from agent.providers.llm_cache import ContextCache, split_system  # noqa: E402

M35, M31 = "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"
# USD / 1M 토큰 (2026-11 Standard 가격표: 입력, 캐시 입력, 출력)
PRICE = {M35: (0.30, 0.03, 2.50), M31: (0.25, 0.025, 1.50)}
STORAGE_PER_MTOK_H = 1.00
CALL_TIMEOUT_S = 20.0

QUESTIONS: list[tuple[str, str | None]] = [
    ("주문번호 2611200105 배송 어디쯤 왔어요?", "track_delivery"),
    ("제 번호 01023456781로 주문한 거 목록 좀 보여주세요.", "get_order_list"),
    ("주문번호 2611200105 취소해 주세요. 단순 변심이에요.", "cancel_order"),
    ("주문번호 2611200105 환불 언제 들어와요?", "get_refund_status"),
    ("01023456781 번호로 쓸 수 있는 쿠폰 있나요?", "get_coupons"),
    ("그냥 상담원 연결해 주세요.", "transfer_to_agent|request_callback"),   # 점심·영업시간 밖이면 콜백이 매뉴얼상 정답
    ("{product} 재고 있어요?", "check_stock"),
    ("영업시간이 어떻게 되나요?", None),
    ("반품하고 싶은데요.", None),
    ("오늘 저녁 메뉴 추천해 주세요.", None),
]
ALL_MODES = ["single35", "single31", "fallback", "hedge800", "hedge1500", "race", "single35_cache",
             "hedge800_cache"]


def load_tools() -> list[dict]:
    raw = json.loads((ROOT / "legacy/shop_mock/tools.json").read_text(encoding="utf-8"))
    return [{"type": "function", "function": t} for t in raw]


def product_name() -> str:
    d = json.loads((ROOT / "legacy/shop_mock/mock_data.json").read_text(encoding="utf-8"))
    return d["products"][0]["name"]


class Usage:
    def __init__(self):
        self.rows: list[dict] = []

    def __call__(self, u: dict) -> None:
        self.rows.append(u)


def make_modes(client, usage: Usage, names: list[str]) -> tuple[dict, list[ContextCache]]:
    caches: list[ContextCache] = []

    def g(model, cache=None):
        return GeminiLLM(client=client, model=model, context_cache=cache, on_usage=usage)

    out = {}
    for n in names:
        cache = None
        if n.endswith("_cache"):
            cache = ContextCache(client=client, ttl_s=1800)
            caches.append(cache)
        base = n.removesuffix("_cache")
        if base == "single35":
            out[n] = g(M35, cache)
        elif base == "single31":
            out[n] = g(M31, cache)
        elif base == "fallback":
            out[n] = FallbackLLM([g(M35, cache), g(M31, cache)], first_event_timeout_s=2.5)
        elif base.startswith("hedge"):
            out[n] = HedgedLLM([g(M35, cache), g(M31, cache)], hedge_delay_s=int(base[5:]) / 1000)
        elif base == "race":
            out[n] = HedgedLLM([g(M35, cache), g(M31, cache)], mode="race")
        else:
            raise SystemExit(f"unknown mode {n}")
    return out, caches


async def one_call(llm, system, messages, tools) -> dict:
    t0 = time.perf_counter()
    first = tool_t = None
    texts: list[str] = []
    calls = []
    err = ""
    try:
        async with asyncio.timeout(CALL_TIMEOUT_S):
            async for ev in llm.stream(system, messages, tools):
                now = time.perf_counter() - t0
                if ev.type in ("text", "tool_call") and first is None:
                    first = now
                if ev.type == "text":
                    texts.append(ev.text)
                elif ev.type == "tool_call":
                    tool_t = tool_t if tool_t is not None else now
                    calls.append(ev.tool_call)
    except (Exception, TimeoutError) as e:  # noqa: BLE001
        err = repr(e)[:200]
    return {"first": first, "tool": tool_t, "total": time.perf_counter() - t0, "text": "".join(texts),
            "calls": calls, "err": err}


def served(llm) -> str:
    return getattr(llm, "last_model", "") or getattr(llm, "model", "")


def hedge_count(llm) -> int:
    st = getattr(llm, "stats", {}) or {}
    return sum(v for k, v in st.items() if k.startswith("hedge:"))


def cost(rows: list[dict]) -> float:
    c = 0.0
    for u in rows:
        pin, pc, pout = PRICE.get(u["model"], PRICE[M35])
        c += ((u["prompt"] - u["cached"]) * pin + u["cached"] * pc + (u["output"] + u["thoughts"]) * pout) / 1e6
    return c


def loser_cost(winner_rows: list[dict], loser_model: str) -> float:
    if not winner_rows:
        return 0.0
    u = winner_rows[0]
    pin, pc, _ = PRICE.get(loser_model, PRICE[M35])
    return ((u["prompt"] - u["cached"]) * pin + u["cached"] * pc) / 1e6


async def run_case(mode: str, llm, q: str, expect: str | None, tools, usage: Usage, k: int) -> dict:
    info = SessionInfo(session_id=f"r2-{k}", now=f"2026-11-{20 + k % 7:02d} {9 + k % 9:02d}:{(k * 7) % 60:02d}")
    system = build_system(info, "네, 고객님.")
    usage.rows.clear()
    h0 = hedge_count(llm)
    msgs = [Message("user", q)]
    r = await one_call(llm, system, msgs, tools)
    hedged = hedge_count(llm) - h0
    model1 = served(llm)
    rows1 = list(usage.rows)
    names = [c.name for c in r["calls"]]
    ok = any(e in names for e in expect.split("|")) if expect else (not names and not r["err"])
    rec = {"mode": mode, "q": q, "expect": expect, "names": names, "ok": ok, "first": r["first"],
           "tool": r["tool"], "total": r["total"], "err": r["err"], "served": model1, "hedged": hedged,
           "text": r["text"][:80], "usage": rows1}
    extra = 0.0
    is_race = isinstance(llm, HedgedLLM) and llm.mode == "race"
    if (hedged or is_race) and isinstance(llm, HedgedLLM) and len(llm.llms) > 1:
        loser = M31 if model1 == M35 else M35
        extra += loser_cost(rows1, loser)
    # 툴 결과 후속 호출(같은 llm 객체 → hedge는 같은 모델로 고정)
    if r["calls"] and not r["err"]:
        follow = msgs + [Message("assistant", tool_calls=r["calls"])] + [
            Message("tool", content=json.dumps({"status": "ok", "data": {"result": "처리되었습니다"}},
                                               ensure_ascii=False), tool_call_id=c.id, name=c.name)
            for c in r["calls"]]
        usage.rows.clear()
        h1 = hedge_count(llm)
        f = await one_call(llm, system, follow, tools)
        rec.update(f_first=f["first"], f_total=f["total"], f_err=f["err"], f_served=served(llm),
                   f_text=f["text"][:60], f_usage=list(usage.rows), f_hedged=hedge_count(llm) - h1)
        if is_race and len(llm.llms) > 1 and rec["f_served"] != model1:
            extra += loser_cost(rec["f_usage"], M31 if rec["f_served"] == M35 else M35)
    rec["cost"] = cost(rec["usage"]) + cost(rec.get("f_usage", [])) + extra
    rec["cost_loser"] = extra
    return rec


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    i = min(len(xs) - 1, max(0, round(p / 100 * (len(xs) - 1))))
    return xs[i]


def fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def summarize(recs: list[dict], modes: list[str]) -> str:
    lines = ["| 모드 | n | 첫 이벤트 p50 | p95 | 최대 | 툴 결정 p50 | p95 | 후속 첫 이벤트 p50 | p95 | 정확도 | 오류·시간초과 | "
             "대체 모델 채택 | hedge 발동 | 입력 토큰 평균 | 캐시 토큰 평균 | 호출당 비용(¢, 후속 포함) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in modes:
        rs = [r for r in recs if r["mode"] == m]
        if not rs:
            continue
        firsts = [r["first"] for r in rs]
        tools = [r["tool"] for r in rs if r["expect"]]
        ff = [r.get("f_first") for r in rs if "f_first" in r]
        us = [u for r in rs for u in r["usage"]]
        alt = sum(1 for r in rs if r["served"] == M31 and not m.startswith("single31"))
        lines.append(
            f"| {m} | {len(rs)} | {fmt(pct(firsts, 50))} | {fmt(pct(firsts, 95))} | "
            f"{fmt(max([x for x in firsts if x is not None], default=None))} | {fmt(pct(tools, 50))} | "
            f"{fmt(pct(tools, 95))} | {fmt(pct(ff, 50))} | {fmt(pct(ff, 95))} | "
            f"{sum(r['ok'] for r in rs)}/{len(rs)} | {sum(1 for r in rs if r['err'] or r.get('f_err'))} | "
            f"{alt} | {sum(1 for r in rs if r['hedged'])} | "
            f"{fmt(statistics.mean([u['prompt'] for u in us]) if us else None, 0)} | "
            f"{fmt(statistics.mean([u['cached'] for u in us]) if us else None, 0)} | "
            f"{fmt(100 * statistics.mean([r['cost'] for r in rs]), 3)} |")
    return "\n".join(lines)


async def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--modes", default=",".join(ALL_MODES))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    tools = load_tools()
    pname = product_name()
    questions = [(q.format(product=pname), e) for q, e in QUESTIONS]
    client = make_gemini_client()
    usage = Usage()
    llms, caches = make_modes(client, usage, modes)

    # 캐시 미리 생성(정상 상태 측정). 생성 시간 기록
    probe_sys = build_system(SessionInfo(session_id="p", now="2026-11-20 10:00"), "네, 고객님.")
    prefix = split_system(probe_sys)[0]
    cache_info = []
    for m, llm in llms.items():
        if not m.endswith("_cache"):
            continue
        for g in (getattr(llm, "llms", None) or [llm]):
            t = time.perf_counter()
            g.context_cache.lookup(g.model, prefix, tools)
            await g.context_cache.wait_pending()
            cache_info.append({"mode": m, "model": g.model, "create_s": round(time.perf_counter() - t, 2),
                               "stats": dict(g.context_cache.stats)})
    print("cache:", json.dumps(cache_info, ensure_ascii=False))

    out = Path(args.out) if args.out else ROOT / "runs/llm_r2" / f"{datetime.now():%m%d_%H%M}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    recs: list[dict] = []
    t_start = time.time()
    k = 0
    with out.open("w", encoding="utf-8") as fh:
        for rnd in range(args.rounds):
            for qi, (q, expect) in enumerate(questions):
                rot = (qi + rnd) % len(modes)
                for m in modes[rot:] + modes[:rot]:
                    k += 1
                    rec = await run_case(m, llms[m], q, expect, tools, usage, k)
                    rec["round"] = rnd
                    rec["ts"] = round(time.time() - t_start, 1)
                    recs.append(rec)
                    fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                    fh.flush()
                    print(f"[{rec['ts']:7.1f}s] r{rnd} {m:15s} first={fmt(rec['first'])} "
                          f"tool={fmt(rec['tool'])} ok={rec['ok']} served={rec['served']} "
                          f"hedged={rec['hedged']} {rec['err'][:60]}", flush=True)
    elapsed = time.time() - t_start
    for c in caches:
        await c.aclose()
    print(f"\n측정 {datetime.now():%Y-%m-%d %H:%M}, {elapsed / 60:.1f}분, 기록 {out}")
    print(summarize(recs, modes))
    for m, llm in llms.items():
        st = getattr(llm, "stats", None)
        gems = getattr(llm, "llms", None) or [llm]
        cst = [dict(g.context_cache.stats) for g in gems if getattr(g, "context_cache", None)]
        print(m, "stats", st or "", "cache", cst or "")


if __name__ == "__main__":
    asyncio.run(main())
