"""한 턴의 함수 처리 전체: 번호 확정 → 조회(병렬 검증) → 접수 판단(게이트) → 템플릿 답.

상담 서버에서는 이렇게 쓴다.

    async with ToolClient(session["tools_url"], session["session_id"]) as tc:
        # 엔드포인트가 잡히면(또는 최종 STT 가 나오면) 바로 시작한다. LLM 응답을 기다리지 않는다.
        task = asyncio.create_task(run_turn(session, utterance, tc.call, llm_commit=None))
        send_filler(...)                      # 미리 합성해 둔 채움말 PCM (첫 소리 < 1.5s)
        out = await task
        if out.speech: tts(out.speech)        # 템플릿 답 → LLM 재호출 없이 바로 TTS
        else:          llm_answer(out.results)  # 템플릿이 없는 경우만 LLM

LLM 이 접수 함수를 고른 경우엔 llm_commit=(name, args) 로 넘기면 게이트(gate_commit)를 거쳐
번호가 코드 값으로 바뀐 뒤 호출된다.
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .planner import Call, gate_commit, plan_lookups, suggest_commit, topic, wants_search
from .slots import Slots, resolve
from .templates import render

CallFn = Callable[[str, dict], Awaitable[dict]]
RANK = {"ok": 0, "already_requested": 1}


@dataclass
class TurnResult:
    slots: Slots
    calls: list[tuple[Call, dict]] = field(default_factory=list)   # 실제로 보낸 호출과 결과
    speech: str | None = None          # 템플릿으로 만든 답 (None 이면 LLM 이 답해야 함)
    commit_reason: str = ""            # 접수를 안 했다면 그 이유 (need_phone, no_consent ...)
    rounds: int = 0                    # 순차 함수 왕복 수 (지연 근사)
    variant_used: str | None = None    # 들은 번호 대신 보정 후보로 찾았으면 그 번호 (끝자리로 확인 안내)
    elapsed: float = 0.0

    @property
    def last(self) -> tuple[Call, dict] | None:
        return self.calls[-1] if self.calls else None


async def _parallel(call: CallFn, calls: list[Call]) -> list[tuple[Call, dict]]:
    res = await asyncio.gather(*(call(c.name, c.arguments) for c in calls))
    return list(zip(calls, res))


def _best(pairs: list[tuple[Call, dict]], name: str) -> tuple[Call, dict] | None:
    cand = [(RANK.get(r.get("status"), 5), i, c, r) for i, (c, r) in enumerate(pairs) if c.name == name]
    if not cand:
        return None
    _, _, c, r = min(cand, key=lambda x: x[:2])
    return c, r


def _absorb(slots: Slots, best_order: tuple[Call, dict] | None):
    if best_order and best_order[1].get("status") == "ok":
        rows = (best_order[1].get("data") or {}).get("orders") or []
        if rows:
            slots.known_orders = rows


async def _variant_round(call, slots, best, out):
    """조회 함수가 모두 not_found 이고 손님이 번호를 말했을 때만. 후보 중 '딱 하나'만 맞으면 그 결과를 쓴다.
    둘 이상 맞으면 남의 기록일 수 있으니 쓰지 않는다. 함수 점수: 조회는 더 불러도 감점이 없다."""
    from .numbers_ko import stt_variants
    tried = [(n, r) for n, r in best.items() if n in ("get_order_info", "get_return_info")]
    if not tried or any(r[1].get("status") == "ok" for _, r in tried):
        return None
    heard = [(c, "phone") for c in slots.utter_phones[:2]] + [(c, "order") for c in slots.utter_orders[:2]]
    if not heard:
        return None
    name = tried[0][0]
    calls, seen = [], set()
    # 발신번호: 들은 번호가 안 맞으면 걸려 온 번호로도 확인한다 (실측에서 정답이 발신번호와 같은 경우가 있었다)
    tried_phones = {re.sub(r"\D", "", str(r[0].arguments.get("buyer_cellphone") or r[0].arguments.get("customer_phone_number") or ""))
                    for _, r in tried}
    if slots.caller_phone and slots.caller_phone not in tried_phones:
        seen.add(slots.caller_phone)
        calls.append(Call(name, {"customer_phone_number": slots.caller_phone} if name == "get_return_info"
                          else {"order_id": None, "buyer_cellphone": slots.caller_phone}, "caller-fallback"))
    for c, kind in heard:
        for v in stt_variants(c.value, kind):
            if v in seen:
                continue
            seen.add(v)
            if name == "get_return_info":
                if kind == "phone":
                    calls.append(Call(name, {"customer_phone_number": v}, "stt-variant"))
            else:
                calls.append(Call(name, {"order_id": v if kind == "order" else None,
                                         "buyer_cellphone": v if kind == "phone" else None}, "stt-variant"))
    if not calls:
        return None
    pairs = await _parallel(call, calls[:40])
    out.rounds += 1
    hits = [(c, r) for c, r in pairs if r.get("status") == "ok"]
    caller_hit = [(c, r) for c, r in hits if c.why == "caller-fallback"]
    if caller_hit:
        c, r = caller_hit[0]
        out.variant_used = slots.caller_phone
        return c, r
    if len(hits) != 1:
        if len(hits) > 1:
            out.commit_reason = "ambiguous_number"
        return None
    c, r = hits[0]
    v = c.arguments.get("buyer_cellphone") or c.arguments.get("customer_phone_number") or c.arguments.get("order_id") or ""
    out.variant_used = re.sub(r"\D", "", v)
    return c, r


async def run_turn(session: dict, utterance: str, call: CallFn,
                   llm_commit: tuple[str, dict] | None = None, use_rules: bool = True) -> TurnResult:
    t0 = time.perf_counter()
    slots = resolve(session, utterance)
    out = TurnResult(slots)
    intent = topic(session, utterance)

    # 1) 조회 — 후보를 한 번에 병렬로
    looks = plan_lookups(session, utterance, slots)
    if wants_search(session, utterance):
        looks.append(Call("search_products", {"query": re.sub(r"\s+", " ", utterance).strip()[:60]}, "product"))
    pairs = await _parallel(call, looks) if looks else []
    out.rounds += bool(looks)
    best = {n: _best(pairs, n) for n in ("get_order_info", "get_return_info", "search_products")}
    best = {k: v for k, v in best.items() if v}

    # 1-2) 들은 번호로 조회가 안 되면, '한 가지 오류'를 고친 후보로 한 번 더 병렬 조회 (음성 인식 오류 대비)
    fixed = await _variant_round(call, slots, best, out)
    if fixed:
        best[fixed[0].name] = fixed

    _absorb(slots, best.get("get_order_info"))
    out.calls.extend(best.values())
    looked_up = "get_order_info" in best and best["get_order_info"][1].get("status") == "ok"

    # 2) 접수 판단
    commit, why = None, ""
    if llm_commit:
        args, why = gate_commit(llm_commit[0], llm_commit[1], session, utterance, slots)
        commit = Call(llm_commit[0], args) if args else None
    elif use_rules:
        commit, why = suggest_commit(session, utterance, slots, after_lookup=looked_up)
    if commit is None and why == "lookup_first" and not looked_up and (slots.phone or slots.order):
        # 주문 조회 결과가 없어서 막힌 경우: 조회 → 다시 판단 (함수 왕복 1회 추가)
        c = Call("get_order_info", {"order_id": slots.order, "buyer_cellphone": slots.phone}, "chain")
        r = await call(c.name, c.arguments)
        out.rounds += 1
        out.calls.append((c, r))
        _absorb(slots, (c, r))
        if r.get("status") == "ok":
            if llm_commit:
                args, why = gate_commit(llm_commit[0], llm_commit[1], session, utterance, slots)
                commit = Call(llm_commit[0], args) if args else None
            else:
                commit, why = suggest_commit(session, utterance, slots, after_lookup=True)
    out.commit_reason = why
    # LLM 에 줄 상황 힌트 (접수 판단 결과보다 구체적인 것으로 덮어쓴다)
    distinct = {r.get("주문번호") for r in slots.known_orders if r.get("주문번호")}
    if intent in {"return_request", "cancel", "exchange"} and not slots.known_orders and not (slots.phone or slots.order):
        out.commit_reason = "need_order"
    elif intent in {"return_request", "cancel", "exchange", "order_status"} and len(distinct) > 1 and not commit \
            and not any(m.get("role") == "tool_call" and m.get("name") in ("return_order", "return_order_item", "cancel_order")
                        for m in session.get("history") or []):   # 이미 접수한 건이 있으면 그 건 이야기다
        out.commit_reason = "multi_order"
    if intent is None and not out.calls and len(re.sub(r"[\d\s\-.,]", "", utterance)) <= 2:
        out.commit_reason = "unclear"             # '1600-0000' 처럼 번호만, 주제 없음
    from .templates import unpaid_stale, PAY_Q
    if unpaid_stale(slots.known_orders, slots.now) and (PAY_Q.search(utterance) or intent in (None, "order_status")) \
            and any(c.name == "get_order_info" for c, _ in out.calls):
        out.commit_reason = "unpaid_stale"         # 며칠 전 주문이 아직 미입금 → 20분 안내 대신 입금 정보 확인
    caller_only = not (slots.utter_phones or slots.utter_orders)
    if caller_only and any(c.name in ("get_order_info", "get_return_info") and r.get("status") == "not_found"
                           for c, r in out.calls):
        out.commit_reason = "caller_not_found"     # 손님이 번호를 말한 적 없는데 '말씀하신 번호'라고 하면 안 된다
    if commit and commit.name == "request_callback" and slots.utter_phones and slots.utter_phones[0].conf < 0.9:
        # 애매하게 들린 번호로 전화 요청을 넣기 전에, 변형 후보 중 주문 기록이 있는 번호를 찾는다
        from .numbers_ko import stt_variants
        vs = [c.value for c in slots.utter_phones[:8]]
        vs += [v for v in stt_variants(slots.utter_phones[0].value, "phone") if v not in vs][:24]
        pairs = await _parallel(call, [Call("get_order_info", {"order_id": None, "buyer_cellphone": v}) for v in vs])
        out.rounds += 1
        hit = _best(pairs, "get_order_info")
        if hit and hit[1].get("status") == "ok":
            out.calls.append(hit)
            commit.arguments["phone_number"] = hit[0].arguments["buyer_cellphone"]
    if commit:
        r = await call(commit.name, commit.arguments)
        out.rounds += 1
        out.calls.append((commit, r))

    # 3) 템플릿 답
    if out.last and out.commit_reason not in ("caller_not_found", "multi_order"):
        c, r = out.last
        out.speech = render(c.name, r, session, intent, utterance, slots.known_orders) if c.name != "search_products" else None
        if out.speech and out.variant_used:
            out.speech = f"끝자리 {out.variant_used[-4:]} 번호로 확인했어요. " + out.speech
    out.elapsed = time.perf_counter() - t0
    return out
