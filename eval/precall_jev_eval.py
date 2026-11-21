"""AG-T31 말하는 중 Jev 선호출 vs 기존(침묵 200ms) 선호출 비교.

기존: 침묵 precall_silence_ms(200)에 그 순간 STT 텍스트로 LLM 선호출 → finalize(+240ms) 텍스트가 다르면 다시 건다.
      음성이 다시 나오면 취소. 확정 때 match_key(선호출 텍스트) == match_key(최종 텍스트)면 재사용.
새 방식: 말하는 동안 tick_ms마다 그 순간 STT 텍스트로 Jev에 "이후 말로 응답이 바뀔 확률"(meaning_will_change)을 묻고,
      p ≤ tau면 그 텍스트로 선호출(음성 재개로 취소하지 않는다). 확정 때 선호출 텍스트와 최종 텍스트가 다르면
      Jev에 "선호출 응답이 최종 발화에도 맞나"(same_reply)를 묻고 p ≥ sigma면 재사용, 아니면 기존 선호출로 폴백.
      정합성 요청은 finalize 텍스트가 나온 순간(없으면 확정 순간)에 보내고, 확정 뒤에 도착하면 그만큼 기다린다.

재생: eval.soniox_endpoint_eval 녹화(rows.jsonl의 Soniox 토큰 이벤트, 실제 도착 시각) + Silero VAD + 기존 결정기.
     세션 _process_frame과 같은 순서(eval.soniox_gate_eval.simulate_detector와 같은 규칙)로 확정 시각·최종 텍스트를 얻는다.
정답: 선호출 텍스트와 최종 텍스트를 실제 LLM(시스템 프롬프트·툴 그대로)에 넣어 첫 행동을 비교한다.
     둘 다 툴 호출이면 eval.score.tool_match exact(자유 텍스트 인자 제외), 둘 다 텍스트면 심사
     (runs/precall/judge_todo.jsonl로 내보내고 Claude 서브에이전트가 judge.jsonl에 {"k","ok","reason"}로 채운다)
     ("부분 발화로 만든 첫 응답을 최종 발화에 그대로 내보내도 되나"), 종류가 다르면 불일치.
     최종 텍스트는 두 번 돌려 LLM 자체 흔들림(바닥값)도 잰다.
지연 모델: L = 최종 텍스트 LLM 첫 행동 시간(툴이면 스트림 끝, 텍스트면 첫 이벤트).
     확정 뒤 남는 LLM 대기 = max(정합성 대기, L − 선행 시간), 선행 시간 = 확정 − 쓴 선호출 발사 시각.

캐시(runs/precall/): jev_cache.jsonl(Jev 본문 해시), llm_first.jsonl(문항+텍스트), judge.jsonl. 다시 돌리면 새 호출 0회.

사용:
  uv run python -m eval.precall_jev_eval --run runs/onedam/ev_audio --manual onedam_cs_testset/agent_world/manual.md \
      --data onedam_cs_testset/agent_world/mock_data.json --tools onedam_cs_testset/agent_world/tools.json \
      --out runs/precall/onedam [--no-call]
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from agent.session import match_key
from agent.turn import jev_turn as jt
from agent.turn.soniox_gate import is_speech_token
from eval.soniox_endpoint_eval import ROOT

CACHE_DIR = ROOT / "runs/precall"

# ───────────────────────── Jev 질문 ─────────────────────────
TICK_SITUATION = (
    "Korean phone customer-service call for an online health-supplement shop. The customer is STILL SPEAKING. "
    "`customer_speech_so_far` is a live speech-to-text transcript of the current turn up to now; it lags the audio "
    "by about a second, may contain recognition errors, and has unreliable punctuation. The agent wants to start "
    "preparing its reply early, before the customer finishes.")

TICK_QUESTIONS: dict[str, dict] = {
    "meaning_will_change": {
        "type": "noul",
        "instructions": (
            "Will what the customer says from here on change what the agent needs to do in reply — which lookup or "
            "action to take, with which values (order or phone number, product, quantity, date, option), or what "
            "answer to give? Judge `customer_speech_so_far` in light of `previous_conversation`."),
        "criteria": {
            "true": ("The rest of the turn will probably change or add to the needed reply: the actual request has "
                     "not been stated yet, it is only a greeting or setup, a number or value is incomplete, the "
                     "customer may correct themselves, or a second request or condition is likely to follow."),
            "false": ("The request and every value needed to act on it are already clear; what remains is probably "
                      "just a sentence ending, politeness, or repetition that would not change the agent's reply."),
        },
    },
}

SAME_SITUATION = (
    "Korean phone customer-service call for an online health-supplement shop. While the customer was still "
    "speaking, the agent started preparing a reply from `early_transcript`, a partial live transcript. "
    "`final_transcript` is what the customer said in the whole turn. Both are speech-to-text and may differ in "
    "punctuation, spacing, and how numbers are written.")

SAME_QUESTIONS: dict[str, dict] = {
    "same_reply": {
        "type": "noul",
        "instructions": (
            "Is the reply prepared from `early_transcript` still correct for `final_transcript` — the same lookup or "
            "action with the same values, or the same answer — so the agent can use it without starting over? "
            "Consider `previous_conversation`."),
        "criteria": {
            "true": ("Yes: the final transcript only adds a sentence ending, politeness, repetition, or "
                     "transcription variants; the needed action and its values or the answer are unchanged."),
            "false": ("No: the final transcript adds or changes the request, a value (number, product, date, "
                      "quantity, option), a negation or condition, a self-correction, or a second request."),
        },
    },
}


def _hist(history: list[dict] | None) -> list[dict]:
    return [{"role": m.get("role"), "text": m.get("text", m.get("content"))} for m in (history or [])]


def tick_body(text: str, history: list[dict] | None, caller_known: bool) -> dict:
    return {"model": jt.MODEL, "questions": TICK_QUESTIONS,
            "state": {"situation": TICK_SITUATION, "previous_conversation": _hist(history),
                      "caller_already_identified": bool(caller_known),
                      "customer_speech_so_far": (text or "").strip()}}


FIRST_SITUATION = (
    "Korean phone customer-service call for an online health-supplement shop. While the customer was still "
    "speaking, the agent took its FIRST STEP from `early_transcript`, a partial live transcript. "
    "`prepared_first_step` is that step: either lookups (tool calls with their arguments) or a spoken reply. "
    "`final_transcript` is what the customer said in the whole turn. If the first step is a lookup, the agent "
    "will write its answer only after the lookup returns, reading `final_transcript`, and can make further lookups "
    "then. Transcripts are speech-to-text and may differ in punctuation, spacing, and how numbers are written.")

FIRST_QUESTIONS: dict[str, dict] = {
    "same_first_step": {
        "type": "noul",
        "instructions": (
            "Is `prepared_first_step` still a correct first step for `final_transcript`, so the agent can keep it "
            "instead of starting over? For lookups: are they the lookups the agent should do first for the final "
            "transcript, with the right values? Extra questions or details in the final transcript that can be "
            "handled after these lookups return do NOT make them wrong. For a spoken reply: it will be said as-is, "
            "so it must fit the whole final transcript. Consider `previous_conversation`."),
        "criteria": {
            "true": ("Keep it: the lookups (tool and argument values) are right for the final transcript, or the "
                     "spoken reply fully fits it."),
            "false": ("Start over: a lookup has a wrong or incomplete value (number, product, option), the wrong "
                      "tool, the final transcript negates or corrects what the step relied on, or a spoken reply "
                      "misses or contradicts what the customer finally asked."),
        },
    },
}


def first_step(action: dict | None) -> Any:
    """LLM 첫 행동 캐시 행 → Jev 상태용 요약."""
    if not action:
        return None
    if action["kind"] == "tool":
        return {"lookups": [{"tool": c["name"], "arguments": c["arguments"]} for c in action["calls"]]}
    return {"spoken_reply": action.get("reply", "")}


def first_body(early: str, final: str, action: dict | None, history: list[dict] | None, caller_known: bool,
               catalog: list[dict] | None = None) -> dict:
    """catalog: [{"tool", "description"}]를 주면 상태에 available_lookups로 넣는다(툴 선택이 맞는지 판단용)."""
    state = {"situation": FIRST_SITUATION, "previous_conversation": _hist(history),
             "caller_already_identified": bool(caller_known),
             "early_transcript": (early or "").strip(), "prepared_first_step": first_step(action),
             "final_transcript": (final or "").strip()}
    if catalog:
        state["available_lookups"] = catalog
    return {"model": jt.MODEL, "questions": FIRST_QUESTIONS, "state": state}


def same_body(early: str, final: str, history: list[dict] | None, caller_known: bool) -> dict:
    return {"model": jt.MODEL, "questions": SAME_QUESTIONS,
            "state": {"situation": SAME_SITUATION, "previous_conversation": _hist(history),
                      "caller_already_identified": bool(caller_known),
                      "early_transcript": (early or "").strip(), "final_transcript": (final or "").strip()}}


def body_key(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:20]


class JevStore:
    """Jev 응답 캐시: {키: {"answers": {q: p}, "ms": 왕복}}. ask(body) → (dict, ms)를 주입할 수 있다."""

    def __init__(self, path: Path | None, ask: Callable[[dict], tuple[dict, float]] | None = None):
        self.path, self._ask = path, ask
        self.d: dict[str, dict] = {}
        self.calls = 0
        if path is not None and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.d[r["k"]] = r

    def get(self, body: dict, q: str) -> tuple[float | None, float]:
        r = self.d.get(body_key(body))
        if not r:
            return None, 0.0
        v = (r.get("answers") or {}).get(q)
        return (None if v is None else float(v)), float(r.get("ms") or 0.0)

    def fill(self, bodies: list[dict], workers: int = 6) -> None:
        todo = {body_key(b): b for b in bodies if body_key(b) not in self.d}
        if not todo or self._ask is None:
            return

        def one(item):
            k, b = item
            resp, ms = self._ask(b)
            ans = {q: (resp.get("answers") or {}).get(q) for q in b["questions"]}
            ans = {q: (v.get("noul") if isinstance(v, dict) else v) for q, v in ans.items()}
            return {"k": k, "model": b["model"], "answers": ans, "ms": round(ms, 1), "usage": resp.get("usage")}

        f = self.path.open("a", encoding="utf-8") if self.path is not None else None
        try:
            with ThreadPoolExecutor(workers) as ex:
                for r in ex.map(one, todo.items()):
                    self.d[r["k"]] = r
                    self.calls += 1
                    if f is not None:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                        f.flush()
        finally:
            if f is not None:
                f.close()


# ───────────────────────── 재생 ─────────────────────────
@dataclass
class Precall:
    t: float          # 발사 시각(초, 오디오 기준)
    text: str
    src: str          # partial | finalize | speech


@dataclass
class Replay:
    commit_t: float | None
    commit_text: str
    reason: str
    survivor: Precall | None                 # 확정 순간 살아 있는 기존(침묵) 선호출
    silence_fires: int                       # 기존 방식 선호출 발사 수(교체 포함)
    final_t: float | None                    # 확정 텍스트(finalize 결과)를 얻은 시각
    ticks: list[tuple[float, str]] = field(default_factory=list)   # 말하는 중 (시각, 텍스트)


def replay(events: list[dict], speech: list[bool], until_s: float, detector: Any, expect_short: bool = False,
           *, tick_ms: int = 400, finalize_ms: float = 240.0, precall_ms: int = 200,
           min_speech_ms: int = 60) -> Replay:
    """세션 _process_frame 순서로 Soniox 이벤트·VAD를 재생한다(simulate_detector와 같은 규칙) + 선호출·tick 기록.
    tick: 발화가 시작된 뒤(speech_ms ≥ min) 침묵이 precall_ms 미만인 동안 tick_ms마다 그 순간 텍스트(확정+미확정)."""
    from agent.core.types import TurnSignals
    finals = [(x.get("start_ms", 0), x["text"]) for ev in events for x in ev.get("toks", [])
              if x.get("is_final") and is_speech_token(x)]
    final, nonfinal, ei = "", "", 0
    speech_ms = sil = 0
    fin_req: float | None = None
    early: str | None = None
    cur: Precall | None = None
    fires = 0
    ticks: list[tuple[float, str]] = []
    next_tick: float | None = None

    def live() -> str:
        return (final + nonfinal).strip()

    def fire(t: float, text: str, src: str) -> None:
        nonlocal cur, fires
        if not text:
            return
        if cur is not None and match_key(cur.text) == match_key(text):
            return
        cur = Precall(t, text, src)
        fires += 1

    def tick(t: float) -> None:
        nonlocal next_tick
        if speech_ms < min_speech_ms:
            return
        if next_tick is None:
            next_tick = t
        if t + 1e-9 >= next_tick:
            ticks.append((round(t, 3), live()))
            next_tick = t + tick_ms / 1000

    for f in range(int(round(until_s / 0.02))):
        t = (f + 1) * 0.02
        while ei < len(events) and events[ei]["t"] <= t:
            toks = events[ei]["toks"]
            final += "".join(x["text"] for x in toks if x.get("is_final") and is_speech_token(x))
            nonfinal = "".join(x["text"] for x in toks if not x.get("is_final") and is_speech_token(x))
            ei += 1
        if f < len(speech) and speech[f]:
            if sil > 0 and fin_req is not None:
                cur = None                                  # 확정 전 음성 재개 → 선호출 취소
            if sil > 0:
                fin_req, early = None, None
            speech_ms += 20
            sil = 0
            tick(t)
            continue
        if speech_ms < min_speech_ms:
            if speech_ms and sil + 20 >= precall_ms:
                speech_ms = sil = 0
            elif speech_ms:
                sil += 20
            continue
        sil += 20
        if sil < precall_ms:
            tick(t)
        if fin_req is None and sil >= precall_ms:
            fin_req = t
            fire(t, live(), "partial")
        if fin_req is not None and early is None and t >= fin_req + finalize_ms / 1000:
            early = "".join(x for st, x in finals if st < fin_req * 1000).strip()
            fire(t, early, "finalize")
        text = early or live()
        dec = detector.decide(TurnSignals(silence_ms=sil, speech_ms=speech_ms, text=text, smart_turn_p=None,
                                          expect_short=expect_short))
        if dec.end:
            ft = (fin_req + finalize_ms / 1000) if (early and fin_req is not None) else None
            return Replay(round(t, 3), text, dec.reason, cur, fires, ft, ticks)
    return Replay(None, live(), "no_end", cur, fires, None, ticks)


# ───────────────────────── 정책 ─────────────────────────
@dataclass(frozen=True)
class SpeechPolicy:
    tau: float = 0.3                 # meaning_will_change ≤ tau면 선호출
    sigma: float = 0.5               # same_reply ≥ sigma면 재사용
    mode: str = "first"              # first: 한 번만 / latest: 통과할 때마다 새 텍스트로 교체
    # jev(same_reply) | jev_first(same_first_step, 첫 행동 포함) | jev_first_tool(첫 행동이 텍스트면 버림)
    # | jev_first_cat(툴만 + 상태에 툴 목록) | none(검사 없이 재사용) | oracle(정답으로 판정) | string(문자열 일치만)
    check: str = "jev"
    min_chars: int = 4               # 정규화 글자 수가 이보다 짧은 텍스트는 묻지 않는다

    def label(self) -> str:
        c = {"jev": f"σ={self.sigma}", "jev_first": f"첫 조회 σ={self.sigma}",
             "jev_first_tool": f"첫 조회·툴만 σ={self.sigma}",
             "jev_first_cat": f"첫 조회·툴만·툴 목록 σ={self.sigma}", "none": "검사 없음", "oracle": "정답 검사",
             "string": "문자열 일치만"}[self.check]
        return f"τ={self.tau} {self.mode} · {c}"


@dataclass
class SpeechFire:
    fire: Precall | None
    jev_calls: int
    fires: int


def speech_fire(ticks: list[tuple[float, str]], commit_t: float, pchange: Callable[[str], tuple[float | None, float]],
                pol: SpeechPolicy) -> SpeechFire:
    """tick마다 Jev(텍스트가 바뀐 tick만). 응답 도착(tick + 왕복)이 확정 전이어야 쓴다."""
    last_key, cur, calls, fires = "", None, 0, 0
    for t, text in ticks:
        k = match_key(text)
        if len(k) < pol.min_chars or k == last_key:
            continue
        last_key = k
        p, ms = pchange(text)
        calls += 1
        arrive = t + ms / 1000
        if p is None or arrive >= commit_t or p > pol.tau:
            continue
        if cur is not None and match_key(cur.text) == k:
            continue
        cur = Precall(round(arrive, 3), text, "speech")
        fires += 1
        if pol.mode == "first":
            break
    return SpeechFire(cur, calls, fires)


@dataclass
class TurnOutcome:
    used: str                  # speech | silence | none
    head_ms: float             # 쓴 선호출의 선행 시간(확정 − 발사)
    check_wait_ms: float       # 정합성 판정 때문에 확정 뒤 기다린 시간
    llm_wait_ms: float | None  # 확정 뒤 남는 LLM 첫 행동 대기
    wrong: bool                # 재사용했는데 정답상 다른 응답
    missed: bool               # 정답상 같은데 버림
    jev_calls: int
    llm_calls: int
    wait_pen_ms: float | None = None   # 잘못 재사용 벌점(한 회차 더)을 넣은 대기. None이면 llm_wait_ms와 같다


def existing_outcome(rp: Replay, L: float | None) -> TurnOutcome:
    s = rp.survivor
    ok = s is not None and match_key(s.text) == match_key(rp.commit_text)
    head = (rp.commit_t - s.t) * 1000 if ok else 0.0
    llm_calls = rp.silence_fires + (0 if ok else 1)
    wait = None if L is None else max(0.0, L - head)
    return TurnOutcome("silence" if ok else "none", head, 0.0, wait, False, False, 0, llm_calls)


def speech_outcome(rp: Replay, sf: SpeechFire, L: float | None, pol: SpeechPolicy,
                   psame: Callable[[str, str], tuple[float | None, float]],
                   oracle: Callable[[str, str], bool | None],
                   early_kind: Callable[[str], str | None] | None = None) -> TurnOutcome:
    """psame: 정책의 정합성 질문(check에 맞는 것)을 넘긴다. early_kind: 선호출 텍스트 → 첫 행동 종류(tool|text)."""
    base = existing_outcome(rp, L)
    llm_calls = base.llm_calls + sf.fires
    if sf.fire is None:
        return TurnOutcome(base.used, base.head_ms, 0.0, base.llm_wait_ms, False, False, sf.jev_calls, llm_calls)
    early, final = sf.fire.text, rp.commit_text
    truth = True if match_key(early) == match_key(final) else oracle(early, final)
    check_wait, jev_calls = 0.0, sf.jev_calls
    if match_key(early) == match_key(final):
        accept = True
    elif pol.check == "none":
        accept = True
    elif pol.check == "string":
        accept = False
    elif pol.check == "oracle":
        accept = bool(truth)
    elif pol.check in ("jev_first_tool", "jev_first_cat") and early_kind is not None and early_kind(early) != "tool":
        accept = False                                  # 첫 행동이 텍스트(그대로 송출)면 재사용하지 않는다
    else:
        p, ms = psame(early, final)
        jev_calls += 1
        t_req = rp.final_t if (rp.final_t is not None and rp.final_t <= rp.commit_t) else rp.commit_t
        check_wait = max(0.0, (t_req + ms / 1000 - rp.commit_t) * 1000)
        accept = p is not None and p >= pol.sigma
    if accept:
        head = (rp.commit_t - sf.fire.t) * 1000
        wait = None if L is None else max(check_wait, L - head)
        o = TurnOutcome("speech", head, check_wait, wait, truth is False, False, jev_calls, llm_calls)
        if o.wrong and wait is not None:
            o.wait_pen_ms = wait + L                    # 틀린 첫 행동 → 최종 텍스트로 한 회차 더(≈L)
        return o
    wait = None if base.llm_wait_ms is None else max(check_wait, base.llm_wait_ms)
    return TurnOutcome(base.used, base.head_ms, check_wait, wait, False, truth is True, jev_calls, llm_calls)


def summarize(outs: list[TurnOutcome]) -> dict:
    n = len(outs)
    waits = [o.llm_wait_ms for o in outs if o.llm_wait_ms is not None]
    heads = [o.head_ms for o in outs if o.used != "none"]
    sp = [o for o in outs if o.used == "speech"]
    pct = lambda xs, q: float(np.percentile(xs, q)) if xs else None  # noqa: E731
    return {
        "n": n,
        "reuse": sum(o.used != "none" for o in outs) / n if n else 0.0,
        "speech_reuse": len(sp) / n if n else 0.0,
        "head_p50": pct(heads, 50),
        "wait_mean": float(np.mean(waits)) if waits else None,
        "wait_p50": pct(waits, 50), "wait_p95": pct(waits, 95),
        "wait_pen_mean": float(np.mean([o.wait_pen_ms if o.wait_pen_ms is not None else o.llm_wait_ms
                                        for o in outs if o.llm_wait_ms is not None])) if waits else None,
        "check_wait_mean": float(np.mean([o.check_wait_ms for o in outs])) if outs else 0.0,
        "wrong": sum(o.wrong for o in outs), "missed": sum(o.missed for o in outs),
        "jev_per_turn": float(np.mean([o.jev_calls for o in outs])) if outs else 0.0,
        "llm_per_turn": float(np.mean([o.llm_calls for o in outs])) if outs else 0.0,
    }


# ───────────────────────── 정답(LLM 첫 행동) ─────────────────────────
class LLMFirst:
    """문항+텍스트 → 실제 LLM 첫 응답(툴 호출들 또는 텍스트)과 시간. runs/precall/llm_first.jsonl 캐시."""

    def __init__(self, path: Path | None, world: dict | None = None, call: bool = True):
        self.path, self.world, self.call = path, world, call
        self.d: dict[str, dict] = {}
        self.calls = 0
        if path is not None and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.d[r["k"]] = r

    @staticmethod
    def key(case_id: str, text: str, rep: int = 0) -> str:
        return hashlib.sha256(f"{case_id}\x00{text.strip()}\x00{rep}".encode()).hexdigest()[:20]

    def get(self, case_id: str, text: str, rep: int = 0) -> dict | None:
        return self.d.get(self.key(case_id, text, rep))

    async def _one(self, case: dict, text: str, rep: int) -> dict:
        from agent.core.types import Message
        from agent.normalize import normalize_args
        from agent.prompts import build_system
        from eval.llm_eval import DEFAULT_OPENER, case_info
        w = self.world
        info = case_info(case)
        system = build_system(info, DEFAULT_OPENER, manual=w["manual"])
        llm = w["llm_factory"]()
        calls, reply = [], ""
        t0 = time.perf_counter()
        first = None
        async for ev in llm.stream(system, [Message("user", text)], w["tools"]):
            if first is None:
                first = time.perf_counter()
            if ev.type == "tool_call" and ev.tool_call is not None:
                tc = ev.tool_call
                try:
                    args = normalize_args(tc.name, dict(tc.arguments or {}), w["schemas"].get(tc.name, {}), info.now)
                except Exception:  # noqa: BLE001
                    args = dict(tc.arguments or {})
                calls.append({"name": tc.name, "arguments": args})
            elif ev.type == "text":
                reply += ev.text
        done = time.perf_counter()
        return {"k": self.key(case["case_id"], text, rep), "case_id": case["case_id"], "text": text.strip(),
                "rep": rep, "kind": "tool" if calls else "text", "calls": calls, "reply": reply.strip(),
                "first_ms": round(((first or done) - t0) * 1000, 1), "done_ms": round((done - t0) * 1000, 1),
                "model": getattr(llm, "last_model", None) or getattr(llm, "model", None)}

    async def fill(self, reqs: list[tuple[dict, str, int]], concurrency: int = 4, retries: int = 3) -> None:
        todo = {}
        for c, t, r in reqs:
            k = self.key(c["case_id"], t, r)
            if k not in self.d and t.strip():
                todo[k] = (c, t, r)
        if not todo or not self.call:
            return
        sem = asyncio.Semaphore(concurrency)
        f = self.path.open("a", encoding="utf-8") if self.path is not None else None

        async def run(c, t, r):
            async with sem:
                for a in range(retries + 1):
                    try:
                        row = await self._one(c, t, r)
                        break
                    except Exception as e:  # noqa: BLE001
                        if a >= retries:
                            print(f"llm fail {c['case_id']}: {e!r}")
                            return
                        await asyncio.sleep(2.0 * 2 ** a)
                self.d[row["k"]] = row
                self.calls += 1
                if f is not None:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
        try:
            await asyncio.gather(*(run(*v) for v in todo.values()))
        finally:
            if f is not None:
                f.close()


def latency_of(row: dict | None) -> float | None:
    """확정 뒤 응답을 시작하려면 필요한 LLM 시간: 툴이면 스트림 끝(툴 실행 시작), 텍스트면 첫 이벤트."""
    if not row:
        return None
    return float(row["done_ms"] if row["kind"] == "tool" else row["first_ms"])


JUDGE_PROMPT = """한국어 전화 상담(건강기능식품 온라인몰) 평가입니다.
상담 에이전트가 고객이 말하는 도중의 부분 받아쓰기만 듣고 첫 응답을 미리 만들었습니다.
고객의 최종 발화 전체에 대해 그 응답을 그대로 내보내도 되는지 판단하세요.

이전 대화: {history}
부분 받아쓰기(미리 만든 기준): {early}
최종 발화: {final}
부분 받아쓰기로 만든 응답: {reply_early}
최종 발화로 만든 응답(참고): {reply_final}

판단 기준: 최종 발화가 요구한 내용에 맞고, 빠뜨린 요청·틀린 값·이미 말한 정보를 다시 묻는 문제가 없으면 ok=true.
표현이 달라도 내용이 맞으면 ok=true. JSON으로만 답하세요: {{"ok": true|false, "reason": "한 문장"}}"""


class Judge:
    def __init__(self, path: Path | None, fn: Callable[[str], dict] | None = None):
        self.path, self.fn = path, fn
        self.d: dict[str, dict] = {}
        self.calls = 0
        if path is not None and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.d[r["k"]] = r

    @staticmethod
    def prompt(history: list, early: str, final: str, ra: str, rb: str) -> str:
        return JUDGE_PROMPT.format(history=json.dumps(history or [], ensure_ascii=False), early=early, final=final,
                                   reply_early=ra or "(없음)", reply_final=rb or "(없음)")

    def get(self, prompt: str) -> bool | None:
        r = self.d.get(body_key({"p": prompt}))
        return None if r is None else bool(r["ok"])

    def fill(self, prompts: list[str], workers: int = 4) -> None:
        todo = {body_key({"p": p}): p for p in prompts if body_key({"p": p}) not in self.d}
        if not todo or self.fn is None:
            return

        def one(item):
            k, p = item
            v = self.fn(p)
            return {"k": k, "ok": bool(v.get("ok")), "reason": v.get("reason", "")}

        f = self.path.open("a", encoding="utf-8") if self.path is not None else None
        try:
            with ThreadPoolExecutor(workers) as ex:
                for r in ex.map(one, todo.items()):
                    self.d[r["k"]] = r
                    self.calls += 1
                    if f is not None:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        finally:
            if f is not None:
                f.close()


def equivalent(early: dict | None, final: dict | None, judged: bool | None) -> bool | None:
    """부분 텍스트 첫 행동이 최종 텍스트 첫 행동과 같은가. 둘 다 툴: tool_match exact, 둘 다 텍스트: 심사."""
    from eval.score import tool_match
    if early is None or final is None:
        return None
    if early["kind"] != final["kind"]:
        return False
    if early["kind"] == "tool":
        return bool(tool_match(final["calls"], early["calls"])["exact"])
    return judged


# ───────────────────────── 실행 ─────────────────────────
TAUS = (0.1, 0.2, 0.3, 0.4, 0.5)
SIGMAS = (0.5, 0.7, 0.85)


def _load_items(run: Path) -> list[tuple[dict, dict]]:
    from eval.soniox_endpoint_eval import load_dataset
    from eval.soniox_gate_eval import attach_vad, load_rows
    cfg = json.loads((run / "summary.json").read_text(encoding="utf-8"))["config"]
    pairs = load_rows(run, load_dataset(cfg["dataset"], cfg.get("cases", ""), cfg.get("audio", "")))
    attach_vad(cfg["dataset"], pairs)
    return pairs


def _world(a) -> dict:
    from agent.prompts import load_manual
    from eval.llm_eval import _default_llm_factory, shop_tools
    from mock_api.legacy_shop import load_shop
    shop = load_shop(data_path=a.data, tools_path=a.tools)
    tools = shop_tools(shop)
    return {"manual": load_manual(Path(a.manual)), "tools": tools, "llm_factory": _default_llm_factory(None, False),
            "schemas": {t["function"]["name"]: t["function"].get("parameters", {}) for t in tools}}


def _pct(xs, q):
    return f"{np.percentile(xs, q):.0f}" if xs else "-"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/onedam/ev_audio")
    ap.add_argument("--cases", default="onedam_cs_testset/agent_world/cases.jsonl")
    ap.add_argument("--manual", default="onedam_cs_testset/agent_world/manual.md")
    ap.add_argument("--data", default="onedam_cs_testset/agent_world/mock_data.json")
    ap.add_argument("--tools", default="onedam_cs_testset/agent_world/tools.json")
    ap.add_argument("--out", default="runs/precall/onedam")
    ap.add_argument("--tick-ms", type=int, default=400)
    ap.add_argument("--no-call", action="store_true", help="캐시만 쓴다(새 API 호출 금지)")
    ap.add_argument("--concurrency", type=int, default=4)
    a = ap.parse_args(argv)
    from eval.jev_turn_eval import load_dotenv
    load_dotenv()
    os.environ.setdefault("DIRECTON_MOCK_DATA", str(Path(a.data).resolve()))
    from agent.turn.detector import make_detector
    from eval.turn_sim import _expect_short

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = {c["case_id"]: c for c in (json.loads(x) for x in Path(a.cases).read_text(encoding="utf-8").splitlines()
                                        if x.strip())}
    pairs = _load_items(Path(a.run))

    # 1) 재생
    reps: dict[str, Replay] = {}
    for it, row in pairs:
        rp = replay(row["events"], it["speech"], it["end_t"] + 2.5, make_detector(),
                    _expect_short(it.get("history") or []), tick_ms=a.tick_ms)
        if rp.commit_t is not None and rp.commit_text:
            reps[it["uid"]] = rp
    ctx = {it["uid"]: (it.get("history") or [], bool(it.get("caller_known"))) for it, _ in pairs}
    print(f"재생 {len(pairs)}문항, 확정 {len(reps)}, tick 평균 {np.mean([len(r.ticks) for r in reps.values()]):.1f}")

    # 2) tick Jev
    ask = None
    if not a.no_call:
        from eval.jev_turn_eval import Jev
        ask = Jev().ask
    jev = JevStore(CACHE_DIR / "jev_cache.jsonl", ask)
    tb = []
    for uid, rp in reps.items():
        h, k = ctx[uid]
        seen = set()
        for _, text in rp.ticks:
            mk = match_key(text)
            if len(mk) >= SpeechPolicy.min_chars and mk not in seen:
                seen.add(mk)
                tb.append(tick_body(text, h, k))
    jev.fill(tb)
    print(f"tick Jev 요청 {len(tb)}, 새 호출 {jev.calls}")

    def pchange_for(uid):
        h, k = ctx[uid]
        return lambda text: jev.get(tick_body(text, h, k), "meaning_will_change")

    def psame_for(uid):
        h, k = ctx[uid]
        return lambda e, f: jev.get(same_body(e, f, h, k), "same_reply")

    # 3) 정책별 발사 → 정합성 Jev·LLM 정답 대상
    pols = [SpeechPolicy(tau, s, m, "jev") for tau in TAUS for s in SIGMAS for m in ("first", "latest")]
    fires: dict[tuple[str, float, str], SpeechFire] = {}
    for uid, rp in reps.items():
        for tau in TAUS:
            for m in ("first", "latest"):
                fires[(uid, tau, m)] = speech_fire(rp.ticks, rp.commit_t, pchange_for(uid), SpeechPolicy(tau, mode=m))
    pair_set = {(uid, sf.fire.text) for (uid, _, _), sf in fires.items()
                if sf.fire is not None and match_key(sf.fire.text) != match_key(reps[uid].commit_text)}
    sb = [same_body(e, reps[uid].commit_text, *ctx[uid]) for uid, e in pair_set]
    c0 = jev.calls
    jev.fill(sb)
    print(f"정합성 Jev 쌍 {len(sb)}, 새 호출 {jev.calls - c0}")

    llmf = LLMFirst(CACHE_DIR / "llm_first.jsonl", _world(a) if not a.no_call else None, call=not a.no_call)
    reqs = [(cases[uid], rp.commit_text, r) for uid, rp in reps.items() for r in (0, 1)]
    reqs += [(cases[uid], e, 0) for uid, e in pair_set]
    asyncio.run(llmf.fill(reqs, a.concurrency))
    print(f"LLM 첫 행동 요청 {len(reqs)}, 새 호출 {llmf.calls}")

    def fbody(uid, e, f):
        return first_body(e, f, llmf.get(uid, e), *ctx[uid])

    fb = [fbody(uid, e, reps[uid].commit_text) for uid, e in pair_set if llmf.get(uid, e)]
    c0 = jev.calls
    jev.fill(fb)
    print(f"첫 조회 정합성 Jev 쌍 {len(fb)}, 새 호출 {jev.calls - c0}")

    tj = json.loads(Path(a.tools).read_text(encoding="utf-8-sig"))
    tj = tj.get("tools", tj) if isinstance(tj, dict) else tj
    catalog = [{"tool": (x.get("function") or x)["name"], "description": (x.get("function") or x).get("description", "")}
               for x in tj]

    def cbody(uid, e, f):
        return first_body(e, f, llmf.get(uid, e), *ctx[uid], catalog=catalog)

    cb = [cbody(uid, e, reps[uid].commit_text) for uid, e in pair_set
          if (llmf.get(uid, e) or {}).get("kind") == "tool"]
    c0 = jev.calls
    jev.fill(cb)
    print(f"첫 조회+툴 목록 정합성 Jev 쌍 {len(cb)}, 새 호출 {jev.calls - c0}")

    def pfirst_for(uid):
        return lambda e, f: jev.get(fbody(uid, e, f), "same_first_step")

    def pcat_for(uid):
        return lambda e, f: jev.get(cbody(uid, e, f), "same_first_step")

    def kind_for(uid):
        return lambda e: (llmf.get(uid, e) or {}).get("kind")

    judge = Judge(CACHE_DIR / "judge.jsonl", None)     # 심사는 바깥(서브에이전트)이 judge_todo.jsonl을 보고 채운다

    def jprompt(uid, early_text, er, fr):
        return Judge.prompt(ctx[uid][0], early_text, reps[uid].commit_text, er["reply"], fr["reply"])

    prompts = []
    for uid, e in pair_set:
        er, fr = llmf.get(uid, e), llmf.get(uid, reps[uid].commit_text)
        if er and fr and er["kind"] == fr["kind"] == "text":
            prompts.append(jprompt(uid, e, er, fr))
    for uid, rp in reps.items():                                  # 바닥값: 최종 텍스트 두 번
        f0, f1 = llmf.get(uid, rp.commit_text, 0), llmf.get(uid, rp.commit_text, 1)
        if f0 and f1 and f0["kind"] == f1["kind"] == "text":
            prompts.append(Judge.prompt(ctx[uid][0], rp.commit_text, rp.commit_text, f1["reply"], f0["reply"]))
    todo = [{"k": body_key({"p": p}), "prompt": p} for p in dict.fromkeys(prompts)
            if body_key({"p": p}) not in judge.d]
    with (CACHE_DIR / "judge_todo.jsonl").open("w", encoding="utf-8") as fh:
        for r in todo:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"심사 대상 {len(prompts)}, 미심사 {len(todo)} → {CACHE_DIR / 'judge_todo.jsonl'}")

    def oracle_for(uid):
        def f(early, final):
            er, fr = llmf.get(uid, early), llmf.get(uid, final)
            j = None
            if er and fr and er["kind"] == fr["kind"] == "text":
                j = judge.get(jprompt(uid, early, er, fr))
            return equivalent(er, fr, j)
        return f

    # 4) 집계
    Ls = {uid: latency_of(llmf.get(uid, rp.commit_text)) for uid, rp in reps.items()}
    results: dict[str, dict] = {}
    detail: dict[str, list] = {}
    base_outs = [existing_outcome(rp, Ls[uid]) for uid, rp in reps.items()]
    results["기존(침묵 200ms 선호출, 문자열 일치)"] = summarize(base_outs)
    allpols = pols + [SpeechPolicy(tau, s, m, c) for c in ("jev_first", "jev_first_tool", "jev_first_cat")
                      for tau in TAUS
                      for s in SIGMAS for m in ("first", "latest")]
    allpols += [SpeechPolicy(tau, 0.5, m, c) for tau in TAUS for m in ("first", "latest")
                for c in ("none", "oracle", "string")]
    for pol in allpols:
        outs = []
        for uid, rp in reps.items():
            ps = (pcat_for(uid) if pol.check == "jev_first_cat" else
                  pfirst_for(uid) if pol.check.startswith("jev_first") else psame_for(uid))
            o = speech_outcome(rp, fires[(uid, pol.tau, pol.mode)], Ls[uid], pol, ps, oracle_for(uid), kind_for(uid))
            outs.append(o)
            detail.setdefault(pol.label(), []).append({"uid": uid, **o.__dict__})
        results[pol.label()] = summarize(outs)

    # 바닥값·판정 품질
    floor = []
    for uid, rp in reps.items():
        f0, f1 = llmf.get(uid, rp.commit_text, 0), llmf.get(uid, rp.commit_text, 1)
        j = judge.get(Judge.prompt(ctx[uid][0], rp.commit_text, rp.commit_text, f1["reply"], f0["reply"])) \
            if f0 and f1 and f0["kind"] == f1["kind"] == "text" else None
        e = equivalent(f1, f0, j)
        if e is not None:
            floor.append(e)
    quality = []
    for uid, e in sorted(pair_set):
        truth = oracle_for(uid)(e, reps[uid].commit_text)
        p, ms = psame_for(uid)(e, reps[uid].commit_text)
        pf, msf = pfirst_for(uid)(e, reps[uid].commit_text)
        pc, _ = pcat_for(uid)(e, reps[uid].commit_text)
        quality.append({"uid": uid, "early": e, "final": reps[uid].commit_text, "truth": truth, "p_same": p,
                        "ms": ms, "p_first": pf, "ms_first": msf, "p_cat": pc, "kind": kind_for(uid)(e)})

    timing = []
    for uid, rp in reps.items():
        it = next(i for i, _ in pairs if i["uid"] == uid)
        sf = fires[(uid, 0.3, "first")]
        timing.append({"uid": uid, "end_t": it["end_t"], "commit_t": rp.commit_t, "final_t": rp.final_t,
                       "survivor": rp.survivor.__dict__ if rp.survivor else None, "commit_text": rp.commit_text,
                       "speech_fire": sf.fire.__dict__ if sf.fire else None, "L": Ls[uid],
                       "ticks": [(t, s, pchange_for(uid)(s)[0]) for t, s in rp.ticks]})

    rep_json = {"config": vars(a), "n": len(reps), "results": results,
                "llm_floor_agree": (sum(floor) / len(floor)) if floor else None, "llm_floor_n": len(floor),
                "check_quality": quality, "timing": timing,
                "api_calls": {"jev": jev.calls, "llm": llmf.calls, "judge": judge.calls}}
    (out / "report.json").write_text(json.dumps(rep_json, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "detail.json").write_text(json.dumps(detail, ensure_ascii=False), encoding="utf-8")

    lines = ["| 방식 | 재사용 | 말하는 중 재사용 | 선행 p50 ms | 확정 뒤 LLM 대기 평균 / p50 / p95 ms | 벌점 포함 평균 | 판정 대기 평균 | 잘못 재사용 | 놓친 재사용 | Jev/턴 | LLM/턴 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: "-" if v is None else f"{v:.0f}"  # noqa: E731
    for lbl, s in results.items():
        lines.append(f"| {lbl} | {s['reuse']:.0%} | {s['speech_reuse']:.0%} | {f(s['head_p50'])} | "
                     f"{f(s['wait_mean'])} / {f(s['wait_p50'])} / {f(s['wait_p95'])} | {f(s['wait_pen_mean'])} | {s['check_wait_mean']:.0f} | "
                     f"{s['wrong']} | {s['missed']} | {s['jev_per_turn']:.1f} | {s['llm_per_turn']:.2f} |")
    from eval.jev_turn_eval import auc
    for name, col, tool_only in (("same_reply", "p_same", False), ("same_first_step", "p_first", False),
                                 ("same_reply · 툴 쌍", "p_same", True), ("same_first_step · 툴 쌍", "p_first", True),
                                 ("same_first_step+툴 목록 · 툴 쌍", "p_cat", True)):
        lab = [q for q in quality if q["truth"] is not None and q[col] is not None
               and (not tool_only or q["kind"] == "tool")]
        if lab:
            lines.append(f"정합성 Jev {name} AUC {auc([q[col] for q in lab], [int(q['truth']) for q in lab])} "
                         f"(쌍 {len(lab)}, 정답 같음 {sum(q['truth'] for q in lab)})")
    lines.append(f"LLM 바닥값(최종 텍스트 2회 첫 행동 일치) {rep_json['llm_floor_agree']} (n={len(floor)})")
    Lv = [v for v in Ls.values() if v is not None]
    lines.append(f"L(최종 텍스트 LLM 첫 행동) p50 {_pct(Lv, 50)} / p95 {_pct(Lv, 95)} ms")
    lines.append(f"API 새 호출: {rep_json['api_calls']}")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
