"""AG-T24 Jev(TypeSafe System One) 턴 종료 판정. 근거·지표: docs/decisions/jev_turn.md

침묵 200ms(LLM 선호출과 같은 시점)에 그 순간의 STT 텍스트 + 대화 맥락으로 Noul 세 개를 한 요청에 묻는다.
  finished       고객이 이번 턴 말을 마쳤나(높을수록 끝). 결정에 쓰는 값
  cut_off        텍스트가 문장·숫자 중간에서 끊겼나(보조, 로그·분석용)
  more_expected  곧 더 말할 것 같은가(보조)
응답(SDK 실측 p50 176ms, p95 247ms → 침묵 ~450ms에 도착)이 오면 표 결정기(TableTurnDetector)의 대기 시간을 조정한다.
  finished ≥ hi → 대기를 fast_ms로 줄인다(표가 더 짧으면 표). 기본 꺼짐(hi > 1)
  finished ≤ lo → 대기를 상한(cap)까지 늘린다(상한 자체는 그대로라 무음 상한은 지킨다)
  그 밖·응답 전·실패·시간 초과 → 표 결정기 그대로
기본 꺼짐. DIRECTON_JEV_TURN=1이면 make_detector()가 JevTurnDetector를 돌려준다.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from agent.core.types import TurnDecision, TurnSignals

log = logging.getLogger(__name__)

MODEL = "jev-1.14.0"          # 평가(data/turnset/jev_cache.jsonl)와 같은 판. 바꾸면 평가를 다시 돌린다
ENV_SWITCH = "DIRECTON_JEV_TURN"

SITUATION = ("Korean phone customer-service call for an online shop. The customer has just gone silent for a "
             "fraction of a second. `customer_speech_so_far` is a live speech-to-text transcript of everything the "
             "customer has said in the current turn; it may contain recognition errors and missing punctuation.")

QUESTIONS: dict[str, dict] = {
    "finished": {
        "type": "noul",
        "instructions": ("Has the customer finished their turn, so that the agent should start replying now? "
                         "Judge `customer_speech_so_far` in light of `previous_conversation`."),
        "criteria": {
            "true": ("The turn is complete: a full request, question, answer, or statement after which a phone "
                     "agent would naturally start replying."),
            "false": ("The customer is probably mid-turn: the transcript stops mid-sentence or mid-number, or what "
                      "was said so far is filler, hesitation, a self-correction, or a setup that is normally "
                      "followed by more (the actual request, an order or phone number, or a corrected value)."),
        },
    },
    "cut_off": {
        "type": "noul",
        "instructions": "Does `customer_speech_so_far` stop in the middle of a sentence, phrase, or number?",
        "criteria": {
            "true": "It ends with an unfinished clause, a connective ending, a particle, filler, or a partial number.",
            "false": "It ends at a grammatically complete point (a finished sentence, question, or short answer).",
        },
    },
    "more_expected": {
        "type": "noul",
        "instructions": ("Is the customer likely to keep talking right after this short pause, for example to add "
                         "an order or phone number, the change they want, a corrected value, or a follow-up question?"),
        "criteria": {
            "true": "More speech from the customer is expected before the agent should reply.",
            "false": "Nothing more is expected; the agent can reply now.",
        },
    },
}


def build_state(text: str, history: list[dict] | None = None, caller_known: bool = False) -> dict:
    return {
        "situation": SITUATION,
        "previous_conversation": [{"role": m.get("role"), "text": m.get("text", m.get("content"))}
                                  for m in (history or [])],
        "caller_already_identified": bool(caller_known),
        "customer_speech_so_far": (text or "").strip(),
    }


def build_body(text: str, history: list[dict] | None = None, caller_known: bool = False,
               model: str = MODEL) -> dict:
    """HTTP 본문(평가기 캐시 키와 같은 모양)."""
    return {"model": model, "state": build_state(text, history, caller_known), "questions": QUESTIONS}


@dataclass(frozen=True)
class JevTurnResult:
    finished: float
    cut_off: float
    more_expected: float
    ms: float = 0.0


def _noul(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        x = v.get("noul")
    else:
        x = getattr(v, "noul", None)
    return None if x is None else float(x)


def parse_answers(resp: Any, ms: float = 0.0) -> JevTurnResult | None:
    """SDK SystemOneResponse, HTTP 응답 dict, 캐시 형식({"answers": {q: p}}) 모두 받는다."""
    if isinstance(resp, dict):
        ans = resp.get("answers") or {}
    else:
        ans = getattr(resp, "nouls", None) or getattr(resp, "answers", None) or {}
    vals = {k: _noul(ans.get(k)) for k in QUESTIONS}
    if any(v is None for v in vals.values()):
        return None
    return JevTurnResult(vals["finished"], vals["cut_off"], vals["more_expected"], ms)


def _dotenv_key() -> str | None:
    env = Path(__file__).resolve().parents[2] / ".env"
    if not env.exists():
        return None
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("TYPESAFE_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


def api_key() -> str | None:
    return os.environ.get("TYPESAFE_API_KEY") or _dotenv_key()


def make_client(timeout_s: float, key: str | None = None) -> Any:
    """typesafe_sdk 비동기 클라이언트. 재시도 없음(침묵 창 안에서만 의미가 있다)."""
    from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy
    return AsyncTypeSafeClient(api_key=key or api_key(), retry=RetryPolicy(max_retries=0),
                               timeout=max(0.1, timeout_s))


class SystemOneClient(Protocol):
    async def system_one(self, state: Any, questions: Any, **kw: Any) -> Any: ...


class JevTurnJudge:
    """client: system_one(state=, questions=, model=)을 가진 비동기 클라이언트(typesafe_sdk.AsyncTypeSafeClient 등).
    None이면 첫 요청 때 만든다. 실패·시간 초과는 None(호출자는 표 결정기 그대로)."""

    def __init__(self, client: SystemOneClient | None = None, *, model: str = MODEL, timeout_s: float = 0.6):
        self._client = client
        self.model = model
        self.timeout_s = timeout_s

    def _get(self) -> SystemOneClient:
        if self._client is None:
            self._client = make_client(self.timeout_s)
        return self._client

    async def judge(self, text: str, history: list[dict] | None = None,
                    caller_known: bool = False) -> JevTurnResult | None:
        text = (text or "").strip()
        if not text:
            return None
        t0 = time.perf_counter()
        try:
            resp = await asyncio.wait_for(
                self._get().system_one(state=build_state(text, history, caller_known), questions=QUESTIONS,
                                       model=self.model),
                self.timeout_s)
        except (Exception, asyncio.TimeoutError) as e:  # noqa: BLE001
            log.info("jev turn failed: %r", e)
            return None
        return parse_answers(resp, (time.perf_counter() - t0) * 1000)


# ── 대기 결합 ──
@dataclass(frozen=True)
class JevTurnPolicy:
    # 기본값 = 늘리기만(docs/decisions/jev_turn.md 4절): 줄이기는 지연 p50을 줄이지만 false cut이 늘고,
    # 대기 ~1초까지는 속도 점수 손실이 없어 이득이 없다. hi > 1이면 줄이기를 끈다.
    hi: float = 1.01              # finished ≥ hi → 대기를 fast_ms로
    lo: float = 0.3               # finished ≤ lo → 대기를 상한까지
    fast_ms: int = 600
    extend: bool = True


DEFAULT_POLICY = JevTurnPolicy()

# AG-T33(AD13 B, 2026-11-21 사용자 확정): Soniox STT에서 쓰는 설정. make_detector()가 쓴다.
# 늘리기만(lo 0.5), 최근 응답 규칙, 침묵당 요청 6. 근거: docs/decisions/soniox_gate.md 7·8절
# (6개 셋 Soniox 재생에서 실제 끊음 62 → 51, miss +4, p95 거의 그대로. lo는 같은 데이터에서 고른 값이라 낙관적).
SONIOX_POLICY = JevTurnPolicy(hi=1.01, lo=0.5)
SONIOX_MATCH = "latest"
SONIOX_MAX_REQUESTS = 6


def combine_wait(table_wait: int, cap: int, p: float | None, pol: JevTurnPolicy = DEFAULT_POLICY) -> tuple[int, str]:
    if p is None:
        return table_wait, ""
    if p >= pol.hi:
        return min(table_wait, pol.fast_ms), f"jev_fast(p={p:.2f})"
    if pol.extend and p <= pol.lo:
        return max(table_wait, cap), f"jev_hold(p={p:.2f})"
    return table_wait, ""


def decide_with_p(s: TurnSignals, base: Any, p: float | None, pol: JevTurnPolicy = DEFAULT_POLICY) -> TurnDecision:
    """table_decide와 같은 순서. p가 없거나 정책 밖이면 base.decide(s)와 같은 결과."""
    c = base.config
    if p is None:
        return base.decide(s)
    if s.speech_ms < c.min_speech_ms and not (s.text or "").strip():
        return base.decide(s)
    if s.silence_ms < c.min_silence_ms:
        return base.decide(s)
    cap = base.cap_ms(s)
    if s.silence_ms >= cap:
        return base.decide(s)
    w0, why = base.wait_ms(s)
    w, tag = combine_wait(w0, cap, p, pol)
    if not tag:
        return base.decide(s)
    w = max(c.min_silence_ms, w)
    if s.silence_ms >= w:
        return TurnDecision(True, f"end {tag} wait={w} {why} silence={s.silence_ms}")
    return TurnDecision(False, f"wait {tag} wait={w} {why} silence={s.silence_ms}")


def _key(text: str) -> str:
    return " ".join((text or "").split())


class JevTurnDetector:
    """TableTurnDetector를 감싼 결정기(상태 있음, 세션마다 새로). 계약: agent.core.types.TurnDetector.

    - judge가 있으면 침묵 request_at_ms(200)에 그 텍스트로 Jev를 비동기로 부른다(실행 중 이벤트 루프가 있을 때만).
      텍스트가 바뀌면(부분 결과 → finalize) 한 침묵 안에서 max_requests까지 다시 부른다.
    - set_jev(text, p)로 바깥에서 확률을 넣을 수도 있다(세션이 선호출과 함께 부르는 경우).
    - match="exact"(기본, AG-T24): p는 그 텍스트와 같을 때만 쓴다. 새 요청을 걸면 진행 중인 요청은 취소한다.
    - match="latest"(AG-T33): 이번 침묵에서 가장 최근에 온 p를 텍스트가 바뀌어도 쓴다. 진행 중인 요청은 취소하지 않는다.
      Soniox 부분 결과는 토큰이 약 1초 늦어 침묵 중에도 바뀌므로("주문" → "주문했는데요") exact면 p가 거의 버려진다.
    - 음성이 다시 시작되면(silence_ms가 줄면) p와 요청을 모두 잊는다.
    """

    def __init__(self, base: Any, judge: JevTurnJudge | None = None, policy: JevTurnPolicy = DEFAULT_POLICY,
                 request_at_ms: int = 200, max_requests: int = 2, match: str = "exact"):
        if match not in ("exact", "latest"):
            raise ValueError(f"match {match!r}")
        self.base = base
        self.judge = judge
        self.policy = policy
        self.request_at_ms = request_at_ms
        self.max_requests = max_requests
        self.match = match
        self.history: list[dict] = []
        self.caller_known = False
        self._p: tuple[str, float] | None = None
        self._tasks: list[tuple[str, asyncio.Task]] = []     # (텍스트 키, 작업) 요청 순
        self._asked: set[str] = set()
        self._requests = 0
        self._last_silence = 0
        self.last_result: JevTurnResult | None = None

    @property
    def config(self) -> Any:
        return self.base.config

    def wait_ms(self, s: TurnSignals) -> tuple[int, str]:
        return self.base.wait_ms(s)

    def cap_ms(self, s: TurnSignals) -> int:
        return self.base.cap_ms(s)

    def set_context(self, history: list[dict] | None, caller_known: bool = False) -> None:
        self.history = list(history or [])
        self.caller_known = bool(caller_known)

    def set_jev(self, text: str, p: float | None) -> None:
        self._p = None if p is None else (_key(text), float(p))

    def latest(self) -> tuple[str, float] | None:
        return self._p

    def close(self) -> None:
        for _, t in self._tasks:
            if not t.done():
                t.cancel()
        self._tasks = []

    def _reset(self) -> None:
        self.close()
        self._p = None
        self._asked = set()
        self._requests = 0

    def _collect(self) -> None:
        """끝난 요청을 요청 순서대로 반영한다(나중 요청이 이긴다)."""
        keep = []
        for k, t in self._tasks:
            if not t.done():
                keep.append((k, t))
                continue
            if t.cancelled() or t.exception() is not None:
                continue
            r = t.result()
            if r is not None:
                self.last_result = r
                self.set_jev(k, r.finished)
        self._tasks = keep

    def _maybe_request(self, s: TurnSignals) -> None:
        if self.judge is None or s.silence_ms < self.request_at_ms or self._requests >= self.max_requests:
            return
        k = _key(s.text)
        if not k or k in self._asked or (self._p is not None and self._p[0] == k):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self.match == "exact":
            self.close()                                    # AG-T24 동작: 최신 텍스트 요청만 남긴다
        self._asked.add(k)
        self._requests += 1
        self._tasks.append((k, loop.create_task(self.judge.judge(s.text, self.history, self.caller_known))))

    def decide(self, s: TurnSignals) -> TurnDecision:
        if s.silence_ms < self._last_silence or s.silence_ms == 0:
            self._reset()                                   # 새 침묵(음성 재개)
        self._last_silence = s.silence_ms
        self._collect()
        self._maybe_request(s)
        if self._p is None:
            p = None
        elif self.match == "latest":
            p = self._p[1]
        else:
            p = self._p[1] if self._p[0] == _key(s.text) else None
        return decide_with_p(s, self.base, p, self.policy)


_shared_judge: JevTurnJudge | None = None


def shared_judge() -> JevTurnJudge | None:
    """세션들이 같이 쓰는 판정기(연결 재사용). 키가 없으면 None."""
    global _shared_judge
    if not api_key():
        return None
    if _shared_judge is None:
        _shared_judge = JevTurnJudge(timeout_s=float(os.environ.get("DIRECTON_JEV_TIMEOUT_S", "0.6")))
    return _shared_judge


def switch_on() -> bool:
    return os.environ.get(ENV_SWITCH, "").strip() == "1"
