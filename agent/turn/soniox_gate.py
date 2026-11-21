"""AG-T30 Soniox `<end>` 게이트: Soniox 엔드포인트마다 "진짜 말 끝인가"를 규칙 또는 Jev로 판정한다.

배경(docs/decisions/soniox_baze.md 3절): Soniox `<end>`만 쓰면 지연은 짧지만(p50 약 0.5초) 짧은 쉼마다 나와
말 끝 전 확정이 turnset 46.5%, 172문항 27.3%다. 그래서 `<end>`를 "후보"로만 보고 텍스트로 거른다.

흐름(한 턴)
  1. `<end>` 도착 → 그때까지의 확정 텍스트로 판정
       rule     : detector.table_kind 묶음이 보류(연결어미·조사·간투사·숫자 미완·자기 정정·쉼표)면 거절, 아니면 수락
       jev      : Jev `finished` ≥ jev_threshold면 수락(응답 실패·시간 초과면 rule로)
       rule_jev : rule이 보류면 거절, 아니면 Jev로 확인
  2. 수락 → 그 시각(Jev면 응답 도착 시각)에 턴 확정
  3. 거절 → 다음 `<end>`를 기다린다. 로컬 VAD 침묵이 hold_ms가 되면 확정(무음 상한)
  4. `<end>`가 오지 않아도 VAD 침묵이 cap_ms면 확정(Soniox가 말 끝을 놓친 경우)
"말하는 중/침묵 길이"는 Soniox 토큰이 아니라 로컬 VAD로 본다. Soniox 토큰은 음성보다 약 1~1.2초 늦게 도착해
(실측, AG-T30) 토큰으로 재면 다음 구절이 이미 시작됐는데도 침묵으로 오판한다.
  - `<end>`가 도착한 순간 VAD가 음성이면(고객이 이미 다음 말을 시작) 그 `<end>`는 지난 것이라 무시한다.
  - Jev 응답을 기다리는 사이 VAD 음성이 나오면 그 판정은 버린다. 음성이 나오면 hold도 풀린다.
빈 텍스트 `<end>`는 잡음으로 보고 무시한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from agent.turn.detector import CALIBRATED_CONFIG, TableTurnConfig, table_kind

HOLD_KINDS = frozenset({"connective", "filler", "particle", "digits_incomplete", "digits_partial",
                        "self_correction", "comma"})
MODES = ("rule", "jev", "rule_jev")


@dataclass(frozen=True)
class GatePolicy:
    mode: str = "rule"
    jev_threshold: float = 0.5
    hold_ms: int = 1000       # 거절 뒤 VAD 침묵이 이만큼이면 확정
    cap_ms: int = 1500        # `<end>`와 무관한 VAD 침묵 상한
    hold_kinds: frozenset = HOLD_KINDS


@dataclass
class GateVerdict:
    accept: bool
    reason: str
    kind: str = ""
    p: float | None = None
    ms: float = 0.0           # 판정에 걸린 시간(Jev 왕복). rule은 0


def rule_verdict(text: str, pol: GatePolicy = GatePolicy(), cfg: TableTurnConfig = CALIBRATED_CONFIG) -> GateVerdict:
    k = table_kind(text, cfg).kind
    if k in pol.hold_kinds:
        return GateVerdict(False, f"rule_hold:{k}", k)
    return GateVerdict(True, f"rule_end:{k}", k)


def combine(text: str, p: float | None, ms: float, pol: GatePolicy,
            cfg: TableTurnConfig = CALIBRATED_CONFIG) -> GateVerdict:
    """모드별 최종 판정. p는 Jev finished(없으면 None → rule로 대신)."""
    r = rule_verdict(text, pol, cfg)
    if pol.mode == "rule":
        return r
    if pol.mode == "rule_jev" and not r.accept:
        return r
    if p is None:
        return GateVerdict(r.accept, f"jev_none→{r.reason}", r.kind, None, ms)
    ok = p >= pol.jev_threshold
    return GateVerdict(ok, f"jev_{'end' if ok else 'hold'}(p={p:.2f})", r.kind, p, ms)


JevFn = Callable[[str, list, bool], Awaitable[Any]]  # → JevTurnResult | None


class SonioxEndpointGate:
    """실시간 판정기. jev: JevTurnJudge.judge 같은 async 함수(rule 모드면 없어도 된다)."""

    def __init__(self, policy: GatePolicy = GatePolicy(), jev: JevFn | None = None,
                 cfg: TableTurnConfig = CALIBRATED_CONFIG):
        if policy.mode not in MODES:
            raise ValueError(f"mode {policy.mode!r} not in {MODES}")
        self.policy, self.jev, self.cfg = policy, jev, cfg

    def needs_jev(self, text: str) -> bool:
        if self.policy.mode == "rule" or self.jev is None:
            return False
        return self.policy.mode == "jev" or rule_verdict(text, self.policy, self.cfg).accept

    async def judge(self, text: str, history: list | None = None, caller_known: bool = False) -> GateVerdict:
        text = (text or "").strip()
        if not text:
            return GateVerdict(False, "empty")
        p, ms = None, 0.0
        if self.needs_jev(text):
            res = await self.jev(text, history or [], caller_known)
            if res is not None:
                p, ms = res.finished, res.ms
        return combine(text, p, ms, self.policy, self.cfg)


# ── 오프라인 시뮬레이션(이벤트 재생) ──
@dataclass
class GateRun:
    decision_t: float | None
    reason: str = ""
    trace: list = field(default_factory=list)


def is_speech_token(tok: dict) -> bool:
    t = tok.get("text", "")
    return t not in ("<end>", "<fin>") and bool(t.strip())


FRAME_S = 0.02


def simulate_gate(events: list[dict], verdict_at: Callable[[str], GateVerdict], pol: GatePolicy,
                  speech: list[bool], until_s: float) -> GateRun:
    """20ms 프레임 단위 재생(세션과 같은 순서: 프레임 VAD → 그 사이 도착한 Soniox 응답 → 대기 확인).
    events: [{"t": 도착 초(스트림 시작 = 오디오 0초), "toks": [{"text", "is_final"}...]}] 시간순.
    speech: 프레임별 VAD 음성 여부(오디오 길이 뒤는 무음). verdict_at(text) → GateVerdict(ms 포함)."""
    final = ""
    ei = 0
    sil_ms = 0
    heard = False
    hold = False
    pending: tuple[float, GateVerdict] | None = None   # Jev 응답 도착 시각, 판정
    trace: list = []
    for f in range(int(round(until_s / FRAME_S))):
        t_end = (f + 1) * FRAME_S
        sp = speech[f] if f < len(speech) else False
        if sp:
            sil_ms, heard, hold, pending = 0, True, False, None
        else:
            sil_ms += 20
        while ei < len(events) and events[ei]["t"] <= t_end:
            ev = events[ei]
            ei += 1
            toks = ev.get("toks", [])
            for x in toks:
                if x.get("is_final") and is_speech_token(x):
                    final += x["text"]
            if not any(x.get("text") == "<end>" for x in toks) or not final.strip():
                continue
            text = final.strip()
            if sil_ms == 0:
                trace.append({"t": ev["t"], "text": text, "accept": None, "reason": "stale(vad_speech)"})
                continue
            v = verdict_at(text)
            trace.append({"t": ev["t"], "text": text, "accept": v.accept, "reason": v.reason, "p": v.p})
            if v.ms > 0:
                pending = (ev["t"] + v.ms / 1000, v)
            elif v.accept:
                return GateRun(ev["t"], v.reason, trace)
            else:
                hold = True
        if pending and pending[0] <= t_end:
            pt, v = pending
            pending = None
            if v.accept:
                return GateRun(pt, v.reason, trace)
            hold = True
        if heard and not sp:
            if hold and sil_ms >= pol.hold_ms:
                return GateRun(t_end, "hold_timeout", trace)
            if sil_ms >= pol.cap_ms:
                return GateRun(t_end, "cap_timeout", trace)
    return GateRun(None, "no_end", trace)
