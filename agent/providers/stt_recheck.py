"""숫자 재인식 래퍼(AG-T10b).

Live STT가 끊어 읽은 숫자를 빠뜨리는 문제(통화 채널·잡음)를 STT 계약 안에서 보정한다.
`DigitRecheckSTT`는 STTProvider를 감싸고, 스트림은 받은 오디오 전체를 모아 둔다.
finalize 결과에 6자리 이상 숫자열이 있는데 자릿수가 기대와 다르면(10자리 주문번호,
01로 시작하는 11자리 휴대폰 번호가 아니면) 모은 오디오 전체를 batch 모델로 다시 받아쓴다.
재인식 결과에 기대 자릿수 숫자열이 더 많을 때만 교체한다. 실패·시간 초과면 원래 결과.
session.py는 바꿀 필요가 없다(finalize 안에서 끝난다).
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Awaitable, Callable

from agent.normalize import _parse_number_token, _strip_suffix

log = logging.getLogger(__name__)

RECHECK_PROMPT = (
    "이 오디오는 온라인 쇼핑몰 고객센터 전화의 고객 발화다. 처음부터 끝까지 모든 말을 들리는 그대로 "
    "한국어로 받아써라. 숫자 아닌 말도 빠짐없이 쓴다. "
    "고객은 숫자를 한 자리씩 끊어 읽는다(공=0, 일=1, 이=2, 삼=3, 사=4, 오=5, 육=6, 칠=7, 팔=8, 구=9). "
    "들린 숫자는 한 자리도 빠뜨리거나 더하거나 바꾸지 말고 들린 순서대로 아라비아 숫자로 쓴다. "
    "추측해서 자릿수를 맞추지 않는다. 받아쓴 텍스트만 출력한다."
)
DEFAULT_MIN_LEN = 6
_LEAD_DIGITS = re.compile(r"\d+")
_TOKEN_RE = re.compile(r"[^\s,.\-+/·~()\[\]:;!?\"']+")   # normalize._SPLIT_RE의 여집합


def digit_run_spans(text: str) -> list[tuple[str, int, int]]:
    """이어진 숫자 토큰(아라비아·한글 낭독)을 합친 (숫자열, 시작, 끝) 목록. 숫자가 아닌 토큰에서 끊는다.
    끝 위치는 마지막 숫자 토큰의 숫자 부분 끝(조사 앞)."""
    runs: list[tuple[str, int, int]] = []
    cur, start, end = "", -1, -1

    def flush() -> None:
        nonlocal cur, start, end
        if cur:
            runs.append((cur, start, end))
        cur, start, end = "", -1, -1

    for m in _TOKEN_RE.finditer(text or ""):
        raw, pos = m.group(0), m.start()
        tok = _strip_suffix(raw)
        d = _parse_number_token(tok) if tok else None
        if d is not None:
            if not cur:
                start = pos
            cur += d                # "공일공에 구팔칠육에"처럼 조사가 붙어도 다음 숫자 토큰과 잇는다
            end = pos + len(tok)
            continue
        lead = _LEAD_DIGITS.match(raw)  # "5678로", "3개"처럼 아라비아 숫자 뒤에 다른 말이 붙은 토큰
        if lead and len(lead.group(0)) >= 3:
            if not cur:
                start = pos
            cur += lead.group(0)
            end = pos + len(lead.group(0))
            flush()
            continue
        flush()
        if lead:
            runs.append((lead.group(0), pos, pos + len(lead.group(0))))
    flush()
    return runs


def digit_runs(text: str) -> list[str]:
    return [r for r, _, _ in digit_run_spans(text)]


def valid_run(run: str) -> bool:
    return len(run) == 10 or (len(run) == 11 and run.startswith("01"))


def needs_recheck(text: str, min_len: int = DEFAULT_MIN_LEN) -> bool:
    return any(len(r) >= min_len and not valid_run(r) for r in digit_runs(text))


def _valid_count(text: str) -> int:
    return sum(1 for r in digit_runs(text) if valid_run(r))


def interim_fix(final: str, interim: str, min_len: int = DEFAULT_MIN_LEN) -> str | None:
    """확정 결과의 틀린 숫자열 하나를, finalize 직전 부분 결과에 있던 맞는 자릿수 숫자열로 바꾼다.
    확정에 틀린 숫자열이 정확히 하나, 부분 결과에 확정에 없는 맞는 숫자열이 정확히 하나일 때만."""
    bad = [r for r in digit_run_spans(final) if len(r[0]) >= min_len and not valid_run(r[0])]
    final_runs = set(digit_runs(final))
    good = [r for r in digit_runs(interim) if valid_run(r) and r not in final_runs]
    if len(bad) != 1 or len(good) != 1:
        return None
    _, a, b = bad[0]
    return final[:a] + good[0] + final[b:]


Recheck = Callable[[bytes, str], Awaitable[str]]


class DigitRecheckStream:
    def __init__(self, inner: Any, recheck: Recheck | None, context: str, timeout: float, min_len: int,
                 interim_fix: bool = True):
        self._interim_fix = interim_fix
        self._inner = inner
        self._recheck = recheck
        self._context = context
        self._timeout = timeout
        self._min_len = min_len
        self._pcm = bytearray()
        self._override: str | None = None
        self.last_recheck: dict = {"triggered": False, "replaced": False, "ms": 0.0, "text": "",
                                   "error": None}

    async def send_audio(self, pcm: bytes) -> None:
        if pcm:
            self._pcm += pcm
            self._override = None
        await self._inner.send_audio(pcm)

    def latest_text(self) -> str:
        return self._override if self._override is not None else self._inner.latest_text()

    async def finalize(self, timeout: float = 1.0) -> str:
        before = self._inner.latest_text()          # finalize 직전 부분 결과
        text = await self._inner.finalize(timeout)
        info = {"triggered": False, "replaced": False, "ms": 0.0, "text": "", "error": None,
                "live_text": text, "source": ""}
        self.last_recheck = info
        if not self._pcm or not needs_recheck(text, self._min_len):
            return text
        info["triggered"] = True
        if self._interim_fix:
            fixed = interim_fix(text, before, self._min_len)
            if fixed is not None:
                info.update(replaced=True, text=fixed, source="interim")
                self._override = fixed
                log.info("digit interim fix %r -> %r", text, fixed)
                return fixed
        if self._recheck is None:
            return text
        t0 = time.perf_counter()
        try:
            new = await asyncio.wait_for(self._recheck(bytes(self._pcm), self._context), self._timeout)
        except asyncio.CancelledError:
            raise
        except BaseException as e:  # noqa: BLE001 - 시간 초과·API 오류 모두 원래 결과 유지
            info["error"] = repr(e) or type(e).__name__
            new = ""
        info["ms"] = (time.perf_counter() - t0) * 1000
        new = (new or "").strip()
        info["text"] = new
        if new and _valid_count(new) > _valid_count(text):
            info["replaced"] = True
            info["source"] = "batch"
            self._override = new
            log.info("digit recheck replaced %r -> %r (%.0fms)", text, new, info["ms"])
            return new
        return text

    async def close(self) -> None:
        self._pcm = bytearray()
        await self._inner.close()


class DigitRecheckSTT:
    """STTProvider 래퍼. recheck(pcm, context) -> 텍스트(None이면 batch 재인식 없이 부분 결과 복원만)."""

    def __init__(self, inner: Any, recheck: Recheck | None, *, timeout: float = 1.5,
                 min_len: int = DEFAULT_MIN_LEN, recheck_model: str = "", interim_fix: bool = True):
        self.interim_fix = interim_fix
        self.inner = inner
        self.recheck = recheck
        self.timeout = float(timeout)
        self.min_len = int(min_len)
        self.recheck_model = recheck_model

    @property
    def primary(self) -> Any:
        return getattr(self.inner, "primary", self.inner)

    async def open_stream(self, context: str = "") -> Any:
        from agent.providers.gemini_stt import GeminiBatchStream

        stream = await self.inner.open_stream(context)
        if isinstance(stream, GeminiBatchStream):       # 이미 batch(폴백) → 재인식 의미 없음
            return stream
        return DigitRecheckStream(stream, self.recheck, context, self.timeout, self.min_len,
                                  interim_fix=self.interim_fix)
