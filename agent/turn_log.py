"""턴 로그: 턴마다 JSON 한 줄 → runs/logs/turns-YYYYMMDD.jsonl.

AG-T17: 시험 음성 저장 금지. DIRECTON_NO_AUDIO_LOG=1이면 고객 발화 원문(user_text, 선호출·조기 finalize
텍스트, 번호 교정 전 들은 값)을 지우고 쓴다. 시간 지표(marks)·상담원 문장·함수 호출 기록은 남긴다.
오디오는 이 모듈이 쓰지 않는다. 다른 모듈은 저장 전에 no_audio_log()를 확인한다.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


NO_AUDIO_LOG_ENV = "DIRECTON_NO_AUDIO_LOG"

# 고객 발화 원문이 담기는 키(최상위·이벤트 공통)
_CUSTOMER_KEYS = frozenset({"user_text", "heard", "stt_text", "transcript", "partial", "final_text", "customer_text"})
# 이 이벤트의 "text"는 고객 발화다(presynth·tts_request·filler의 text는 상담원 문장이라 남긴다)
_CUSTOMER_TEXT_EVENTS = frozenset({"precall", "finalize_early", "stt", "stt_partial", "stt_final", "user"})


def no_audio_log(env=None) -> bool:
    """시험 음성 저장 금지 모드인가(DIRECTON_NO_AUDIO_LOG=1/true/yes/on)."""
    env = os.environ if env is None else env
    return str(env.get(NO_AUDIO_LOG_ENV, "")).strip().lower() in ("1", "true", "yes", "on")


def scrub_customer_text(record: dict) -> dict:
    """고객 발화 원문을 뺀 사본."""
    out = {k: v for k, v in record.items() if k not in _CUSTOMER_KEYS}
    events = []
    for e in record.get("events") or []:
        if not isinstance(e, dict):
            continue
        e2 = {k: v for k, v in e.items() if k not in _CUSTOMER_KEYS}
        if e2.get("name") in _CUSTOMER_TEXT_EVENTS:
            e2.pop("text", None)
        events.append(e2)
    if "events" in record:
        out["events"] = events
    return out


class TurnLog:
    """marks: 한 번만 일어나는 시점(ms, 턴 시작 기준). events: 반복되는 시점(툴, 문장 TTS)."""

    def __init__(self, session_id: str, clock: Callable[[], float] = time.perf_counter,
                 t0: float | None = None):
        self.session_id = session_id
        self.clock = clock
        self.t0 = clock() if t0 is None else t0
        self.marks: dict[str, float] = {}
        self.events: list[dict] = []
        self.fields: dict[str, Any] = {}

    def _ms(self) -> float:
        return round((self.clock() - self.t0) * 1000, 1)

    def mark(self, name: str, value: float | None = None, *, overwrite: bool = False) -> None:
        if overwrite or name not in self.marks:
            self.marks[name] = self._ms() if value is None else value

    def event(self, name: str, **extra: Any) -> None:
        self.events.append({"name": name, "t": self._ms(), **extra})

    def set(self, key: str, value: Any) -> None:
        self.fields[key] = value

    def to_dict(self) -> dict:
        return {"session_id": self.session_id, "wall": time.time(), **self.fields,
                "marks": self.marks, "events": self.events}


class TurnLogWriter:
    def __init__(self, log_dir: str | Path = "runs/logs",
                 date_fn: Callable[[], str] = lambda: datetime.now().strftime("%Y%m%d")):
        self.dir = Path(log_dir)
        self.date_fn = date_fn

    def write(self, record: dict) -> None:
        if no_audio_log():
            record = scrub_customer_text(record)
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"turns-{self.date_fn()}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
