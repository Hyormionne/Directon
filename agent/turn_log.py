"""턴 로그: 턴마다 JSON 한 줄 → runs/logs/turns-YYYYMMDD.jsonl."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


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
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"turns-{self.date_fn()}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
