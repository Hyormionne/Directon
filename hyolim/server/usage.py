"""크레딧 사용량 기록 — 턴마다 Soniox 로 보낸 음성 초, TTS 로 보낸 글자 수(캐시로 아낀 글자 수 포함)를 남긴다.

usage/usage.jsonl   턴별 기록 (한 줄 = 한 턴)
usage/total.json    누적 합계 — 측정을 몇 번 돌렸는지와 상관없이 계속 더해진다

누적값을 보고 크레딧이 얼마나 남았는지 가늠한다. 기록만 하고, 서버 동작을 막지는 않는다(최종 측정에 영향 없음).
"""
from __future__ import annotations

import json
import os
import threading
import time

from . import config as C

DIR = os.path.join(C.ROOT, "usage")
_lock = threading.Lock()


def _total_path():
    return os.path.join(DIR, "total.json")


def add_turn(case_id: str, stt_sec: float, tts_chars: int, tts_cached_chars: int):
    """한 턴 사용량을 기록하고 누적에 더한다. 실패해도 응대에는 영향이 없게 조용히 넘어간다."""
    try:
        os.makedirs(DIR, exist_ok=True)
        rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "case": case_id, "stt_backend": C.STT_BACKEND,
               "tts_backend": C.TTS_BACKEND, "stt_sec": round(stt_sec, 2), "tts_chars": tts_chars,
               "tts_cached_chars": tts_cached_chars}
        with _lock:
            with open(os.path.join(DIR, "usage.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tot = json.load(open(_total_path(), encoding="utf-8")) if os.path.exists(_total_path()) else {}
            k = f"{C.STT_BACKEND}/{C.TTS_BACKEND}"
            t = tot.setdefault(k, {"turns": 0, "stt_min": 0.0, "tts_chars": 0, "tts_cached_chars": 0})
            t["turns"] += 1
            t["stt_min"] = round(t["stt_min"] + stt_sec / 60, 3)
            t["tts_chars"] += tts_chars
            t["tts_cached_chars"] += tts_cached_chars
            json.dump(tot, open(_total_path(), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    except Exception:
        pass


def summary() -> str:
    if not os.path.exists(_total_path()):
        return "사용량 기록 없음"
    tot = json.load(open(_total_path(), encoding="utf-8"))
    return " / ".join(f"{k}: 턴 {v['turns']}, STT {v['stt_min']:.1f}분, TTS {v['tts_chars']}자(캐시로 아낀 {v['tts_cached_chars']}자)"
                      for k, v in tot.items())


if __name__ == "__main__":
    print(summary())
