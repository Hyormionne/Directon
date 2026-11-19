"""동시 세션 부하 도구(AG-T18). 실행 중인 상담원 서버에 N세션 동시 × M회를 보내 실패·첫 소리·메모리를 본다.

실행 예(서버·Mock은 따로 띄운다):
  uv run python -m mock_api.server --port 8100
  uv run python -m agent.server --port 9000
  uv run python -m eval.stress --server ws://127.0.0.1:9000 --sessions 4 --rounds 2 \
    [--cases data/testset/cases.jsonl --audio data/testset/audio --mock http://127.0.0.1:8100] \
    [--pid <서버 PID>] [--out runs/stress_X.json]

- 측정기와 같은 방식: session.start → 20ms 실시간 PCM(말이 끝나도 무음 계속) → turn.end까지 받는다.
- 첫 소리 = RMS ≥ 300인 첫 바이너리 도착 − speech_end(meta.json). 음수면 말 끝 전 발화(속도 0점).
- 실패 = 연결·프로토콜 오류, 또는 꼬리 시간(--tail, 기본 25초) 안에 turn.end 없음.
- 메모리: 서버 프로세스 RSS를 회차 전후·진행 중 1초마다 잰다. --pid가 없으면 포트로 PID를 찾는다(netstat).
- 측정 서버와 같은 번호 정규화·채점은 하지 않는다(그건 eval.runner/score). 여기서는 안정성만 본다.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
import uuid
import wave
from typing import Any, Callable
from urllib.parse import urlparse

import numpy as np

RATE = 16000
FRAME_BYTES = 640           # 20ms
SILENCE_RMS = 300.0
TAIL_TIMEOUT = 25.0


# ── 순수 함수 ─────────────────────────────────────────
def rms(b: bytes) -> float:
    a = np.frombuffer(b[: len(b) // 2 * 2], dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0


def first_voiced(frames: list[tuple[float, bytes]]) -> float | None:
    """[(speech_end 기준 시각, bytes)] → RMS ≥ 300인 첫 프레임 시각."""
    for t, b in frames:
        if rms(b) >= SILENCE_RMS:
            return t
    return None


def percentile(xs: list[float], p: float) -> float | None:
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return None
    return float(np.percentile(xs, p))


def pick_cases(cases: list[dict], n: int, round_idx: int) -> list[dict]:
    """회차마다 이어서 n개(모자라면 처음부터 돌려 쓴다)."""
    if not cases:
        return []
    start = round_idx * n
    return [cases[(start + i) % len(cases)] for i in range(n)]


def pid_from_netstat(text: str, port: int) -> int | None:
    """`netstat -ano`(Windows) 출력에서 port를 LISTEN하는 PID."""
    pat = re.compile(rf"^\s*TCP\s+\S*:{port}\s+\S+\s+LISTEN(?:ING)?\s+(\d+)\s*$", re.I)
    for line in text.splitlines():
        m = pat.match(line)
        if m:
            return int(m.group(1))
    return None


def _ok(r: dict) -> bool:
    return "error" not in r and r.get("turn_end") is not None


def summarize(runs: list[dict], mem: list[dict]) -> dict:
    ok = [r for r in runs if _ok(r)]
    fs = [r["first_sound"] for r in ok if r.get("first_sound") is not None]
    te = [r["turn_end"] for r in ok]
    rss = [m["rss_mb"] for m in mem if m.get("rss_mb") is not None]
    out = {
        "sessions": len(runs),
        "ok": len(ok),
        "failures": len(runs) - len(ok),
        "errors": sum(1 for r in runs if "error" in r),
        "no_turn_end": sum(1 for r in runs if "error" not in r and r.get("turn_end") is None),
        "no_sound": sum(1 for r in ok if r.get("first_sound") is None),
        "spoke_before_end": sum(1 for r in runs if r.get("spoke_before_end")),
        "first_sound_p50": percentile(fs, 50),
        "first_sound_p95": percentile(fs, 95),
        "first_sound_max": max(fs) if fs else None,
        "turn_end_p50": percentile(te, 50),
        "turn_end_p95": percentile(te, 95),
        "mem_start_mb": rss[0] if rss else None,
        "mem_peak_mb": max(rss) if rss else None,
        "mem_end_mb": rss[-1] if rss else None,
        "mem_growth_mb": (rss[-1] - rss[0]) if rss else None,
    }
    return out


# ── 메모리 ─────────────────────────────────────────────
def find_pid(port: int) -> int | None:
    try:
        if sys.platform == "win32":
            txt = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True,
                                 timeout=10).stdout
            return pid_from_netstat(txt, port)
        txt = subprocess.run(["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"], capture_output=True,
                             text=True, timeout=10).stdout
        return int(txt.split()[0]) if txt.split() else None
    except Exception:  # noqa: BLE001
        return None


def rss_mb(pid: int) -> float | None:
    """프로세스 RSS(MB). Windows는 psapi, 그 밖은 /proc 또는 ps."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.OpenProcess.restype = wintypes.HANDLE
            h = k32.OpenProcess(0x1000 | 0x0010, False, pid)     # QUERY_LIMITED_INFORMATION | VM_READ
            if not h:
                return None
            try:
                pmc = PMC()
                pmc.cb = ctypes.sizeof(PMC)
                if not k32.K32GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
                    return None
                return pmc.WorkingSetSize / 1e6
            finally:
                k32.CloseHandle(h)
        status = f"/proc/{pid}/status"
        if os.path.exists(status):
            for line in open(status, encoding="utf-8"):
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e3
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout
        return int(out.strip()) / 1e3 if out.strip() else None
    except Exception:  # noqa: BLE001
        return None


# ── 세션 하나 ─────────────────────────────────────────
def load_wav(path: str) -> bytes:
    with wave.open(path) as w:
        assert w.getframerate() == RATE and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return w.readframes(w.getnframes())


async def run_session(server: str, case: dict, pcm: bytes, speech_end_off: float, mock: str,
                      tail: float = TAIL_TIMEOUT) -> dict:
    import websockets

    sid = f"STRESS-{case['case_id']}-{uuid.uuid4().hex[:6]}"
    out: dict[str, Any] = {"case_id": case["case_id"], "session_id": sid}
    frames: list[tuple[float, bytes]] = []
    turn_end: float | None = None
    texts: list[str] = []
    try:
        async with websockets.connect(server, max_size=None, open_timeout=10) as ws:
            await ws.send(json.dumps({
                "type": "session.start", "session_id": sid, "case_id": case["case_id"],
                "now": case.get("now", ""), "history": case.get("history") or [],
                "caller": case.get("customer_profile"), "tools_url": mock,
                "audio": {"encoding": "pcm_s16le", "sample_rate": RATE, "channels": 1}}, ensure_ascii=False))
            t0 = time.perf_counter()
            speech_end = t0 + speech_end_off
            done = asyncio.Event()

            async def sender() -> None:
                k = 0
                silence = b"\0" * FRAME_BYTES
                while not done.is_set():
                    i = k * FRAME_BYTES
                    chunk = pcm[i:i + FRAME_BYTES] if i < len(pcm) else silence
                    await ws.send(chunk.ljust(FRAME_BYTES, b"\0"))
                    k += 1
                    wait = t0 + k * 0.02 - time.perf_counter()
                    if wait > 0:
                        await asyncio.sleep(wait)
                    if time.perf_counter() > speech_end + tail:
                        done.set()

            async def receiver() -> None:
                nonlocal turn_end
                async for m in ws:
                    now = time.perf_counter() - speech_end
                    if isinstance(m, bytes):
                        frames.append((now, m))
                    else:
                        e = json.loads(m)
                        if e.get("type") == "agent.transcript":
                            texts.append(str(e.get("text", "")))
                        elif e.get("type") == "turn.end":
                            turn_end = now
                            done.set()
                            return

            tasks = [asyncio.create_task(sender()), asyncio.create_task(receiver())]
            await done.wait()
            for t in tasks:
                t.cancel()
            for t in tasks:
                if t.done() and not t.cancelled() and t.exception() is not None:
                    raise t.exception()        # type: ignore[misc]
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)[:300]
        return out
    fs = first_voiced(frames)
    out.update(first_sound=fs, turn_end=turn_end, spoke_before_end=bool(fs is not None and fs < 0),
               agent_audio_sec=round(sum(len(b) for _, b in frames) / 2 / RATE, 2),
               agent_text=" ".join(texts))
    return out


async def _sample_mem(pid: int | None, mem: list[dict], stop: asyncio.Event, t0: float,
                      reader: Callable[[int], float | None] = rss_mb) -> None:
    while not stop.is_set():
        if pid:
            mem.append({"t": round(time.perf_counter() - t0, 1), "rss_mb": reader(pid)})
        try:
            await asyncio.wait_for(stop.wait(), 1.0)
        except asyncio.TimeoutError:
            pass


# ── 진입점 ────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="상담원 서버 동시 세션 부하(AG-T18)")
    ap.add_argument("--server", required=True)
    ap.add_argument("--cases", default="data/testset/cases.jsonl")
    ap.add_argument("--audio", default="data/testset/audio")
    ap.add_argument("--mock", default="http://127.0.0.1:8100")
    ap.add_argument("--sessions", type=int, default=4, help="동시 세션 수(측정기는 4)")
    ap.add_argument("--rounds", type=int, default=3, help="회차 수")
    ap.add_argument("--offset", type=int, default=0, help="문항 시작 위치")
    ap.add_argument("--pid", type=int, default=0, help="서버 PID(없으면 포트로 찾는다)")
    ap.add_argument("--tail", type=float, default=TAIL_TIMEOUT)
    ap.add_argument("--out", default="")
    return ap.parse_args(argv)


async def main(argv: list[str] | None = None) -> dict:
    a = parse_args(argv)
    with open(a.cases, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    with open(os.path.join(a.audio, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    cases = [c for c in cases if c["case_id"] in meta][a.offset:]
    pid = a.pid or find_pid(urlparse(a.server).port or 80)
    if not pid:
        print("서버 PID를 찾지 못했다(메모리 생략). --pid로 지정할 수 있다.", flush=True)
    mem: list[dict] = []
    stop = asyncio.Event()
    t0 = time.perf_counter()
    sampler = asyncio.create_task(_sample_mem(pid, mem, stop, t0))
    runs: list[dict] = []
    for r in range(a.rounds):
        batch = pick_cases(cases, a.sessions, r)
        res = await asyncio.gather(*(
            run_session(a.server, c, load_wav(os.path.join(a.audio, c["case_id"] + ".wav")),
                        meta[c["case_id"]]["speech_end"], a.mock, a.tail) for c in batch))
        for x in res:
            x["round"] = r
            fs = x.get("first_sound")
            state = ("ERROR " + x["error"]) if "error" in x else (
                "turn.end 없음" if x.get("turn_end") is None else
                f"첫 소리 {fs:.2f}s" if fs is not None else "소리 없음")
            print(f"[{r}] {x['case_id']}: {state}{' [말 끝 전 발화]' if x.get('spoke_before_end') else ''}"
                  f" | {x.get('agent_text', '')[:50]}", flush=True)
        runs.extend(res)
    stop.set()
    await sampler
    if pid:
        mem.append({"t": round(time.perf_counter() - t0, 1), "rss_mb": rss_mb(pid)})
    summary = summarize(runs, mem)
    summary.update(server=a.server, concurrency=a.sessions, rounds=a.rounds, pid=pid,
                   wall_s=round(time.perf_counter() - t0, 1))
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "runs": runs, "mem": mem}, f, ensure_ascii=False, indent=1)
    return summary


if __name__ == "__main__":
    asyncio.run(main())
