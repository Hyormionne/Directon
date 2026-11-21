"""Soniox 엔드포인트(`<end>`) vs 기존 턴 결정기 비교 (AG-T29).

오디오를 20ms씩 실시간 속도로 Soniox에 흘리고(`enable_endpoint_detection: true`),
`<end>` 토큰이 **도착한 벽시계 시각**(스트림 시작 기준, 네트워크 지연 포함)을 말 끝 라벨과 비교한다.
오디오가 끝나면 무음을 이어 보내 말 끝 + tail_s까지 기다린다.

판정은 eval/turn_sim.py와 같다.
  - 첫 `<end>`가 말 끝보다 앞: false cut(쉼 구간이면 그 쉼 유형, 아니면 intra/mid)
  - 말 끝 + window_s(1.5초) 안: ok, 지연 = `<end>` 시각 − 말 끝
  - 그 뒤이거나 없음: miss (늦게 온 `<end>`는 late_ms로 따로 남긴다)

데이터:
  --set turnset : data/turnset test split(228발화, 쉼 유형 라벨)
  --set ts172   : data/testset 172문항(meta.speech_end)

사용:
  uv run python -m eval.soniox_endpoint_eval --set turnset --out runs/endpoint/soniox_turnset
      [--extra '{"max_endpoint_delay_ms": 1000}'] [--concurrency 6] [--limit 20]
연결: factory.make_stt(STT_PROVIDER=soniox). 직접 키 대신 BAZE로 보내려면 --set-env SONIOX_API_KEY=
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from agent.core.types import SAMPLE_RATE

ROOT = Path(__file__).resolve().parents[1]
FRAME_MS = 20
FRAME_BYTES = SAMPLE_RATE * 2 * FRAME_MS // 1000
PAUSE_TOL_S = 0.06  # turn_sim.PAUSE_TOL_S와 같은 의미


@dataclass
class Pause:
    t: float
    dur_ms: float
    cut_type: str = "pause"


@dataclass
class EndpointResult:
    uid: str
    outcome: str                      # ok | false_cut | miss
    end_t: float
    first_end_t: float | None = None
    latency_ms: float | None = None
    late_ms: float | None = None      # miss인데 window 뒤에 온 경우의 지연
    where: str = ""
    ends: list[float] = field(default_factory=list)
    text: str = ""
    error: str = ""
    meta: dict = field(default_factory=dict)


def classify_endpoint(uid: str, ends: list[float], end_t: float, pauses: list[Pause],
                      window_s: float = 1.5, mid_label: str = "intra") -> EndpointResult:
    """`<end>` 도착 시각 목록 → turn_sim과 같은 판정."""
    if not ends:
        return EndpointResult(uid, "miss", end_t, ends=[])
    t = min(ends)
    if t + 1e-9 >= end_t:
        lat = round((t - end_t) * 1000, 1)
        if t - end_t <= window_s + 1e-9:
            return EndpointResult(uid, "ok", end_t, t, latency_ms=lat, ends=list(ends))
        return EndpointResult(uid, "miss", end_t, t, late_ms=lat, ends=list(ends))
    where = mid_label
    for p in pauses:
        if p.t - PAUSE_TOL_S <= t <= p.t + p.dur_ms / 1000 + PAUSE_TOL_S:
            where = p.cut_type or "pause"
            break
    return EndpointResult(uid, "false_cut", end_t, t, where=where, ends=list(ends))


async def stream_endpoints(stt: Any, pcm: bytes, until_s: float, *, extra: dict | None = None,
                           speed: float = 1.0, clock=time.monotonic, sleep=asyncio.sleep) -> tuple[list[float], str]:
    """pcm을 실시간으로 보내고(끝나면 무음) until_s까지 `<end>` 도착 시각(초)을 모은다. → (ends, final_text)"""
    msg = stt.start_message()
    msg["enable_endpoint_detection"] = True
    msg.update(extra or {})
    ws = await stt.connect(stt.url, {"Authorization": f"Bearer {stt.api_key}"})
    await ws.send(json.dumps(msg, ensure_ascii=False))
    ends: list[float] = []
    text: list[str] = []
    err: list[str] = []
    t0 = clock()

    async def rx():
        try:
            while True:
                res = json.loads(await ws.recv())
                now = clock() - t0
                if res.get("error_code"):
                    err.append(f"{res.get('error_code')} {res.get('error_message')}")
                    return
                for tok in res.get("tokens", []):
                    if tok.get("text") == "<end>":
                        ends.append(round(now, 3))
                    elif tok.get("is_final") and tok.get("text") not in ("<fin>",):
                        text.append(tok["text"])
                if res.get("finished"):
                    return
        except Exception as e:  # 정상 종료(ConnectionClosedOK)는 오류가 아니다
            if not err and "ConnectionClosedOK" not in type(e).__name__:
                err.append(repr(e))

    task = asyncio.create_task(rx())
    silence = b"\x00" * FRAME_BYTES
    total = max(int(until_s * 1000 / FRAME_MS), (len(pcm) + FRAME_BYTES - 1) // FRAME_BYTES)
    try:
        for i in range(total):
            if task.done():
                break
            chunk = pcm[i * FRAME_BYTES:(i + 1) * FRAME_BYTES] or silence
            await ws.send(chunk)
            if speed > 0:
                wait = t0 + (i + 1) * FRAME_MS / 1000 / speed - clock()
                if wait > 0:
                    await sleep(wait)
        if not task.done():
            await ws.send("")
            try:
                await asyncio.wait_for(task, 3.0)
            except asyncio.TimeoutError:
                pass
    finally:
        task.cancel()
        try:
            await task
        except BaseException:
            pass
        try:
            await ws.close()
        except Exception:
            pass
    if err:
        raise RuntimeError(err[0])
    return ends, "".join(text).strip()


# ───────────────────────── 데이터 ─────────────────────────
def _jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_turnset(split: str = "test") -> list[dict]:
    out = []
    for r in _jsonl(ROOT / "data/turnset/labels.jsonl"):
        if split and r["split"] != split:
            continue
        end = [p for p in r["points"] if p["kind"] == "end"][-1]["t"]
        pauses = [Pause(p["t"], p.get("dur_ms", 0), p.get("cut_type") or "pause")
                  for p in r["points"] if p["kind"] == "pause"]
        out.append({"uid": r["uid"], "wav": ROOT / f"data/turnset/audio/{r['uid']}.wav", "end_t": end,
                    "pauses": pauses, "mid": "intra",
                    "meta": {"type": r.get("type"), "channel": r["env"]["channel"], "noise": r["env"]["noise"]}})
    return out


def load_ts172() -> list[dict]:
    meta = json.loads((ROOT / "data/testset/audio/meta.json").read_text(encoding="utf-8"))
    cases = {c["case_id"]: c for c in _jsonl(ROOT / "data/testset/cases.jsonl")}
    out = []
    for cid, m in meta.items():
        c = cases.get(cid, {})
        tags = c.get("tags") or []
        out.append({"uid": cid, "wav": ROOT / f"data/testset/audio/{cid}.wav", "end_t": float(m["speech_end"]),
                    "pauses": [], "mid": "mid",
                    "meta": {"type": "mid_pause" if "mid_pause" in tags else "other",
                             "channel": m["env"]["channel"], "noise": m["env"]["noise"]}})
    return out


# ───────────────────────── 요약 ─────────────────────────
def summarize(rs: list[EndpointResult], group_keys: tuple[str, ...] = ("type", "channel", "noise")) -> dict:
    def one(xs: list[EndpointResult]) -> dict:
        n = len(xs)
        ok = [r.latency_ms for r in xs if r.outcome == "ok"]
        d = {"n": n,
             "false_cut": sum(r.outcome == "false_cut" for r in xs),
             "false_cut_rate": sum(r.outcome == "false_cut" for r in xs) / n if n else 0.0,
             "miss_rate": sum(r.outcome == "miss" for r in xs) / n if n else 0.0,
             "errors": sum(bool(r.error) for r in xs)}
        if ok:
            a = np.asarray(ok, float)
            d.update(latency_p50=float(np.percentile(a, 50)), latency_p95=float(np.percentile(a, 95)),
                     latency_mean=float(a.mean()))
        return d
    out = one(rs)
    where: dict[str, int] = {}
    for r in rs:
        if r.outcome == "false_cut":
            where[r.where] = where.get(r.where, 0) + 1
    out["false_cut_where"] = where
    late = [r.late_ms for r in rs if r.late_ms is not None]
    out["late_count"] = len(late)
    out["by"] = {g: {k: one([r for r in rs if str(r.meta.get(g)) == k])
                     for k in sorted({str(r.meta.get(g)) for r in rs})} for g in group_keys}
    return out


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


async def amain(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="dataset", default="turnset", choices=["turnset", "ts172"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", default="", help="시작 메시지에 더할 JSON(예: max_endpoint_delay_ms)")
    ap.add_argument("--set-env", action="append", default=[], help="KEY=VALUE (ProviderConfig 환경 덮어쓰기)")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tail", type=float, default=2.5, help="말 끝 뒤 기다릴 초")
    ap.add_argument("--window", type=float, default=1.5)
    ap.add_argument("--retries", type=int, default=1)
    a = ap.parse_args(argv)
    _load_dotenv()
    from agent import audio as au
    from agent.providers.factory import ProviderConfig, make_stt
    env = dict(os.environ)
    env["STT_PROVIDER"] = "soniox"
    for kv in a.set_env:
        k, v = kv.split("=", 1)
        env[k] = v
    stt = make_stt(ProviderConfig.from_env(env))
    extra = json.loads(a.extra) if a.extra else {}
    items = load_turnset() if a.dataset == "turnset" else load_ts172()
    if a.limit:
        items = items[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(a.concurrency)
    results: list[EndpointResult] = []

    async def one(it: dict) -> None:
        pcm, sr = au.wav_read(it["wav"])
        assert sr == SAMPLE_RATE, it["wav"]
        async with sem:
            err = ""
            for attempt in range(a.retries + 1):
                try:
                    ends, text = await stream_endpoints(stt, pcm, it["end_t"] + a.tail, extra=extra)
                    err = ""
                    break
                except Exception as e:
                    err, ends, text = str(e), [], ""
        r = classify_endpoint(it["uid"], ends, it["end_t"], it["pauses"], a.window, it["mid"])
        r.text, r.error, r.meta = text, err, it["meta"]
        results.append(r)
        print(f"{r.uid} {r.outcome:9s} lat={r.latency_ms} where={r.where} ends={r.ends} {err}", flush=True)

    await asyncio.gather(*(one(it) for it in items))
    results.sort(key=lambda r: r.uid)
    with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    s = summarize(results)
    s["config"] = {"dataset": a.dataset, "extra": extra, "window_s": a.window, "model": stt.model,
                   "url_host": stt.url.split("/")[2]}
    (out / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in s.items() if k != "by"}, ensure_ascii=False, indent=1))
    return s


def main(argv=None) -> None:
    asyncio.run(amain(argv))


if __name__ == "__main__":
    main()
