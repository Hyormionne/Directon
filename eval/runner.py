"""Starter Kit 측정기(`DIRECT_ON 2026 Starter Kit/eval/run_eval.py`) 복사·확장판. 원본은 수정하지 않는다.

실행 예:
  uv run python -m eval.runner --server ws://127.0.0.1:9000 \
    --cases legacy/testset/cases.jsonl --audio legacy/testset/audio \
    --mock http://127.0.0.1:8100 --out runs/X [--limit N] [--concurrency 4] [--only T0001,T0002]

원본과 같은 프로토콜·시간 계산(20ms 실시간 송신, RMS≥300 첫 프레임 − speech_end, 25초 꼬리).
추가 기록: agent_text, events 전체, tool_calls(name·arguments·matched·t_response), after_tool_sound·after_tool_ms
(마지막 함수 결과 → 그 뒤 첫 유성 프레임, AG-T21),
turn_end, first_sound, first_frame, spoke_before_end, agent_audio_sec, 그리고 cases.jsonl의
category/tags/env/expected_*/customer_utterance/voice. 파일 입출력은 UTF-8.
"""
import argparse, asyncio, json, os, time, uuid, wave
import numpy as np
import httpx, websockets

RATE = 16000
FRAME = 320                 # 20ms
SILENCE_RMS = 300           # 이보다 작은 프레임은 무음으로 본다
TAIL_TIMEOUT = 25.0         # 말 끝 뒤 최대 대기(초)
CASE_FIELDS = ("category", "tags", "env", "expected_tools", "expected_points", "must_not",
               "customer_utterance", "voice", "voice_style", "segments", "history", "customer_profile")


def load_wav(path):
    with wave.open(path) as w:
        assert w.getframerate() == RATE and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return w.readframes(w.getnframes())


def rms(b):
    a = np.frombuffer(b, dtype=np.int16).astype(np.float32)
    return float(np.sqrt((a * a).mean())) if len(a) else 0.0


def summarize(agent_frames, events, calls=None, wall_speech_end=0.0):
    """측정 결과 요약(순수 함수).

    agent_frames: [(speech_end 기준 상대시각, bytes)], events: [{"t": 상대시각, ...}],
    calls: Mock /calls 응답 목록(t_response는 벽시계), wall_speech_end: speech_end의 벽시계 시각.
    """
    out = {}
    voiced = [t for t, b in agent_frames if rms(b) >= SILENCE_RMS]
    out["first_sound"] = voiced[0] if voiced else None
    out["first_frame"] = agent_frames[0][0] if agent_frames else None
    out["spoke_before_end"] = bool(voiced and voiced[0] < 0)
    out["turn_end"] = next((e["t"] for e in events if e.get("type") == "turn.end"), None)
    out["agent_text"] = " ".join(e.get("text", "") for e in events if e.get("type") == "agent.transcript")
    out["events"] = events
    out["agent_audio_sec"] = sum(len(b) for _, b in agent_frames) / 2 / RATE
    calls = calls or []
    out["tool_calls"] = [{"name": x.get("name"), "arguments": x.get("arguments"), "matched": x.get("matched"),
                          "t_response": (x["t_response"] - wall_speech_end) if x.get("t_response") is not None
                          else None} for x in calls]
    times = [c["t_response"] for c in out["tool_calls"] if c["t_response"] is not None]
    if times:
        last = max(times)
        after = [t for t in voiced if t > last]
        out["after_tool_sound"] = (after[0] - last) if after else None
        # AG-T21: 공식 속도 규칙(함수 문항 = 첫 소리 점수와 "함수 결과 → 답" 점수의 평균)용 ms 표기
        out["after_tool_ms"] = round(out["after_tool_sound"] * 1000, 1) if after else None
    return out


def attach_case(row, case):
    """cases.jsonl의 채점용 필드를 결과 행에 붙인다(측정값은 덮어쓰지 않는다)."""
    for k in CASE_FIELDS:
        if k in case and k not in row:
            row[k] = case[k]
    return row


async def run_case(case, meta, args):
    sid = f"{case['case_id']}-{uuid.uuid4().hex[:8]}"
    pcm = load_wav(os.path.join(args.audio, case["case_id"] + ".wav"))
    speech_end_off = meta["speech_end"]          # 음성 파일 안에서 말이 끝나는 시각(초)
    out = {"case_id": case["case_id"], "session_id": sid, "speech_end": speech_end_off}
    agent_frames = []                            # (도착시각, bytes)
    events = []
    speech_end = None
    try:
        async with websockets.connect(args.server, max_size=None, open_timeout=10) as ws:
            await ws.send(json.dumps({
                "type": "session.start", "session_id": sid, "case_id": case["case_id"],
                "now": case["now"], "history": case["history"], "caller": case.get("customer_profile"), "tools_url": args.mock,
                "audio": {"encoding": "pcm_s16le", "sample_rate": RATE, "channels": 1}}, ensure_ascii=False))
            t0 = time.perf_counter()
            speech_end = t0 + speech_end_off
            done = asyncio.Event()

            async def sender():
                silence = b"\0" * (FRAME * 2)
                k = 0
                while not done.is_set():
                    i = k * FRAME * 2
                    chunk = pcm[i:i + FRAME * 2] if i < len(pcm) else silence   # 말이 끝나도 전화선처럼 무음을 계속 보낸다
                    if len(chunk) < FRAME * 2:
                        chunk += b"\0" * (FRAME * 2 - len(chunk))
                    await ws.send(chunk)
                    k += 1
                    wait = t0 + k * 0.02 - time.perf_counter()
                    if wait > 0:
                        await asyncio.sleep(wait)
                    if time.perf_counter() > speech_end + TAIL_TIMEOUT:
                        done.set()

            async def receiver():
                async for m in ws:
                    now = time.perf_counter()
                    if isinstance(m, bytes):
                        agent_frames.append((now, m))
                    else:
                        e = json.loads(m)
                        events.append({"t": now - speech_end, **e})
                        if e.get("type") == "turn.end":
                            done.set()
                            return

            st = asyncio.create_task(sender())
            rt = asyncio.create_task(receiver())
            await done.wait()
            st.cancel(); rt.cancel()
    except Exception as e:
        out["error"] = repr(e)[:300]
        out["events"] = events
        return attach_case(out, case)

    # 함수 호출 기록 (모의 서버)
    wall_se = time.time() - (time.perf_counter() - speech_end)            # perf → 벽시계 환산
    calls = []
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            calls = (await c.get(args.mock.rstrip("/") + "/calls", params={"session_id": sid})).json()
    except Exception as e:
        out["tool_calls_error"] = repr(e)[:300]
    out.update(summarize([(t - speech_end, b) for t, b in agent_frames], events, calls, wall_se))

    # 답 음성 저장
    os.makedirs(os.path.join(args.out, "agent_audio"), exist_ok=True)
    with wave.open(os.path.join(args.out, "agent_audio", case["case_id"] + ".wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes(b"".join(b for _, b in agent_frames))
    return attach_case(out, case)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--mock", default="http://localhost:8100")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--only", default="", help="쉼표로 구분한 case_id")
    return ap.parse_args(argv)


async def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    with open(args.cases, encoding="utf-8") as f:
        cases = [json.loads(l) for l in f if l.strip()]
    with open(os.path.join(args.audio, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    cases = [c for c in cases if c["case_id"] in meta]
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    if only:
        cases = [c for c in cases if c["case_id"] in only]
    path = os.path.join(args.out, "results.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {json.loads(l)["case_id"] for l in f if l.strip()}
    todo = [c for c in cases if c["case_id"] not in done][: args.limit or None]
    sem = asyncio.Semaphore(args.concurrency)

    async def one(c):
        async with sem:
            return await run_case(c, meta[c["case_id"]], args)

    with open(path, "a", encoding="utf-8") as f:
        for fut in asyncio.as_completed([one(c) for c in todo]):
            r = await fut
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            fs = r.get("first_sound")
            if "error" in r:
                print(f"{r['case_id']}: ERROR {r['error']}", flush=True)
            else:
                timing = f"첫 소리 {fs:.2f}s" if fs is not None else "소리 없음"
                if "after_tool_ms" in r:
                    at = r["after_tool_ms"]
                    timing += f" | 조회 뒤 {at / 1000:.2f}s" if at is not None else " | 조회 뒤 소리 없음"
                flag = " [말 끝 전 발화]" if r.get("spoke_before_end") else ""
                print(f"{r['case_id']}: {timing}{flag} | 함수 {len(r.get('tool_calls', []))}개", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
