"""Provider 실측 벤치(AG-T04). 실제 Gemini API를 호출한다.

    uv run python scripts/bench_providers.py                # 전체, 결과 → docs/decisions/bench_providers.md
    uv run python scripts/bench_providers.py --only tts --n 5

- STT: Gemini TTS로 한국어 문장 5개를 합성(16k) → 실시간 속도로 스트리밍 → 마지막 음성 직후 finalize
       → 최종 텍스트까지 지연 + CER. live/batch 둘 다.
- TTS: 짧은 문장 TTFB·전체 시간, 모델 3종, 16k 직접 출력 옵션 동작 여부, Interactions API 비교.
- LLM: 가짜 툴 14개 + 질문(툴 5 + 일반 2)으로 첫 이벤트 지연, 툴 호출 결정 지연, 정확도, 툴 결과 왕복.
API 키는 환경변수(GEMINI_API_KEY)에서만 읽고 출력하지 않는다.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import platform
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from agent.core.types import FRAME_BYTES, Message, ToolCall  # noqa: E402
from agent.providers._gemini import make_gemini_client  # noqa: E402
from agent.providers.edge_tts import EdgeTTS  # noqa: E402
from agent.providers.gemini_llm import GeminiLLM  # noqa: E402
from agent.providers.gemini_stt import GeminiBatchSTT, GeminiLiveSTT  # noqa: E402
from agent.providers.gemini_tts import GeminiTTS  # noqa: E402

STT_SENTENCES = [
    "주문한 운동화가 아직 안 왔는데 언제 도착하나요?",
    "어제 산 바지를 교환하고 싶어요.",
    "배송지를 회사 주소로 바꿀 수 있을까요?",
    "환불은 보통 며칠 걸리나요?",
    "상담원이랑 직접 통화하고 싶습니다.",
]
TTS_SENTENCES = [
    "네, 확인해 드릴게요.",                      # 11
    "주문하신 상품은 내일 도착 예정입니다.",       # 20
    "잠시만 기다려 주시겠어요?",                  # 14
    "반품 접수가 완료되었습니다.",                # 15
    "고객님, 주문번호를 말씀해 주시겠어요?",       # 20
]
TTS_MODELS = ["gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts", "gemini-2.5-flash-preview-tts"]
LLM_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.5-flash"]


def _fn(name, desc, props, req=None):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": {k: {"type": "string", "description": v}
                                                        for k, v in props.items()},
                       "required": req if req is not None else list(props)}}}


TOOLS = [
    _fn("get_member_info", "전화번호로 회원 정보 조회", {"phone_number": "하이픈 없는 휴대폰 번호"}),
    _fn("get_order_list", "전화번호로 최근 주문 목록 조회", {"phone_number": "하이픈 없는 휴대폰 번호"}),
    _fn("get_order_info", "주문번호(숫자 10자리)로 주문 상세 조회", {"order_id": "주문번호 10자리"}),
    _fn("track_delivery", "주문번호로 배송 위치·도착 예정일 조회", {"order_id": "주문번호 10자리"}),
    _fn("cancel_order", "주문 취소", {"order_id": "주문번호", "reason": "취소 사유"}),
    _fn("request_return", "반품 접수", {"order_id": "주문번호", "reason": "반품 사유"}),
    _fn("request_exchange", "교환 접수", {"order_id": "주문번호", "reason": "교환 사유"}),
    _fn("change_shipping_address", "배송지 변경", {"order_id": "주문번호", "new_address": "새 주소"}),
    _fn("get_refund_status", "환불 진행 상태 조회", {"order_id": "주문번호"}),
    _fn("get_product_info", "상품 정보 조회", {"product_name": "상품명"}),
    _fn("check_stock", "상품 옵션 재고 확인", {"product_name": "상품명", "option": "색상·사이즈 등"}),
    _fn("get_coupons", "보유 쿠폰 조회", {"phone_number": "하이픈 없는 휴대폰 번호"}),
    _fn("request_callback", "상담원 콜백 예약", {"phone_number": "휴대폰 번호", "preferred_time": "희망 시간"}),
    _fn("transfer_to_agent", "상담원 연결", {"reason": "연결 사유"}),
]
SYSTEM = ("너는 온라인 쇼핑몰 '디렉트몰'의 한국어 음성 고객센터 상담원이다. 지금은 2026-11-21 11:00이다. "
          "고객 질문에 1~2문장으로 짧고 공손하게 답한다. 조회·접수가 필요하면 반드시 제공된 함수를 호출하고, "
          "필요한 정보(주문번호, 전화번호 등)가 없으면 함수를 부르지 말고 되묻는다. "
          "영업시간은 평일 9시~18시, 반품은 수령 후 7일 이내 가능하다. 숫자는 아라비아 숫자로 함수 인자에 넣는다.")
LLM_CASES = [
    ("주문번호 2611150231 배송 어디쯤 왔어요?", {"track_delivery", "get_order_info"}),
    ("제 번호 01012345678로 주문한 거 목록 좀 보여주세요.", {"get_order_list"}),
    ("주문번호 2611150231 취소해 주세요. 단순 변심이에요.", {"cancel_order"}),
    ("검정 후드티 L 사이즈 재고 있어요?", {"check_stock"}),
    ("그냥 상담원 연결해 주세요.", {"transfer_to_agent"}),
    ("영업시간이 어떻게 되나요?", set()),
    ("반품하고 싶은데요.", set()),   # 주문번호 없음 → 되묻기
]


# ── 통계 ─────────────────────────────────────────────
def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def fmt(x, unit="s"):
    if x is None:
        return "-"
    return f"{x:.2f}" if unit == "s" else f"{x:.1%}"


def stat_row(xs):
    xs = [x for x in xs if x is not None]
    return fmt(pct(xs, 50)), fmt(pct(xs, 95)), (fmt(statistics.mean(xs)) if xs else "-"), len(xs)


def cer(ref: str, hyp: str) -> float:
    norm = lambda s: re.sub(r"[\s\.,\?!~·]", "", s)
    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / len(r)


async def with_timeout(coro, t):
    try:
        return await asyncio.wait_for(coro, t)
    except Exception as e:  # noqa: BLE001
        return e


# ── STT ──────────────────────────────────────────────
async def synth16(client, text: str, voice: str, model: str = "gemini-3.8-flash-tts") -> bytes:
    tts = GeminiTTS(client=client, model=model, voice=voice)
    return b"".join([c async for c in tts.synthesize(text)])


async def edge16(text: str, voice: str) -> bytes:
    """edge-tts(MP3) → ffmpeg 16k PCM. Gemini TTS 할당량이 없을 때 STT 테스트 음성용."""
    import subprocess

    import edge_tts
    buf = bytearray()
    async for ch in edge_tts.Communicate(text, voice).stream():
        if ch.get("type") == "audio" and ch.get("data"):
            buf += ch["data"]
    p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", "s16le",
                        "-ac", "1", "-ar", "16000", "pipe:1"], input=bytes(buf), capture_output=True, check=True)
    return p.stdout


def is_quota(e) -> bool:
    return "429" in str(e)[:60] or "RESOURCE_EXHAUSTED" in str(e)


async def stream_realtime(stream, pcm: bytes, lead_ms=300):
    silence = b"\x00" * FRAME_BYTES
    frames = [silence] * (lead_ms // 20) + [pcm[i:i + FRAME_BYTES].ljust(FRAME_BYTES, b"\x00")
                                            for i in range(0, len(pcm), FRAME_BYTES)]
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    first_partial = None
    speech_start = t0 + lead_ms / 1000
    for i, f in enumerate(frames):
        await asyncio.sleep(max(0.0, t0 + i * 0.02 - loop.time()))
        await stream.send_audio(f)
        if first_partial is None and stream.latest_text():
            first_partial = loop.time() - speech_start
    return first_partial


async def silence_feeder(stream, stop: asyncio.Event):
    silence = b"\x00" * FRAME_BYTES
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    i = 0
    while not stop.is_set():
        await asyncio.sleep(max(0.0, t0 + i * 0.02 - loop.time()))
        await stream.send_audio(silence)
        i += 1


async def bench_stt(client, n: int, log) -> dict:
    voices = ["Puck", "Leda", "Charon", "Aoede", "Fenrir"]
    edge_voices = ["ko-KR-InJoonNeural", "ko-KR-SunHiNeural"]
    log("STT: 테스트 음성 합성 중...")
    audios, sources = [], []
    for i, s in enumerate(STT_SENTENCES):
        pcm = b""
        for model in ("gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"):
            try:
                pcm = await synth16(client, s, voices[i % len(voices)], model)
                sources.append(f"{model}:{voices[i % len(voices)]}")
                break
            except Exception as e:  # noqa: BLE001
                log(f"  {model} 합성 실패({str(e)[:40]}), 다음")
        if not pcm:
            v = edge_voices[i % 2]
            pcm = await edge16(s, v)
            sources.append(f"edge:{v}")
        audios.append(pcm)
    log(f"  합성 완료: {[round(len(a) / 32000, 2) for a in audios]}초 {sources}")
    res: dict = {"sources": sources}
    for kind in ("live", "batch"):
        stt = (GeminiLiveSTT(client=client) if kind == "live" else GeminiBatchSTT(client=client))
        rows = []
        reps = max(1, round(n / len(STT_SENTENCES))) if kind == "live" else 1   # batch는 할당량 절약
        for r in range(reps):
            for i, (ref, pcm) in enumerate(zip(STT_SENTENCES, audios)):
                t = time.perf_counter()
                try:
                    stream = await stt.open_stream()
                except Exception as e:  # noqa: BLE001
                    rows.append({"err": f"open: {e}"[:200]})
                    continue
                connect = time.perf_counter() - t
                fp = await stream_realtime(stream, pcm)
                t_end = time.perf_counter()
                stop = asyncio.Event()
                feeder = asyncio.create_task(silence_feeder(stream, stop))
                text = await with_timeout(stream.finalize(timeout=5.0), 30)
                lat = time.perf_counter() - t_end
                stop.set()
                await feeder
                await stream.close()
                if isinstance(text, Exception):
                    rows.append({"err": str(text)[:200]})
                    continue
                c = cer(ref, text)
                rows.append({"connect": connect, "first_partial": fp, "final_latency": lat,
                             "cer": c, "ref": ref, "hyp": text})
                log(f"  {kind} {r}-{i}: final {lat:.2f}s partial {fp if fp is None else round(fp, 2)} "
                    f"CER {c:.2f} | {text}")
        res[kind] = rows
    return res


# ── TTS ──────────────────────────────────────────────
async def probe_16k(client, model: str) -> str:
    out = []
    try:
        s = await client.aio.interactions.create(
            model=model, input=[{"type": "user_input", "content": [{"type": "text", "text": "네."}]}],
            response_format={"type": "audio", "mime_type": "audio/l16", "sample_rate": 16000},
            generation_config={"speech_config": [{"voice": "Kore"}]}, stream=True)
        async for ev in s:
            if ev.event_type == "step.delta" and getattr(ev.delta, "type", None) == "audio":
                out.append(f"OK rate={ev.delta.sample_rate}")
                break
    except Exception as e:  # noqa: BLE001
        out.append("거부: " + str(e).split("\n")[-1][:90])
    try:
        from google.genai import types
        cfg = types.GenerateContentConfig(response_modalities=["AUDIO"], response_mime_type="audio/l16;rate=16000")
        await client.aio.models.generate_content(model=model, contents="네.", config=cfg)
        out.append("generateContent OK")
    except Exception as e:  # noqa: BLE001
        out.append("generateContent 거부(400)" if "400" in str(e) else f"generateContent 오류 {str(e)[:60]}")
    return " / ".join(out)


async def interactions_ttfb(client, model: str, text: str) -> float | None:
    t = time.perf_counter()
    try:
        s = await client.aio.interactions.create(
            model=model, input=[{"type": "user_input", "content": [{"type": "text", "text": text}]}],
            response_format={"type": "audio"}, generation_config={"speech_config": [{"voice": "Kore"}]},
            stream=True)
        async for ev in s:
            if ev.event_type == "step.delta" and getattr(ev.delta, "type", None) == "audio":
                base64.b64decode(ev.delta.data)
                return time.perf_counter() - t
    except Exception:  # noqa: BLE001
        return None
    return None


async def bench_tts(client, n: int, log) -> dict:
    res: dict = {"models": {}, "probe16k": {}, "interactions": [], "style": {}}
    for model in ["edge:ko-KR-SunHiNeural", "edge:ko-KR-InJoonNeural"] + TTS_MODELS:
        if model.startswith("edge:"):
            tts = EdgeTTS(voice=model.split(":", 1)[1])
        else:
            tts = GeminiTTS(client=client, model=model, voice="Kore")
        rows = []
        for i in range(n):
            text = TTS_SENTENCES[i % len(TTS_SENTENCES)]
            t = time.perf_counter()
            ttfb = None
            nbytes = 0
            nchunks = 0
            try:
                async def run():
                    nonlocal ttfb, nbytes, nchunks
                    async for c in tts.synthesize(text):
                        if ttfb is None:
                            ttfb = time.perf_counter() - t
                        nbytes += len(c)
                        nchunks += 1
                await asyncio.wait_for(run(), 30)
                total = time.perf_counter() - t
                rows.append({"text": text, "ttfb": ttfb, "total": total, "audio_s": nbytes / 32000,
                             "chunks": nchunks})
                log(f"  {model} {i}: TTFB {ttfb:.2f}s total {total:.2f}s audio {nbytes / 32000:.2f}s chunks {nchunks}")
            except Exception as e:  # noqa: BLE001
                rows.append({"text": text, "err": str(e)[:200]})
                log(f"  {model} {i}: ERR {str(e)[:120]}")
                if is_quota(e):
                    log(f"  {model}: 할당량 초과 → 이 모델 중단")
                    break
        res["models"][model] = rows
        if model.startswith("edge:"):
            res["probe16k"][model] = "해당 없음(MP3→ffmpeg 16k)"
        elif any("err" not in r for r in rows):
            pr = await with_timeout(probe_16k(client, model), 30)
            res["probe16k"][model] = pr if isinstance(pr, str) else f"시간 초과/오류 {str(pr)[:60]}"
        else:
            res["probe16k"][model] = "할당량 초과로 미측정"
        log(f"  16k probe {model}: {res['probe16k'][model]}")
    for i in range(3):
        v = await with_timeout(interactions_ttfb(client, "gemini-3.8-flash-tts", TTS_SENTENCES[i]), 20)
        v = v if isinstance(v, float) else None
        if v is None:
            res["interactions"].append(None)
            break
        res["interactions"].append(v)
        log(f"  interactions TTFB: {v}")
    # 스타일 지시가 음성으로 읽히는지: 길이 비교 + 전사 확인
    text = "네, 확인해 드릴게요."
    ok_models = [m for m, rows in res["models"].items()
                 if not m.startswith("edge:") and any("err" not in r for r in rows)]
    if not ok_models:
        res["style"] = {"skipped": "할당량 없음"}
        return res
    sm = ok_models[0]
    try:
        plain = await synth16(client, text, "Kore", sm)
        styled_tts = GeminiTTS(client=client, model=sm, voice="Kore",
                               style="친절하고 차분한 고객센터 상담원 말투로 말하세요")
        styled = b"".join([c async for c in styled_tts.synthesize(text)])
        stream = await GeminiLiveSTT(client=client).open_stream()
        await stream_realtime(stream, styled, lead_ms=0)
        tr = await stream.finalize(timeout=5.0)
        await stream.close()
        res["style"] = {"model": sm, "plain_s": len(plain) / 32000, "styled_s": len(styled) / 32000,
                        "styled_transcript": tr}
    except Exception as e:  # noqa: BLE001
        res["style"] = {"skipped": str(e)[:100]}
    log(f"  style: {res['style']}")
    return res


# ── LLM ──────────────────────────────────────────────
async def llm_once(llm: GeminiLLM, question: str, timeout: float):
    t = time.perf_counter()
    first = first_tool = None
    calls: list[ToolCall] = []
    text = ""

    async def run():
        nonlocal first, first_tool, text
        async for ev in llm.stream(SYSTEM, [Message(role="user", content=question)], TOOLS):
            now = time.perf_counter() - t
            if ev.type in ("text", "tool_call") and first is None:
                first = now
            if ev.type == "tool_call":
                calls.append(ev.tool_call)
                if first_tool is None:
                    first_tool = now
            elif ev.type == "text":
                text += ev.text
    r = await with_timeout(run(), timeout)
    total = time.perf_counter() - t
    return {"first": first, "first_tool": first_tool, "total": total, "calls": calls, "text": text,
            "err": (str(r)[:200] or type(r).__name__) if isinstance(r, Exception) else None}


async def llm_roundtrip(llm: GeminiLLM, timeout: float) -> str:
    """툴 호출 → 결과 → 다시 생성(thought_signature 처리 확인)."""
    q = LLM_CASES[0][0]
    first = await llm_once(llm, q, timeout)
    if not first["calls"]:
        return "툴 호출 없음"
    tc = first["calls"][0]
    msgs = [Message("user", q), Message("assistant", first["text"], tool_calls=[tc]),
            Message("tool", json.dumps({"status": "배송중", "eta": "2026-11-22"}, ensure_ascii=False),
                    tool_call_id=tc.id, name=tc.name)]
    t = time.perf_counter()
    text = ""
    try:
        async def run():
            nonlocal text
            async for ev in llm.stream(SYSTEM, msgs, TOOLS):
                if ev.type == "text":
                    text += ev.text
        await asyncio.wait_for(run(), timeout)
        return f"OK {time.perf_counter() - t:.2f}s: {text.strip()[:60]}"
    except Exception as e:  # noqa: BLE001
        return f"ERR {str(e)[:120]}"


async def bench_llm(client, n: int, log, timeout: float, models: list[str] | None = None) -> dict:
    res: dict = {}
    reps = max(1, n // len(LLM_CASES))   # 무료 티어 할당량 때문에 기본 1회
    for model in models or LLM_MODELS:
        llm = GeminiLLM(client=client, model=model)
        rows = []
        fails = 0
        for r in range(reps):
            for q, expected in LLM_CASES:
                if fails >= 3:
                    rows.append({"q": q, "err": "skipped(연속 실패)"})
                    continue
                o = await llm_once(llm, q, timeout)
                names = {c.name for c in o["calls"]}
                ok = (bool(names & expected) if expected else not names)
                rows.append({"q": q, "expected": sorted(expected), "names": sorted(names),
                             "args": [c.arguments for c in o["calls"]], "ok": ok, **{k: o[k] for k in
                             ("first", "first_tool", "total", "text", "err")}})
                fails = fails + 1 if o["err"] else 0
                if o["err"] and is_quota(o["err"]):
                    fails = 3
                log(f"  {model} {r}: first {o['first']} tool {o['first_tool']} ok={ok} "
                    f"{sorted(names)} {o['text'][:40]!r} {o['err'] or ''}")
        rt = await llm_roundtrip(llm, timeout)
        log(f"  {model} roundtrip: {rt}")
        res[model] = {"rows": rows, "roundtrip": rt,
                      "thinking": llm._thinking_idx.get(model)}
    return res


# ── 보고서 ───────────────────────────────────────────
def report(res: dict, n: int) -> str:
    L = ["# Provider 실측 (AG-T04)", "",
         f"- 측정: {datetime.now():%Y-%m-%d %H:%M} · `scripts/bench_providers.py --n {n}` · "
         f"{platform.system()} · 키 `GEMINI_API_KEY`(무료/유료 티어 미확인)",
         "- 단위: 초. p50/p95는 성공한 시도 기준. 네트워크 위치: 개발 PC(한국 가정용 회선 추정).", ""]
    if "stt" in res:
        L += ["## STT (한국어 합성 음성 5문장, 실시간 스트리밍)", "",
              "지연 = 마지막 음성 프레임 송신 → `finalize()` 반환(최종 텍스트). batch는 이 시간에 전사 요청 전체가 들어간다.", "",
              "| 방식 | 모델 | final p50 | final p95 | 평균 | 성공 | 첫 부분결과 p50 | 연결 p50 | CER 평균 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for kind, model in (("live", "gemini-3.5-transcribe-live"), ("batch", "gemini-3.5-transcribe")):
            rows = res["stt"].get(kind, [])
            ok = [r for r in rows if "err" not in r]
            p50, p95, mean, cnt = stat_row([r["final_latency"] for r in ok])
            fp = fmt(pct([r["first_partial"] for r in ok if r["first_partial"] is not None], 50))
            con = fmt(pct([r["connect"] for r in ok], 50))
            c = f"{statistics.mean([r['cer'] for r in ok]):.3f}" if ok else "-"
            L.append(f"| {kind} | `{model}` | {p50} | {p95} | {mean} | {cnt}/{len(rows)} | {fp} | {con} | {c} |")
        L += ["", f"- 테스트 음성 출처: {res['stt'].get('sources')}", "", "<details><summary>전사 결과</summary>", ""]
        for kind in ("live", "batch"):
            for r in res["stt"].get(kind, []):
                L.append(f"- {kind}: " + (f"`{r['hyp']}` (CER {r['cer']:.2f})" if "err" not in r else f"ERR {r['err']}"))
        L += ["", "</details>", ""]
    if "tts" in res:
        t = res["tts"]
        L += ["## TTS (짧은 문장 11~20자. edge: MP3→ffmpeg 16k 스트림 / Gemini: `generate_content_stream` + 24k→16k soxr)", "",
              "| 모델 | TTFB p50 | TTFB p95 | 전체 p50 | 전체 p95 | 성공 | 오디오 평균(초) | 16k 직접 출력 |",
              "|---|---|---|---|---|---|---|---|"]
        for model, rows in t["models"].items():
            ok = [r for r in rows if "err" not in r]
            if not ok and rows and all(is_quota(r.get("err", "")) for r in rows):
                L.append(f"| `{model}` | 한도 소진 | | | | 0/{len(rows)} | | {t['probe16k'].get(model, '-')} |")
                continue
            a = stat_row([r["ttfb"] for r in ok])
            b = stat_row([r["total"] for r in ok])
            au = f"{statistics.mean([r['audio_s'] for r in ok]):.2f}" if ok else "-"
            L.append(f"| `{model}` | {a[0]} | {a[1]} | {b[0]} | {b[1]} | {len(ok)}/{len(rows)} | {au} | {t['probe16k'].get(model, '-')} |")
        it = [x for x in t["interactions"] if x is not None]
        L += ["", f"- Interactions API 스트림(`gemini-3.8-flash-tts`) TTFB: "
              + (", ".join(f"{x:.2f}" for x in it) if it else "실패"),
              f"- 스타일 지시(프롬프트 앞에 `스타일: 텍스트`): {t['style']}", ""]
    if "llm" in res:
        L += ["## LLM (가짜 툴 14개, 질문 7개: 툴 5 + 툴 없음 2, 스트리밍)", "",
              "첫 이벤트 = 요청 → 첫 텍스트 또는 툴 호출. 툴 결정 = 요청 → 첫 tool_call(툴 질문만). 정확도 = 기대 함수 포함(툴 없음 질문은 호출 안 함).", "",
              "| 모델 | 첫 이벤트 p50 | p95 | 툴 결정 p50 | p95 | 정확도 | 성공 | thinking | 툴 결과 왕복 |",
              "|---|---|---|---|---|---|---|---|---|"]
        from agent.providers.gemini_llm import THINKING_CANDIDATES
        for model, d in res["llm"].items():
            rows = [r for r in d["rows"] if not r.get("err")]
            a = stat_row([r["first"] for r in rows])
            b = stat_row([r["first_tool"] for r in rows if r["expected"]])
            acc = f"{sum(r['ok'] for r in rows) / len(rows):.0%}" if rows else "-"
            th = THINKING_CANDIDATES[d["thinking"]] if d["thinking"] is not None else "-"
            L.append(f"| `{model}` | {a[0]} | {a[1]} | {b[0]} | {b[1]} | {acc} | {len(rows)}/{len(d['rows'])} | {th} | {d['roundtrip'][:50]} |")
        L += ["", "<details><summary>툴 호출 상세</summary>", ""]
        for model, d in res["llm"].items():
            for r in d["rows"]:
                L.append(f"- `{model}` {r['q']} → {r.get('names')} {r.get('args')} "
                         f"{'OK' if r.get('ok') else 'X'} {r.get('err') or ''} {(r.get('text') or '')[:40]!r}")
        L += ["", "</details>", ""]
    L += ["## 권장", "", "(측정 후 작성)", ""]
    return "\n".join(L)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--only", choices=["stt", "tts", "llm"], action="append")
    ap.add_argument("--out", default=str(ROOT / "docs/decisions/bench_providers.md"))
    ap.add_argument("--raw", default="", help="원시 결과 JSON 경로(선택)")
    ap.add_argument("--llm-timeout", type=float, default=30.0)
    ap.add_argument("--llm-models", default="", help="쉼표 구분(기본: 후보 4종)")
    a = ap.parse_args()
    only = set(a.only or ["stt", "tts", "llm"])
    client = make_gemini_client()

    def log(s):
        print(s, flush=True)

    res: dict = {}
    if "tts" in only:
        log("== TTS ==")
        res["tts"] = await bench_tts(client, a.n, log)
    if "llm" in only:
        log("== LLM ==")
        res["llm"] = await bench_llm(client, a.n, log, a.llm_timeout,
                                     [m for m in a.llm_models.split(",") if m] or None)
    if "stt" in only:
        log("== STT ==")
        res["stt"] = await bench_stt(client, a.n, log)
    md = report(res, a.n)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(md, encoding="utf-8")
    if a.raw:
        Path(a.raw).write_text(json.dumps(res, ensure_ascii=False, default=str, indent=1), encoding="utf-8")
    log(f"→ {a.out}")


if __name__ == "__main__":
    asyncio.run(main())
