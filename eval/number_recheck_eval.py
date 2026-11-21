"""AG-T46 숫자 2차 인식 오프라인 비교: 번호를 말하는 연습 문항 음성을 Soniox(stt-rt-v6)에 여러 방식으로 다시 넣어
정답 번호(reference 함수 인자의 주문번호·전화번호)를 그대로 받아 적는 비율을 잰다. 실제 API를 부른다(BAZE 경유).

변형:
  base   전체 음성, 기존 문맥(쇼핑몰 상담)            — 실시간 1차와 같은 조건
  ctx    전체 음성, 숫자 문맥(번호 형식 안내)
  crop   1차 토큰 시각으로 번호 구간만 잘라(앞뒤 여유) + 숫자 문맥
  pad    crop + 앞뒤 무음 0.5초 + 음량 정규화
  slow   crop을 0.8배속(WSOLA, 음높이 유지) + 숫자 문맥
  slow2  crop을 0.65배속 + 숫자 문맥

사용: uv run python -m eval.number_recheck_eval --cases runs/number_cases.jsonl \
        --audio directon-2026-voice-cs/data/audio/practice --out runs/t45_offline.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import wave
from pathlib import Path

import numpy as np

from agent.providers.stt_recheck import digit_runs

SR = 16000
ID_KEYS = ("order_id", "order_number", "buyer_cellphone", "customer_phone_number", "phone_number")
BASE_CTX = {"general": [{"key": "domain", "value": "쇼핑몰 고객센터 상담"}]}
DIGIT_CTX = {
    "general": [{"key": "domain", "value": "쇼핑몰 고객센터 전화 상담"},
                {"key": "topic", "value": "고객이 휴대폰 번호나 주문번호를 불러 준다"}],
    "text": ("고객은 휴대폰 번호(010으로 시작하는 11자리, 예: 010-1234-5678)나 주문번호(날짜 8자리와 7자리, "
             "예: 20261001-0005987)를 숫자로 끊어 읽는다. 숫자는 아라비아 숫자로 받아 적는다."),
    "terms": ["010", "공일공", "주문번호", "전화번호", "휴대폰 번호"],
}


def digits(v) -> str:
    return re.sub(r"\D", "", str(v or ""))


def truth_numbers(case: dict) -> list[str]:
    out = []
    for tc in (case.get("reference") or {}).get("tool_calls") or []:
        for k in ID_KEYS:
            d = digits((tc.get("arguments") or {}).get(k))
            if len(d) >= 7 and d not in out:
                out.append(d)
    return out


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)


def to_pcm(x: np.ndarray) -> bytes:
    return np.clip(x, -32768, 32767).astype(np.int16).tobytes()


def wsola(x: np.ndarray, rate: float, win: int = 512, tol: int = 128) -> np.ndarray:
    """간단한 WSOLA 시간 늘이기(rate < 1이면 느리게, 음높이 유지)."""
    hs = win // 2
    ha = hs * rate
    w = np.hanning(win).astype(np.float32)
    n_out = int(len(x) / rate) + win
    y = np.zeros(n_out, np.float32)
    norm = np.zeros(n_out, np.float32)
    xp = np.pad(x, (tol, win + tol))
    prev = None
    k = 0
    while True:
        a = int(k * ha) + tol
        if a + win + tol > len(xp):
            break
        if prev is not None and prev + hs + win > len(xp):
            break
        if prev is None:
            best = a
        else:
            ref = xp[prev + hs: prev + hs + win]
            cands = range(a - tol, a + tol + 1, 4)
            best = max(cands, key=lambda s: float(np.dot(xp[s:s + win], ref)))
        o = k * hs
        if o + win > n_out:
            break
        y[o:o + win] += xp[best:best + win] * w
        norm[o:o + win] += w
        prev = best
        k += 1
    norm[norm < 1e-3] = 1.0
    return (y / norm)[: int(len(x) / rate)]


async def transcribe(pcm: bytes, context: dict, url: str, key: str) -> tuple[str, list[dict]]:
    import websockets
    async with websockets.connect(url, additional_headers={"Authorization": f"Bearer {key}"},
                                  max_size=2 ** 22) as ws:
        await ws.send(json.dumps({"model": "stt-rt-v6", "audio_format": "pcm_s16le", "sample_rate": SR,
                                  "num_channels": 1, "language_hints": ["ko"], "context": context},
                                 ensure_ascii=False))
        tokens: list[dict] = []

        async def rx():
            while True:
                res = json.loads(await ws.recv())
                if res.get("error_code"):
                    raise RuntimeError(f"{res.get('error_code')} {res.get('error_message')}")
                for t in res.get("tokens", []):
                    if t.get("is_final") and t.get("text") not in ("<end>", "<fin>"):
                        tokens.append(t)
                if res.get("finished"):
                    return

        task = asyncio.create_task(rx())
        step = SR // 10 * 2                      # 100ms
        for i in range(0, len(pcm), step):
            await ws.send(pcm[i:i + step])
            await asyncio.sleep(0.02)            # 5배속 전송
        await ws.send("")
        await asyncio.wait_for(task, 30)
    return "".join(t["text"] for t in tokens).strip(), tokens


def number_span(tokens: list[dict], margin_ms: int = 400) -> tuple[int, int] | None:
    """숫자가 든 토큰들의 (시작, 끝) ms. 숫자 토큰이 없으면 None."""
    hits = [t for t in tokens if re.search(r"[0-9공영일이삼사오육칠팔구]", t.get("text", ""))
            and t.get("start_ms") is not None]
    hits = [t for t in hits if re.search(r"\d", t["text"])] or hits
    if not hits:
        return None
    return max(0, hits[0]["start_ms"] - margin_ms), hits[-1]["end_ms"] + margin_ms


def score(text: str, truths: list[str]) -> dict:
    runs = digit_runs(text)
    alld = digits(text)
    return {"exact": all(t in runs for t in truths), "contained": all(t in alld for t in truths), "runs": runs}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="base,ctx,crop,pad,slow,slow2")
    args = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv()
    key = os.environ.get("BAZE_API_KEY", "")
    from agent.providers.baze_tts import BAZE_BASE_URL
    base = (os.environ.get("BAZE_BASE_URL") or BAZE_BASE_URL).rstrip("/")
    url = "wss://" + (base + "/soniox/transcribe-websocket").split("://", 1)[1]
    variants = args.variants.split(",")
    cases = [json.loads(l) for l in open(args.cases, encoding="utf-8")]
    out = open(args.out, "w", encoding="utf-8")
    totals = {v: [0, 0, 0] for v in variants}
    for c in cases:
        truths = truth_numbers(c)
        if not truths:
            continue
        x = read_wav(Path(args.audio) / f"{c['case_id']}.wav")
        text0, toks = await transcribe(to_pcm(x), BASE_CTX, url, key)
        span = number_span(toks)
        crop = x[int(span[0] * SR / 1000): int(span[1] * SR / 1000)] if span else x
        row = {"case_id": c["case_id"], "truth": truths, "span": span}
        for v in variants:
            if v == "base":
                text = text0
            else:
                if v == "ctx":
                    audio = x
                elif v == "crop":
                    audio = crop
                elif v == "pad":
                    peak = float(np.max(np.abs(crop))) or 1.0
                    sil = np.zeros(SR // 2, np.float32)
                    audio = np.concatenate([sil, crop * (20000.0 / peak), sil])
                elif v == "slow":
                    audio = wsola(crop, 0.8)
                elif v == "slow2":
                    audio = wsola(crop, 0.65)
                else:
                    raise SystemExit(f"unknown variant {v}")
                try:
                    text, _ = await transcribe(to_pcm(audio), DIGIT_CTX, url, key)
                except Exception as e:  # noqa: BLE001
                    text = f"<error {e!r}>"
            s = score(text, truths)
            totals[v][0] += s["exact"]
            totals[v][1] += s["contained"]
            totals[v][2] += 1
            row[v] = {"text": text, **s}
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
        print(c["case_id"], truths, " | ".join(f"{v}:{'O' if row[v]['exact'] else ('o' if row[v]['contained'] else 'x')}"
                                              for v in variants), flush=True)
    print("\nvariant  exact  contained  n")
    for v, (e, ct, n) in totals.items():
        print(f"{v:7s} {e:3d}/{n}  {ct:3d}/{n}")


if __name__ == "__main__":
    asyncio.run(main())
