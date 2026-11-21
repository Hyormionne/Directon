"""오프라인 STT 평가기(AG-T10b).

테스트셋 오디오를 실시간 속도(--speed, 0이면 대기 없음)로 STT 스트림에 20ms씩 흘리고,
실제 세션처럼 말 끝(meta.speech_end) 뒤 침묵 200ms 지점에서 finalize를 부른다.
finalize 동안에도 남은 오디오(꼬리 무음·잡음)를 계속 보낸다(세션도 확정 전까지 보낸다).

지표: CER(원문 / 숫자 정규화), 숫자 정확도(digits 태그), finalize 지연 p50/p95,
말 끝 시점·finalize 시점 latest_text 빈 비율, 첫 부분 결과 시각, 채널·잡음별 표.

사용:
  uv run python -m eval.stt_eval --out runs/stt_eval/<이름> [--tags digits] [--sample 40 --seed 1]
      [--speed 1.0] [--concurrency 3] [--set STT_DIGIT_RECHECK=1 --set GEMINI_STT_LIVE_MODEL=...]
      [--context "주문번호,휴대폰 번호"] [--stt gemini_batch] [--only T0001,T0002]
      [--cases legacy/testset/cases.jsonl --audio legacy/testset/audio]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import sys
import time
import wave
from pathlib import Path
from typing import Any, Callable

from agent.normalize import _parse_number_token, _strip_suffix, spoken_digits
from agent.providers.stt_recheck import digit_runs

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
FRAME_MS = 20
FRAME_BYTES = SR * 2 * FRAME_MS // 1000
KEY_ARGS = ("phone_number", "order_id")


# ── 텍스트 지표 ─────────────────────────────────────
def _digitize(text: str) -> str:
    out = []
    for tok in (text or "").split():
        core = re.sub(r"[^\w]+$", "", re.sub(r"^[^\w]+", "", tok))
        stem = _strip_suffix(core)
        d = _parse_number_token(stem) if stem else None
        if d is not None:
            tok = d + core[len(stem):]
        else:
            # "010-1234-5678"처럼 구분자로 이어진 숫자
            parts = re.split(r"[-.·/]", core)
            if len(parts) > 1 and all(p.isdigit() for p in parts if p):
                tok = "".join(parts)
        out.append(tok)
    return " ".join(out)


def norm_text(text: str, norm_digits: bool = False) -> str:
    if norm_digits:
        text = _digitize(text)
    return re.sub(r"[\W_]+", "", text or "").lower()


def _edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str, norm_digits: bool = False) -> float:
    r, h = norm_text(ref, norm_digits), norm_text(hyp, norm_digits)
    if not r:
        return 0.0 if not h else 1.0
    return _edit_distance(r, h) / len(r)


# ── 숫자 정확도 ─────────────────────────────────────
def digit_targets(case: dict) -> list[str]:
    out: list[str] = []
    for t in case.get("expected_tools") or []:
        for k in KEY_ARGS:
            v = (t.get("arguments") or {}).get(k)
            if v and str(v) not in out:
                out.append(str(v))
    return out


def _has(text: str, target: str) -> bool:
    return target in digit_runs(text)


def digit_correct(case: dict, hyp: str) -> bool | None:
    """digits 태그 문항만 판정. 정답 인자가 정답 텍스트에 이어 읽혀 있으면 그 숫자열이 그대로 있는지,
    아니면(정정으로 갈라짐·기대 함수 없음) 정답 텍스트의 숫자 전체를 그대로 재현했는지."""
    if "digits" not in (case.get("tags") or []):
        return None
    ref = case.get("customer_utterance", "")
    targets = digit_targets(case)
    if targets and all(_has(hyp, t) for t in targets):
        return True
    if targets and all(_has(ref, t) for t in targets):
        return False
    return (spoken_digits(hyp) or "") == (spoken_digits(ref) or "")


# ── 집계 ────────────────────────────────────────────
def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return float(xs[lo] + (xs[hi] - xs[lo]) * (k - lo))


def _group(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("error")]
    dig = [r for r in ok if r.get("digit_ok") is not None]
    fin = [r["finalize_ms"] for r in ok if r.get("finalize_ms") is not None]
    rech = [r for r in ok if (r.get("recheck") or {}).get("triggered")]
    n = len(ok)
    mean = (lambda k: statistics.fmean(r[k] for r in ok) if n else None)
    return {
        "n": n,
        "errors": len(rows) - n,
        "cer": mean("cer"),
        "cer_norm": mean("cer_norm"),
        "digit_n": len(dig),
        "digit_acc": (sum(1 for r in dig if r["digit_ok"]) / len(dig)) if dig else None,
        "fin_p50": _pct(fin, 0.5),
        "fin_p95": _pct(fin, 0.95),
        "empty_at_end": (sum(1 for r in ok if not (r.get("text_at_end") or "").strip()) / n) if n else None,
        "empty_at_prep": (sum(1 for r in ok if not (r.get("text_at_prep") or "").strip()) / n)
        if n and all("text_at_prep" in r for r in ok) else None,
        "recheck_n": len(rech),
        "recheck_replaced": sum(1 for r in rech if r["recheck"].get("replaced")),
    }


def summarize(rows: list[dict]) -> dict:
    out = {"overall": _group(rows), "by_channel": {}, "by_noise": {}}
    for key, name in (("channel", "by_channel"), ("noise", "by_noise")):
        vals = sorted({r.get(key) for r in rows if r.get(key)})
        out[name] = {v: _group([r for r in rows if r.get(key) == v]) for v in vals}
    return out


def _f(v: Any, pct: bool = False, ms: bool = False) -> str:
    if v is None:
        return "-"
    if pct:
        return f"{v * 100:.1f}%"
    if ms:
        return f"{v:.0f}"
    return f"{v:.3f}"


def render_markdown(summary: dict, title: str = "STT 평가") -> str:
    head = ("| 구분 | n | 오류 | CER | CER(숫자정규화) | 숫자 정확도 | finalize p50/p95 ms | 말 끝 빈 부분결과 "
            "| finalize 시점 빈 부분결과 | 재인식(교체) |\n|---|---|---|---|---|---|---|---|---|---|\n")

    def row(name: str, g: dict) -> str:
        dig = f"{_f(g['digit_acc'], pct=True)} ({g['digit_n']})" if g["digit_n"] else "-"
        return (f"| {name} | {g['n']} | {g['errors']} | {_f(g['cer'])} | {_f(g['cer_norm'])} | {dig} | "
                f"{_f(g['fin_p50'], ms=True)} / {_f(g['fin_p95'], ms=True)} | {_f(g['empty_at_end'], pct=True)} | "
                f"{_f(g['empty_at_prep'], pct=True)} | {g['recheck_n']} ({g['recheck_replaced']}) |\n")

    md = f"# {title}\n\n" + head + row("전체", summary["overall"])
    for name, label in (("by_channel", "채널"), ("by_noise", "잡음")):
        for k, g in summary[name].items():
            md += row(f"{label}:{k}", g)
    return md


# ── 재생 ────────────────────────────────────────────
async def run_case(stt: Any, pcm: bytes, speech_end: float, *, speed: float = 1.0, prep_ms: int = 200,
                   finalize_timeout: float = 1.0, context: str = "") -> dict:
    """한 문항을 흘리고 결과(dict)를 돌려준다. speed=0이면 대기 없이 보낸다."""
    stream = await stt.open_stream(context)
    # 정확히 speech_end+prep 바이트까지 보낸 뒤 finalize(프레임 경계와 무관하게)
    end_bytes = int(speech_end * SR) * 2
    prep_bytes = int((speech_end + prep_ms / 1000) * SR) * 2
    loop = asyncio.get_running_loop()
    t_start = loop.time()
    sent = 0          # 송신 바이트(파일보다 길면 무음으로 채운다)
    i = 0
    res: dict = {"text_at_end": None, "text_at_prep": "", "first_partial_ms": None}

    async def send_next(limit: int) -> None:
        nonlocal sent, i
        if speed > 0:
            delay = t_start + i * FRAME_MS / 1000 / speed - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
        n = min(FRAME_BYTES, limit - sent)
        chunk = pcm[sent:sent + n]
        chunk += bytes(n - len(chunk))
        await stream.send_audio(chunk)
        sent += n
        i += 1
        if res["first_partial_ms"] is None and stream.latest_text().strip():
            res["first_partial_ms"] = sent / 2 / SR * 1000

    try:
        # 1) speech_end + prep 까지
        while sent < prep_bytes:
            await send_next(prep_bytes)
            if res["text_at_end"] is None and sent >= end_bytes:
                res["text_at_end"] = stream.latest_text()
        if res["text_at_end"] is None:
            res["text_at_end"] = stream.latest_text()
        res["text_at_prep"] = stream.latest_text()
        # 2) finalize + 남은 오디오 계속 송신(실시간 재생일 때)
        t0 = time.perf_counter()
        fin = asyncio.create_task(stream.finalize(finalize_timeout))
        while speed > 0 and not fin.done() and sent < len(pcm):
            await send_next(len(pcm))
        final = await fin
        res["finalize_ms"] = (time.perf_counter() - t0) * 1000
        res["final"] = (final or "").strip()
        rc = getattr(stream, "last_recheck", None)
        if rc is not None:
            res["recheck"] = dict(rc)
        ev = getattr(stream, "events", None)
        if ev is not None:
            res["events"] = list(ev)[-40:]
        return res
    finally:
        try:
            await stream.close()
        except Exception:  # noqa: BLE001
            pass


def score_row(case: dict, res: dict) -> dict:
    env = case.get("env") or {}
    ref = case.get("customer_utterance", "")
    hyp = res.get("final", "")
    row = {
        "case_id": case["case_id"], "channel": env.get("channel"), "noise": env.get("noise"),
        "voice": case.get("voice"), "tags": case.get("tags"), "ref": ref, **res,
        "cer": cer(ref, hyp), "cer_norm": cer(ref, hyp, norm_digits=True),
        "digit_ok": digit_correct(case, hyp), "digit_targets": digit_targets(case), "error": None,
    }
    rc = res.get("recheck") or {}
    if rc.get("triggered") and "live_text" in rc:
        row["digit_ok_live"] = digit_correct(case, rc["live_text"])
    return row


async def evaluate(stt: Any, cases: list[dict], load: Callable[[str], tuple[bytes, float]], *,
                   speed: float = 1.0, concurrency: int = 3, retries: int = 2, retry_wait: float = 2.0,
                   prep_ms: int = 200, finalize_timeout: float = 1.0, context: str = "",
                   log: Callable[[str], None] | None = None) -> list[dict]:
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(case: dict) -> dict:
        async with sem:
            pcm, speech_end = load(case["case_id"])
            err = None
            for attempt in range(retries + 1):
                try:
                    res = await run_case(stt, pcm, speech_end, speed=speed, prep_ms=prep_ms,
                                         finalize_timeout=finalize_timeout, context=context)
                    row = score_row(case, res)
                    row["attempts"] = attempt + 1
                    if log:
                        log(f"{case['case_id']} cer={row['cer_norm']:.2f} dig={row['digit_ok']} "
                            f"fin={row['finalize_ms']:.0f}ms end={row['text_at_end']!r} -> {row['final']!r}")
                    return row
                except Exception as e:  # noqa: BLE001
                    err = repr(e)
                    if attempt < retries:
                        await asyncio.sleep(retry_wait * (attempt + 1))
            env = case.get("env") or {}
            if log:
                log(f"{case['case_id']} ERROR {err}")
            return {"case_id": case["case_id"], "channel": env.get("channel"), "noise": env.get("noise"),
                    "tags": case.get("tags"), "error": err}

    return list(await asyncio.gather(*(one(c) for c in cases)))


# ── CLI ─────────────────────────────────────────────
def load_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return w.readframes(w.getnframes())


def select_cases(cases: list[dict], audio_dir: Path, meta: dict, *, only: str = "", tags: str = "",
                 sample: int = 0, seed: int = 0) -> list[dict]:
    have = [c for c in cases if c["case_id"] in meta and (audio_dir / f"{c['case_id']}.wav").exists()]
    if only:
        ids = {x.strip() for x in only.split(",") if x.strip()}
        return [c for c in have if c["case_id"] in ids]
    sel = have
    if tags:
        want = {t.strip() for t in tags.split(",") if t.strip()}
        tagged = [c for c in have if want & set(c.get("tags") or [])]
        rest = [c for c in have if c not in tagged]
        sel = tagged
        if sample:
            sel = tagged + random.Random(seed).sample(rest, min(sample, len(rest)))
    elif sample:
        sel = random.Random(seed).sample(have, min(sample, len(have)))
    return sorted(sel, key=lambda c: c["case_id"])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", default=str(ROOT / "legacy/testset/cases.jsonl"))
    p.add_argument("--audio", default=str(ROOT / "legacy/testset/audio"))
    p.add_argument("--out", required=True)
    p.add_argument("--only", default="")
    p.add_argument("--tags", default="", help="이 태그 문항 전부(+ --sample이면 나머지에서 무작위 표본)")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--prep-ms", type=int, default=200)
    p.add_argument("--finalize-timeout", type=float, default=1.0)
    p.add_argument("--context", default="", help="세션 stt_context와 같음(쉼표 구분 어휘)")
    p.add_argument("--stt", default="", help="STT_PROVIDER 덮어쓰기(gemini_live|gemini_batch|soniox)")
    p.add_argument("--set", action="append", default=[], help="환경변수 덮어쓰기 KEY=VALUE (ProviderConfig)")
    p.add_argument("--label", default="")
    return p.parse_args(argv)


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
    args = parse_args(argv)
    _load_dotenv()
    from agent.providers.factory import ProviderConfig, make_stt

    env = dict(os.environ)
    for kv in args.set:
        k, v = kv.split("=", 1)
        env[k] = v
    if args.stt:
        env["STT_PROVIDER"] = args.stt
    cfg = ProviderConfig.from_env(env)
    stt = make_stt(cfg)
    audio_dir = Path(args.audio)
    meta = json.loads((audio_dir / "meta.json").read_text(encoding="utf-8"))
    with open(args.cases, encoding="utf-8") as f:
        cases = [json.loads(l) for l in f if l.strip()]
    sel = select_cases(cases, audio_dir, meta, only=args.only, tags=args.tags, sample=args.sample,
                       seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(sel)} cases → {out}", flush=True)

    def load(cid: str) -> tuple[bytes, float]:
        return load_wav(audio_dir / f"{cid}.wav"), float(meta[cid]["speech_end"])

    t0 = time.time()
    rows = await evaluate(stt, sel, load, speed=args.speed, concurrency=args.concurrency,
                          retries=args.retries, prep_ms=args.prep_ms, finalize_timeout=args.finalize_timeout,
                          context=args.context, log=lambda s: print(s, flush=True))
    with open(out / "rows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = summarize(rows)
    summary["config"] = {"set": args.set, "stt": args.stt, "context": args.context, "speed": args.speed,
                         "cases": len(sel), "label": args.label, "elapsed_s": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    md = render_markdown(summary, title=args.label or out.name)
    (out / "report.md").write_text(md, encoding="utf-8")
    print(md)
    return summary


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(amain(argv))


if __name__ == "__main__":
    main()
