"""오프라인 LLM 평가기(AG-T10a): 오디오 없이 프롬프트·툴 루프만 잰다.

cases.jsonl의 정답 텍스트(customer_utterance)·history·caller(customer_profile)·now로
실제 시스템 프롬프트(agent.prompts.build_system)와 실제 Responder 툴 루프를 돌린다.
- TTS: 문장 텍스트만 모으는 fake(TextTTS). 첫 문장이 TTS로 넘어간 시각을 잰다.
- 툴: mock_api의 load_shop().lookup 직접 호출(서버 불필요). 정규화 훅 agent.normalize.normalize_args.
- 함수 일치: eval.score.tool_match(이름 + 키 인자 정확 일치, 자유 텍스트 인자 제외, 기대 []인데 호출하면 0).
- 옵션 --judge: eval.score.make_gemini_judge로 expected_points/must_not 점수(eval.score.judge_points).
- 동시성 제한(기본 4), 429/503 등 재시도(eval.synth.is_retryable).
결과: runs/llm_eval_<시각>/{results.jsonl, summary.json, report.md} + 콘솔 요약(범주별).
- 옵션 --stt-text(AG-T15a): 정답 텍스트에는 STT 숫자 오인식이 없으므로, 문항별 실제 STT 최종 텍스트를 LLM 입력으로 쓴다.
  파일 형식은 자동 판별: stt_eval rows.jsonl(case_id+final), 세션 턴 로그 turns-*.jsonl(session_id+user_text,
  --stt-sessions RUN_DIR로 그 실행의 세션만), {case_id: text} JSON. 심사·기록의 고객 발화는 정답 그대로 두고
  행에 input_utterance(실제 입력)·stt_missing(STT 없어 정답으로 대체)을 남긴다. 되묻기 응답 수(reprompts)도 센다.

사용:
  uv run python -m eval.llm_eval [--cases legacy/testset/cases.jsonl] [--category followup ...]
      [--ids T0001 T0002] [--limit N] [--concurrency 4] [--judge] [--model gemini-3.5-flash-lite]
      [--no-fallback] [--out runs/llm_eval_X]
      [--stt-text runs/stt_eval/final_full/rows.jsonl | runs/logs/turns-YYYYMMDD.jsonl --stt-sessions runs/R]
  uv run python -m eval.llm_eval --compare runs/A runs/B     # 두 실행의 범주별 비교표
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.config import Settings  # noqa: E402
from agent.core.types import Message, SessionInfo  # noqa: E402
from agent.normalize import normalize_args  # noqa: E402
from agent.prompts import build_system, load_manual  # noqa: E402
from agent.responder import Responder, ResponderError  # noqa: E402
from eval.score import judge_points, tool_match  # noqa: E402
from eval.synth import is_retryable  # noqa: E402

DEFAULT_CASES = ROOT / "legacy" / "testset" / "cases.jsonl"
DEFAULT_OPENER = Settings().openers[0]


# ───────────────────────── 입력 ─────────────────────────
def load_cases(path: str | Path = DEFAULT_CASES, ids: list[str] | None = None,
               categories: list[str] | None = None, limit: int | None = None) -> list[dict]:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = ROOT / p
    with open(p, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if ids:
        want = set(ids)
        rows = [r for r in rows if r.get("case_id") in want]
    if categories:
        cats = set(categories)
        rows = [r for r in rows if r.get("category") in cats]
    if limit:
        rows = rows[:limit]
    return rows


# ───────────────────────── STT 텍스트 모드(AG-T15a) ─────────────────────────
def run_session_ids(run_dir: str | Path) -> set[str]:
    """측정 실행(eval.runner) results.jsonl의 session_id 집합."""
    out: set[str] = set()
    with open(Path(run_dir) / "results.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sid = json.loads(line).get("session_id")
                if sid:
                    out.add(sid)
    return out


def _case_of_session(sid: str) -> str:
    return sid.rsplit("-", 1)[0] if "-" in sid else sid


def load_stt_texts(path: str | Path, sessions: set[str] | None = None) -> dict[str, str]:
    """문항별 STT 최종 텍스트 {case_id: text}. 형식 자동 판별(모듈 설명 참고). 빈 텍스트·오류 행은 뺀다.
    턴 로그는 sessions가 있으면 그 세션만, 한 세션에 턴이 여럿이면 이어 붙인다."""
    raw = Path(path).read_text(encoding="utf-8").strip()
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and obj and all(isinstance(v, str) for v in obj.values()):
            return {k: v.strip() for k, v in obj.items() if v.strip()}
    out: dict[str, str] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "session_id" in r and "user_text" in r and "case_id" not in r:
            sid = r["session_id"]
            if sessions is not None and sid not in sessions:
                continue
            cid, text = _case_of_session(sid), (r.get("user_text") or "").strip()
            if text:
                out[cid] = f"{out[cid]} {text}" if cid in out else text
            continue
        cid = r.get("case_id")
        if not cid or r.get("error"):
            continue
        text = next((r[k] for k in ("final", "stt_text", "text", "user_text") if isinstance(r.get(k), str)), "")
        if text.strip():
            out[cid] = text.strip()
    return out


_REPROMPT = re.compile(r"(다시|한 번 더|한번 더)\s*(한 번\s*)?(말씀|불러|알려)|한 자리씩")


def is_reprompt(text: str) -> bool:
    """응답이 번호를 다시 말해 달라는 되묻기인가(처음 묻기는 제외)."""
    return bool(_REPROMPT.search(text or ""))


def case_info(case: dict) -> SessionInfo:
    return SessionInfo(session_id=f"llm-eval-{case.get('case_id', '')}", case_id=case.get("case_id", ""),
                       now=case.get("now", ""), history=case.get("history") or [],
                       caller=case.get("customer_profile") or case.get("caller"))


def shop_tools(shop: Any) -> list[dict]:
    """GET /tools와 같은 모양(OpenAI 형식)."""
    return [{"type": "function", "function": t} for t in shop.tools]


# ───────────────────────── fake·계측 ─────────────────────────
class TextTTS:
    """TTSProvider fake: 문장 텍스트와 요청 시각만 기록한다."""

    def __init__(self, clock: Callable[[], float]):
        self.clock = clock
        self.requests: list[tuple[float, str]] = []

    async def synthesize(self, text: str):
        self.requests.append((self.clock(), text))
        yield b"\x00\x00"


class FillerCache:
    """TTSCache fake: 고정 문구(필러 등)는 캐시에 있다고 답하고, 꺼낸 시각을 기록한다."""

    def __init__(self, phrases: list[str] | tuple[str, ...], clock: Callable[[], float]):
        self.phrases = set(p for p in phrases if p)
        self.clock = clock
        self.hits: list[tuple[float, str]] = []

    def get(self, text: str) -> bytes | None:
        if text in self.phrases:
            self.hits.append((self.clock(), text))
            return b"\x00\x00"
        return None


class TimedLLM:
    """LLM 호출마다 첫 이벤트 지연과 첫 이벤트 종류를 기록한다."""

    def __init__(self, inner: Any, clock: Callable[[], float]):
        self.inner = inner
        self.clock = clock
        self.calls: list[dict] = []

    async def stream(self, system, messages, tools):
        rec = {"start": self.clock(), "first": None, "kind": None}
        self.calls.append(rec)
        async for ev in self.inner.stream(system, messages, tools):
            if rec["first"] is None:
                rec["first"] = self.clock()
                rec["kind"] = ev.type
            yield ev


class ShopCaller:
    """Mock 쇼핑몰 직접 호출 + 호출 기록(정규화된 인자, status, matched)."""

    def __init__(self, shop: Any):
        self.shop = shop
        self.calls: list[dict] = []

    async def __call__(self, name: str, args: dict) -> dict:
        args = dict(args or {})
        if name not in self.shop.tool_names:
            res, matched = {"status": "unknown_tool", "message": f"정의되지 않은 함수: {name}"}, False
        else:
            res, matched = self.shop.lookup(name, args)
        self.calls.append({"name": name, "arguments": args,
                           "status": res.get("status") if isinstance(res, dict) else None,
                           "matched": bool(matched)})
        return res


_HONORIFIC = re.compile(r"^\s*(?:네[,.]?\s*)?(?:고객님|[가-힣]{2,4}\s?고객님|안녕하세요)")


def honorific_repeat(text: str) -> bool:
    """본 응답 첫머리에 호칭(고객님, ○○○ 고객님)이나 인사(안녕하세요)를 다시 넣었는가."""
    return bool(text) and bool(_HONORIFIC.match(text))


def _nchars(s: str) -> int:
    return sum(1 for c in s if not c.isspace())


def _ms(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round((b - a) * 1000, 1)


def _retryable(e: BaseException) -> bool:
    if isinstance(e, ResponderError):
        return False
    if isinstance(e, (TimeoutError, asyncio.TimeoutError)):
        return True
    return is_retryable(e)  # type: ignore[arg-type]


# ───────────────────────── 실행 ─────────────────────────
async def run_case_once(case: dict, llm: Any, shop: Any, *, tools: list[dict] | None = None,
                        opener: str = DEFAULT_OPENER, manual: str | None = None,
                        settings: Settings | None = None, utterance: str | None = None,
                        clock: Callable[[], float] = time.perf_counter) -> dict:
    """한 문항 1회 실행. LLM 예외는 그대로 올린다(재시도는 evaluate가 한다).
    utterance를 주면 정답 텍스트 대신 그것을 고객 발화로 넣는다(STT 텍스트 모드)."""
    settings = settings or Settings()
    info = case_info(case)
    if manual is None:
        manual = load_manual(ROOT / settings.manual_path)
    system = build_system(info, opener, manual=manual)
    tts = TextTTS(clock)
    tllm = TimedLLM(llm, clock)
    caller = ShopCaller(shop)
    cache = FillerCache(settings.cached_phrases(), clock)
    raw_args: list[dict] = []

    def norm(name: str, args: dict, schema: dict, now: str) -> dict:
        raw_args.append({"name": name, "arguments": dict(args)})
        return normalize_args(name, args, schema, now)

    responder = Responder(tllm, tts, caller, tools=tools if tools is not None else shop_tools(shop),
                          settings=settings, cache=cache, normalize_args=norm, now=info.now)

    async def sink(_pcm: bytes) -> None:
        return None

    spoken: list[str] = []
    warning = None
    t0 = clock()
    try:
        utt = utterance if utterance is not None else case.get("customer_utterance", "")
        await responder.respond(system, [Message("user", utt)], sink,
                                spoken=spoken)
    except ResponderError as e:          # 빈 응답: 툴 호출은 채점한다
        warning = str(e)
    t_end = clock()
    response = " ".join(s for s in spoken if s).strip()
    first = tllm.calls[0] if tllm.calls else {}
    row = {
        "tool_calls": caller.calls,
        "raw_tool_args": raw_args,
        "response": response,
        "agent_text": f"{opener} {response}".strip() if opener else response,
        "spoken": spoken,
        "sentences": len(spoken),
        "response_chars": _nchars(response),
        "honorific_repeat": honorific_repeat(response),
        "reprompt": is_reprompt(response),
        "rounds": len(tllm.calls),
        "llm_calls": [{"first_ms": _ms(c["start"], c["first"]), "kind": c["kind"]} for c in tllm.calls],
        "llm_first_event_ms": _ms(first.get("start"), first.get("first")),
        "first_sentence_ms": _ms(t0, tts.requests[0][0]) if tts.requests else None,
        "first_audio_ms": _ms(t0, min([t for t, _ in tts.requests[:1]] + [t for t, _ in cache.hits[:1]]))
        if (tts.requests or cache.hits) else None,
        "filler": bool(cache.hits),
        "total_ms": _ms(t0, t_end),
        "model": getattr(llm, "last_model", None) or getattr(llm, "model", None),
    }
    if warning:
        row["warning"] = warning
    return row


_CASE_FIELDS = ("case_id", "category", "tags", "now", "history", "customer_profile", "customer_utterance",
                "expected_tools", "expected_points", "must_not")


def _score_row(row: dict, exclude_free_text: bool = True) -> None:
    sc: dict = {"tool": None, "tool_exact": None, "judge": None}
    if "error" not in row and row.get("expected_tools") is not None:
        m = tool_match(row["expected_tools"],
                       [{"name": c["name"], "arguments": c["arguments"]} for c in row.get("tool_calls") or []],
                       exclude_free_text)
        sc.update(tool=m["score"], tool_exact=m["exact"], tool_missing=m["missing"], tool_extra=m["extra"],
                  tool_name_match=m["name_match"])
    row["scores"] = sc


async def _judge_row(row: dict, judge_fn: Callable[[dict], dict]) -> None:
    sc = row["scores"]
    try:
        v = judge_fn(row)
        if asyncio.iscoroutine(v):
            v = await v
        sc["judge"] = judge_points(v, len(row.get("expected_points") or []))
        if v.get("comment"):
            sc["judge_comment"] = v["comment"]
        if v.get("violations"):
            sc["judge_violations"] = v["violations"]
        if v.get("points") is not None:
            sc["judge_points"] = v["points"]
    except Exception as e:  # noqa: BLE001
        sc["judge"] = None
        sc["judge_error"] = repr(e)[:300]


async def evaluate(cases: list[dict], llm_factory: Callable[[], Any], *, shop: Any = None,
                   tools: list[dict] | None = None, judge_fn: Callable[[dict], Any] | None = None,
                   concurrency: int = 4, retries: int = 3, sleep: Callable[[float], Any] = asyncio.sleep,
                   opener: str = DEFAULT_OPENER, manual: str | None = None, settings: Settings | None = None,
                   judge_in_thread: bool = False, progress: Callable[[dict], None] | None = None,
                   stt_texts: dict[str, str] | None = None,
                   clock: Callable[[], float] = time.perf_counter) -> list[dict]:
    """문항들을 동시성 제한으로 돌려 행 목록(입력 순서)을 돌려준다. llm_factory는 문항마다 새 LLM.
    stt_texts가 있으면 문항별 STT 텍스트를 LLM 입력으로 쓴다(없는 문항은 정답 텍스트, stt_missing 표시)."""
    if shop is None:
        from mock_api.server import load_shop
        shop = load_shop()
    tools = tools if tools is not None else shop_tools(shop)
    settings = settings or Settings()
    if manual is None:
        manual = load_manual(ROOT / settings.manual_path)
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(case: dict) -> dict:
        async with sem:
            row: dict = {k: case.get(k) for k in _CASE_FIELDS}
            utt = None
            if stt_texts is not None:
                utt = stt_texts.get(case.get("case_id"))
                row["stt_missing"] = utt is None
                if utt is None:
                    utt = case.get("customer_utterance", "")
            row["input_utterance"] = utt if utt is not None else case.get("customer_utterance", "")
            attempt = 0
            while True:
                attempt += 1
                try:
                    res = await run_case_once(case, llm_factory(), shop, tools=tools, opener=opener,
                                              manual=manual, settings=settings, utterance=utt, clock=clock)
                    row.update(res)
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt <= retries and _retryable(e):
                        await sleep(min(30.0, 2.0 * 2 ** (attempt - 1)))
                        continue
                    row["error"] = repr(e)[:300]
                    row.setdefault("tool_calls", [])
                    break
            row["attempts"] = attempt
            _score_row(row)
            if judge_fn is not None and "error" not in row:
                if judge_in_thread:
                    await _judge_row(row, lambda r: asyncio.to_thread(judge_fn, r))
                else:
                    await _judge_row(row, judge_fn)
        if progress is not None:
            progress(row)
        return row

    return list(await asyncio.gather(*(one(c) for c in cases)))


# ───────────────────────── 집계 ─────────────────────────
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    i = min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))
    return xs[i]


def _stats(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    sc = [r.get("scores") or {} for r in ok]
    fe = [r.get("llm_first_event_ms") for r in ok]
    fs = [r.get("first_sentence_ms") for r in ok]
    fa = [r.get("first_audio_ms") for r in ok]
    return {
        "n": len(rows),
        "errors": len(rows) - len(ok),
        "tool": _mean([s.get("tool") for s in sc]),
        "tool_exact": _mean([None if s.get("tool_exact") is None else float(s["tool_exact"]) for s in sc]),
        "extra_calls": sum(1 for s in sc if s.get("tool_extra")),
        "missing_calls": sum(1 for s in sc if s.get("tool_missing")),
        "judge": _mean([s.get("judge") for s in sc]),
        "judged": sum(1 for s in sc if s.get("judge") is not None),
        "chars": _mean([r.get("response_chars") for r in ok]),
        "sentences": _mean([r.get("sentences") for r in ok]),
        "rounds": _mean([r.get("rounds") for r in ok]),
        "honorific_repeat": sum(1 for r in ok if r.get("honorific_repeat")),
        "reprompts": sum(1 for r in ok if r.get("reprompt")),
        "llm_first_event_p50": _pct(fe, 0.5), "llm_first_event_p95": _pct(fe, 0.95),
        "first_sentence_p50": _pct(fs, 0.5), "first_sentence_p95": _pct(fs, 0.95),
        "first_audio_p50": _pct(fa, 0.5), "first_audio_p95": _pct(fa, 0.95),
        "fillers": sum(1 for r in ok if r.get("filler")),
    }


def summarize(rows: list[dict]) -> dict:
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(str(r.get("category") or "unknown"), []).append(r)
    return {"overall": _stats(rows), "by_category": {k: _stats(v) for k, v in sorted(groups.items())}}


def _f(v, nd=1):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


COLS = [("n", "n", 0), ("errors", "오류", 0), ("tool", "함수", 1), ("tool_exact", "함수일치율", 2),
        ("extra_calls", "여분호출", 0), ("missing_calls", "누락", 0), ("judge", "응대", 1),
        ("chars", "글자", 0), ("sentences", "문장", 1), ("honorific_repeat", "호칭반복", 0), ("reprompts", "되묻기", 0),
        ("llm_first_event_p50", "LLM첫p50", 0), ("llm_first_event_p95", "LLM첫p95", 0),
        ("first_sentence_p50", "첫문장p50", 0), ("first_sentence_p95", "첫문장p95", 0),
        ("first_audio_p50", "첫소리p50", 0), ("first_audio_p95", "첫소리p95", 0), ("fillers", "필러", 0)]


def render_summary(summary: dict) -> str:
    lines = ["| 범주 | " + " | ".join(c[1] for c in COLS) + " |", "|---|" + "---|" * len(COLS)]
    for k, st in [("전체", summary["overall"]), *summary["by_category"].items()]:
        lines.append(f"| {k} | " + " | ".join(_f(st.get(c[0]), c[2]) for c in COLS) + " |")
    return "\n".join(lines) + "\n"


def _failures(rows: list[dict]) -> list[str]:
    out = []
    for r in rows:
        sc = r.get("scores") or {}
        why = []
        if "error" in r:
            why.append(f"오류 {r['error'][:80]}")
        if sc.get("tool_exact") is False:
            fmt = lambda xs: ", ".join(f"{m['name']}{json.dumps(m.get('arguments') or {}, ensure_ascii=False)}"  # noqa: E731
                                       for m in xs) or "-"
            why.append(f"함수(누락 {fmt(sc.get('tool_missing') or [])} / 여분 {fmt(sc.get('tool_extra') or [])})")
        if sc.get("judge") is not None and sc["judge"] < 50:
            why.append(f"응대 {sc['judge']:.0f} {sc.get('judge_comment', '')}".rstrip())
        if r.get("honorific_repeat"):
            why.append("호칭반복")
        if why:
            resp = (r.get("response") or "").replace("|", "/")[:120]
            out.append(f"| {r.get('case_id')} | {r.get('category')} | {'; '.join(why)} | {resp} |")
    return out


def render_report(rows: list[dict], summary: dict, meta: dict | None = None) -> str:
    L = ["# LLM 오프라인 평가", ""]
    if meta:
        L += [f"- {k}: {v}" for k, v in meta.items()] + [""]
    L += ["## 범주별", "", render_summary(summary)]
    fails = _failures(rows)
    L += [f"## 감점 문항 ({len(fails)}건)", "", "| case | 범주 | 사유 | 응답 |", "|---|---|---|---|", *fails]
    return "\n".join(L) + "\n"


def write_run(out_dir: str | Path, rows: list[dict], meta: dict | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = summarize(rows)
    (out / "summary.json").write_text(json.dumps({**summary, "meta": meta or {}}, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    (out / "report.md").write_text(render_report(rows, summary, meta), encoding="utf-8")
    return out


def compare(a: dict, b: dict, keys=("tool", "tool_exact", "judge", "chars", "reprompts", "honorific_repeat",
                                    "llm_first_event_p50", "first_sentence_p50")) -> str:
    """summary.json 두 개 → 범주별 'A → B' 표."""
    cats = ["전체", *sorted(set(a["by_category"]) | set(b["by_category"]))]
    get = lambda s, c: s["overall"] if c == "전체" else s["by_category"].get(c, {})  # noqa: E731
    lines = ["| 범주 | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for c in cats:
        cells = []
        for k in keys:
            nd = 2 if k == "tool_exact" else (0 if k.endswith(("p50", "repeat", "chars", "reprompts")) else 1)
            cells.append(f"{_f(get(a, c).get(k), nd)} → {_f(get(b, c).get(k), nd)}")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ───────────────────────── CLI ─────────────────────────
def _default_llm_factory(model: str | None, no_fallback: bool) -> Callable[[], Any]:
    from agent.providers._gemini import make_gemini_client
    from agent.providers.factory import ProviderConfig, make_llm_with_fallback
    over: dict = {}
    if model:
        over["gemini_llm_model"] = model
    if no_fallback:
        over["llm_fallback_models"] = "none"
    cfg = ProviderConfig.from_env(**over)
    client = make_gemini_client(cfg.gemini_api_key)
    return lambda: make_llm_with_fallback(cfg, gemini_client=client)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="오프라인 LLM 평가(프롬프트·툴 루프)")
    ap.add_argument("--cases", default=str(DEFAULT_CASES))
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--category", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--judge", action="store_true", help="Gemini 응대 심사(expected_points/must_not)")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--model", default=None, help="기본 LLM 모델(기본: GEMINI_LLM_MODEL 또는 gemini-3.5-flash-lite)")
    ap.add_argument("--no-fallback", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"))
    ap.add_argument("--stt-text", default=None,
                    help="문항별 STT 텍스트 파일(stt_eval rows.jsonl / 턴 로그 / {case_id: text} JSON)을 LLM 입력으로")
    ap.add_argument("--stt-sessions", default=None, help="턴 로그일 때 이 측정 실행(results.jsonl)의 세션만 쓴다")
    a = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    if a.compare:
        sa, sb = (json.loads((Path(p) / "summary.json").read_text(encoding="utf-8")) for p in a.compare)
        print(compare(sa, sb))
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except Exception:  # noqa: BLE001
        pass

    cases = load_cases(a.cases, ids=a.ids, categories=a.category, limit=a.limit)
    stt_texts = None
    if a.stt_text:
        sids = run_session_ids(a.stt_sessions) if a.stt_sessions else None
        stt_texts = load_stt_texts(a.stt_text, sessions=sids)
        miss = [c["case_id"] for c in cases if c["case_id"] not in stt_texts]
        print(f"STT 텍스트 {len(stt_texts)}개 로드, 문항 중 없음 {len(miss)}개(정답 텍스트로 대체)", flush=True)
    factory = _default_llm_factory(a.model, a.no_fallback)
    judge_fn = None
    if a.judge:
        from eval.score import JUDGE_MODEL, make_gemini_judge
        judge_fn = make_gemini_judge(a.judge_model or JUDGE_MODEL)
    out = Path(a.out) if a.out else ROOT / "runs" / f"llm_eval_{datetime.now():%Y%m%d_%H%M%S}"
    done = {"n": 0}

    def progress(row: dict) -> None:
        done["n"] += 1
        sc = row.get("scores") or {}
        mark = "E" if "error" in row else ("O" if sc.get("tool_exact") else "X")
        print(f"[{done['n']}/{len(cases)}] {row['case_id']} {row.get('category')} {mark}"
              f" judge={_f(sc.get('judge'), 0)} | {(row.get('response') or row.get('error') or '')[:70]}",
              flush=True)

    t = time.perf_counter()
    rows = asyncio.run(evaluate(cases, factory, judge_fn=judge_fn, concurrency=a.concurrency,
                                retries=a.retries, judge_in_thread=True, progress=progress, stt_texts=stt_texts))
    meta = {"label": a.label, "cases": str(a.cases), "n": len(rows), "model": a.model or "default",
            "fallback": not a.no_fallback, "judge": bool(a.judge), "concurrency": a.concurrency,
            "stt_text": a.stt_text or "", "stt_sessions": a.stt_sessions or "",
            "stt_missing": sum(1 for r in rows if r.get("stt_missing")),
            "wall_s": round(time.perf_counter() - t, 1), "time": f"{datetime.now():%Y-%m-%d %H:%M}",
            "llm_calls": sum(r.get("rounds") or 0 for r in rows),
            "served": dict(sorted({m: sum(1 for r in rows if r.get("model") == m)
                                   for m in {r.get("model") for r in rows}}.items(), key=lambda kv: str(kv[0])))}
    write_run(out, rows, meta)
    print()
    print(render_summary(summarize(rows)))
    print(f"결과: {out}")


if __name__ == "__main__":
    main()
