"""AG-T05a: 사람이 쓴 시나리오 원본 → legacy/testset/cases.jsonl (docs/AGENT_PLAN.md 3.2~3.4).

사용:
  uv run python scripts/gen_scenarios.py            # cases.jsonl 생성
  uv run python scripts/gen_scenarios.py --stats    # 분포표(마크다운)만 출력

- 발화 내용·기대 함수·기대 응대는 `legacy/testset/scenarios_src.yaml`에 사람이 직접 쓴다.
- 이 스크립트는 case_id(파일 순서대로 T0001~), voice, env(3.3절), 기본 now를 seed 고정 균형 배분으로 붙인다.
- customer_profile은 mock_data 회원 정보에서 채운다({"name","phone","email":null}).
- 외부 호출 없음. 같은 원본 → 같은 출력(tests/test_testset.py가 확인).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "legacy" / "testset" / "scenarios_src.yaml"
OUT = ROOT / "legacy" / "testset" / "cases.jsonl"
MOCK = ROOT / "legacy" / "shop_mock" / "mock_data.json"
SEED = 20261120

# gemini:Kore는 상담원 음성이라 쓰지 않는다(과적합 방지).
VOICES = [
    "gemini:Puck", "gemini:Charon", "gemini:Fenrir", "gemini:Aoede", "gemini:Leda",
    "gemini:Orus", "gemini:Zephyr", "gemini:Despina",
    "edge:ko-KR-SunHiNeural", "edge:ko-KR-InJoonNeural", "edge:ko-KR-HyunsuMultilingualNeural",
]
# (값, 비중) — 비중대로 개수를 정한 뒤 섞는다.
ENV_AXES = {
    "channel": [("wideband", 1), ("phone", 1), ("phone_ulaw", 1)],
    "noise": [("none", 0.36), ("white", 0.16), ("pink", 0.16), ("babble", 0.16), ("hum", 0.16)],
    "snr_db": [(30, 0.25), (20, 0.30), (10, 0.30), (5, 0.15)],
    "gain_db": [(0, 0.6), (-12, 0.2), (6, 0.2)],
    "speed": [(1.0, 0.6), (0.85, 0.2), (1.2, 0.2)],
    "packet_loss": [(0.0, 0.88), (0.02, 0.12)],
}
LEAD_MIN, LEAD_MAX = 200, 1500
DEFAULT_DATE = "2026-11-20"   # 데이터 기준일(금). 기본 now는 이 날 영업시간(점심 제외) 중에서 고른다.


def _rng(axis: str) -> random.Random:
    return random.Random(f"{SEED}:{axis}")


def balanced(choices, n: int, rng: random.Random) -> list:
    """비중에 맞춰 개수를 정하고(최대 잔여 방식) 섞은 길이 n 목록."""
    total = sum(w for _, w in choices)
    raw = [w / total * n for _, w in choices]
    counts = [int(r) for r in raw]
    order = sorted(range(len(choices)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in order[: n - sum(counts)]:
        counts[i] += 1
    out = [v for (v, _), c in zip(choices, counts) for _ in range(c)]
    rng.shuffle(out)
    return out


def lead_values(n: int, rng: random.Random) -> list[int]:
    if n == 1:
        return [LEAD_MIN]
    vals = [int(round((LEAD_MIN + (LEAD_MAX - LEAD_MIN) * i / (n - 1)) / 10) * 10) for i in range(n)]
    rng.shuffle(vals)
    return vals


def default_times(n: int, rng: random.Random) -> list[str]:
    slots = [f"{DEFAULT_DATE} {h:02d}:{m:02d}" for h in range(9, 18) if h != 12 for m in range(5, 60, 5)]
    return [rng.choice(slots) for _ in range(n)]


def voice_values(n: int, rng: random.Random) -> list[str]:
    out = []
    while len(out) < n:
        block = VOICES[:]
        rng.shuffle(block)
        out.extend(block)
    return out[:n]


def load_src(path: Path = SRC) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _segments(s) -> list[dict]:
    if isinstance(s, str):
        return [{"text": s.strip(), "pause_after_ms": 0}]
    segs = []
    for i, el in enumerate(s):
        last = i == len(s) - 1
        if isinstance(el, str):
            text, pause = el, 0
        else:
            text, pause = el[0], int(el[1])
        if last:
            pause = 0
        segs.append({"text": text.strip(), "pause_after_ms": pause})
    return segs


def _tools(x) -> list[dict]:
    out = []
    for el in x or []:
        call = {"name": el[0], "arguments": {k: str(v) for k, v in (el[1] or {}).items()}}
        if len(el) > 2 and el[2] == "nf":
            call["expect_not_found"] = True
        out.append(call)
    return out


def build_cases(src_path: Path = SRC) -> list[dict]:
    src = load_src(src_path)
    members = {m["phone_number"]: m for m in json.loads(MOCK.read_text(encoding="utf-8"))["members"]}
    n = len(src)
    env_cols = {axis: balanced(ch, n, _rng(axis)) for axis, ch in ENV_AXES.items()}
    leads = lead_values(n, _rng("lead_ms"))
    voices = voice_values(n, _rng("voice"))
    times = default_times(n, _rng("now"))

    rows = []
    for i, sc in enumerate(src):
        segs = _segments(sc["s"])
        who = sc.get("who")
        profile = None
        tags = list(sc.get("t") or [])
        if who:
            m = members[str(who)]
            profile = {"name": m["name"], "phone": m["phone_number"], "email": None}
            tags.append("caller_known")
        row = {
            "case_id": f"T{i + 1:04d}",
            "now": sc.get("now") or times[i],
            "history": [{"role": "agent" if r == "a" else "customer", "text": t} for r, t in (sc.get("h") or [])],
            "customer_profile": profile,
            "customer_utterance": " ".join(s["text"] for s in segs),
            "segments": segs,
            "voice": voices[i],
        }
        if sc.get("style"):
            row["voice_style"] = sc["style"]
        row["env"] = {
            "channel": env_cols["channel"][i],
            "noise": env_cols["noise"][i],
            "snr_db": env_cols["snr_db"][i],
            "gain_db": env_cols["gain_db"][i],
            "speed": env_cols["speed"][i],
            "lead_ms": leads[i],
            "packet_loss": env_cols["packet_loss"][i],
        }
        row.update({
            "category": sc["c"],
            "tags": tags,
            "expected_tools": _tools(sc.get("x")),
            "expected_points": list(sc.get("p") or []),
            "must_not": list(sc.get("n") or []),
        })
        rows.append(row)
    return rows


def write(rows: list[dict], out: Path = OUT):
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stats_md(rows: list[dict]) -> str:
    n = len(rows)

    def table(title, cnt, order=None):
        keys = order or sorted(cnt, key=lambda k: (-cnt[k], str(k)))
        lines = [f"| {title} | 문항 | 비율 |", "|---|---|---|"]
        lines += [f"| {k} | {cnt[k]} | {cnt[k] / n:.0%} |" for k in keys]
        return "\n".join(lines)

    parts = [f"총 {n}문항\n"]
    parts.append(table("category", Counter(r["category"] for r in rows)))
    tag_cnt = Counter(t for r in rows for t in r["tags"])
    tag_cnt["(history 있음)"] = sum(1 for r in rows if r["history"])
    tag_cnt["(segments ≥ 2)"] = sum(1 for r in rows if len(r["segments"]) > 1)
    parts.append(table("tag", tag_cnt))
    for axis in ("channel", "noise", "snr_db", "gain_db", "speed", "packet_loss"):
        parts.append(table(axis, Counter(r["env"][axis] for r in rows)))
    leads = [r["env"]["lead_ms"] for r in rows]
    parts.append(f"lead_ms: {min(leads)}~{max(leads)} (고르게 분포)")
    parts.append(table("voice", Counter(r["voice"] for r in rows)))
    off = sum(1 for r in rows if not ("09:00" <= r["now"][11:] < "18:00") or r["now"][11:13] == "12"
              or r["now"][:10] in ("2026-11-21", "2026-11-22"))
    tools = Counter(t["name"] for r in rows for t in r["expected_tools"])
    parts.append(table("expected_tools 함수", tools))
    parts.append(f"영업시간 외 now: {off}문항 / expect_not_found 호출: "
                 f"{sum(1 for r in rows for t in r['expected_tools'] if t.get('expect_not_found'))}개")
    return "\n\n".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true", help="분포표만 출력")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    rows = build_cases()
    if a.stats:
        sys.stdout.reconfigure(encoding="utf-8")
        print(stats_md(rows))
        return
    write(rows, a.out)
    print(f"{len(rows)}문항 → {a.out}")


if __name__ == "__main__":
    main()
