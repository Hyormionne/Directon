"""스모크 세트: data/testset/cases.jsonl에서 범주·태그·채널이 고르게 섞인 N문항을 골라 voice만 edge로 바꾼 사본.

AG-T09: 본 테스트셋 음성(Gemini 목소리)은 결제 반영 대기 중이라 edge 목소리로 먼저 E2E를 돌린다.
상담원 목소리(edge SunHi)와 겹치지 않게 InJoon·HyunsuMultilingual 위주, SunHi는 일부만.

사용:
  uv run python scripts/build_smoke_set.py [--n 20] [--out data/testset/smoke/cases.jsonl] [--build]
  --build: 이어서 eval.build_testset로 data/testset/smoke/audio/에 음성 생성(edge만 호출)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "testset" / "cases.jsonl"
OUT = ROOT / "data" / "testset" / "smoke" / "cases.jsonl"
AUDIO = ROOT / "data" / "testset" / "smoke" / "audio"
# 5문항 주기: InJoon 2, Hyunsu 2, SunHi 1 → 20문항이면 8/8/4
VOICES = ("edge:ko-KR-InJoonNeural", "edge:ko-KR-HyunsuMultilingualNeural", "edge:ko-KR-InJoonNeural",
          "edge:ko-KR-HyunsuMultilingualNeural", "edge:ko-KR-SunHiNeural")
# 함수 일치를 볼 문항(조회·실행·후속)을 더 넣는다. 20문항이면 4/4/3/3/2/2/2
QUOTA_WEIGHTS = {"tool_lookup": 4, "tool_action": 4, "followup": 3, "need_info": 3,
                 "faq": 2, "complaint_handoff": 2, "out_of_scope": 2}
# 테스트셋 본체에서 흔하고 정규화·턴 감지에 중요한 태그
TAG_WEIGHTS = {"digits": 2.0, "mid_pause": 1.5, "caller_known": 0.7}


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def select(cases: list[dict], n: int) -> list[dict]:
    """결정적 탐욕 선택: 범주를 돌아가며, 그 범주 안에서 아직 덜 나온 태그·채널·잡음·기대 함수 수를
    가장 많이 늘리는 문항을 고른다(동점이면 원래 순서)."""
    by_cat: dict[str, list[dict]] = {}
    for c in cases:
        by_cat.setdefault(c["category"], []).append(c)
    cats = sorted(by_cat)
    tag_n: Counter = Counter()
    chan_n: Counter = Counter()
    noise_n: Counter = Counter()
    ntool_n: Counter = Counter()
    picked: list[dict] = []
    used: set[str] = set()

    def gain(c: dict) -> float:
        env = c.get("env") or {}
        g = sum(TAG_WEIGHTS.get(t, 1.0) / (1 + tag_n[t]) for t in c.get("tags") or [])
        g += 1.0 / (1 + chan_n[env.get("channel", "wideband")])
        g += 0.5 / (1 + noise_n[env.get("noise", "none")])
        g += 0.5 / (1 + ntool_n[len(c.get("expected_tools") or [])])
        return g

    quota = {c: max(1, round(n * w / sum(QUOTA_WEIGHTS.get(k, 1) for k in cats)))
             for c, w in ((c, QUOTA_WEIGHTS.get(c, 1)) for c in cats)}
    taken: Counter = Counter()
    i = 0
    while len(picked) < n and any(c["case_id"] not in used for c in cases):
        cat = cats[i % len(cats)]
        i += 1
        pool = [c for c in by_cat[cat] if c["case_id"] not in used]
        if not pool or (taken[cat] >= quota[cat] and any(taken[k] < quota[k] for k in cats)):
            continue
        taken[cat] += 1
        best = max(pool, key=lambda c: (gain(c), -cases.index(c)))
        used.add(best["case_id"])
        picked.append(best)
        env = best.get("env") or {}
        tag_n.update(best.get("tags") or [])
        chan_n[env.get("channel", "wideband")] += 1
        noise_n[env.get("noise", "none")] += 1
        ntool_n[len(best.get("expected_tools") or [])] += 1
    return picked


def to_smoke(picked: list[dict]) -> list[dict]:
    out = []
    for k, c in enumerate(picked):
        c = dict(c)
        c["source_voice"] = c.get("voice")
        c["voice"] = VOICES[k % len(VOICES)]
        c.pop("voice_style", None)                    # edge는 말투 지시를 지원하지 않는다
        out.append(c)
    return out


def write(cases: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def summary(cases: list[dict]) -> str:
    cat = Counter(c["category"] for c in cases)
    tags = Counter(t for c in cases for t in c.get("tags") or [])
    chan = Counter((c.get("env") or {}).get("channel") for c in cases)
    voice = Counter(c["voice"] for c in cases)
    nt = Counter(len(c.get("expected_tools") or []) for c in cases)
    return (f"범주 {dict(cat)}\n태그 {dict(tags)}\n채널 {dict(chan)}\n음성 {dict(voice)}\n"
            f"기대 함수 수 {dict(sorted(nt.items()))}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--build", action="store_true", help="이어서 음성 생성")
    ap.add_argument("--audio", default=str(AUDIO))
    a = ap.parse_args(argv)
    smoke = to_smoke(select(load(Path(a.src)), a.n))
    write(smoke, Path(a.out))
    print(f"{len(smoke)}문항 → {a.out}\n{summary(smoke)}")
    if a.build:
        from eval import build_testset
        build_testset.main(["--cases", a.out, "--out", a.audio])


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
