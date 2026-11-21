"""cases.jsonl → <out>/<case_id>.wav + <out>/meta.json (docs/AGENT_PLAN.md 3.2·3.3).

사용:
  uv run python -m eval.build_testset [--cases legacy/testset/cases.jsonl] [--out legacy/testset/audio]
      [--only T0001,T0002] [--workers 4] [--force] [--cache legacy/testset/cache]

- 세그먼트(segments[{text,pause_after_ms}])를 각각 합성 → trim → pause로 잇는다. 없으면 customer_utterance 한 덩어리.
- 깨끗한 음성은 음성 구간 RMS 3000(약 -21dBFS)으로 맞춘 뒤 env(3.3절)로 증강. seed는 env.seed가 없으면 case_id 해시.
- babble 잡음은 이번 실행에서 합성한 다른 문항의 깨끗한 음성(부족하면 캐시의 다른 음성)으로 만든다.
- meta.json = {case_id: {"speech_end": 초, ...}} — Starter Kit 측정기가 읽는 형식. 기존 항목과 병합.
- 이미 wav와 meta 항목이 있으면 건너뛴다(--force로 재생성).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from eval import augment as A

ROOT = Path(__file__).resolve().parents[1]
POOL_MIN = 4
TARGET_RMS = 3000.0           # 깨끗한 음성 레벨 통일(약 -21dBFS). gain_db는 이 기준에서 더한다
POOL_MAX_FROM_CACHE = 12


def load_cases(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def case_seed(case: dict) -> int:
    s = (case.get("env") or {}).get("seed")
    if s is not None:
        return int(s)
    return int(hashlib.sha256(case["case_id"].encode("utf-8")).hexdigest()[:8], 16)


def case_segments(case: dict) -> list[dict]:
    segs = case.get("segments")
    if segs:
        return segs
    return [{"text": case["customer_utterance"], "pause_after_ms": 0}]


def write_wav(path: Path, pcm: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(A.SR)
        w.writeframes(np.asarray(pcm, np.int16).tobytes())


def _cache_pool(synth, case: dict, exclude_keys: set[str]) -> list[np.ndarray]:
    cache_dir = getattr(synth, "cache_dir", None)
    if not cache_dir or not Path(cache_dir).is_dir():
        return []
    files = sorted(p for p in Path(cache_dir).glob("*.pcm") if p.stem not in exclude_keys)
    if not files:
        return []
    rng = np.random.default_rng(case_seed(case))
    pick = rng.choice(len(files), size=min(POOL_MAX_FROM_CACHE, len(files)), replace=False)
    return [np.fromfile(files[int(i)], dtype=np.int16) for i in sorted(pick)]


def _segment_keys(synth, case: dict) -> set[str]:
    key = getattr(synth, "cache_key", None)
    if key is None:
        return set()
    return {key(s["text"], case["voice"], case.get("voice_style")) for s in case_segments(case)}


def build(cases: list[dict], out_dir, synth, force: bool = False, only: set[str] | None = None,
          workers: int = 4, log=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    targets = [c for c in cases if not only or c["case_id"] in only]
    todo = [c for c in targets
            if force or c["case_id"] not in meta or not (out_dir / f"{c['case_id']}.wav").exists()]
    if not todo:
        log(f"건너뜀: {len(targets)}개 모두 존재 (--force로 재생성)")
        return meta

    # 1) 깨끗한 음성(세그먼트 조립) — 병렬 합성
    def clean_of(case):
        voice, style = case["voice"], case.get("voice_style")
        clean = A.assemble_segments(case_segments(case), lambda t: synth.synthesize(t, voice, style))
        return np.round(A.normalize_level(clean, TARGET_RMS)).astype(np.int16)

    failures: dict[str, str] = {}

    def safe_clean(case):
        try:
            return clean_of(case)
        except Exception as e:  # 한 문항 실패로 전체를 멈추지 않는다
            failures[case["case_id"]] = repr(e)[:300]
            log(f"{case['case_id']}: 합성 실패 {e!r}"[:300])
            return None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        cleans = {c["case_id"]: a for c, a in zip(todo, ex.map(safe_clean, todo)) if a is not None}
    todo = [c for c in todo if c["case_id"] in cleans]

    # 2) 증강 + 저장 (문항 순서대로 결정적)
    lock = threading.Lock()

    def finish(case):
        try:
            _finish(case)
        except Exception as e:
            failures[case["case_id"]] = repr(e)[:300]
            log(f"{case['case_id']}: 증강·저장 실패 {e!r}"[:300])

    def _finish(case):
        cid = case["case_id"]
        env = case.get("env") or {}
        pool = None
        if env.get("noise") == "babble":
            pool = [cleans[k] for k in sorted(cleans) if k != cid]
            if len(pool) < POOL_MIN:
                pool += _cache_pool(synth, case, _segment_keys(synth, case))
        r = A.augment(cleans[cid], env, seed=case_seed(case), babble_pool=pool)
        if r.speech_end is None:
            raise RuntimeError(f"{cid}: 음성이 검출되지 않았다")
        write_wav(out_dir / f"{cid}.wav", r.audio)
        entry = {"speech_end": round(float(r.speech_end), 3),
                 "duration": round(len(r.audio) / A.SR, 3),
                 "voice": case["voice"], "voice_style": case.get("voice_style"),
                 "env": {k: v for k, v in A.Env.from_dict(env).__dict__.items() if k != "unknown"},
                 **{k: v for k, v in r.info.items() if k in ("clipped_frac", "dropped_frames",
                                                             "noise_fallback", "speed_method",
                                                             "unknown_env_keys")}}
        with lock:
            meta[cid] = entry
        log(f"{cid}: speech_end={entry['speech_end']:.3f}s dur={entry['duration']:.2f}s "
            f"{env.get('channel', 'wideband')}/{env.get('noise', 'none')}")

    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            list(ex.map(finish, todo))
    finally:
        meta_path.write_text(json.dumps(dict(sorted(meta.items())), ensure_ascii=False, indent=1),
                             encoding="utf-8")
    if failures:
        log(f"실패 {len(failures)}건: {', '.join(sorted(failures))}")
    return meta


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="cases.jsonl → wav + meta.json")
    ap.add_argument("--cases", default=str(ROOT / "legacy" / "testset" / "cases.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "legacy" / "testset" / "audio"))
    ap.add_argument("--only", default="", help="쉼표로 구분한 case_id")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cache", default=str(ROOT / "legacy" / "testset" / "cache"))
    ap.add_argument("--max-gemini", type=int, default=3, help="Gemini 동시 호출 수")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    from eval.synth import Synthesizer
    synth = Synthesizer(cache_dir=a.cache, max_gemini_concurrency=a.max_gemini)
    only = {s.strip() for s in a.only.split(",") if s.strip()} or None
    meta = build(load_cases(a.cases), a.out, synth, force=a.force, only=only, workers=a.workers)
    print(f"완료: meta {len(meta)}개, 캐시 hit={synth.stats['hit']} miss={synth.stats['miss']} "
          f"retry={synth.stats['retry']}", file=sys.stderr)


if __name__ == "__main__":
    main()
