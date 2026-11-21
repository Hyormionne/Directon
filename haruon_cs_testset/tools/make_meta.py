"""audio/meta.json 생성 — 공식 측정기(run_eval.py) 형식 {case_id: {"speech_end": 초, "voice": 화자}}.
사용: python tools/make_meta.py audio/ metadata.jsonl

speech_end = 파일 안에서 마지막 말소리가 끝나는 시각. check_audio와 같은 적응형 무음 기준으로 찾고,
끝의 0.3초 미만 소리가 0.5초 넘는 무음 너머 떨어져 있으면(숨소리·클릭) 말로 보지 않는다.
측정기는 이 시각을 기준으로 첫 소리 지연과 끼어듦(spoke_before_end)을 잰다.
"""
import json, os, sys, wave
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_audio import FRAME, frames_db  # noqa: E402
from fix_pauses import threshold  # noqa: E402

SR = 16000


def speech_end(pcm):
    db = frames_db(pcm)
    m = db >= threshold(db)
    segs, st = [], None
    for i, v in enumerate(m):
        if v and st is None: st = i
        if not v and st is not None: segs.append([st, i]); st = None
    if st is not None: segs.append([st, len(m)])
    if not segs:
        return None
    gap, short = 0.5 * SR / FRAME, 0.3 * SR / FRAME
    merged = [segs[0]]
    for s, e in segs[1:]:
        if s - merged[-1][1] < gap: merged[-1][1] = e
        else: merged.append([s, e])
    while len(merged) > 1 and merged[-1][1] - merged[-1][0] < short:
        merged.pop()
    return round(merged[-1][1] * FRAME / SR, 2)


def main():
    adir, meta_path = sys.argv[1], sys.argv[2]
    recs = [json.loads(l) for l in open(meta_path, encoding="utf-8")]
    out, missing = {}, []
    for r in recs:
        p = os.path.join(adir, f"{r['id']}.wav")
        if not os.path.exists(p):
            missing.append(r["id"]); continue
        with wave.open(p) as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        end = speech_end(pcm)
        dur = len(pcm) / SR
        rc = r["recording"]
        out[r["id"]] = {"speech_end": end, "voice": f"{rc['speaker']}_{rc['speaker_profile']}"}
        if end is None or dur - end > 0.8 or end < 0.5:
            print(f"! {r['id']}: speech_end {end}s / 파일 {dur:.2f}s — 확인 필요")
    with open(os.path.join(adir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"meta.json: {len(out)}개" + (f" · 음성 없음 {len(missing)}개({missing[0]}…)" if missing else ""))


if __name__ == "__main__":
    main()
