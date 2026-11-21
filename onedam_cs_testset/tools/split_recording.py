"""화자 한 명이 대본 순서대로 이어 읽은 녹음 파일을 문항별 q###.wav로 자른다.

사용:
  python tools/split_recording.py 신호진_녹음.m4a --speaker S5
  python tools/split_recording.py 재녹음.m4a --ids q022,q096      # 일부 문항만 다시 녹음한 경우

동작:
  1) ffmpeg로 16kHz·16bit·mono 변환
  2) faster-whisper(로컬)로 단어 타임스탬프 받아쓰기 — 외부 API로 보내지 않음
  3) 대본(metadata.jsonl의 utterance)과 받아쓰기를 글자 단위로 정렬해 문항 경계를 찾음
  4) check_audio.py와 같은 적응형 무음 기준으로 앞뒤를 다듬어 audio/q###.wav로 저장
  5) 문항별 받아쓰기 결과와 대본을 나란히 보여 주고 check_audio 검사까지 실행

전제: 지정한 문항을 metadata.jsonl의 id 순서대로 한 번씩 읽었음(재녹음·건너뛴 문항은 --ids로 맞춤).
필요: ffmpeg, numpy, faster-whisper (pip install faster-whisper). 첫 실행 때 모델을 내려받음.
"""
import argparse, difflib, json, os, re, subprocess, sys, tempfile, wave
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SR, F = 16000, 320            # 20ms 프레임
LEAD, TRAIL = 0.12, 0.30      # 말 앞뒤로 남길 여유(초)
MINRUN = 5                    # 0.1s 이상 이어진 유성 구간만 말소리로 본다(잡음 블립 무시)


DIGIT_KO = str.maketrans("0123456789", "공일이삼사오육칠팔구")


def norm(s):
    # 받아쓰기는 숫자를 아라비아 숫자로 내므로, 대본의 한 자리씩 읽기(이공이육…)에 맞춰 바꿔 비교
    return re.sub(r"[^0-9A-Za-z가-힣]", "", s).lower().translate(DIGIT_KO)


def to_wav(src, dst):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-ar", str(SR), "-ac", "1",
                    "-sample_fmt", "s16", dst], check=True)
    with wave.open(dst) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16)


def transcribe(pcm, model_size):
    from faster_whisper import WhisperModel
    m = WhisperModel(model_size, device="cpu", compute_type="int8")
    segs, _ = m.transcribe(pcm.astype(np.float32) / 32768, language="ko", word_timestamps=True,
                           beam_size=5, condition_on_previous_text=False)
    return [{"s": w.start, "e": w.end, "w": w.word} for s in segs for w in s.words]


def item_starts(words, recs):
    """대본 글자열과 받아쓰기 글자열을 정렬해, 각 문항 첫 글자에 대응하는 단어 인덱스를 찾는다."""
    exp, bounds = "", []
    for r in recs:
        bounds.append(len(exp)); exp += norm(r["utterance"])
    hyp, owner = "", []
    for i, w in enumerate(words):
        t = norm(w["w"]); hyp += t; owner += [i] * len(t)
    blocks = [b for b in difflib.SequenceMatcher(None, exp, hyp, autojunk=False).get_matching_blocks() if b.size]
    starts = []
    for k, b0 in enumerate(bounds):
        b1 = bounds[k + 1] if k + 1 < len(bounds) else len(exp)
        hit = next((bl for bl in blocks if bl.a + bl.size > b0 and bl.a < b1), None)
        if hit is None:
            sys.exit(f"{recs[k]['id']}: 받아쓰기에서 이 문항을 찾지 못함 (건너뛰었다면 --ids로 빼 주세요)")
        hpos = hit.b + max(0, b0 - hit.a)
        starts.append(owner[min(hpos, len(owner) - 1)])
    if any(b <= a for a, b in zip(starts, starts[1:])):
        sys.exit(f"문항 경계 순서가 꼬임 {starts} — 대본 순서대로 읽었는지 확인하세요")
    return starts


def runs(mask):
    out, st = [], None
    for i, v in enumerate(mask):
        if v and st is None: st = i
        if not v and st is not None: out.append((st, i)); st = None
    if st is not None: out.append((st, len(mask)))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔(cp949)에서 한글·기호 출력이 깨지지 않게
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--speaker", help="S1~S5: 그 화자의 문항 전체")
    g.add_argument("--ids", help="쉼표로 구분한 문항 id (읽은 순서대로)")
    ap.add_argument("--meta", default=os.path.join(ROOT, "metadata.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "audio"))
    ap.add_argument("--model", default="medium", help="faster-whisper 모델 크기 (기본 medium)")
    ap.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않고 경계만 출력")
    a = ap.parse_args()

    allrecs = [json.loads(l) for l in open(a.meta, encoding="utf-8")]
    if a.speaker:
        recs = [r for r in allrecs if r["recording"]["speaker"] == a.speaker]
    else:
        byid = {r["id"]: r for r in allrecs}
        recs = [byid[i.strip()] for i in a.ids.split(",")]
    if not recs: sys.exit("해당 문항 없음")

    with tempfile.TemporaryDirectory() as td:
        x = to_wav(a.recording, os.path.join(td, "full.wav"))
    print(f"녹음 {len(x) / SR:.1f}s · 문항 {len(recs)}개 · 받아쓰기 중({a.model})…", flush=True)
    words = transcribe(x, a.model)
    starts = item_starts(words, recs)
    spans = []
    for k, si in enumerate(starts):
        ei = (starts[k + 1] if k + 1 < len(starts) else len(words)) - 1
        spans.append((words[si]["s"], words[ei]["e"], "".join(w["w"] for w in words[si:ei + 1]).strip()))

    n = len(x) // F
    db = 20 * np.log10(np.sqrt((x[: n * F].reshape(n, F).astype(np.float32) ** 2).mean(1)) + 1e-9) - 90.3
    os.makedirs(a.out, exist_ok=True)
    for k, (ws, we, text) in enumerate(spans):
        # 탐색 창: 앞 문항 끝~다음 문항 시작의 중간점까지로 제한해 옆 문항이 섞이지 않게
        lo = max(0.0, ws - 0.5) if k == 0 else max((spans[k - 1][1] + ws) / 2, ws - 0.5)
        hi = we + 0.5 if k + 1 == len(spans) else min((we + spans[k + 1][0]) / 2, we + 0.5)
        lo_f, hi_f = int(lo / .02), min(n, int(hi / .02))
        d = db[lo_f:hi_f]
        thr = min(max(np.percentile(d, 10) + 10, np.percentile(d, 90) - 30), np.percentile(d, 90) - 6)  # check_audio와 같은 식
        rs = [(s + lo_f, e + lo_f) for s, e in runs(d >= thr) if e - s >= MINRUN]
        on, off = (rs[0][0], rs[-1][1]) if rs else (int(ws / .02), int(we / .02))
        s0, s1 = max(0, on * F - int(LEAD * SR)), min(len(x), off * F + int(TRAIL * SR))
        r = recs[k]
        sim = difflib.SequenceMatcher(None, norm(r["utterance"]), norm(text)).ratio()
        flag = "  ← 대본과 많이 다름, 들어 보세요" if sim < 0.8 else ""
        print(f"{r['id']} {s0 / SR:7.2f}-{s1 / SR:7.2f} ({(s1 - s0) / SR:.2f}s) 일치 {sim:.2f}{flag}\n"
              f"   대본: {r['utterance']}\n   인식: {text}")
        if not a.dry_run:
            with wave.open(os.path.join(a.out, f"{r['id']}.wav"), "wb") as o:
                o.setnchannels(1); o.setsampwidth(2); o.setframerate(SR); o.writeframes(x[s0:s1].tobytes())

    if not a.dry_run:
        print("\n— check_audio —")
        sys.path.insert(0, HERE)
        from check_audio import check
        bad = 0
        for r in recs:
            issues, _ = check(os.path.join(a.out, f"{r['id']}.wav"), r)
            if issues: bad += 1; print(f"✗ {r['id']}: " + " | ".join(issues))
        print(f"통과 {len(recs) - bad} / {len(recs)}")


if __name__ == "__main__":
    main()
