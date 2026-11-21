"""녹음 파일 검사: 형식(16kHz·16bit·mono), 앞뒤 무음, 대본의 중간 쉼이 실제로 들어갔는지.
사용: python tools/check_audio.py audio/ metadata.jsonl [--fix]
--fix: 형식이 다르면 ffmpeg로 16kHz·16bit·mono 변환(원본은 *.orig.wav로 보존)
"""
import json, os, shutil, subprocess, sys, wave
import numpy as np

FRAME = 320  # 20ms @16kHz (= 640바이트)

def frames_db(pcm):
    n = len(pcm) // FRAME
    x = pcm[: n * FRAME].reshape(n, FRAME).astype(np.float32)
    rms = np.sqrt((x ** 2).mean(axis=1)) + 1e-9
    return 20 * np.log10(rms / 32768)

def silent_runs(db, thr):
    runs, start = [], None
    for i, s in enumerate(db < thr):
        if s and start is None: start = i
        if not s and start is not None: runs.append((start, i)); start = None
    if start is not None: runs.append((start, len(db)))
    return runs

def check(path, rec):
    issues = []
    with wave.open(path) as w:
        fmt = (w.getframerate(), w.getsampwidth() * 8, w.getnchannels())
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if fmt != (16000, 16, 1):
        return [f"형식 {fmt} ≠ (16000,16,1)"], fmt
    db = frames_db(pcm)
    if len(db) == 0: return ["빈 파일"], fmt
    thr = min(max(np.percentile(db, 10) + 10, np.percentile(db, 90) - 30), np.percentile(db, 90) - 6)   # 적응형 무음 기준
    voiced = np.where(db >= thr)[0]
    if len(voiced) == 0: return ["음성 없음"], fmt
    lead, trail = voiced[0] * 0.02, (len(db) - 1 - voiced[-1]) * 0.02
    if lead > 0.5: issues.append(f"앞 무음 {lead:.2f}s(>0.5)")
    if trail > 0.8: issues.append(f"뒤 무음 {trail:.2f}s(>0.8)")
    inner = [(e - s) * 0.02 for s, e in silent_runs(db[voiced[0]: voiced[-1] + 1], thr)]
    found = sorted([d for d in inner if d >= 0.45], reverse=True)
    want = sorted([p["sec"] for p in rec["pauses"] if p["sec"] >= 0.8], reverse=True)
    for i, wsec in enumerate(want):
        if i >= len(found): issues.append(f"{wsec}s 쉼이 안 보임"); continue
        if abs(found[i] - wsec) > 0.4: issues.append(f"쉼 {wsec}s 지시 → 실제 약 {found[i]:.1f}s")
    if not want and found and found[0] >= 1.0:
        issues.append(f"지시 없는 긴 쉼 {found[0]:.1f}s (엔드포인팅 판정이 달라질 수 있음)")
    dur = len(pcm) / 16000
    est = rec["recording"]["est_duration_sec"]
    if not (0.5 * est <= dur <= 2.2 * est): issues.append(f"길이 {dur:.1f}s (예상 {est}s)")
    return issues, fmt

def main():
    adir, meta = sys.argv[1], sys.argv[2]
    fix = "--fix" in sys.argv
    recs = [json.loads(l) for l in open(meta, encoding="utf-8")]
    ok = missing = bad = 0
    for r in recs:
        p = os.path.join(adir, f"{r['id']}.wav")
        if not os.path.exists(p):
            missing += 1; continue
        issues, fmt = check(p, r)
        if fix and issues and issues[0].startswith("형식"):
            orig = p.replace(".wav", ".orig.wav"); shutil.move(p, orig)
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", orig, "-ar", "16000", "-ac", "1",
                            "-sample_fmt", "s16", p], check=True)
            issues, fmt = check(p, r)
        if issues:
            bad += 1; print(f"✗ {r['id']}: " + " | ".join(issues))
        else:
            ok += 1
    print(f"\n통과 {ok} · 문제 {bad} · 없음 {missing} / 전체 {len(recs)}")

if __name__ == "__main__":
    main()
