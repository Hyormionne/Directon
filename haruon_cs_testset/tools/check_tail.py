"""말끝 잘림 검사: 파일 마지막 0.1초가 말소리 크기에 가까우면(90분위 대비 -12dB 이상) 단어 중간에서 끊겼을 가능성.
사용: python tools/check_tail.py audio/"""
import glob, os, sys, wave
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_audio import frames_db

bad = []
files = sorted(f for f in glob.glob(os.path.join(sys.argv[1], "h*.wav")) if not f.endswith(".orig.wav"))
for f in files:
    with wave.open(f) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    db = frames_db(pcm)
    rel = float(db[-5:].mean() - np.percentile(db, 90))
    if rel > -12:
        bad.append((os.path.basename(f)[:-4], round(rel, 1)))
print(f"말끝 잘림 의심 {len(bad)} / {len(files)}:", bad)
