"""녹음 속 말 중간 쉼을 대본 지시에 맞춘다.
사용: python tools/fix_pauses.py audio/ metadata.jsonl [--ids h001,h002] [--dry]

- 무음 판정은 check_audio.py와 같은 적응형 기준(20ms 프레임).
- 대본의 [n초 쉼]은 글자 위치(char_offset)로 녹음 속 무음 구간에 짝지어 지시 길이로 맞춘다(길면 줄이고, 0.15초 넘게 짧으면 늘림).
- 지시한 쉼을 녹음에서 못 찾으면(거의 안 쉬고 읽음) 늘리지 않고 '찾지 못한 쉼'으로 알린다 → 재녹음 대상.
- 짝이 없는 무음 중 0.6초를 넘는 것은 0.5초로 줄인다(지시 없는 긴 쉼 → 말 끝 판단 오작동 방지).
- 줄일 때는 무음 가운데를 잘라 앞뒤 자연스러운 감쇠를 남기고, 늘릴 때는 그 무음 구간의 잡음을 반복해 채운다.
- 처음 고칠 때 원본을 h###.orig.wav로 남기고, 다시 돌리면 항상 원본에서 다시 계산한다(--again이면 고친 파일에서 한 번 더).
"""
import argparse, json, os, shutil, sys, wave
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_audio import FRAME, frames_db, silent_runs  # noqa: E402

SR = 16000
MIN_RUN = 0.3        # 이보다 짧은 무음은 쉼으로 보지 않음(파열음 폐쇄 등)
FREE_MAX, FREE_TO = 0.6, 0.5
TOL = 0.15


def threshold(db):
    return min(max(np.percentile(db, 10) + 10, np.percentile(db, 90) - 30), np.percentile(db, 90) - 6)


def plan(pcm, rec):
    db = frames_db(pcm)
    thr = threshold(db)
    voiced = np.where(db >= thr)[0]
    if len(voiced) == 0:
        return [], [], None
    # 가장자리 잡음: 맨 앞/뒤의 0.3초 미만 소리가 0.5초 이상 무음 너머 따로 떨어져 있으면(숨소리·클릭·이전 문항 꼬리) 말이 아님
    vmask = db >= thr
    vsegs, st = [], None
    for i, m in enumerate(vmask):
        if m and st is None: st = i
        if not m and st is not None: vsegs.append([st, i]); st = None
    if st is not None: vsegs.append([st, len(vmask)])
    gap, short = 0.5 * SR / FRAME, 0.3 * SR / FRAME
    merged = [vsegs[0]]
    for s_, e_ in vsegs[1:]:
        if s_ - merged[-1][1] < gap:
            merged[-1][1] = e_
        else:
            merged.append([s_, e_])
    vsegs = merged
    while len(vsegs) > 1 and vsegs[0][1] - vsegs[0][0] < short and vsegs[1][0] - vsegs[0][1] >= gap:
        vsegs.pop(0)
    while len(vsegs) > 1 and vsegs[-1][1] - vsegs[-1][0] < short and vsegs[-1][0] - vsegs[-2][1] >= gap:
        vsegs.pop()
    v0, v1 = vsegs[0][0], vsegs[-1][1]
    trim = (max(0, v0 - int(0.12 * SR / FRAME)), min(len(db), v1 + int(0.3 * SR / FRAME)))
    runs = [(s + v0, e + v0) for s, e in silent_runs(db[v0:v1], thr) if (e - s) * FRAME / SR >= MIN_RUN]
    pauses = rec["pauses"]
    text_len = max(1, len(rec["recording"].get("read_text", rec["utterance"])))
    total_p = sum(p["sec"] for p in pauses) * SR / FRAME
    span = max(1, (v1 - v0) - total_p)
    # 쉼 j가 녹음에서 나타날 것으로 예상되는 프레임
    exp, acc = [], 0.0
    for p in pauses:
        exp.append(v0 + p["char_offset"] / text_len * span + acc)
        acc += p["sec"] * SR / FRAME
    assigned = {}
    for j in sorted(range(len(pauses)), key=lambda j: -pauses[j]["sec"]):
        free = [k for k in range(len(runs)) if k not in assigned.values()]
        if not free:
            break
        k = min(free, key=lambda k: abs((runs[k][0] + runs[k][1]) / 2 - exp[j]) - 0.5 * (runs[k][1] - runs[k][0]))
        assigned[j] = k
    target = {}
    for j, k in assigned.items():
        target[k] = pauses[j]["sec"]
    edits, notes = [], []
    for k, (s, e) in enumerate(runs):
        cur = (e - s) * FRAME / SR
        if k in target:
            t = target[k]
            if cur - t > TOL or t - cur > TOL:
                edits.append((s, e, t)); notes.append(f"쉼 {cur:.1f}→{t}s")
        elif cur > FREE_MAX:
            edits.append((s, e, FREE_TO)); notes.append(f"지시 없는 쉼 {cur:.1f}→{FREE_TO}s")
    missing = [pauses[j]["sec"] for j in range(len(pauses)) if j not in assigned]
    if missing:
        notes.append(f"찾지 못한 쉼 {missing}")
    cut_front, cut_back = trim[0] * FRAME / SR, (len(db) - trim[1]) * FRAME / SR
    if cut_front > 0.3 or cut_back > 0.5:
        notes.insert(0, f"가장자리 잡음·무음 제거 앞 {cut_front:.1f}s 뒤 {cut_back:.1f}s")
    else:
        trim = None
    return edits, notes, trim


def apply(pcm, edits, trim=None):
    out, pos = [], 0
    for s, e, t in sorted(edits):
        a, b = s * FRAME, e * FRAME
        seg = pcm[a:b]
        want = int(t * SR)
        out.append(pcm[pos:a])
        if len(seg) >= want:
            h = want // 2
            out += [seg[:h], seg[len(seg) - (want - h):]]
        else:
            fill = np.resize(seg[len(seg) // 4: 3 * len(seg) // 4] if len(seg) >= 8 else np.zeros(1, np.int16), want - len(seg))
            mid = len(seg) // 2
            out += [seg[:mid], fill.astype(np.int16), seg[mid:]]
        pos = b
    out.append(pcm[pos:])
    new = np.concatenate(out).astype(np.int16)
    if trim:  # 편집으로 줄어든 길이만큼 뒤쪽 경계를 당긴다
        removed = sum((e - s) * FRAME - int(t * SR) for s, e, t in edits)
        new = new[trim[0] * FRAME: trim[1] * FRAME - removed]
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adir"); ap.add_argument("meta")
    ap.add_argument("--ids"); ap.add_argument("--dry", action="store_true")
    ap.add_argument("--again", action="store_true", help="원본 대신 이미 고친 파일에서 한 번 더 다듬기")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.meta, encoding="utf-8")]
    ids = set(a.ids.split(",")) if a.ids else None
    changed = 0
    for r in recs:
        p = os.path.join(a.adir, f"{r['id']}.wav")
        if (ids and r["id"] not in ids) or not os.path.exists(p):
            continue
        orig = p.replace(".wav", ".orig.wav")
        src = orig if os.path.exists(orig) and not a.again else p
        with wave.open(src) as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        edits, notes, trim = plan(pcm, r)
        if not edits and not trim:
            if notes:
                print(f"· {r['id']}: " + " | ".join(notes))
            continue
        changed += 1
        print(f"✎ {r['id']}: " + " | ".join(notes))
        if a.dry:
            continue
        if not os.path.exists(orig):
            shutil.copy2(p, orig)
        new = apply(pcm, edits, trim)
        with wave.open(p, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(new.tobytes())
    print(f"\n수정 {changed}개" + (" (dry run)" if a.dry else ""))


if __name__ == "__main__":
    main()
