"""잘라 낸 문항 파일을 하나씩 다시 받아써 대본과 비교한다(말끝 잘림·옆 문항 섞임 검사).
사용: uv run --with openvino-genai python tools/verify_clips.py audio/ metadata.jsonl [--ids h001,...] [--min 0.8]
OpenVINO GenAI WhisperPipeline(GPU)로 로컬에서 받아쓴다. 외부로 보내지 않는다.
"""
import argparse, difflib, json, os, sys, wave
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from split_recording import norm, PROJECT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adir"); ap.add_argument("meta")
    ap.add_argument("--ids"); ap.add_argument("--min", type=float, default=0.8)
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--ov-model", default=os.path.join(PROJECT, "models", "whisper-large-v3-int8-ov"))
    a = ap.parse_args()
    import openvino_genai as ovg
    pipe = ovg.WhisperPipeline(a.ov_model, a.device, CACHE_DIR=os.path.join(PROJECT, "models", "ov_cache"))
    ids = set(a.ids.split(",")) if a.ids else None
    recs = [json.loads(l) for l in open(a.meta, encoding="utf-8")]
    low, n = [], 0
    for r in recs:
        p = os.path.join(a.adir, f"{r['id']}.wav")
        if (ids and r["id"] not in ids) or not os.path.exists(p):
            continue
        with wave.open(p) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        text = pipe.generate(np.concatenate([x, np.zeros(8000, np.float32)]).tolist(),
                             language="<|ko|>", task="transcribe").texts[0].strip()
        ref = r["recording"].get("read_text", r["utterance"])
        sim = difflib.SequenceMatcher(None, norm(ref), norm(text)).ratio()
        n += 1
        if sim < a.min:
            low.append(r["id"])
            print(f"✗ {r['id']} 일치 {sim:.2f}\n   대본: {ref}\n   인식: {text}")
    print(f"\n검증 {n}개 · 일치 {a.min} 미만 {len(low)}개: {','.join(low)}")


if __name__ == "__main__":
    main()
