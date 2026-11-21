"""agent_world/audio(실녹음)에 통화 환경을 입혀 별도 폴더로 낸다. speech_end는 원본 그대로(앞뒤 무음 추가 없음).

사용(프로젝트 루트에서):
  uv run python onedam_cs_testset/tools/augment_world.py --name phone --channel phone_ulaw
  uv run python onedam_cs_testset/tools/augment_world.py --name phone_babble --channel phone_ulaw --noise babble --snr 12
출력: onedam_cs_testset/agent_world/audio_<name>/ (wav + meta.json, env 기록)
"""
import argparse, json, os, sys, wave
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.dirname(HERE)
ROOT = os.path.dirname(DS)
sys.path.insert(0, ROOT)
from eval.augment import augment  # noqa: E402

SRC = os.path.join(DS, "agent_world", "audio")


def read(p):
    with wave.open(p) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, p
        return np.frombuffer(w.readframes(w.getnframes()), np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--channel", default="wideband")
    ap.add_argument("--noise", default="none")
    ap.add_argument("--snr", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(SRC, "meta.json"), encoding="utf-8"))
    ids = sorted(meta)
    clean = {i: read(os.path.join(SRC, f"{i}.wav")) for i in ids}
    out = os.path.join(DS, "agent_world", f"audio_{a.name}")
    os.makedirs(out, exist_ok=True)
    env = {"channel": a.channel, "noise": a.noise, "snr_db": a.snr, "lead_ms": 0, "tail_ms": 0}
    new_meta = {}
    for k, i in enumerate(ids):
        pool = [clean[j].astype(np.float32) for j in ids if j != i][:30]
        r = augment(clean[i].astype(np.float32), env, seed=a.seed * 1000 + k, babble_pool=pool)
        with wave.open(os.path.join(out, f"{i}.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(r.audio.tobytes())
        m = dict(meta[i]); m["env"] = {kk: env[kk] for kk in ("channel", "noise", "snr_db")}
        new_meta[i] = m
    json.dump(new_meta, open(os.path.join(out, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(ids)}개 → {out} env={env}")


if __name__ == "__main__":
    main()
