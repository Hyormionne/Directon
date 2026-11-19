"""턴 감지 모델을 models/에 내려받는다. 이미 있으면 건너뛴다.

사용: uv run python scripts/download_models.py [--force]

출처·라이선스
- Silero VAD (silero_vad.onnx, v6 계열, 16kHz 512샘플 입력)
  저장소: https://github.com/snakers4/silero-vad  (MIT License)
  파일:   https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
- Pipecat Smart Turn v3.2 (smart-turn-v3.2-cpu.onnx, int8, 약 8MB)
  저장소: https://github.com/pipecat-ai/smart-turn  (BSD-2-Clause)
  모델:   https://huggingface.co/pipecat-ai/smart-turn-v3
  파일:   https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

MODELS: dict[str, str] = {
    "silero_vad.onnx":
        "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
    "smart-turn-v3.2-cpu.onnx":
        "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx",
}

MIN_BYTES = 100_000  # 받다 만 파일·HTML 오류 페이지를 걸러낸다


def ensure(name: str, url: str, models_dir: Path = MODELS_DIR, force: bool = False) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    dst = models_dir / name
    if dst.exists() and dst.stat().st_size >= MIN_BYTES and not force:
        print(f"skip  {name} ({dst.stat().st_size:,} bytes)")
        return dst
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"get   {name} <- {url}")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 16):
                f.write(chunk)
    size = tmp.stat().st_size
    if size < MIN_BYTES:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{name}: 파일이 너무 작다({size} bytes)")
    tmp.replace(dst)
    print(f"ok    {name} ({size:,} bytes)")
    return dst


def ensure_all(models_dir: Path = MODELS_DIR, force: bool = False) -> dict[str, Path]:
    return {n: ensure(n, u, models_dir, force) for n, u in MODELS.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="이미 있어도 다시 받는다")
    ap.add_argument("--dir", type=Path, default=MODELS_DIR)
    a = ap.parse_args(argv)
    ensure_all(a.dir, a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
