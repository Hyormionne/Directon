"""제출 ZIP 만들기: python scripts/make_submission.py [출력 경로]  (기본 ../submission.zip)

넣는 것: start_server.sh, start_mock.sh, requirements.txt, .env(있으면), agent/, mock_api/, official/, models/
빼는 것: .venv, tests, runs, __pycache__ 등. official/·.env는 커밋하지 않지만 ZIP에는 들어간다(반출 금지 자료 — ZIP은 제출용으로만).
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["start_server.sh", "start_mock.sh", "requirements.txt", ".env"]
DIRS = ["agent", "mock_api", "official", "models"]
NEEDED = ["official/manual.md", "official/style.md", "official/tools.json", "official/mock_db.json",
          "models/silero_vad.onnx", "models/smart-turn-v3.2-cpu.onnx"]


def main(out: Path) -> None:
    missing = [p for p in NEEDED if not (ROOT / p).is_file()]
    if missing:
        sys.exit(f"없는 파일: {missing} (scripts/build_manual.py, scripts/download_models.py 먼저)")
    env = ROOT / ".env"
    if not env.is_file():
        print("경고: .env 없음 — 측정 때 API 키가 없다")
    elif any(ln.strip().startswith(("OPENAI_API_KEY=", "SONIOX_API_KEY=", "GEMINI_API_KEY="))
             for ln in env.read_text(encoding="utf-8").splitlines()):
        print("경고: .env에 직접 키가 있다 — 제출용은 BAZE_API_KEY(대회 크레딧)만 넣는다. 직접 키가 게이트웨이보다 우선한다")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            if (ROOT / f).is_file():
                z.write(ROOT / f, f)
        for d in DIRS:
            for p in sorted((ROOT / d).rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                    z.write(p, p.relative_to(ROOT).as_posix())
    print(f"{out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "submission.zip")
