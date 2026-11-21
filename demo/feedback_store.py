"""데모 통화 묶음 저장 — 본체는 feedback_kit/store.py(AG-T53에서 이식용 키트로 옮김).

예전 이름(demo.feedback_store)과 데모 기본 저장 위치(runs/demo_feedback/sessions)를 유지한다.
CLI: uv run python -m demo.feedback_store <bundle.json>... [--out DIR]
"""
from __future__ import annotations

from pathlib import Path

from feedback_kit import store as _store
from feedback_kit.store import (  # noqa: F401  (다시 내보내기)
    RATE, SCHEMA, TRACKS, BundleError, build_cases, decode_wav, interleave, slice_ms, wav_bytes,
)

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "runs" / "demo_feedback" / "sessions"


def write_session(bundle: dict, root: Path | str = DEFAULT_ROOT, *, saved_at: str | None = None) -> dict:
    return _store.write_session(bundle, root, saved_at=saved_at)


def main(argv: list[str] | None = None) -> int:
    return _store.main(argv, default_out=DEFAULT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
