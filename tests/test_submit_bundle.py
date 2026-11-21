"""제출 묶음(scripts/package.sh)과 묶음 안내 문서(SUBMIT_README.md)의 최소 검사(AG-T31).

묶음을 실제로 만들지는 않는다(.env와 공식 키트가 필요하다). 스크립트·문서의 글자와 코드의 기본값이 맞는지만 본다.
"""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from agent.config import ENV_PREFIX, Settings
from agent.providers import factory
from agent.providers.baze_tts import BAZE_BASE_URL
from agent.providers.factory import ENV_MAP, ProviderConfig

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "SUBMIT_README.md").read_text(encoding="utf-8")
PACKAGE = (ROOT / "scripts" / "package.sh").read_text(encoding="utf-8")


def _env_rows() -> dict[str, list[str]]:
    """환경 변수 표: 이름 → [필수 여부, 기본값, 뜻]."""
    rows = {}
    for line in README.splitlines():
        m = re.match(r"\| `([A-Z][A-Z0-9_]+)` \|(.+)\|$", line)
        if m:
            rows[m.group(1)] = [c.strip() for c in m.group(2).split("|")]
    return rows


def test_package_puts_readme_on_top():
    assert 'cp "$ROOT/SUBMIT_README.md" "$OUT/README.md"' in PACKAGE
    assert 'need "$ROOT/SUBMIT_README.md"' in PACKAGE
    check = PACKAGE[PACKAGE.index("for f in start_server.sh"):]
    for name in ("start_server.sh", "start_mock.sh", "requirements.txt", ".env", "README.md",
                 "models/silero_vad.onnx", "data/brand/tools.json", "data/brand/scenarios.md",
                 "data/brand/information.json", "data/brand/style.json", "data/mock_db.json"):
        assert name in check.split("do need")[0], name


def test_package_env_allowlist_has_no_dev_key_and_no_tts_off():
    allow = re.search(r"^for k in ([A-Z0-9_ ]+); do$", PACKAGE, re.M).group(1).split()
    assert "BAZE_API_KEY" in allow
    assert "OPENAI_API_KEY" not in allow and "TYPESAFE_API_KEY" not in allow and "ELEVENLABS_API_KEY" not in allow
    assert "DIRECTON_TTS_PROVIDER=off*" in PACKAGE           # 개발용 TTS 끔 설정은 건너뛴다
    assert 'grep -q "^BAZE_API_KEY=" "$OUT/.env"' in PACKAGE


def test_package_rejects_things_that_must_not_ship():
    for name in ("tests", "docs", "runs", ".venv", "__pycache__", "audio", '"*.wav"', '"practice_cases*"',
                 '"examples.json"'):
        assert f"-name {name}" in PACKAGE, name
    assert "practice_cases" not in PACKAGE.split("bad=")[0]   # 연습 문항을 복사하는 줄이 없다
    assert "/data/audio" not in PACKAGE


def test_readme_start_commands_and_ports():
    for s in ("pip install -r requirements.txt", "bash start_mock.sh", "bash start_server.sh", "9000", "8100",
              "0.0.0.0"):
        assert s in README, s
    assert "--port 9000" in (ROOT / "start_server.sh").read_text(encoding="utf-8")
    assert "--port 8100" in (ROOT / "start_mock.sh").read_text(encoding="utf-8")


def test_readme_env_table_matches_code():
    rows = _env_rows()
    assert rows["BAZE_API_KEY"][0] == "필수"
    assert all(v[0] == "선택" for k, v in rows.items() if k != "BAZE_API_KEY")
    settings_env = {ENV_PREFIX + f.name.upper() for f in dataclasses.fields(Settings)}
    source = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "agent").rglob("*.py"))
    for name in rows:                                         # 코드가 실제로 읽는 이름만 적는다
        assert name in ENV_MAP.values() or name in settings_env or f'"{name}"' in source, name
    for name in ENV_MAP.values():                             # 조립에 쓰는 이름은 빠짐없이 적는다
        assert name in rows, name

    d = ProviderConfig()
    assert rows["BAZE_BASE_URL"][1] == f"`{BAZE_BASE_URL}`" == f"`{d.baze_base_url}`"
    assert rows["SONIOX_MODEL"][1] == f"`{d.soniox_model}`"
    assert rows["STT_LANGUAGE"][1] == f"`{d.stt_language}`"
    assert rows["OPENAI_MODEL"][1] == f"`{d.openai_model}`"
    assert rows["OPENAI_REASONING_EFFORT"][1] == f"`{d.openai_reasoning}`"
    assert rows["DIRECTON_TTS_PROVIDER"][1] == f"`{d.tts}`"
    assert rows["GEMINI_TTS_VOICE"][1] == f"`{d.tts_voice}`"
    assert f"`{factory.TTS_MODEL_BAZE}`" in rows["GEMINI_TTS_MODEL"][1]
    assert f"`{factory.TTS_MODEL_DIRECT}`" in rows["GEMINI_TTS_MODEL"][1]
    s = Settings()
    assert rows["DIRECTON_PRELOAD_ORDERS"][1] == ("`1`" if s.preload_orders else "`0`")
    assert rows["DIRECTON_LOG_DIR"][1] == f"`{s.log_dir}`"
    assert rows["DIRECTON_TTS_CACHE_DIR"][1] == f"`{s.tts_cache_dir}`"


def test_readme_models_and_privacy_notes():
    d = ProviderConfig()
    for s in (d.soniox_model, d.openai_model, factory.TTS_MODEL_BAZE, factory.TTS_MODEL_DIRECT,
              "저장하지 않는다", "DIRECTON_NO_AUDIO_LOG", "runs/logs"):
        assert s in README, s


def test_readme_has_no_key_values():
    assert not re.search(r"(sk-[A-Za-z0-9_-]{16,}|AIza[0-9A-Za-z_-]{20,}|_API_KEY\s*=\s*\S)", README)


def test_requirements_cover_runtime_packages():
    req = {re.split(r"[<>=!~\[ ]", ln.strip(), maxsplit=1)[0].lower()
           for ln in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
           if ln.strip() and not ln.lstrip().startswith("#")}
    assert {"websockets", "httpx", "numpy", "onnxruntime", "soxr", "openai", "google-genai", "python-dotenv"} <= req
