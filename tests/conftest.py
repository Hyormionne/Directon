import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


import pytest


@pytest.fixture(autouse=True)
def _mini_knowledge(monkeypatch):
    """테스트는 공식 키트 대신 지어낸 작은 지식 폴더를 쓴다(키트가 없는 환경에서도 같은 결과)."""
    from agent import knowledge

    monkeypatch.setenv(knowledge.KNOWLEDGE_ENV, str(ROOT / "tests" / "fixtures" / "official_mini" / "brand"))
    knowledge.reset()
    yield
    knowledge.reset()
