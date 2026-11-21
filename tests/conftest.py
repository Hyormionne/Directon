import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


import pytest  # noqa: E402


@pytest.fixture
def legacy_world(monkeypatch):
    """옛 다이렉트몰 체계(주문번호 10자리·상품 카탈로그) 전제로 쓴 테스트용(AG-T38 임시, AG-T39·T40에서 제거).
    공식 기본값은 카탈로그 파일 없음·주문번호 15·16자리다."""
    from agent import catalog as catalog_mod
    monkeypatch.delenv("DIRECTON_MOCK_DATA", raising=False)
    monkeypatch.setenv("DIRECTON_CATALOG_PATH", str(catalog_mod.LEGACY_CATALOG_PATH))
    catalog_mod.reset_default_catalog()
    yield
    catalog_mod.reset_default_catalog()
