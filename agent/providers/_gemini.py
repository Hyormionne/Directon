"""Gemini 클라이언트 생성(지연 import). 테스트는 fake client를 주입하므로 여기를 거치지 않는다."""
from __future__ import annotations

import os
from typing import Any


def make_gemini_client(api_key: str = "", base_url: str = "") -> Any:
    """base_url을 주면 그 주소로 보낸다(AG-T16: BAZE 게이트웨이 `.../gemini`)."""
    from google import genai

    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    opts: dict = {"api_version": "v1beta"}
    if base_url:
        opts["base_url"] = base_url
    return genai.Client(api_key=key or None, http_options=opts)
