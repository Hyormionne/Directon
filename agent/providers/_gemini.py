"""Gemini 클라이언트 생성(지연 import). 테스트는 fake client를 주입하므로 여기를 거치지 않는다."""
from __future__ import annotations

import os
from typing import Any


def make_gemini_client(api_key: str = "") -> Any:
    from google import genai

    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    return genai.Client(api_key=key or None, http_options={"api_version": "v1beta"})
