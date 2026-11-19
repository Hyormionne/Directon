"""Mock API(함수 서버) 클라이언트. 모든 호출에 X-Session-Id를 붙인다."""
from __future__ import annotations

import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)


def _base(url: str) -> str:
    return url.rstrip("/")


class ToolsClient:
    def __init__(self, http: httpx.AsyncClient | None = None, timeout: float = 10.0):
        self.http = http or httpx.AsyncClient(timeout=timeout)
        self.timeout = timeout
        self._tools: dict[str, list[dict]] = {}

    async def list_tools(self, tools_url: str) -> list[dict]:
        """GET {tools_url}/tools (URL별 캐시). 실패하면 []를 주고 캐시하지 않는다."""
        key = _base(tools_url)
        if key in self._tools:
            return self._tools[key]
        try:
            r = await self.http.get(f"{key}/tools", timeout=self.timeout)
            r.raise_for_status()
            tools = r.json()
            if not isinstance(tools, list):
                raise ValueError("tools is not a list")
        except Exception as e:  # noqa: BLE001
            log.warning("list_tools failed %s: %r", key, e)
            return []
        self._tools[key] = tools
        return tools

    async def call(self, tools_url: str, session_id: str, name: str, args: dict[str, Any]) -> dict:
        """POST {tools_url}/tools/{name}. 실패해도 예외 없이 {"status":"error",...}."""
        url = f"{_base(tools_url)}/tools/{name}"
        try:
            r = await self.http.post(url, json=args, headers={"X-Session-Id": session_id},
                                     timeout=self.timeout)
        except Exception as e:  # noqa: BLE001
            log.warning("tool %s failed: %r", name, e)
            return {"status": "error", "message": "network error"}
        try:
            body = r.json()
        except Exception:  # noqa: BLE001
            body = None
        if r.status_code >= 400:
            out = {"status": "error", "http_status": r.status_code}
            if isinstance(body, dict):
                out.update({k: v for k, v in body.items() if k != "status"})
            return out
        if body is None:
            return {"status": "error", "message": "invalid response"}
        return body if isinstance(body, dict) else {"status": "ok", "result": body}

    async def aclose(self) -> None:
        await self.http.aclose()
