"""AG-T07 tools_client: GET /tools 캐시, POST /tools/{name} + X-Session-Id, 오류 처리."""
import json

import httpx
import pytest

from agent.tools_client import ToolsClient

TOOLS = [{"type": "function", "function": {
    "name": "get_order_info", "description": "주문 조회",
    "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                   "required": ["order_id"]}}}]


def make_client(handler):
    return ToolsClient(http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def test_list_tools_cached_per_url():
    hits = []

    def handler(req: httpx.Request):
        hits.append(str(req.url))
        assert req.method == "GET" and req.url.path == "/tools"
        return httpx.Response(200, json=TOOLS)

    c = make_client(handler)
    assert await c.list_tools("http://mock:8100") == TOOLS
    assert await c.list_tools("http://mock:8100/") == TOOLS     # 끝 슬래시는 같은 URL
    assert len(hits) == 1
    await c.list_tools("http://other:8100")
    assert len(hits) == 2


async def test_list_tools_error_returns_empty_and_not_cached():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("down", request=req)
        return httpx.Response(200, json=TOOLS)

    c = make_client(handler)
    assert await c.list_tools("http://mock:8100") == []
    assert await c.list_tools("http://mock:8100") == TOOLS


async def test_call_posts_with_session_header_and_returns_json():
    seen = {}

    def handler(req: httpx.Request):
        seen["method"] = req.method
        seen["path"] = req.url.path
        seen["sid"] = req.headers.get("X-Session-Id")
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"status": "ok", "order": {"id": "2611150231"}})

    c = make_client(handler)
    out = await c.call("http://mock:8100/", "S-1", "get_order_info", {"order_id": "2611150231"})
    assert out == {"status": "ok", "order": {"id": "2611150231"}}
    assert seen == {"method": "POST", "path": "/tools/get_order_info", "sid": "S-1",
                    "body": {"order_id": "2611150231"}}


async def test_call_network_error_returns_status_error():
    def handler(req):
        raise httpx.ConnectError("refused", request=req)

    out = await make_client(handler).call("http://mock:8100", "S-1", "x", {})
    assert out["status"] == "error"


async def test_call_non_json_or_http_error_returns_status_error():
    def handler(req):
        if req.url.path.endswith("/bad"):
            return httpx.Response(200, text="not json")
        return httpx.Response(404, json={"detail": "unknown tool"})

    c = make_client(handler)
    assert (await c.call("http://m", "S", "bad", {}))["status"] == "error"
    out = await c.call("http://m", "S", "nope", {})
    assert out["status"] == "error" and out.get("http_status") == 404
