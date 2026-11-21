"""함수 호출 클라이언트 + 후보 병렬 검증.

- 모든 호출에 X-Session-Id 를 코드가 자동으로 붙인다 (빠지면 '호출 안 함'으로 채점됨).
- 같은 함수의 후보 인수 여러 개를 동시에 보내고, 'ok' 가 나온 것 중 순위가 가장 높은 결과를 쓴다.
  중계가 함수마다 고정 지연(1.0~1.5초)을 넣으므로 병렬로 보내면 지연은 한 번만 든다.
- 같은 인수의 중복 호출은 캐시로 막는다.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

import httpx

from .planner import Call, LOOKUP_FUNCS


@dataclass
class CallRecord:
    name: str
    arguments: dict
    result: dict
    t_start: float
    t_end: float


@dataclass
class ToolClient:
    tools_url: str
    session_id: str
    timeout: float = 8.0
    log: list[CallRecord] = field(default_factory=list)
    _client: httpx.AsyncClient | None = None
    _inflight: dict = field(default_factory=dict)

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=self.timeout, headers={"X-Session-Id": self.session_id})
        return self

    async def __aexit__(self, *exc):
        await self._client.aclose()

    def _key(self, name, args):
        return name + json.dumps(args, sort_keys=True, ensure_ascii=False)

    async def call(self, name: str, args: dict) -> dict:
        """같은 호출이 진행 중이면 그 결과를 같이 기다린다(선호출과 LLM 호출이 겹칠 때)."""
        k = self._key(name, args)
        if k not in self._inflight:
            self._inflight[k] = asyncio.ensure_future(self._do(name, args))
        return await self._inflight[k]

    async def _do(self, name, args):
        t0 = time.perf_counter()
        try:
            r = await self._client.post(f"{self.tools_url.rstrip('/')}/tools/{name}", json=args)
            res = r.json()
        except Exception as e:                      # 네트워크 실패도 응대는 이어가야 한다
            res = {"status": "error", "message": repr(e)[:200]}
        self.log.append(CallRecord(name, args, res, t0, time.perf_counter()))
        return res


async def run_lookups(tc: ToolClient, calls: list[Call]) -> dict[str, tuple[Call, dict]]:
    """조회 후보들을 한꺼번에 보내고 함수별로 가장 좋은 (호출, 결과)를 고른다.
    순위: status ok > already_requested > not_found/invalid/error, 같은 등급이면 먼저 나온 후보."""
    if not calls:
        return {}
    results = await asyncio.gather(*(tc.call(c.name, c.arguments) for c in calls))
    rank = {"ok": 0, "already_requested": 1}
    best: dict[str, tuple[int, int, Call, dict]] = {}
    for i, (c, r) in enumerate(zip(calls, results)):
        score = (rank.get(r.get("status"), 5), i)
        if c.name not in best or score < best[c.name][:2]:
            best[c.name] = (*score, c, r)
    return {k: (v[2], v[3]) for k, v in best.items()}


def start_lookups(tc: ToolClient, calls: list[Call]) -> asyncio.Task:
    """엔드포인트 직후(또는 번호가 다 들린 순간) 바로 띄워 두고, 나중에 await 한다."""
    return asyncio.ensure_future(run_lookups(tc, [c for c in calls if c.name in LOOKUP_FUNCS]))
