"""상담원 WebSocket 서버. 실행: uv run python -m agent.server --port 9000

실제 Provider·턴 감지·정규화는 build_deps()에서 지연 import한다(다른 Task의 모듈).
테스트는 fake deps로 handle(ws, deps)만 검증한다.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import logging
from typing import Any, Callable

from agent.config import Settings
from agent.session import Session, SessionDeps

log = logging.getLogger("agent.server")

# build_deps가 찾는 이름(앞에서부터 처음 있는 것을 쓴다)
EXPECTED = {
    "agent.providers.factory": {"stt": ["make_stt"], "tts": ["make_tts"], "llm": ["make_llm"]},
    "agent.turn.vad": {"vad": ["make_vad", "SileroVAD"]},
    "agent.turn.smart_turn": {"smart_turn": ["make_smart_turn", "SmartTurnONNX", "SmartTurnV3"]},
    "agent.turn.detector": {"detector": ["make_detector", "KoreanTurnDetector", "TurnDetector"]},
    "agent.normalize": {"normalize_args": ["normalize_args"]},
}


class DepsImportError(RuntimeError):
    pass


async def handle(ws: Any, deps: SessionDeps) -> None:
    await Session(ws, deps).run()


def _pick(module: str, names: list[str]) -> Any:
    try:
        mod = importlib.import_module(module)
    except ImportError as e:
        raise DepsImportError(f"{module} 를 import할 수 없습니다({e}). "
                              f"해당 Task 구현이 필요합니다: {', '.join(names)}") from e
    for n in names:
        if hasattr(mod, n):
            return getattr(mod, n)
    raise DepsImportError(f"{module} 에 {' / '.join(names)} 중 아무것도 없습니다.")


def _adapt_normalize(fn: Callable) -> Callable:
    """normalize_args(tool, args, schema[, now]) 둘 다 받는다."""
    try:
        n = len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        n = 4
    if n >= 4:
        return fn
    return lambda tool, args, schema, now: fn(tool, args, schema)


def build_deps(settings: Settings | None = None) -> SessionDeps:
    from agent.tools_client import ToolsClient
    from agent.tts_cache import TTSCache
    from agent.turn_log import TurnLogWriter

    s = settings or Settings.from_env()
    f = EXPECTED
    make_stt = _pick("agent.providers.factory", f["agent.providers.factory"]["stt"])
    make_tts = _pick("agent.providers.factory", f["agent.providers.factory"]["tts"])
    make_llm = _pick("agent.providers.factory", f["agent.providers.factory"]["llm"])
    vad_factory = _pick("agent.turn.vad", f["agent.turn.vad"]["vad"])
    smart_cls = _pick("agent.turn.smart_turn", f["agent.turn.smart_turn"]["smart_turn"])
    detector_factory = _pick("agent.turn.detector", f["agent.turn.detector"]["detector"])
    normalize = _adapt_normalize(_pick("agent.normalize", f["agent.normalize"]["normalize_args"]))
    return SessionDeps(stt=make_stt(), tts=make_tts(), llm=make_llm(), vad_factory=vad_factory,
                       smart_turn=smart_cls(), detector_factory=detector_factory,
                       tools=ToolsClient(), cache=TTSCache(s.tts_cache_dir, voice=s.tts_voice_key),
                       settings=s, normalize_args=normalize, log_writer=TurnLogWriter(s.log_dir))


async def serve(port: int, host: str = "0.0.0.0", deps: SessionDeps | None = None) -> None:
    import websockets

    deps = deps or build_deps()
    n = await deps.cache.warm(deps.settings.cached_phrases(), deps.tts)
    log.info("tts cache warm: %d synthesized", n)
    async with websockets.serve(lambda ws: handle(ws, deps), host, port, max_size=None):
        log.info("agent server on :%d", port)
        await asyncio.Future()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:  # noqa: BLE001
        pass
    try:
        asyncio.run(serve(a.port, a.host))
    except DepsImportError as e:
        raise SystemExit(f"의존 모듈 준비 안 됨: {e}")


if __name__ == "__main__":
    main()
