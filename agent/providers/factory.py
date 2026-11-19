"""환경변수·설정으로 Provider를 고른다(AGENT_PLAN 3.1절).

- STT: SONIOX_API_KEY가 있으면 Soniox, 없으면 Gemini live(연결 실패 시 Gemini batch로 폴백)
- LLM: OPENAI_API_KEY가 있으면 OpenAI(gpt-5.4-mini), 없으면 Gemini(GEMINI_LLM_MODEL)
- TTS: DIRECTON_TTS_PROVIDER(edge|gemini), 기본 edge (AD6: Gemini TTS 무료 티어는 모델당 하루 10요청)
- LLM 폴백(AG-T09): make_llm_with_fallback()이 기본 LLM 뒤에 GEMINI_LLM_FALLBACK_MODELS(쉼표 구분,
  "none"이면 끔)를 붙여 FallbackLLM으로 감싼다. 첫 이벤트 제한은 LLM_FIRST_EVENT_TIMEOUT_S(기본 2.5초).
cfg는 ProviderConfig, dict(필드명 또는 환경변수 이름 키), None(os.environ) 중 하나.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, fields
from typing import Any, Mapping

from agent.providers.edge_tts import EDGE_VOICE, EdgeTTS
from agent.providers.fallback_llm import DEFAULT_FIRST_EVENT_TIMEOUT_S, FallbackLLM
from agent.providers.gemini_llm import LLM_MODEL as GEMINI_LLM_DEFAULT
from agent.providers.gemini_llm import GeminiLLM
from agent.providers.gemini_stt import BATCH_MODEL, LIVE_MODEL, GeminiBatchSTT, GeminiLiveSTT
from agent.providers.gemini_tts import TTS_MODEL, GeminiTTS
from agent.providers.openai_llm import OPENAI_MODEL, OpenAILLM
from agent.providers.soniox_stt import SonioxSTT

log = logging.getLogger(__name__)

# 필드명 → 환경변수
ENV_MAP = {
    "gemini_api_key": "GEMINI_API_KEY",
    "soniox_api_key": "SONIOX_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
    "stt": "STT_PROVIDER",
    "llm": "LLM_PROVIDER",
    "stt_live_model": "GEMINI_STT_LIVE_MODEL",
    "stt_batch_model": "GEMINI_STT_BATCH_MODEL",
    "soniox_model": "SONIOX_MODEL",
    "stt_language": "STT_LANGUAGE",
    "gemini_llm_model": "GEMINI_LLM_MODEL",
    "openai_model": "OPENAI_MODEL",
    "tts_model": "GEMINI_TTS_MODEL",
    "tts_voice": "GEMINI_TTS_VOICE",
    "tts_style": "GEMINI_TTS_STYLE",
    "tts": "DIRECTON_TTS_PROVIDER",
    "edge_voice": "EDGE_TTS_VOICE",
    "llm_fallback_models": "GEMINI_LLM_FALLBACK_MODELS",
    "llm_first_event_timeout_s": "LLM_FIRST_EVENT_TIMEOUT_S",
}

DEFAULT_LLM_FALLBACK = "gemini-3.1-flash-lite,gemini-2.5-flash-lite"


@dataclass
class ProviderConfig:
    gemini_api_key: str = ""
    soniox_api_key: str = ""
    openai_api_key: str = ""
    stt: str = "auto"                 # auto | soniox | gemini_live | gemini_batch
    llm: str = "auto"                 # auto | openai | gemini
    stt_live_model: str = LIVE_MODEL
    stt_batch_model: str = BATCH_MODEL
    stt_fallback: bool = True         # live 연결 실패 시 batch
    soniox_model: str = "stt-rt-v6"
    stt_language: str = "ko-KR"
    gemini_llm_model: str = GEMINI_LLM_DEFAULT
    openai_model: str = OPENAI_MODEL
    tts_model: str = TTS_MODEL
    tts_voice: str = "Kore"
    tts_style: str = ""
    tts: str = "edge"                 # edge | gemini
    edge_voice: str = EDGE_VOICE
    llm_fallback_models: str = DEFAULT_LLM_FALLBACK   # 쉼표 구분, "none"이면 폴백 없음
    llm_first_event_timeout_s: float = DEFAULT_FIRST_EVENT_TIMEOUT_S

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **overrides: Any) -> "ProviderConfig":
        env = os.environ if env is None else env
        kw: dict = {}
        for f, var in ENV_MAP.items():
            v = env.get(var)
            if v not in (None, ""):
                kw[f] = v
        kw.update(overrides)
        return cls(**kw)

    @classmethod
    def coerce(cls, cfg: "ProviderConfig | Mapping[str, Any] | None") -> "ProviderConfig":
        if isinstance(cfg, ProviderConfig):
            return cfg
        if cfg is None:
            return cls.from_env()
        names = {f.name for f in fields(cls)}
        env_part = {k: v for k, v in cfg.items() if k.isupper()}
        overrides = {k: v for k, v in cfg.items() if k in names}
        return cls.from_env(env_part if env_part else None, **overrides)


class FallbackSTT:
    """primary.open_stream이 실패하면 fallback으로 연다."""

    def __init__(self, primary: Any, fallback: Any):
        self.primary, self.fallback = primary, fallback

    async def open_stream(self, context: str = "") -> Any:
        try:
            return await self.primary.open_stream(context)
        except Exception as e:
            log.warning("STT primary failed (%s), using fallback", e)
            return await self.fallback.open_stream(context)


def _gemini(cfg: ProviderConfig, client: Any) -> Any:
    if client is not None:
        return client
    from agent.providers._gemini import make_gemini_client
    return make_gemini_client(cfg.gemini_api_key)


def make_stt(cfg: Any = None, *, gemini_client: Any = None, soniox_connect: Any = None) -> Any:
    cfg = ProviderConfig.coerce(cfg)
    choice = cfg.stt
    if choice == "auto":
        choice = "soniox" if cfg.soniox_api_key else "gemini_live"
    if choice == "soniox":
        if not cfg.soniox_api_key:
            raise ValueError("STT soniox 선택됨, SONIOX_API_KEY 없음")
        return SonioxSTT(api_key=cfg.soniox_api_key, connect=soniox_connect, model=cfg.soniox_model,
                         language_hints=[cfg.stt_language.split("-")[0]])
    client = _gemini(cfg, gemini_client)
    batch = GeminiBatchSTT(client=client, model=cfg.stt_batch_model, language=cfg.stt_language)
    if choice == "gemini_batch":
        return batch
    if choice == "gemini_live":
        live = GeminiLiveSTT(client=client, model=cfg.stt_live_model, language=cfg.stt_language)
        return FallbackSTT(live, batch) if cfg.stt_fallback else live
    raise ValueError(f"알 수 없는 STT: {cfg.stt}")


def make_tts(cfg: Any = None, *, gemini_client: Any = None) -> Any:
    cfg = ProviderConfig.coerce(cfg)
    if cfg.tts == "edge":
        return EdgeTTS(voice=cfg.edge_voice)
    if cfg.tts != "gemini":
        raise ValueError(f"알 수 없는 TTS: {cfg.tts}")
    return GeminiTTS(client=_gemini(cfg, gemini_client), model=cfg.tts_model, voice=cfg.tts_voice,
                     style=cfg.tts_style)


def make_llm(cfg: Any = None, *, gemini_client: Any = None, openai_client: Any = None) -> Any:
    cfg = ProviderConfig.coerce(cfg)
    choice = cfg.llm
    if choice == "auto":
        choice = "openai" if cfg.openai_api_key else "gemini"
    if choice == "openai":
        if not cfg.openai_api_key and openai_client is None:
            raise ValueError("LLM openai 선택됨, OPENAI_API_KEY 없음")
        return OpenAILLM(client=openai_client, model=cfg.openai_model, api_key=cfg.openai_api_key)
    if choice == "gemini":
        return GeminiLLM(client=_gemini(cfg, gemini_client), model=cfg.gemini_llm_model)
    raise ValueError(f"알 수 없는 LLM: {cfg.llm}")


def fallback_chain(cfg: ProviderConfig, primary_model: str = "") -> list[str]:
    raw = (cfg.llm_fallback_models or "").strip()
    if raw.lower() in ("", "none", "off", "0"):
        return []
    out: list[str] = []
    for m in raw.split(","):
        m = m.strip()
        if m and m != primary_model and m not in out:
            out.append(m)
    return out


def make_llm_with_fallback(cfg: Any = None, *, gemini_client: Any = None, openai_client: Any = None) -> Any:
    """기본 LLM + Gemini 대체 모델 체인(FallbackLLM). 체인이 비면 기본 LLM 그대로."""
    cfg = ProviderConfig.coerce(cfg)
    primary = make_llm(cfg, gemini_client=gemini_client, openai_client=openai_client)
    chain = fallback_chain(cfg, getattr(primary, "model", ""))
    if not chain:
        return primary
    if isinstance(primary, GeminiLLM):
        client = primary.client
    elif cfg.gemini_api_key or gemini_client is not None:
        client = _gemini(cfg, gemini_client)
    else:
        return primary
    llms = [primary] + [GeminiLLM(client=client, model=m) for m in chain]
    return FallbackLLM(llms, first_event_timeout_s=float(cfg.llm_first_event_timeout_s))
