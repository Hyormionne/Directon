"""행사 지정 모델 세 개만 만든다(대체·폴백 없음).

- STT: Soniox `stt-rt-v6` (SONIOX_API_KEY)
- TTS: Gemini `gemini-3.8-flash-tts` (GEMINI_API_KEY, Q&A Q8: lite 아님)
- LLM: OpenAI `gpt-5.4-mini` (OPENAI_API_KEY)

모델 이름은 고정이다. 음성·말투(GEMINI_TTS_VOICE, GEMINI_TTS_STYLE)만 환경변수로 바꾼다.
cfg는 ProviderConfig, dict(필드명 또는 환경변수 이름 키), None(os.environ) 중 하나.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any, Mapping

from agent.providers.gemini_tts import TTS_MODEL, GeminiTTS
from agent.providers.openai_llm import OPENAI_MODEL, OpenAILLM
from agent.providers.soniox_stt import SonioxSTT

STT_MODEL = "stt-rt-v6"

# 필드명 → 환경변수
ENV_MAP = {
    "gemini_api_key": "GEMINI_API_KEY",
    "soniox_api_key": "SONIOX_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
    "stt_language": "STT_LANGUAGE",
    "tts_voice": "GEMINI_TTS_VOICE",
    "tts_style": "GEMINI_TTS_STYLE",
}


@dataclass
class ProviderConfig:
    gemini_api_key: str = ""
    soniox_api_key: str = ""
    openai_api_key: str = ""
    stt_language: str = "ko-KR"
    tts_voice: str = "Kore"
    tts_style: str = ""

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


def make_stt(cfg: Any = None, *, soniox_connect: Any = None) -> Any:
    cfg = ProviderConfig.coerce(cfg)
    if not cfg.soniox_api_key:
        raise ValueError("SONIOX_API_KEY 없음")
    return SonioxSTT(api_key=cfg.soniox_api_key, connect=soniox_connect, model=STT_MODEL,
                     language_hints=[cfg.stt_language.split("-")[0]])


def make_tts(cfg: Any = None, *, gemini_client: Any = None) -> Any:
    cfg = ProviderConfig.coerce(cfg)
    if gemini_client is None:
        if not cfg.gemini_api_key:
            raise ValueError("GEMINI_API_KEY 없음")
        from agent.providers._gemini import make_gemini_client
        gemini_client = make_gemini_client(cfg.gemini_api_key)
    return GeminiTTS(client=gemini_client, model=TTS_MODEL, voice=cfg.tts_voice, style=cfg.tts_style)


def make_llm(cfg: Any = None, *, openai_client: Any = None) -> Any:
    cfg = ProviderConfig.coerce(cfg)
    if openai_client is None and not cfg.openai_api_key:
        raise ValueError("OPENAI_API_KEY 없음")
    return OpenAILLM(client=openai_client, model=OPENAI_MODEL, api_key=cfg.openai_api_key)
