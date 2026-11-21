"""행사 지정 모델 조립 (AG-T31). 환경변수(또는 인자)로 고른다.

- STT: Soniox 실시간(`SONIOX_MODEL`, 기본 stt-rt-v6)
- LLM: OpenAI(`OPENAI_MODEL`, 기본 gpt-5.4-mini, Responses API)
- TTS: Gemini TTS. `DIRECTON_TTS_PROVIDER`:
    auto(기본) = GEMINI_API_KEY가 있으면 gemini(직접, 스트리밍), 없으면 baze(게이트웨이, 문장 통째 응답)
    gemini     = Gemini API 직접. 모델 기본 gemini-3.8-flash-lite-tts
    baze       = BAZE 게이트웨이 `/audio/speech/`. 모델 기본 gemini-3.8-flash-tts
                 (2026-11-21 실측: 게이트웨이에 flash-lite-tts가 없고 스트리밍도 없다. 첫 바이트 2.2~3.0초)
  모델은 `GEMINI_TTS_MODEL`, 음성은 `GEMINI_TTS_VOICE`, 말투 지시는 `GEMINI_TTS_STYLE`로 바꾼다.

키: 직접 키(`SONIOX_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`)가 있으면 그 서비스는 직접 부르고,
없으면 `BAZE_API_KEY`로 `BAZE_BASE_URL`(기본 https://api.baze.co/v1/gateway)을 거친다.
Soniox `wss://…/soniox/transcribe-websocket`, OpenAI Responses `…/responses`.
쓸 수 있는 키가 없으면 다른 모델로 넘어가지 않고 ValueError를 낸다.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, fields
from typing import Any, Mapping

from agent.providers.baze_tts import BAZE_BASE_URL, BazeTTS
from agent.providers.gemini_tts import GeminiTTS
from agent.providers.openai_llm import OPENAI_MODEL, OpenAILLM
from agent.providers.soniox_stt import SonioxSTT

log = logging.getLogger(__name__)

TTS_MODEL_DIRECT = "gemini-3.8-flash-lite-tts"   # 행사 지정 모델(Gemini API 직접)
TTS_MODEL_BAZE = "gemini-3.8-flash-tts"          # 게이트웨이에 있는 가장 가까운 모델

# 필드명 → 환경변수
ENV_MAP = {
    "gemini_api_key": "GEMINI_API_KEY",
    "soniox_api_key": "SONIOX_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
    "baze_api_key": "BAZE_API_KEY",
    "baze_base_url": "BAZE_BASE_URL",
    "soniox_model": "SONIOX_MODEL",
    "stt_language": "STT_LANGUAGE",
    "openai_model": "OPENAI_MODEL",
    "tts": "DIRECTON_TTS_PROVIDER",
    "tts_model": "GEMINI_TTS_MODEL",
    "tts_voice": "GEMINI_TTS_VOICE",
    "tts_style": "GEMINI_TTS_STYLE",
}


@dataclass
class ProviderConfig:
    gemini_api_key: str = ""
    soniox_api_key: str = ""
    openai_api_key: str = ""
    baze_api_key: str = ""
    baze_base_url: str = BAZE_BASE_URL
    soniox_model: str = "stt-rt-v6"
    stt_language: str = "ko-KR"
    openai_model: str = OPENAI_MODEL
    tts: str = "auto"                 # auto | gemini | baze
    tts_model: str = ""               # ""이면 연결 방식별 기본 모델
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
        return cls.from_env(env_part if env_part else {}, **overrides)


def baze_url(cfg: ProviderConfig, path: str = "") -> str:
    return (cfg.baze_base_url or BAZE_BASE_URL).rstrip("/") + path


def make_stt(cfg: Any = None, *, soniox_connect: Any = None) -> Any:
    cfg = ProviderConfig.coerce(cfg)
    hints = [cfg.stt_language.split("-")[0]]
    if cfg.soniox_api_key:
        return SonioxSTT(api_key=cfg.soniox_api_key, connect=soniox_connect, model=cfg.soniox_model,
                         language_hints=hints)
    if cfg.baze_api_key:
        url = "wss://" + baze_url(cfg, "/soniox/transcribe-websocket").split("://", 1)[1]
        return SonioxSTT(api_key=cfg.baze_api_key, connect=soniox_connect, model=cfg.soniox_model,
                         language_hints=hints, url=url)
    raise ValueError("STT(Soniox) 키가 없습니다: SONIOX_API_KEY 또는 BAZE_API_KEY")


def make_llm(cfg: Any = None, *, openai_client: Any = None) -> Any:
    cfg = ProviderConfig.coerce(cfg)
    if cfg.openai_api_key or openai_client is not None:
        return OpenAILLM(client=openai_client, model=cfg.openai_model, api_key=cfg.openai_api_key)
    if cfg.baze_api_key:
        return OpenAILLM(model=cfg.openai_model, api_key=cfg.baze_api_key, base_url=baze_url(cfg))
    raise ValueError("LLM(OpenAI) 키가 없습니다: OPENAI_API_KEY 또는 BAZE_API_KEY")


def tts_choice(cfg: ProviderConfig, gemini_client: Any = None) -> str:
    choice = (cfg.tts or "auto").strip().lower()
    if choice == "auto":
        if cfg.gemini_api_key or gemini_client is not None:
            return "gemini"
        if cfg.baze_api_key:
            return "baze"
        raise ValueError("TTS 키가 없습니다: GEMINI_API_KEY 또는 BAZE_API_KEY")
    if choice not in ("gemini", "baze"):
        raise ValueError(f"알 수 없는 TTS: {cfg.tts}")
    return choice


def make_tts(cfg: Any = None, *, gemini_client: Any = None) -> Any:
    cfg = ProviderConfig.coerce(cfg)
    choice = tts_choice(cfg, gemini_client)
    if choice == "baze":
        if not cfg.baze_api_key:
            raise ValueError("TTS baze 선택됨, BAZE_API_KEY 없음")
        return BazeTTS(api_key=cfg.baze_api_key, base_url=baze_url(cfg), model=cfg.tts_model or TTS_MODEL_BAZE,
                       voice=cfg.tts_voice, style=cfg.tts_style)
    if gemini_client is None:
        if not cfg.gemini_api_key:
            raise ValueError("TTS gemini 선택됨, GEMINI_API_KEY 없음")
        from agent.providers import _gemini as g
        gemini_client = g.make_gemini_client(cfg.gemini_api_key)
    return GeminiTTS(client=gemini_client, model=cfg.tts_model or TTS_MODEL_DIRECT, voice=cfg.tts_voice,
                     style=cfg.tts_style)
