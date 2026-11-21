"""환경변수·설정으로 Provider를 고른다(AGENT_PLAN 3.1절).

- STT: SONIOX_API_KEY가 있으면 Soniox, 없으면 Gemini live(연결 실패 시 Gemini batch로 폴백)
- LLM: OPENAI_API_KEY가 있으면 OpenAI(gpt-5.4-mini), 없으면 Gemini(GEMINI_LLM_MODEL)
- TTS: DIRECTON_TTS_PROVIDER(edge|gemini), 기본 edge (AD6: Gemini TTS 무료 티어는 모델당 하루 10요청)
- LLM 폴백(AG-T09): make_llm_with_fallback()이 기본 LLM 뒤에 GEMINI_LLM_FALLBACK_MODELS(쉼표 구분,
  "none"이면 끔)를 붙여 FallbackLLM으로 감싼다. 첫 이벤트 제한은 LLM_FIRST_EVENT_TIMEOUT_S(기본 2.5초).
- LLM 동시 요청(AG-T15c, docs/decisions/llm_latency_r2.md): LLM_MODE=hedge|race|fallback|single.
  hedge = 기본 모델에 보내고 LLM_HEDGE_DELAY_MS 안에 첫 이벤트가 없으면 대체 모델에도 보내 먼저 온 쪽 채택
  (HedgedLLM, FallbackLLM 상속). race = 처음부터 둘 다. single = 폴백 없이 기본 모델만.
  서버 build_deps는 make_llm_with_fallback을 찾으므로 모드 전환은 이 함수 안에서 한다.
- 컨텍스트 캐시(AG-T15c): LLM_CONTEXT_CACHE=1이면 Gemini LLM들이 ContextCache 하나를 공유한다(모델별 키).
- STT 튜닝(AG-T10b, docs/decisions/stt_tuning.md): gemini_live는 DigitRecheckSTT로 감싼다.
  STT_DIGIT_INTERIM_FIX(기본 1: 확정 숫자 자릿수가 틀리면 직전 부분 결과로 복원, 지연 0),
  STT_DIGIT_RECHECK(기본 0: batch 재인식, +1.5~3초). live 노브: STT_LIVE_CHUNK_MS, STT_VOCABULARY,
  STT_LIVE_MODE, STT_VAD_SILENCE_MS, STT_VAD_END_SENSITIVITY, STT_LANGUAGE("auto"=자동 감지).
- BAZE 게이트웨이(AG-T16, AD9): 직접 키가 없는 서비스는 BAZE_API_KEY로 BAZE_BASE_URL을 거친다.
  Soniox → .../soniox/transcribe-websocket, OpenAI → base_url, Gemini → .../gemini.
  게이트웨이에 transcribe 모델·gemini-2.5-flash-lite가 없어 STT는 Soniox, 기본 폴백은 3.1-flash-lite만.
  DIRECTON_TTS_PROVIDER=baze면 BazeTTS(.../audio/speech/).
- TTS 대체(AG-T18, docs/decisions/stability.md): 서버 build_deps는 make_tts_with_fallback을 찾는다.
  1차가 edge이고 GEMINI_API_KEY(또는 주입한 gemini_client)가 있으면 FallbackTTS(edge → GeminiTTS)로 감싼다.
  DIRECTON_TTS_FALLBACK=none이면 끔. 첫 덩어리 제한 TTS_FALLBACK_TIMEOUT_S(기본 2.0), 1차 쿨다운
  TTS_FALLBACK_COOLDOWN_S(기본 30). BAZE는 비활성(토큰 소진)이라 BAZE 키만으로는 대체를 붙이지 않는다.
cfg는 ProviderConfig, dict(필드명 또는 환경변수 이름 키), None(os.environ) 중 하나.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, fields
from typing import Any, Mapping

from agent.providers.baze_tts import BAZE_BASE_URL, BazeTTS
from agent.providers.edge_tts import EDGE_VOICE, EdgeTTS
from agent.providers.fallback_llm import DEFAULT_FIRST_EVENT_TIMEOUT_S, FallbackLLM
from agent.providers.fallback_tts import DEFAULT_COOLDOWN_S, DEFAULT_FIRST_CHUNK_TIMEOUT_S, FallbackTTS
from agent.providers.gemini_llm import LLM_MODEL as GEMINI_LLM_DEFAULT
from agent.providers.gemini_llm import GeminiLLM
from agent.providers.gemini_stt import BATCH_MODEL, LIVE_MODEL, GeminiBatchSTT, GeminiLiveSTT
from agent.providers.gemini_tts import TTS_MODEL, GeminiTTS
from agent.providers.hedged_llm import DEFAULT_HEDGE_DELAY_S, HedgedLLM
from agent.providers.llm_cache import DEFAULT_TTL_S, ContextCache
from agent.providers.openai_llm import OPENAI_MODEL, OpenAILLM
from agent.providers.soniox_stt import SonioxSTT
from agent.providers.stt_recheck import RECHECK_PROMPT, DigitRecheckSTT

log = logging.getLogger(__name__)

# 필드명 → 환경변수
ENV_MAP = {
    "gemini_api_key": "GEMINI_API_KEY",
    "soniox_api_key": "SONIOX_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
    "baze_api_key": "BAZE_API_KEY",
    "baze_base_url": "BAZE_BASE_URL",
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
    "tts_fallback": "DIRECTON_TTS_FALLBACK",
    "tts_fallback_timeout_s": "TTS_FALLBACK_TIMEOUT_S",
    "tts_fallback_cooldown_s": "TTS_FALLBACK_COOLDOWN_S",
    "llm_fallback_models": "GEMINI_LLM_FALLBACK_MODELS",
    "llm_first_event_timeout_s": "LLM_FIRST_EVENT_TIMEOUT_S",
    "llm_mode": "LLM_MODE",
    "llm_hedge_delay_ms": "LLM_HEDGE_DELAY_MS",
    "llm_context_cache": "LLM_CONTEXT_CACHE",
    "llm_cache_ttl_s": "LLM_CACHE_TTL_S",
    "llm_same_model_hedge": "LLM_SAME_MODEL_HEDGE",
    "stt_live_chunk_ms": "STT_LIVE_CHUNK_MS",
    "stt_vocabulary": "STT_VOCABULARY",
    "stt_live_mode": "STT_LIVE_MODE",
    "stt_vad_silence_ms": "STT_VAD_SILENCE_MS",
    "stt_vad_end_sensitivity": "STT_VAD_END_SENSITIVITY",
    "stt_digit_interim_fix": "STT_DIGIT_INTERIM_FIX",
    "stt_digit_recheck": "STT_DIGIT_RECHECK",
    "stt_recheck_model": "STT_RECHECK_MODEL",
    "stt_recheck_timeout_s": "STT_RECHECK_TIMEOUT_S",
}

DEFAULT_LLM_FALLBACK = "gemini-3.1-flash-lite,gemini-2.5-flash-lite"
DEFAULT_LLM_MODE = "hedge"              # AG-T15c 실측(docs/decisions/llm_latency_r2.md): h=500ms + 캐시
DEFAULT_LLM_CONTEXT_CACHE = True        # AG-T15c 실측: 입력 비용 약 -70~80%, p50 +0.1s 안팎
BAZE_LLM_FALLBACK = "gemini-3.1-flash-lite"   # 게이트웨이에 2.5-flash-lite 없음(2026-11-19 실측 404)
DEFAULT_STT_VOCABULARY = ""
DEFAULT_DIGIT_RECHECK = False
RECHECK_MODEL = "gemini-3.5-flash-lite"
DEFAULT_RECHECK_TIMEOUT_S = 1.5


def _truthy(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


@dataclass
class ProviderConfig:
    gemini_api_key: str = ""
    soniox_api_key: str = ""
    openai_api_key: str = ""
    baze_api_key: str = ""
    baze_base_url: str = BAZE_BASE_URL
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
    tts: str = "edge"                 # edge | gemini | baze
    edge_voice: str = EDGE_VOICE
    tts_fallback: str = "gemini"      # gemini | none (1차 edge 실패 시 대체 TTS, AG-T18)
    tts_fallback_timeout_s: float = DEFAULT_FIRST_CHUNK_TIMEOUT_S
    tts_fallback_cooldown_s: float = DEFAULT_COOLDOWN_S
    llm_fallback_models: str = DEFAULT_LLM_FALLBACK   # 쉼표 구분, "none"이면 폴백 없음
    llm_first_event_timeout_s: float = DEFAULT_FIRST_EVENT_TIMEOUT_S
    llm_mode: str = DEFAULT_LLM_MODE                # hedge | race | fallback | single
    llm_hedge_delay_ms: float = DEFAULT_HEDGE_DELAY_S * 1000
    llm_context_cache: Any = DEFAULT_LLM_CONTEXT_CACHE
    llm_cache_ttl_s: int = DEFAULT_TTL_S
    llm_same_model_hedge: Any = True                # AG-T41: OpenAI 기본이면 대체 모델 대신 같은 모델 사본에 hedge
    stt_live_chunk_ms: int = 100                    # 문서 권장 100ms. 50·40·200 실험에서 유의한 차이 없음(AG-T10b)
    stt_vocabulary: str = DEFAULT_STT_VOCABULARY    # 쉼표 구분 기본 어휘, "none"이면 없음
    stt_live_mode: str = ""                         # "" | VERBATIM | SMART
    stt_vad_silence_ms: int = 0                     # 0이면 서버 기본
    stt_vad_end_sensitivity: str = ""               # "" | LOW | HIGH
    stt_digit_interim_fix: Any = True                # AG-T10b: 확정 숫자 자릿수가 틀리면 직전 부분 결과로 복원(0/1)
    stt_digit_recheck: Any = DEFAULT_DIGIT_RECHECK   # AG-T10b: 자릿수 불일치 시 batch 재인식(0/1)
    stt_recheck_model: str = RECHECK_MODEL
    stt_recheck_timeout_s: float = DEFAULT_RECHECK_TIMEOUT_S

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


def baze_url(cfg: ProviderConfig, path: str = "") -> str:
    return (cfg.baze_base_url or BAZE_BASE_URL).rstrip("/") + path


def _gemini(cfg: ProviderConfig, client: Any) -> Any:
    if client is not None:
        return client
    from agent.providers import _gemini as g
    if not cfg.gemini_api_key and cfg.baze_api_key:
        return g.make_gemini_client(cfg.baze_api_key, base_url=baze_url(cfg, "/gemini"))
    return g.make_gemini_client(cfg.gemini_api_key)


def make_stt(cfg: Any = None, *, gemini_client: Any = None, soniox_connect: Any = None) -> Any:
    cfg = ProviderConfig.coerce(cfg)
    choice = cfg.stt
    if choice == "auto":
        choice = "soniox" if (cfg.soniox_api_key or cfg.baze_api_key) else "gemini_live"
    if choice == "soniox":
        hints = [cfg.stt_language.split("-")[0]]
        if cfg.soniox_api_key:
            return SonioxSTT(api_key=cfg.soniox_api_key, connect=soniox_connect, model=cfg.soniox_model,
                             language_hints=hints)
        if cfg.baze_api_key:
            url = "wss://" + baze_url(cfg, "/soniox/transcribe-websocket").split("://", 1)[1]
            return SonioxSTT(api_key=cfg.baze_api_key, connect=soniox_connect, model=cfg.soniox_model,
                             language_hints=hints, url=url)
        raise ValueError("STT soniox 선택됨, SONIOX_API_KEY·BAZE_API_KEY 없음")
    client = _gemini(cfg, gemini_client)
    batch = GeminiBatchSTT(client=client, model=cfg.stt_batch_model, language=cfg.stt_language)
    if choice == "gemini_batch":
        return batch
    if choice == "gemini_live":
        live = GeminiLiveSTT(client=client, model=cfg.stt_live_model, language=cfg.stt_language,
                             chunk_ms=int(cfg.stt_live_chunk_ms), vocabulary=_vocab(cfg.stt_vocabulary),
                             mode=(cfg.stt_live_mode or "").upper(), vad=_vad(cfg))
        stt = FallbackSTT(live, batch) if cfg.stt_fallback else live
        return _with_recheck(cfg, client, stt)
    raise ValueError(f"알 수 없는 STT: {cfg.stt}")


def _vocab(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if raw.lower() in ("", "none", "off", "0"):
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def _vad(cfg: ProviderConfig) -> dict:
    vad: dict = {}
    if int(cfg.stt_vad_silence_ms or 0) > 0:
        vad["silence_duration_ms"] = int(cfg.stt_vad_silence_ms)
    sens = (cfg.stt_vad_end_sensitivity or "").strip().upper()
    if sens in ("LOW", "HIGH"):
        vad["end_of_speech_sensitivity"] = f"END_SENSITIVITY_{sens}"
    return vad


def _with_recheck(cfg: ProviderConfig, client: Any, stt: Any) -> Any:
    interim = _truthy(cfg.stt_digit_interim_fix)
    batch = _truthy(cfg.stt_digit_recheck)
    if not (interim or batch):
        return stt
    timeout = float(cfg.stt_recheck_timeout_s)
    recheck = None
    if batch:
        rb = GeminiBatchSTT(client=client, model=cfg.stt_recheck_model, language=cfg.stt_language,
                            prompt=RECHECK_PROMPT, thinking_level=_recheck_thinking(cfg.stt_recheck_model),
                            request_timeout=timeout)
        recheck = rb.transcribe
    return DigitRecheckSTT(stt, recheck, timeout=timeout, recheck_model=cfg.stt_recheck_model,
                           interim_fix=interim)


def _recheck_thinking(model: str) -> str | None:
    return "MINIMAL" if "flash" in model and "transcribe" not in model else None


def make_tts(cfg: Any = None, *, gemini_client: Any = None) -> Any:
    cfg = ProviderConfig.coerce(cfg)
    if cfg.tts == "edge":
        return EdgeTTS(voice=cfg.edge_voice)
    if cfg.tts == "baze":
        if not cfg.baze_api_key:
            raise ValueError("TTS baze 선택됨, BAZE_API_KEY 없음")
        return BazeTTS(api_key=cfg.baze_api_key, base_url=baze_url(cfg), model=cfg.tts_model,
                       voice=cfg.tts_voice, style=cfg.tts_style)
    if cfg.tts != "gemini":
        raise ValueError(f"알 수 없는 TTS: {cfg.tts}")
    return GeminiTTS(client=_gemini(cfg, gemini_client), model=cfg.tts_model, voice=cfg.tts_voice,
                     style=cfg.tts_style)


def make_tts_with_fallback(cfg: Any = None, *, gemini_client: Any = None) -> Any:
    """서버 build_deps의 TTS 진입점(AG-T18). 1차 edge + 2차 Gemini TTS(FallbackTTS).
    1차가 edge가 아니거나, Gemini 키·클라이언트가 없거나, DIRECTON_TTS_FALLBACK=none이면 make_tts 그대로."""
    cfg = ProviderConfig.coerce(cfg)
    primary = make_tts(cfg, gemini_client=gemini_client)
    fb = (cfg.tts_fallback or "").strip().lower()
    if fb in ("", "none", "off", "0") or not isinstance(primary, EdgeTTS):
        return primary
    if fb != "gemini":
        raise ValueError(f"알 수 없는 DIRECTON_TTS_FALLBACK: {cfg.tts_fallback}")
    if not (cfg.gemini_api_key or gemini_client is not None):
        return primary
    secondary = GeminiTTS(client=_gemini(cfg, gemini_client), model=cfg.tts_model, voice=cfg.tts_voice,
                          style=cfg.tts_style)
    return FallbackTTS(primary, secondary, first_chunk_timeout_s=float(cfg.tts_fallback_timeout_s),
                       cooldown_s=float(cfg.tts_fallback_cooldown_s))


def make_llm(cfg: Any = None, *, gemini_client: Any = None, openai_client: Any = None) -> Any:
    cfg = ProviderConfig.coerce(cfg)
    choice = cfg.llm
    if choice == "auto":
        choice = "openai" if (cfg.openai_api_key or cfg.baze_api_key) else "gemini"
    if choice == "openai":
        if cfg.openai_api_key or openai_client is not None:
            return OpenAILLM(client=openai_client, model=cfg.openai_model, api_key=cfg.openai_api_key)
        if cfg.baze_api_key:
            return OpenAILLM(model=cfg.openai_model, api_key=cfg.baze_api_key, base_url=baze_url(cfg))
        raise ValueError("LLM openai 선택됨, OPENAI_API_KEY·BAZE_API_KEY 없음")
    if choice == "gemini":
        return GeminiLLM(client=_gemini(cfg, gemini_client), model=cfg.gemini_llm_model)
    raise ValueError(f"알 수 없는 LLM: {cfg.llm}")


def fallback_chain(cfg: ProviderConfig, primary_model: str = "") -> list[str]:
    raw = (cfg.llm_fallback_models or "").strip()
    if raw == DEFAULT_LLM_FALLBACK and cfg.baze_api_key and not cfg.gemini_api_key:
        raw = BAZE_LLM_FALLBACK
    if raw.lower() in ("", "none", "off", "0"):
        return []
    out: list[str] = []
    for m in raw.split(","):
        m = m.strip()
        if m and m != primary_model and m not in out:
            out.append(m)
    return out


def _attach_cache(cfg: ProviderConfig, llms: list[Any]) -> None:
    gems = [m for m in llms if isinstance(m, GeminiLLM)]
    if not gems or not _truthy(cfg.llm_context_cache):
        return
    cache = ContextCache(client=gems[0].client, ttl_s=int(cfg.llm_cache_ttl_s))
    for m in gems:
        m.context_cache = cache


def _llm_chain(cfg: ProviderConfig, gemini_client: Any, openai_client: Any) -> list[Any]:
    primary = make_llm(cfg, gemini_client=gemini_client, openai_client=openai_client)
    if isinstance(primary, OpenAILLM):
        # AG-T41(AD17): 대회 규정 LLM은 gpt-5.4-mini 하나. Gemini 폴백을 붙이지 않는다.
        # hedge·race·fallback은 같은 모델 사본(같은 키·경로)으로만 한다(OpenAI는 store=False라 사본끼리 상태 없음).
        if not _truthy(cfg.llm_same_model_hedge):
            return [primary]
        replica = OpenAILLM(client=primary.client, model=primary.model, reasoning_effort=primary.reasoning_effort,
                            max_output_tokens=primary.max_output_tokens, api_key=primary.api_key,
                            base_url=primary.base_url)
        return [primary, replica]
    chain = fallback_chain(cfg, getattr(primary, "model", ""))
    if not chain:
        return [primary]
    if isinstance(primary, GeminiLLM):
        client = primary.client
    elif cfg.gemini_api_key or cfg.baze_api_key or gemini_client is not None:
        client = _gemini(cfg, gemini_client)
    else:
        return [primary]
    return [primary] + [GeminiLLM(client=client, model=m) for m in chain]


DEFAULT_ALLOWED_LLM_MODELS = OPENAI_MODEL    # AG-T41(AD17): 대회 규정 LLM


def allowed_llm_models(env: Mapping[str, str] | None = None) -> str:
    """LLM_ALLOWED_MODELS(쉼표 구분, '*'면 검사 끔). 기본은 대회 규정 모델."""
    env = os.environ if env is None else env
    return (env.get("LLM_ALLOWED_MODELS") or "").strip() or DEFAULT_ALLOWED_LLM_MODELS


def llm_models(llm: Any) -> list[str]:
    """LLM 객체(FallbackLLM·HedgedLLM 포함)가 실제로 부를 수 있는 모델 이름들."""
    return [str(getattr(m, "model", m)) for m in (getattr(llm, "llms", None) or [llm])]


def check_llm_policy(llm: Any, allowed: str) -> None:
    """허용 목록 밖 모델이 체인에 하나라도 있으면 ValueError."""
    allowed = (allowed or "").strip()
    if allowed == "*":
        return
    ok = {m.strip() for m in allowed.split(",") if m.strip()}
    bad = [m for m in llm_models(llm) if m not in ok]
    if bad:
        raise ValueError(f"허용되지 않은 LLM 모델: {bad} (허용: {sorted(ok)}, LLM_ALLOWED_MODELS로 변경)")


def make_llm_hedged(cfg: Any = None, *, gemini_client: Any = None, openai_client: Any = None,
                    mode: str = "") -> Any:
    """기본 LLM + 대체 모델 동시 요청(HedgedLLM). 체인이 비면 기본 LLM 그대로."""
    cfg = ProviderConfig.coerce(cfg)
    llms = _llm_chain(cfg, gemini_client, openai_client)
    _attach_cache(cfg, llms)
    if len(llms) == 1:
        return llms[0]
    m = (mode or cfg.llm_mode or "hedge").strip().lower()
    return HedgedLLM(llms, hedge_delay_s=float(cfg.llm_hedge_delay_ms) / 1000.0,
                     mode="race" if m == "race" else "hedge",
                     first_event_timeout_s=float(cfg.llm_first_event_timeout_s))


def make_llm_with_fallback(cfg: Any = None, *, gemini_client: Any = None, openai_client: Any = None) -> Any:
    """서버 build_deps의 LLM 진입점. LLM_MODE에 따라
    fallback: 기본 LLM + Gemini 대체 모델 체인(FallbackLLM) / hedge·race: HedgedLLM / single: 기본 LLM만.
    체인이 비면 기본 LLM 그대로."""
    cfg = ProviderConfig.coerce(cfg)
    mode = (cfg.llm_mode or DEFAULT_LLM_MODE).strip().lower()
    if mode in ("hedge", "race"):
        return make_llm_hedged(cfg, gemini_client=gemini_client, openai_client=openai_client, mode=mode)
    if mode == "single":
        primary = make_llm(cfg, gemini_client=gemini_client, openai_client=openai_client)
        _attach_cache(cfg, [primary])
        return primary
    if mode != "fallback":
        raise ValueError(f"알 수 없는 LLM_MODE: {cfg.llm_mode}")
    llms = _llm_chain(cfg, gemini_client, openai_client)
    _attach_cache(cfg, llms)
    if len(llms) == 1:
        return llms[0]
    return FallbackLLM(llms, first_event_timeout_s=float(cfg.llm_first_event_timeout_s))
