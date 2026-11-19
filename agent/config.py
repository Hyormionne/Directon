"""상담원 서버 설정. 턴 감지 파라미터는 detector가 가진다. 여기는 응답·세션 동작값만 둔다.

환경변수(접두사 DIRECTON_)로 덮어쓸 수 있다. 문구 목록은 '|'로 구분한다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Mapping

ENV_PREFIX = "DIRECTON_"


@dataclass
class Settings:
    # 캐시 문구
    openers: tuple[str, ...] = ("네, 고객님.",)
    fillers: tuple[str, ...] = ("확인해 드릴게요, 잠시만요.",)
    apology: str = "죄송합니다, 잠시 확인이 어렵습니다. 잠시 후 다시 말씀해 주시겠어요?"
    reprompt: str = "죄송합니다, 잘 듣지 못했어요. 한 번만 더 말씀해 주시겠어요?"
    opener_use_name: bool = False       # True면 "네, ○○ 고객님."(실시간 합성 필요)

    # 타이밍
    precall_silence_ms: int = 200       # 이 침묵에서 finalize·Smart Turn·LLM 선호출 시작
    min_speech_ms: int = 60             # 이보다 짧은 음성은 발화로 보지 않는다(잡음 클릭)
    vad_threshold: float = 0.5
    filler_delay_ms: int = 700          # 툴 실행이 이보다 길면 필러 재생
    safety_timeout_s: float = 20.0      # 확정 → turn.end 전체 상한
    llm_timeout_s: float = 8.0          # LLM 이벤트 사이 최대 대기
    tts_timeout_s: float = 5.0          # 문장 TTS 첫 바이트 최대 대기
    finalize_timeout_s: float = 1.0
    tools_timeout_s: float = 5.0        # GET /tools 대기

    # 응답
    max_tool_rounds: int = 3            # LLM 호출 최대 횟수(마지막 회차는 툴 없이)
    min_sentence_chars: int = 6         # 이보다 짧은 문장 조각은 다음과 합친다
    out_chunk_bytes: int = 3200         # 송신 청크(100ms)
    tts_level_rms: float = 3000.0       # 캐시·실시간 TTS 음성 구간 목표 RMS(0이면 정규화 끔)
    smart_turn_window_s: float = 8.0
    stt_context: str = ""

    # 경로
    log_dir: str = "runs/logs"
    tts_cache_dir: str = "assets/tts_cache"
    tts_voice_key: str = "default"      # 캐시 키 네임스페이스(서버는 여기에 TTS 음성·레벨을 덧붙인다)
    manual_path: str = "cases/manual.md"

    # 서버 시작 예열
    warmup_llm: bool = True             # LLM 체인 모델마다 짧은 요청 1회
    warmup_timeout_s: float = 20.0

    def cached_phrases(self) -> list[str]:
        """서버 시작 때 미리 합성해 둘 문구."""
        out: list[str] = []
        for p in (*self.openers, *self.fillers, self.apology, self.reprompt):
            if p and p not in out:
                out.append(p)
        return out

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        kw = {}
        for f in fields(cls):
            raw = env.get(ENV_PREFIX + f.name.upper())
            if raw is None:
                continue
            default = getattr(cls, f.name)
            if isinstance(default, tuple):
                kw[f.name] = tuple(p.strip() for p in raw.split("|") if p.strip())
            elif isinstance(default, bool):
                kw[f.name] = raw.strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(default, int):
                kw[f.name] = int(raw)
            elif isinstance(default, float):
                kw[f.name] = float(raw)
            else:
                kw[f.name] = raw
        return cls(**kw)
