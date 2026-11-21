"""상담원 서버 설정. 턴 감지 파라미터는 detector가 가진다. 여기는 응답·세션 동작값만 둔다.

환경변수(접두사 DIRECTON_)로 덮어쓸 수 있다. 문구 목록은 '|'로 구분한다.
AG-T17 당일 교체: DIRECTON_MANUAL은 manual_path의 별칭(DIRECTON_MANUAL_PATH보다 우선).
브랜드·말투 파일(DIRECTON_BRAND, DIRECTON_STYLE)은 agent.prompts.build_system이 직접 읽는다.
AG-T25: 바지인·속도 조절·첫 마디 정책 값도 여기 둔다(DIRECTON_BARGE_IN, DIRECTON_BARGE_IN_MS, DIRECTON_PACE_LEAD_MS,
DIRECTON_OPENER_ADAPTIVE 등).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Mapping

ENV_PREFIX = "DIRECTON_"
# 당일 체크리스트에서 쓰는 짧은 이름 → 필드
ENV_ALIASES = {"manual_path": "DIRECTON_MANUAL"}


@dataclass
class Settings:
    # 캐시 문구
    openers: tuple[str, ...] = ("네, 고객님.",)
    fillers: tuple[str, ...] = ("확인해 드릴게요, 잠시만요.",)
    apology: str = "죄송합니다, 잠시 확인이 어렵습니다. 잠시 후 다시 말씀해 주시겠어요?"
    reprompt: str = "죄송합니다, 잘 듣지 못했어요. 한 번만 더 말씀해 주시겠어요?"
    opener_use_name: bool = False       # True면 "네, ○○ 고객님."(실시간 합성 필요)
    # 텍스트 기반 첫 마디(AG-T10c): 확정 시점 텍스트의 키워드로 고른다(tts_cache.select_opener_for_text).
    # 캐시에 없으면 기본 첫 마디를 쓴다. 조회 첫 마디를 쓴 턴의 툴 필러는 opener_lookup_filler로 바꾼다("확인해" 반복 방지).
    opener_by_text: bool = True
    opener_apology: str = "네, 불편을 드려 죄송합니다."
    opener_lookup: str = "네, 확인해 드릴게요."
    opener_lookup_filler: str = "잠시만 기다려 주세요."

    # 타이밍
    precall_silence_ms: int = 200       # 이 침묵에서 finalize·Smart Turn·LLM 선호출 시작
    precall_refresh: bool = True        # AG-T10c: finalize 후 선호출 재시작·정규화 비교·첫 문장 선합성·결정기에 finalize 텍스트(False면 이전 동작)
    min_speech_ms: int = 60             # 이보다 짧은 음성은 발화로 보지 않는다(잡음 클릭)
    vad_threshold: float = 0.5
    filler_delay_ms: int = 150          # 툴 회차 시작부터 이 안에 다음 LLM 텍스트가 없으면 필러 재생(AG-T10a)
    filler_skip_tools: tuple[str, ...] = ("transfer_to_agent",)   # 이 함수를 부르는 회차엔 필러 없음
    safety_timeout_s: float = 20.0      # 확정 → turn.end 전체 상한
    llm_timeout_s: float = 8.0          # LLM 이벤트 사이 최대 대기
    tts_timeout_s: float = 10.0         # 문장 TTS 첫 바이트 최대 대기(게이트웨이 TTS는 한 번에 응답, 긴 문장 4.7초 실측)
    finalize_timeout_s: float = 1.0
    tools_timeout_s: float = 5.0        # GET /tools 대기

    # 응답
    drafts: bool = True                 # 함수 결과 전에 결과별 답 후보를 받아 결과 뒤 LLM 재호출 없이 말한다(agent.drafts)
    draft_presynth_n: int = 3           # 빈칸 없는 후보 중 가능성 높은 순 몇 개를 결과 전에 미리 합성할지
    max_tool_rounds: int = 3            # LLM 호출 최대 횟수(마지막 회차는 툴 없이)
    ground_number_args: bool = True     # LLM 번호 인자를 고객 발화의 번호에 맞춘다(AG-T10a)
    validate_number_length: bool = True # 스키마 설명의 'N자리'와 다른 번호는 툴에 보내지 않는다(AG-T10a)
    ensure_promised_call: bool = True   # 말로 접수·조회를 약속했는데 함수가 없으면 확인 회차를 한 번 더 돈다
    ensure_transfer_call: bool = True  # 상담원 연결을 말했는데 transfer_to_agent가 없으면 부른다(AG-T10a)
    max_response_chars: int = 300       # 한 응답 상한(공백 제외 글자). 넘으면 LLM 스트림을 끊는다(폭주 생성 방지, AG-T10a)
    min_sentence_chars: int = 6         # 이보다 짧은 문장 조각은 다음과 합친다
    out_chunk_bytes: int = 3200         # 송신 청크(100ms)
    tts_level_rms: float = 3000.0       # 캐시·실시간 TTS 음성 구간 목표 RMS(0이면 정규화 끔)
    smart_turn_window_s: float = 8.0
    # Soniox 문맥 용어(쉼표 구분): 브랜드·제품명이 "하루 온"·"하루원"·"온갖"처럼 깨지는 것을 줄인다(브랜드 정보 기준)
    stt_context: str = ("하루온, 하루온 고객센터, 간 가드 맥스2, 기관지 가드 브리즈콜, 눈가드 맥스4, 눈가드 맥스3, "
                        "맑은 오메가3, 멀티비타민 올인원, 위 가드 루틴, 이눌린 배변 가드, 전립선 가드 포 맨, "
                        "조인플렉스 관절&연골 가드, 콜레스테롤 밸런스, 프로폴리스 프로, 혈당 가드 맥스2, 혈관 순환 세트, "
                        "가뿐 슬림 세트, 갱년기 가드 라폰티신, 환불보장제, 주문번호, 송장번호, 반품, 환불, 입금")

    # 바지인·첫 마디(AG-T19·T20, AG-T25에서 설정으로 옮김). session.start의 "barge_in"/"opener_adaptive"가 있으면 그쪽이 우선.
    # 기본값은 agent.session.SESSION_DEFAULTS와 같다(tests/test_wiring_t25.py).
    barge_in: bool = True               # DIRECTON_BARGE_IN=0으로 끔
    barge_in_ms: int = 300              # 이만큼 연속 음성이면 끼어들기
    barge_in_gap_ms: int = 60           # 연속 음성 사이 이 이하 끊김은 이어진 것으로
    barge_in_arm_ms: int = 1000         # 첫 소리 뒤 이 시간 이후에 시작된 말만 끼어들기로 본다
    barge_in_preroll_ms: int = 200      # 끼어든 말 시작 전 이만큼을 새 턴 STT에 같이 넣는다
    pace_lead_ms: int = 1500            # 응답 오디오를 재생 시계보다 이 이상 앞서 보내지 않는다(0이면 끔)
    pace_cap_s: float = 10.0            # 확정 뒤 이 시간이 지나면 속도 조절 없이 보낸다(turn.end 25초 한도)
    opener_adaptive: bool = False       # AG-T20 첫 마디 정책. 측정 전까지 끔(사용자 결정, DIRECTON_OPENER_ADAPTIVE=1로 켬)
    opener_grace_ms: int = 200          # 선합성이 진행 중이면 이만큼 기다려 본다

    # Jev 인계 판정(AG-T25, 스위치는 DIRECTON_JEV_HANDOFF=1 — agent.handoff_jev)
    jev_handoff_wait_s: float = 0.3     # 응답 시작 때 판정 결과를 기다리는 최대 시간(보통 이미 끝나 있다)

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
        for p in (*self.openers, *self.fillers, self.apology, self.reprompt,
                  self.opener_apology, self.opener_lookup, self.opener_lookup_filler):
            if p and p not in out:
                out.append(p)
        return out

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        kw = {}
        for f in fields(cls):
            raw = env.get(ENV_ALIASES[f.name]) if f.name in ENV_ALIASES else None
            if raw is None:
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
