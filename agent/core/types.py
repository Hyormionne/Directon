"""모듈 간 계약. 여기 정의된 시그니처는 Task 사이의 인터페이스이므로 바꾸려면 트랙 문서에 먼저 적는다.

오디오 규약: 모든 PCM은 16kHz, signed 16-bit little-endian, mono bytes.
시간 규약: 세션 내부 시각은 time.perf_counter() 초(float).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Protocol

SAMPLE_RATE = 16000
FRAME_BYTES = 640          # 20ms
FRAME_MS = 20


# ── 세션 ──────────────────────────────────────────────
@dataclass
class SessionInfo:
    session_id: str
    case_id: str = ""
    now: str = ""                       # "2026-11-21 11:00"
    history: list[dict] = field(default_factory=list)
    caller: dict | None = None          # {"name","phone","email"}
    tools_url: str = "http://localhost:8100"

    @classmethod
    def from_start(cls, msg: dict) -> "SessionInfo":
        return cls(session_id=msg["session_id"], case_id=msg.get("case_id", ""),
                   now=msg.get("now", ""), history=msg.get("history") or [],
                   caller=msg.get("caller"), tools_url=msg.get("tools_url") or "http://localhost:8100")


# ── STT ──────────────────────────────────────────────
class STTStream(Protocol):
    async def send_audio(self, pcm: bytes) -> None: ...
    def latest_text(self) -> str: ...                     # 확정 + 미확정 텍스트(즉시 반환)
    async def finalize(self, timeout: float = 1.0) -> str: ...  # 지금까지 오디오의 최종 텍스트
    async def close(self) -> None: ...


class STTProvider(Protocol):
    async def open_stream(self, context: str = "") -> STTStream: ...


# ── TTS ──────────────────────────────────────────────
class TTSProvider(Protocol):
    def synthesize(self, text: str) -> AsyncIterator[bytes]: ...   # 16k PCM 덩어리를 순서대로


# ── LLM ──────────────────────────────────────────────
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)   # role=assistant
    tool_call_id: str = ""                                      # role=tool
    name: str = ""                                              # role=tool: 함수명


@dataclass
class LLMEvent:
    type: Literal["text", "tool_call", "done"]
    text: str = ""
    tool_call: ToolCall | None = None


class LLMProvider(Protocol):
    def stream(self, system: str, messages: list[Message],
               tools: list[dict]) -> AsyncIterator[LLMEvent]: ...
    # tools: OpenAI 형식 [{"type":"function","function":{name,description,parameters}}]


# ── 턴 감지 ──────────────────────────────────────────
@dataclass
class TurnSignals:
    silence_ms: int                 # 마지막 음성 프레임 이후 무음 길이
    speech_ms: int                  # 이번 발화의 누적 음성 길이
    text: str = ""                  # 현재 STT 텍스트(없으면 "")
    smart_turn_p: float | None = None   # 오디오 기반 완료 확률
    expect_short: bool = False      # 직전 상담원 발화가 질문(단답 예상)


@dataclass
class TurnDecision:
    end: bool
    reason: str = ""


class VAD(Protocol):
    # 상태가 있으므로 세션마다 새 인스턴스(인자 없는 생성자/팩토리 지원). TurnDetector도 같다. SmartTurn은 공유 가능.
    def process(self, frame: bytes) -> float: ...      # 20ms 프레임 → 최신 음성 확률(0~1)
    def reset(self) -> None: ...


class SmartTurn(Protocol):
    def predict(self, pcm: bytes) -> float: ...        # 최근 최대 8초 PCM → 발화 완료 확률(0~1)


class TurnDetector(Protocol):
    def decide(self, s: TurnSignals) -> TurnDecision: ...
