"""AG-T09: 실제 구성 조립(build_deps)·LLM 폴백·첫 소리 레벨·서버 예열. 네트워크 없음."""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from agent.config import Settings
from agent.core.types import LLMEvent, Message, ToolCall

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MEASURE_RMS = 300.0        # 측정기 무음 기준(바이너리 메시지 단위)


# ── fakes ────────────────────────────────────────────
class FakeSTT:
    def __init__(self):
        self.opened = 0

    async def open_stream(self, context: str = ""):
        self.opened += 1
        return FakeStream()


class FakeStream:
    closed = False

    async def send_audio(self, pcm):
        pass

    def latest_text(self):
        return ""

    async def finalize(self, timeout=1.0):
        return ""

    async def close(self):
        self.closed = True


class ToneTTS:
    """amp 진폭 사인파(+앞 무음, 느린 시작)를 chunk 단위로 준다."""

    def __init__(self, amp: float = 400.0, seconds: float = 1.0, lead_ms: int = 0, fade_ms: int = 0,
                 chunk: int = 2048):
        self.amp, self.seconds, self.lead_ms, self.fade_ms, self.chunk = amp, seconds, lead_ms, fade_ms, chunk
        self.requests: list[str] = []
        self.voice = "fake-voice"

    def pcm(self) -> bytes:
        n = int(16000 * self.seconds)
        t = np.arange(n) / 16000
        x = self.amp * np.sin(2 * np.pi * 220 * t)
        if self.fade_ms:
            k = int(16 * self.fade_ms)
            x[:k] *= np.linspace(0.02, 1.0, k)
        x = np.concatenate([np.zeros(16 * self.lead_ms), x])
        return np.round(x).astype(np.int16).tobytes()

    async def synthesize(self, text):
        self.requests.append(text)
        b = self.pcm()
        for i in range(0, len(b), self.chunk):
            yield b[i:i + self.chunk]


def rms(b: bytes) -> float:
    a = np.frombuffer(b[: len(b) // 2 * 2], dtype=np.int16).astype(np.float64)
    return float(np.sqrt(np.mean(a * a))) if len(a) else 0.0


class ScriptLLM:
    """script: 이벤트·지연(float)·예외의 목록. 호출 기록과 정리(aclose) 여부를 남긴다."""

    def __init__(self, name: str, script: list):
        self.model = name
        self.script = script
        self.calls = 0
        self.closed = 0

    async def stream(self, system, messages, tools):
        self.calls += 1
        try:
            for item in self.script:
                if isinstance(item, (int, float)):
                    await asyncio.sleep(item)
                elif isinstance(item, BaseException):
                    raise item
                else:
                    yield item
        finally:
            self.closed += 1


class APIError(Exception):
    def __init__(self, code: int, msg: str = ""):
        super().__init__(f"{code} {msg}")
        self.code = code


def text(t: str) -> LLMEvent:
    return LLMEvent(type="text", text=t)


DONE = LLMEvent(type="done")


async def collect(agen) -> list[LLMEvent]:
    return [ev async for ev in agen]


# ══════════════════════════════════════════════════════
# build_deps: 실제 클래스명으로 조립
# ══════════════════════════════════════════════════════
@pytest.fixture
def fake_factory(monkeypatch):
    import agent.providers.factory as factory

    made = NS(stt=FakeSTT(), tts=ToneTTS(), llm=ScriptLLM("fake", [text("안녕하세요."), DONE]))
    monkeypatch.setattr(factory, "make_stt", lambda *a, **k: made.stt)
    monkeypatch.setattr(factory, "make_tts", lambda *a, **k: made.tts)
    monkeypatch.setattr(factory, "make_llm", lambda *a, **k: made.llm)
    monkeypatch.setattr(factory, "make_llm_with_fallback", lambda *a, **k: made.llm, raising=False)
    return made


class FakeVAD:
    def process(self, frame):
        return 0.0

    def reset(self):
        pass


class FakeSmart:
    def __init__(self):
        self.calls = 0

    def predict(self, pcm):
        self.calls += 1
        return 0.5


def test_build_deps_uses_real_class_names(fake_factory, monkeypatch, tmp_path):
    """make_* 팩토리가 없을 때 SileroVAD·SmartTurnV3·RuleTurnDetector 이름으로 찾는다."""
    import agent.turn.detector as det
    import agent.turn.smart_turn as st
    import agent.turn.vad as vad
    from agent.server import build_deps
    from agent.session import SessionDeps

    for mod, name in ((vad, "make_vad"), (st, "make_smart_turn"), (st, "SmartTurnONNX"),
                      (det, "make_detector"), (det, "KoreanTurnDetector"), (det, "TurnDetector")):
        monkeypatch.delattr(mod, name, raising=False)
    monkeypatch.setattr(vad, "SileroVAD", FakeVAD)
    monkeypatch.setattr(st, "SmartTurnV3", FakeSmart)

    deps = build_deps(Settings(tts_cache_dir=str(tmp_path / "c"), log_dir=str(tmp_path / "l")))
    assert isinstance(deps, SessionDeps)
    assert isinstance(deps.vad_factory(), FakeVAD)
    assert isinstance(deps.smart_turn, FakeSmart)
    d = deps.detector_factory()
    assert isinstance(d, det.RuleTurnDetector)
    assert deps.detector_factory() is not d                    # 세션마다 새 결정기
    assert deps.stt is fake_factory.stt and deps.llm is fake_factory.llm
    assert deps.normalize_args is not None
    assert deps.normalize_args("get_order_info", {"order_id": "이육일일 일오공 이삼일"},
                               {"properties": {"order_id": {"type": "string"}}},
                               "2026-11-20 10:00")["order_id"] == "2611150231"


def test_build_deps_prefers_make_detector(fake_factory, monkeypatch, tmp_path):
    import agent.turn.detector as det
    import agent.turn.smart_turn as st
    import agent.turn.vad as vad
    from agent.server import build_deps

    sentinel = det.RuleTurnDetector()
    monkeypatch.setattr(det, "make_detector", lambda: sentinel, raising=False)
    monkeypatch.setattr(vad, "SileroVAD", FakeVAD)
    monkeypatch.setattr(st, "SmartTurnV3", FakeSmart)
    deps = build_deps(Settings(tts_cache_dir=str(tmp_path / "c"), log_dir=str(tmp_path / "l")))
    assert deps.detector_factory() is sentinel


@pytest.mark.skipif(not (MODELS / "silero_vad.onnx").exists()
                    or not (MODELS / "smart-turn-v3.2-cpu.onnx").exists(), reason="models/ 없음")
def test_build_deps_with_real_turn_modules(fake_factory, tmp_path):
    from agent.server import build_deps
    from agent.turn.smart_turn import SmartTurnV3
    from agent.turn.vad import SileroVAD

    deps = build_deps(Settings(tts_cache_dir=str(tmp_path / "c"), log_dir=str(tmp_path / "l")))
    assert isinstance(deps.vad_factory(), SileroVAD)
    assert isinstance(deps.smart_turn, SmartTurnV3)
    assert hasattr(deps.detector_factory(), "decide")


def test_build_deps_wraps_tts_with_level_normalizer(fake_factory, monkeypatch, tmp_path):
    import agent.turn.smart_turn as st
    import agent.turn.vad as vad
    from agent.server import build_deps
    from agent.tts_cache import LevelTTS

    monkeypatch.setattr(vad, "SileroVAD", FakeVAD)
    monkeypatch.setattr(st, "SmartTurnV3", FakeSmart)
    deps = build_deps(Settings(tts_cache_dir=str(tmp_path / "c"), log_dir=str(tmp_path / "l")))
    assert isinstance(deps.tts, LevelTTS) and deps.tts.inner is fake_factory.tts
    deps2 = build_deps(Settings(tts_cache_dir=str(tmp_path / "c"), log_dir=str(tmp_path / "l"),
                                tts_level_rms=0))
    assert deps2.tts is fake_factory.tts                      # 0이면 정규화 끔


# ══════════════════════════════════════════════════════
# 서버 시작 예열
# ══════════════════════════════════════════════════════
async def test_warmup_loads_models_cache_and_llm(tmp_path):
    from agent.server import warmup
    from agent.session import SessionDeps
    from agent.tools_client import ToolsClient
    from agent.tts_cache import TTSCache

    s = Settings(tts_cache_dir=str(tmp_path / "c"))
    smart = FakeSmart()
    vads = []
    llm = ScriptLLM("p", [text("네."), DONE])
    tts = ToneTTS(amp=3000)
    stt = FakeSTT()
    deps = SessionDeps(stt=stt, tts=tts, llm=llm, vad_factory=lambda: vads.append(FakeVAD()) or vads[-1],
                       smart_turn=smart, detector_factory=lambda: None, tools=ToolsClient(),
                       cache=TTSCache(s.tts_cache_dir), settings=s)
    info = await warmup(deps)
    assert smart.calls >= 1 and vads                         # 모델 로드·첫 추론
    assert llm.calls == 1                                    # LLM 연결 예열
    for p in s.cached_phrases():
        assert deps.cache.get(p) is not None                 # 첫 마디·필러·사과 캐시
    assert info["cache_synthesized"] == len(s.cached_phrases())


async def test_warmup_survives_llm_and_stt_failure(tmp_path):
    from agent.server import warmup
    from agent.session import SessionDeps
    from agent.tools_client import ToolsClient
    from agent.tts_cache import TTSCache

    class BadSTT:
        async def open_stream(self, context=""):
            raise RuntimeError("no network")

    s = Settings(tts_cache_dir=str(tmp_path / "c"))
    deps = SessionDeps(stt=BadSTT(), tts=ToneTTS(amp=3000),
                       llm=ScriptLLM("p", [APIError(503, "UNAVAILABLE")]),
                       vad_factory=FakeVAD, smart_turn=None, detector_factory=lambda: None,
                       tools=ToolsClient(), cache=TTSCache(s.tts_cache_dir), settings=s)
    info = await warmup(deps)                                 # 예외 없이 끝난다
    assert info["llm_ok"] is False


# ══════════════════════════════════════════════════════
# 첫 소리 레벨(측정기는 바이너리 메시지 단위 RMS ≥ 300)
# ══════════════════════════════════════════════════════
def _speech(amp: float, seconds: float = 1.0, lead_ms: int = 0, fade_ms: int = 0) -> bytes:
    return ToneTTS(amp=amp, seconds=seconds, lead_ms=lead_ms, fade_ms=fade_ms).pcm()


def test_normalize_pcm_raises_quiet_speech_to_target():
    from agent.tts_cache import normalize_pcm, speech_rms

    out = normalize_pcm(_speech(150, lead_ms=300), target_rms=3000)
    assert abs(speech_rms(out) - 3000) < 150
    assert len(out) == len(_speech(150, lead_ms=300))          # 길이 유지


def test_normalize_pcm_prevents_clipping():
    from agent.tts_cache import normalize_pcm

    # 짧은 큰 피크 + 작은 본문: 목표 RMS까지 올리면 피크가 넘친다 → 피크 한도에서 멈춘다
    x = np.frombuffer(_speech(300), dtype=np.int16).astype(np.float64)
    x[8000:8010] = 20000
    out = np.frombuffer(normalize_pcm(np.round(x).astype(np.int16).tobytes(), target_rms=3000,
                                      peak_limit=30000), dtype=np.int16)
    assert np.abs(out.astype(np.int32)).max() <= 30000


def test_normalize_pcm_silence_unchanged():
    from agent.tts_cache import normalize_pcm

    z = b"\x00\x00" * 16000
    assert normalize_pcm(z) == z


def test_prepare_cached_pcm_first_chunk_is_voiced():
    from agent.tts_cache import prepare_cached_pcm

    # 작은 목소리 + 앞 무음 + 느린 시작(페이드 인 400ms)
    out = prepare_cached_pcm(_speech(200, lead_ms=250, fade_ms=400), chunk_bytes=3200)
    assert rms(out[:3200]) >= MEASURE_RMS


async def test_cache_warm_first_chunk_rms_at_least_300(tmp_path):
    from agent.tts_cache import TTSCache

    tts = ToneTTS(amp=120, lead_ms=200, fade_ms=300)
    c = TTSCache(tmp_path, chunk_bytes=3200)
    await c.warm(["네, 고객님."], tts)
    pcm = c.get("네, 고객님.")
    assert rms(pcm[:3200]) >= MEASURE_RMS
    pcm2 = await TTSCache(tmp_path / "b", chunk_bytes=3200).get_or_synth("확인해 드릴게요.", tts)
    assert rms(pcm2[:3200]) >= MEASURE_RMS


async def test_level_tts_streaming_normalizes_quiet_and_loud():
    from agent.tts_cache import LevelTTS, speech_rms

    quiet = LevelTTS(ToneTTS(amp=300, lead_ms=150), target_rms=3000)
    a = b"".join([c async for c in quiet.synthesize("첫 문장.")])
    b = b"".join([c async for c in quiet.synthesize("둘째 문장.")])
    assert abs(speech_rms(a) - 3000) < 300 and abs(speech_rms(b) - 3000) < 300
    assert len(a) == len(ToneTTS(amp=300, lead_ms=150).pcm())  # 길이·순서 유지
    loud = LevelTTS(ToneTTS(amp=20000), target_rms=3000)
    c = b"".join([x async for x in loud.synthesize("크게.")])
    assert abs(speech_rms(c) - 3000) < 300
    assert quiet.voice == "fake-voice"                          # 내부 속성 위임


async def test_level_tts_yields_incrementally_after_first_sentence():
    """한 번 레벨을 잰 뒤에는 덩어리를 모으지 않고 바로 흘려보낸다(TTFB 유지)."""
    from agent.tts_cache import LevelTTS

    inner = ToneTTS(amp=3000, chunk=640)
    lt = LevelTTS(inner, target_rms=3000)
    _ = [c async for c in lt.synthesize("예열.")]
    chunks = [c async for c in lt.synthesize("본문.")]
    assert len(chunks) == len(inner.pcm()) // 640
