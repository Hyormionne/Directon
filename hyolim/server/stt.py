"""받아쓰기(Soniox 실시간) + 말 끝 판단.

말 끝 판단은 두 겹이다.
1. Soniox 엔드포인트(<end> 토큰). max_endpoint_delay_ms 를 짧게 잡는다.
2. 로컬 음량 VAD 안전장치: <end> 가 왔어도 직전 200ms 에 말소리가 있으면 기다린다(말 도중 끼어들기 = 0점).
   반대로 말소리 뒤 무음이 길게 이어지는데 <end> 가 안 오면 finalize 를 보내 <fin> 으로 끝낸다.
"""
from __future__ import annotations

import asyncio
import json
import time

import numpy as np
import websockets

from . import config as C

SPEECH_RMS = 400
# 연습 음성의 문장 안 쉼은 최대 1.82초(P0130-01). 음량만 보는 판단은 이보다 길게 기다려야 끼어들지 않는다.
VAD_END_SILENCE_MS = 1900
FORCE_FINALIZE_SILENCE_MS = 1900
NUMBER_HOLD_MS = 2000          # 번호를 부르다 만 것 같으면 <end> 를 무시하고 이만큼 더 듣는다


def incomplete_number(text: str) -> bool:
    """문장 끝이 덜 불린 전화번호·주문번호면 True. (010 + 8자리 미만, 날짜 8자리 + 7자리 미만)"""
    from fncall.numbers_ko import find_number_spans
    spans = find_number_spans(text, min_digits=3)
    if not spans or len(text.rstrip()) - spans[-1].end > 3:
        return False
    d = spans[-1].digits
    if d.startswith("01"):
        return len(d) < 10
    if d.startswith("20") and len(d) >= 8:
        return len(d) < 15
    return False


def rms(pcm: bytes) -> float:
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(a * a))) if len(a) else 0.0


def soniox_context() -> dict:
    """도메인 단어를 알려 숫자·제품명 인식률을 높인다."""
    import os
    terms = ["하루온", "주문번호", "송장번호", "전화번호", "반품", "환불", "교환", "취소", "배송", "입금", "무통장"]
    try:
        info = json.load(open(os.path.join(C.ROOT, "data", "brand", "information.json"), encoding="utf-8"))
        for x in info:
            if "제품명" in x.get("tag", ""):
                terms += [t.strip() for t in x["content"].replace("\n", ",").split(",") if 1 < len(t.strip()) < 30][:60]
    except Exception:
        pass
    return {"general": [{"key": "domain", "value": "건강기능식품 쇼핑몰 전화 고객센터"},
                        {"key": "topic", "value": "주문번호와 휴대폰 번호를 숫자로 불러 줌"}],
            # 숫자 형식 설명: 휴대폰 010-XXXX-XXXX, 주문번호 2026MMDD-NNNNNNN, 한 자리씩 끊어 읽는다
            "text": ("손님은 휴대폰 번호(010으로 시작하는 11자리, 예: 공일공 일이삼사 오육칠팔 = 010-1234-5678)나 "
                     "주문번호(2026으로 시작하는 8자리와 7자리, 예: 20260101-0001234)를 한 자리씩 끊어 불러 준다. "
                     "'공'과 '영'은 0이다."),
            "terms": list(dict.fromkeys(terms))[:100]}


class Listener:
    """손님 음성을 받아 최종 문장과 말 끝 시각을 낸다."""

    def __init__(self):
        self.final = ""              # 확정된 텍스트
        self.partial = ""            # 아직 확정 안 된 텍스트
        self.heard_speech = False
        self.silent_ms = 0
        self.last_voice_t = 0.0
        self.endpoint = asyncio.Event()
        self.error: str | None = None
        self._ws = None
        self._reader = None
        self._finalize_sent = False
        self._pending: list[bytes] = []        # STT 연결 전에 들어온 음성
        self._held = False                      # 번호 때문에 말 끝을 미룬 상태
        self._preroll: list[bytes] = []          # 말소리 전 무음은 보내지 않고 직전 0.3초만 들고 있다 (크레딧 절약)
        self.sent_bytes = 0                      # STT 로 실제 보낸 음성 (사용량 기록)

    @property
    def text(self) -> str:
        return (self.final + self.partial).strip()

    async def start(self):
        if C.FAKE:
            return
        try:
            await self._connect()
        except Exception as e:
            self.error = f"STT 연결 실패: {e!r}"[:200]

    async def _connect(self):
        headers = {"Authorization": f"Bearer {C.SONIOX_API_KEY}"}
        try:
            ws = await websockets.connect(C.STT_URL, additional_headers=headers, max_size=None, open_timeout=5)
        except TypeError:                                    # websockets 12 이하
            ws = await websockets.connect(C.STT_URL, extra_headers=headers, max_size=None, open_timeout=5)
        # 설정을 보내기 전까지는 feed 가 버퍼(_pending)에 쌓는다
        cfg = {
            "model": C.STT_MODEL, "audio_format": "pcm_s16le", "sample_rate": C.RATE, "num_channels": 1,
            "language_hints": ["ko"], "enable_endpoint_detection": True,
            "max_endpoint_delay_ms": C.STT_MAX_ENDPOINT_DELAY_MS, "context": soniox_context(),
        }
        if not C.SONIOX_API_KEY.startswith("baze"):          # Soniox 직접 키는 예전 방식(본문)도 함께. 베이즈 키는 헤더로만
            cfg["api_key"] = C.SONIOX_API_KEY
        await ws.send(json.dumps(cfg, ensure_ascii=False))
        while self._pending:
            f = self._pending.pop(0)
            await ws.send(f)
            self.sent_bytes += len(f)
        self._ws = ws
        self._reader = asyncio.create_task(self._read())

    async def _read(self):
        try:
            async for m in self._ws:
                r = json.loads(m)
                if r.get("error_code"):
                    self.error = f"{r.get('error_type')}: {r.get('error_message')}"
                    return
                part = ""
                for t in r.get("tokens", []):
                    tx = t.get("text", "")
                    if tx in ("<end>", "<fin>"):
                        self._on_endpoint()
                        continue
                    if t.get("is_final"):
                        self.final += tx
                    else:
                        part += tx
                self.partial = part
                if r.get("finished"):
                    self._on_endpoint()
                    return
        except Exception as e:
            self.error = repr(e)[:200]            # 말 도중일 수 있으니 바로 끝내지 않고 음량 판단으로 넘긴다

    def _on_endpoint(self):
        if not self.heard_speech:
            return
        # 안전장치 1: 직전 200ms 안에 말소리가 있었으면 아직 말하는 중으로 본다
        if time.perf_counter() - self.last_voice_t < 0.2:
            return
        # 안전장치 2: "공일공 사구오칠" 처럼 번호를 부르다 쉰 경우 — 조금 더 듣는다
        if self.silent_ms < NUMBER_HOLD_MS and incomplete_number(self.text):
            self._held = True
            return
        self.endpoint.set()

    PREROLL_FRAMES = 15                          # 20ms × 15 = 말 시작 직전 0.3초

    def _gate(self, pcm: bytes) -> list[bytes]:
        """말소리가 나오기 전 무음은 보내지 않는다. 말이 시작되면 직전 0.3초를 붙여 함께 보낸다(첫 음절 보존)."""
        if not self.heard_speech:
            self._preroll.append(pcm)
            if len(self._preroll) > self.PREROLL_FRAMES:
                self._preroll.pop(0)
            return []
        if self._preroll:
            out, self._preroll = self._preroll + [pcm], []
            return out
        return [pcm]

    async def feed(self, pcm: bytes):
        if rms(pcm) >= SPEECH_RMS:
            self.heard_speech = True
            self.silent_ms = 0
            self.last_voice_t = time.perf_counter()
        elif self.heard_speech:
            self.silent_ms += len(pcm) * 1000 // (C.RATE * 2)
        if C.FAKE or self.error:
            # 가짜 모드, 또는 STT 연결이 끊긴 경우: 음량만으로 말 끝을 판단한다 (답은 못 알아들은 것으로 처리)
            if self.heard_speech and self.silent_ms >= VAD_END_SILENCE_MS:
                self.endpoint.set()
            return
        if self._held and self.silent_ms >= NUMBER_HOLD_MS:
            self.endpoint.set()                              # 미뤘는데 더 말이 없으면 끝
        frames = self._gate(pcm)
        if self._ws is None:
            self._pending.extend(frames)
            return
        try:
            for f in frames:
                await self._ws.send(f)
                self.sent_bytes += len(f)
            if self.heard_speech and self.silent_ms >= FORCE_FINALIZE_SILENCE_MS and not self._finalize_sent:
                self._finalize_sent = True
                await self._ws.send(json.dumps({"type": "finalize"}))
            if self._finalize_sent and self.silent_ms >= FORCE_FINALIZE_SILENCE_MS + 1000:
                self.endpoint.set()                          # STT 응답이 끝내 없으면 그냥 넘어간다
        except Exception as e:
            self.error = repr(e)[:200]            # 말 도중일 수 있으니 바로 끝내지 않고 음량 판단으로 넘긴다

    async def wait_transcript(self, timeout: float = 4.0) -> str:
        return self.text                                     # Soniox 는 <end> 때 글이 이미 확정돼 있다

    async def close(self):
        if self._ws is not None:
            try:
                await self._ws.send("")                      # 빈 텍스트 프레임 = 스트림 끝
                await asyncio.wait_for(self._ws.close(), 1.0)
            except Exception:
                pass
        if self._reader:
            self._reader.cancel()


# ───────────────────────────────────────────────────────────────────────────
# 임시: OpenAI 실시간 받아쓰기 (Soniox 키가 없을 때 지연 측정용). STT_BACKEND=openai
#   wss://api.openai.com/v1/realtime?intent=transcription, session.update(type=transcription)
#   입력 PCM 은 24kHz 만 받으므로 16k → 24k 로 올려 보낸다. 말 끝은 서버 VAD(speech_stopped).
# ───────────────────────────────────────────────────────────────────────────
import base64

from scipy.signal import resample_poly


class OpenAIListener(Listener):
    URL = "wss://api.openai.com/v1/realtime?intent=transcription"

    def __init__(self):
        super().__init__()
        self.transcript_ready = asyncio.Event()
        self.t_speech_stopped: float | None = None
        self.t_transcript: float | None = None
        self._items: dict[str, str] = {}
        self._order: list[str] = []          # 커밋된 순서 (말 덩어리 순서대로 이어 붙인다)
        self._done: set[str] = set()         # 받아쓰기가 끝난 덩어리
        self.t_commit: float | None = None

    async def _connect(self):
        headers = {"Authorization": f"Bearer {C.OPENAI_API_KEY}"}
        try:
            ws = await websockets.connect(self.URL, additional_headers=headers, max_size=None, open_timeout=5)
        except TypeError:
            ws = await websockets.connect(self.URL, extra_headers=headers, max_size=None, open_timeout=5)
        await ws.send(json.dumps({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {
            "format": {"type": "audio/pcm", "rate": 24000},
            "transcription": {"model": C.OPENAI_STT_MODEL, "language": "ko",
                              "prompt": "건강기능식품 쇼핑몰 고객센터 통화. 전화번호와 주문번호를 숫자로 불러 줌. 하루온, 반품, 환불, 배송"},
            "turn_detection": {"type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300,
                               "silence_duration_ms": C.OPENAI_VAD_SILENCE_MS}}}}}, ensure_ascii=False))
        while self._pending:
            f = self._pending.pop(0)
            await ws.send(self._enc(f))
            self.sent_bytes += len(f)
        self._ws = ws
        self._reader = asyncio.create_task(self._read())

    @staticmethod
    def _enc(pcm16k: bytes) -> str:
        a = np.frombuffer(pcm16k, dtype=np.int16).astype(np.float32)
        b = np.clip(resample_poly(a, 3, 2), -32768, 32767).astype(np.int16).tobytes()
        return json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(b).decode()})

    async def _read(self):
        try:
            async for m in self._ws:
                e = json.loads(m)
                t = e.get("type", "")
                if t == "error":
                    self.error = str(e.get("error"))[:200]
                elif t == "input_audio_buffer.committed":
                    iid = e.get("item_id")
                    if iid and iid not in self._order:
                        self._order.append(iid)
                    self.t_commit = time.perf_counter()
                    self._check_ready()
                elif t == "input_audio_buffer.speech_stopped":
                    self.t_speech_stopped = time.perf_counter()
                    self._on_endpoint()
                elif t == "conversation.item.input_audio_transcription.delta":
                    self._items[e.get("item_id")] = self._items.get(e.get("item_id"), "") + (e.get("delta") or "")
                    self.partial = " ".join(self._items.values())
                elif t == "conversation.item.input_audio_transcription.completed":
                    iid = e.get("item_id")
                    self._items[iid] = e.get("transcript") or ""
                    self._done.add(iid)
                    if iid not in self._order:
                        self._order.append(iid)
                    self.final = " ".join(self._items[i] for i in self._order if self._items.get(i))
                    self.partial = ""
                    self.t_transcript = time.perf_counter()
                    self._check_ready()
        except Exception as ex:
            self.error = repr(ex)[:200]

    @property
    def text(self) -> str:
        return (self.final or self.partial).strip()

    def _check_ready(self):
        """말 끝이 정해졌고, 그때까지 커밋된 말 덩어리가 '모두' 받아써졌을 때만 글을 넘긴다.
        (긴 말은 여러 덩어리로 나뉘는데, 앞 덩어리 글만 먼저 와서 뒷부분이 잘리는 문제를 막는다)"""
        last_committed = self.t_commit is not None and (self.t_speech_stopped is None or self.t_commit >= self.t_speech_stopped)
        if self.endpoint.is_set() and last_committed and self._order and all(i in self._done for i in self._order):
            self.transcript_ready.set()

    def _on_endpoint(self):
        # 음량 안전장치(직전 200ms 말소리)만 적용. 번호 보류는 글이 늦게 오므로 여기선 쓰지 않는다.
        if not self.heard_speech or time.perf_counter() - self.last_voice_t < 0.2:
            return
        self.endpoint.set()
        self._check_ready()

    async def wait_transcript(self, timeout: float = 4.0) -> str:
        try:
            await asyncio.wait_for(self.transcript_ready.wait(), timeout)
        except asyncio.TimeoutError:
            pass
        return self.text

    async def feed(self, pcm: bytes):
        if rms(pcm) >= SPEECH_RMS:
            self.heard_speech, self.silent_ms, self.last_voice_t = True, 0, time.perf_counter()
        elif self.heard_speech:
            self.silent_ms += len(pcm) * 1000 // (C.RATE * 2)
        if self.error or (self.heard_speech and self.silent_ms >= VAD_END_SILENCE_MS + 1000):
            if self.heard_speech and self.silent_ms >= VAD_END_SILENCE_MS:
                self.endpoint.set()
                self.transcript_ready.set()
            return
        frames = self._gate(pcm)
        if self._ws is None:
            self._pending.extend(frames)
            return
        try:
            for f in frames:
                await self._ws.send(self._enc(f))
                self.sent_bytes += len(f)
        except Exception as ex:
            self.error = repr(ex)[:200]

    async def close(self):
        if self._ws is not None:
            try:
                await asyncio.wait_for(self._ws.close(), 1.0)
            except Exception:
                pass
        if self._reader:
            self._reader.cancel()


def make_listener() -> Listener:
    return OpenAIListener() if C.STT_BACKEND == "openai" and not C.FAKE else Listener()
