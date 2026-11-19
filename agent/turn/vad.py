"""Silero VAD(onnxruntime 직접 실행, torch 없음). 계약: agent.core.types.VAD

모델: models/silero_vad.onnx (snakers4/silero-vad, MIT). scripts/download_models.py로 받는다.
원본 OnnxWrapper(utils_vad.py)와 같은 방식: 16kHz에서 512샘플 청크 + 앞 청크의 64샘플 컨텍스트,
RNN state (2,1,128)을 청크 사이에 이어 준다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from agent.core.types import SAMPLE_RATE

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "silero_vad.onnx"
CHUNK = 512        # 16kHz에서 모델이 받는 유일한 길이(32ms)
CONTEXT = 64


class SileroVAD:
    """20ms(320샘플) 프레임을 받아 512샘플 단위로 추론하고 최신 음성 확률을 돌려준다."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL, threads: int = 1):
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = threads
        so.log_severity_level = 3
        self._sess = ort.InferenceSession(str(model_path), sess_options=so,
                                          providers=["CPUExecutionProvider"])
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT), dtype=np.float32)
        self._buf = np.zeros(0, dtype=np.float32)
        self.last_prob = 0.0
        self.n_chunks = 0

    def _infer(self, chunk: np.ndarray) -> float:
        x = np.concatenate([self._context, chunk[None, :]], axis=1)
        out, self._state = self._sess.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._context = x[:, -CONTEXT:]
        self.n_chunks += 1
        return float(out[0, 0])

    def process(self, frame: bytes) -> float:
        """프레임(길이 무관, 보통 640B)을 넣는다. 512샘플이 모일 때마다 추론, 최신 확률 반환."""
        x = np.frombuffer(frame[: len(frame) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0
        self._buf = np.concatenate([self._buf, x]) if self._buf.size else x
        while self._buf.size >= CHUNK:
            self.last_prob = self._infer(self._buf[:CHUNK])
            self._buf = self._buf[CHUNK:]
        return self.last_prob


# ── AG-T10c 잡음 강건 VAD ─────────────────────────────
# Silero는 웅성거림(babble)을 음성으로 본다(172문항 베이스라인: babble 꼬리 프레임의 43%가 p≥0.5).
# 그래서 말이 끝난 뒤에도 '침묵'이 시작되지 않아 확정이 잡음 끝까지 밀렸다.
# NoiseGate:
#  - 잡음 바닥: 100ms 평활 에너지의 최근 3초 최솟값(최소 통계). 바닥 > noisy_floor_db면 '잡음 있음'(noisy).
#  - 엄격 판정(active)은 잡음이 있고 다음 중 하나일 때만 건다.
#      · 출렁이는 잡음: 스트림 첫 200ms 프레임 에너지 표준편차 ≥ fluct_on_db(babble 1.2~27dB, white·hum ≤0.9dB)
#      · 내부 VAD가 고른 잡음에 속음: 바닥 근처(바닥+3dB 미만)가 300ms 이상 이어진 프레임에서 p≥p_base 비율이 높음
#    white·pink·hum처럼 Silero가 이미 잘 거르는 잡음에 걸면 낮은 SNR의 약한 음절을 깎아 말 중간 침묵 오인이 는다.
#  - 엄격 판정: 시작은 (a) p ≥ p_noisy (b) 에너지 ≥ 바닥 + margin_db (c) onset_frames 연속,
#    유지는 p ≥ p_hold 그리고 에너지 ≥ 바닥 + hold_margin_db.
#  - 잡음이 없거나 엄격 판정이 아니면 내부 VAD 확률을 그대로 돌려준다.
# 값은 eval/vad_noise_eval.py로 골랐다(docs/decisions/session_tuning.md).
from collections import deque  # noqa: E402
from dataclasses import dataclass  # noqa: E402


@dataclass(frozen=True)
class NoiseGateConfig:
    p_base: float = 0.5             # 내부 VAD 임계(세션 vad_threshold와 같게)
    p_noisy: float = 0.6            # 엄격: 음성 시작 임계(0.7보다 말 중간 침묵 오인이 적었다)
    p_hold: float = 0.5             # 엄격: 음성 유지 임계
    margin_db: float = 6.0          # 엄격: 음성 시작 에너지 = 바닥 + margin
    hold_margin_db: float = 3.0     # 엄격: 음성 유지 에너지 = 바닥 + hold_margin
    onset_frames: int = 3           # 엄격: 시작 조건 연속 프레임 수(60ms)
    noisy_floor_db: float = -60.0   # 바닥이 이 값(dBFS)을 넘으면 잡음 있음
    smooth_frames: int = 5          # 바닥 추정용 평활(100ms)
    floor_window_frames: int = 150  # 최소 통계 창(3초)
    min_frames: int = 10            # 이만큼 본 뒤부터 바닥을 믿는다(200ms)
    fluct_frames: int = 10          # 출렁임 판정에 쓰는 첫 프레임 수(200ms). 한 번 정하면 reset 뒤에도 유지
    fluct_on_db: float = 1.5
    fool_near_db: float = 3.0       # 속음 판정: 바닥 근처 기준
    fool_settle_frames: int = 15    # 바닥 근처가 이만큼(300ms) 이어진 뒤부터 센다(말 사이 VAD 꼬리 제외)
    fool_alpha: float = 0.05
    fool_on: float = 0.4


def frame_db(frame: bytes) -> float:
    a = np.frombuffer(frame[: len(frame) // 2 * 2], dtype=np.int16).astype(np.float64)
    if a.size == 0:
        return -120.0
    return float(10.0 * np.log10(np.mean(a * a) / (32768.0 ** 2) + 1e-12))


class NoiseGate:
    """프레임 에너지 + 내부 확률 → 음성 여부. reset()은 음성 상태만 지우고 잡음 추정은 유지한다(같은 통화 환경)."""

    def __init__(self, config: NoiseGateConfig | None = None):
        self.c = config or NoiseGateConfig()
        self._pow: deque[float] = deque(maxlen=max(1, self.c.smooth_frames))
        self._smooth: deque[float] = deque(maxlen=max(1, self.c.floor_window_frames))
        self._first: list[float] = []
        self.floor_db: float | None = None
        self.fluct_db: float | None = None
        self.fool = 0.0
        self.fooled = False
        self.noisy = False
        self._near = 0
        self.reset()

    @property
    def active(self) -> bool:
        """엄격 판정 중인지."""
        return self.noisy and (self.fooled or (self.fluct_db is not None and self.fluct_db >= self.c.fluct_on_db))

    def reset(self) -> None:
        self.in_speech = False
        self._run = 0
        self.last_db = -120.0

    def update(self, frame: bytes, p: float) -> bool:
        return self.update_db(frame_db(frame), p)

    def update_db(self, e: float, p: float) -> bool:
        """프레임 에너지(dBFS)와 내부 확률로 갱신하고 음성 여부를 돌려준다(오프라인 측정은 이걸 직접 부른다)."""
        c = self.c
        self.last_db = e
        self._pow.append(10.0 ** (e / 10.0))
        self._smooth.append(10.0 * np.log10(sum(self._pow) / len(self._pow) + 1e-12))
        if len(self._smooth) >= c.min_frames:
            self.floor_db = min(self._smooth)
        if self.fluct_db is None:
            self._first.append(e)
            if len(self._first) >= max(2, c.fluct_frames):
                self.fluct_db = float(np.std(self._first))
        self.noisy = self.floor_db is not None and self.floor_db > c.noisy_floor_db
        near = self.noisy and e < self.floor_db + c.fool_near_db
        self._near = self._near + 1 if near else 0
        if self._near >= c.fool_settle_frames:
            self.fool += c.fool_alpha * ((1.0 if p >= c.p_base else 0.0) - self.fool)
            self.fooled = self.fool >= c.fool_on
        if not self.active:
            self.in_speech = p >= c.p_base
            self._run = 1 if self.in_speech else 0
            return self.in_speech
        fl = self.floor_db
        if self.in_speech:
            self.in_speech = p >= c.p_hold and e >= fl + c.hold_margin_db
            self._run = 0
            return self.in_speech
        cand = p >= c.p_noisy and e >= fl + c.margin_db
        self._run = self._run + 1 if cand else 0
        self.in_speech = self._run >= max(1, c.onset_frames)
        return self.in_speech

    def output(self, p: float, speech: bool) -> float:
        """세션은 확률 ≥ 임계를 음성으로 본다: 엄격 판정이 아니면 원래 확률, 엄격이면 음성일 때만 확률(≥p_base)."""
        if not self.active:
            return p
        return max(p, self.c.p_base) if speech else 0.0


class NoiseRobustVAD:
    """임의의 VAD(process/reset)를 NoiseGate로 감싼다."""

    def __init__(self, inner, config: NoiseGateConfig | None = None):
        self.inner = inner
        self.gate = NoiseGate(config)
        self.last_raw_prob = 0.0

    @property
    def noisy(self) -> bool:
        return self.gate.noisy

    @property
    def active(self) -> bool:
        return self.gate.active

    def process(self, frame: bytes) -> float:
        p = float(self.inner.process(frame))
        self.last_raw_prob = p
        return self.gate.output(p, self.gate.update(frame, p))

    def reset(self) -> None:
        self.inner.reset()
        self.gate.reset()


class NoiseRobustSileroVAD(SileroVAD):
    """SileroVAD + NoiseGate(서버 기본). SileroVAD 계약(reset·n_chunks·last_prob)은 그대로."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL, threads: int = 1,
                 config: NoiseGateConfig | None = None):
        self.gate = NoiseGate(config)
        self.last_raw_prob = 0.0
        super().__init__(model_path, threads)

    @property
    def noisy(self) -> bool:
        return self.gate.noisy

    @property
    def active(self) -> bool:
        return self.gate.active

    def reset(self) -> None:
        super().reset()
        gate = getattr(self, "gate", None)
        if gate is not None:
            gate.reset()

    def process(self, frame: bytes) -> float:
        p = super().process(frame)
        self.last_raw_prob = p
        return self.gate.output(p, self.gate.update(frame, p))


def make_vad() -> SileroVAD:
    """서버 build_deps가 세션마다 부르는 팩토리.

    기본은 원래 SileroVAD(AD8, 2026-11-19 사용자 확정: 잡음 래퍼가 보정셋 false cut을 16.7%→18.4%로 늘림).
    DIRECTON_VAD_NOISE_GATE=1이면 NoiseRobustSileroVAD.
    """
    import os

    if os.environ.get("DIRECTON_VAD_NOISE_GATE", "0").strip().lower() in ("1", "true", "yes", "on"):
        return NoiseRobustSileroVAD()
    return SileroVAD()
