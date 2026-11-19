"""Pipecat Smart Turn v3.2 (ONNX, CPU int8). 계약: agent.core.types.SmartTurn

모델: models/smart-turn-v3.2-cpu.onnx (pipecat-ai/smart-turn-v3, BSD-2-Clause).
원본 inference.py와 같은 전처리:
  1) truncate_audio_to_last_n_seconds(8): 8초 초과면 뒤 8초, 미만이면 앞쪽 0패딩
  2) WhisperFeatureExtractor(chunk_length=8)(..., padding="max_length", max_length=128000,
     do_normalize=True) → (1, 80, 800)
  3) ONNX 출력 = sigmoid 확률(완료)
2)는 transformers 없이 numpy로 다시 구현했다(transformers numpy 경로와 수치 일치, 테스트로 확인).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort

from agent.core.types import SAMPLE_RATE

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "smart-turn-v3.2-cpu.onnx"
MAX_SECONDS = 8
N_FFT = 400
HOP = 160
N_MELS = 80


def fit_last_n_seconds(x: np.ndarray, n_seconds: int = MAX_SECONDS, sr: int = SAMPLE_RATE) -> np.ndarray:
    """원본 audio_utils.truncate_audio_to_last_n_seconds와 동일."""
    m = n_seconds * sr
    if len(x) > m:
        return x[-m:]
    if len(x) < m:
        return np.pad(x, (m - len(x), 0), mode="constant", constant_values=0)
    return x


def _hz_to_mel(f: np.ndarray) -> np.ndarray:  # slaney
    f = np.asarray(f, dtype=np.float64)
    min_log_hz, min_log_mel, logstep = 1000.0, 15.0, 27.0 / np.log(6.4)
    mel = 3.0 * f / 200.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) * logstep, mel)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:  # slaney
    m = np.asarray(m, dtype=np.float64)
    min_log_hz, min_log_mel, logstep = 1000.0, 15.0, np.log(6.4) / 27.0
    hz = 200.0 * m / 3.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), hz)


@lru_cache(maxsize=1)
def mel_filters() -> np.ndarray:
    """transformers.audio_utils.mel_filter_bank(201, 80, 0, 8000, 16000, norm='slaney', mel_scale='slaney')."""
    n_freq = N_FFT // 2 + 1
    mel_pts = np.linspace(_hz_to_mel(0.0), _hz_to_mel(SAMPLE_RATE / 2), N_MELS + 2)
    filter_freqs = _mel_to_hz(mel_pts)
    fft_freqs = np.linspace(0, SAMPLE_RATE // 2, n_freq)
    diff = np.diff(filter_freqs)
    slopes = filter_freqs[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    fb = np.maximum(0.0, np.minimum(down, up))
    enorm = 2.0 / (filter_freqs[2:N_MELS + 2] - filter_freqs[:N_MELS])
    return np.ascontiguousarray(fb * enorm[None, :])   # (201, 80)


@lru_cache(maxsize=1)
def _hann() -> np.ndarray:
    return np.hanning(N_FFT + 1)[:-1]    # periodic hann


def whisper_log_mel(x: np.ndarray) -> np.ndarray:
    """8초(128000샘플) float 파형 → (80, 800) float32 log-mel. WhisperFeatureExtractor numpy 경로 재현."""
    x = np.asarray(x, dtype=np.float32)
    # do_normalize: 전체 길이(attention_mask 전부 1) 기준 zero-mean unit-var
    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    w = np.pad(x.astype(np.float64), (N_FFT // 2, N_FFT // 2), mode="reflect")
    n_frames = 1 + (len(w) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    spec = np.abs(np.fft.rfft(w[idx] * _hann()[None, :], n=N_FFT, axis=1)) ** 2   # (frames, 201)
    # (frames,201)@(201,80): mel_filters().T @ spec.T와 같은 값이지만 이 배치가 OpenBLAS에서 ~100배 빠르다
    mel = np.maximum(1e-10, spec @ mel_filters())                                # (frames, 80)
    log_spec = np.log10(mel).T.astype(np.float32)[:, :-1]
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)


class SmartTurnV3:
    """최근 최대 8초 PCM → 발화 완료 확률.

    threads: onnxruntime intra-op 스레드. 16논리코어 개발 PC 실측(ONNX만): 1스레드 ~50ms, 4 ~15-20ms,
    8 ~15ms, 16은 spin-wait 켜면 ~48ms·끄면 ~17ms. 그래서 spin-wait를 끄고(서버 이벤트 루프·다른
    세션과 CPU를 다투지 않게) 기본 4스레드로 둔다. 동시 세션이 많으면 더 줄인다.
    """

    def __init__(self, model_path: str | Path = DEFAULT_MODEL, threads: int = 4,
                 allow_spinning: bool = False):
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        so.add_session_config_entry("session.intra_op.allow_spinning", "1" if allow_spinning else "0")
        self._sess = ort.InferenceSession(str(model_path), sess_options=so,
                                          providers=["CPUExecutionProvider"])

    def predict_float(self, x: np.ndarray) -> float:
        feats = whisper_log_mel(fit_last_n_seconds(np.asarray(x, dtype=np.float32)))[None]
        p = float(self._sess.run(None, {"input_features": feats})[0].reshape(-1)[0])
        return min(1.0, max(0.0, p))

    def predict(self, pcm: bytes) -> float:
        # 8초 넘는 앞부분은 변환 전에 버린다(결과 동일, 비용 절약)
        pcm = pcm[-MAX_SECONDS * SAMPLE_RATE * 2:] if len(pcm) > MAX_SECONDS * SAMPLE_RATE * 2 else pcm
        x = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0
        return self.predict_float(x)
