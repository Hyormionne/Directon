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
