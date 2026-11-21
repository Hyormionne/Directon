"""합성 API를 부르지 않는 TTS(개발용). `DIRECTON_TTS_PROVIDER=off`일 때 쓴다.

문장마다 짧은 신호음(16kHz s16le mono)을 낸다. 크레딧을 쓰지 않고 첫 소리·조회 뒤 답 같은 시간 측정과
함수 호출 확인은 그대로 할 수 있다. 말소리가 없으므로 응대 채점(받아쓰기)에는 쓸 수 없다. 제출용이 아니다.
"""
from __future__ import annotations

import math
import struct
from typing import AsyncIterator

from agent.core.types import SAMPLE_RATE


class ToneTTS:
    model = "off"
    voice = "tone"

    def __init__(self, ms: int = 300, hz: float = 440.0, amp: int = 6000):
        n = SAMPLE_RATE * ms // 1000
        self._pcm = b"".join(struct.pack("<h", int(amp * math.sin(2 * math.pi * hz * i / SAMPLE_RATE)))
                             for i in range(n))

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        if (text or "").strip():
            yield self._pcm
