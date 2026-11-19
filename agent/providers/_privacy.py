"""로그용 고객 발화 가리기(AG-T18). 시험 음성 저장 금지 모드(DIRECTON_NO_AUDIO_LOG=1)에서는
콘솔 로그에도 원문을 남기지 않는다(콘솔을 파일로 돌리면 디스크에 남는다). 길이·숫자 개수만 남긴다."""
from __future__ import annotations

import re


def _no_audio_log() -> bool:
    try:
        from agent.turn_log import no_audio_log
        return bool(no_audio_log())
    except Exception:  # noqa: BLE001 — 판단 불가면 안전하게 가린다
        return True


def redact(text: object) -> str:
    """로그에 넣을 표현. 저장 금지 모드면 '<N자, 숫자 M개>', 아니면 repr."""
    s = "" if text is None else str(text)
    if _no_audio_log():
        return f"<{len(s)}자, 숫자 {len(re.findall(r'[0-9]', s))}개>"
    return repr(s)
