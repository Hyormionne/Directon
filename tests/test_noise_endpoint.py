"""AG-T55 소음 속 말 끝: 짧은 소리로 침묵이 깨지지 않기, 손님 목소리 크기 기준, 받아쓰기 글 멈춤으로 확정.

시간은 프레임 수 기준(20ms/프레임). FakeVAD는 0이 아닌 프레임을 모두 음성(확률 1.0)으로 본다 —
주변 말소리·TV처럼 VAD가 음성으로 보는 소음을 흉내 낸다.
"""
from test_session import SILENCE, SPEECH, FakeDetector, FakeLLM, FakeWS, make_deps, run, start_msg, text_events

QUIET = b"\xe8\x03" * 320            # 20ms, 진폭 1000(SPEECH보다 20dB 작다)


def first_audio(ws):
    bins = ws.binaries()
    return bins[0][0] if bins else None


def _run_case(tmp_path, frames, **settings_kw):
    ws = FakeWS(start_msg(), frames)
    llm = FakeLLM(text_events("확인해 드릴게요."), ws=ws)
    kw = {"barge_in": False, **settings_kw}
    deps, det = make_deps(tmp_path, ws=ws, llm=llm, detector=FakeDetector(end_ms=600), settings_kw=kw)
    det.ws = ws
    return ws, deps, det


async def test_short_blip_counts_as_silence(tmp_path):
    frames = [SPEECH] * 50 + [SILENCE] * 10 + [SPEECH] * 2 + [SILENCE] * 40
    ws, deps, det = _run_case(tmp_path, frames)
    await run(ws, deps)
    assert det.decided_at == [80]                     # 40ms 소리는 침묵으로 세어 50 + 30프레임에 확정


async def test_short_blip_resets_silence_when_switch_off(tmp_path):
    frames = [SPEECH] * 50 + [SILENCE] * 10 + [SPEECH] * 2 + [SILENCE] * 40
    ws, deps, det = _run_case(tmp_path, frames, resume_min_ms=0)
    await run(ws, deps)
    assert det.decided_at == [92]                     # 이전 동작: 소리 뒤 다시 30프레임


async def test_real_resume_still_resets_silence(tmp_path):
    frames = [SPEECH] * 30 + [SILENCE] * 15 + [SPEECH] * 20 + [SILENCE] * 40
    ws, deps, det = _run_case(tmp_path, frames)
    await run(ws, deps)
    assert det.decided_at == [30 + 15 + 20 + 30]      # 100ms 넘게 이어진 말은 그대로 재개


async def test_continuous_noise_without_stall_waits_for_silence(tmp_path):
    frames = [SPEECH] * 400 + [SILENCE] * 40          # 말이 끝난 뒤에도 소음이 8초
    ws, deps, det = _run_case(tmp_path, frames, text_stall_cap_ms=0)
    await run(ws, deps)
    assert first_audio(ws) >= 430                     # 소음이 그칠 때까지 확정 못 함(문제 상황)


async def test_text_stall_cap_commits_under_continuous_noise(tmp_path):
    frames = [SPEECH] * 400 + [SILENCE] * 40
    ws, deps, det = _run_case(tmp_path, frames, text_stall_cap_ms=3000)
    await run(ws, deps)
    t = first_audio(ws)
    assert t is not None and 150 <= t <= 175          # 글이 6프레임째에 나온 뒤 3초 + 끝맺음 대기
    assert ws.texts()[-1] == {"type": "turn.end"}
    assert deps.stt.streams[0].finalize_calls >= 1    # 확정 전에 받아쓰기 끝맺음을 요청했다


async def test_text_stall_end_uses_detector_rule(tmp_path):
    frames = [SPEECH] * 400 + [SILENCE] * 40
    ws, deps, det = _run_case(tmp_path, frames, text_stall_end_ms=1500, text_stall_cap_ms=0)
    await run(ws, deps)
    t = first_audio(ws)
    assert t is not None and 95 <= t <= 120           # 멈춤 1.5초 뒤부터 침묵 200ms로 넘겨 결정기 600ms에 확정


async def test_speaker_gate_treats_quiet_voice_as_silence(tmp_path):
    frames = [SPEECH] * 50 + [QUIET] * 60 + [SILENCE] * 40
    ws, deps, det = _run_case(tmp_path, frames, speaker_gate_db=12.0, speaker_gate_min_ms=600)
    await run(ws, deps)
    assert det.decided_at == [80]                     # 20dB 작은 소리는 침묵으로 보고 50 + 30프레임에 확정


async def test_speaker_gate_off_by_default(tmp_path):
    frames = [SPEECH] * 50 + [QUIET] * 60 + [SILENCE] * 40
    ws, deps, det = _run_case(tmp_path, frames)
    await run(ws, deps)
    assert det.decided_at == [50 + 60 + 30]
