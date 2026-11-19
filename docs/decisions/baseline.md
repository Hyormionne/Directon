# 베이스라인 (AG-T09b) — 172문항, 커밋 aedb19d

- 실행: `bash runs/run_baseline.sh` (mock 8100, agent 9000, `eval.runner --concurrency 4`, `eval.score`, judge 없음)
- 구성: STT gemini-3.5-transcribe-live / LLM gemini-3.5-flash-lite(폴백 3.1-flash-lite) / TTS edge SunHi / Silero VAD + Smart Turn v3.2 + RuleTurnDetector 기본 설정
- 상세표: `baseline_172_report.md`(eval.score 출력 사본)

## 요약

| 지표 | 값 |
|---|---|
| 속도 점수 평균 | 67.4 |
| 말 끝 전 발화(속도 0점) | **55 / 172 (32%)** |
| 첫 소리 p50 / p95 (정상 확정) | 0.51s / 1.90s |
| 함수 점수 / 완전 일치율 | 56.4 / 0.52 |
| 에러 | 0 |
| turn.end p50 / p95 | 3.07s / 5.14s |
| 추정(응대 제외) | 63.0 |

## 관찰

1. **조기 확정이 가장 큰 손실.** 말 끝 전 발화 55건의 첫 소리는 중앙값 기준 실제 말 끝보다 3.4초 앞 → 발화 초반의 쉼에서 확정. 태그: digits 21, mid_pause 19, filler_words 8, self_correction 7(8건 중 7건), connective_end 6. 숫자를 끊어 읽는 쉼(TTS가 쉼표마다 쉼)에서 많이 끊긴다.
2. **babble 잡음**: 첫 소리 p50 1.89s — 말 끝 뒤에도 잡음이 이어져 VAD가 음성으로 보고 잡음이 끝난 뒤(파일 끝)에야 확정.
3. **함수**: complaint_handoff 0.12(transfer_to_agent 미호출), tool_action 0.35, digits 0.35. 숫자 오인식(예: 2611194212, 25120384, 0107543210)과 조기 확정에 따른 부분 발화가 원인.
4. 채널: phone_ulaw가 가장 나쁨(속도 0점 23건, 함수 0.40).

## 개선 Task
- AG-T12/T10c: 턴 결정(쉼·숫자 끊어 읽기·간투사·자기 정정), babble VAD
- AG-T10b: STT 숫자 인식
- AG-T10a: 인계 함수 호출 등 LLM 응대
