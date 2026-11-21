# Soniox STT 도입(BAZE 경유) + 엔드포인팅 비교 (AG-T29)

> 2026-11-21 · 모델 `stt-rt-v6` · 경로 `wss://api.baze.co/v1/gateway/soniox/transcribe-websocket`(BAZE 키) ·
> 도구 `eval/stt_eval.py`(STT), `eval/soniox_endpoint_eval.py`(새, 엔드포인트), `eval/turn_sim.py`(기존 결정기) ·
> 원시 결과 `runs/stt_eval/baze_soniox_full/`, `runs/endpoint/soniox_{turnset,ts172}/`(gitignore 영역)

## 1. 요약

- **STT는 Soniox(BAZE)로 바꾼다.** 같은 172문항에서 Gemini live보다 정확하고 빠르다.
  숫자 정규화 CER 0.065 → **0.040**, 숫자 정확도 74.1 → **79.6%**, finalize p50 397 → **238ms**.
- **엔드포인팅은 기존 결정기(VAD + Smart Turn + 한국어 규칙)를 유지한다.** Soniox `<end>`는 빠르지만(p50 약 0.5~0.6초)
  말 끝 전 확정이 3~5배 많다(turnset 10.1 → 46.5%, 172문항 7.0 → 27.3%). 말 끝 전 응답은 그 문항 속도 0점이라 손해가 더 크다.
  `SonioxSTT(endpoint_detection=False)` 기본값을 그대로 둔다.
- 직접 키(`SONIOX_API_KEY`)는 잔액 소진(`402 organization_balance_exhausted`). 인증은 통과한다.

## 2. STT 정확도·지연 (172문항, 실시간 속도, 말 끝 + 200ms에 finalize)

| 지표 | Gemini `gemini-3.5-transcribe-live` (AG-T10b final_full) | Soniox `stt-rt-v6` (BAZE) |
|---|---|---|
| CER 원문 | 0.222 | **0.203** |
| CER 숫자 정규화 | 0.065 | **0.040** |
| 숫자 정확도(54문항) | 74.1% | **79.6%** |
| └ phone | 68.4% | **78.9%** |
| └ phone_ulaw | 70.6% | 70.6% |
| └ wideband | 83.3% | **88.9%** |
| finalize p50 / p95 | 397 / 490ms | **238 / 327ms** |
| finalize 시점 빈 부분 결과 | 4.7% | 5.8% |
| 최종 텍스트 빈 문항 | — | 0 |
| 오류 | 0 | 0 |

- Gemini 쪽에는 부분 결과 숫자 복원(STT_DIGIT_INTERIM_FIX, 교체 1건)이 들어 있고 Soniox에는 없다. 그래도 Soniox가 앞선다.
- 잡음별로 babble·white 숫자 정확도는 둘 다 66.7%로 같다. 개선은 잡음 없음·hum·phone에서 나왔다.
- "빈 부분 결과" 열은 말 끝 + 200ms 시점의 부분 결과다. 최종 텍스트가 빈 문항은 없다.

## 3. 엔드포인팅: 기존 결정기 vs Soniox `<end>`

판정은 같다(`eval/turn_sim.py`): 첫 판정이 말 끝보다 앞이면 false cut, 말 끝 + 1.5초 안이면 ok(지연), 그 밖은 miss.
Soniox 지연은 `<end>` **도착 벽시계 시각**이라 네트워크(BAZE 경유) 지연이 들어 있다. 기존 결정기 지연은 결정 시각이고,
세션은 침묵 200ms에 finalize를 선호출하므로 STT 확정 시간은 대부분 그 안에 숨는다.

| 데이터 | 방식 | 말 끝 전 확정 | miss | 지연 p50 / p95 |
|---|---|---|---|---|
| turnset test 228발화 | 기존 결정기 (`make_detector`, STT 텍스트, 빈 텍스트 10%) | **10.1%** (23) | 9.2% | 1045 / 1356ms |
| | Soniox `<end>` | 46.5% (106) | 9.6% | **570 / 925ms** |
| 172문항 | 기존 결정기 (`testset-eval`) | **7.0%** (12) | 7.6% | 835 / 1416ms |
| | Soniox `<end>` | 27.3% (47) | **2.9%** | **496 / 794ms** |

Soniox 말 끝 전 확정 위치(turnset): trap(완결 문장 뒤 쉼) 27, intra 45, digits 11, filler 8, particle 6, 자기 정정 6, connective 3.
유형별 비율: trap 85%, 자기 정정 74%, 간투사 61%, 연결어미 50%, 숫자 35%, 완결 문장 12%. 172문항 mid_pause 태그는 74%.

해석:
- Soniox `<end>`는 짧은 쉼(약 0.4~0.6초)에서 바로 나온다. 한국어 연결어미·숫자 묶음·자기 정정 같은 "말이 더 온다" 단서를 쓰지 않는다.
- 기존 결정기는 그 단서로 최대 1.3~2초까지 기다려 지연을 내주고 말 끝 전 확정을 막는다. 이것이 0점(말 끝 전 응답)을 막는 핵심이다.
- `max_endpoint_delay_ms`(서버가 받아들임, 3문항 스모크에서 500ms도 말 끝 전 확정 그대로)는 늦은 쪽 상한일 뿐 이른 확정은 줄이지 못한다.

## 4. 결론과 남은 것

- STT: Soniox(BAZE) 기본. `.env`에 `STT_PROVIDER=soniox`, 직접 키는 주석 처리(BAZE가 쓰이도록). LLM은 `LLM_PROVIDER=gemini`로 AD10 유지
  (BAZE 키가 있으면 factory가 LLM도 OpenAI-BAZE로 고르기 때문).
- 엔드포인팅: 기존 결정기 유지(AD12 권장 A).
- 이 측정은 STT만 바꿨다. 결정기 입력 텍스트가 Soniox 부분 결과로 바뀌면 연결어미·숫자 판정이 달라질 수 있다 → 172문항 E2E 재측정(`eval.runner`)이 필요하다.
- 발견 이슈: Soniox 오류(잔액·인증)는 연결이 열린 뒤 오므로 서버 예열이 통과하고 모든 턴이 빈 텍스트가 된다. Soniox → Gemini 폴백은 없다(AGENT_PLAN 13절).

## 5. 재현

```
uv run python -m eval.stt_eval --out runs/stt_eval/baze_soniox_full --stt soniox --set SONIOX_API_KEY= --concurrency 4
uv run python -m eval.soniox_endpoint_eval --set turnset --out runs/endpoint/soniox_turnset --set-env SONIOX_API_KEY= --concurrency 8
uv run python -m eval.soniox_endpoint_eval --set ts172 --out runs/endpoint/soniox_ts172 --set-env SONIOX_API_KEY= --concurrency 8
uv run python -m eval.turn_sim eval --split test --empty-text 0.10
uv run python -m eval.turn_sim testset-eval
```

API 호출: Soniox(BAZE) 스트림 약 580회(STT 174, 엔드포인트 408), 직접 키 2회(402).
