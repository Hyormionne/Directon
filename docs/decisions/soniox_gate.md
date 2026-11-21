# Soniox `<end>` + 규칙/Jev 게이트 엔드포인팅 비교 (AG-T30)

> 2026-11-21 · 이전 비교 `soniox_baze.md` 3절 · 게이트 `agent/turn/soniox_gate.py` · 녹화 `eval/soniox_endpoint_eval.py` ·
> 비교 `eval/soniox_gate_eval.py` · 원시 결과 `runs/endpoint/ev_{turnset,ts172}/`, `runs/endpoint/gate_report.{md,json}`(gitignore 영역)

## 1. 요약

- Soniox `<end>`마다 "진짜 말 끝인가"를 텍스트로 거르는 게이트를 만들었다(규칙 / Jev / 규칙+Jev). 거절하면 다음 `<end>`를 기다리고,
  VAD 침묵이 hold(1초)를 넘으면 확정한다.
- **어느 게이트도 기존 결정기를 이기지 못했다.** 말 끝 전 확정(0점)이 turnset 10.1% → 22.8~25.4%, 172문항 7.0% → 11.0~15.1%.
  얻는 것은 지연이다(172문항 p50 835 → 497~667ms, p95 1416 → 794~940ms).
- 남는 말 끝 전 확정의 대부분은 **"완결 문장 + 쉼 + 다음 문장"(trap)** 이다(turnset 22~25건, 기존 6건). 텍스트는 이미 완결이라
  규칙도 Jev도 "끝"이라 한다. 기존 결정기는 완결 문장에도 0.7~1.3초 기다려 이걸 막는데, Soniox `<end>`는 약 0.5초에 온다.
- **권장: 기존 결정기 유지(AD12 A 그대로).** 게이트는 기본 꺼짐 모듈로 남긴다. 세션에는 연결하지 않았다.

## 2. 설계

| 단계 | 규칙 | 근거 |
|---|---|---|
| `<end>` 도착 | 그때까지 확정 텍스트로 판정 | Soniox는 `<end>` 직전 토큰을 모두 확정한다 |
| 규칙 | `detector.table_kind`가 연결어미·조사·간투사·숫자 미완/부분·자기 정정·쉼표면 거절 | 기존 결정기의 보류 묶음과 같다 |
| Jev | `finished` ≥ 0.5면 수락, 판정 시각 = `<end>` + 실측 왕복 | AG-T24 질문·캐시 그대로(`jev-1.14.0`) |
| 규칙+Jev | 규칙이 보류면 거절, 아니면 Jev | |
| 거절 뒤 | VAD 침묵 hold_ms(1000)면 확정 | |
| 공통 상한 | `<end>`와 무관하게 VAD 침묵 1500ms면 확정 | Soniox가 말 끝을 놓친 경우 |
| 지난 `<end>` | 도착 순간 VAD가 음성이면 무시 | 아래 3절 |

### 처음 설계에서 바꾼 점: 침묵은 Soniox 토큰이 아니라 로컬 VAD로 잰다

녹화 이벤트를 보면 Soniox 토큰은 그 음성보다 **약 1~1.2초 늦게** 도착한다(예: TS0005 오디오 5.28초 음절이 6.49초에 도착).
처음에는 "새 토큰이 오면 말하는 중"으로 hold를 쟀는데, 다음 구절 토큰이 오기 전에 hold가 끝나 말 끝 전 확정이 오히려 늘었다
(turnset 규칙 게이트 39.0%, intra 52건). 세션은 20ms마다 Silero VAD를 돌리므로 그것으로 바꾸자 39.0 → 25.4%가 됐다.
"Soniox + VAD만" 행이 이 효과만 보인다(텍스트 판정 없이 지난 `<end>`만 버림: 46.9 → 33.3%).

## 3. 결과 (같은 오디오, 같은 판정: 말 끝 전이면 false cut, 말 끝 + 1.5초 안이면 지연, 그 밖은 miss)

| 데이터 | 방식 | 말 끝 전 확정 | miss | 지연 p50 / p95 ms |
|---|---|---|---|---|
| turnset test 228 | **기존 결정기** | **10.1%** (23) | **9.2%** | 1045 / 1356 |
| | Soniox 단독 | 46.9% (107) | 10.1% | 576 / 875 |
| | Soniox + VAD만 | 33.3% (76) | 14.9% | 547 / 847 |
| | Soniox + 규칙 | 25.4% (58) | 14.9% | 568 / 1051 |
| | Soniox + Jev | 23.7% (54) | 16.2% | 722 / 1078 |
| | Soniox + 규칙 + Jev | 22.8% (52) | 16.2% | 722 / 1082 |
| 172문항 | **기존 결정기** | **7.0%** (12) | **7.6%** | 835 / 1416 |
| | Soniox 단독 | 26.7% (46) | 3.5% | 498 / 757 |
| | Soniox + VAD만 | 20.3% (35) | 9.3% | 499 / 756 |
| | Soniox + 규칙 | 15.1% (26) | 9.3% | 497 / 794 |
| | Soniox + Jev | 11.6% (20) | 11.0% | 667 / 911 |
| | Soniox + 규칙 + Jev | 11.0% (19) | 11.0% | 663 / 940 |

말 끝 전 확정 위치(turnset):

| 방식 | trap | intra | digits | self_corr | particle | filler | connective |
|---|---|---|---|---|---|---|---|
| 기존 결정기 | 6 | 6 | 4 | 0 | 1 | 3 | 3 |
| Soniox + 규칙 | 25 | 9 | 4 | 5 | 6 | 6 | 3 |
| Soniox + 규칙 + Jev | 22 | 8 | 4 | 4 | 6 | 5 | 3 |

- 규칙·Jev는 숫자·자기 정정·간투사에서 Soniox 단독보다 확실히 줄인다(숫자 12 → 4). trap은 거의 그대로다(26 → 22~25).
- Jev는 172문항(대화 맥락 있음)에서 규칙보다 낫다(15.1 → 11.6%). turnset(맥락 없음)에서는 차이가 작다. Jev 왕복만큼(약 170ms) 늦다.

### 설정 탐색 (test 데이터에서 고른 값이라 낙관적)

| 데이터 | 설정 | 말 끝 전 확정 | miss | 지연 p50 / p95 |
|---|---|---|---|---|
| turnset | 규칙 hold 700 / 1000 / 1300 / 1600 | 31.1 / 25.4 / 22.4 / 21.1% | 14.9 / 14.9 / 14.9 / 18.9% | 560~578 / 842~1272 |
| turnset | Jev thr 0.3 / 0.5 / 0.7 / 0.85 | 24.6 / 23.7 / 22.8 / 19.7% | 16.2 / 16.2 / 16.7 / 16.7% | 713~1020 / 1000~1151 |
| 172문항 | 규칙 hold 700 / 1000 / 1300 / 1600 | 18.6 / 15.1 / 14.5 / 14.5% | 9.3 / 9.3 / 9.3 / 10.5% | 495~498 / 755~794 |
| 172문항 | Jev thr 0.3 / 0.5 / 0.7 / 0.85 | 12.2 / 11.6 / 8.7 / 7.6% | 11.0 / 11.0 / 11.0 / 12.2% | 667~670 / 911~1112 |

- 가장 가까운 설정은 172문항 Jev thr 0.85(말 끝 전 7.6% vs 7.0%, p50 −165ms, p95 −300ms)지만 miss가 +4.6%p이고,
  같은 설정이 turnset에서는 19.7%(기존의 2배)다.

### miss가 늘어난 이유

Soniox + 규칙 turnset miss 34건 중 27건이 babble 잡음이다(babble 62문항 중 44%). VAD가 babble을 음성으로 봐서 말 끝 `<end>`를 "지난 것"으로 버리고,
침묵 상한도 시작되지 않는다. 기존 결정기도 같은 VAD 약점이 있다(AD8 잡음 게이트 끔). 나머지는 1.5초를 조금 넘긴 늦은 확정이다(1.54~2.4초).

## 4. 결론

- 엔드포인팅은 기존 결정기 유지. Soniox `<end>`는 trap(완결 문장 뒤 쉼)에서 구조적으로 이르다. 텍스트 게이트는 그걸 못 막는다.
- Soniox를 엔드포인팅에 쓴다면 "확정 신호"보다 **기존 결정기의 대기를 줄이는 근거**가 맞다(예: 표 결정기가 기다리는 중
  `<end>` + Jev 높음이면 대기 단축). 다만 AG-T24에서 Jev "줄이기"는 대기 1초 안쪽에서 점수 이득이 없어 껐다. 이번 결과도 같은 방향이다.
- 게이트 모듈(`SonioxEndpointGate`)과 녹화·재생 도구는 남긴다. 당일 규격이 바뀌어(예: 쉼이 짧은 발화만) 지연이 더 중요해지면 다시 쓸 수 있다.

## 5. 재현

```
uv run python -m eval.soniox_endpoint_eval --set turnset --out runs/endpoint/ev_turnset --set-env SONIOX_API_KEY= --concurrency 8
uv run python -m eval.soniox_endpoint_eval --set ts172 --out runs/endpoint/ev_ts172 --set-env SONIOX_API_KEY= --concurrency 8
uv run python -m eval.soniox_gate_eval --out runs/endpoint/gate_report.md
```

API 호출: Soniox(BAZE) 녹화 400 스트림, Jev 438회(`data/turnset/jev_cache.jsonl`에 추가, 다시 돌리면 0회).

## 6. 실녹음 평가: onedam_cs_testset (AG-T31, 2026-11-21)

- 데이터: `onedam_cs_testset/agent_world/` — 사람 녹음 100문항(화자 5명, 건강기능식품 CS), 3판: 원본 wideband / phone_ulaw / phone_ulaw + babble(SNR 12).
  1초 이상 쉼(`long_pause`) 30문항. 오디오 앞뒤 무음이 거의 없어(꼬리 0~0.36초) 송신 때 무음을 덧붙였다(babble판은 꼬리부터 잡음 없음).
- 기존 결정기는 **Soniox 이벤트 재생**(`simulate_detector`)으로 쟀다: 세션과 같은 VAD·finalize(침묵 200ms 요청, +240ms) 순서, 텍스트는 Soniox 부분 결과.
  같은 재생을 turnset·172문항에 돌리면 turn_sim(Gemini 지점 텍스트)과 비슷하거나 조금 낫다: turnset 10.1 → 9.2%, 172문항 7.0 → 6.4%
  (172문항 miss 7.6 → 11.0%는 대부분 babble 꼬리 1.5초 잡음 때문). **Soniox 토큰 지연이 기존 결정기를 해치지 않는다.**
- 라벨 주의: `speech_end`가 실제 음성 끝보다 늦은 문항이 있다(VAD 끝이 라벨보다 0.2초 이상 이른 문항 14개, 보통은 VAD가 +80ms).
  그래서 "말 끝 전 확정" 중 판정 시각 뒤에 (원본 녹음 기준) VAD 음성이 100ms 이상 없는 것은 "라벨 늦음"으로 따로 셌다.

| 판 | 방식 | 말 끝 전 확정(라벨) | 실제 끊음(뒤에 음성 있음) | miss | 지연 p50 / p95 ms |
|---|---|---|---|---|---|
| wideband | **기존 결정기** | 14 | **14** | 4 | 730 / 1360 |
| | Soniox 단독 | 36 | 29 | 3 | 548 / 764 |
| | Soniox + 규칙 | 28 | 22 | 3 | 551 / 1170 |
| | **Soniox + Jev** | 15 | **13** | 4 | 694 / 1000 |
| | Soniox + 규칙 + Jev | 15 | 13 | 4 | 699 / 1160 |
| phone | **기존 결정기** | 14 | **14** | 3 | 740 / 1378 |
| | Soniox 단독 | 35 | 28 | 0 | 535 / 795 |
| | Soniox + 규칙 | 28 | 21 | 0 | 545 / 1089 |
| | **Soniox + Jev** | 17 | **14** | 0 | 683 / 981 |
| | Soniox + 규칙 + Jev | 17 | 14 | 0 | 683 / 1078 |
| phone + babble | **기존 결정기** | 2 | **2** | 7 | 900 / 1430 |
| | Soniox 단독 | 37 | 31 | 5 | 606 / 1123 |
| | Soniox + 규칙 | 16 | 12 | 10 | 608 / 1308 |
| | **Soniox + Jev** | 5 | **3** | 14 | 758 / 1300 |
| | Soniox + 규칙 + Jev | 5 | 3 | 13 | 763 / 1300 |

(n=100이라 건수 = %. Jev thr 0.5, hold 1000ms. thr 0.3~0.85 사이에서 결과가 거의 같다.)

쉼 유형별(wideband, 라벨 기준 말 끝 전 확정):

| 방식 | long_pause 30 | 그 밖 70 |
|---|---|---|
| 기존 결정기 | 8 | 6 |
| Soniox + 규칙 | 9 | 19 |
| Soniox + Jev | 7 | 8 |

해석:
- **실녹음에서는 Soniox + Jev가 기존 결정기와 비슷하다**(실제 끊음 13 vs 14, 14 vs 14). 합성셋(turnset 23.7% vs 10.1%)과 다른 결과다.
  합성셋은 "완결 문장 + 쉼"(trap)을 일부러 많이 넣었고 대화 맥락이 없다. onedam은 실제 말투에 대화 맥락·고객 정보가 있다.
- 둘이 틀리는 문항이 다르다(wideband 겹침 9). 기존은 "-는데요"(soft_final) 뒤 짧은 쉼(q001·q023·q090)과 "-싶은데"(q097)에서 끊고,
  Jev는 그걸 잡는다(p 0.69~0.96으로 끝까지 기다림). Jev는 1.1~1.8초 긴 쉼(q004 "주문번호는 잠시만요")에서 hold/상한(1.0/1.5초)에
  걸려 끊고(q006 "오늘 보내주시면"도 상한), "빨라서요,"(q041)·"접수해 주시고요,"(q074)를 끝으로 본다(쉼표 뒤 0.6~0.7초 쉼).
- 지연: Soniox + Jev는 p95가 약 0.36~0.4초 짧다(1360 → 1000, 1378 → 981). p50은 36~57ms 짧다.
- babble: 기존 결정기가 끊음은 가장 적지만(2) VAD가 웅성거림을 음성으로 봐서 늦다(p50 900). Soniox + Jev는 miss 14로 더 나쁘다(같은 VAD 약점 + `<end>` 무시).
- 규칙 게이트는 실녹음에서도 약하다(22, 21). Jev가 결정적이다.

결론: 합성셋까지 합치면 여전히 기존 결정기가 안전하다. 다만 실녹음에서 **Soniox + Jev는 끊음이 같고 p95가 0.4초 짧아** 후보가 된다.
가장 유망한 조합은 "기존 결정기 + `<end>`·Jev 높음이면 대기 단축"(서로 다른 문항을 틀리므로 AND/OR 결합 여지)이다. 실녹음 표본이 100문항·화자 5명(20대)이라 확정하기에는 작다.

재현:
```
C=onedam_cs_testset/agent_world/cases.jsonl
for v in audio audio_phone audio_phone_babble; do
  uv run python -m eval.soniox_endpoint_eval --set onedam_$v --cases $C --audio onedam_cs_testset/agent_world/$v --out runs/onedam/ev_$v --set-env SONIOX_API_KEY=
done
uv run python -m eval.soniox_gate_eval --run runs/onedam/ev_audio --run runs/onedam/ev_audio_phone --run runs/onedam/ev_audio_phone_babble --out runs/onedam/gate_report.md
```
API: Soniox(BAZE) 녹화 300 스트림, Jev 238회(캐시). 중간에 STT 비교(Gemini·Soniox, wideband)도 돌렸으나 STT는 Soniox 확정이라 중단했다
(wideband 숫자 정규화 CER 둘 다 0.041, finalize p50 339 vs 225ms).

## 7. 기존 결정기 + Jev 결합 (AG-T32, 2026-11-21)

모두 Soniox 이벤트 재생(`simulate_detector`)이고, 판정 규칙은 위와 같다. onedam은 "실제 끊음"(판정 뒤 원본 녹음에 음성 있음)으로 센다.

| 방식 | turnset 228 | 172문항 | onedam wideband | onedam phone | onedam babble | **합계 828** | miss 합계 | 지연 p95 (turnset / 172 / onedam) |
|---|---|---|---|---|---|---|---|---|
| 기존 결정기 | 21 | 11 | 14 | 14 | 2 | **62** | 55 | 1355 / 1259 / 1360 |
| Soniox + Jev (게이트) | 54 | 20 | 13 | 14 | 3 | 104 | 74 | 1078 / 910 / 1000 |
| 기존 + Jev 늘리기 lo 0.3 — AG-T24 그대로(텍스트 정확 일치, 침묵당 2회) | 21 | 10 | 14 | 14 | 2 | 61 | 55 | 1359 / 1259 / 1360 |
| 기존 + Jev 늘리기 lo 0.3 — 최근 응답, 침묵당 6회 | 21 | 10 | 12 | 11 | 1 | 55 | 57 | 1371 / 1259 / 1360 |
| **기존 + Jev 늘리기 lo 0.5 — 최근 응답, 침묵당 6회** | 21 | **9** | **10** | **10** | **1** | **51** | 59 | 1387 / 1259 / 1360 |
| 기존 + Jev 늘리기 lo 0.3 + 줄이기 hi 0.95→600ms — 최근 응답 | 23 | 11 | 14 | 13 | 1 | 62 | 55 | 1372 / 1252 / 1360 |
| 기존 + `<end>`·Jev (p ≥ 0.9면 즉시, ≤ 0.3이면 늘리기) | 26 | 13 | 15 | 17 | 2 | 73 | 52 | 1354 / 1173 / 983 |

(지연 p50은 늘리기 방식 모두 기존과 ±35ms 안. 전체 표는 `runs/endpoint/combo_report.md`, 실제 끊음은 `combo_real.json`.)

해석:
- **AG-T24 결합(`JevTurnDetector`)은 Soniox에서 사실상 효과가 없다**(62 → 61). Jev 확률을 "지금 텍스트와 같은 텍스트"일 때만 쓰고 침묵당 2회만 묻는데,
  Soniox 부분 결과는 침묵 중에도 계속 바뀐다("오메가3 주문" → "주문했" → "주문했는데요"). Jev는 앞 조각에 0.04~0.26(안 끝남)을 줬지만
  결정 시점 텍스트와 달라 버려졌다(q001·q023·q090).
- **"최근 응답" 규칙으로 바꾸면 늘리기가 작동한다**(같은 침묵에서 가장 최근에 온 p를 텍스트가 바뀌어도 쓰고, 텍스트가 바뀔 때마다 최대 6회 묻는다).
  lo 0.5에서 실제 끊음 62 → **51(−18%)**. 다섯 셋 모두 나빠지지 않는다. 대가는 miss +4건(55 → 59), turnset p95 +32ms. 172·onedam p95는 그대로다.
  늘리기만 하므로 Jev가 실패·지연되면 표 결정기 그대로다(안전한 방향).
- **줄이기(hi)·`<end>` 즉시 확정은 끊음을 늘린다**(62, 73). 지연(p95 onedam 1360 → 983)을 얻지만 0점 위험이 커진다. AG-T24 결론과 같다.
- lo 0.5는 같은 test 데이터에서 고른 값이다(lo 0.3도 55로 이득). 쓰기 전에 다른 데이터로 확인이 필요하다.

권장: 엔드포인팅 = **기존 결정기 + Jev 늘리기(lo 0.3~0.5, 최근 응답 규칙)**. 그러려면 `JevTurnDetector`의 텍스트 일치 규칙과 요청 상한을 바꿔야 한다
(지금 코드는 시뮬레이션의 "AG-T24 그대로" 행과 같다). 외부 API(Jev) 의존이 생기므로 스위치(`DIRECTON_JEV_TURN`)로 둔다.

재현: `uv run python -m eval.soniox_gate_eval --combos --run runs/endpoint/ev_turnset --run runs/endpoint/ev_ts172 --run runs/onedam/ev_audio --run runs/onedam/ev_audio_phone --run runs/onedam/ev_audio_phone_babble --out runs/endpoint/combo_report.md`
API: Jev 1,676회(침묵 지점 부분 텍스트, 캐시). Soniox 새 호출 없음.
