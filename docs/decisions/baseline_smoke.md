# 스모크 베이스라인 (AG-T09 1부)

- 측정: 2026-11-19 18:27·18:31 (2회, 결과 거의 같음) · Windows 개발 PC · Gemini 무료 티어
- 구성: `uv run python -m agent.server --port 9000` 기본값
  - STT `gemini-3.5-transcribe-live`(실패 시 batch) / LLM `gemini-3.5-flash-lite` → 폴백 `gemini-3.1-flash-lite`(첫 이벤트 2.5초 제한. `gemini-2.5-flash-lite`는 이 키에서 404라 예열 때 자동 제외)
  - TTS edge `ko-KR-SunHiNeural` + 레벨 정규화(음성 구간 RMS 3000) / Silero VAD / Smart Turn v3.2 / 결정기 `make_detector()`(표 방식 `TableTurnDetector`, 기본 표) / `normalize_args`
- 세트: `data/testset/smoke/cases.jsonl` 20문항(`scripts/build_smoke_set.py`). 범주 쿼터: 조회 4·실행 4·후속 3·정보부족 3·FAQ 2·불만 2·범위밖 2. 음성은 edge InJoon 8·Hyunsu 8·SunHi 4. 채널 wideband 6·phone 9·phone_ulaw 5. 본 테스트셋 음성(Gemini 목소리)은 결제 반영 후 다시 잰다.
- 명령: `eval.runner --concurrency 2` → `eval.score`(응대 심사는 한도 때문에 생략). 원시 결과는 `runs/smoke_e2e_20261119_1831/`, 턴 로그는 `runs/logs/turns-20261119.jsonl`
- 아래 표와 수치는 2회차(18:31, 최종 코드) 결과다.

## 요약

| 지표 | 값 | 목표(구현계획 4.9) |
|---|---|---|
| first_sound p50 / p95 (20문항 전체, 음수 포함) | **0.50s / 1.94s** | p95 ≤ 1.3s |
| first_sound p50 (말 끝 뒤에 낸 12문항만) | 0.51s (최대 1.94s) | |
| spoke_before_end (말 끝 전 발화 = 속도 0점) | **8 / 20** | 잘못 끊기 ≤ 1% |
| 함수 완전 일치 | **7 / 20 (35%)**, 함수 점수 평균 37.5 | ≥ 95% |
| 함수 완전 일치(잘리지 않은 12문항만) | 7 / 12 | |
| 오류(error·turn.end 없음) | 0 | |
| 속도 점수 평균 / 응대 제외 추정 | 59.5 / 50.7 | |
| LLM 폴백 발생 | 1회(3.5-flash-lite 2.5초 시간 초과 → 3.1-flash-lite가 0.7초에 응답) | |

결론: **조립·프로토콜·첫 소리 속도는 동작한다**(첫 마디 캐시가 확정 6ms 뒤 나가고, 말 끝 뒤 첫 소리는 대부분 0.4~0.6초). 점수를 깎는 건 (1) 문장 사이 쉼에서 너무 빨리 끊는 것, (2) STT가 한국어로 끊어 읽은 숫자를 빠뜨리는 것이다.

## 문항별 결과

first_sound는 말 끝(speech_end) 기준 초, 음수면 말 끝 전 발화다. 함수 칸에서 자유 텍스트 인자(reason·new_address)는 뺐다.

| case | 범주 | 태그 | 음성 | 채널/잡음 | first_sound | 말 끝 전 | 기대 함수 | 실제 호출 | 일치 | agent_text(첫 마디 뒤) 요약 |
|---|---|---|---|---|---|---|---|---|---|---|
| T0129 | need_info | banmal,short | InJoon | wideband/pink | 0.61 | | - | - | O | 주문번호나 휴대폰 번호를 되묻는다 |
| T0039 | complaint_handoff | long,caller_known | InJoon | ulaw/none | -4.76 | Y | transfer_to_agent | get_order_list(01078123456) | X | 첫 문장에서 끊긴 뒤 "상담원 연결"을 말로만 하고 함수는 안 부름 |
| T0042 | complaint_handoff | mid_pause,connective_end | InJoon | ulaw/hum | -3.36 | Y | transfer_to_agent | - | X | 부상 사과 + 상담원 연결을 말로만 함 |
| T0123 | tool_action | mid_pause,connective_end,filler,caller_known | Hyunsu | phone/none | -5.14 | Y | get_order_list; cancel_order(2611190412) | - | X | "어." 뒤에서 끊겨 "무엇을 도와드릴까요" |
| T0018 | faq | banmal,filler | Hyunsu | phone/babble | 0.46 | | - | - | O | 전액 환불·출고 전 취소 안내 |
| T0009 | faq | banmal,short | Hyunsu | phone/babble | **1.94** | | - | - | O | STT가 "택배 어디 거 써?"를 "어디 갔어?"로 → 배송 조회 되묻기 |
| T0171 | followup | mid_pause,connective_end,filler,caller_known | SunHi | wideband/babble | 0.40 | | - | get_order_list(01076543210) | X | 회수 일정 안내(불필요한 조회 1건) |
| T0062 | tool_lookup | digits,self_correction | InJoon | phone/white | -2.20 | Y | track_delivery(2611160457) | get_order_info(2511801604) | X | 정정 중간에 끊김 → 번호 재확인 |
| T0167 | followup | digits,self_correction | InJoon | wideband/none | -1.99 | Y | get_member_info(01043210987) | - | X | "다시 할게요"에서 끊김 → 번호 재요청 |
| T0127 | need_info | digits | InJoon | ulaw/pink | 0.51 | | - | get_order_info(261023) | X | STT 숫자 누락(6자리) → 재확인 |
| T0109 | tool_action | mid_pause,connective_end,digits | Hyunsu | wideband/hum | 0.50 | | cancel_order(2611200517) | cancel_order(261200517) | X | STT 숫자 누락(9자리) → not_found → 재확인 |
| T0134 | need_info | filler,long | Hyunsu | phone/none | -1.78 | Y | - | - | O | "어…" 뒤에서 끊김 |
| T0033 | out_of_scope | mid_pause,filler | Hyunsu | wideband/none | 0.46 | | - | - | O | 번역은 범위 밖이라고 안내 |
| T0025 | out_of_scope | banmal,short | SunHi | wideband/white | 0.52 | | - | - | O | 로또 번호 추천 거절 |
| T0080 | tool_lookup | digits,long | SunHi | phone/hum | 0.50 | | track_delivery(2611160712) | track_delivery(621210711) | X | STT 숫자 오인식 → 재확인 |
| T0116 | tool_action | mid_pause,digits,self_correction,negation | InJoon | phone/white | -7.33 | Y | change_shipping_address(2611190871) | - | X | 첫 세그먼트 뒤에서 끊김. STT "뉴욕행 19087편"(1회차엔 항공편 응대) |
| T0118 | tool_action | mid_pause,connective_end,digits,banmal | InJoon | ulaw/hum | 1.13 | | cancel_order(2611160457) | get_order_info(16110457) | X | 상한 1.1초로 확정. STT 숫자 누락(8자리) |
| T0085 | tool_lookup | digits,negation | Hyunsu | phone/hum | -4.45 | Y | get_order_info(2611180321) | - | X | "아니고요," 뒤에서 끊김 |
| T0146 | followup | negation,short,caller_known | SunHi | phone/none | 0.56 | | - | - | O | 취소하지 않고 유지 |
| T0086 | tool_lookup | mid_pause,connective_end,caller_known | Hyunsu | ulaw/pink | 0.46 | | get_order_list; track_delivery(2611180956) | get_order_list(01043210987) | X | 목록 결과(상품준비중)로 답하고 track_delivery는 생략 |

## 구간별 지연 (턴 로그, 20턴)

기준점은 결정기 확정(commit). "확정까지"만 마지막 침묵 시작(silence_start, 즉 VAD가 본 말 끝)부터 잰다.

| 구간 | 정의 | p50 | p95 | 최대 |
|---|---|---|---|---|
| 확정까지 | silence_start → commit | 382ms | 528ms | 1066ms |
| 첫 마디 송신 | commit → opener_sent(캐시 PCM 송신 완료) | 6ms | 8ms | 8ms |
| STT 최종 | commit → finalize_done(finalize는 침묵 200ms에 미리 시작) | 190ms | 289ms | 292ms |
| LLM 첫 이벤트 | commit → llm_first_token | 1049ms | 1451ms | 3427ms(폴백 1회) |
| 툴 호출 | tool_call → tool_result (Mock 로컬) | 6ms | 7ms | 7ms |
| TTS 첫 바이트 | 문장 tts_request → tts_first_byte (edge, 40문장) | 663ms | 1077ms | 1271ms |
| 본 응답 첫 오디오 | commit → 첫 문장 TTS 첫 바이트 | 1858ms | 2932ms | 4353ms |
| turn.end | commit → turn_end | 2444ms | 3649ms | 5023ms |
| LLM 선호출 재사용 | 침묵 200ms 선호출이 그대로 쓰인 턴 | 2 / 20 | | |

- 첫 소리는 "확정까지 + 6ms + 망 지연"이라 0.4~0.6초에 나온다. LLM·TTS 지연은 첫 소리 점수에 영향이 없고, 첫 마디("네, 고객님." 약 1초)와 본 응답 사이에 약 0.9초 빈틈이 생긴다(응대 품질 문제).
- 선호출 재사용이 2/20뿐이다. 침묵 200ms 시점에 STT `latest_text()`가 대부분 비어 있어(아래 문제 3) 선호출을 시작하지 못하고, finalize 뒤에 LLM을 부른다.
- 서버 시작 예열: 모델 0.15초, 캐시 4문구 합성 + LLM 체인 예열 + STT 연결 1회를 병렬로 해서 첫 기동 5.1초(캐시가 있으면 1.1초). 첫 문항도 다른 문항과 지연이 같았다(T0129 commit→LLM 1.1초).
- 캐시 첫 100ms 청크 RMS: 2477~3120(측정기 기준 300의 8배 이상). edge 원음은 음성 구간 RMS 4500~5800이라 원래도 300을 넘지만, 정규화로 캐시와 실시간 문장을 같은 레벨(3000)로 맞췄다.

## Starter Kit 원본 측정기 호환

`"DIRECT_ON 2026 Starter Kit/eval/run_eval.py"`로 데모 3건(결과 `runs/starter_demo_20261119_1831/`): 세 건 모두 오류 없이 PCM·`agent.transcript`·`turn.end`를 받았다 → **프로토콜 호환 확인.**
- first_sound: DEMO1 -0.03s, DEMO2 -1.70s, DEMO3 7.31s. 13절에 적힌 대로 데모 meta.json의 speech_end가 wav와 맞지 않아서다(턴 로그 기준 확정은 침묵 400·400·900ms 뒤였다). 데모 음성은 영어가 아니라 한국어였고 STT도 한국어로 받아 적었다(예: "카드 사용이 어렵네요.").
- 원본 측정기는 results.jsonl을 시스템 인코딩(cp949)으로 쓴다. 읽을 때 주의.

## 관찰된 문제

1. **문장 사이 쉼에서 너무 빨리 끊는다(8/20, 가장 큼).** 8건 모두 Smart Turn p가 0.66~0.99로 높고 결정기가 400~600ms 대기로 확정했다. 유형:
   - 여러 문장 발화의 첫 문장 끝(T0039, T0042, T0116): edge가 마침표에서 400~600ms 쉰다. 첫 문장만 보면 정말로 끝난 것처럼 들린다.
   - 정정·나열 중간(T0062, T0167, T0085): "…, 아 잠깐만요," / "다시 할게요," / "아니고요,"의 쉼표 쉼.
   - 간투사 "어…" 뒤(T0123, T0134): 확정 순간 STT 텍스트가 비어 있어 규칙이 `neutral`이었다. 같은 텍스트("어.")가 들어갔다면 `filler`(hold 1100ms)였다.
2. **STT가 끊어 읽은 숫자를 빠뜨린다.** "이육일일 일오공 이삼" → `261023`, "이육일일 이공공 오일칠" → `261를 200517`, "이육일일 일육공 칠일이" → `621210711`, "이육일일 일육공 사오칠" → `16110457`. 실측(bench)의 깨끗한 음성에선 CER 0이었는데, 통화 채널·잡음·끊어 읽기에서 무너진다. normalize_args는 빠진 숫자를 되살릴 수 없다. LLM은 자릿수 확인 규칙대로 되묻는다(조회·실행 실패).
3. **STT 부분 결과(latest_text)가 확정 시점에 자주 비어 있다.** 4.7초 발화(T0109)에서도 확정 순간 텍스트가 비어 결정기가 텍스트 없이 판단했다(`text=neutral:+0.0`이 20턴 중 7턴). 그래서 텍스트 규칙(hold/연결어미)과 선호출이 거의 쓰이지 않는다. Gemini live 부분 결과 지연(bench: 말 시작부터 1.3초)보다 더 늦다. 원인 조사 필요.
4. **babble 잡음을 음성으로 본다.** T0009(phone/babble): 실제 발화 약 1.5초인데 speech_ms 2860, 첫 소리 1.94초. VAD 0.5 임계가 웅성거림에 걸린다.
5. **LLM 응대 문제(함수 점수):** 상담원 연결을 말로만 하고 `transfer_to_agent`를 부르지 않음(T0039, T0042 — 둘 다 끊긴 탓도 있음), 후속 턴에서 불필요한 `get_order_list`(T0171), 목록 결과로 답하고 `track_delivery` 생략(T0086). 응답 첫머리에 "고객님,"이 첫 마디 "네, 고객님."과 겹치는 경우가 있고, 숫자를 "여섯 이일, 일오"처럼 어색하게 읽는다.
6. **Gemini 3.5-flash-lite가 가끔 첫 이벤트 2.5초를 넘긴다**(2회차 20턴 중 1회). 폴백이 0.7초 안에 받아 줬다. 무료 티어 혼잡 때문이라 시간대에 따라 더 잦을 수 있다.

## AG-T10 개선 후보 (우선순위)

1. **조기 확정 억제(속도 0점 8건)** — 결정기 보정(AG-T12와 함께): 텍스트가 비어 있거나 `neutral`이면 fast 경로(400ms) 금지, 쉼표·"아"·"잠깐만요"·"다시"·"아니고요" 뒤는 hold, 여러 문장 발화(speech_ms 길고 텍스트가 문장 하나뿐)는 대기를 늘린다. 확정 뒤 음성이 다시 들리면 응답을 멈추는 바지-인 처리는 점수상 의미가 없으니(이미 0점) 확정 전 판단에 집중. *잘못 끊기 0점이 지연 몇백 ms보다 훨씬 비싸다* — 현재 첫 소리 p50 0.5초라 대기를 600~900ms로 늘려도 1.5초 안이다.
2. **숫자 인식** — STT 문맥 힌트(`DIRECTON_STT_CONTEXT`: "숫자는 들은 그대로 한 자리씩 아라비아 숫자로") 실험, finalize 텍스트와 부분 결과 비교, 자릿수가 틀리면 batch STT로 숫자 구간 재인식, Soniox 키(AG-U01) 비교.
3. **STT 부분 결과 공백 조사** — `GeminiLiveSTT.latest_text()` 갱신 경로 확인. 부분 결과가 살아나면 1번 텍스트 규칙과 선호출(현재 2/20 재사용)이 같이 좋아진다. 그 전엔 finalize가 끝났으면 그 텍스트를 결정기에 주는 방법도 있다(침묵 200ms + 약 190ms).
4. **babble 잡음 VAD** — 임계 상향 또는 에너지·스펙트럼 조건 추가, 잡음 바닥 추정. T0009처럼 첫 소리가 1초 이상 밀린다.
5. **프롬프트** — 상담원 연결 요청·불만 에스컬레이션은 반드시 `transfer_to_agent` 호출, history로 이미 아는 정보는 다시 조회하지 않기, 응답 첫머리 "고객님" 반복 금지, 숫자 읽기 방식.
6. **첫 마디와 본 응답 사이 빈틈(약 0.9초)** — 선호출 재사용률을 올리거나, 첫 마디를 조금 길게("네, 고객님. 확인해 보겠습니다.")·상황별로 고르기.
7. **LLM 폴백 튜닝** — 첫 이벤트 제한 2.5초를 1.5초로 낮추거나 두 모델에 동시 요청(hedging) 실험. 3.1-flash-lite는 툴 정확도가 낮다(bench 5/7)는 점을 같이 볼 것.

## 재현

```
uv run python scripts/build_smoke_set.py --build
uv run python -m mock_api.server --port 8100 &
uv run python -m agent.server --port 9000 &
uv run python -m eval.runner --server ws://127.0.0.1:9000 --cases data/testset/smoke/cases.jsonl \
  --audio data/testset/smoke/audio --mock http://127.0.0.1:8100 --out runs/smoke_e2e_<시각> --concurrency 2
uv run python -m eval.score --run runs/smoke_e2e_<시각>
```
구간별 지연 집계 스크립트는 `runs/server_logs/latency.py`(gitignore 영역, 임시).
