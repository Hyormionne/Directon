# 조회 뒤 답까지 시간(after_tool) 측정과 단축 (AG-T21)

- 측정: 2026-11-20 00:13~01:10. 툴 문항 121개 중 case_id 순으로 3개에 1개씩 고른 **40문항**(`runs/after_tool/cases40.txt`). mock 8121 / agent 9121, `--concurrency 4`, judge(Gemini 3.8 flash) 포함
- 조건: LLM은 Gemini 직접 키(hedge 500ms + 컨텍스트 캐시, `llm_latency_r2.md` 기본값), BAZE 끔, TTS edge. 스크립트 `runs/after_tool/run.sh <이름> <POST_TOOL_HEDGE> <TOOL_TEMPLATE>`, 비교표 `runs/after_tool/analyze.py`, LLM·TTS 구간 `runs/after_tool/llm_gap.py`. 턴 로그는 `DIRECTON_LOG_DIR=runs/after_tool/<이름>/logs`
- 왜: 공식 속도 점수는 함수 문항에서 **"말 끝 → 첫 소리" 점수와 "함수 결과 → 실제 답" 점수의 평균**이다(`docs/04_목표_점검.md`). 지금까지는 앞쪽만 줄였다.

## 결론

| 스위치(환경변수) | 코드 기본값 | 권장 | 의미 |
|---|---|---|---|
| `DIRECTON_POST_TOOL_HEDGE` | **1** | 1 | 툴 결과 뒤 LLM 호출도 동시 요청(툴을 낸 모델 먼저, 500ms 뒤 대체 모델) |
| `DIRECTON_TOOL_TEMPLATE` | 0 | **1** | 툴 결과에 적힌 사실만으로 만든 첫 문장을 LLM보다 먼저 말한다 |

깨끗한 비교(B2·C2, 같은 코드로 연달아 실행, 서버 경고 0~1건):

| 지표 | B2: hedge만 | **C2: hedge + 템플릿** |
|---|---|---|
| 조회 뒤 첫 유성음(측정기 정의, 채움말 포함) p50 / p95 | 0.15 / 1.83 s | 0.39 / **0.84 s** |
| 조회 뒤 본 응답(채움말 제외, 턴 로그) p50 / p90 / p95 | 2.32 / 2.95 / 3.15 s | **0.73** / 2.68 / 2.92 s |
| 공식 속도(채움말을 답으로 침) | 91.6 | **92.5** |
| 공식 속도(채움말을 답으로 안 침, 보수적) | 83.7 | **90.5** |
| 첫 소리 p50 / p95 | 0.81 / 1.39 s | 0.79 / 1.41 s |
| 함수 점수 / 정확률 | 82.5 / 0.78 | 80.0 / 0.75 |
| 응대(judge) | 70.0 | 70.4 |
| 추정총점(공식 속도) | 79.0 | 79.0 |
| 템플릿 발화 / 반복 문장 제거 | - | 25회 / 1회 |

- **템플릿을 켜면 실제 답이 툴 결과 뒤 0.7초에 나온다**(p50, 2.3초 → 0.7초). 측정기 정의(채움말 포함)의 p95도 1.83초 → 0.84초로, 모든 함수 문항이 1.5초(100점) 안에 들어온다. 응대·함수 점수는 표본 오차 범위 안에서 같다(1문항 = 2.5점).
- 템플릿을 코드 기본으로 켜지 않은 이유: 기존 `tests/test_llm_eval.py`의 필러 테스트 2개(`test_filler_recorded_in_tool_turn`, `test_exact_match_with_normalized_args_and_tool_result_fed_back`)가 "툴 결과 뒤 필러" 동작을 단정한다. 기존 단정을 바꾸지 않기 위해 코드 기본은 끄고, **서버 실행 시 `DIRECTON_TOOL_TEMPLATE=1`을 권장**한다(당일 체크리스트·`.env`에 반영 필요 — 다른 트랙 파일이라 이 Task에서는 바꾸지 않았다).
- 후속 hedge는 p50을 바꾸지 않고 꼬리만 자른다: 툴 결과 → 다음 LLM 첫 이벤트 최대값이 3.97~4.41초(고정 모델, 2.5초 제한 뒤 폴백)에서 1.99~2.05초로 줄었다(아래 2절).

## 1. 측정 지표 (eval)

- `eval/runner.py`: `after_tool_ms` = 마지막 함수 응답(`/calls`의 `t_response`, 측정기 중계 시각과 같은 의미) → 그 뒤 첫 유성 프레임(RMS ≥ 300). Starter Kit `run_eval.py`의 `after_tool_sound`를 ms로 같이 적는다.
- 턴 로그(`agent/responder.py`): `after_tool_ms`(서버가 마지막 툴 결과를 받은 뒤 첫 유성 PCM 송신, 채움말 포함)와 `after_tool_answer_ms`(채움말을 뺀 본 응답 문장 — LLM 또는 템플릿). 측정기 값과 서버 값은 50ms 안에서 일치했다(A_base 9건 대조).
- `eval/score.py`: 기존 `speed`(첫 소리만)·`total*`은 그대로 두고 `speed_after_tool`, `speed_official`(함수 문항 = 평균), `speed_official_answer`(턴 로그의 본 응답 기준), `total_official`·`total_no_judge_official`을 추가했다. run 폴더의 `logs/*.jsonl`을 자동으로 읽어 session_id로 붙인다(`--turn-logs`로 다른 위치 추가). 보고서에 "조회 뒤" 줄과 열(속도(공식), 조회뒤p50/p95, 조회뒤답p50, 추정총점(공식))이 생겼다.
- **"실제 답"의 해석이 열려 있다.** Starter Kit 코드는 함수 응답 뒤 첫 유성음만 보므로 채움말도 답으로 친다. 공식 사이트는 "조회 결과 뒤 실제 답"이라고 적었다. 두 값을 모두 낸다. 채움말을 줄이는 AG-T20이 들어가면 두 값이 가까워진다.

### 기준선(A_base, 템플릿·후속 hedge 끔)
| 지표 | 값 |
|---|---|
| 조회 뒤 첫 유성음 p50 / p95 | 0.14 / 2.36 s |
| 조회 뒤 본 응답 p50 / p90 / p95 | 1.96 / 2.54 / 2.99 s |
| 툴 결과 → LLM 첫 이벤트 p50 / 최대 | 1.22 / 3.97 s |
| 문장 TTS 첫 바이트(edge) p50 / p90 | 0.65 / 0.97 s |
| 공식 속도 / 보수적 / 첫 소리만 | 90.1 / 86.6 / 92.5 |

- p50이 0.14초인 것은 필러("확인해 드릴게요, 잠시만요.", 툴 회차 시작 150ms 뒤)가 덮기 때문이다. 필러가 없는 문항이 꼬리를 만든다: **`transfer_to_agent`(필러 제외 목록)**, **같은 턴의 두 번째 툴 회차**(필러는 턴당 한 번, 예: get_order_list → cancel_order). 이 9건이 1.6~4.7초였다.
- 본 응답까지는 LLM 재호출(약 1.2초) + 문장 TTS 첫 바이트(약 0.65초)다. Gemini 스트림은 첫 이벤트에 첫 문장이 통째로 오는 일이 많아 "문장 단위 즉시 TTS"(이미 구현돼 있음, 테스트 `test_first_post_tool_sentence_goes_to_tts_before_stream_ends`)만으로는 더 줄지 않는다.

## 2. 후속 hedge

- HedgedLLM은 툴 후속 라운드를 툴을 낸 모델에 고정하고(thought_signature 때문), 그 모델이 2.5초 안에 첫 이벤트를 못 내야 다른 모델로 넘긴다. 응답기는 provider를 고치지 않고 같은 모델 목록으로 **새 HedgedLLM을 만들어**(툴을 낸 모델을 앞에, `_pins` 기록으로 찾음) 후속 호출에 쓴다. 대체 모델은 서명 없는 호출 id에 GeminiLLM의 공식 우회 서명을 붙인다(기존 폴백 경로와 같다).

| 실행 | 툴 결과 → LLM 첫 이벤트 p50 / p90 / 최대 (ms) |
|---|---|
| A_base(고정) | 1217 / 1474 / 3974 |
| B_hedge | 1268 / 1470 / 1990 |
| A2_base(고정) | 1392 / 3594 / 4406 |
| B2_hedge | 1355 / 1736 / 2020 |
| C2_template(hedge) | 1243 / 1879 / 2052 |

- 비용: 후속 호출도 500ms 뒤 대체 모델에 한 번 더 보낸다(입력은 캐시 단가). 질문당 약 0.02¢ 이하 추정.

## 3. 결과 템플릿

- `tool_result_template(tool, result)`: 결과 status가 `ok`·`not_allowed`·`no_refund`일 때만. ① 결과 `message`의 첫 문장이 40자 이하이고 숫자·지시 표현("고객에게", "안내하" 등)이 없으면 그대로. ② 아니면 **조회 함수**(get_/track_/check_…)이고 data에 상태 값이 있을 때만 "확인해 보니 현재 {상태} 상태입니다." 쓰기 함수는 상태만으로 결과를 단정하지 않는다. 우리 차단 메시지(`invalid_arguments`)·오류는 말하지 않는다. 한 회차에 툴이 하나일 때만 쓴다.
- LLM에는 툴 결과에 `_already_said`(말한 문장)와 `_note`("되풀이하지 말고 이어서")를 붙인다. LLM 문장이 템플릿과 같거나 어미만 바꾼 반복이면 뺀다(`repeats_template`).
- 템플릿이 나가면 아직 안 나간 필러는 취소한다(답 뒤에 "잠시만요"가 오지 않게).
- C2에서 말한 템플릿 25개는 모두 툴 결과에 있는 문장·상태였다. 남은 문제: 약 5/25건은 LLM이 같은 내용을 다른 말로 한 번 더 말했다(예: "배송지가 변경되었습니다. 새로 말씀하신 주소로 배송지가 변경되었어요."). 반복 제거 규칙 보강 뒤(어미 변형) 일부는 걸러진다. 상태 문자열이 "회수완료·환불진행중"처럼 어색한 경우가 있다(데이터 그대로).
- 당일 공식 함수: message가 없거나 다른 형식이면 ①이 빠지고 ②(조회 함수의 status)만 쓴다. 둘 다 없으면 템플릿 없이 기존 동작이다.

## 4. 측정 중 이상(오염된 실행)

- 다른 트랙이 같은 작업 트리를 고치는 중에 측정했다. **B_hedge·C_template·A2_base는 비교에서 뺐다.** B_hedge부터 turn_end p50이 4.4초 → 10.8초로 늘었다(본 응답 첫 오디오 뒤 약 5초 — 세션 송신을 재생 속도로 바꾼 다른 트랙 변경으로 보인다). A_base와 다른 세션 코드였다. C_template은 edge-tts 시간 초과 → GeminiTTS 대체 → **429 할당량 초과**와 STT 1차 실패가 겹쳐 함수 문항 25개만 남았다(함수 53.8, 응대 40.0). A2_base도 응답 실패 18건(429 등)이 있었다(응대 37.5). 깨끗한 B2·C2만 표에 썼다.
- 결과적으로 "템플릿·hedge 모두 끔"의 깨끗한 짝 비교는 A_base(이전 세션 코드)뿐이다. 본 응답 p50은 A_base 1.96초, B2 2.32초로 세션 송신 방식 차이가 섞여 있다.

## 5. 자릿수 가정 제거

- `REPAIR_*`를 번호 길이 비율로 바꿨다(`repair_thresholds(L)`: L=10 → (7, 2, 6, 2) 그대로, L=4 → (3, 1, 3, 1)). `match_order_id`는 길이 인자가 없으면 알려진 주문번호(get_order_list 결과)의 가장 흔한 길이를 쓴다.
- 발화 번호 추출 최소 자리 수는 `min(7, catalog.order_id_lengths(), catalog.phone_lengths())`(AG-T17 helper). 기본 데이터는 7 그대로, 4자리 데이터면 4. 7자리보다 짧은 번호는 한 자리 차이까지만 교정한다(4자리에서 편집 거리 3은 사실상 아무 번호다).
- 차단 메시지의 "주문번호는 열 자리"를 스키마의 N으로 읽는다("네 자리", "열한 자리").
- 번호 교정 로그는 `_privacy.redact`로 가린다(저장 금지 모드).

## 6. 재현·비용
```
JUDGE=1 bash runs/after_tool/run.sh B2_hedge 1 0
JUDGE=1 bash runs/after_tool/run.sh C2_template 1 1
uv run python runs/after_tool/analyze.py A_base B2_hedge C2_template
uv run python runs/after_tool/llm_gap.py A_base B2_hedge C2_template
```
API 호출(전체 6회 실행): 240문항 × LLM 약 2~3회(+hedge 대체) + judge 240회 + STT. 추정 비용 약 $0.5 이하.
