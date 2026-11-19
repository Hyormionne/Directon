# AGENT_PLAN — 한국어 음성 CS 에이전트 구현

> 작성일 2026-11-19 · 기준 `main`(초기 커밋) · 브랜치 `main` · Task 접두사 `AG-` · 결정 접두사 `AD` · 사용자 작업 `AG-U*` · 커밋 scope `agent|turn|providers|mock|eval|docs`
> 상태: **진행 중.** AD1~AD5 확정(2026-11-19).

- 근거: 사용자 요청(2026-11-19).
  1. "한번 구현해봐. 가능하면 TestCase용 음성도 생성해서 최적화해. 통화 환경과 발화 내용 등 여러 변수를 고려해서 테스트를 제작해. 서브에이전트 적극 사용해서 빠르게 병렬작업해. /fast-dev-workflow 참고해서 진행해"
- 관련 문서: `docs/01_구현계획.md`(설계), `docs/00_대회규격_분석.md`(규격), `docs/research/01~03`(조사)
- 진행 규칙은 CLAUDE.md와 같다.

---

## 1. 현재 흐름 (2026-11-19)

- 코드 없음. Starter Kit만 있다: `example_server/server.py`(RMS 800ms 침묵 → 삐 소리), `mock_tools/mock_api_template.py`(`lookup()` 미구현), `eval/run_eval.py`(측정기), `eval/score_demo.py`(속도 표).
- 사용 가능한 키: `GEMINI_API_KEY`만 있다. Soniox·OpenAI·TypeSafe·ElevenLabs 키는 없다.
- 이 키로 쓸 수 있는 모델(2026-11-19 `models.list` 확인): `gemini-3.5-transcribe-live`(bidi STT), `gemini-3.5-transcribe`, `gemini-3.8-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-3.8-flash-tts`, `gemini-3.8-flash-lite-tts`, `gemini-2.5-flash-preview-tts`, `gemini-embedding-2`.
- 하드웨어: CPU 16스레드, GPU 없음.

## 2. 문제 진단

| # | 문제 | 근거 |
|---|---|---|
| 1 | 계획서의 STT(Soniox)·LLM(OpenAI) 키가 없다 | 환경변수 확인 |
| 2 | 한국어 테스트 음성이 없다(데모는 영어 3개) | `Starter Kit/data/audio/demo` |
| 3 | 공식 14개 함수·매뉴얼이 미공개 | `Starter Kit/docs/MOCK_API_TASK.md` |

## 3. 설계

`docs/01_구현계획.md` 3~4절을 따른다. 차이점만 적는다.

### 3.1 Provider 교체 구조 (AD1, AD2)
- 모든 외부 서비스는 `agent/core/types.py`의 Protocol 뒤에 둔다. `agent/providers/factory.py`가 환경변수와 설정으로 고른다.
- STT 기본값: `SONIOX_API_KEY`가 있으면 Soniox, 없으면 Gemini(`gemini-3.5-transcribe-live` 스트리밍, 실패하면 `gemini-3.5-transcribe` 일괄).
- LLM 기본값: `OPENAI_API_KEY`가 있으면 `gpt-5.4-mini`, 없으면 Gemini Flash 계열 중 0단계 실측에서 가장 빠른 모델.
- TTS: `gemini-3.8-flash-tts`(대안 flash-lite-tts). 첫 마디·필러·고정 문구는 캐시.

### 3.2 테스트셋 형식 (AG-T05/T06/T09 사이의 계약)
- `data/testset/cases.jsonl`: 한 줄에 한 문항.
  ```json
  {"case_id":"T0001","now":"2026-11-21 11:00","history":[],"customer_profile":null,
   "customer_utterance":"주문한 거 언제 와요?",
   "segments":[{"text":"주문번호가요","pause_after_ms":700},{"text":"이육일일일오공이삼일이요. 언제 와요?","pause_after_ms":0}],
   "voice":"gemini:Kore",
   "env":{"channel":"phone_ulaw","noise":"babble","snr_db":15,"gain_db":-6,"speed":1.0,"lead_ms":400,"packet_loss":0.0},
   "category":"tool_lookup","tags":["mid_pause","digits"],
   "expected_tools":[{"name":"get_order_info","arguments":{"order_id":"2611150231"}}],
   "expected_points":["배송 상태(배송중)를 안내한다"],"must_not":["주문 취소를 실행한다"]}
  ```
  - `history` 원소: `{"role":"agent"|"customer","text":"..."}` (공식 형식이 나오면 맞춘다)
  - Starter Kit `run_eval.py`가 읽는 필드(`case_id, now, history, customer_profile`)를 그대로 포함한다.
- `data/testset/audio/<case_id>.wav`(16k mono), `data/testset/audio/meta.json` = `{case_id: {"speech_end": 초}}`. 오디오는 git에 넣지 않는다.

### 3.3 통화 환경 변수 (AG-T06)
| 축 | 값 |
|---|---|
| channel | `wideband`(원음) / `phone`(8kHz 다운샘플 + 300~3400Hz 대역) / `phone_ulaw`(phone + G.711 μ-law) |
| noise | `none` / `white` / `pink` / `babble`(다른 발화를 겹쳐 만든 웅성거림) / `hum`(60Hz 험) |
| snr_db | 30 / 20 / 10 / 5 |
| gain_db | 0 / −12(작은 목소리) / +6(클리핑 근접) |
| packet_loss | 0 / 0.02(20ms 프레임 무작위 무음화) |
| speed | 0.85 / 1.0 / 1.2 |
| lead_ms | 말 시작 전 무음 200~1500 |

### 3.4 발화 내용 변수 (AG-T05)
- 범주: `faq`(정책·영업시간), `tool_lookup`(조회), `tool_action`(취소·반품·교환·주소 변경·콜백), `need_info`(정보 부족 → 되묻기), `out_of_scope`, `complaint_handoff`(불만·상담원 요청), `followup`(history 있는 후속 턴, 숫자만 말하기 등)
- 태그: `mid_pause`(발화 중 300~1200ms 쉼), `filler_words`(어, 음), `connective_end`(연결어미에서 쉼), `digits`(숫자 읽기: 공/영, 끊어 읽기), `self_correction`, `negation`, `banmal`, `long`, `short`, `caller_known`

### 3.5 조립 규약 (AG-T07에서 확정)
- `agent/server.py:build_deps()`가 지연 import: `agent.providers.factory.make_stt/make_tts/make_llm()`(인자 없음), `agent.turn.vad.make_vad|SileroVAD`(세션마다), `agent.turn.smart_turn.make_smart_turn|SmartTurnONNX|SmartTurnV3`(공유), `agent.turn.detector.make_detector|KoreanTurnDetector|TurnDetector`(세션마다), `agent.normalize.normalize_args(tool, args, schema, now)`.
- 첫 마디·필러·사과·재질문 문구는 `agent/config.py:Settings`(환경변수 `DIRECTON_<필드>`), 캐시는 `assets/tts_cache/`.
- transcript는 응답 끝에 한 번(첫 마디·필러 포함), 그다음 turn.end.

## 4. 결정 필요 사항

| ID | 주제 | 선택지 | 권장 | 결정 |
|---|---|---|---|---|
| AD1 | 개발용 STT | A: Gemini transcribe-live(키 있음) / B: Soniox 키 발급 후 진행 | **A.** 지금 키가 있고, Soniox 어댑터도 같이 만들어 키가 생기면 바로 교체 | **A** (2026-11-19 사용자 확정) |
| AD2 | 개발용 LLM | A: Gemini Flash 계열 / B: OpenAI 키 대기 | **A.** 같은 이유. 실측으로 모델 선택 | **A** (2026-11-19 사용자 확정) |
| AD3 | 테스트 음성 합성 | A: Gemini TTS 여러 음성 + edge-tts(무료, 키 불필요) / B: Gemini만 | **A.** 상담원 음성(Gemini)과 다른 엔진이 섞여야 과적합을 피한다. 가상 문장만 보낸다 | **A** (2026-11-19 사용자 확정) |
| AD4 | 가상 쇼핑몰 함수 | A: 공식 미공개이므로 쇼핑몰 함수 14개를 가정해 구현 / B: 데모 2개만 | **A.** 공식 공개 시 데이터만 교체 | **A** (2026-11-19 사용자 확정) |
| AD6 | 상담원 TTS(Gemini TTS 무료 하루 10회) | A: Gemini 결제 연결 / B: 개발 중 edge-tts 대체 / C: 다른 TTS 키 | **A.** 비용 약 $5 | **B** (2026-11-19 사용자 확정) |
| AD7 | 테스트 음성 화자 | A: edge 3종+Heami에 피치·속도 변형 / B: 결제 연결 후 Gemini 음성 사용 | **A** | **B** (2026-11-19 사용자 확정) — T05b는 AG-U04 이후. T12 보정셋은 edge+피치로 먼저 만들고 Gemini 음성은 결제 후 추가 |
| AD8 | 잡음 VAD 래퍼 기본값 | A: 끔 / B: 켬 / C: 결정기 재보정 후 | **A.** 잘못 끊기(0점) 손해가 말 끝 놓침(80~90점)보다 큼 | **A** (2026-11-19 사용자 확정) — `DIRECTON_VAD_NOISE_GATE=1`로 켬 |
| AD5 | 저장소 | A: 이 폴더를 git 저장소로 / B: git 없이 | **A.** Task별 커밋이 워크플로 전제 | **A** (2026-11-19 사용자 확정) |

## 5. 사용자가 직접 확인할 것

| ID | 할 일 | 왜 |
|---|---|---|
| AG-U01 | Soniox·OpenAI 키 발급, `.env`에 추가 | 계획서 기본 스택으로 전환 |
| AG-U02 | 운영진 질문 발송(`00_대회규격_분석.md` 6절) | 허용 API·실행 위치·채점 방식 |
| AG-U03 | 생성된 한국어 테스트 음성 샘플 청취 | 합성 품질 확인 |
| AG-U04 | Gemini API 결제 연결(Tier 1) 또는 대체 TTS 키 | 무료 티어 TTS 하루 10회 — ✅ 2026-11-19 프로젝트 158777464759 결제 연결 확인 |

## 6. Task 상태표

| ID | Task | 의존 | 상태 | 완료일 |
|---|---|---|---|---|
| AG-T01 | 골격: git, uv, 계약(`agent/core/types.py`), CLAUDE.md | — | DONE | 2026-11-19 |
| AG-T02 | Mock API 서버 + 가상 쇼핑몰 함수 14개·데이터·응대 매뉴얼 | AG-T01, AD4 | DONE | 2026-11-19 |
| AG-T03 | 오디오 유틸 + VAD + Smart Turn + 한국어 규칙 + 턴 결정기 | AG-T01 | DONE | 2026-11-19 |
| AG-T04 | Provider(Gemini STT/TTS/LLM, Soniox, OpenAI, edge TTS) + 실측 벤치 | AG-T01, AD1, AD2 | DONE | 2026-11-19 |
| AG-T06 | 테스트 음성 도구(합성·통화환경 증강) + 측정기·채점기 | AG-T01, AD3 | DONE | 2026-11-19 |
| AG-T07 | 세션·응답기·툴 클라이언트·TTS 캐시·서버 | AG-T01 | DONE | 2026-11-19 |
| AG-T08 | 인자 정규화(한국어 숫자·철자·전화번호·날짜) | AG-T01 | DONE | 2026-11-19 |
| AG-T05 | 테스트 시나리오 작성 + 음성 생성 — 분할됨 | AG-T02, AG-T06 | 분할됨 | |
| AG-T05a | 시나리오 텍스트(cases.jsonl) 작성 | AG-T02 | DONE | 2026-11-19 |
| AG-T05b | 테스트 음성 생성(build_testset) | AG-T05a, AG-T06, AG-U04 | DONE | 2026-11-19 |
| AG-T09 | 실제 구성 연결 + 첫 측정(베이스라인) — 분할됨 | AG-T03, T04, T07, T08 | 분할됨 | |
| AG-T09a | 조립·LLM 폴백·레벨 정규화·예열 + E2E 스모크 20문항 | 위와 같음 | DONE | 2026-11-19 |
| AG-T09b | 172문항 베이스라인 측정 | AG-T09a, AG-T05b | DONE | 2026-11-19 |
| AG-T12 | 턴 감지 보정 데이터셋(600발화, 쉼/말끝 자동 라벨) + 오프라인 보정 | AG-T03, AG-T06 | DONE | 2026-11-19 |
| AG-T10 | 최적화 루프 — 분할됨 | AG-T05b, AG-T09, AG-T12 | 분할됨 | |
| AG-T10a | LLM 응대: 프롬프트(인계 함수 호출, 재조회·호칭 반복 금지), 첫 마디-본 응답 공백 | AG-T09a | DONE | 2026-11-19 |
| AG-T10b | STT 숫자 인식: 문맥 힌트, 부분 결과 공백, 자릿수 불일치 시 재인식 | AG-T09a | DONE | 2026-11-19 |
| AG-T10c | 세션: 선호출 재사용(finalize 후 재시작·정규화 비교), 텍스트 기반 첫 마디, babble 잡음 VAD, 빈 텍스트 짧은 발화 처리 | AG-T09a | DONE | 2026-11-19 |
| AG-T10d | 상품명·옵션 별칭 정규화(엑스라지↔XL, 회색↔그레이, 부분 상품명) | AG-T02, AG-T08 | DONE | 2026-11-19 |
| AG-T13 | 벤치마크: 단순 STT+LLM+TTS 조합 172문항 점수 (2026-11-19 사용자 요청) | AG-T09b | DONE | 2026-11-19 |
| AG-T11 | 전체 회귀 + 결과 보고 | 위 전부 | TODO | |

## 7. Task 상세 (테스트 먼저)

### 공통 규칙
- 테스트: `uv run pytest -q`. 외부 API는 fake(CLAUDE.md 표). 실제 호출 테스트는 `@pytest.mark.live`.
- 실측·음성 생성 스크립트(`scripts/`, `eval/`)는 실제 API를 불러도 된다(사용자 요청: 테스트 음성 생성과 최적화). 호출 수는 필요한 만큼만.
- 모듈은 생성자 주입으로 만들고, 전역 상태를 두지 않는다.

### AG-T02 Mock API 서버 + 가상 쇼핑몰
- 테스트 먼저(`tests/test_mock_api.py`):
  - `GET /tools`가 14개 함수를 `[{"type":"function","function":{...}}]` 형식으로 준다
  - 정의된 입력이면 정의된 출력 + `/calls`에 `matched=true`, `X-Session-Id` 기록
  - 정의되지 않은 입력이면 함수별 not_found 응답 + `matched=false`
  - 모르는 함수는 404, 잘못된 JSON은 400
- 구현: `mock_api/server.py`(Starter Kit 템플릿과 같은 경로·형식), `mock_api/data/tools.json`, `mock_api/data/mock_data.json`, `cases/manual.md`(가상 쇼핑몰 응대 매뉴얼)
  - 함수 후보: get_member_info(phone_number), get_order_list(phone_number), get_order_info(order_id), track_delivery(order_id), cancel_order(order_id, reason), request_return(order_id, reason), request_exchange(order_id, reason), change_shipping_address(order_id, new_address), get_refund_status(order_id), get_product_info(product_name), check_stock(product_name, option), get_coupons(phone_number), request_callback(phone_number, preferred_time), transfer_to_agent(reason)
  - 주문번호는 숫자 10자리, 전화번호는 `010`+8자리(하이픈 없음)

### AG-T03 오디오·턴 감지
- 테스트 먼저(`tests/test_audio.py`, `tests/test_turn.py`):
  - `rms`, `trim_silence`, `resample`(24k→16k 길이 2/3), `frames()` 20ms 분할
  - Silero VAD: 무음 프레임 확률 < 0.2, 데모 음성 구간 평균 확률 > 0.5, 20ms→512샘플 재분할
  - Smart Turn: 데모 음성 전체 → [0,1] float, 8초 초과 입력은 뒤 8초만 사용
  - 한국어 규칙: "배송 언제 와요?"→끝남, "주문번호가요"→끝남 아님 수준의 판별(`end_score`), "그리고"·"어"·연결어미 → 보류
  - 결정기: 침묵 < 200ms면 항상 false, 최대 1100ms면 항상 true, 400/600/900ms 경계 규칙, `expect_short`이면 단축
- 구현: `agent/audio.py`, `agent/turn/vad.py`, `agent/turn/smart_turn.py`, `agent/turn/rules.py`, `agent/turn/detector.py`, `scripts/download_models.py`(→ `models/`)

### AG-T04 Provider
- 테스트 먼저(`tests/test_providers.py`): fake client로 각 Provider가 계약대로 동작
  - STT: 오디오를 보내고 `latest_text`가 갱신되고 `finalize`가 최종 텍스트를 반환
  - TTS: 16k PCM 덩어리를 순서대로 yield(24k면 변환)
  - LLM: 텍스트 이벤트, 툴 호출 이벤트(인자 dict), done. `Message` → 각 API 형식 변환(툴 결과 포함)
  - factory: 환경변수에 따라 Provider 선택
- 구현: `agent/providers/{gemini_stt,gemini_tts,gemini_llm,soniox_stt,openai_llm,factory}.py`, `scripts/bench_providers.py`(실측 → `docs/decisions/bench_providers.md`)

### AG-T06 테스트 음성 도구 + 측정·채점
- 테스트 먼저(`tests/test_eval_tools.py`):
  - 증강: 각 channel·noise·gain·packet_loss가 길이를 유지하고 결정적(seed)이며, SNR이 목표 ±1dB
  - speech_end 계산: 앞 무음 + 음성 + 쉼이 있는 합성 신호에서 마지막 음성 끝을 20ms 오차로
  - 세그먼트 조립: pause_after_ms 반영
  - 채점: 속도 표(Starter Kit과 동일), 함수 일치(이름+인자), 태그별 집계
- 구현: `eval/augment.py`, `eval/synth.py`(Gemini TTS·edge-tts, 해시 캐시), `eval/build_testset.py`(cases.jsonl → wav + meta), `eval/runner.py`(Starter Kit 측정기 확장: agent_text·tool_calls 기록), `eval/score.py`(속도·함수·LLM 심사 옵션·태그별 표 → `report.md`)

### AG-T07 세션·응답기
- 테스트 먼저(`tests/test_session.py`, `tests/test_responder.py`, `tests/test_tools_client.py`): fake STT/TTS/LLM/VAD/결정기 + `httpx.MockTransport`
  - 말 끝 확정 전에는 오디오를 보내지 않는다
  - 확정 직후 첫 오디오는 캐시된 첫 마디
  - LLM 툴 호출 → `X-Session-Id` 헤더로 호출 → 결과를 넣어 다시 LLM → 문장 단위 TTS
  - `agent.transcript`(전체 텍스트) 후 `turn.end`가 마지막 메시지
  - 확정 전 음성 재개 시 선호출 취소
  - LLM·TTS 실패 시 캐시된 사과 문구 + `turn.end`
- 구현: `agent/session.py`, `agent/responder.py`, `agent/tools_client.py`, `agent/prompts.py`, `agent/tts_cache.py`, `agent/config.py`, `agent/turn_log.py`, `agent/server.py`

### AG-T08 인자 정규화
- 테스트 먼저(`tests/test_normalize.py`):
  - "공일공 일이삼사 오육칠팔" / "영일영..." / "010-1234-5678" → "01012345678"
  - "이육일일 일오공 이삼일" → "2611150231"
  - 영문 철자 읽기 "디 이 엠 오" → "DEMO"
  - "내일", "모레", "다음 주 월요일" + now → 날짜
  - `normalize_args(tool_name, args, schema, now)`: 파라미터 이름(phone_number, order_id 등)으로 규칙 적용, 모르는 필드는 그대로
- 구현: `agent/normalize.py`

### AG-T05 시나리오 + 음성 생성
- 3.2·3.4절 형식으로 `data/testset/cases.jsonl` 150문항 이상(Mock 데이터와 일치하는 기대 함수·인자). `eval/build_testset.py`로 음성 생성, 3.3절 환경을 고르게 배분.
- 테스트 먼저(`tests/test_testset.py`): 모든 문항이 스키마를 지키고, expected_tools가 Mock API에서 matched가 되며, 범주·환경 분포가 기준을 넘는다.

### AG-T12 턴 감지 보정 (2026-11-19 사용자 요청: "네가 데이터셋 만들어서 보정할 수 있어?" → "ㄱㄱ")
- 데이터: 테스트셋과 별개의 발화 600개(연결어미·조사·숫자 중단·간투사·완결처럼 들리는 함정에서 끊기는 세그먼트 구조). 세그먼트 조립으로 쉼 위치=미완료, 마지막 끝=완료 라벨을 자동 생성. 음성 11종 이상 + 3.3절 통화 환경. 쉼 지점까지 오디오의 실제 STT 텍스트를 함께 저장. 학습/평가는 음성·문장 비중복 분할. 위치 `data/turnset/`(오디오 gitignore).
- 보정 1단계: Smart Turn 확률·end_score·침묵·발화 길이를 결합한 결정 규칙/로지스틱 + 대기 시간 재설정. 프레임 단위 오프라인 재생 시뮬레이터로 잘못 끊기 ≤1% 제약 하 평균 대기 최소화. 결과 `docs/decisions/turn_calibration.md`, 확정 파라미터는 `agent/turn/detector.py` 설정값으로.
- 보정 2단계(필요 시): Smart Turn 한국어 미세조정.
- 테스트 먼저: 라벨 생성(쉼/끝 위치) 정확성, 시뮬레이터가 결정기와 같은 판정을 내는지, 분할 비중복.

### AG-T09 연결 + 베이스라인
- `agent/server.py`가 실제 Provider·턴 감지·정규화를 엮어 실행. Starter Kit 측정기와 `eval/runner.py`로 테스트셋 전체를 측정해 `docs/decisions/baseline.md`.

### AG-T10 최적화
- 베이스라인 지표에서 병목을 골라 반복: 턴 대기 임계값, 첫 마디, 프롬프트, 모델 선택, 병렬화. 회차마다 `runs/`와 `docs/decisions/opt_log.md`에 기록.

## 8. 수동 체크리스트

| 항목 | 기대 | 결과 |
|---|---|---|
| 생성 음성 샘플 10개 청취 | 자연스러운 한국어, 통화 환경 구분 | |
| 상담원 응답 음성 10개 청취 | 응답이 질문에 맞고 끊김 없음 | |

## 9. 기존 테스트 영향
없음(신규 저장소).

## 11. 성공 지표
`docs/01_구현계획.md` 4.9절: first_sound p95 ≤ 1.3초, 잘못 끊기 ≤ 1%, 함수 일치 ≥ 95%.

## 12. 리스크
`docs/01_구현계획.md` 8절 + Gemini STT 지연이 Soniox보다 클 수 있음(실측으로 확인).

## 13. 발견 이슈 · 범위 밖
- (AG-T02) 키 인자(phone_number, order_id, product_name, option)만 정확 일치로 matched. 자유 텍스트(reason, new_address, preferred_time)는 판정 제외 → 채점기도 키 인자만 비교. X-Session-Id는 ASCII여야 한다(httpx). 데이터 기준일 2026-11-20(금).
- (AG-T07) 측정기는 바이너리 메시지 단위로 RMS를 잰다 → 캐시 PCM 앞 무음 제거 + 100ms 청크 송신. 실제 TTS 음량이 작으면 첫 청크 RMS<300 가능 → AG-T09에서 확인.
- (AG-T03) Starter Kit 데모 meta.json의 speech_end가 wav와 맞지 않는다(DEMO1 3.087s인데 음성 ~8.1s, DEMO2 5.337s인데 파일 3.65s). 운영진 질문 후보. 데모 음성 무음 구간에 RMS 20~30 잡음 있음.
- (AG-T03) Smart Turn은 여러 문장 클립에서 낮은 확률(DEMO1 0.04, DEMO3 0.03) → 한국어 테스트셋으로 보정 필요(AG-T10).
- (AG-T03→T09) 결정기 클래스명은 `RuleTurnDetector`, build_deps가 찾는 이름(make_detector 등)과 다름 → AG-T09에서 연결.
- (AG-T05a) 2단계 함수 문항(get_order_list → cancel 등) 있음 → 채점기의 순서·부분 일치 규칙 필요. "…이삼일이요"처럼 숫자 뒤 '이요' 어미 → 정규화 확인. 새 문항은 원본 끝에 추가(case_id 밀림 방지).
- **(AG-T06) Gemini TTS 무료 티어는 모델당 하루 10요청.** 상담원 TTS(문항마다 호출)와 테스트 음성 합성 모두 막힌다 → 유료 결제 연결(AG-U04) 또는 대체 TTS 필요. 테스트 음성은 edge-tts 위주로 전환.
- (AG-T06) google-genai는 TTS에 `api_version=v1beta` 필요. Gemini TTS 응답은 RIFF WAV(+메타 청크) → 청크 파싱. flash-lite-tts는 스타일 지시를 소리 내 읽음 → 스타일 요청은 제외.
- (AG-T04) 실측(p50/p95): STT live finalize→최종 0.43/0.47s(CER 0), edge TTS TTFB 0.47/0.67s, LLM gemini-3.5-flash-lite 첫 이벤트 0.78/2.96s(툴 정확도 7/7). Gemini 3.x Flash는 503·시간 초과 잦음. Gemini TTS 16k 직접 출력은 400 거부 → soxr 변환. 본 응답 p50 약 1.7~2.2s → 첫 마디 캐시가 필수.
- (AG-T09a) 스모크 20문항: first_sound p50 0.50s/p95 1.94s, **말 끝 전 발화 8/20**(문장 사이 쉼·쉼표·"어…" 뒤에서 400~600ms 확정, Smart Turn 0.66~0.99), 함수 완전 일치 7/20. STT가 끊어 읽은 숫자를 빠뜨림(통화 채널·잡음). 확정 시점 STT 부분 결과가 비는 턴 7/20. babble을 음성으로 오인. LLM이 transfer_to_agent를 말로만 함. 첫 마디와 본 응답 사이 ~0.9s 공백. `gemini-2.5-flash-lite`는 이 키에서 404.
- (AG-T09a) Starter Kit 데모 음성은 실제로 한국어. 원본 측정기는 결과를 cp949로 쓴다.
- (AG-T10b) STT 설정으로는 숫자 정확도 개선 없음(digits 74.1%, phone 63~68%, edge 음성 54%). transcribe-live는 시스템 지시 불가. batch 재인식은 1.5~3s 추가·개선 0~1건 → 기본 끔. 부분 결과 숫자 복원(지연 0)만 기본 켬. 남은 대안: 자릿수 불일치 시 LLM 되묻기(프롬프트), Soniox 비교, 사람 녹음 확인.
- (AG-T10b) T09a의 '확정 시점 텍스트 공백 7/20'은 오독(`neutral`은 규칙 중립). 실제 공백 8.4%, 모두 1.3s 이하 짧은 발화(첫 부분 결과가 음성 시작 후 ~1.1s). 선호출 재사용률이 낮은 원인은 부분 결과가 마지막 단어를 놓쳐 finalize와 불일치 → 비교 규칙 완화는 세션 쪽(T10c).
- (AG-T10a) 선호출 재사용 2/20: 침묵 200ms 부분 결과가 마지막 어절을 놓치고 `_respond_turn`이 정확 일치만 재사용 → 제안: finalize 직후 선호출 재시작, 정규화 비교, 텍스트 기반 첫 마디 선택(T10c).
- (AG-T10a) LLM 폭주 생성 1건(같은 문장 11,544회) → 300자 상한으로 차단, 원인 미상. 옵션·상품명 표기 불일치(엑스라지↔XL, 회색↔그레이, 후드티↔기모 후드티)로 not_found. history에 주문번호 없는 후속 문항 6~7개는 기대값 점검 필요.
- (AG-T10d) 프롬프트의 "상품명이 분명하지 않으면 확인" 규칙 때문에 분명한 상품명도 되묻는다(T0075) → 카탈로그 정규화가 생겼으니 규칙 완화 후보. Gemini 404가 실행마다 1~2건 간헐 발생(재시도 시 정상).
- (AG-T12) 보정셋 771발화(test 153): false cut 42.5% → 15.7%, 1.1s 이하 쉼 false cut 22.9% → 1.3%, 지연 p50 504 → 965ms·p95 1114ms, miss 7.8% → 9.2%. **false cut ≤1%는 구조적으로 불가**: 남은 오류 대부분이 1.1s 초과 쉼과 함정(완결 문장+쉼+추가 요청).
- (AG-T12) Smart Turn은 한국어 통화 음성에서 변별력 없음(AUC 0.57, 함정 쉼 평균 p 0.97 > 진짜 끝 0.88) → 채택 설정에서 미사용. 텍스트 규칙 AUC 0.83, 텍스트+음성 길이 0.894. 미세조정은 실제 평가 음성 확보 후 재검토.
- (AG-T12) rules.py 숫자 판정 버그 2건(4자리 이하를 weak_final, '공일공에 …'를 미완으로) → 결정기 층에서 보정. 잡음이 적을수록 false cut 많음(잡음이 VAD 침묵을 쪼갬). 보정셋 Gemini 음성은 429로 미포함.
- (AG-T10c) 잡음 VAD 래퍼: babble 말 끝 놓침 34% → 21%지만 보정셋 false cut 16.7% → 18.4% → 기본 끔(AD8). 측정 시간대에 flash-lite 첫 이벤트 시간 초과가 잦아 확정 → LLM 첫 이벤트 ~4s(폴백) — LLM 지연이 본 응답 공백의 주원인.
- (AG-T13) 응대(judge) round1 60.4 < 단순 조합 64.8: round1 말 끝 전 발화 16건이 응대 0점, 번호 자릿수 되묻기 16건(단순 0건), 측정 당시 폴백 모델 의존. 말 끝 전 발화를 빼면 66.0 vs 67.7. → 되묻기를 조건부(not_found 등)로 완화 검토. judge는 인사 반복·공백을 감점하지 않음. 서버 로그는 cp949. run_baseline.sh의 kill이 uv 자식 python을 남길 수 있음.
- (AG-T08) 숫자가 아닌 말 속의 수 단어도 잡힌다("이 번호요 공일공" → "2010"). "다시" 정정은 앞부분을 대체하지 않고 이어 붙인다. 고유어 수(하나, 둘) 미지원. 콜백 시간 인자 형식은 Mock API 정의와 맞춰야 한다.

## 14. 작업 로그
- 2026-11-19 | AG-T01 | DONE | git init, uv(Python 3.12) 의존성, `agent/core/types.py` 계약, CLAUDE.md, pytest 설정, 테스트 0개(골격)
- 2026-11-19 | AG-T08 | DONE | agent/normalize.py, tests/test_normalize.py, 94개 RED ModuleNotFoundError → GREEN 94 passed, normalize_args에 now 인자 추가(설계 반영), 한계는 13절
- 2026-11-19 | AG-T02 | DONE | mock_api/{server.py,data/tools.json,mock_data.json,README.md}, cases/manual.md, tests/test_mock_api.py, RED ModuleNotFoundError → GREEN 54 passed, status 5종(ok/not_allowed/no_refund/not_found/invalid_arguments)
- 2026-11-19 | AG-T07 | DONE | agent/{config,tools_client,tts_cache,prompts,responder,session,turn_log,server}.py, tests 3개, RED ModuleNotFoundError → GREEN 40 passed, 전체 428 passed, VAD·결정기는 팩토리 주입(계약 주석 반영)
- 2026-11-19 | AG-T03 | DONE | agent/audio.py, agent/turn/{vad,smart_turn,rules,detector}.py, scripts/download_models.py, tests 2개, RED ImportError → GREEN 130 passed, VAD 0.1~0.2ms/청크, Smart Turn 23~31ms(4스레드), 잡음성 짧은 발화(<250ms, 텍스트 없음)는 상한에서도 끝 아님, 단답은 400ms
- 2026-11-19 | AG-T05a | DONE | data/testset/{scenarios_src.yaml,cases.jsonl,README.md}, scripts/gen_scenarios.py, tests/test_testset.py, RED 21 errors → GREEN 21 passed, 172문항(범주 7종·음성 11종·채널 3종 균형)
- 2026-11-19 | AG-T06 | DONE | eval/{augment,synth,build_testset,runner,score}.py, tests/test_eval_tools.py, RED ModuleNotFoundError → GREEN 120 passed, 전체 489 passed, 실제 스모크(edge·Gemini 합성, 증강 4종, runner+score), 레벨 정규화(RMS 3000) 추가, .gitignore에 캐시 경로 추가
- 2026-11-19 | AG-T04 | DONE | agent/providers/{_gemini,gemini_stt,gemini_tts,edge_tts,gemini_llm,soniox_stt,openai_llm,factory}.py, scripts/bench_providers.py, docs/decisions/bench_providers.md, tests/test_providers.py, RED 29 실패 → GREEN 35 passed, live 스모크 통과(edge TTS→live STT→flash-lite 툴 호출), 기본 LLM gemini-3.5-flash-lite·TTS edge
- 2026-11-19 | AG-T09a | DONE | agent/{server,tts_cache,config}.py, agent/providers/{fallback_llm,factory}.py, scripts/build_smoke_set.py, data/testset/smoke/, tests/test_server_wiring.py, docs/decisions/baseline_smoke.md, RED 24 실패 → GREEN 27 passed, 회귀 521 passed, E2E 스모크·Starter Kit 측정기 호환 확인, 개선 후보는 baseline_smoke.md
- 2026-11-19 | AG-T05b | DONE | data/testset/audio/ 172 wav + meta.json(gitignore, `eval.build_testset`로 재생성), 1차 161건 + 429 실패 11건 순차 재시도로 완료. Tier 1에서도 TTS 동시 3개는 분당 한도에 걸림 → 동시 1~2 권장
- 2026-11-19 | AG-T09b | DONE | docs/decisions/baseline.md(+report 사본), runs/run_baseline.sh, 속도 67.4·말 끝 전 발화 55/172·첫 소리 p50 0.51s·함수 56.4(완전 일치 0.52)·에러 0, 추정(응대 제외) 63.0
- 2026-11-19 | AG-T10b | DONE | eval/stt_eval.py, agent/providers/{stt_recheck,gemini_stt,factory}.py, tests 2개, docs/decisions/stt_tuning.md, RED ImportError → GREEN 34 passed, 회귀 610 passed, 숫자 정확도 74.1% 유지(설정 실험 무효), 부분 결과 숫자 복원 기본 켬·batch 재인식 기본 끔, live 세션 약 950회
- 2026-11-19 | AG-T10a | DONE | eval/llm_eval.py, agent/{prompts,responder,config}.py, tests/test_llm_eval.py·test_prompts.py, docs/decisions/llm_tuning.md, RED ImportError → GREEN 57 passed, 회귀 612 passed, 오프라인 함수 완전 일치 0.70~0.72 → 0.87~0.90, judge 69.5 → 73.0, 호칭 반복 22 → 0, 번호 인자 교정·자릿수 검사·인계 안전망·필러 150ms·응답 300자 상한, LLM 약 2,750회+judge 344회
- 2026-11-19 | AG-T10d | DONE | agent/catalog.py, agent/normalize.py(catalog 인자 추가), tests/test_catalog.py, docs/decisions/catalog_normalize.md, RED ImportError → GREEN 73 passed, 회귀 247 passed·전체 749 passed, 상품 관련 21문항 함수 완전 일치 0.71 → 0.79
- 2026-11-19 | AG-T12 | DONE | scripts/gen_turnset.py, eval/{turn_sim,turn_calibrate}.py, agent/turn/detector.py(TableTurnDetector·CALIBRATED_CONFIG·make_detector 추가), data/turnset/(라벨·보정 결과, 오디오 제외), tests 2개, docs/decisions/turn_calibration.md, RED ImportError → GREEN 194 passed, 전체 759 passed, Gemini STT 약 700회·edge 700회
- 2026-11-19 | AG-T10c | DONE | agent/{session,config,tts_cache}.py, agent/turn/vad.py(NoiseGate·NoiseRobustSileroVAD·make_vad), eval/vad_noise_eval.py, tests/test_session_precall.py·test_vad_noise.py, docs/decisions/session_tuning.md, RED ImportError → GREEN 28 passed, 선호출 재사용 2/20 → 24/30, 텍스트 기반 첫 마디, 잡음 래퍼는 AD8로 기본 끔(새 테스트의 기본값 단언을 결정에 맞게 수정)
- 2026-11-19 | — | 측정 | round1(40751d7): 속도 67.4 → 85.6, 말 끝 전 발화 55 → 23, 함수 56.4 → 74.4(완전 일치 0.52 → 0.72), 추정(응대 제외) 63.0 → 81.1, turn.end p50 3.07 → 5.99s(flash-lite 시간 초과로 폴백 다발). docs/decisions/round1.md
- 2026-11-19 | AG-T13 | DONE | bench/{naive_server.py,run_naive.sh}, tests/test_naive_server.py, docs/decisions/naive_benchmark.md, 6 passed·전체 783 passed, 단순 조합 N0 추정 총점 53.1 vs round1 70.7(속도 25.2 vs 85.6, 함수 65.7 vs 74.4, 응대 64.8 vs 60.4), N1(문장 TTS만) 효과 ~0.2s
