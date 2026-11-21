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
- (AG-T29, AD11) 실제 운영 STT는 **BAZE 경유 Soniox `stt-rt-v6`**(`.env`의 `STT_PROVIDER=soniox`, 직접 키는 잔액 소진으로 주석). LLM은 AD17로 gpt-5.4-mini 고정. 엔드포인팅은 기존 결정기 유지, Soniox `<end>`는 끔(AD12, `docs/decisions/soniox_baze.md`).
- LLM: **gpt-5.4-mini 고정**(AD17, 대회 규정). `LLM_PROVIDER=openai`(또는 auto + OpenAI/BAZE 키). OpenAI 기본이면 Gemini 폴백을 붙이지 않고 hedge·race·fallback은 같은 모델 사본으로만 한다(AG-T44). `agent.server.serve`는 체인에 `LLM_ALLOWED_MODELS`(기본 `gpt-5.4-mini`, `*`면 끔) 밖 모델이 있으면 시작하지 않는다. Gemini 컨텍스트 캐시는 OpenAI 경로에서 쓰이지 않는다.
- TTS: `gemini-3.8-flash-tts`(대안 flash-lite-tts). 첫 마디·필러·고정 문구는 캐시.
- BAZE 게이트웨이(AG-T16, AD9): 직접 키(`SONIOX_API_KEY`·`OPENAI_API_KEY`·`GEMINI_API_KEY`)가 없고 `BAZE_API_KEY`가 있으면 그 서비스를 `BAZE_BASE_URL`(기본 `https://api.baze.co/v1/gateway`)로 보낸다. Soniox `wss://…/soniox/transcribe-websocket`, OpenAI Responses `…/`, Gemini `…/gemini`(v1beta). 게이트웨이에 없는 모델: `gemini-3.5-transcribe(-live)`, `gemini-2.5-flash-lite` → BAZE 모드 STT는 Soniox만(폴백 없음), LLM 기본 폴백은 `gemini-3.1-flash-lite`만. TTS는 `DIRECTON_TTS_PROVIDER=baze`일 때 `POST …/audio/speech/`(24k PCM 한 번에 응답 → 16k 변환), 기본은 edge 유지.

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
| AD2 | 개발용 LLM | A: Gemini Flash 계열 / B: OpenAI 키 대기 | **A.** 같은 이유. 실측으로 모델 선택 | **A** (2026-11-19 사용자 확정) → **AD17로 폐기**(2026-11-21, 대회 규정 LLM gpt-5.4-mini) |
| AD3 | 테스트 음성 합성 | A: Gemini TTS 여러 음성 + edge-tts(무료, 키 불필요) / B: Gemini만 | **A.** 상담원 음성(Gemini)과 다른 엔진이 섞여야 과적합을 피한다. 가상 문장만 보낸다 | **A** (2026-11-19 사용자 확정) |
| AD4 | 가상 쇼핑몰 함수 | A: 공식 미공개이므로 쇼핑몰 함수 14개를 가정해 구현 / B: 데모 2개만 | **A.** 공식 공개 시 데이터만 교체 | **A** (2026-11-19 사용자 확정) |
| AD6 | 상담원 TTS(Gemini TTS 무료 하루 10회) | A: Gemini 결제 연결 / B: 개발 중 edge-tts 대체 / C: 다른 TTS 키 | **A.** 비용 약 $5 | **B** (2026-11-19 사용자 확정) |
| AD7 | 테스트 음성 화자 | A: edge 3종+Heami에 피치·속도 변형 / B: 결제 연결 후 Gemini 음성 사용 | **A** | **B** (2026-11-19 사용자 확정) — T05b는 AG-U04 이후. T12 보정셋은 edge+피치로 먼저 만들고 Gemini 음성은 결제 후 추가 |
| AD8 | 잡음 VAD 래퍼 기본값 | A: 끔 / B: 켬 / C: 결정기 재보정 후 | **A.** 잘못 끊기(0점) 손해가 말 끝 놓침(80~90점)보다 큼 | **A** (2026-11-19 사용자 확정) — `DIRECTON_VAD_NOISE_GATE=1`로 켬 |
| AD9 | BAZE 게이트웨이 범위 (2026-11-19 사용자 요청) | A: Claude Code만 / B: 프로젝트 provider만 / C: 둘 다. TTS: 기본 edge 유지+baze 선택 / baze 기본 / 안 함. STT 폴백: Soniox만 / gemini-3.5-flash 일괄 | — | **B**, TTS는 edge 기본+`baze` 선택, STT 폴백 없음 (2026-11-19 사용자 확정) |
| AD10 | BAZE 크레딧 소진 → 서비스 선택 | — | — | **Gemini 직접 키로 복귀** (2026-11-19 사용자 지시). `.env`의 `BAZE_API_KEY` 주석 처리(값 보존). 다시 켜려면 주석 해제 → **AD17로 폐기**(2026-11-21, 대회 규정 LLM gpt-5.4-mini) |
| AD11 | STT 기본 (2026-11-21 사용자 요청: "BAZE에 SONIOX가 있어. 그걸로 한번 도입해봐") | A: BAZE 경유 Soniox / B: Gemini live 유지 | **A.** 172문항 숫자 정규화 CER 0.065→0.040, 숫자 74.1→79.6%, finalize p50 397→238ms | **A** (2026-11-21 사용자 지시) |
| AD12 | 엔드포인팅 (2026-11-21 사용자 요청: "엔드포인팅도 기존 방안과 SONIOX의 방안을 비교해줘") | A: 기존 결정기 유지(Soniox `<end>` 끔) / B: Soniox `<end>`로 교체 | **A.** Soniox는 지연 p50 약 0.5초 빠르지만 말 끝 전 확정 turnset 10.1→46.5%, 172문항 7.0→27.3%(0점 조건). AG-T30 규칙/Jev 게이트를 붙여도 22.8~25.4% / 11.0~15.1%(`soniox_gate.md`) | |
| AD13 | 엔드포인팅 결합 (AG-T32) | A: 기존 결정기만 / B: 기존 + Jev 늘리기(최근 응답 규칙, lo 0.3~0.5, 스위치) / C: Soniox + Jev 게이트 | **B.** 5개 셋 실제 끊음 62 → 51(lo 0.5) / 55(lo 0.3), miss +2~4, p95 거의 그대로. 늘리기만이라 Jev 실패 시 기존 그대로. 단 외부 API 의존 | **B** (2026-11-21 사용자 확정) |
| AD14 | 공식 자료 위치 (AG-T36, 2026-11-21) | A: `tools.json`만 저장소(`mock_api/data/`)에 복사, `mock_db.json`은 공식 폴더(gitignore)에서 `--db`로 읽고 테스트는 합성 소형 DB / B: 둘 다 복사해 커밋 / C: 둘 다 공식 폴더 경로만 참조 | **A.** `tools.json`은 개인정보 없는 계약이라 테스트·프롬프트가 기대야 한다. `mock_db.json`은 실제 상담 기반이라 반출 금지(DATA.md) → 저장소 밖에 둔다. 제출 묶음에는 `start_mock.sh`가 인자로 받는다 || **A** (2026-11-21 사용자 확정) |
| AD15 | 옛 함수 14개 체계 (AG-T36~T38) | A: 완전 교체 — 옛 함수·mock 데이터·관련 단언 삭제·갱신 / B: 스위치로 둘 다 유지 | **A.** 공식 규격이 확정됐고 시험도 같은 8개 함수. 두 체계 유지는 프롬프트·교정·테스트를 두 벌로 만든다. 옛 함수명이 걸린 테스트 약 25개 단언을 바꾼다(9절에 기록) || **A** (2026-11-21 사용자 확정) |
| AD16 | 브랜드 자료 교체 범위 | A: 이번 묶음에 포함(AG-T39: `cases/manual.md` → `data/brand/` 시나리오·정보 96개·말투) / B: 함수만 먼저, 브랜드는 다음 | **A.** 응대 점수(50%)는 하루온 시나리오·정보로 채점된다. 함수 규칙과 매뉴얼이 같은 프롬프트에 있어 따로 하면 프롬프트를 두 번 고친다 || **A** (2026-11-21 사용자 확정) |
| AD17 | 상담원 LLM (2026-11-21 사용자: "gpt-5.4-mini가 대회에서 규정된 제한이야.") | A: gpt-5.4-mini 단독, 다른 모델로 새는 경로 차단 / B: Gemini 유지 | **A.** 대회 규정 | **A** (2026-11-21 사용자 확정) — hedge는 같은 모델 사본(`LLM_SAME_MODEL_HEDGE=1`), 서버는 `LLM_ALLOWED_MODELS`(기본 `gpt-5.4-mini`) 밖 모델이 체인에 있으면 시작하지 않는다(AG-T44) |
| AD18 | 상담원 LLM reasoning effort (AG-T50) (2026-11-21 사용자 요청: "LLM 엔진 추론 레벨을 미디움으로 올려") | A: none 유지 / B: medium + effort별 타이밍·reasoning item 되돌리기 | **B.** 사용자 지시. 속도 점수 위험(조회 뒤 응답에 두 번째 LLM 호출이 들어간다)은 실측으로 확인하고, `OPENAI_REASONING_EFFORT=none`이면 예전 동작 그대로 | **B** (2026-11-21 사용자 지시) — 타이밍 기본값은 잠정(실측 전) |
| AD19 | 상담원 LLM reasoning effort 기본값 재결정 (AG-T51) (2026-11-21 사용자: "1번만 해" — low 기본값 변경 확정) | A: medium 유지(AD18) / B: low / C: none | **B.** 4차 측정(docs/14 7절)에서 low·medium 참고 총점이 같음(86.99 vs 87.05), 응대 둘 다 82.9, 조회 뒤 답 3.5초 초과 low 1건 vs medium 9건. 실제 BAZE 꼬리 지연에 여유가 크고 추론 토큰·타임아웃 위험이 작다 | **B** (2026-11-21 사용자 확정) — AD18의 기본값(medium)을 대체. `OPENAI_REASONING_EFFORT`로 되돌릴 수 있다 |
| AD5 | 저장소 | A: 이 폴더를 git 저장소로 / B: git 없이 | **A.** Task별 커밋이 워크플로 전제 | **A** (2026-11-19 사용자 확정) |

## 5. 사용자가 직접 확인할 것

| ID | 할 일 | 왜 |
|---|---|---|
| AG-U01 | Soniox·OpenAI 키 발급, `.env`에 추가 | 계획서 기본 스택으로 전환 — 2026-11-21 Soniox 직접 키 추가됐으나 잔액 소진(402). 충전 전까지 BAZE 경유(AG-T29). OpenAI 키 없음 |
| AG-U02 | 운영진 질문 발송(`00_대회규격_분석.md` 6절) | 허용 API·실행 위치·채점 방식 |
| AG-U03 | 생성된 한국어 테스트 음성 샘플 청취 | 합성 품질 확인 |
| AG-U05 | 체험 데모 실제 마이크로 확인(`powershell -ExecutionPolicy Bypass -File demo
un_demo.ps1`, 가이드 `docs/03_사용자_테스트_가이드.md`) — 마이크 권한, 에코, 음질, Ctrl+C 종료, 다중 턴 | 실제 마이크·스피커 경로는 미검증 |
| AG-U04 | Gemini API 결제 연결(Tier 1) 또는 대체 TTS 키 | 무료 티어 TTS 하루 10회 — ✅ 2026-11-19 프로젝트 158777464759 결제 연결 확인 |
| AG-U06 | AI Hub [민원(콜센터) 질의-응답](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=98) 신청·다운로드(본인 인증, 내국인만) → `data/aihub_98/raw/`에 압축 해제 | AG-T27 실제 통화 음성 평가의 입력. 재배포 금지 데이터라 git에 넣지 않는다 |

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
| AG-T14 | 사용자 체험 데모: 브라우저 마이크 데모 페이지 + 데모 서버 + 사용자 테스트 가이드 (2026-11-19 사용자 요청) | AG-T10c | DONE | 2026-11-19 |
| AG-T16 | BAZE 게이트웨이 연동: BAZE_API_KEY로 Soniox·OpenAI·Gemini 라우팅 + BAZE TTS 어댑터 (2026-11-19 사용자 요청) | AG-T04, AD9 | DONE | 2026-11-19 |
| AG-T14b | 체험 데모 이전 대화(history) 주입 UI + 예시 프리셋 (2026-11-19 사용자 요청) | AG-T14 | DONE | 2026-11-19 |
| AG-T15a | 2차: 번호 되묻기 조건부 완화 + 응대 품질(judge) 개선 | AG-T13 | DONE | 2026-11-19 |
| AG-T15b | 2차: 말 끝 전 응답 감소(숫자 묶음·자기 정정·긴 쉼 규칙) | AG-T12, AG-T13 | DONE | 2026-11-20 |
| AG-T15c | 2차: LLM 동시 요청(hedging) + 컨텍스트 캐시 | AG-T13 | DONE | 2026-11-19 |
| AG-T15d | 2차 통합 172문항 측정(round2) + judge | AG-T15a~c, AG-T14, AG-T21, AG-T25 — 첫 마디 정책 켬/끔·Jev 인계 켬/끔 A/B 포함(2026-11-20 사용자: 첫 마디 정책은 측정 후 결정) | DONE | 2026-11-20 |
| AG-T17 | 당일 교체: Mock `--data/--tools`, 매뉴얼·브랜드·말투 파일 교체, 번호 자릿수 데이터 추론, 시험 음성 저장 끄기, 당일 체크리스트 (2026-11-19 사용자 요청: 공식 사이트 기반 작업 착수) | AG-T09 | DONE | 2026-11-20 |
| AG-T18 | 외부 접속·안정성: 터널 스크립트, 동시 4세션 연속 부하, edge-tts 장애 시 TTS 대체, 측정 중 예외 격리 | AG-T09 | DONE | 2026-11-20 |
| AG-T19 | 바지인(중간 개입): 상담원 발화 중 고객 음성 → 즉시 중단 → 이어 듣고 재응답 | AG-T10c | DONE | 2026-11-20 |
| AG-T20 | 첫 마디 재설계: 같은 채움말 반복 금지, 답 준비 시 첫 마디 생략, 상황별 다양한 첫 마디 | AG-T10c | DONE | 2026-11-20 |
| AG-T21 | 조회 뒤 답까지 시간 단축(after_tool_sound) + 측정 지표 | AG-T15c | DONE | 2026-11-20 |
| AG-T22 | 건강기능식품 연습판: 매뉴얼·상품·결제·몸 불편 인계를 가정한 Mock 데이터 + 시나리오 40개 + 음성 | AG-T17 계약 | DONE | 2026-11-20 |
| AG-T23 | 기획서·발표 초안 + 1장 검토 보고서(결정 모델 선준비·빠른 서빙) | — | DONE | 2026-11-20 |
| AG-T24 | Jev 결정 모델: 말 끝 판단 Noul(오프라인 평가 + 기본 끔 스위치), 사람 인계 판단 Noul(몸 불편·직접 요청·욕설·반복 실패·법적·예외, 기본 끔 스위치), 여유 있으면 의도 Choice (2026-11-20 사용자 요청: "Jev 추가해보기로 결정했어", "사람 인계 여부도 Jev로 판단하도록 해") | AG-T15b | DONE | 2026-11-20 |
| AG-T25 | 통합 배선: Jev 인계·말 끝 문맥을 세션·서버에 연결(스위치), 첫 마디·바지인 설정 필드(config), 워밍업에 첫 마디 문구, 끊긴 답 프롬프트 규칙 | AG-T19, AG-T20, AG-T24 | DONE | 2026-11-20 |
| AG-T26 | 최종 결과 보고서 HTML: 모든 측정 결과를 각 선택의 trade-off 중심으로 정리 (2026-11-20 사용자 요청: "모든 측정 끝나고 보기좋은 HTML 페이지로 결과보고서 정리해줘. 각 선택의 Trade-off를 중심으로") | AG-T15d | DONE | 2026-11-20 |
| AG-T27 | 실제 통화 평가셋: AI Hub 98번(다산콜센터 녹취) 고객 발화 → 16k 평가셋 변환 + STT·턴 감지 실음성 측정, 합성 테스트셋과 격차 비교 (2026-11-20 사용자 요청: "응 98번으로 Task 적어줘") | AG-U06, AG-T06, AG-T12 | TODO | |
| AG-T28 | 번호 확인 되묻기: 주문번호가 막히거나(자릿수, 교정 실패) not_found면 발신 고객 주문 중 가장 가까운 1~2개를 끝자리·상품명으로 확인 질문, 기본 끔 스위치 `DIRECTON_CONFIRM_ORDER_ID` (2026-11-20 사용자 요청: "선택지 중 가장 가까운 것을 하나 골라서 0000 맞으실까요?하고 되묻는 식으로", "기본 끔 스위치로 구현해줘") | AG-T15a | DONE | 2026-11-20 |
| AG-T29 | Soniox STT 도입(BAZE 경유): 172문항 STT 비교 + 엔드포인트(`<end>`) vs 기존 결정기 비교 + `.env` 라우팅 (2026-11-21 사용자 요청) | AG-T04, AG-T16, AD11 | DONE | 2026-11-21 |
| AG-T30 | Soniox `<end>` 게이트: `<end>`마다 규칙/Jev로 진짜 말 끝 판정(VAD 침묵 hold·상한), 기존 / Soniox+규칙 / Soniox+Jev 비교 (2026-11-21 사용자 요청) | AG-T29, AG-T24 | DONE | 2026-11-21 |
| AG-T31 | onedam_cs_testset(실녹음 100문항 × wideband·phone·babble) 엔드포인팅 평가: 기존(Soniox 텍스트 재생) / Soniox+규칙 / Soniox+Jev (2026-11-21 사용자 요청) | AG-T30 | DONE | 2026-11-21 |
| AG-T32 | 기존 결정기 + Jev 결합 비교(침묵 200ms 늘리기·줄이기, `<end>` 트리거, 텍스트 일치 규칙) — 5개 셋 Soniox 재생 (2026-11-21 사용자 요청) | AG-T31, AG-T24 | DONE | 2026-11-21 |
| AG-T33 | `JevTurnDetector` 최근 응답 규칙(텍스트 바뀌어도 이번 침묵의 최근 p, 요청 상한 6) + 늘리기 lo 결정 + 다른 데이터 확인 | AG-T32, AD13 | DONE | 2026-11-21 |
| AG-T34 | 말하는 중 Jev 선호출(주기 tick `meaning_will_change` ≤ τ면 선호출, 확정 때 Jev `same_reply`로 정합성 판정) vs 기존 침묵 200ms 선호출 — onedam 재생 비교 (2026-11-21 사용자 요청) | AG-T31, AG-T10c | DONE | 2026-11-21 |
| AG-T34 | 공식 연습 데이터(directon-2026-voice-cs 80문항) A·B·C 엔드포인팅 평가 (2026-11-21 사용자 요청) | AG-T32 | DONE | 2026-11-21 |
| AG-T35 | 규칙 보정: 채팅식 키워드 말 끝 "문의"·"~방법은"을 조사로 보지 않기(공식 연습 miss 3) | AG-T34 | DONE | 2026-11-21 |
| AG-T36 | 공식 규격 mock 서버: `mock_api/server.py`를 공식 `tools.json`(8개) + `--db` mock_db 형식으로 교체(조회 표기 무시 일치·not_found, 접수 required→ok/invalid, 검색 최근접, 미정의 404), 루트 `start_mock.sh` (2026-11-21 사용자 요청: "모든 mock 서버와 함수 등을 다 directon-2026-voice-cs를 기준으로 먼저 맞춰줘") | AD14, AD15 | DONE | 2026-11-21 |
| AG-T37 | history의 `tool_call`·`tool_result`를 프롬프트에 반영(지금은 빈 "고객:" 줄) + 이전 결과 재사용 규칙 | AG-T36 | DONE | 2026-11-21 |
| AG-T38 | agent 함수 체계 교체(코드): 자동 연결 안전망·`transfer_to_agent` 의존 삭제, 번호 교정·되묻기를 `get_order_info(buyer_cellphone)` + 한글 결과 키로, 공식 번호 형식(주문 15·16자리, 전화 11자리)으로 자릿수 검사, `order_number` 정규화·교정, 카탈로그 기본 파일 없음(mock 데이터 안 읽음), 템플릿 한글 키, 인계 판정은 함수 없는 안내. 프롬프트 문구는 AG-T39 | AG-T36, AD15 | DONE | 2026-11-21 |
| AG-T39 | 프롬프트 교체: 고정부 함수 규칙(8개 함수, 인계는 말로 + 전화 문의 방법)·caller 줄·영업시간(평일 10~17시, 점심 12~13시)을 `data/brand/`(scenarios.md, information.json, style.json) 기준으로, 매뉴얼을 브랜드 자료로 | AG-T38, AD16 | DONE | 2026-11-21 |
| AG-T40 | 측정 정렬: 공식 측정기(`directon-2026-voice-cs/eval/run_eval.py`·`score.py`, 최종 평가와 같은 도구)로 직접 측정 — 우리 `eval.runner` 이식 대신(범위 조정 2026-11-21). 첫 측정: BAZE 경유 Soniox STT + gpt-5.4-mini (2026-11-21 사용자 요청: "다 하고 directon-2026-voice-cs 으로 BAZE API에 아래 모델들로 벤치마크 돌려서 정량점수 측정해줘") | AG-T39 | DONE | 2026-11-21 |
| AG-T41 | 발신번호 교정 확장: 자리 수는 맞지만 발신번호와 거의 같은(공통 부분열 기준 기존 `_close_enough`) 전화번호 인자는 발신번호로 교정(STT 숫자 오인식, P0119-04·P0133-03·P0194-03) | AG-T40 | DONE | 2026-11-21 |
| AG-T42 | 막힌 함수 호출(번호 자리 수 틀림·필수 번호 없음)의 되묻기 다듬기: 막힐 회차엔 채움말을 틀지 않고, 되묻기 안내는 짧게 사과 + 다시 불러 달라는 한 문장만(자리 수 규칙·정책 안내 덧붙이지 않음) | AG-T40 | DONE | 2026-11-21 |
| AG-T43 | 번호 후보 컨텍스트 + 확신도별 처리: 이미 아는 번호(발신번호, 이전 대화의 번호·조회 결과 주문번호, 이번 턴에 받은 고객 주문)를 후보로 프롬프트에 넣고, 들은 번호가 후보와 강하게 맞으면 그 후보로 바꿔 호출, 약하게 맞으면 그 후보 끝자리로 되묻기(needs_confirmation) (2026-11-21 사용자 요청: "숫자 인식률 개선을 위해 이미 아는 정보 기반으로 가능한 응답 경우의 수를 LLM에 컨텍스트로 입력해서 인식 결과가 좀 달라도 강한 확신 -> 그냥 가장 가까운 경우의 수로 swap, 약한 확신 -> 해당 경우의 수로 되묻기 방식을 구현하고 테스트해줘") | AG-T41, AG-T42 | DONE | 2026-11-21 |
| AG-T44 | LLM 규정 고정: OpenAI 기본이면 Gemini 폴백 제거·같은 모델 사본 hedge(`LLM_SAME_MODEL_HEDGE`), `check_llm_policy`·`LLM_ALLOWED_MODELS`로 서버 시작 검사 (2026-11-21 사용자 요청: "gpt-5.4-mini가 대회에서 규정된 제한이야." → "응 진행해줘") | AD17 | DONE | 2026-11-21 |
| AG-T45 | 함수 호출 안전장치(`agent/tool_guard.py`): 전체 반품의 return_order_item → return_order 변환(조회한 주문 기준, 일부 반품 표현이면 그대로), 접수 함수 없이 처리 완료를 말하면 그 문장을 내보내지 않고 한 번 다시 생성, 함수 호출 문자열·JSON은 TTS 전에 거르고 알려진 호출이면 실제 호출로 (2026-11-21 사용자 요청, `docs/14_공식측정_결과보고서.md` 6절 1~3) | AG-T42 | DONE | 2026-11-21 |
| AG-T46 | 숫자 2차 인식(Soniox) 오프라인 비교: 번호 문항 음성을 전체 재인식·숫자 문맥·번호 구간 자르기·무음/정규화·0.8/0.65배속으로 다시 받아 적어 정답 번호 일치율 비교(`eval/number_recheck_eval.py`). 효과 있으면 `DigitRecheckSTT` 재인식 함수를 Soniox로 바꿔 연결 (2026-11-21 사용자 요청: "Soniox 2차 인식부터 해서 25개로 측정해줘") | AG-T43 | DONE | 2026-11-21 |
| AG-T48 | 근거 없는 사실 금지: 프롬프트 '사실 근거 규칙'(숫자·기간·섭취량·정책은 함수 결과·브랜드 정보에 있는 것만, 조회 없이 조회 결과 금지, 며칠분 계산·할인 적용 순서·다른 제품 정보 옮기기 금지, 섭취 문구 그대로)과 '결제 수단별 환불 정리'(선불금·페이/카드/무통장·가상계좌/계좌이체/휴대폰, 환불보장제), 조회 없이 '입금 확인이 안 된다'·'조회가 어렵다'를 말하면 AG-T45 SpeechGuard로 붙잡아 한 번 다시 생성 (2026-11-21 사용자 요청, 공식 측정 main 17380d6 즉시 0점 5건 중 4건이 지어낸 사실) | AG-T45 | DONE | 2026-11-21 |
| AG-T49 | 번호를 받고도 조회 없이 번호 전체를 다시 묻는 문제(공식 측정 runs/official_t45 약 15건): 막힌 조회는 발신번호로 대신 조회해 결과를 LLM에 주고(P0173-02), 같은 회차에 통과한 같은 조회가 있으면 막힌 호출은 건너뜀, get_order_info의 틀린 선택 인자는 다른 번호가 맞으면 버림, 번호 인자가 모두 빈 조회는 발신번호로 채우거나 부르지 않음(이미 물었으면 턴 끝), 들린 번호가 어떤 형식에도 맞지 않으면 첫 마디 "네, 확인해 드릴게요." 대신 기본 첫 마디, 이전 대화에서 결과가 없던 번호는 다시 묻지 않게 요약 줄 (2026-11-21 사용자 요청) | AG-T43, AG-T45 | DONE | 2026-11-21 |
| AG-T50 | LLM reasoning effort 설정화(기본 medium, `OPENAI_REASONING_EFFORT`) + effort별 hedge·첫 이벤트 기본값 + store=False 툴 라운드 사이 encrypted reasoning item 되돌리기(`OPENAI_REASONING_INCLUDE`) + reasoning 스트림 이벤트 무시 (2026-11-21 사용자 요청: "LLM 엔진 추론 레벨을 미디움으로 올려") | AD17, AD18 | DONE | 2026-11-21 |
| AG-T51 | reasoning effort 기본값 medium → low(`DEFAULT_REASONING_EFFORT`), 4차 low 측정 기록 (2026-11-21 사용자 요청: "기록하고 low도 돌려봐" → "1번만 해. 이제 평가할거야") | AG-T50, AD19 | DONE | 2026-11-21 |
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

### AG-T16 BAZE 게이트웨이 연동 (2026-11-19 사용자 요청: "아래 gateway를 통해 원안의 AI 모델들을 사용하고자 해")
- 테스트 먼저(`tests/test_providers.py`): BAZE 키만 있을 때 factory가 Soniox(게이트웨이 URL·BAZE 키)·OpenAI(base_url)·Gemini(base_url) 선택, 직접 키가 있으면 직접 키 우선, BAZE 모드 기본 폴백 체인에서 2.5-flash-lite 제외, `BazeTTS`가 `httpx.MockTransport`로 요청 형식(model·input·voice·instructions·Bearer)을 지키고 24k → 16k PCM을 순서대로 yield, 빈 텍스트 무출력, HTTP 오류 시 예외.
- 구현: `agent/providers/{factory,baze_tts,_gemini,openai_llm}.py`, `openai` 의존성 추가(`OpenAILLM`이 지연 import하지만 의존성에 없었다).

### AG-T17~T23 공식 사이트 기반 작업 (2026-11-19 사용자 요청: "공식사이트 기반 작업들 착수해줘", 근거 `docs/04_목표_점검.md`)
공통 계약(병렬 작업 전 확정):
- Mock: `python -m mock_api.server --data <mock_data.json> --tools <tools.json>`(둘 다 생략 시 기존 `mock_api/data/`). 환경변수 `DIRECTON_MOCK_DATA`, `DIRECTON_MOCK_TOOLS`도 같은 뜻.
- 상담원 매뉴얼·브랜드·말투: `DIRECTON_MANUAL`(기본 `cases/manual.md`), `DIRECTON_BRAND`, `DIRECTON_STYLE`(선택, 없으면 생략). 카탈로그는 기존 `DIRECTON_CATALOG_PATH`.
- 시험 음성 저장 금지: `DIRECTON_NO_AUDIO_LOG=1`이면 오디오·고객 발화 원문을 디스크에 쓰지 않는다(측정 모드 기본 켬).
- after_tool 지표: 턴 로그 `after_tool_ms` = 마지막 함수 결과 수신 → 그 뒤 첫 유성 PCM 전송.

- **AG-T17** `mock_api/server.py`, `agent/config.py`, `agent/prompts.py`, `agent/normalize.py`, `agent/catalog.py`, `agent/turn_log.py`, `docs/05_당일_체크리스트.md`. 테스트 먼저: `--data/--tools` 로드, 매뉴얼·브랜드·말투 파일이 시스템 프롬프트에 들어감, 4자리 주문번호 데이터에서 자릿수 규칙이 4자리로 바뀜(고정 10자리 제거), 저장 끄기 시 파일 없음.
- **AG-T18** `agent/server.py`, `agent/providers/factory.py`, `agent/providers/edge_tts.py`, 새 `agent/providers/fallback_tts.py`, `scripts/tunnel.*`, 새 `eval/stress.py`. 테스트 먼저: TTS 1차 실패 시 대체 TTS로 이어 말함, 세션 하나의 예외가 다른 세션·서버를 죽이지 않음, fake로 4세션 동시 처리.
- **AG-T19·T20** `agent/session.py`, `agent/tts_cache.py`, `demo/*`. 테스트 먼저: 재생 중 고객 음성 300ms 이상 → 남은 TTS 중단·`turn.end` 규칙 유지·이어 들은 말로 재응답, 측정기 경로(끼어들지 않음) 결과 불변. 연속 턴에서 같은 첫 마디 반복 없음, 선호출 답이 준비됐으면 첫 마디 없이 본 응답.
- **AG-T21** `agent/responder.py`, `eval/runner.py`, `eval/score.py`. 테스트 먼저: 함수 결과 뒤 첫 문장이 끝나는 즉시 TTS로 넘어감, `after_tool_ms` 기록·집계, responder의 10자리 가정 제거(알려진 주문번호 길이 기준).
- **AG-T22** 새 `data/practice_health/`(mock_data.json, tools.json, manual.md, cases.jsonl, README), `eval/build_testset.py`(경로 인자). 테스트 먼저: 모든 문항 기대 함수가 연습 Mock에서 matched, 업무 7종·몸 불편 인계 포함.
- **AG-T23** `docs/06_기획서_초안.md`, `docs/07_발표_초안.md`, `docs/08_검토보고서_선준비.md`. 코드 변경 없음.

### AG-T27 실제 통화 평가셋 (2026-11-20 사용자 요청: "실제 고객 통화 녹음 음성데이터를 구할 수 있는지 찾아봐" → "응 98번으로 Task 적어줘")
- 왜: 지금 테스트셋(172문항)·보정셋(960발화)은 모두 TTS 합성 + 인공 통화 환경이다. 실제 전화 음질·말버릇·쉼에서 STT CER과 턴 감지(잘못 끊기·말 끝 놓침)가 얼마나 달라지는지 모른다.
- 데이터: AI Hub dataSetSn=98, 음성 440시간 이상, 상담사 0 / 고객 1 화자 구분. 실제 녹취는 다산콜센터 부분만(금융/보험은 구축자가 직접 건 통화, 개인정보는 창작). 서울시 민원 내용이라 쇼핑 Mock·함수 일치·judge는 측정하지 않는다 → **STT·턴 감지·첫 소리 지연만**.
- 0단계(코드 전): 받은 파일의 라벨 형식·샘플레이트·발화 단위(문장/대화)를 확인해 이 절에 적는다. 8kHz면 경계에서 16kHz로 변환(불변 규칙).
- 새 `eval/build_realset.py`: `--raw data/aihub_98/raw --out data/realset --n 200 --seed 0`. 다산콜센터·고객(화자 1) 발화만, 길이·숫자 포함·문장 내 쉼(VAD 300ms 이상)으로 층화 표본. 출력은 3.2절 형식 그대로(`cases.jsonl`, `audio/<case_id>.wav` 16k mono s16le, `audio/meta.json`의 `speech_end`는 Silero VAD 마지막 음성 끝). `customer_utterance` = 원 전사, `category: "real_call"`, `env.channel: "real_phone"`, `expected_tools: []`. 턴 시뮬용으로 쉼 구간·구간별 전사를 보정셋(`data/turnset`) 라벨 형식으로도 낸다.
- 측정: `eval.stt_eval --cases data/realset/cases.jsonl --audio data/realset/audio`(현재 기본 STT + Soniox 키 있으면 비교), `eval.turn_sim`(realset 라벨), 서버 실측은 `eval.runner`로 첫 소리 p50/p95·말 끝 전 응답 수만. 결과 `docs/decisions/realset.md`에 합성 테스트셋 대비 표(CER·숫자 정확도·false cut·miss·지연).
- 테스트 먼저(`tests/test_realset.py`, 네트워크·실데이터 없음): 작은 가짜 라벨 + 8kHz/16kHz 짧은 wav 픽스처로 ① 고객 발화만 뽑힘 ② 출력 wav가 16k mono s16le ③ cases가 3.2절 필수 필드를 가짐(Starter Kit `run_eval.py` 필드 포함) ④ `speech_end` ≤ 오디오 길이 ⑤ 같은 seed면 같은 표본. 라벨 파서는 0단계에서 확인한 형식 기준.
- 지키기: `data/aihub_98/`, `data/realset/audio/`는 `.gitignore`. 문서·보고서에는 집계 수치만, 발화 원문 인용 금지. 결과가 합성셋과 크게 다르면 보정(AG-T12 방식)을 다음 Task로 따로 연다.

### AG-T28 번호 확인 되묻기 (2026-11-20 사용자 요청)
- 왜: 지금은 번호를 확실히 고칠 수 없으면 "번호를 한 자리씩 천천히 다시 말씀해 주시겠어요?"로 되묻는다. 고객 본인 주문 중 가까운 것이 있으면 "끝자리 공이삼일 무선 충전기 주문 맞으실까요?"로 물어 "네" 한마디로 끝나게 한다. round2 R2-best 기준 해당 후보는 막힌 호출 14 + not_found 7(상한, 발신 번호 모르는 문항 포함).
- 범위: 기존 조용한 교정(자릿수 틀림 + 후보 하나뿐 → 바로 호출, AG-T15a)은 그대로. 확인 질문은 **원래 되묻던 경우만** 바꾼다(그 턴에 함수 호출이 없다는 점은 기존과 같아 함수 점수 불변).
  1. 자릿수가 틀리고 교정이 안 된 경우(후보가 여럿이라 애매).
  2. 자릿수는 맞는데 함수 결과가 `not_found`인 경우.
- 후보: **발신 고객 본인의 주문(`get_order_list`)에서만**(남의 주문번호를 읽어 주지 않는다). 발신 번호를 모르거나 목록 함수가 없으면 기존 동작. 공통 부분열 기준(`_close_enough`)을 통과한 주문 중 1등과 차이가 기준(`margin`) 안인 것이 1~2개일 때만, 3개 이상이면 기존 되묻기.
- 결과: `{"status": "needs_confirmation", "original_status", "candidates": [{order_id, last_digits, product_name}], "message"}`. message는 끝자리(4자리 이하 번호면 전체)를 한글 숫자로 읽어 확인만 하라는 지시. 주문번호 전체는 읽지 않고, 조회 결과를 지어내지 않는다.
- 스위치: `DIRECTON_CONFIRM_ORDER_ID=1`(기본 끔). 끄면 이전과 같다.
- 테스트 먼저(`tests/test_order_confirm.py`): 후보 고르기(1개·2개·3개 이상·먼 번호·같은 번호 제외), 끝자리 읽기, 스위치 끔이면 기존 invalid_arguments·not_found 그대로, 켜면 애매한 자릿수 오류와 not_found가 needs_confirmation으로 바뀜, 발신 번호 없으면 그대로, 목록은 턴당 한 번.

### AG-T29 Soniox STT 도입 (2026-11-21 사용자 요청)
- 근거: "soniox 도입해줘" → 직접 키 잔액 소진(402) 확인 → "BAZE에 SONIOX가 있어. 그걸로 한번 도입해봐", "엔드포인팅도 기존 방안과 SONIOX의 방안을 비교해줘".
- 코드 변경 없음(어댑터·BAZE 라우팅은 AG-T04·T16에 있음). `.env`: 직접 Soniox 키 주석, `STT_PROVIDER=soniox`, `LLM_PROVIDER=gemini`.
- 새 도구 `eval/soniox_endpoint_eval.py`: `enable_endpoint_detection`을 켜고 실시간 송신, `<end>` 도착 시각을 turn_sim과 같은 판정으로 비교. `--extra`로 시작 메시지 필드 추가.
- 테스트 먼저(`tests/test_soniox_endpoint.py`): 판정(ok 지연·쉼 유형 false cut·intra·늦은 miss·없음), fake websocket으로 시작 메시지(endpoint 켬, extra)·`<end>` 시각·무음 꼬리·빈 프레임 종료, 오류 메시지면 예외.
- 결과: `docs/decisions/soniox_baze.md`.

### AG-T30 Soniox `<end>` 게이트 (2026-11-21 사용자 요청)
- 근거: "그 Soniox에 <end> 태그마다 규칙기반과 jev 기반 엔드포인팅으로 진짜 끝인지를 판단하는 로직을 추가해봐. 기존 / Soniox+Jev / Soniox + 규칙 이렇게 비교해줘"
- 구현: `agent/turn/soniox_gate.py`(`GatePolicy`·`rule_verdict`·`combine`·`SonioxEndpointGate.judge`·`simulate_gate`), `eval/soniox_endpoint_eval.py` 토큰 이벤트 녹화, `eval/soniox_gate_eval.py`(재생 비교, Jev 캐시 공유). 세션 연결 없음(기본 꺼짐).
- 설계 변경: 침묵·"말하는 중"은 Soniox 토큰이 아니라 로컬 VAD로(토큰이 음성보다 약 1~1.2초 늦게 도착). 도착 순간 VAD 음성인 `<end>`는 무시.
- 테스트 먼저(`tests/test_soniox_gate.py`): 규칙 보류/수락, Jev 임계·실패 시 규칙, 규칙+Jev에서 규칙 보류 우선, 재생(거절 뒤 다음 `<end>` 수락, VAD hold, 지난 `<end>` 무시, Jev 지연·음성 재개 시 버림, `<end>` 없을 때 상한), judge가 필요할 때만 Jev 호출.
- 결과: `docs/decisions/soniox_gate.md`. 기존 결정기가 여전히 낫다(trap이 구조적 원인).

### AG-T31 onedam_cs_testset 엔드포인팅 평가 (2026-11-21 사용자 요청)
- 근거: "그리고 onedam_cs_testset으로도 한번 평가해봐", "사실 SST는 이미 SONIOX로 확정이라 그건 평가 필요 없고 각 방안의 평가가 중요해", "끝나면 결과 표로 정리해줘".
- 구현: `eval/soniox_endpoint_eval.py` 일반 테스트셋 로더(`load_cases_dir`, `--cases --audio`), `eval/soniox_gate_eval.py` `simulate_detector`(기존 결정기를 Soniox 이벤트로 재생, 세션과 같은 VAD·finalize 순서)·임의 wav VAD 캐시·`--run` 여러 개.
- 테스트(`tests/test_soniox_gate.py`): finalize 지연 뒤 finalize 텍스트 사용, 음성 재개 시 finalize 버림·짧은 잡음 무시.
- 결과: `docs/decisions/soniox_gate.md` 6절. 실녹음에서 Soniox+Jev ≈ 기존(실제 끊음 13 vs 14), p95 약 0.4초 짧음. 합성셋까지 보면 기존 유지.
- `onedam_cs_testset/`·zip은 커밋하지 않았다(외부 데이터, 녹음 원본 포함).

### AG-T32 기존 결정기 + Jev 결합 비교 (2026-11-21 사용자 요청)
- 근거: "기존 결정기랑 Jev 결합하는 방안도 비교해줘"
- 구현: `eval/soniox_gate_eval.py` `simulate_detector(jev=, combo=)`·`Combo(trigger, policy, end_hi, max_requests, match)`·`collect_jev_texts`·`--combos`. 결합 규칙은 `agent/turn/jev_turn.decide_with_p` 그대로.
- 테스트(`tests/test_soniox_gate.py` +3): 늘리기로 soft_final 쉼 보호, `<end>` 트리거 즉시 확정, 최근 응답 규칙은 텍스트가 바뀌어도 p를 쓴다.
- 결과: `docs/decisions/soniox_gate.md` 7절. AG-T24 규칙(텍스트 정확 일치·2회)은 Soniox에서 효과 없음(62 → 61), 최근 응답 규칙이면 62 → 51.

### AG-T34 말하는 중 Jev 선호출 비교 (2026-11-21 사용자 요청)
- 근거: "선호출을 묵음 구간이 아니라 말이 진행되고 있을 때 일정 주기로 jev로 그 이후에 의미가 바뀔 확률을 계산해서 임계값을 넘으면 선호출을 하고 엔드포인팅이 되었을 때 선호출 결과와 최종 발화본의 정합성을 jev로 판단하는 방식으로 테스트해보고자 해. onedam_cs_testset 기준으로", "심사 GEMINI 말고 네가 서브에이전트로 진행해"
- 구현: `eval/precall_jev_eval.py`(Soniox 녹화 재생 + 선호출·tick 기록, Jev 질문 2개, 정책 τ·σ·first/latest·대조군, 실제 LLM 첫 행동 정답, 심사는 `judge_todo.jsonl` → 서브에이전트가 `judge.jsonl`). 세션 변경 없음.
- 테스트(`tests/test_precall_jev.py` 7개): tick은 말하는 중만, 침묵 선호출·finalize 교체·음성 재개 취소, first/latest 발사와 확정 뒤 도착 무시, 판정 대기·수락·거절 폴백·검사 없음 오답, 첫 행동 동등성, Jev 캐시 주입.
- 결과: `docs/decisions/precall_jev.md`. 이득 작음(τ0.5·σ0.5 확정 뒤 LLM 대기 평균 434 → 404ms, 오답 0) → 세션 미적용.
- 후속(사용자 요청 "정합성 질문을 첫 조회 기준으로 바꿔서 다시 테스트해봐"): `same_first_step` 질문·`first_body`(선호출 첫 행동·툴 목록 선택)·정책 `jev_first`/`jev_first_tool`/`jev_first_cat`·벌점 포함 대기. 결과 5절, 미적용 유지.

### AG-T34 공식 연습 데이터 A·B·C 평가 (2026-11-21 사용자 요청)
- 근거: "A B C directon-2026-voice-cs 데이터셋으로 편가 한번 돌려줘". 데이터 반출 조항·Jev 비지원 모델을 물었고 사용자가 "A·B·C 모두 돌리기"를 골랐다(연습 문항만, 시험 문항 없음).
- 구현: `eval/soniox_endpoint_eval.py` `load_cases_dir`가 글 없는 history(tool_call·tool_result)를 뺀다. 나머지는 AG-T32 도구 그대로.
- 결과: `docs/decisions/soniox_gate.md` 8절. A = B(끊음 1, miss 3), C는 Soniox `<end>`가 36%에서 안 나와 탈락.

### AG-T36~T40 공식 규격(directon-2026-voice-cs) 정렬 (2026-11-21 사용자 요청)
- 근거: "모든 mock 서버와 함수 등을 다 directon-2026-voice-cs를 기준으로 먼저 맞춰줘". 직전 논의(함수 호출 UX, 응답 템플릿 사전 생성 제안)는 이 정렬 뒤로 미룬다.
- 공식 규격과 지금 코드의 차이(코드로 확인):
  - 함수: 공식 8개(`get_order_info(order_id?, buyer_cellphone?)`, `get_return_info(customer_phone_number)`, `search_products(query)`, `cancel_order`, `return_order(order_number, …)`, `return_order_item(return_items, …)`, `exchange_order(order_number, …)`, `request_callback(customer_name, phone_number)`). 지금은 가정한 14개(`mock_api/data/tools.json`). `transfer_to_agent`·`get_order_list`·`track_delivery` 등 없음.
  - 인계: 공식은 함수 없이 말로 넘긴다(scenarios.md 10절: 고객센터 번호·운영 시간 안내). 정답에 없는 접수 함수를 부르면 그 문항 함수 0점(SCORING.md) → `responder._ensure_transfer`가 `request_callback`을 대신 부르면 안 된다.
  - 채점: 조회를 더 부르는 것은 감점 없음, 같은 결과가 history에 있으면 재조회 불필요. 상품 검색은 함수 점수 제외. 번호는 숫자만 비교.
  - history: `{"role":"tool_call","name","arguments"}`, `{"role":"tool_result","name","result"}`가 들어온다(연습 46개). `prompts._history_lines`는 `text`만 읽어 빈 "고객:" 줄을 만든다.
  - 결과 데이터 키가 한글(`주문번호`, `배송상태`, `송장번호` …)이다. 응답 템플릿(`_STATUS_KEYS`)·되묻기(`_order_rows`)는 영문 키를 본다.
  - 함수 지연은 측정기 중계가 넣는다(0.8~1.5초, 점수 제외). mock 서버는 지연을 넣지 않는다.
- **AG-T36** `mock_api/server.py`, `mock_api/data/tools.json`(공식 사본), `start_mock.sh`, `tests/test_mock_api.py`. 테스트 먼저(합성 소형 DB): GET /tools가 공식 8개와 같음, 조회는 하이픈·공백 무시 일치 / 없으면 `not_found`, 접수는 required 다 있으면 기록 결과(없으면 `{"status":"ok"}`) / 빠지면 `invalid`, 검색은 가장 가까운 query, 미정의 함수 404, `X-Session-Id` 기록, `--db` 인자.
- **AG-T37** `agent/prompts.py`. 테스트 먼저: `tool_call`·`tool_result`가 "상담원이 조회함: 함수(인자) → 결과" 줄로 들어감(긴 결과는 필요한 키만), 빈 줄 없음, `history_summary`가 결과 속 번호도 이미 나온 번호로 본다, 재조회 금지 규칙 문구.
- **AG-T38** `agent/responder.py`, `agent/config.py`, `agent/normalize.py`, `agent/catalog.py`, `agent/handoff_jev.py`. 테스트 먼저(`tests/test_official_tools.py`): 인계 약속 문장에도 함수 호출 없음, `order_number`·전화 3종 정규화, 공식 스키마에서 자릿수 틀린 주문번호(15·16자리 아님)·전화번호(11자리 아님)는 막힘 / 맞으면 호출, 발신 고객 주문(`get_order_info(buyer_cellphone)` → `orders[].주문번호`)으로 교정·되묻기, 카탈로그 기본 파일 없음 → 주문번호 길이 {15, 16}, 템플릿이 `orders[0].배송상태`를 읽음, 인계 판정 안내에 함수명 없음. (범위 조정 2026-11-21: 고정 프롬프트가 함수·브랜드 규칙을 한 문단에 섞고 있어 문구 변경은 AG-T39로 옮김)
- **AG-T39** `agent/prompts.py`, `agent/config.py`(`brand_dir`). 테스트 먼저: 시나리오 10개·정보 96개·말투 규칙이 시스템 프롬프트 고정부에 들어감, 첫 인사는 style.json.
- **AG-T40** `eval/runner.py`, `eval/score.py`. 테스트 먼저: 공식 cases 형식(`customer_utterance`, `reference.tool_calls`, `customer_profile`) 읽기, 중계가 함수별 지연을 넣고 호출을 세션별로 기록, 함수 판정이 공식 `tool_match`와 같은 결과(연습 문항 표본).
- **AG-T41** `agent/responder.py`(`_repair`). 테스트 먼저: 발신번호 01052072967인데 request_callback phone_number=01050207267(11자리) → 발신번호로 교정되어 전송, 전혀 다른 11자리 번호는 그대로, 발신번호 없으면 그대로, 교정 끄기 환경변수(DIRECTON_REPAIR_ORDER_ID=0)면 그대로.
- **AG-T42** `agent/responder.py`(`_start_filler`, `_length_error` 안내문). 테스트 먼저: 모든 호출이 자리 수 검사에 막히는 회차엔 필러가 재생되지 않음, 통과하는 호출이 있으면 필러 그대로, invalid_arguments 안내에 '한 자리씩 천천히'는 있고 '먼저 안내'·자리 수 규칙 고지 지시는 없음.
- **AG-T43** `agent/prompts.py`(`number_candidates`, 이번 통화 `번호 후보` 줄), `agent/responder.py`(`classify_number`, `_repair`, `_candidate_confirm`). 확신도: 강함 = 기존 `_close_enough`(공통 부분열) + 1·2등 차이(`repair_thresholds` margin) → 후보로 교체. 약함 = 형식이 같은 앞자리(010·주문 날짜)를 빼고 끝 8자리 공통 부분열이 전화 5개·주문 6개 이상(무작위 번호 오탐 시뮬레이션 전화 4.5%·주문 8.7%, 실제 번호 검출 99%·97%) → 함수를 부르지 않고 끝자리 확인 질문. 약함은 자리 수가 틀려 어차피 막힐 번호와 not_found에만 쓴다(자리 수가 맞는 번호는 고객이 다른 번호를 말했을 수 있어 강함일 때만 교체). 끄기 `DIRECTON_NUMBER_CANDIDATES=0`(프롬프트 줄·이전 대화 후보·약한 확인). 테스트 먼저(`tests/test_number_candidates.py`): 발신번호·이전 조회 주문번호가 후보 줄에 들어감, 후보 없으면 줄 없음, 끄기 스위치, 시스템 프롬프트에서 후보 읽기, 확신도 분류(강함·약함·없음·동점), 이전 대화 주문번호와 한 자리 다른 15자리 → 교체, 이전 대화 전화번호와 거의 같은 번호 → 교체, 자리 수 틀린 전화번호가 발신번호와 약하게 맞음 → needs_confirmation(끝자리 4351), 맞는 후보 없음 → invalid_arguments 그대로.
- 범위 밖(다음 묶음 후보): 행사 지원 모델로 교체(LLM GPT-5.4 mini, TTS `gemini-3.8-flash-lite-tts`, STT Soniox `stt-rt-v6` 직접), 제출 환경 Python 3.11·`start_server.sh`·`requirements.txt`·동시 4접속 확인.

### AG-T33 JevTurnDetector 최근 응답 규칙 (2026-11-21 사용자 확정 AD13 B)
- 근거: "B ㄱㄱ".
- 구현: `agent/turn/jev_turn.py` `JevTurnDetector(match="exact"|"latest")`. latest는 진행 중인 요청을 취소하지 않고, 끝난 응답을 요청 순서대로 반영해 이번 침묵의 가장 최근 p를 텍스트가 바뀌어도 쓴다. 같은 텍스트는 다시 묻지 않는다. `SONIOX_POLICY`(늘리기만 lo 0.5)·`SONIOX_MATCH`·`SONIOX_MAX_REQUESTS`(6). `agent/turn/detector.make_detector`는 `DIRECTON_JEV_TURN=1`일 때 이 설정을 쓴다(기본 꺼짐 그대로, exact 기본값도 그대로).
- 테스트(`tests/test_jev_turn.py` +3): 텍스트가 바뀐 뒤에도 latest는 p를 씀, 진행 중 요청 유지·요청 상한, make_detector가 Soniox 설정을 씀.
- lo 0.5는 같은 데이터(AG-T32·T34)에서 고른 값이라 낙관적이다. 켜기 전 Jev 사용 가능 여부는 주최 측 확인이 필요하다.

### AG-T35 규칙 보정: 키워드형 말 끝 (2026-11-21 사용자 요청)
- 근거: "AG-T35 ㄱㄱ".
- 구현: `agent/turn/rules.py` `PARTICLE_NOUN_SUFFIXES`("문의"·"회의"·"동의" 등 '의'로 끝나는 명사, 접미 일치)·`TOPIC_QUESTION_SUFFIXES`("방법은")로 끝나면 조사로 보지 않는다(→ neutral, 대기 약 0.95초. 전에는 particle 1.4~1.9초).
- 테스트(`tests/test_turn.py` +15): 키워드 말 끝 9개는 particle·connective·filler 아님, 조사 6개("제품의"·"배송은"·"방법을" 등)는 그대로 particle.
- 결과(Soniox 재생 6개 셋, 기존 결정기): 공식 연습 miss 3 → 0(끊음 1 그대로, p95 1325 → 1322), B도 miss 3 → 0. turnset·172·onedam 3종은 숫자 변화 없음.
- 일부러 넓히지 않은 것: "가격은"·"비용은" 같은 다른 주제어(실제 말에서 "가격은 … 얼마예요" 중간 쉼과 구별이 안 됨, 근거 사례 없음).

### AG-T44 LLM 규정 고정 (2026-11-21 사용자 확정 AD17)
- 근거: "gpt-5.4-mini가 대회에서 규정된 제한이야." → "응 진행해줘".
- 문제(코드 확인): `factory._llm_chain`이 OpenAI 기본 뒤에도 Gemini 폴백(BAZE 키면 `BAZE_LLM_FALLBACK=gemini-3.1-flash-lite`)을 붙였고, 기본 hedge(500ms)는 그 모델을 동시에 불러 먼저 첫 이벤트를 낸 답을 썼다.
- 구현: `agent/providers/factory.py` `_llm_chain`(OpenAILLM이면 [기본, 같은 client·키·경로의 사본], `LLM_SAME_MODEL_HEDGE=0`이면 [기본]), `llm_models`·`check_llm_policy`·`allowed_llm_models`(`LLM_ALLOWED_MODELS`, 기본 `gpt-5.4-mini`, `*`면 끔). `agent/server.py` `serve`: deps를 직접 조립할 때만 검사하고 위반이면 예열 전에 SystemExit(테스트·벤치가 넣는 deps는 제외).
- 테스트 먼저(`tests/test_llm_policy.py` 23개): 키 조합 3 × 모드 4에서 체인에 Gemini 없음, 사본 hedge(같은 client·BAZE 경로·키), 스위치 끔이면 단일, `GEMINI_LLM_FALLBACK_MODELS`를 줘도 무시, 정책 허용·거부·`*`·목록, 기본 허용값, auto + BAZE → OpenAI, serve 거부·주입 deps 통과.
- `.env`(git 밖) `LLM_PROVIDER=gemini` → `openai`로 바꿨다(그대로면 서버가 시작을 거부한다).

### AG-T45 함수 호출 안전장치 (2026-11-21 사용자 요청, 근거 `docs/14_공식측정_결과보고서.md` 6절 1~3)
- 새 모듈 `agent/tool_guard.py`. 응답기는 문장이 TTS로 가기 직전(`SpeechGuard.filter`·`finish`), 툴 실행 직전(`fix_full_returns`), 툴 결과 직후(`note_results`)에만 부른다. AG-T43과 병합하기 쉽게 `responder.py`·`prompts.py` 수정은 몇 줄로 한정.
- 전체 반품: `return_order_item`의 상품·수량이 이미 조회한 주문(이번 턴 get_order_info tool 메시지 또는 시스템 프롬프트 이전 대화의 `(조회 결과) get_order_info` 줄) 전체와 같으면 같은 id의 `return_order(order_number, reason, [refund_account, pickup_address])`로 바꾼다. 주문 정보가 없거나, 여러 주문에 걸치거나, 상품·수량 일부거나, 최근 고객 말(이번 통화 3개 + 이전 대화 3개)에 일부 반품 표현("일부", "~만 반품", "빼고")이 있으면 그대로. 공식 P0147-07(상품 1개 주문 + "일부반품")이 return_order_item 정답이라 고객 표현 검사가 필요하다. 프롬프트 반품 규칙도 "기본 return_order, 일부만일 때 return_order_item"으로 바꾸고 키 이름을 실제 결과의 `상세주문번호`로 고쳤다.
- 접수 주장: "접수해 두었습니다", "반품 접수가 완료되었습니다", "요청을 남겨 두었습니다" 같은 처리 완료 문장(질문·조건·미래형 제외)은 이번 턴에 접수 함수(함수 정의에서 조회 접두사가 아닌 것: cancel_order·return_order·return_order_item·exchange_order·request_callback)가 ok/already_requested였거나, 상태 조회(get_*) 결과에 처리 상태가 있거나, 이전 대화에 접수 기록·접수 완료 말이 있으면 그대로 나간다(지연 없음). 아니면 그 문장(과 뒤 문장)을 붙잡고, 그 회차에 함수 호출이 없으면 버린 뒤 교정 지시를 시스템 프롬프트에 붙여 한 번 다시 생성한다(이때만 회차 +1, 함수 사용 가능). 다시 생성해도 주장하면 그 문장만 버리고, 아무 말도 없으면 `FALLBACK_TEXT`(상담원 확인 안내).
- 호출 문자열: 알려진 함수 이름(밑줄·대소문자·`functions.` 무시), `이름({`, JSON(`{"키":`)이 든 문장은 TTS 전에 거른다. 열린 괄호는 다음 문장까지 추적하고, 호출 앞뒤의 한국어 말은 남긴다. 그 회차에 실제 호출이 없고 글로 쓴 호출이 알려진 함수 + JSON 인자면 실제 호출로 바꿔 실행한다(대화 기록의 상담원 글은 비움). 바꿀 수 없고 남은 말도 없으면 한 번 다시 생성.
- 테스트 먼저(`tests/test_tool_guard.py` 49개): 접수 함수 목록, 주문 정보 읽기(이전 대화 줄·tool 메시지), 전체 반품 변환(상품 2개·수량 4개), 진짜 일부 반품(상품 일부·수량 일부·"일부반품")·주문 정보 없음·다른 함수는 그대로, 주장 문장 판별 양성 8·음성 10, 호출 문자열 판별 양성 4·음성 4, 글로 쓴 호출 변환, 프롬프트 규칙, 응답기 배선 11개(전체 반품 변환과 대화 기록, 거짓 주장 → 다시 생성해 실제 호출, 성공한 접수 뒤 주장 통과, 실패한 접수 뒤 주장 차단, 반복 주장 → 대체 문장, 이전 대화 접수 기록이면 통과, 반품 현황 조회 결과면 통과, 글로 쓴 호출 → 실제 호출, JSON → 다시 생성, 여러 문장에 걸친 호출 미송출, 일반 문장 그대로).

### AG-T48 근거 없는 사실 금지 (2026-11-21 사용자 요청, 근거 runs/official_t45·official_t42_merged UX 채점)
- 진단: 공식 측정(main 17380d6) 즉시 0점 5건 중 4건과 D(맥락 적합성) 감점 다수가 함수 결과·브랜드 정보에 없는 사실이었다. P0010-11(조회 없이 '입금 확인이 되지 않았어요/조회가 어려워서', 발화에 주문번호·발신번호 있음), P0067-01(할인 먼저·배송비 나중 계산 순서 추측), P0158-01(결과 '정 수 미확인'인데 1일 1정으로 '1박스 1달' 계산), P0197-03(선불금 환불을 카드 기간 '3~4일'로), P0099-03(선불금인데 계좌 요청), P0149·P0135·P0120·P0060(검색 결과에 없거나 다른 섭취 방법·제품 단정), 2차 P0137-01(합배송 정책), P0121-01(환불보장제인데 반품 배송비 안내).
- 원인: 기존 고정부에 "모르는 내용은 지어내지 않는다" 한 줄뿐이라 '값이 없으면 어떻게 말하라'가 없었다. 브랜드 정보의 환불 기간은 항목 5개에 흩어져 서로 다르다(선불금 '즉시'/'1~2일', 계좌이체 '1~2일'/'2~3일', 카드 '3~4일'/'최대 3일'). 조회 결과 주장은 AG-T45 가드가 접수 주장만 봤다.
- 구현: 새 모듈 `agent/grounding.py`(`GROUNDING_RULES`, `REFUND_TABLE`, `refund_table_for`). `agent/prompts.py` `build_system`이 고정부 바로 뒤에 근거 규칙을, 브랜드 정보 뒤에 환불 정리를 넣는다(둘 다 캐시되는 고정부, 환불 정리는 브랜드 정보에 '선불금'이 있을 때만). `agent/tool_guard.py`: `is_lookup_claim`(주문·입금·결제·반품·내역 같은 대상 + '조회가 어렵/안 되', '확인이 되지 않았/안 됐', '찾을 수 없', '내역이 없'; 질문·조건('안 되면')·가능성('~ㄹ 수 있')·못 들었다는 말·다른 판매처 안내는 제외, '확인이 어렵다'는 정책 안내에도 쓰여 제외). `SpeechGuard`가 이번 턴 조회 함수(get_·search_ 등, 결과 무관)나 실패한 접수 함수(not_found 등)가 없고 이전 대화에도 조회 기록이 없으면 그 문장을 접수 주장과 같은 방식으로 붙잡아 `LOOKUP_NOTE`(번호가 있으면 조회, 없을 때만 묻기)로 한 번 다시 생성한다. 정상 경로엔 LLM 호출 추가 없음.
- 오탐 점검(오프라인, 기존 runs 함수 호출 없는 응답 5,335문장): 걸린 10문장 중 P0010-11 2문장 외 8문장은 자리 수 규칙을 설명한 옛 되묻기(프롬프트가 이미 금지)였다.
- 테스트 먼저(`tests/test_grounding.py` 27개): 근거 규칙이 고정부에 있음, 환불 정리는 선불금 브랜드에만, 조회 결과 주장 양성 6·음성 12, 응답기 배선 7개(조회 없는 주장 → 다시 생성해 get_order_info, 이번 턴 조회 뒤 통과, 이전 대화 조회 뒤 통과, 실패한 접수 뒤 통과, 이전 대화에 접수 성공만 있으면 다시 생성(P0010-11 형태), 정책 문장 그대로, 반복 주장 → 대체 문장).
- 남는 것: 섭취량·며칠분·계산 순서·다른 제품 정보 같은 내용 오류는 프롬프트 규칙으로만 막는다(넓은 숫자 검사기는 정상 답 오탐이 커서 두지 않음). 재측정 전.

### AG-T49 번호 되묻기 정리 (2026-11-21 사용자 요청, 근거 runs/official_t45 ux_scores·turns-20261121.jsonl)
- 사례별 원인(STT 결과 vs 정답 인자, 턴 로그):
  - 진짜 STT 오인식, 대체 식별자 없음(발신번호·이전 번호 없음) → 되묻기 자체는 맞다: P0052-02("51375. 5499. 451." ← 01079044351), P0086-02(주문·전화 둘 다 깨짐), P0154-02("5109580? 5497."), P0193-02("010-3710. 35."), P0196-02("018738842"), P0032-09("515110736", 이름만 있음), P0034-03("516. 5166273"), P0110-02("20162031. 008422" 14자리, LLM이 스스로 되물음), P0175-02(16자리로 들려 통과 → not_found). 7건은 길이 검사가 정상적으로 막았다(쓸 수 있는 번호를 막은 경우 없음). 감점 원인은 첫 마디 "네, 확인해 드릴게요."(번호 4자리 이상이면 조회 첫 마디) 뒤 "번호가 정확히 들리지 않았어요"로 엇갈린 것.
  - P0173-02(40점): LLM은 깨진 주문번호 하나만 불렀고, AG-T15a 교정용 발신번호 자동 조회(auto_order_list)는 성공했지만 그 결과는 교정에만 쓰이고 LLM에는 막힌 결과(invalid_arguments)만 갔다 → 다시 물음.
  - P0095-06(40점): "전화번호는 동일"이 STT에서 "전화번호는. 01."로 바뀌었고, 이전 대화에서 그 번호 조회 결과가 비었다는 사실이 요약에 없어 같은 번호를 다시 물음.
  - P0104-01·P0124-01·P0152-01: 번호 없이 get_order_info({}) 호출 → 필러 → invalid → 번호 질문. P0104-01은 같은 회차 글에서 이미 물은 뒤 다시 생성해 같은 질문 두 번.
  - P0194-03: 발신번호 교정(AG-T41)은 성공, 반품 내역 없음 뒤 주문 조회 대신 주문번호·제품명을 물음 → 이번 범위 밖(발신번호가 있으니 묻지 않는다는 기존 프롬프트 규칙을 LLM이 안 지킴).
  - AG-T43 약한 확인(`number_confirm`)이 안 나온 이유: 위 사례 대부분은 후보가 0개(발신번호·이전 번호 없음). P0173-02는 후보(발신번호)가 있지만 들린 번호와 끝 8자리 공통이 적어 '없음'. 문턱을 낮추면 무작위 번호 오탐이 늘어(AG-T43 측정) 바꾸지 않았다.
- 구현(`agent/responder.py`, 몇 군데 추가만):
  - `_caller_fallback`: 막힌 조회(get_*)가 AG-T28 확인·AG-T43 약한 확인으로도 안 풀리면 발신번호로 대신 조회한다(get_order_info는 같은 턴 자동 조회 결과 `_caller_orders_res` 재사용, 호출 1회). 결과가 있으면 그 결과 + `_note`(발신번호로 조회한 결과, 번호를 다시 묻지 않는다), 없으면 원래 invalid_arguments에 "발신번호로도 없었다"를 붙인다.
  - `_run_tools`: 같은 회차에 같은 함수의 다른 호출이 통과하면 막힌 호출은 보내지 않고 `skipped`(그 결과로 답하고 다시 묻지 않는다).
  - `_repair`: get_order_info의 번호 인자가 모두 비면 발신번호로 채운다(`number_filled`). 맞는 번호 인자가 하나라도 있으면 자리 수가 틀린 선택 인자는 None(`number_dropped`, P0086-02 유형에서 전화번호만 맞게 들린 경우).
  - `_length_error`: 번호 인자가 모두 선택 인자이고 모두 비면 보내지 않는다(invalid_arguments "번호를 받지 못했습니다", 필러 없음 — AG-T42 규칙 그대로). 같은 회차 글에 이미 질문이 있으면 그 호출을 버리고 다시 생성하지 않고 턴을 끝낸다(`empty_lookup_dropped`).
- `agent/tts_cache.py` `_garbled_number`: 아라비아 숫자 4자리 이상이 있는데 이어진 숫자 묶음 어디에도 전화(01로 시작 11자리)·주문(데이터 길이, 공식 15·16) 형식이 없으면 `select_opener_for_text`·`opener_kind`가 조회 첫 마디를 쓰지 않는다(불만 첫 마디는 그대로). 공식 측정 80턴 중 16턴이 바뀌고 그중 12턴이 되묻기 턴, 나머지 4턴은 "네, 고객님." 뒤 조회(필러는 기본 "확인해 드릴게요, 잠시만요.").
- `agent/prompts.py` `history_summary`: 이전 대화의 get_* 결과가 not_found거나 목록이 모두 비면 "그 번호로 조회했지만 결과가 없었다. 같은 번호·동일하다고 하면 다시 묻지 말고 찾지 못했다고 말한 뒤 주문번호나 다른 연락처를 묻는다" 줄(P0095-06). 고정 프롬프트는 건드리지 않았다(AG-T48과 겹치지 않게).
- 고치지 못한 것: 대체 식별자가 없는 STT 오인식(위 9건)은 여전히 되묻는다(이제 엇갈린 첫 마디 없이 한 번). P0175-02처럼 자리 수가 맞게 잘못 들린 번호는 not_found 뒤 되묻기.
- 테스트 먼저(`tests/test_number_reask.py` 24개): P0173-02 자동 조회 결과 전달(호출 1회), 같은 회차 skipped, 주문·반품 발신번호 대신 조회, 대신 조회도 없음 → 되묻기 + 사실 전달, 발신번호 없으면 그대로, 틀린 선택 인자 버림, 빈 조회 미호출·필러 없음, 빈 조회 발신번호 채움, 이미 물은 회차는 턴 끝(같은 질문 1번), 첫 마디 깨진 번호 5개·정상 6개, 이전 대화 결과 없음 줄 3개.

### AG-T50 LLM reasoning effort medium (2026-11-21 사용자 요청, AD18)
- 근거: "LLM 엔진 추론 레벨을 미디움으로 올려". 이전에는 `OpenAILLM(reasoning_effort="none")` 하드코딩(factory가 넘기지 않음).
- 설정(`agent/providers/factory.py`): `OPENAI_REASONING_EFFORT`(none|minimal|low|medium|high|xhigh, 기본 **medium**, 틀린 값은 ValueError). 직접 키·BAZE 둘 다 넘기고, 같은 모델 hedge 사본은 effort·include·reasoning 저장소를 그대로 받는다. `OpenAILLM` 생성자 기본도 medium.
- 타이밍(`REASONING_TIMING`, `llm_timing`): `LLM_HEDGE_DELAY_MS`·`LLM_FIRST_EVENT_TIMEOUT_S`를 비우면 기본 LLM이 OpenAI일 때 effort별 기본값. none 500ms/2.5s(AG-T15c 그대로), minimal 1000ms/3.5s, low 1500ms/4.0s, **medium 3000ms/6.0s**, high·xhigh 6000ms/12.0s. Gemini 기본이면 effort와 무관하게 500ms/2.5s. 환경변수를 주면 그 값이 이긴다. 응답기의 툴 후속 hedger는 기본 HedgedLLM의 값을 복사하므로 같이 바뀐다.
  - 이유: 추론을 켜면 첫 텍스트·툴 호출 전에 reasoning 토큰을 만들므로 첫 이벤트가 늦어진다. hedge 500ms를 두면 거의 모든 요청에서 사본이 발동해(비용 2배) 이득 없이 같은 대기를 두 번 한다. hedge 지연은 '대부분은 사본 없이, 꼬리만 자르는' 값(예상 첫 이벤트 p90 근처), 첫 이벤트 제한은 정상 요청을 끊고 처음부터 다시 시작하지 않도록 그 2배 안팎. **잠정값, 실측 전** — gpt-5.4-mini medium의 첫 이벤트 분포를 실측해 다시 정한다.
  - 그대로 둔 것: 응답기 `llm_timeout_s`(8초, `DIRECTON_LLM_TIMEOUT_S`) — 이벤트 사이 상한이라 effort와 무관하게 두었다. 단 툴 후속 고정 경로는 기본 모델 6초 뒤 사본으로 넘어가면 남은 2초 안에 첫 이벤트가 와야 한다. 실측에서 꼬리가 길면 `DIRECTON_LLM_TIMEOUT_S=12` 또는 `LLM_FIRST_EVENT_TIMEOUT_S`를 줄인다.
- reasoning item(`agent/providers/openai_llm.py`): effort가 none이 아니고 include가 켜져 있으면 `include=["reasoning.encrypted_content"]`를 요청하고, 응답의 reasoning item(암호화 내용 있는 것만, `id`·`summary`·`encrypted_content`)을 그 응답의 function_call id들에 묶어 `reasoning_store`(최대 2048)에 둔다. 다음 라운드 입력에서 그 호출이 든 assistant 메시지 앞에 한 번만 되돌린다(store=False라 서버 상태가 없어, OpenAI 권장 방식). `agent/core/types.py` 계약은 그대로(인스턴스 내부 저장, Gemini thought_signature와 같은 방식). HedgedLLM은 툴 후속을 모델 이름으로 고정해 사본이 이긴 라운드도 기본 쪽으로 보내므로 factory가 저장소를 공유시킨다.
  - `OPENAI_REASONING_INCLUDE`: 비우면 auto — 직접 OpenAI 키면 켬, **BAZE 경유면 끔**(게이트웨이가 `include`를 통과시키는지 미확인, 거부하면 모든 LLM 요청이 실패하므로 보수적으로). `1`/`0`으로 강제. 끄면 추론은 그대로 하되 라운드마다 처음부터 추론한다.
- 스트림: reasoning 이벤트(`response.reasoning_summary_*`, `response.reasoning_text.delta`, reasoning output item)는 LLMEvent로 내보내지 않는다 — 말하지 않고, hedge·첫 이벤트 제한의 '첫 이벤트'는 첫 텍스트·툴 호출·done 기준(추론 시간 포함한 실제 출력 시각). summary는 요청하지 않는다.
- 테스트 먼저(`tests/test_reasoning_t50.py` 23개): 기본 medium·include, none이면 include 없음, include 끔, 틀린 effort, reasoning 이벤트 미송출, 툴 후속 입력에 reasoning item 되돌림·암호화 없음/끔이면 안 넣음·병렬 호출 한 번만·모르는 id 무시, factory 기본·환경변수·BAZE auto·사본 상속·저장소 공유(사본 라운드 → 기본 후속), effort별 타이밍 5종(hedge·fallback), 환경변수 우선, Gemini 기본은 예전 값. `tests/test_providers.py` 기존 단언 none → medium.
- 실측 때 볼 것: 첫 이벤트 p50/p90(첫 소리 ≤1.5s 만점 — 첫 소리는 캐시 첫 마디라 직접 영향은 적지만 첫 문장 이음새), 조회 뒤 응답 p50/p90(≤3.5s 만점, none에서 2.17/2.73s 중 두 번째 LLM 호출 ~1.5s), hedge 발동률(`stats`의 `hedge:`), `fallback`·첫 이벤트 시간 초과·응답기 apology 횟수, reasoning 토큰 수와 비용, BAZE에서 `OPENAI_REASONING_INCLUDE=1`이 400 없이 encrypted_content를 돌려주는지.
- 실측 결과(4차 공식 측정, docs/14 7절, 직접 키 + BAZE 지연 시뮬레이션 0.54초): medium은 none보다 응대 78.2 → 82.9, 속도 99.4 → 97.9, 함수 81.2 같음, 참고 총점 85.2 → 87.0. LLM 첫 이벤트 p50/p90 첫 라운드 940/1394 → 2487/3707ms, 툴 후속 737/955 → 1853/3426ms. 조회 뒤 답 p50/p90 1.59/1.98 → 2.67/4.36초, 3.5초 초과 0 → 9건. 즉시 0점 3 → 1건(사실 지어내기 사라짐). 기본 medium 유지, low 측정 진행 중. BAZE의 include 동작은 아직 미확인(이번엔 직접 키).

### AG-T51 reasoning effort 기본값 low (2026-11-21 사용자 확정, AD19)
- 근거: 4차 공식 측정에 low를 더 돌림(`runs/official_t50_low/`, docs/14 7절). 참고 총점 none 85.19 / low 86.99 / medium 87.05. 응대 78.2 / 82.9 / 82.9, 함수 81.2 / 78.8 / 81.2, 속도 99.4 / 99.3 / 97.9. 조회 뒤 답 p50/p90 1.59/1.98 · 1.57/2.45 · 2.67/4.36초, 3.5초 초과 0 / 1 / 9건. LLM 첫 이벤트 툴 후속 p50/p90 737/955 · 917/1399 · 1853/3426ms.
- 결정: 점수가 medium과 같고 꼬리 지연 여유가 커서 low를 기본값으로 한다. 0.54초 시뮬레이션은 BAZE p50만 흉내 내므로 실제 BAZE에서는 medium의 3.5초 초과가 더 늘 수 있다.
- 구현: `agent/providers/openai_llm.py` `DEFAULT_REASONING_EFFORT = "low"`(factory `ProviderConfig.openai_reasoning_effort`가 이 값을 씀), 모듈 docstring 2곳. low 타이밍은 AG-T50 표 그대로(hedge 1.5초 / 첫 이벤트 4.0초).
- 테스트 먼저: `tests/test_reasoning_t50.py` 기본값 단언 4개(OpenAILLM 기본 effort, factory 기본 effort, BAZE 경로 effort, 기본 타이밍 1.5/4.0)와 `tests/test_providers.py::test_openai_llm_stream`의 effort 단언을 low로 바꿈 → 5 failed(RED) → 기본값 변경 후 GREEN.
- 남는 것: low 채점자는 같은 문항의 none·medium 점수를 보고 채점해 기존 점수에 끌렸을 수 있다. 실제 BAZE 경유·지정 TTS 조건 측정은 아직이다.

## 8. 수동 체크리스트

| 항목 | 기대 | 결과 |
|---|---|---|
| 생성 음성 샘플 10개 청취 | 자연스러운 한국어, 통화 환경 구분 | |
| 상담원 응답 음성 10개 청취 | 응답이 질문에 맞고 끊김 없음 | |

## 9. 기존 테스트 영향
없음(신규 저장소).

AG-T36~T38(AD15 완전 교체, 2026-11-21 사용자 확정)로 바뀐 기존 단언:
| 파일 | 바뀐 것 | 이유 |
|---|---|---|
| tests/test_mock_api.py | 전면 교체(옛 14개 함수 → 공식 8개, 합성 DB 20개) | AG-T36 |
| tests/test_prompts.py | 자동 연결 안전망 테스트 3개 삭제, 필러 예외 테스트는 `filler_skip_tools`를 명시, 주문번호 기본 "10자리" → "15자리나 16자리" | 정답에 없는 접수 함수 호출은 함수 0점, 공식 번호 형식 |
| tests/test_jev_handoff.py, tests/test_wiring_t25.py | 인계 안내에 `transfer_to_agent`/`request_callback` 대신 "함수를 부르지"·"전화 문의 방법" | 공식 함수에 인계 함수 없음 |
| tests/test_catalog.py, tests/test_dayof_swap.py | 기본 카탈로그 None, 기본 주문번호 길이 {15, 16} | 상담 서버는 mock 데이터를 읽지 않는다(README 7장) |
| tests/test_session.py, tests/test_session_debug.py | 장난감 주문번호 "1" → "20261115-0000231" | 주문번호 키에 형식 검사 기본 적용 |
| tests/test_responder.py, tests/test_after_tool.py, tests/test_prompts.py | 모듈에 `legacy_world` fixture(옛 카탈로그, 10자리), 장난감 "1" → "2611150231", 정규화 훅 테스트는 형식 검사 끔 | 옛 체계 전제 테스트, AG-T39·T40에서 정리 |
| tests/test_prompts.py, tests/test_session_precall.py | 필러 테스트의 장난감 주문번호 "1" → 유효 번호("2611150231"·"20261115-0000231") | AG-T42: 자리 수 검사에 막힐 회차엔 필러를 틀지 않는다(단언 그대로) |
| 옛 쇼핑몰 import(test_testset·test_llm_eval·test_demo_history·test_order_confirm·test_reprompt_policy·test_dayof_swap) | `mock_api.server` → `mock_api.legacy_shop` | AG-T36(단언 그대로) |
| tests/test_providers.py | `test_baze_mode_fallback_chain_drops_missing_models`: OpenAI 기본 + BAZE 체인 [gpt-5.4-mini, gemini-3.1-flash-lite] → [gpt-5.4-mini, gpt-5.4-mini] | AG-T44, AD17(대회 규정 밖 LLM 금지) |
| tests/test_order_confirm.py(`test_off_by_default_not_found_unchanged`, `test_off_ambiguous_length_error_unchanged`), tests/test_reprompt_policy.py(`test_no_repair_when_ambiguous_blocks`) | 단언 그대로, `DIRECTON_NUMBER_CANDIDATES=0`을 걸어 '약한 확인까지 끈 옛 동작' 테스트로 고정 | AG-T43: 약한 확신 되묻기(needs_confirmation)가 기본 켜짐이라 애매한 자리 수 오류·not_found가 확인 질문으로 바뀐다(사용자 요청: 약한 확신 → 되묻기) |
| tests/test_session_precall.py(`test_select_opener_for_text_rules`) | 장난감 주문번호 "2611150231"(10자리) → "20261115-0000231" | AG-T49: 형식(공식 15·16자리)에 맞지 않는 번호는 STT 오인식으로 보고 조회 첫 마디를 쓰지 않는다(단언 그대로) |

## 11. 성공 지표
`docs/01_구현계획.md` 4.9절: first_sound p95 ≤ 1.3초, 잘못 끊기 ≤ 1%, 함수 일치 ≥ 95%.

## 12. 리스크
`docs/01_구현계획.md` 8절 + Gemini STT 지연이 Soniox보다 클 수 있음(실측으로 확인).

## 13. 발견 이슈 · 범위 밖
- (AG-T40) 측정 환경: Edge TTS 타임아웃 → Gemini TTS 대체가 할당량 소진(429)이면 응답 전체가 "잠시 확인이 어렵습니다"로 끝난다(첫 측정 80건 중 21건). 측정 때는 `DIRECTON_TTS_FALLBACK=none`. 대체 TTS 실패가 응답 실패로 번지지 않게 하는 것이 후보 Task.
- (AG-T40) 정상 번호 거절이 가장 큰 감점: 11자리 전화번호·15자리 주문번호를 받고도 "15·16자리여야"/"010으로 시작하는 11자리여야"로 되묻는다(P0052-02, P0154-02, P0193-02, P0148-02, P0106-02 등, 함수 감점 23건 중 '호출 안 함' 13건 대부분). 원인 확인(2026-11-21, debug 세션 재현 7/8건): **번호 검증은 정상, STT가 숫자를 못 받아 적는다.** 예: "01079044351" → Soniox "51375. 5499. 451.", "20261001-0005987" → "2026년 2020. 00987". 파이프라인 없이 원본 wav를 Soniox에 직접 넣어도 같고(context 유무·language_hints_strict 무관), gpt-4o-transcribe도 "오유 삼지로 오사구구 사오인"처럼 틀린다 → 연습 음성 자체의 숫자 발음이 알아듣기 어렵다. 번호를 말하는 문항 21개 중 caller 있음 7개. 대응 후보: (1) 자리 수는 맞지만 발신번호와 거의 같은 전화번호는 발신번호로 교정(`_repair`는 지금 자리 수가 틀릴 때만 교정 → P0119-04·P0133-03·P0194-03을 놓침), (2) caller 없는 문항은 되묻기가 맞으나 "잠시만 기다려 주세요" 뒤 되묻기·묻지 않은 안내 덧붙이기를 없앤다.
- (AG-T40) 번호 옮겨 적기 오류: request_callback 3건·get_order_info/get_return_info/return_order 각 1건. 발신번호·직전 조회 결과로 교정 필요.
- (AG-T40) 전체 반품에 return_order_item 3건(P0018-05, P0099-03, P0197-03) → 정답에 없는 접수 함수 = 함수 0점. 부분 반품일 때만 쓰도록 규칙 강화 후보.
- (AG-T40) P0130-01 고객 발화 중 소리(첫 소리 -3.23s, 속도 0점). 접수 약속 후 함수 미호출(P0148-08, P0164-05). 제품 정보 섞기(P0060-02, P0067-01, P0158-01). 모든 턴이 "네, 확인해 드릴게요. 잠시만 기다려 주세요."로 시작, 답이 4~5문장.
- (AG-T40) 포트 9000을 다른 상담원 서버(출처 미상)가 쓰고 있어 측정 서버는 9001.
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
- (AG-T14) 브라우저 경로: 연결 전 마이크 오디오는 버려짐("연결됐습니다" 후 말하기 안내). 브라우저 노이즈 억제·AGC가 쉼 길이를 바꿔 조기 확정 1/4회 관찰. 16kHz AudioContext로 다운샘플 품질 문제 해결.
- (AG-T15a) BAZE 게이트웨이 모드에서는 phone_number가 "[PHONE]"으로 가려짐 → responder가 원문·caller로 복원. judge 비교 1은 게이트웨이 gpt-5.4-mini 혼합 조건(1회) — Gemini 단독 judge 재측정은 AG-T15d에서. 폴백 체인의 gemini-2.5-flash-lite 404. eval 로그에 `aclose(): asynchronous generator is already running` 경고(비치명).
- (AG-T14b) history 마지막이 '?'이면 expect_short로 대기 −100ms → 띄어 읽은 번호가 중간에 끊김(1/2회). 함수 결과와 다른 금액·요일 착오 발화 관찰(LLM).
- (AG-T15c) 컨텍스트 캐시는 prompts.py의 `## 이번 통화` 마커에 의존(바뀌면 캐시 꺼짐). 캐시 요청엔 system_instruction·tools 동시 지정 불가 → 가변부를 별도 user 턴으로. DEFAULT_LLM_FALLBACK의 gemini-2.5-flash-lite는 404 → 제거 권장.
- (AG-T08) 숫자가 아닌 말 속의 수 단어도 잡힌다("이 번호요 공일공" → "2010"). "다시" 정정은 앞부분을 대체하지 않고 이어 붙인다. 고유어 수(하나, 둘) 미지원. 콜백 시간 인자 형식은 Mock API 정의와 맞춰야 한다.

- (AG-T16) BAZE 경유 LLM 첫 이벤트 지연이 들쭉날쭉: 3회 실측 gpt-5.4-mini 4.75/1.40/1.94초, gemini-3.5-flash-lite 1.50/1.02/5.66초. `LLM_FIRST_EVENT_TIMEOUT_S`(2.5초)를 넘기면 3.1-flash-lite로 폴백된다(스모크 1회 발생). BAZE TTS 첫 오디오 3.24초(비스트리밍, 2.8초 분량). → BAZE 모드 속도 점수는 AG-T10/T11 측정에서 확인. 게이트웨이 Soniox는 stt-rt-v6·16kHz 정상.

- (AG-T15d) 당일 TTS 대체 위험: gemini-3.8-flash-tts Tier 1 일일 100회 한도 소진 상태 — edge 장애 시 대체 실패(01:35~01:47 장애 구간에서 사과문 응답 발생). 대체 모델을 2.5-flash-preview-tts로 바꿀지 검토(코드 변경, 별도 Task).
- (AG-T15d) R2-best 남은 실패: 말 끝 전 발화 12건(대부분 tool_action), 첫 소리 1.5초 초과 13건(모두 babble), caller 문항에서 get_order_list 선호출 약 12건(기대값 엄격성 점검 필요), STT 숫자 오인식 약 6, 되묻기 약 8 — round2.md 7절.
- (AG-T15d) 4자리 주문번호 STT가 앞자리 "일"을 빠뜨림(연습판 함수 실패 6건 중 5건). 템플릿 뒤 LLM 세부 누락·반복. Jev 인계 질문당 약 5.5회 호출(선호출 텍스트 바뀔 때마다) — 단가 미확인.
- (AG-T29) Soniox 오류(잔액 402·인증)는 연결이 열린 뒤 첫 응답으로 온다 → `_warm_stt`(open만)가 통과하고 모든 턴이 빈 텍스트가 된다. `FallbackSTT`는 gemini_live에만 있고 Soniox → Gemini 폴백이 없다. 예열에서 무음을 보내 finalize까지 확인 + 스트림 오류 시 Gemini로 전환하는 Task 후보.
- (AG-T29) STT만 바꿔 측정했다. 결정기 입력 텍스트(부분 결과)가 Soniox로 바뀌면 연결어미·숫자 판정이 달라질 수 있다 → 172문항 E2E 재측정(`eval.runner`) 필요.
- (AG-T30) Soniox 토큰은 음성보다 약 1~1.2초 늦게 도착한다(BAZE 경유 실측). 세션의 텍스트 기반 판단(결정기 입력 텍스트)도 Soniox 부분 결과가 늦으면 침묵 200ms 시점 텍스트가 비거나 한 구절 앞일 수 있다 → AG-T29의 E2E 재측정에서 확인.
- (AG-T31) onedam `speech_end` 라벨이 실제 음성 끝보다 0.2초 이상 늦은 문항 14개(원본 VAD 기준). 빠른 방식이 "말 끝 전"으로 잘못 잡힌다 → 실제 끊음(뒤에 음성 있음)을 따로 셌다.
- (AG-T31) 기존 결정기와 Soniox+Jev가 틀리는 문항이 다르다(겹침 9/14). "-는데요" 뒤 짧은 쉼은 Jev가, 1초 넘는 쉼("잠시만요")은 기존이 낫다 → 결합 Task 후보(기존 대기 중 `<end>`+Jev 높음이면 단축, 또는 Jev 낮음이면 연장).
- (AG-T32) `JevTurnDetector`는 Jev p를 지금 텍스트와 같을 때만 쓰고 침묵당 2회만 묻는다. Soniox 부분 결과는 침묵 중에도 바뀌어(토큰 약 1초 지연) p가 거의 버려진다 → `DIRECTON_JEV_TURN=1`로 켜도 Soniox에서는 효과가 없다. AG-T33.
- (AG-T34) Jev `same_reply`(응답 전체가 같은가)는 prefetch에 필요한 "첫 행동이 같은가"보다 엄격하다(AUC 0.75, 놓침 10/16). 뒤에 질문이 붙어도 첫 조회는 같은 경우를 거절한다. 다시 본다면 첫 조회 기준 질문 + 선호출 툴 호출을 상태에 넣기(상한 평균 −81ms). → 재테스트(5절): `same_first_step`(첫 행동 상태 포함)·툴만이면 재사용 14%, 대기 평균 352ms(벌점 포함 376, 상한 353)지만 툴 쌍 AUC 0.45(거의 다 수락), 틀린 툴 선택 3건은 못 거름. 값 누락 거르기는 규칙(인자 숫자 vs 최종 텍스트 숫자)으로 대체 가능
- (AG-T34) 공식 연습 음성에서 Soniox `<end>`가 80개 중 29개에서 말 끝 + 2.5초 안에 안 온다(19개는 아예 없음, `max_endpoint_delay_ms` 1000도 같음). Soniox 엔드포인트에 기대는 설계는 피한다. 다른 데이터(합성·onedam)에서는 miss 0~10%였다.
- (AG-T34) 규칙이 "~문의"(명사)를 조사로 본다(NOUN_EXCEPTIONS에 없음) → 키워드형 말에서 1.6~1.9초 대기. AG-T35.
- (2026-11-21 함수 UX 점검) 자릿수 검사로 막힌 호출에도 필러가 나간다(로그 11/11: "확인해 드릴게요. 잠시만 기다려 주세요. 번호를 … 다시 말씀해 주시겠어요?"). 필러는 턴당 1회라 2회차 함수 대기(30턴)와 교정용 자동 조회 대기는 무음. `ToolsClient()` POST 타임아웃 10초. AG-T38 이후 처리 후보.
- (AG-T36) 공식 mock_db에 같은 인자로 결과가 둘인 기록 3건(같은 손님을 다른 시점에 조회, 연습 문항도 P0196은 앞·P0177은 뒤를 씀) → 인자만으로 못 고르니 첫 기록. 시험 DB도 같을 수 있다.
- (AG-T36) 데모(`demo/`)·`bench/`·`runs/*/run.sh`는 `python -m mock_api.server`를 옛 데이터 전제로 띄운다 → 이제 공식 8개 함수가 뜬다. 데모 프리셋·발신자는 AG-T38·T39 뒤 맞춘다. `start_mock.sh`는 공식 예시처럼 `python3`(제출 컨테이너 기준).
- (AG-T38) `agent/catalog.py`·`config.py`·`handoff_jev.py`는 원래 CRLF였는데 편집 중 LF로 바뀌어 223de83 diff가 파일 전체로 잡혔다(기능 무관, 저장소 나머지는 LF).
- (AG-T38) 다른 세션이 동시에 같은 저장소를 커밋했다(223de83에 진행 중이던 AG-T38 일부가 묶임). 이후 AG-T38 마무리 커밋에 나머지를 담았다.
- (AG-T38) 고정 프롬프트 되읽기 예시가 아직 10자리(`2611150231`)인데 길이 안내는 15·16자리 → AG-T39.
- (AG-T39) 공식 `now`는 "2026-11-04 (수) 12:39" 형식인데 기존 파서는 앞 16자만 읽어 영업시간·요일 판단이 모두 빠지고 있었다 → 고침.
- (AG-T39) 브랜드 자료도 공식 폴더에서 읽는다(DIRECTON_BRAND_DIR). 제출 묶음에 `directon-2026-voice-cs/data/brand/`를 함께 넣어야 한다.
- (2026-11-21) 제출 환경은 Python 3.11인데 `pyproject.toml`은 `>=3.12`. 3.12 전용 문법 사용 여부 확인 필요.
- (AG-T35) 규칙 원문 기준으로 "틱톡광고"처럼 붙여 쓴 복합 명사 "~광고"는 connective가 된다(`NOUN_EXCEPTIONS`는 낱말 전체 일치). Soniox는 띄어 써서 영향 없음. 붙여 쓰는 STT로 바꾸면 다시 볼 것.
- (AG-T44) AG-T40·T42 공식 측정 서버 env에 `GEMINI_LLM_FALLBACK_MODELS=none`이 없었다면 BAZE 키 + hedge 기본값이라 LLM 체인이 [gpt-5.4-mini, gemini-3.1-flash-lite]였다. gpt-5.4-mini 첫 이벤트가 500ms를 넘는 턴은 Gemini 답이 쓰였을 수 있다. 턴 로그에 답한 모델이 없어 확인할 수 없다 → AG-T44 이후 재측정 필요, 턴 로그에 served 모델 기록 후보. hedge 500ms·첫 이벤트 2.5초는 Gemini 실측값이라 gpt-5.4-mini 기준 재튜닝 후보(같은 모델 사본 hedge는 요청 비용 최대 2배).
- (AG-T43) 번호 문항 25개 중 함수 실패 13건 가운데 12건은 발신번호도 이전 대화 번호도 없어 번호 후보가 0개다(P0032-09는 주문번호 후보뿐인데 전화번호가 필요). 후보 기반 교정·확인은 이 문항들에 쓸 수 없다. 남은 대응 후보: 번호 구간만 숫자 특화 STT로 다시 받아 적기, 자리 수가 틀린 번호의 되묻기 대신 들린 숫자 묶음을 되읽어 고객이 고치게 하기.
- (AG-T46) Soniox 2차 인식은 어떤 변형으로도 번호를 살리지 못한다(정답 일치 5/24로 모든 변형 동일, 1차에서 맞은 5건뿐). 공통 오류는 '010'의 0(공)을 5(오)로 듣는 것(01018167368 → 5151107368, 01098505497 → 510-9580-5497). 숫자 문맥은 앞자리 010을 가끔 되살리지만(P0154·P0047) 가운데 자리가 틀려 정확히 맞지 않는다. 평균 편집 거리 base 5.2 / ctx 4.85 / crop 5.0 / pad 4.85 / slow 4.95. 음성 자체 문제(T40의 gpt-4o-transcribe도 틀림)로 보고 연결하지 않았다.

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
- 2026-11-19 | AG-T16 | DONE | agent/providers/{baze_tts.py(신규),factory.py,_gemini.py,openai_llm.py}, pyproject.toml·uv.lock(openai 추가), tests/test_providers.py(BAZE 6개), RED 6 failed → GREEN 41 passed, 전체 802 passed, 실호출 스모크(라우팅·도구 호출·TTS) 정상, LLM 지연 편차는 발견 이슈로
- 2026-11-19 | AG-T14 | DONE | demo/{server.py,static/*,run_demo.ps1,run_demo.sh}, agent/session.py(debug:true일 때만 debug 이벤트, 끔이면 바이트 동일), tests/test_demo_server.py·test_session_debug.py, docs/03_사용자_테스트_가이드.md(시나리오 16개), RED ImportError → GREEN 13 passed, 전체 802 passed, WS 스모크·헤드리스 크롬(가짜 마이크) 확인
- 2026-11-19 | AG-T15a | DONE | agent/{prompts,responder}.py, eval/llm_eval.py(--stt-text), tests/test_reprompt_policy.py, docs/decisions/llm_tuning_r2.md, RED ImportError → GREEN 35 passed·전체 915 passed, 후보 C 채택: judge(STT 텍스트, 146문항) 58.9 → 65.3, 함수 유지(84.2 → 83.6, Gemini 직접), caller 주문 목록 기반 주문번호 교정, 응답 내용 보강(3~4문장), DIRECTON_PROMPT_DETAIL·DIRECTON_REPAIR_ORDER_ID로 끔
- 2026-11-19 | AG-T14b | DONE | demo/{presets/history_presets.json,server.py,static/*}, docs/03_사용자_테스트_가이드.md(6절), tests/test_demo_history.py, RED 2 실패+6 에러 → GREEN 8 passed, 회귀 21 passed, 프리셋 8개 mock matched·실서버 스모크
- 2026-11-19 | AG-T15c | DONE | agent/providers/{hedged_llm,llm_cache,gemini_llm,factory}.py, scripts/bench_llm_r2.py, tests/test_hedged_llm.py, docs/decisions/llm_latency_r2.md, RED 26 실패 → GREEN 27 passed·전체 924 passed, 기본 LLM_MODE=hedge(500ms)+컨텍스트 캐시: 첫 이벤트 p95 3.73 → 1.64s, 공백 p90 2231 → 207ms, 질문당 비용 0.248 → 0.074¢
- 2026-11-20 | AG-T15b | DONE | detector.py 13묶음 표·정정/'에' 이어 부르기/쉼표 규칙·상한 1.3초, turn_sim·turn_calibrate r2, 보정셋 771→960, 172 시뮬 말 끝 전 응답 28→13(교차 검증 18), 지연 p95 1292ms, miss +1.2%p(교차 +1.7%p), test_turn_r2 43 RED → 52 GREEN, 전체 924 passed. 이슈: 테스트셋 불만 문항 음성에 TTS 스타일 지시가 읽혀 들어감(T0035~37 재합성 필요), 실서버 재측정은 T15d
- 2026-11-20 | AG-T23 | DONE | docs/06_기획서_초안.md(공식 12개 목차), 07_발표_초안.md(9장·420초·예상 질문 8), 08_검토보고서_선준비.md(1장), 코드 변경 없음. 당일 값은 〔당일 측정〕 표시(round2, after_tool, T17~T22 결과)
- 2026-11-20 | AG-T17 | DONE | mock_api --data/--tools(+Starter Kit 형식 {함수: [{arguments,result}]}), DIRECTON_MANUAL/BRAND/STYLE(캐시 고정부 유지), 주문번호·전화 자릿수 데이터 추론(catalog.order_id_lengths), DIRECTON_NO_AUDIO_LOG 고객 텍스트 제거, docs/05_당일_체크리스트.md, .gitignore official/. test_dayof_swap 22 RED → 23 GREEN. 이슈: stt_recheck.py:139,158 INFO 로그에 고객 발화(콘솔을 파일로 돌리면 저장됨), responder MIN_NUMBER_DIGITS=7이 4자리 번호를 버림(T21에 전달)
- 2026-11-20 | AG-T22 | DONE | data/practice_health/(가상 브랜드 온결헬스, 함수 15개(결제 2개 신규), 상품 15·주문 23(4자리), Starter Kit 매핑 형식 mock_data 280항목, manual.md, cases 40(업무 7종·몸 불편 인계 3·history 3), generate.py), 음성 40개(Gemini 73회, gitignore). test_practice_health 23 RED → 24 GREEN. 이슈: 매핑 모드 transfer_to_agent는 reason 무관 matched, H0001 '천이백삼십사' 읽기 정규화 미확인
- 2026-11-20 | AG-T18 | DONE | fallback_tts.py(edge→Gemini, 첫 청크 2초·쿨다운 30초), factory.make_tts_with_fallback, server GuardedWS·세션 사망 시 사과+turn.end 열화 모드, _privacy.redact(stt_recheck·openai_llm 로그), scripts/tunnel.ps1·sh(cloudflared), eval/stress.py, docs/decisions/stability.md. test_stability 23 RED → 32 GREEN, 전체 1073 passed. 실측 4×2 동시: 실패 0, 첫 소리 p50 0.86 / p95 1.21s, RSS 159→246MB. 이슈: Windows에서 두 서버가 같은 포트 9000 공유 가능(당일 단일 프로세스 확인), responder 번호 로그·session 사과 송신 미보호(T21·T19에 전달)
- 2026-11-20 | AG-T19·T20 | DONE | session.py 바지인(응답 중 음성 300ms·첫 소리 뒤 1초 무장, 남은 TTS 중단·들린 만큼 transcript·turn.end·끼어든 말+200ms를 다음 턴 STT로, 송신 페이싱 1.5초 앞까지·10초 뒤 해제), 실패 시 turn.end 보장·_cleanup 상태 초기화, tts_cache.choose_opener(답 준비 시 생략·직전 반복 금지·상황별)·choose_filler, 데모 바지인 신호·첫 마디 체크박스, 가이드 S17. 새 테스트 29개 RED(ImportError) → GREEN, 전체 1078 passed. 첫 마디 정책은 기본 끔(session.start opener_adaptive) — 켜면 기존 단언 9개 변경 필요(결정 대기). 이슈: server 워밍업에 opener_phrases 미포함, config 필드 없음, 끊긴 답 프롬프트 규칙, 함수 실행 중 끼어들기
- 2026-11-20 | AG-T24 | DONE | agent/turn/jev_turn.py(말 끝 Noul 3개, 200ms 침묵에 자체 요청, DIRECTON_JEV_TURN=1), agent/handoff_jev.py(인계 Noul 6개·기준별 임계·영업시간 외 콜백, DIRECTON_JEV_HANDOFF=1), eval/jev_turn_eval.py·jev_handoff_eval.py, typesafe-sdk 0.7.5, docs/decisions/jev_turn.md·jev_handoff.md. 테스트 25개 RED(수집 오류) → GREEN, 전체 1079 passed. Jev 지연 p50 176 / p95 247ms. 말 끝: AUC 0.916(규칙 0.826)이나 표와 결합 시 false cut 1건 감소뿐 → 선택 사용. 인계: testset 16/16·오탐 0(LLM 10/16), 연습판 4/4 — 같은 문항으로 조정한 표본 안 수치. 세션 배선 미적용(제안 패치 문서화). Jev 호출 1,456회. 의도 Choice는 생략
- 2026-11-20 | AG-T25 | DONE | session.py SessionDeps.handoff(선택 필드, types.py 계약 변경 없음)·선호출과 동시에 Jev 인계 요청·확정 뒤 ≤0.3초 대기·인계면 힌트(동적부) 추가 후 힌트 없는 선호출 폐기·failures 카운트·detector.set_context, config 바지인·페이싱·opener_adaptive(기본 False)·jev_handoff_wait_s 필드(DIRECTON_*), server 워밍업에 첫 마디 문구·_make_handoff, prompts 끊긴 답 규칙, handoff_jev 힌트에 중복 사과 금지. test_wiring_t25 13 RED → 18 GREEN, 전체 1097 passed. 실측 5건(round1 LLM 누락분): 3건 transfer_to_agent 호출, 2건은 말 끝 전 확정으로 부분 텍스트(턴 결정기 문제)
- 2026-11-20 | AG-T21 | DONE | responder.py 조회 뒤 hedge(DIRECTON_POST_TOOL_HEDGE 기본 1)·결과 템플릿 첫 문장(DIRECTON_TOOL_TEMPLATE 기본 0, 켜면 기존 test_llm_eval 2개 단언과 충돌)·번호 자릿수 상대화(repair_thresholds)·번호 로그 redact, runner/score after_tool_ms·speed_official·total_official. test_after_tool 24개 RED → GREEN, 전체 1099 passed. 40문항 실측(B2 hedge vs C2 +템플릿): 조회 뒤 실제 답 p50 2.32→0.73s, 공식 속도 91.6→92.5(채움말 제외 83.7→90.5), judge 70.0→70.4, 함수 82.5→80.0. 이슈: 템플릿 바꿔 말한 반복 약 5/25, validate_number_length가 스키마 'N자리'에 의존, 페이싱 뒤 turn_end p50 4.4→10.8s, edge 타임아웃 시 Gemini TTS 429
- 2026-11-20 | AG-T15d | DONE | eval/synth.py 말투 지시 분리 프롬프트(STYLE_PROMPT_VERSION 3)·per_day 한도 판별, test_eval_tools 4개 추가, 오염 음성 5문항 재합성, docs/decisions/round2.md. 172문항 10회(A/A 포함): 잡음 폭 judge ±4·시스템 ±2.5. R2-best(템플릿+Jev 인계+첫 마디 정책) 시스템 79.8(÷5=16.0), 공식 속도(채움말 제외) 88.3, 조회 뒤 답 p50 0.94s, 인계 15/16 오탐 0. 템플릿 2.3→1.0s 유효, Jev 인계 재현율 13→15, Jev 말 끝 효과 없음, 첫 마디 정책 시스템 중립. 연습판 40문항 시스템 74.5. 전체 1103 passed. 이슈: TTS 대체 일일 한도, 4자리 번호 STT, 남은 실패 목록(13절)
- 2026-11-20 | AG-T26 | DONE | docs/09_결과보고서.html(선택 11개 trade-off 카드, 점수 흐름·조회 뒤 지연 차트, 당일 설정, 남은 실패, 연습판, 비용), artifact https://claude.ai/artifact/NgVkk6fBVUhNbW7sTQSxbb. 코드 변경 없음
- 2026-11-20 | AG-T28 | DONE | agent/responder.py(confirm_candidates·last_digits·digits_ko·confirm_message, _confirm: 막힌 호출·not_found → 고객 본인 주문 후보 1~2개 needs_confirmation, get_order_list 결과에서 상품명 보관, 스위치 DIRECTON_CONFIRM_ORDER_ID 기본 끔), tests/test_order_confirm.py 13개 RED ImportError → GREEN, 전체 1116 passed. 실서버 측정 안 함(당일 A/B 후보). 다음 턴에서 "네"를 받았을 때 이전 대화의 끝자리로 주문을 찾는 처리는 없음(LLM이 목록 조회로 찾아야 함)
- 2026-11-21 | AG-T29 | DONE | eval/soniox_endpoint_eval.py(새), tests/test_soniox_endpoint.py(새), docs/decisions/soniox_baze.md, .env 라우팅(커밋 안 함), 측정 도구를 먼저 쓰고 테스트 7개 추가(RED 단계 없음) 7 passed, 전체 1124 passed. STT 172문항: 숫자정규화 CER 0.065→0.040, 숫자 74.1→79.6%, finalize p50 397→238ms. 엔드포인트: Soniox 말 끝 전 확정 46.5%/27.3%(기존 10.1%/7.0%) → 기존 결정기 유지 권장(AD12 결정 대기). 남은 이슈: Soniox→Gemini 폴백 없음, E2E 재측정
- 2026-11-21 | AG-T30 | DONE | agent/turn/soniox_gate.py(새), eval/soniox_gate_eval.py(새), eval/soniox_endpoint_eval.py(토큰 이벤트 녹화, 쉼 여유 0.10), tests/test_soniox_gate.py(새 9개), tests/test_soniox_endpoint.py(이벤트 단언), data/turnset/jev_cache.jsonl(+438), docs/decisions/soniox_gate.md. 테스트 먼저 쓰지 않고 구현과 함께 작성, 토큰 기반 hold 실패 → VAD 기반으로 재설계 후 9 passed, 전체 1133 passed. 말 끝 전 확정 turnset 기존 10.1% / 규칙 25.4% / Jev 23.7% / 규칙+Jev 22.8%, 172문항 7.0% / 15.1% / 11.6% / 11.0%. 기존 결정기 유지 권장
- 2026-11-21 | AG-T31 | DONE | eval/soniox_endpoint_eval.py(일반 로더), eval/soniox_gate_eval.py(simulate_detector·VAD 캐시·--run), tests/test_soniox_gate.py(+2), data/turnset/jev_cache.jsonl(+238), docs/decisions/soniox_gate.md 6절. 18 passed, 전체 1135 passed. onedam 실제 끊음 wideband 기존 14 / 규칙 22 / Jev 13, phone 14 / 21 / 14, babble 2 / 12 / 3. Soniox 재생 기존 결정기 turnset 9.2%·172문항 6.4%(토큰 지연 영향 없음). STT 비교는 사용자 지시로 중단
- 2026-11-21 | AG-T32 | DONE | eval/soniox_gate_eval.py(Combo·simulate_detector Jev 결합·collect_jev_texts·--combos), tests/test_soniox_gate.py(+3), data/turnset/jev_cache.jsonl(+1676), docs/decisions/soniox_gate.md 7절. 14 passed, 전체 1138 passed. 5개 셋 실제 끊음 합계 기존 62 / AG-T24식 61 / 최근 응답 lo0.3 55 / lo0.5 51 / Soniox+Jev 104 / <end>·Jev 73. 남은 일 AG-T33, AD13 결정 대기
- 2026-11-21 | AG-T34 | DONE | eval/precall_jev_eval.py(새), tests/test_precall_jev.py(새 7개), docs/decisions/precall_jev.md. 테스트는 구현 뒤 작성, 7 passed, 전체 1145 passed. onedam wideband 99문항: 기존 확정 뒤 LLM 대기 434/484/894ms(평균/p50/p95), 새 τ0.5·σ0.5 404/468/894(말하는 중 재사용 6%, 오답 0), 정답 판정 상한 353/403/798(16%), 판정 없음 324/336/798(오답 6). Jev 458회·LLM 227회·심사 32쌍(Claude 서브에이전트). 세션 미적용 권장
- 2026-11-21 | AG-T34 | 후속 | eval/precall_jev_eval.py(same_first_step·툴만·툴 목록 변형·벌점 포함 대기), tests/test_precall_jev.py(+2, 9 passed), docs/decisions/precall_jev.md 5절. τ0.5 첫 조회·툴만 σ0.5: 말하는 중 재사용 14%, 대기 352/403/798ms(벌점 포함 376), 잘못 재사용 3(첫 조회 선택, 텍스트 오답 0). 툴 쌍 AUC same_reply 0.73 / same_first_step 0.45 / +툴 목록 0.37. Jev +49회. 미적용 유지
- 2026-11-21 | AG-T34 | DONE | eval/soniox_endpoint_eval.py(글 없는 history 제외), data/turnset/jev_cache.jsonl(+479, 해시·확률만), docs/decisions/soniox_gate.md 8절. 공식 연습 80: A 끊음 1·miss 3·p95 1325, B(lo 0.5 최근 응답) 1·3·1365, C 3·29(36%)·1116. 데이터는 gitignore, 문서에 원문 없음
- 2026-11-21 | AG-T33 | DONE | agent/turn/jev_turn.py(match latest·SONIOX_POLICY lo 0.5·요청 6), agent/turn/detector.py(DIRECTON_JEV_TURN=1이면 Soniox 설정), tests/test_jev_turn.py(+3). 3개 RED → GREEN 19 passed, 전체 1150 passed. 켜려면 DIRECTON_JEV_TURN=1, Jev 사용 허가는 주최 측 확인 대기
- 2026-11-21 | AG-T36 | DONE | mock_api/server.py(공식 규격 Store: 조회 표기 무시·둘 다>주문번호>전화 순·not_found/invalid, 접수 required→invalid·번호 맞으면 기록 결과·없으면 ok, 검색 최근접, 404, 기본 0.0.0.0·--db), mock_api/data/tools.json(공식 8개 사본), start_mock.sh, 옛 쇼핑몰 → mock_api/legacy_shop.py + data/legacy_tools.json(테스트·eval import만 변경, 단언 그대로), tests/test_mock_api.py(합성 DB 20개). RED ImportError → GREEN 20 passed, 전체 1113 passed(옛 mock 테스트 제거분 감소). 공식 mock_db 76기록 자기 일치 73(중복 인자 3)
- 2026-11-21 | AG-T36 | 수정 | mock_api/server.py: 404 거부 전에 body를 읽는다(전체 회귀에서 test_unknown_function_is_404가 WinError 10053로 간헐 실패)
- 2026-11-21 | AG-T37 | DONE | agent/prompts.py(history tool_call → "(상담원이 함수 호출) 함수 {값 있는 인자}", tool_result → "(조회 결과) 함수 → 압축 JSON"(2,000자 상한), 번호 요약은 함수 인자 키로 전화/주문 판단, 직전 상담원 말은 함수 기록 건너뜀, 조회 결과 재사용 규칙), tests/test_prompts.py(+6). 4개 RED → GREEN 44 passed, 전체 1119 passed. 세션 `_context_history`(Jev·결정기 문맥)는 text 없는 함수 항목을 그대로 넘긴다 → AG-T38에서 확인
- 2026-11-21 | AG-T35 | DONE | agent/turn/rules.py(PARTICLE_NOUN_SUFFIXES·TOPIC_QUESTION_SUFFIXES), tests/test_turn.py(+15), docs/decisions/soniox_gate.md 8절. 8개 RED → GREEN, turn 176 passed, 전체 1134 passed. 6개 셋 재생: 공식 miss 3 → 0, 나머지 변화 없음
- 2026-11-21 | AG-T38 | DONE | (223de83 진행 중 커밋 + 마무리) agent/responder.py(number_formats: 'N자리' 없으면 전화 11·주문 15/16, 하이픈 무시 `_num`, 교정·되묻기 `get_order_info(buyer_cellphone)`+한글 키 `orders[].주문번호/상품명`, 자동 연결 안전망·promises_transfer 삭제, 템플릿 단일 주문 `배송상태`), agent/config.py(filler_skip_tools=(), ensure_transfer_call 삭제), agent/normalize.py(order_number→id), agent/catalog.py(기본 파일 없음, 기본 주문번호 {15,16}, LEGACY_CATALOG_PATH), agent/handoff_jev.py(인계 안내는 함수 없이, 전화 문의 방법), agent/turn/jev_turn.py·handoff_jev.build_state(함수 기록 제외), mock_api/legacy_shop.py(404 전 body 읽기), tests/test_official_tools.py(16개) + 기존 단언 갱신(9절). 10개 RED → GREEN, 전체 1147 passed
- 2026-11-21 | AG-T39 | DONE | agent/prompts.py(고정부 하루온·공식 8개 함수 규칙·인계는 말로+전화 문의 방법·결과 status ok/already_requested/not_found/invalid/error, load_brand: scenarios.md·information.json(링크 제거)·style.json을 고정부에, 영업시간 평일 10~17시 점심 12~13시, `now` "(수)" 형식 파싱, caller 줄 공식 인자, 되읽기 예시 15자리), agent/normalize.py(15자리 주문번호 4-4-3-4 묶음), agent/config.py(manual_path 기본 ""), agent/handoff_jev.py(영업시간 10~17·공식 now 형식), tests/test_brand_prompt.py(14개) + 옛 프롬프트 단언 갱신(test_prompts 8·test_dayof_swap 3·test_reprompt_policy 1). RED ImportError → GREEN, 전체 1161 passed. 실제 시스템 프롬프트 22,222자(고정부 21,749)
- 2026-11-21 | AG-T40 | DONE | 공식 run_eval/score로 연습 80문항 첫 측정(코드 변경 없음). 서버: BAZE 경유 Soniox STT + gpt-5.4-mini(reasoning none), TTS edge, DIRECTON_JEV_TURN=1, 결과 runs/official_baze_gpt54mini_merged(첫 실행 + TTS 대체 429 실패 21건 재실행 병합). 공식 score.py --no-judge: 함수 70.0 / 속도 99.1(첫 소리 p50 1.02s·p90 1.35s, 조회 뒤 p50 2.27s·p90 2.79s). 응대는 공식 judge 대신 docs/13 A~F 기준 서브에이전트 채점: 67.5(A 1.12 B 1.77 C 1.46 D 0.91 E 1.49 F 1.66, fatal 6, 40점 이하 20건, 100점 8건). 참고 총점(응대×0.5+함수×0.2+속도×0.3, 응대는 공식 judge 아님) 77.5. 이슈는 13절
- 2026-11-21 | AG-T41 | DONE | agent/responder.py `_repair`: 자리 수가 맞아도 발신번호와 거의 같은(`_close_enough`, 11자리면 공통 9자리 이상) 전화번호는 발신번호로 교정 (무작위 010 번호끼리 오교정 0.05%), tests/test_number_recovery.py. 3개 RED → GREEN
- 2026-11-21 | AG-T42 | DONE | agent/responder.py `_start_filler`(모든 호출이 자리 수 검사에 막히면 필러 없음), `_length_error` 안내문(자리 수 규칙 설명 금지, 고객이 직접 물은 정책만 한 문장, 묻지 않은 안내 금지), agent/prompts.py invalid_arguments 규칙 같은 내용, tests/test_number_recovery.py + 기존 필러 테스트 입력 번호 갱신(9절). 4개 RED → GREEN, 전체 1172 passed. 재측정 전(AG-T41·T42 효과 미측정)
- 2026-11-21 | AG-T44 | DONE | agent/providers/factory.py(OpenAI 기본이면 Gemini 폴백 없음·같은 모델 사본 hedge, check_llm_policy·LLM_ALLOWED_MODELS), agent/server.py(serve 시작 검사), tests/test_llm_policy.py 20개 RED → 23 GREEN, test_providers 단언 1개 갱신(9절). 전체 1195 passed(스테이징본 기준, 다른 세션의 AG-T43 진행 중 파일 제외). AD2·AD10 폐기, AD17 추가, .env LLM_PROVIDER=openai
- 2026-11-21 | AG-T43 | DONE | agent/prompts.py(number_candidates·candidates_line, 이번 통화 '번호 후보' 줄, _text_numbers로 history_summary 번호 추출 공유), agent/responder.py(number_candidates_from_system, classify_number 강함·약함·없음, _pool, _repair 강함이면 후보로 교체, _candidate_confirm 약함이면 needs_confirmation, DIRECTON_NUMBER_CANDIDATES 기본 켬), tests/test_number_candidates.py(새 20개 RED ImportError → GREEN), 기존 테스트 3개 스위치 끔 고정(9절), 전체 1215 passed. 공식 측정기 번호 문항 25개(runs/number_cases.jsonl, gpt-5.4-mini·Soniox·DIRECTON_JEV_TURN=1, 동시 2) 같은 코드 스위치 끔/켬: 함수 44.0 → 48.0(11 → 12건, 차이는 P0010-11 한 건으로 LLM 편차 범위), 속도 100.0 / 99.6. 약한 확인은 한 번도 발동하지 않음(실패 13건 중 12건이 후보 0개, 13절). 결과 runs/t43_num_on·t43_num_off
- 2026-11-21 | AG-T45 | DONE | agent/tool_guard.py(새: fix_full_return·is_claim·is_leak·parse_leaked_calls·SpeechGuard), agent/responder.py(guard 배선 몇 줄: 문장 송신 전 filter/finish, 툴 실행 전 fix_full_returns, 결과 뒤 note_results, 다시 생성 1회분 회차, 대체 문장), agent/prompts.py(반품 규칙 기본 return_order·상세주문번호, 호출 형식·JSON 글자로 쓰지 않기), tests/test_tool_guard.py 49개. RED(ImportError, 이후 배선 전 8개 실패) → GREEN, 전체 1244 passed(test_stt_eval 타이밍 테스트 1개 간헐 실패, 단독·재실행 통과). 연습 문항 sanity: 주문 정보가 history에 있는 반품 7문항에서 전체 호출을 가정하면 return_order 정답 6건은 변환, P0147-07(일부반품)은 그대로. 재측정 전
- 2026-11-21 | AG-T46 | DONE | eval/number_recheck_eval.py(새, 실제 API 실험 스크립트, 자동 테스트 없음). 번호 문항 24개(정답 번호 있는 것) × 6변형 Soniox 재인식: 정답 일치 base·ctx·crop·pad·slow·slow2 모두 5/24(1차에서 맞은 5건), 추가로 살린 문항 0. 효과가 없어 DigitRecheckSTT 연결·공식 측정은 하지 않음(13절). 결과 runs/t45_offline*.jsonl
- 2026-11-21 | AG-T45 | 측정 | 3차 공식 측정(커밋 17380d6, AG-T43·T44·T45 반영, runs/official_t45, 환경 실패 0건) — 함수 78.8(63/80, 2차 56), 속도 99.4, 응대(docs/13 기준) 74.9, 참고 총점 83.0(2차 80.3). AG-T45 목표: 전체 반품 4/5, 말로만 접수 3/4, 호출 문자열 송출 0, 거짓 접수 0. 즉시 0점 5건 중 4건 사실 지어내기. docs/14 1·2·3절 3차 열, 6절(3차) 신설, 7·8절 갱신
- 2026-11-21 | AG-T48 | DONE | agent/grounding.py(새: 사실 근거 규칙·결제 수단별 환불 정리), agent/prompts.py(build_system 훅 3줄), agent/tool_guard.py(is_lookup_claim·LOOKUP_NOTE, SpeechGuard 조회 기록 추적·주장 종류별 교정 지시), tests/test_grounding.py 27개. RED(ImportError) → GREEN, 전체 1286 passed. 재측정 전
- 2026-11-21 | AG-T49 | DONE | agent/responder.py(_caller_fallback 막힌 조회 → 발신번호 조회·자동 조회 결과 재사용, _run_tools 같은 회차 skipped, _repair 빈 조회 발신번호 채움·틀린 선택 인자 버림, _length_error 빈 조회 차단, 이미 물은 회차 턴 끝), agent/tts_cache.py(_garbled_number: 형식 없는 번호면 조회 첫 마디 안 씀), agent/prompts.py(history_summary 결과 없던 번호 줄), tests/test_number_reask.py 24개(RED 16 → GREEN), test_session_precall 장난감 번호 1개 갱신(9절). 전체 1283 passed, 8 skipped. 재측정 전
- 2026-11-21 | AG-T50 | DONE | agent/providers/openai_llm.py(reasoning_effort 검증·기본 medium, include_reasoning·reasoning_store, encrypted reasoning item 보관·되돌리기, reasoning 이벤트 무시), agent/providers/factory.py(OPENAI_REASONING_EFFORT·OPENAI_REASONING_INCLUDE, REASONING_TIMING·llm_timing — medium 기본 hedge 3000ms·첫 이벤트 6.0s, none·Gemini는 500ms·2.5s 그대로, 사본 상속·저장소 공유), tests/test_reasoning_t50.py(새 23개), tests/test_providers.py(effort 단언 medium). 실측 전 — 타이밍 잠정
- 2026-11-21 | AG-T50 | 측정 | 4차 공식 측정(커밋 29e7990, AG-T48·T49·T50 반영, LLM 직접 키 + DIRECTON_SIM_LLM_DELAY_S=0.54, TTS edge, runs/official_t50_none·official_t50_medium, 환경 실패 0건) — none/medium: 응대(docs/13 기준) 78.2/82.9, 함수 81.2/81.2(65/80), 속도 99.4/97.9, 참고 총점 85.2/87.0. 조회 뒤 답 p50 1.59/2.67초, 3.5초 초과 0/9건, 즉시 0점 3/1건. 기본 medium 유지, low 측정 진행 중. docs/14 1절 4차 열, 7절(4차) 신설, 8·9절 번호 이동·결과 위치 갱신
- 2026-11-21 | AG-T51 | DONE | 4차 low 측정(runs/official_t50_low, 함수 78.8·속도 99.3·응대 82.9·참고 총점 86.99, 조회 뒤 3.5초 초과 1건) 기록, agent/providers/openai_llm.py 기본 effort medium → low, tests 단언 5개 low로(RED 5 → GREEN), docs/14 7절에 low 열·결정 추가, AD19
