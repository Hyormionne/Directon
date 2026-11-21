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
- (AG-T29, AD11) 실제 운영 STT는 **BAZE 경유 Soniox `stt-rt-v6`**(`.env`의 `STT_PROVIDER=soniox`, 직접 키는 잔액 소진으로 주석). LLM은 `LLM_PROVIDER=gemini`로 AD10 유지. 엔드포인팅은 기존 결정기 유지, Soniox `<end>`는 끔(AD12, `docs/decisions/soniox_baze.md`).
- LLM 기본값: `OPENAI_API_KEY`가 있으면 `gpt-5.4-mini`, 없으면 Gemini Flash 계열 중 0단계 실측에서 가장 빠른 모델.
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
| AD2 | 개발용 LLM | A: Gemini Flash 계열 / B: OpenAI 키 대기 | **A.** 같은 이유. 실측으로 모델 선택 | **A** (2026-11-19 사용자 확정) |
| AD3 | 테스트 음성 합성 | A: Gemini TTS 여러 음성 + edge-tts(무료, 키 불필요) / B: Gemini만 | **A.** 상담원 음성(Gemini)과 다른 엔진이 섞여야 과적합을 피한다. 가상 문장만 보낸다 | **A** (2026-11-19 사용자 확정) |
| AD4 | 가상 쇼핑몰 함수 | A: 공식 미공개이므로 쇼핑몰 함수 14개를 가정해 구현 / B: 데모 2개만 | **A.** 공식 공개 시 데이터만 교체 | **A** (2026-11-19 사용자 확정) |
| AD6 | 상담원 TTS(Gemini TTS 무료 하루 10회) | A: Gemini 결제 연결 / B: 개발 중 edge-tts 대체 / C: 다른 TTS 키 | **A.** 비용 약 $5 | **B** (2026-11-19 사용자 확정) |
| AD7 | 테스트 음성 화자 | A: edge 3종+Heami에 피치·속도 변형 / B: 결제 연결 후 Gemini 음성 사용 | **A** | **B** (2026-11-19 사용자 확정) — T05b는 AG-U04 이후. T12 보정셋은 edge+피치로 먼저 만들고 Gemini 음성은 결제 후 추가 |
| AD8 | 잡음 VAD 래퍼 기본값 | A: 끔 / B: 켬 / C: 결정기 재보정 후 | **A.** 잘못 끊기(0점) 손해가 말 끝 놓침(80~90점)보다 큼 | **A** (2026-11-19 사용자 확정) — `DIRECTON_VAD_NOISE_GATE=1`로 켬 |
| AD9 | BAZE 게이트웨이 범위 (2026-11-19 사용자 요청) | A: Claude Code만 / B: 프로젝트 provider만 / C: 둘 다. TTS: 기본 edge 유지+baze 선택 / baze 기본 / 안 함. STT 폴백: Soniox만 / gemini-3.5-flash 일괄 | — | **B**, TTS는 edge 기본+`baze` 선택, STT 폴백 없음 (2026-11-19 사용자 확정) |
| AD10 | BAZE 크레딧 소진 → 서비스 선택 | — | — | **Gemini 직접 키로 복귀** (2026-11-19 사용자 지시). `.env`의 `BAZE_API_KEY` 주석 처리(값 보존). 다시 켜려면 주석 해제 |
| AD11 | STT 기본 (2026-11-21 사용자 요청: "BAZE에 SONIOX가 있어. 그걸로 한번 도입해봐") | A: BAZE 경유 Soniox / B: Gemini live 유지 | **A.** 172문항 숫자 정규화 CER 0.065→0.040, 숫자 74.1→79.6%, finalize p50 397→238ms | **A** (2026-11-21 사용자 지시) |
| AD12 | 엔드포인팅 (2026-11-21 사용자 요청: "엔드포인팅도 기존 방안과 SONIOX의 방안을 비교해줘") | A: 기존 결정기 유지(Soniox `<end>` 끔) / B: Soniox `<end>`로 교체 | **A.** Soniox는 지연 p50 약 0.5초 빠르지만 말 끝 전 확정 turnset 10.1→46.5%, 172문항 7.0→27.3%(0점 조건). AG-T30 규칙/Jev 게이트를 붙여도 22.8~25.4% / 11.0~15.1%(`soniox_gate.md`) | |
| AD13 | 엔드포인팅 결합 (AG-T32) | A: 기존 결정기만 / B: 기존 + Jev 늘리기(최근 응답 규칙, lo 0.3~0.5, 스위치) / C: Soniox + Jev 게이트 | **B.** 5개 셋 실제 끊음 62 → 51(lo 0.5) / 55(lo 0.3), miss +2~4, p95 거의 그대로. 늘리기만이라 Jev 실패 시 기존 그대로. 단 외부 API 의존 | |
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
| AG-T33 | `JevTurnDetector` 최근 응답 규칙(텍스트 바뀌어도 이번 침묵의 최근 p, 요청 상한 6) + 늘리기 lo 결정 + 다른 데이터 확인 | AG-T32, AD13 | TODO | |
| AG-T34 | 말하는 중 Jev 선호출(주기 tick `meaning_will_change` ≤ τ면 선호출, 확정 때 Jev `same_reply`로 정합성 판정) vs 기존 침묵 200ms 선호출 — onedam 재생 비교 (2026-11-21 사용자 요청) | AG-T31, AG-T10c | DONE | 2026-11-21 |
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
- (AG-T34) Jev `same_reply`(응답 전체가 같은가)는 prefetch에 필요한 "첫 행동이 같은가"보다 엄격하다(AUC 0.75, 놓침 10/16). 뒤에 질문이 붙어도 첫 조회는 같은 경우를 거절한다. 다시 본다면 첫 조회 기준 질문 + 선호출 툴 호출을 상태에 넣기(상한 평균 −81ms).

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
