# Directon — DIRECT:ON 2026 한국어 음성 상담원 (제출 묶음)

## 구성

| 경로 | 내용 |
|---|---|
| `start_server.sh` | 상담 서버 시작(WebSocket, 9000번, `0.0.0.0`) |
| `start_mock.sh` | 모의 함수 서버 시작(HTTP, 8100번, `0.0.0.0`). 첫 인자가 데이터 파일 |
| `requirements.txt` | 실행에 필요한 패키지(Python 3.11) |
| `.env` | API 키(`이름=값`). 상담 서버가 시작할 때 읽는다 |
| `agent/` | 상담 서버 코드(턴 감지, STT·LLM·TTS 연결, 함수 호출 검사) |
| `mock_api/` | 모의 함수 서버 코드 |
| `models/silero_vad.onnx` | 음성 구간 감지 모델(CPU에서 돈다) |
| `data/brand/` | `tools.json`, `scenarios.md`, `information.json`, `style.json`(공식 키트 그대로) |
| `data/mock_db.json` | 연습용 모의 함수 데이터(공식 키트 그대로). `start_mock.sh`에 인자가 없을 때만 쓴다 |
| `assets/tts_cache/` | 고정 문구(첫 마디, 안내 문구) 합성 음성. 없는 문구는 서버 시작 때 합성한다 |

실행 중에 생기는 것: `runs/logs/turns-YYYYMMDD.jsonl`(턴 기록), `assets/tts_cache/*.pcm`(새로 합성한 고정 문구).

## 설치와 시작

```bash
pip install -r requirements.txt
bash start_mock.sh <모의 함수 데이터 파일>   # 8100번. 인자가 없으면 data/mock_db.json
bash start_server.sh                          # 9000번
```

- 두 스크립트는 자기 폴더로 이동한 뒤 `python3 -m mock_api.server`, `python3 -m agent.server`를 실행한다. 데이터 파일은 절대 경로나 이 폴더 기준 경로로 준다.
- 상담 서버는 예열(STT 연결 1회, LLM 요청 2회, 캐시에 없는 고정 문구 합성)을 마친 뒤 포트를 연다. 예열 상한은 25초다. 로그에 `agent server on :9000`이 나오면 준비가 끝난 것이다.
- 모의 함수 서버는 시작하면 `mock tools on http://0.0.0.0:8100 functions=8 records=N`을 찍는다. 함수 정의는 `data/brand/tools.json`에서 읽는다.
- 상담 서버는 `session.start`로 받은 `tools_url`로만 함수를 부른다(주소를 코드에 고정하지 않았다). 모든 호출에 `X-Session-Id`를 붙인다.
- ffmpeg, GPU는 쓰지 않는다.

## 환경 변수

상담 서버는 `.env`를 읽되, 프로세스에 이미 있는 환경 변수가 우선한다.

| 이름 | 필수 | 기본값 | 뜻 |
|---|---|---|---|
| `BAZE_API_KEY` | 필수 | 없음 | 대회 게이트웨이 키. STT·LLM·TTS를 모두 이 키로 부른다 |
| `BAZE_BASE_URL` | 선택 | `https://api.baze.co/v1/gateway` | 게이트웨이 주소 |
| `SONIOX_API_KEY` | 선택 | 없음 | 있으면 STT를 Soniox에 직접 연결한다 |
| `OPENAI_API_KEY` | 선택 | 없음 | 있으면 LLM을 OpenAI에 직접 부른다 |
| `GEMINI_API_KEY` | 선택 | 없음 | 있으면 TTS를 Gemini API에 직접 부른다(모델 기본값이 `gemini-3.8-flash-lite-tts`로 바뀐다) |
| `SONIOX_MODEL` | 선택 | `stt-rt-v6` | STT 모델 |
| `STT_LANGUAGE` | 선택 | `ko-KR` | STT 언어 힌트 |
| `OPENAI_MODEL` | 선택 | `gpt-5.4-mini` | LLM 모델 |
| `OPENAI_REASONING_EFFORT` | 선택 | `medium` | LLM 추론 수준(`none`, `low`, `medium`) |
| `DIRECTON_TTS_PROVIDER` | 선택 | `auto` | `auto`(GEMINI_API_KEY가 있으면 직접, 없으면 게이트웨이), `gemini`, `baze`, `off`(신호음만 내는 개발용. 제출 묶음에는 넣지 않는다) |
| `GEMINI_TTS_MODEL` | 선택 | 게이트웨이 `gemini-3.8-flash-tts`, 직접 `gemini-3.8-flash-lite-tts` | TTS 모델 |
| `GEMINI_TTS_VOICE` | 선택 | `Kore` | TTS 음성 |
| `GEMINI_TTS_STYLE` | 선택 | 없음 | TTS 말투 지시문 |
| `DIRECTON_PRELOAD_ORDERS` | 선택 | `0` | `1`이면 연결 직후 발신자 번호로 주문을 미리 조회한다 |
| `DIRECTON_ORDER_DIGITS` | 선택 | `15` | 주문번호 자릿수(말 끝 판단용). `.env`가 아니라 프로세스 환경 변수로 줘야 적용된다 |
| `DIRECTON_NO_AUDIO_LOG` | 선택 | `1`(상담 서버로 띄울 때) | `1`이면 턴 기록에서 고객 발화 원문을 지운다 |
| `DIRECTON_KNOWLEDGE_DIR` | 선택 | `data/brand` | 시나리오·브랜드 정보·말투·함수 정의 폴더 |
| `DIRECTON_LOG_DIR` | 선택 | `runs/logs` | 턴 기록 폴더 |
| `DIRECTON_TTS_CACHE_DIR` | 선택 | `assets/tts_cache` | 고정 문구 음성 캐시 폴더 |

`agent/config.py`의 `Settings` 필드는 모두 `DIRECTON_<필드 이름 대문자>`로 덮어쓸 수 있다. 제출 묶음의 `.env`에는 `BAZE_API_KEY`만 들어 있고, 나머지는 기본값으로 돈다.

## 쓰는 모델

| 단계 | 모델 | 연결 |
|---|---|---|
| STT | Soniox `stt-rt-v6`(실시간) | 게이트웨이 `/soniox/transcribe-websocket` |
| LLM | OpenAI `gpt-5.4-mini`(Responses API, 추론 수준 medium) | 게이트웨이 `/responses` |
| TTS | Gemini `gemini-3.8-flash-tts`, 음성 Kore | 게이트웨이 `/audio/speech/` |
| 음성 구간 감지 | Silero VAD(`models/silero_vad.onnx`) | 로컬(CPU) |

TTS 지정 모델은 `gemini-3.8-flash-lite-tts`지만 게이트웨이에 이 모델이 없어(2026-11-21 확인) 게이트웨이에 있는 `gemini-3.8-flash-tts`를 쓴다. `GEMINI_API_KEY`를 주면 Gemini API를 직접 불러 `gemini-3.8-flash-lite-tts`를 쓴다.

## 시험 음성과 고객 발화

- 받은 음성(PCM)을 파일로 저장하지 않는다. 음성은 STT로만 보낸다.
- 턴 기록(`runs/logs/turns-*.jsonl`)에는 시간 지표, 상담원 문장, 함수 호출 기록만 남는다. 고객 발화 원문은 `DIRECTON_NO_AUDIO_LOG=1`(기본)일 때 지우고 쓴다.
- `assets/tts_cache/`에 저장되는 것은 상담원의 고정 문구 합성 음성뿐이다.

## 문제가 생기면

| 증상 | 볼 곳 |
|---|---|
| 상담 서버가 바로 끝난다 | 표준 오류의 마지막 줄. `… 키가 없습니다: … BAZE_API_KEY`면 `.env`를 확인한다. `ModuleNotFoundError`면 `pip install -r requirements.txt`를 다시 한다 |
| 9000번이 늦게 열린다 | 예열 중이다(상한 25초). `warmup: {…}` 줄의 `llm_ok`, `stt_ok`, `cache_missing`을 본다 |
| 답이 사과 문구만 나온다 | 로그의 `llm`, `tts`, `stt` 경고(게이트웨이 429, 키, 크레딧)와 `runs/logs/turns-*.jsonl`의 해당 턴 |
| 모의 함수 서버가 끝난다(종료 코드 2) | `모의 함수 서버를 시작할 수 없습니다: …` 줄의 파일 경로 |
| 모의 함수 서버 확인 | `GET http://localhost:8100/tools`가 함수 8개를 돌려준다 |
