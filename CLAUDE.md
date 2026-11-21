# Directon — DIRECT:ON 2026 한국어 음성 CS 에이전트

공식 규격과 우리 대응은 `docs/12_공식자료_분석.md`(공식 키트 원본은 `DIRECT_ON 2026 Starter Kit/`). `docs/00`~`11`은 공식 자료 공개 전 기록이다. 모든 기록은 이 폴더 안에 둔다.

## 현황

| 트랙 | 문서 | Task 접두사 | 결정 접두사 | 상태 |
|---|---|---|---|---|
| AGENT | `docs/AGENT_PLAN.md` | `AG-` | `AD` | 진행 중 |

완료된 트랙 문서는 `docs/archive/`로 옮긴다.

## 명령

- 의존성: `uv sync` (개발은 Python 3.12 `.venv`. 제출 환경은 Python 3.11 + `pip install -r requirements.txt`라 3.11에서 도는 문법만 쓴다)
- 테스트: `uv run pytest -q` (외부 API를 부르는 테스트는 `@pytest.mark.live`, 기본 실행에서 제외. `uv run pytest -m live`로 따로 실행)
- 상담원 서버: `uv run python -m agent.server --port 9000`
- Mock API: `uv run python -m mock_api.server --port 8100` (기본 데이터는 공식 키트의 `data/mock_db.json`, 바꾸려면 `--db`)
- 측정: 공식 측정기·채점기를 원본 그대로 실행한다. `cd "DIRECT_ON 2026 Starter Kit/eval"` 뒤 `python run_eval.py --server ws://localhost:9000 --mock http://localhost:8100 --cases ../data/practice_cases.jsonl --audio ../data/audio/practice --out ../../runs/official/<이름>`, 이어서 `python score.py --run … --cases …`(응대 채점은 `OPENAI_API_KEY` 필요, 없으면 `--no-judge`)
- 제출 묶음: `bash scripts/package.sh` → `dist/directon-submit.zip` (맨 위에 `start_server.sh`, `start_mock.sh`, `requirements.txt`, `.env`)

## 불변 규칙

- 모듈 간 계약은 `agent/core/types.py`. 시그니처를 바꾸려면 트랙 문서에 먼저 적는다.
- PCM은 항상 16kHz s16le mono. 다른 샘플레이트는 경계에서 변환한다.
- 측정기 프로토콜(`DIRECT_ON 2026 Starter Kit/SPEC.md`)을 깨지 않는다: `session.start` → PCM → (우리) PCM + `agent.transcript` + `turn.end`.
- 고객 발화가 끝나기 전에는 유성음을 보내지 않는다(속도 0점 조건).
- 함수 호출에는 항상 `X-Session-Id` 헤더를 붙인다.
- `DIRECT_ON 2026 Starter Kit/`는 공식 키트 원본(11/21 현장 공개본)이다. 수정하지 않는다. 실제 상담 데이터가 들어 있어 `.gitignore` 대상이다(커밋·외부 전송 금지).
- 모델은 행사 지정 모델만 쓴다: STT Soniox `stt-rt-v6`, LLM GPT-5.4 mini, TTS Gemini TTS. 조립은 `agent/providers/factory.py`.
- API 키는 환경변수(`BAZE_API_KEY` 대회 게이트웨이, `OPENAI_API_KEY`·`SONIOX_API_KEY`·`GEMINI_API_KEY` 직접 키, `TYPESAFE_API_KEY`) 또는 `.env`에서만 읽는다. 코드·로그·문서에 키를 쓰지 않는다. 제출 묶음의 `.env`에는 대회 키만 넣는다(`scripts/package.sh`가 걸러 낸다).
- 시험 음성·고객 발화를 저장하지 않는다. 서버는 기본으로 `DIRECTON_NO_AUDIO_LOG=1`이다.

## 외부 서비스 fake

자동 테스트(`live` 아님)에서 실제 네트워크 호출 금지. Provider는 생성자에 클라이언트/전송 함수를 주입받게 만들고, 테스트에서는 fake를 넣는다.

| 서비스 | 위치 | fake 방법 |
|---|---|---|
| Soniox STT | `agent/providers/soniox_stt.py` | fake websocket |
| OpenAI LLM | `agent/providers/openai_llm.py` | fake client |
| Gemini TTS | `agent/providers/gemini_tts.py`(직접), `baze_tts.py`(게이트웨이) | 주입한 fake client, `httpx.MockTransport` |
| Mock API | `agent/tools_client.py` | `httpx.MockTransport` |

## 커밋

- 형식: `<type>(<scope>): <Task ID> <요약>` — type: feat, fix, test, docs, chore, perf, refactor / scope: agent, turn, providers, mock, eval, docs
- Task당 커밋 1개, `git add`는 경로 지정. push는 요청 시에만.
- 사용자 작업은 `AG-U*`.
