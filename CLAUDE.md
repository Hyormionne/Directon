# Directon — DIRECT:ON 2026 한국어 음성 CS 에이전트

설계 근거는 `docs/01_구현계획.md`, 대회 규격은 `docs/00_대회규격_분석.md`. 모든 기록은 이 폴더 안에 둔다.

## 현황

| 트랙 | 문서 | Task 접두사 | 결정 접두사 | 상태 |
|---|---|---|---|---|
| AGENT | `docs/AGENT_PLAN.md` | `AG-` | `AD` | 진행 중 |

완료된 트랙 문서는 `docs/archive/`로 옮긴다.

## 명령

- 의존성: `uv sync` (Python 3.12, `.venv`)
- 테스트: `uv run pytest -q` (외부 API를 부르는 테스트는 `@pytest.mark.live`, 기본 실행에서 제외. `uv run pytest -m live`로 따로 실행)
- 상담원 서버: `uv run python -m agent.server --port 9000`
- Mock API: `uv run python -m mock_api.server --port 8100`
- 측정: `uv run python -m eval.runner ...` (AGENT 트랙에서 정의)

## 불변 규칙

- 모듈 간 계약은 `agent/core/types.py`. 시그니처를 바꾸려면 트랙 문서에 먼저 적는다.
- PCM은 항상 16kHz s16le mono. 다른 샘플레이트는 경계에서 변환한다.
- 측정기 프로토콜(`DIRECT_ON 2026 Starter Kit/docs/SPEC_STARTER.md`)을 깨지 않는다: `session.start` → PCM → (우리) PCM + `agent.transcript` + `turn.end`.
- 고객 발화가 끝나기 전에는 유성음을 보내지 않는다(속도 0점 조건).
- 함수 호출에는 항상 `X-Session-Id` 헤더를 붙인다.
- `DIRECT_ON 2026 Starter Kit/`는 원본이다. 수정하지 않는다.
- API 키는 환경변수(`GEMINI_API_KEY`, `SONIOX_API_KEY`, `OPENAI_API_KEY`, `TYPESAFE_API_KEY`, `ELEVENLABS_API_KEY`) 또는 `.env`에서만 읽는다. 코드·로그·문서에 키를 쓰지 않는다.

## 외부 서비스 fake

자동 테스트(`live` 아님)에서 실제 네트워크 호출 금지. Provider는 생성자에 클라이언트/전송 함수를 주입받게 만들고, 테스트에서는 fake를 넣는다.

| 서비스 | 위치 | fake 방법 |
|---|---|---|
| Gemini (STT/TTS/LLM) | `agent/providers/gemini_*.py` | 주입한 fake client |
| Soniox | `agent/providers/soniox_stt.py` | fake websocket |
| OpenAI | `agent/providers/openai_llm.py` | fake client |
| Mock API | `agent/tools_client.py` | `httpx.MockTransport` |

## 커밋

- 형식: `<type>(<scope>): <Task ID> <요약>` — type: feat, fix, test, docs, chore, perf, refactor / scope: agent, turn, providers, mock, eval, docs
- Task당 커밋 1개, `git add`는 경로 지정. push는 요청 시에만.
- 사용자 작업은 `AG-U*`.
