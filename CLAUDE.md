# Directon — DIRECT:ON 2026 음성 AI 상담원 (476ssy 포크)

대회 규칙·채점·데이터 분석은 상위 폴더 `../CLAUDE.md`에 있다(이 레포에 올리지 않음). 세션 인수인계는 `../04_내작업/handoff.md`. 이 파일에는 이 코드에만 해당하는 것만 둔다. 우선순위는 루트 `todo.md`(git 제외).

## 공식 자료

- 규격 기준은 11/21 공식 자료 `../02_대회자료/directon-2026-voice-cs/`(`SPEC.md`·`SCORING.md`·`DATA.md`). Starter Kit은 참고용.
- 공식 자료는 레포 안에서는 `official/`에만 둔다(`.gitignore`). 데이터·음성·분석 결과를 커밋·push하지 않는다(대회 밖 반출 금지).
- 모델은 지정된 것만 쓴다: STT `stt-rt-v6`(Soniox), TTS `gemini-3.8-flash-tts`(Q&A Q8: lite 아님), LLM `gpt-5.4-mini`. 다른 모델로 폴백하지 않는다.

## 명령

- 의존성: `uv sync` (`.venv`)
- 테스트: `uv run pytest -q` (외부 API를 부르는 테스트는 `@pytest.mark.live`, 기본 실행에서 제외. `uv run pytest -m live`로 따로 실행)
- 상담원 서버: `uv run python -m agent.server --port 9000`
- Mock API: `uv run python -m mock_api.server --port 8100`
- 공식 매뉴얼 생성: `python scripts/build_manual.py` → `official/manual.md`·`style.md`. 서버는 `DIRECTON_MANUAL`·`DIRECTON_STYLE`·`DIRECTON_MOCK_DATA`를 `official/`로(`start_server.sh`가 기본값으로 설정)
- 글 전용 확인(STT·TTS 크레딧 0, LLM만): `python scripts/text_check.py <case_id…> [-v]` (Mock 8100 필요)
- 글 전용 평가(공식 judge·tool_match 재사용, 채점기에 `OPENAI_API_KEY` 필요): `python scripts/text_eval.py --out runs/<새 폴더> [--ids A,B] [--cases <jsonl> --rubric <json>]`. medium 추론은 실행마다 응대 ±4점 흔들리니 한 번 결과로 판단하지 않는다
- 무음 TTS 평가 모드: 상담 서버를 `DIRECTON_SILENT_TTS=1`로 띄우면 답 본문은 무음(캐시된 첫 마디·채움말은 실제 음성, 무음은 캐시 파일로 저장하지 않음). 공식 `run_eval.py` 결과를 `python scripts/stt_score.py --run runs/<폴더>`로 채점(상담원 글 기준). 조회 뒤 답 속도는 이 모드로 잴 수 없다
- STT 문맥 힌트 비교: `python scripts/stt_check.py <case_id…>` (힌트는 `agent/config.py`의 `stt_context`)
- 체험 데모: `powershell -ExecutionPolicy Bypass -File demo\run_demo.ps1 -Force` (공식 데이터·매뉴얼, `.venv` 파이썬, Mock 8100·서버 9000·페이지 8080, 실제 TTS). 공식 데이터 기준 프리셋은 git 밖 `official/demo_presets.json`
- 공식 측정: `python ../02_대회자료/directon-2026-voice-cs/eval/run_eval.py --server ws://127.0.0.1:9000 --cases <jsonl> --audio <음성 폴더> --mock http://localhost:8100 --out runs/<새 폴더>` → `score.py --run … --cases … --no-judge`(OpenAI 채점 생략). 특정 문항만 재려면 그 줄만 뽑은 jsonl을 `--cases`로
- 제출 ZIP: `python scripts/make_submission.py` (→ `../submission.zip`, `.env`는 `BAZE_API_KEY`만)
- 제출 환경은 리눅스·Python 3.11·`requirements.txt`·`start_server.sh`/`start_mock.sh <데이터파일>`. 실행 환경을 고치면 이 절을 갱신한다.

## 불변 규칙

- 모듈 간 계약은 `agent/core/types.py`. 시그니처를 바꾸면 이유를 커밋 메시지에 적는다.
- PCM은 항상 16kHz s16le mono. 다른 샘플레이트는 경계에서 변환한다.
- 측정기 프로토콜을 깨지 않는다: `session.start` → PCM → (우리) PCM + `agent.transcript` + `turn.end`.
- 고객 발화가 끝나기 전에는 유성음을 보내지 않는다(속도 0점 조건).
- 함수 호출은 `session.start`의 `tools_url`로만, 항상 `X-Session-Id` 헤더를 붙인다.
- `DIRECT_ON 2026 Starter Kit/`는 원본이다. 수정하지 않는다.
- API 키는 환경변수(`BAZE_API_KEY` = 대회 크레딧 게이트웨이, 직접 키 `SONIOX_API_KEY`·`GEMINI_API_KEY`·`OPENAI_API_KEY`가 있으면 우선) 또는 `.env`에서만 읽는다. 제출 ZIP의 `.env`에는 `BAZE_API_KEY`만 넣는다. 코드·로그·문서에 키를 쓰지 않는다.

## 외부 서비스 fake

자동 테스트(`live` 아님)에서 실제 네트워크 호출 금지(크레딧 보호). Provider는 생성자에 클라이언트/전송 함수를 주입받게 만들고, 테스트에서는 fake를 넣는다.

| 서비스 | 위치 | fake 방법 |
|---|---|---|
| Gemini TTS | `agent/providers/gemini_tts.py` | 주입한 fake client |
| Soniox | `agent/providers/soniox_stt.py` | fake websocket |
| OpenAI | `agent/providers/openai_llm.py` | fake client |
| Mock API | `agent/tools_client.py` | `httpx.MockTransport` |

## 커밋

- 형식: `<type>(<scope>): <요약>` — type: feat, fix, test, docs, chore, perf, refactor / scope: agent, turn, providers, mock, eval, docs
- 작업당 커밋 1개, `git add`는 경로 지정. push는 요청 시에만.
