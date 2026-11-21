# hyolim — 함수 호출 정확도 + 지연 최소화 음성 상담 서버 (초기 버전)

공식 Starter Kit(하루온) 위에서 동작하는 음성 상담 서버와 함수 처리 모듈이다.
번호 확정·함수 선택·접수 판단은 코드가 맡고, LLM(GPT-5.4 mini)은 말할 내용만 만든다.

## 구성
| 폴더 | 내용 |
|---|---|
| `fncall/` | 함수 처리: 번호 추출(한글 숫자·고쳐 말하기·010 규칙), 번호 확정, 주제 판단, 1·2차 조회(후보 병렬·발신번호), 접수 게이트, 템플릿, 위험도 판단. 평가 스크립트 포함 |
| `server/` | WebSocket 상담 서버: 말 끝 판단(Soniox `<end>` + 안전장치), 채움말 즉시 출력, 템플릿/GPT/안전 문장 세 갈래 답, TTS 스트리밍, 사용량 기록·TTS 캐시 |
| `demo/` | 브라우저로 직접 통화해 보는 도구와 채점기 |
| `mock_tools/mock_server.py` | 공식 뼈대에 `lookup()` 을 채운 모의 함수 서버 |
| `eval/risk_cases.jsonl` | 직접 만든 위험 문항 10개 (복약·증상·임신 등) |

자세한 설명은 `server/README.md`, `fncall/README.md`.

## 실행 준비 (대회 데이터는 저장소에 넣지 않음)
이 폴더에는 대회 데이터와 공식 측정 도구가 없다. 공식 Starter Kit 압축을 푼 폴더에서 아래를 이 폴더로 복사한다.
- `data/` (브랜드 정보·연습 문항·음성·mock_db)
- `eval/run_eval.py`, `eval/score.py`

```bash
uv venv -p 3.11 .venv && source .venv/bin/activate && uv pip install -r requirements.txt
cp .env.example .env          # 키 입력 (베이즈 게이트웨이: BAZE_API_KEY, BAZE_WS_URL=wss://api.baze.co/v1/gateway/soniox/transcribe-websocket)
python3 -m server.check_keys  # 키 확인 (크레딧 거의 0)
```

## 평가 명령
| 명령 | 내용 |
|---|---|
| `python3 -m fncall.eval_fncall --stt-noise 0.5` | 함수 오프라인 평가 (키 불필요) |
| `python3 -m server.eval_text --limit 0` | 글 기준 응대·함수 (GPT) |
| `python3 -m server.eval_bench` | Soniox 받아쓰기 → 함수 → GPT 답 → 채점 (음성합성 없음) |
| `bash start_mock.sh data/mock_db.json` + `bash start_server.sh` + `eval/run_eval.py` | 대회 방식 실제 음성 측정 |

## 지금까지 결과 (연습 80문항, 2026-11-21)
| 측정 | 응대 | 함수 | 비고 |
|---|---|---|---|
| Soniox stt-rt-v6 받아쓰기, 음성합성 없음 | 63.4 | 81.2 | 받아쓰기 글자 오류율 9.9%, 번호 맞힘 5/23 |
| 대회 방식 실제 음성 (OpenAI 받아쓰기·합성으로 임시 대체) | 54~58 | 80~84 | 속도 99.4, 첫 소리 약 0.94초 |
| 글 기준 (받아쓰기 완벽 가정) | 약 81 | 98.8 | |

감점의 대부분은 번호 음성 인식 오류다. Soniox 와 OpenAI 모두 같은 문항에서 같은 식으로 틀렸다.

## 주의
- 제출에는 Soniox·Gemini·GPT-5.4 mini 만 쓴다. `STT_BACKEND=openai`, `TTS_BACKEND=openai` 는 측정용 임시 대체.
- `.env`, `usage/`, `server/cache/`, `server/assets/`, `eval/runs/`, `eval/heard_*.json` 은 올리지 않는다(.gitignore).
