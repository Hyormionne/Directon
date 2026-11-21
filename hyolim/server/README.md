# server — 음성 상담 서버 (fncall 연결)

```
session.start → 손님 음성 → Soniox STT(<end>) → 채움말 PCM 즉시 → fncall.run_turn (조회 병렬 검증·접수 게이트)
             → 템플릿 답(LLM 없음) 또는 GPT 스트리밍 → 문장 단위 Gemini TTS(24k→16k) → turn.end
```

| 파일 | 내용 |
|---|---|
| `app.py` | WebSocket 핸들러, 채움말, 템플릿/LLM 분기, 20초 예산, 오류 시 대체 음성 |
| `stt.py` | Soniox 실시간 + 말 끝 안전장치 (직전 200ms 말소리·덜 불린 번호면 대기, 음량 대체는 1.9초) |
| `llm.py` | 프롬프트(고정 앞부분 → 캐시), 스트리밍, 문장 분할. LLM 에는 도구를 주지 않음 |
| `tts.py` | Gemini 스트리밍 TTS, 16kHz 변환 |
| `make_assets.py` | 채움말 PCM 미리 합성 → `server/assets/` (제출물에 포함) |

## 실행

```bash
cp .env.example .env               # SONIOX_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY 채우기
python3 -m server.make_assets      # 채움말 음성 생성 (한 번)
bash start_mock.sh data/mock_db.json
bash start_server.sh
cd eval && python3 run_eval.py --server ws://localhost:9000 --mock http://localhost:8100 \
  --cases ../data/practice_cases.jsonl --audio ../data/audio/practice --out runs/real1 --limit 20
```

키 없이 배관만 확인: `FAKE_APIS=1 FAKE_TRANSCRIPTS=data/practice_cases.jsonl bash start_server.sh`
(STT 는 연습 문항 원문으로 대체, LLM·TTS 는 가짜. **제출 시에는 설정하지 않는다.**)

## 가짜 모드 측정 (80문항, 실제 측정기·중계·모의 서버)

| | 결과 |
|---|---|
| 함수 점수 | 98.8 |
| 손님 말 도중 끼어듦 | 0건 (음량 기준 700ms 일 때 8건 → 1.9초로 올림) |
| 조회 뒤 답 | 템플릿 0.01초 + 실제 TTS 첫 조각 시간 |
| 첫 소리 | 1.89초 — 가짜 모드는 음량 판단만 써서 느림. 실제는 Soniox `<end>`(최대 0.5초) + 채움말 즉시 |

## 아직 확인 안 된 것 (키 필요)

- Soniox 실제 한국어 숫자 출력 형태와 `<end>` 가 문장 안 쉼(최대 1.8초)에서 일찍 오는지
- Gemini TTS 스트리밍 응답 형식 (`inlineData` / `delta` 둘 다 찾게 해 둠), 첫 조각 지연
- `gpt-5.4-mini` 가 `reasoning_effort: none` 을 받는지 (안 받으면 자동으로 빼고 재시도)
- 응대 점수 (OPENAI_API_KEY 로 `score.py` 채점)

## 위험도 가드 (`fncall/risk.py`, `server/answer.py`)

| 등급 | 예 | 처리 |
|---|---|---|
| HIGH (지병·약·임신 섭취 문의) | 혈압약 먹는데 같이 먹어도 돼요 / 당뇨 있는데 / 임신 중인데 | LLM 자유 생성 금지 → 매뉴얼 [지병 및 복용 중인 약…] 문구("전문의 상담 후 섭취 권장"). **상담원에게 넘기지 않음** |
| HIGH (섭취 후 이상 증상) | 먹고 두드러기 / 배가 아파요 | 섭취 중단 안내 → 상담원 연결 (매뉴얼이 이 경우를 위 문구 대상에서 제외하고, 시나리오가 연결을 지시) |
| MEDIUM | 제품을 특정 못 함 / 며칠분·할인 금액 / 쿠팡 등 다른 판매처 / 효과 보장 요구 | LLM 에 제약 힌트 + 문장 검사 |
| LOW | 일반 문의 | 일반 응대 (템플릿 우선) |

문장 검사(`guard_sentence`): TTS 직전 한 문장씩, 효능 보장·치료·'부작용 없음' 표현과 근거에 없는 '숫자+단위'(30일분, 69,500원 등)를 안전 문장으로 바꾼다.
연습 80문항: HIGH 0 / MEDIUM 10 / LOW 70, 위험도 가드 전후 응대 점수 75.6 → 75.1 (차이 없음), 임의 수치 안내로 틀리던 P0158 은 맞음으로.

위험 문항 10개(`eval/risk_cases.jsonl`, 매뉴얼 문구를 참고 답으로 직접 작성): 응대 94.0 (맞음 8 / 괜찮음 2 / 틀림 0).
`python3 -m server.eval_text --cases eval/risk_cases.jsonl --all-cases --limit 0`
