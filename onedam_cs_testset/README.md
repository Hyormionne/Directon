# 온담헬스 음성 CS 테스트셋 (100문항)

가상 건강기능식품 브랜드 **온담헬스** 고객센터에 걸려 온 통화를 가정한 테스트셋입니다. 측정기 규격에 맞춰 **1문항 = 연결 1개 = 고객 발화 1턴**으로 만들었습니다.

## 구성

| 파일 | 내용 |
|---|---|
| `metadata.jsonl` | 문항별 전체 메타데이터(아래 필드 설명). 평가 스크립트에서 읽는 기준 파일 |
| `metadata.csv` | 같은 내용을 한 줄씩 펼친 표(엑셀/시트에서 보기용, UTF-8 BOM) |
| `session_start/qNNN.json` | 연결 직후 보낼 `session.start` 메시지 100개 |
| `recording_script.md` | 녹음 대본. 녹음 규칙과 화자별로 묶은 대사, 쉼 표시 포함 |
| `recording_script.csv` | 녹음 진행 체크용(`done` 열) |
| `mock_db.json` | 고객 15명·주문·구독·제품 10종·정책. **모든 기대 응답은 이 데이터 기준** |
| `tools_assumed.json` | 가정한 함수 20개(공식 규격 나오면 교체) |
| `stats.json` | 분포 통계 |
| `tools/split_recording.py` | 화자 한 명이 이어 읽은 녹음을 문항별 `audio/qNNN.wav`로 자르기(로컬 Whisper 정렬 + 검사) |
| `tools/check_audio.py` | 녹음 검사: 형식, 앞뒤 무음, 지시한 쉼이 실제로 있는지 |
| `tools/simulate.py` | 로컬 측정기 흉내 + 목업 함수 서버. 끼어듦·지연·타임아웃·함수 호출 측정 |
| `gen/` | 생성 스크립트(`python gen/build.py out`). 문항은 `gen/items.py`에서 수정 |

## metadata.jsonl 필드

- `session_start`: `type`, `session_id`, `timestamp`(현재 시각), `history`(이번 말 이전 대화), `caller`(발신자), `tools_url`
- `utterance`: STT 정답 텍스트 / `recording.script`: 쉼 표시가 들어간 녹음용 대사
- `pauses`: 말 중간 쉼 `[{after, char_offset, sec}]`. **1초 이상 쉼이 있는 문항은 30개**로, 엔드포인팅 함정 테스트용
- `intent_reveal`: 용건이 확정되는 단어와 그 지점까지의 접두 텍스트, 비율(`ratio`). 미리 답을 준비하는 시점 평가용
- `expected_route`: `small`(기존 케이스, 61개) / `large`(복합·정책 예외·건강 안전·신규, 39개). JEV 라우팅 정답 라벨
- `expected_tools`: 기대 함수 호출(이름 + 인자). 함수 없이 답해야 하는 문항은 11개
- `expected_response`: 응답에 들어가야 할 핵심 내용 / `forbidden`: 하면 안 되는 말(진단, 효능 단정, 카드번호 요청 등)
- `tags`: `long_pause`, `has_history`, `multi_intent`, `self_correction`, `number_dictation`, `time_dependent`, `policy_edge`, `elderly`, `emergency`, `off_topic`, `caller_unmatched` 등

## 분포

배송조회 14 · 주문취소/변경 10 · 교환/반품/환불 12 · 정기구독 11 · 제품정보 12 · 섭취/안전 10 · 이상반응 5 · 결제/혜택/증빙 10 · 계정 4 · 상담원/불만/예외 12
대화 이력 있음 12 · 미등록/발신번호 없음 25 · **화자 5명(남 3·여 2, 모두 20대) × 20문항**, 같은 고객은 같은 화자 · 예상 녹음 총량 약 7.5분

## 한계

화자가 모두 20대라 실제 고령 화자의 음성(STT 인식률·발화 속도)은 검증되지 않습니다. 고령 고객 10문항은 말 속도·쉼·말버릇으로만 재현합니다.

## 11/21 공식 자료가 나오면 맞출 것 (가정한 부분)

1. `session.start` 필드명: `timestamp`, `history[].role`(customer/agent), `caller` 구조. `gen/build.py`의 `caller_obj`와 레코드 조립부만 고치면 됨
2. 함수 목록과 호출 형식: `tools_assumed.json`은 임시안. `simulate.py`의 목업 서버는 `POST {tools_url}`에 `{"name", "arguments"}` 형식을 가정
3. 시각 기준: 문항 시각은 2026-11-24(화) 09:05부터 5분 간격. 시간에 따라 답이 달라지는 문항(q005·q006·q039·q040·q079·q092)은 시각을 따로 지정함

## 사용 순서

```bash
# 1) 녹음 → audio/q001.wav … q100.wav
#    화자별로 대본 순서대로 이어 읽은 파일이면 자동으로 자르기(형식 변환·검사 포함)
pip install faster-whisper numpy
python tools/split_recording.py 신호진_녹음.m4a --speaker S5
python tools/split_recording.py 재녹음.m4a --ids q022,q096     # 일부만 다시 녹음했을 때
#    문항별로 따로 녹음했다면 검사만
python tools/check_audio.py audio/ metadata.jsonl --fix
# 2) 내 서버로 측정
pip install websockets numpy
python tools/simulate.py ws://localhost:9000/ws --audio audio --meta metadata.jsonl --db mock_db.json --out results
```
`results/results.jsonl`에 문항별 끼어듦 여부, 말 끝 기준 첫 응답 지연, turn.end까지 걸린 시간, 호출한 함수와 기대 함수의 일치율이 저장되고, 응답 음성은 `qNNN_resp.wav`로 저장됩니다.
