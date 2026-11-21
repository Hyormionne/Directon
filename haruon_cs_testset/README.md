# 하루온 음성 CS 녹음 테스트셋 (200문항)

공식 과제 자료(`directon-2026-voice-cs/data/brand/`: 시나리오 10개, 브랜드 정보, 함수 8개)를 바탕으로 만든 **직접 녹음용** 테스트셋입니다. 측정기 규격대로 **1문항 = 연결 1개 = 손님 발화 1턴**입니다. 고객·주문·번호는 모두 새로 지은 가짜 값이고, 공식 연습 데이터의 번호와 겹치지 않습니다(빌드할 때 검사).

## 파일

| 파일 | 내용 |
|---|---|
| `recording_script.pdf` / `.md` / `.html` | 녹음 대본. 녹음 규칙, 화자별 분량, 화자별 대사(쉼 표시, 직전 상담원 말) |
| `recording_script.csv` | 녹음 진행 체크용(`done` 열) |
| `metadata.jsonl` / `.csv` | 문항별 전체 정보: `session_start`, `utterance`(STT 정답, 숫자는 숫자로), `recording.read_text`(읽는 그대로), 쉼, 기대 함수 호출·결과, 답에 들어갈 내용, 하면 안 되는 말, 태그 |
| `cases.jsonl` | 공식 `practice_cases.jsonl`과 같은 모양. 공식 측정기 `--cases`에 바로 넣을 수 있음 |
| `session_start/hNNN.json` | 연결 직후 보낼 `session.start` 200개(공식 SPEC 필드: `case_id`, `now`, `history`, `caller`, `tools_url`, `audio`) |
| `mock_db.json` | 가상 고객 29명, 주문 29건, 반품 4건. 공식 모양 `{함수명: [{arguments, result}]}` |
| `gen/` | 생성기. 문항은 `gen/items.py`, 고객·주문은 `gen/world.py`에서 고친 뒤 `python gen/build.py --pdf` |
| `tools/` | `split_recording.py`(이어 읽은 녹음을 문항별로 자르기), `check_audio.py`(형식·무음·쉼 검사) |

## 구성

- 시나리오: 배송·주문확인 30, 배송지변경 10, 오배송·누락·미수령 14, 상품문의·추천 26, 주문취소 18, 반품·환불요청 26, 반품·환불진행 16, 교환·불량 12, 결제·전화주문 22, 상담원연결·기타 26
- **화자 5명(여 2: 신소연·손효림, 남 3: 윤진·최준호·신호진) × 40문항**. 같은 고객은 같은 화자가 읽고, 고객 성별과 화자 성별이 같음. 미등록 고객 문항 중 16개는 분량을 맞추려고 성별과 상관없는 대사를 남성 화자에게 배정함(`voice_gender_free` 태그)
- 함수 호출 117문항(get_order_info 68, search_products 19, get_return_info 13, return_order 6, request_callback 5, cancel_order 3, return_order_item 2, exchange_order 2), 함수 없이 답해야 하는 문항 83
- 대화 이력 있음 42, 발신자 정보 없음(caller null) 74, 1초 이상 쉼 27, 고령 고객 28, 번호를 불러 주는 문항 22
- 예상 녹음 총량 약 14분(화자당 2.5~3.5분)

## 사용

```bash
# 녹음: 화자별로 대본 순서대로 이어 읽은 파일 → audio/h001.wav …
uv pip install faster-whisper
uv run python haruon_cs_testset/tools/split_recording.py 신소연_녹음.m4a --speaker S1 --meta haruon_cs_testset/metadata.jsonl
uv run python haruon_cs_testset/tools/check_audio.py haruon_cs_testset/audio haruon_cs_testset/metadata.jsonl --fix

# 측정: 이 데이터로 mock 서버를 띄우고 공식 측정기로 돌림
uv run python -m mock_api.server --port 8100 --db haruon_cs_testset/mock_db.json
```

## 주의

- `mock_db.json`의 `search_products`는 공식 `mock_db.json`에서 빌드할 때 복사합니다. 공식 데이터는 반출 금지라 이 파일은 커밋하지 말고, 필요할 때 다시 빌드하세요.
- 접수 함수(cancel/return/exchange/callback)는 기록이 없으면 mock 서버가 `ok`를 돌려줍니다. 예외: `20261113-0048826`의 `return_order`는 `already_requested`.
- 기대 함수의 자유 텍스트 인자(`reason`, `pickup_address`)는 참고용입니다. 채점할 때는 번호 인자만 비교하세요.
