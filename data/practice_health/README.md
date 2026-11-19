# 건강기능식품 연습판 (AG-T22)

공식 도메인(건강기능식품 쇼핑몰, 업무 7종)을 가정한 **연습용** 데이터. 가상 브랜드 **온결헬스**(실제 회사 아님). 당일 공식 자료(11/21 09:00)가 나오면 공식 파일로 교체하는 리허설에 쓴다. 근거: `docs/04_목표_점검.md`, `docs/AGENT_PLAN.md` AG-T17~T23 공통 계약.

| 파일 | 내용 |
|---|---|
| `generate.py` | 아래 JSON 파일들의 **원본**(함수 정의·데이터·시나리오). 고친 뒤 `uv run python data/practice_health/generate.py` (테스트가 동기화를 검사) |
| `tools.json` | 함수 15개(기존 13개를 건강기능식품용으로 바꾸고 `get_payment_info`·`check_deposit` 추가, `get_coupons` 제외) |
| `source_data.json` | 사람이 읽는 쇼핑몰 형식 원본: 회원 8명, 상품 15개(성분·섭취 방법·주의사항·보관·가격), 주문 23건(**주문번호 4자리**, 예: 1234), 결제 23건(무통장입금 7건 중 입금대기 3건), 환불 3건 |
| `mock_data.json` | Mock 서버용 **Starter Kit 매핑 형식** `{함수: [{"arguments", "result"}]}` 280항목(공식 당일 데이터와 같은 경로로 리허설) |
| `manual.md` | 응대 매뉴얼(업무 7종 + 섭취 후 이상 반응 → 상담원 연결, 의학적 판단 금지) |
| `cases.jsonl` | 시나리오 40개(`data/testset`과 같은 스키마 + `task_type`) |
| `audio/`, `cache/` | `eval.build_testset`로 만드는 음성·합성 캐시(git 제외) |

데이터 기준일 **2026-11-20(금)**. 정책 숫자는 지금 `mock_api/server.py`에 박힌 문구와 같게 맞췄다: 무료배송 5만 원, 배송비 3,000원, 반품·교환 수령 후 7일, 단순변심 왕복 6,000원, 카드 환불 3~5영업일·그 외 1~3영업일, 회수 1~2영업일.

## 실행

```
# Mock (AG-T17 계약)
uv run python -m mock_api.server --port 8100 --data data/practice_health/mock_data.json --tools data/practice_health/tools.json

# 상담원
DIRECTON_MANUAL=data/practice_health/manual.md DIRECTON_CATALOG_PATH=data/practice_health/mock_data.json uv run python -m agent.server --port 9000

# 음성(Gemini TTS 8종, 이미 있으면 건너뜀)
uv run python -m eval.build_testset --cases data/practice_health/cases.jsonl \
    --out data/practice_health/audio --cache data/practice_health/cache --workers 2 --max-gemini 1

# 측정
uv run python -m eval.runner --cases data/practice_health/cases.jsonl --audio data/practice_health/audio \
    --mock http://127.0.0.1:8100 --out runs/practice_health
```

## 업무 7종 분포

| task_type | 문항 | 함수 |
|---|---|---|
| 배송 | 6 | track_delivery, change_shipping_address(가능·불가), FAQ |
| 주문취소 | 6 | cancel_order(즉시·요청 접수·불가), 고민 중(실행 금지), 정보 요청, 2턴 |
| 반품환불 | 6 | request_return(접수·기한 지남), get_refund_status(완료·예정·계좌), 개봉 상품 FAQ |
| 교환 | 4 | request_exchange(오배송·파손·정정 발화·기한 지남) |
| 결제 | 7 | get_payment_info(입금 계좌·수단·기한), check_deposit(확인·미확인), FAQ, 2턴 |
| 상품 | 6 | get_product_info(섭취 방법·약 복용 주의·보관·연령), check_stock(품절), 2턴 |
| 상담원연결 | 5 | 섭취 후 몸 불편 3(복통·설사 / 메스꺼움·두드러기 / 두근거림·어지러움), 직접 요청, 영업시간 외 콜백 |

history 있는 문항 3개(H0012·H0029·H0035). caller(통화 정보) 있는 문항 9개. 주문번호는 대부분 한 자리씩("일삼공이"), H0001만 "천이백삼십사"(자릿수 읽기)로 읽는다.

## Mock 데이터 형식

- `mock_data.json`은 매핑 형식이라 `mock_api/server.py`(AG-T17)의 매핑 모드로 뜬다. 서버 코드 변경 없이 결제 함수까지 전부 응답한다. 정의 안 된 입력은 서버가 `not_found`(matched=false).
- 기존 함수의 `result`는 생성 시점에 서버의 쇼핑몰 규칙(`Shop._fn_*`)으로 계산한다(문구가 기존 Mock과 같다). 결제 함수와 `입금대기` 상태는 `generate.py`가 만든다:
  - `get_payment_info(order_id)` → 수단, 금액, 상태(결제완료/입금대기/환불완료), 카드사·할부 또는 은행·계좌·예금주·입금자명·입금 기한·확인 여부.
  - `check_deposit(order_id)` → 무통장입금이면 `deposit_confirmed`·`deposited_at`·`deposit_deadline`, 아니면 `not_allowed`(matched=true).
  - `입금대기` 주문: `cancel_order` 즉시 취소(환불 0원), `track_delivery` 입금 확인 전 안내, 반품·교환 `not_allowed`, 배송지 변경 가능.
- `get_product_info` 결과에 `volume`·`ingredients`·`how_to_take`·`cautions`·`storage`를 더했다.
- 매핑 모드 한계: `transfer_to_agent`는 빈 `arguments` 항목 하나라 어떤 `reason`에도 맞는다. `check_stock`에 없는 옵션을 넣으면 옵션 없는 항목에 걸려 matched=true(전체 옵션 재고)가 된다.
- 쇼핑몰 형식 원본(`source_data.json`)을 `--data`로 주면 결제 함수 두 개는 서버에 `_fn_`이 없어 `not_found`, `입금대기` 주문의 `cancel_order`·`track_delivery`는 예외가 난다 → 매핑 파일을 쓴다.

## 한계

- 연습용 가정이다. 공식 함수 이름·인자·응답 형식은 다를 수 있다(공식 Starter Kit 예시는 `{함수: [{arguments, result}]}` 고정 응답 형식).
- 개봉 여부·맛 변경 같은 반품 조건은 서버가 검사하지 않는다(매뉴얼·LLM 판단).
- 상품 `description`에도 섭취 방법·주의사항 요약을 넣었다(쇼핑몰 형식으로 띄울 때 대비).
- 오디오는 Gemini TTS 8종(Kore 제외)만 썼다. edge 음성은 없다.
