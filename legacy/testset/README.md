# 테스트셋 (AG-T05a)

다이렉트몰 음성 CS 에이전트 평가용 한국어 시나리오. 형식은 `docs/AGENT_PLAN.md` 3.2(스키마)·3.3(통화 환경)·3.4(범주·태그), 응대 근거는 `legacy/shop_mock/manual.md`, 함수·데이터는 `legacy/shop_mock/`.

| 파일 | 내용 |
|---|---|
| `scenarios_src.yaml` | **사람이 직접 쓴** 시나리오 원본(발화·쉼·history·caller·기대 함수·기대 응대) |
| `cases.jsonl` | 생성 결과(한 줄 한 문항). 직접 고치지 말고 원본을 고친 뒤 다시 생성 |
| `audio/` | AG-T05b가 `eval/build_testset.py`로 만드는 음성(git 제외) |

## 생성

```
uv run python scripts/gen_scenarios.py          # cases.jsonl 다시 생성
uv run python scripts/gen_scenarios.py --stats  # 아래 분포표 출력
uv run pytest -q tests/test_testset.py          # 스키마·Mock 일치·분포·원본 동기화 검사
```

- 원본(`scenarios_src.yaml`)에 사람이 쓰는 것: 발화(`s`, 쉼이 있으면 `[텍스트, 쉼ms]` 세그먼트), 범주(`c`), 태그(`t`), caller 전화번호(`who`), history(`h`), 영업시간 외 테스트용 `now`, 기대 함수(`x`), 기대 응대(`p`), 금지 응대(`n`), Gemini 말투 지시(`style` → `voice_style`).
- 생성기가 붙이는 것(seed `20261120` 고정, 같은 원본 → 같은 출력):
  - `case_id`: 원본 순서대로 `T0001~`. **원본 중간에 문항을 넣으면 뒤 번호가 밀리므로** 새 문항은 끝에 추가한다.
  - `voice`: 11종을 블록 단위로 섞어 고르게 배분(gemini 8종 + edge 3종). `gemini:Kore`는 상담원 음성이라 쓰지 않는다.
  - `env`: 축마다 비중대로 개수를 정해 섞는다. `lead_ms`는 200~1500을 고르게 나눠 섞는다.
  - `now`: 원본에 없으면 2026-11-20(금, 데이터 기준일) 영업시간(09:05~17:55, 점심 제외) 중 하나.
  - `customer_profile`: `who` 번호로 mock_data 회원을 찾아 `{"name","phone","email":null}`. 이때 `caller_known` 태그를 자동으로 붙인다.

## 작성 규칙

- 발화는 TTS가 읽을 **사람의 원문**이다. 숫자는 한글 낭독으로 적는다("이육일일 일오공 이삼일", "공일공에 구팔칠육에 오사삼이", "영일영…", "이 육 일 일…"). 아라비아 숫자 금지(테스트로 검사).
- `mid_pause`·`connective_end`: 세그먼트 사이 쉼 300~1200ms. 쉼 앞은 끝나지 않은 말("주문을 잘못 넣어서" [800])이거나, 완결 문장처럼 들리지만 뒤에 요청이 이어지는 함정("배송 조회 좀 하려고요." [700] "아 그리고 주소도…").
- `expected_tools`: `mock_api.server.load_shop().lookup()`에서 `matched=true`여야 한다. 필수 자유 텍스트 인자(`new_address`, `transfer_to_agent.reason`)는 넣고, 채점은 키 인자(`phone_number`, `order_id`, `product_name`, `option`)만 비교한다. 키 인자가 의도적으로 틀린 시나리오(없는 주문번호·비회원 번호)는 `"expect_not_found": true`.
- caller가 있으면 전화번호 인자는 caller 번호, 주문번호는 그 회원의 주문이어야 한다(테스트로 검사).
- 범주 규칙: `faq`·`out_of_scope`는 함수 없음. `complaint_handoff`는 영업시간 내 `transfer_to_agent`, 영업시간 외(평일 18시 이후·점심·주말) `request_callback`. `need_info`는 함수 없음, 또는 caller가 있으면 `get_order_list`/`get_member_info`만. `followup`은 history가 있고 마지막이 상담원 발화.
- `expected_points` 1~3개, `must_not` 0~2개. 매뉴얼 조항과 mock 데이터 값(금액·날짜·상태)에 근거한 구체 문장.

## 분포 (172문항)

| category | 문항 | 기준 |
|---|---|---|
| tool_lookup | 39 | ≥ 12 |
| tool_action | 34 | ≥ 12 |
| followup | 30 | ≥ 12 |
| faq | 19 | ≥ 12 |
| need_info | 19 | ≥ 12 |
| complaint_handoff | 17 | ≥ 12 |
| out_of_scope | 14 | ≥ 12 |

| tag | 문항 | 기준 |
|---|---|---|
| caller_known | 55 | ≥ 30 |
| digits | 54 | ≥ 30 |
| mid_pause | 34 | ≥ 30 |
| short | 31 | |
| history 있음 | 30 | ≥ 20 |
| banmal | 21 | ≥ 15 |
| connective_end | 16 | ≥ 15 |
| filler_words | 16 | ≥ 15 |
| long | 11 | |
| negation | 9 | ≥ 8 |
| self_correction | 8 | ≥ 8 |

| 환경 축 | 분포 |
|---|---|
| channel | wideband 58 / phone 57 / phone_ulaw 57 |
| noise | none 62(36%) / white 28 / pink 28 / babble 27 / hum 27 |
| snr_db | 30: 43 / 20: 52 / 10: 51 / 5: 26 |
| gain_db | 0: 103 / −12: 35 / +6: 34 |
| speed | 1.0: 103 / 0.85: 35 / 1.2: 34 |
| packet_loss | 0: 151 / 0.02: 21 |
| lead_ms | 200~1500 균등 |

| voice | 문항 |
|---|---|
| gemini: Puck, Charon, Fenrir, Leda, Zephyr | 각 16 |
| gemini: Aoede, Despina, Orus | 각 15 |
| edge: ko-KR-SunHiNeural, ko-KR-HyunsuMultilingualNeural | 각 16 |
| edge: ko-KR-InJoonNeural | 15 |

| expected_tools 함수 | 호출 수 |
|---|---|
| get_order_list | 21 |
| cancel_order | 17 |
| transfer_to_agent | 16 |
| track_delivery | 11 |
| check_stock, request_callback | 각 10 |
| get_order_info | 8 |
| change_shipping_address, get_refund_status, request_exchange, request_return | 각 7 |
| get_coupons, get_product_info | 각 5 |
| get_member_info | 3 |

영업시간 외 `now` 5문항(평일 19:30·19:40·20:10, 점심 12:20, 토요일 10:30), `expect_not_found` 호출 2개.

## 한계

- 함수 기대값은 한 번의 응답 안에서 부를 함수 전체다. caller만 있고 주문번호가 없으면 `get_order_list` → 해당 주문 함수의 2단계로 적었다(채점기가 순서·부분 일치를 어떻게 볼지는 AG-T06에서 정한다).
- `snr_db`는 `noise=none`이어도 값이 들어 있다(증강기가 무시).
- history 형식은 공식 형식이 나오면 맞춘다(3.2절).
