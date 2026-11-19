# 03. JEV(TypeSafe System One) + 플랜 B(로컬 임베딩) 조사

- 작성일: 2026-11-19
- 목적: 한국어 쇼핑몰 CS 음성 상담에서 고객 발화(부분/최종 STT)를 약 20개 케이스와 "해당 없음", "의도 아직 안 드러남"으로 분류하고, 턴 종료 확률을 얻는다.
- 표기 규칙: **[검증]**은 공식 문서에서 확인한 내용, **[커뮤니티]**는 비공식 가이드나 블로그 내용, **[미검증]**은 추정이거나 자료가 없는 내용이다.

---

## 0. 용어 정정

- 초안의 "Noul"이 **정식 명칭이 맞다**. 확률을 돌려주는 예/아니오 프리미티브의 이름은 `noul`이다. [검증] https://docs.typesafe.ai/primitives/noul.md
- 프리미티브는 네 가지다. `choice`(하나 고르기), `noul`(예일 확률), `score`(최대 10단계 순서 척도), `box`(이미지 영역, 1.14 전용이며 실험 기능).

---

## 1. JEV / System One API

### 1.1 엔드포인트와 인증 [검증]
- `POST https://api.typesafe.ai/v1/systemone`
- 헤더는 `Authorization: Bearer <API_KEY>`와 `Content-Type: application/json`이다. SDK는 환경변수 `TYPESAFE_API_KEY`를 읽는다.
- 모델 목록은 `GET /v1/models`로 조회한다. 계정에서 쓸 수 있는 alias가 나오고, 버전 ID는 목록에 없어도 호출할 수 있다.
- 선택 헤더
  - `TypeSafe-Danger-Zone-Long-Context: true`: 컨텍스트를 262k로 늘린다. 대신 느려지고 정확도가 떨어지므로 우리는 쓰지 않는다.
  - `TypeSafe-Danger-Zone-Images: true`: 이미지를 넣을 때 쓴다. 우리와는 무관하다.
- 요청 본문은 gzip 또는 zstd로 압축할 수 있다. 짧은 요청에서는 이득이 작다. [검증] https://docs.typesafe.ai/api/errors.md
- 출처: https://docs.typesafe.ai/api.md , https://typesafe-jev.com/en/guides/api/

### 1.2 모델 ID와 버전 고정 [검증]
| ID | 설명 |
|---|---|
| `jev-1.14.0` | 2026-10-14 출시된 최신 버전. choice, score, noul, box 지원. 변경 내역에 "Better multilingual support"가 있다. |
| `jev-1.13.0` | 이전 버전. choice, score, noul 지원. |
| `jev-latest` | 현재 `jev-1.14.0`을 가리킨다. SDK 기본값이다. |
| `jev-preview` | 현재 `jev-1.14.0`을 가리킨다. 프리뷰 빌드가 나오면 그쪽으로 먼저 옮겨간다. |

- 초안의 "jev-latest → 1.14"는 **맞다**. 2026-11-10 기준이며 커뮤니티 가이드와 공식 models 페이지가 일치한다.
- 응답의 `model` 필드에는 실제로 답한 버전 ID가 들어온다. 이 값을 로그로 남겨야 한다.
- **임계값을 튜닝했다면 `jev-1.14.0`처럼 버전 ID를 고정하라**는 것이 공식 권고다. alias는 새 버전이 나오면 예고 없이 옮겨간다. 폐기(deprecation) 정책은 문서에 없다.
- 출처: https://docs.typesafe.ai/models.md , https://docs.typesafe.ai/changelog.md

### 1.3 요청 형식 [검증]
```json
{
  "model": "jev-1.14.0",
  "state": { "...": "문자열, 객체, 배열 중 아무거나" },
  "questions": {
    "<내가 정한 ID>": { "type": "choice", "instructions": "...", "criteria": { "옵션명": "설명 또는 null" } },
    "<ID2>": { "type": "noul", "instructions": "...", "criteria": { "true": "...", "false": "..." } }
  }
}
```
- `state`: 모든 질문이 공유하는 입력이다. 필드 이름을 붙인 JSON 객체를 권장한다(예: `{"agent_last_turn": ..., "customer_partial": ...}`). 질문 안에서 `` `state.customer_partial` ``처럼 백틱 경로로 가리킬 수 있다.
- 질문 ID는 모델에 전달되지 않는다. 그러므로 의미는 모두 instructions와 criteria 안에 써야 한다.
- `instructions`와 `criteria` 항목에는 문자열, 객체, 배열을 쓸 수 있다. 헷갈리는 옵션에는 `{"covers": ..., "excludes": ..., "examples": [...]}` 같은 객체를 쓰는 것이 좋다. 필드 이름은 자유다.
- 출처: https://docs.typesafe.ai/primitives/choice.md

### 1.4 응답 형식 [검증]
```json
{
  "model": "jev-1.14.0",
  "answers": {
    "case":     { "type": "choice", "choice": "refund", "confidence": 0.81,
                  "probabilities": { "refund": 0.86, "exchange": 0.09, "...": 0.05 } },
    "turn_end": { "type": "noul", "noul": 0.93 }
  },
  "usage": { "input_tokens": 640, "output_tokens": 30 }
}
```
- **Choice는 모든 옵션의 확률을 돌려준다**(합은 약 1). `choice`는 확률이 가장 높은 옵션이다.
- Choice의 `confidence`는 `(p_max − 1/n) / (1 − 1/n)`로 계산한다. 즉 최상위 확률을 변환한 값일 뿐이다.
- Noul은 예일 확률 `noul` 하나만 돌려주고 confidence는 없다. 확신도 형태의 숫자가 필요하면 `|2p−1|`을 쓴다.
- 출처: https://docs.typesafe.ai/confidence.md

### 1.5 크기 제한 [검증]
| 항목 | 제한 |
|---|---|
| Choice 옵션 수 | 1~255. 22개는 문제없다. |
| Score 단계 수 | 최대 10 |
| 요청당 토큰 | 64k (state와 모든 질문 합산) |
| state + 질문 하나 | 32k 토큰 |
| 본문 크기 | 32 MiB |
| JSON 중첩 깊이 | 64 |
| 요청당 질문 수 | **최대값이 명시되어 있지 않다** [미검증]. 64k 토큰 안에서는 자유로 보인다. |
| 텍스트 길이 | 별도 제한 없이 토큰 한도로만 제한된다 |

### 1.6 한 요청에 여러 질문 묶기 [검증]
- 한 요청 안의 질문들은 **병렬로 평가되고 서로의 답을 보지 않는다**. 질문을 추가해도 지연은 거의 늘지 않는다고 문서에 나온다. 다만 토큰 비용은 늘어난다.
- 그러므로 **케이스 Choice, 턴 종료 Noul, (선택) "어느 케이스든 해당되는가" Noul을 한 요청으로 묶는 것이 정석**이다.
- 서로 다른 state 여러 개를 한 요청에 넣는 기능은 없다. state는 요청당 하나다.
- 출처: https://docs.typesafe.ai/primitives/noul.md , https://docs.typesafe.ai/cookbooks/parallel_questions

### 1.7 오류와 재시도 [검증]
- 400, 401, 413, 415, 422는 재시도하지 않는다. `max_tokens_exceeded`는 400으로 온다.
- 429(rate limit)와 529(과부하)는 지수 백오프로 재시도한다. `retry-after` 헤더가 있으면 따른다.
- SDK는 기본으로 2회 재시도한다. **실시간 경로에서는 `RetryPolicy(max_retries=0)`으로 재시도를 꺼야 한다.** 250ms 예산 안에서 재시도할 여유는 없다.

---

## 2. 지연, 리전, 한국어, 가격, 클라이언트

### 2.1 지연
- 벤더가 내세우는 수치는 호출당 **70~500ms**다. 이 수치는 **미국 서부에서 노트북으로 잰 값**이다. [커뮤니티] https://innfactory.ai/en/blog/jev-system-one-model-classifier-not-llm/
- 공식 docs.typesafe.ai에는 지연 수치가 없다. [검증: 없음]
- 서드파티 벤치마크 인용 중에 "0.4s"라는 수치가 있다. 조건은 알 수 없다. [커뮤니티]
- **호스팅 리전은 US West Coast이고 EU나 아시아 리전은 언급이 없다.** [커뮤니티] innfactory 글 기준이며, 공식 문서는 리전을 밝히지 않는다.
- **한국 서버에서 호출할 때 추정치** [미검증]
  - 서울↔미국 서부 RTT는 대략 120~160ms다.
  - keep-alive 연결이면 RTT 1회에 서버 처리 70~500ms가 더해져 **약 200~650ms**가 걸린다.
  - 새 연결이면 TCP와 TLS 핸드셰이크로 RTT 1~2회가 추가된다. 이 경우 **350~950ms**다.
  - 따라서 **250ms 타임아웃은 상당 비율의 요청에서 초과될 가능성이 높다.** 먼저 실측해야 한다(§6 참고).
- 22개 옵션 Choice와 Noul 1~2개를 묶은 요청의 입력은 약 500~1,000 토큰으로 추정한다 [미검증]. 짧은 입력이므로 지연은 500ms보다 70ms 쪽에 가까울 것으로 기대된다. 이 역시 실측이 필요하다.

### 2.2 한국어 품질 [미검증]
- 주 학습 언어는 영어이고, 다른 언어는 "처리하지만 영어만큼은 아니다". [커뮤니티] innfactory
- 1.14 changelog에 "Better multilingual support"라는 한 줄이 있고, 지원 언어는 명시하지 않는다. [검증]
- 한국어 정확도를 다룬 공개 자료는 없다. 한국어 블로그와 커뮤니티 글은 영어 수치를 반복할 뿐이다.
- **결론: 라벨된 한국어 CS 발화로 자체 평가해야 한다.** 1.13과 1.14를 비교하는 것도 권장한다.
- 우회 방법: `instructions`와 `criteria`(옵션 설명)를 **영어로 쓰고 state만 한국어로 두는 구성**과, 모두 한국어로 쓰는 구성을 A/B 테스트한다. 옵션 이름은 영어 snake_case로 둔다.

### 2.3 가격 [검증]
- 입력 토큰 **$0.042 / 1M tokens**이고 출력은 무료다(1.13, 1.14 동일). https://docs.typesafe.ai/models.md
- 요청당 1,000 토큰이면 약 $0.000042다. 통화 한 건에 부분 발화 호출을 수십 번 해도 비용은 무시할 수 있는 수준이다.

### 2.4 Rate limit [검증, 수치 미공개]
- 조직 단위로 초당 토큰과 초당 요청 수로 설정된다. 실제 수치는 개발자 콘솔에서만 볼 수 있다.
- 커스텀 또는 엔터프라이즈 플랜에서 상향할 수 있다.
- 얼리 액세스와 대기열이 있었고, 한도는 동적으로 조정된다는 언급이 있다. [커뮤니티]
- **부분 발화마다 호출하면 RPS가 빠르게 늘어난다.** 디바운스가 필수다(§6).

### 2.5 데이터와 보안 [검증/커뮤니티]
- 고객 데이터로 학습하지 않는다. 엔터프라이즈 플랜은 ZDR(zero data retention)을 지원한다. [검증]
- DPA와 SCC를 제공한다. [커뮤니티]
- 고객 개인정보(이름, 주소, 전화번호)는 state에 넣지 않도록 **STT 텍스트에서 마스킹한 뒤 보내는 것을 권장**한다.

### 2.6 Python 클라이언트 선택
| 방식 | 장점 | 단점 |
|---|---|---|
| **httpx 직접 호출 (권장)** | 의존성이 최소이고 타임아웃과 연결 풀을 완전히 제어할 수 있다. HTTP/2로 연결을 유지할 수 있다. | 타입 래핑을 직접 해야 한다. |
| 공식 SDK `typesafe-sdk` 0.7.5 (Python 3.10+) | `AsyncTypeSafeClient`, `Choice`/`Noul` 타입, `RetryPolicy`, zstd 압축 제공 | 내부적으로 `httpx2`를 쓴다. 이 의존성은 확인이 필요하다 [미검증]. |
| pydantic-ai `pydantic-ai-slim[typesafe]` | `Literal`/`bool` 필드로 선언적으로 쓸 수 있다. | 에이전트 추상화가 더해져 오버헤드가 있다. **확률 분포를 직접 다루기가 번거롭다**(`provider_details['confidence']`만 제공). 실시간 분류에는 과하다. |

- SDK의 `AsyncTypeSafeClient`는 `timeout`, `retry=RetryPolicy(max_retries=0)`, `http_client`, `compression`을 지원한다. https://docs.typesafe.ai/sdk/python/api/clients/async.md
- pydantic-ai 출처: https://pydantic.dev/docs/ai/models/typesafe/

### 2.7 최소 async httpx 예제 (22개 옵션 Choice + 턴 종료 Noul을 한 요청으로)
아래 코드는 문서의 스키마대로 작성했다. **실제 호출로는 검증하지 않았다.** [미검증]

```python
import asyncio, time, httpx

JEV_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.14.0"  # 임계값 튜닝 후 버전 고정

# 옵션 이름은 영어 snake_case로 쓴다(코드용). 의미는 설명에 쓴다. 22개 = 케이스 20 + none + not_yet
CASES: dict[str, str | dict] = {
    "order_status":   "주문/배송 진행 상황 문의 (언제 와요, 어디쯤이에요)",
    "delivery_delay": "배송이 예정보다 늦어 불만/확인",
    "cancel_order":   {"covers": "주문 취소를 원함", "excludes": "취소하지 않겠다는 말(예: '취소는 안 할게요')"},
    "refund":         "환불 요청 또는 환불 진행 문의",
    "exchange":       "다른 사이즈/색상/상품으로 교환",
    "return_pickup":  "반품 회수(택배 기사 수거) 일정",
    "defect":         "상품 불량/파손 신고",
    "wrong_item":     "다른 상품이 오배송됨",
    "missing_item":   "구성품/일부 상품 누락",
    "change_address": "배송지 변경",
    "payment_issue":  "결제 실패/이중 결제/결제수단 문제",
    "receipt":        "영수증·세금계산서·현금영수증 발행",
    "coupon_point":   "쿠폰·적립금·포인트 사용/소멸 문의",
    "stock_restock":  "재고·재입고 문의",
    "product_info":   "상품 사양·사이즈·소재 문의",
    "membership":     "회원정보·로그인·탈퇴",
    "price_adjust":   "가격 변동 차액 보상 요청",
    "complaint":      "서비스 불만 제기(구체 업무 요청 없음)",
    "human_agent":    "상담원 연결 요청",
    "repeat_request": "방금 안내를 다시 말해 달라는 요청",
    "none":           "쇼핑몰 CS 업무와 무관하거나 위 케이스 어디에도 해당하지 않음",
    "not_yet":        {"covers": "아직 말이 끝나지 않아 어떤 요청인지 드러나지 않음",
                       "examples": ["저기 제가 어제", "주문한 거 그게", "네 그 혹시"]},
}

def build_body(agent_last: str, customer_text: str, is_final: bool) -> dict:
    return {
        "model": MODEL,
        "state": {
            "agent_last_turn": agent_last,      # 직전 상담봇 발화(짧은 대답 해석에 필수)
            "customer_text": customer_text,     # STT 결과(개인정보 마스킹 후)
            "stt_is_final": is_final,
        },
        "questions": {
            "case": {
                "type": "choice",
                "instructions": ("Which request is the customer making in `state.customer_text` "
                                 "(Korean speech-to-text from an online shopping-mall support call)? "
                                 "Interpret short answers in light of `state.agent_last_turn`. "
                                 "Pay attention to Korean negation (안/못/-지 않-)."),
                "criteria": CASES,
            },
            "turn_end": {
                "type": "noul",
                "instructions": ("Has the customer finished their turn in `state.customer_text` "
                                 "and is now waiting for the agent to respond?"),
                "criteria": {
                    "true": "The sentence is complete (e.g., ends with -요/-니다/-까요/-주세요, or a complete short answer like 네/아니요).",
                    "false": "The sentence is cut off or trails with a connective (e.g., -는데, -고, -서, -면, 그, 저기) so more speech is expected.",
                },
            },
            # 선택: 'none' 옵션이 잘 동작하지 않을 때 쓰는 companion Noul(문서 권장 패턴)
            "any_case": {
                "type": "noul",
                "instructions": "Is the customer making any shopping-mall customer-service request (order, delivery, payment, return, product, account)?",
            },
        },
    }

async def classify(client: httpx.AsyncClient, agent_last: str, text: str, is_final: bool,
                   timeout_s: float = 0.25) -> dict | None:
    t0 = time.perf_counter()
    try:
        r = await client.post(JEV_URL, json=build_body(agent_last, text, is_final), timeout=timeout_s)
        r.raise_for_status()
    except (httpx.TimeoutException, httpx.HTTPError):
        return None                             # 호출자가 플랜 B 결과를 쓴다
    data = r.json()
    a = data["answers"]
    probs = a["case"]["probabilities"]
    top2 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:2]
    return {
        "model": data["model"],                 # 실제 답한 버전 로깅
        "case": a["case"]["choice"],
        "p_case": top2[0][1],
        "ratio": top2[0][1] / max(top2[1][1], 1e-6),
        "p_turn_end": a["turn_end"]["noul"],
        "p_any_case": a["any_case"]["noul"],
        "ms": (time.perf_counter() - t0) * 1000,
    }

async def main():
    # 커넥션을 미리 열어 TLS 핸드셰이크 비용을 통화 시작 전에 치른다. HTTP/2는 pip install httpx[http2]
    async with httpx.AsyncClient(
        http2=True,
        headers={"Authorization": "Bearer YOUR_KEY"},
        limits=httpx.Limits(max_keepalive_connections=20, keepalive_expiry=300),
    ) as client:
        await classify(client, "", "안녕하세요", True, timeout_s=3.0)  # 워밍업(결과 무시)
        print(await classify(client, "무엇을 도와드릴까요?", "어제 주문한 거 취소하고 싶은데요", True))

asyncio.run(main())
```

- 턴 종료 Noul만 따로, 더 짧게 보내고 싶다면 `questions`에 `turn_end`만 넣는다. 다만 질문을 묶어도 지연이 거의 늘지 않으므로 **한 요청으로 묶는 편이 유리하다**.
- Choice는 **첫 번째 옵션 쪽으로 치우치는 경향**이 있다. [검증] https://docs.typesafe.ai/model-jaggedness.md
  - 옵션 순서를 고정한 채 평가하라.
  - 옵션 순서를 바꿔 일관성을 점검하라(consistency cookbook 참고).
  - `none`과 `not_yet`을 맨 앞에 두지 마라.
- 공식 문서는 "none of the above" 옵션이 잘 동작하지 않을 수 있다고 하며, **companion Noul**(위 `any_case`)을 함께 쓰라고 권한다. https://docs.typesafe.ai/patterns/companion-noul

---

## 3. Confidence 보정과 임계값 설정 [검증 + 권고]

공식 가이드 요약(https://docs.typesafe.ai/confidence.md):
1. 결과를 세 구간으로 나눈다. 높음이면 자동 실행하고, 중간이면 확인 질문을 하거나 조심스럽게 진행하고, 낮음이면 실행하지 않는다(사람 연결, 되묻기, 폴백).
2. 위험도에 따라 임계값을 다르게 둔다. 되돌릴 수 없는 행동(주문 취소 실행 등)은 더 높게 잡는다.
3. **Choice는 confidence가 아니라 실행할 옵션의 확률 `p_top`으로 판단한다.** 2순위와 비교할 때는 top/2nd 비율을 쓴다.
4. Noul의 출발점은 **yes ≥ 0.7, no ≤ 0.3**이고, 그 사이 구간은 별도 결과로 처리한다.
5. 보수적으로 시작한다. 자체 라벨 데이터로 "높은 확신 구간의 실제 정답률"을 재고 조정한다. 모델이나 입력 분포가 바뀌면 다시 점검한다.
6. 학습 방식(RLCD)이 보정(calibration)을 목표로 한다고 벤더가 주장한다. 다만 보정은 집단 수준의 성질이지 개별 정답을 보장하지 않는다. [커뮤니티]

실무 절차(권고):
- 케이스별 라벨 발화를 200~500개 확보한다. 부분 발화 접두사도 포함해 "not_yet" 라벨을 만든다.
- JEV와 플랜 B를 모두 돌려 reliability diagram을 그리고 ECE를 계산한다.
- `p_top` 구간별 정밀도를 보고 **정밀도 ≥ 95%가 되는 최소 임계값**을 자동 실행선으로 삼는다.
- 위 작업은 반드시 버전을 고정한 `jev-1.14.0`으로 한다.

---

## 4. 플랜 B: 로컬 임베딩 유사도

### 4.1 한국어 임베딩 모델 비교
| 모델 | 기반 | 파라미터 | 차원 | 최대 길이 | 라이선스 | 비고 |
|---|---|---|---|---|---|---|
| **nlpai-lab/KURE-v1** | BAAI/bge-m3 파인튜닝 | ~568M | 1024 (dense) | 8192 | **MIT** | 한국어 검색 벤치마크에서 bge-m3와 BGE-m3-ko보다 높다. sentence-transformers로 바로 쓸 수 있다. [검증] https://huggingface.co/nlpai-lab/KURE-v1 |
| **nlpai-lab/KURE-v2** | skt/A.X-Encoder-base | 154M | **토큰당 128 (multi-vector, ColBERT식 MaxSim)** | 8192 | Apache-2.0 | MTEB(kor, v2) 검색 nDCG@10 0.816(KURE-v1은 0.762). 질의 인코딩 13.8ms(카드 기준). **단일 벡터가 아니어서** PyLate나 `MultiVectorEncoder`가 필요하다. 짧은 발화 분류에 쓸 때는 MaxSim으로 직접 구현해야 한다. [검증] https://huggingface.co/nlpai-lab/KURE-v2 |
| dragonkue/BGE-m3-ko | bge-m3 | 568M | 1024 | 8192 | 모델 카드 확인 필요 [미검증] | KURE-v2 표 기준 0.7547 |
| BAAI/bge-m3 | XLM-R large | 568M | 1024 | 8192 | MIT | 다국어 기본선(0.7509) |
| nlpai-lab/KoE5 | multilingual-e5-large | 560M | 1024 | 512 | 카드 확인 필요 [미검증] | 0.7337. 입력 앞에 `query: ` 접두사가 필요하다. |
| intfloat/multilingual-e5-small / base / large | XLM-R | 118M / 278M / 560M | 384 / 768 / 1024 | 512 | MIT | small은 CPU에서 매우 빠르다. 한국어 의도 분류 품질은 실측이 필요하다. [일반 지식, 미검증] |
| OpenAI text-embedding-3-small (API) | - | - | 1536 (축소 가능) | 8191 | 상용 API | 서드파티 측정 p50 약 230~270ms. 미국 리전 왕복이 더해지므로 **로컬보다 느리고 JEV와 지연 특성이 같다.** 플랜 B로는 부적합하다. [커뮤니티] https://openrouter.ai/openai/text-embedding-3-small |

- 주의: 위 벤치마크는 **문서 검색(retrieval)** 성능이다. 짧은 구어 발화의 의도 분류 성능과는 다를 수 있다. 자체 평가가 필수다.

### 4.2 추론 지연(짧은 문장 10~30토큰, batch=1) [미검증 추정, 실측 필수]
| 구성 | 추정 지연 |
|---|---|
| bge-m3/KURE-v1 (568M) GPU(T4/A10급) fp16 | 약 5~15ms |
| bge-m3/KURE-v1 CPU fp32 (8코어 x86) | 약 40~120ms |
| bge-m3/KURE-v1 CPU ONNX int8 | 약 20~60ms |
| multilingual-e5-small CPU | 약 5~20ms |
| KURE-v2 (154M) | 카드 기준 질의 13.8ms. 하드웨어 조건은 명시되지 않았다. |

- bge-m3 계열의 짧은 문장 CPU 지연을 공식 수치로 측정한 자료는 찾지 못했다. 공개된 ONNX 카드의 수치는 긴 문서 기준이다.
- **배포 서버에서 batch=1로 직접 측정할 것.** `torch.set_num_threads`나 ONNX Runtime intra-op 스레드 수를 고정해서 잰다.
- asyncio 서버에서는 인코딩이 CPU 바운드이므로 이벤트 루프를 막는다. **`asyncio.to_thread` 또는 별도 프로세스 풀**에서 실행해야 한다. GPU라면 전용 워커 하나에 큐를 두는 방식을 권장한다.

### 4.3 semantic-router (aurelio-labs) [검증: GitHub 소스와 PyPI]
- 최신 버전은 **1.1.2**(2026-11-09)이고 Python 3.10~3.13, MIT 라이선스다. 1.x는 2026-10월 말에 나온 **매우 새 버전**이며, 0.x는 버그 수정만 받는다.
- 1.x 구조
  - `Route(name, utterances=[...])` 목록으로 `IndexedRouter(encoder, routes, top_k=5, aggregation="mean", confidence_threshold=None, calibrator=None, sparse_encoder=None, alpha=0.7)`를 만든다.
  - 기본 인덱스는 메모리 기반 `LocalIndex`다. Pinecone, Qdrant, Postgres 같은 외부 인덱스는 선택 사항이다.
  - 질의 벡터와 가장 가까운 예문 top_k개를 찾고, 라우트별 점수를 집계해(mean 등) 가장 높은 라우트를 고른다. 라우트별로 `confidence_threshold`를 따로 줄 수도 있다.
  - `router.fit(X, y)`에서 `y=None`은 "어느 라우트에도 해당 없음"을 뜻한다. 라벨 데이터로 **Calibrator를 학습해 원점수를 확률로 보정**하고, 정확도와 ECE를 담은 `FitReport`를 돌려준다. `evaluate(X, y)`로는 정확도를 낸다.
  - `acall`과 `afit`라는 async API가 있다.
- 로컬 인코더
  - `LocalEncoder(name="nlpai-lab/KURE-v1")`: sentence-transformers 기반이다.
  - `HuggingFaceEncoder(name=...)`: transformers 기반이다.
  - `FastEmbedEncoder`: ONNX 기반이다. 단, KURE가 지원 목록에 있는지는 [미검증].
  - 설치는 `pip install "semantic-router[local]"`.
- 출처: https://github.com/aurelio-labs/semantic-router , https://pypi.org/project/semantic-router/

**쓸 만한가, 아니면 numpy 50줄로 충분한가?**
- 우리 규모(케이스 22개 × 예문 20~50개, 즉 벡터 수백~천여 개)에서는 인덱스가 필요 없다. 행렬곱 한 번(0.1ms 미만)이면 된다.
- semantic-router의 실질적인 가치는 **`fit()`의 확률 보정과 None 라벨 처리**뿐이다. 이것은 scikit-learn의 `LogisticRegression`이나 `IsotonicRegression` 수십 줄로 재현할 수 있다.
- 1.x가 막 나와 API가 바뀔 위험이 있다. 내부의 `run()` sync/async 래핑 때문에 우리 이벤트 루프와 섞일 때 복잡도가 생긴다. 인코더 호출이 루프를 막지 않는지도 검증이 필요하다.
- **권고: 직접 구현(numpy + 선택적으로 sklearn 보정).** semantic-router는 보정 방식과 top_k 집계를 참고하는 용도로만 쓴다.

직접 구현의 골자:
```python
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbedRouter:
    def __init__(self, model_name: str, examples: dict[str, list[str]], top_k: int = 3):
        self.m = SentenceTransformer(model_name, device="cuda")  # 또는 "cpu"
        self.labels, texts = [], []
        for case, utts in examples.items():
            self.labels += [case] * len(utts); texts += utts
        self.E = self.m.encode(texts, normalize_embeddings=True)       # (N, d)
        self.labels = np.array(self.labels); self.cases = list(examples)
        self.top_k = top_k

    def scores(self, text: str) -> dict[str, float]:
        q = self.m.encode([text], normalize_embeddings=True)[0]
        sims = self.E @ q                                               # 코사인
        out = {}
        for c in self.cases:
            s = np.sort(sims[self.labels == c])[::-1][: self.top_k]
            out[c] = float(s.mean())                                    # 케이스별 top-k 평균
        return out

    def decide(self, text: str, thr: float = 0.65, margin: float = 0.05):
        sc = sorted(self.scores(text).items(), key=lambda kv: kv[1], reverse=True)
        (c1, s1), (_, s2) = sc[0], sc[1]
        return (c1 if s1 >= thr and s1 - s2 >= margin else None), s1, s1 - s2
# 보정: (s1, s1-s2) 특징 → sklearn LogisticRegression으로 P(정답) 학습 → JEV p_top과 같은 척도로 비교
# 호출은 await asyncio.to_thread(router.decide, text)
```

- 턴 종료 판단은 임베딩 유사도로 하기 어렵다. 플랜 B에서는 **규칙 기반**을 권장한다.
  - 종결어미 정규식: `(요|니다|니까|까요|세요|죠|네요|래요|게요)[.?!]?$`이거나 "네", "아니요" 같은 단독 응답이면 종료로 본다.
  - 연결어미나 지시어로 끝나면 비종료로 본다: `(는데|고|서|면|니까|거든|그|저기|어|음)$`.
  - 여기에 VAD 무음 길이를 결합한다.
  - 여유가 있으면 소형 분류기(e5-small 임베딩 + 로지스틱 회귀)를 붙인다.

---

## 5. 한국어 의도 분류 팁

1. **SOV 어순: 의도를 나타내는 동사가 맨 끝에 온다.**
   - "어제 주문한 니트를 … 취소/교환/환불"처럼 부분 STT 단계에서는 의도가 드러나지 않는 경우가 많다.
   - 그래서 `not_yet` 클래스가 필수다. 부분 발화에서는 명사구만 보고 판정을 확정하지 말고 **잠정 판정(프리페치용)**으로만 쓴다.
2. **부정**
   - 부정 표현: "안 왔어요", "못 받았어요", "취소 안 할게요", "환불 말고 교환", "-지 않-", "-지 말-", "없어요".
   - 임베딩은 부정을 잘 구별하지 못한다. "취소할게요"와 "취소 안 할게요"의 코사인 유사도가 높게 나온다.
   - 대응 1: 예문 세트에 **부정 하드 네거티브**를 넣는다. 예를 들어 `cancel_order`의 반례를 `none`이나 별도 케이스 예문으로 등록한다.
   - 대응 2: 부정어 정규식 플래그로 후처리한다.
   - 대응 3: "A 말고 B" 패턴에서는 B를 우선한다.
   - JEV도 이중부정과 간접 표현에 약하다고 공식 문서에 나온다. instructions에 부정을 처리하라는 지시를 명시한다.
3. **존댓말과 종결어미 변이**
   - 같은 의도라도 "-해 주세요", "-해줘", "-하고 싶은데요", "-할 수 있나요", "-되나요", "-하려고요"처럼 변이가 많다.
   - 예문은 **어미 변이를 섞어 케이스당 20~50개**를 준비한다. 반말, 해요체, 합쇼체, 의문형, 평서형을 모두 포함한다.
   - 임베딩 전에 어미를 정규화하는 것은 오히려 정보를 잃을 수 있다(턴 종료 판단에 필요하다). **분류용 텍스트와 턴 종료용 텍스트를 분리**한다.
4. **간투사와 STT 노이즈**
   - "어, 음, 그, 저기, 아니 근데" 같은 간투사는 앞부분만 제거한다. 단, "아니요"는 의미가 있으므로 남긴다.
   - 띄어쓰기 오류("취소할게요", "취소 할게요")는 공백을 정규화해 처리한다.
   - 숫자와 주문번호는 마스킹한다(예: `<주문번호>`).
5. **짧은 대답의 문맥 의존**
   - "네", "아니요", "그거요", "두 번째요"는 직전 봇 발화 없이는 분류할 수 없다.
   - JEV에는 `agent_last_turn`을 state로 넣는다.
   - 플랜 B에서는 이런 대답을 별도 "대화 행위(긍정/부정/선택)" 소분류로 처리하고 현재 대화 상태 머신에 맡긴다.
6. **복합 의도**
   - 예: "교환하거나 안 되면 환불해 주세요".
   - Choice는 하나만 고르므로 2순위 확률과 비율을 보고 되묻기를 판단한다. 필요하면 케이스별 Noul을 다중 라벨로 쓴다(공식 권장 패턴).

---

## 6. 설계에 주는 시사점

### 6.1 JEV가 지연 예산 안에 들어오는가
- **최종 STT 이후 동기 호출 1회에 250ms 타임아웃을 거는 방식은 위험하다.**
  - 서버가 한국이고 서비스는 미국 서부(리전 미공개, 커뮤니티 정보)라는 점을 감안하면 예상 왕복은 200~650ms다.
  - 이 경우 타임아웃 비율이 상당할 것으로 본다. 단, 이는 [미검증]이며 **1순위 작업은 한국 서버에서 실측하는 것**이다. warm 연결로 p50, p90, p99를 잰다.
- 대신 **부분 STT에 대한 추측(speculative) 호출**로 결과를 미리 확보해 두는 구조라면 1.5초 예산 안에서 충분히 활용할 수 있다. 질문을 묶어도 지연이 거의 늘지 않으므로 Choice와 턴 종료 Noul을 함께 받는다.
- 비용($0.042/Mtok)은 문제가 되지 않는다. **rate limit과 한국어 품질이 실제 제약**이다.

### 6.2 플랜 B와 병렬로 돌리는 방법 (권장 아키텍처)
1. 통화가 시작되면 `httpx.AsyncClient(http2=True)` 연결을 미리 열고 워밍업 호출을 한다. 핸드셰이크 비용을 대화 경로에서 빼기 위해서다.
2. 부분 STT가 갱신되면 다음을 한다.
   - **플랜 B(로컬)는 매번** 실행한다. `to_thread`나 GPU 워커에서 수~수십 ms가 걸린다.
   - **JEV는 디바운스(예: 텍스트가 2어절 이상 늘었거나 마지막 갱신 후 200ms가 지났을 때)** 조건에서 fire-and-forget으로 호출한다. 결과에는 "어떤 텍스트에 대한 결과인지"를 함께 저장한다. 텍스트가 바뀌면 이전 태스크는 취소하거나 결과를 버린다(신선도 체크).
3. 최종 STT 또는 턴 종료 시점에는 다음 순서로 결정한다.
   - (a) 최종 텍스트와 같거나 거의 같은 텍스트에 대한 JEV 결과가 이미 있으면 즉시 사용한다.
   - (b) 없으면 JEV와 플랜 B를 `asyncio.wait(..., timeout=0.25)`로 동시에 기다린다. JEV가 시간 안에 오면 사용하고, 아니면 플랜 B를 쓴다.
   - (c) 둘이 일치하면 확신도를 올리고, 불일치하면 둘 중 보정 확률이 높은 쪽을 택한다. 둘 다 낮으면 되묻기로 간다.
4. 턴 종료 판단은 **VAD 무음, 규칙 기반 어미 판정, (가능하면) JEV `turn_end` Noul**을 결합한다. JEV가 늦으면 규칙만으로 판단하고, 지연된 JEV 결과는 다음 부분 발화에 반영한다.
5. 모든 호출에서 `model`(답한 버전), 지연, 두 시스템의 판정을 로깅해 오프라인 보정 데이터로 쓴다.

### 6.3 임계값 출발점 (라벨 데이터로 반드시 재조정)
| 신호 | 자동 실행 | 확인/되묻기 | 미판정 |
|---|---|---|---|
| JEV Choice `p_top` | ≥ 0.70 그리고 top/2nd ≥ 2.0 | 0.40~0.70 (부분 발화에서는 프리페치만) | < 0.40 또는 `not_yet`이 1위 |
| JEV companion Noul `any_case` | ≥ 0.7이면 케이스 판정을 신뢰 | 0.3~0.7 | ≤ 0.3이면 `none` |
| JEV `turn_end` Noul | ≥ 0.70 그리고 무음 ≥ 200~300ms이면 응답 시작 | 0.3~0.7이면 무음 ≥ 600~700ms까지 대기 | ≤ 0.30이면 무음 1.0~1.2s까지 대기(안전 상한) |
| 플랜 B 코사인 (KURE-v1, top-3 평균) | s1 ≥ 0.65 그리고 s1−s2 ≥ 0.05 | 0.55~0.65 | < 0.55이면 none [미검증: 모델별 코사인 분포가 다르므로 첫 주 데이터로 재설정] |
| 되돌릴 수 없는 행동(취소/환불 접수) | 위 기준보다 높은 p ≥ 0.85와 명시적 확인 발화 | | |

### 6.4 다음 액션
1. 한국 서버에서 JEV 지연을 실측한다. warm 연결로 1,000회 호출해 p50, p90, p99를 구하고 타임아웃 비율을 본다. 그 결과로 250ms 타임아웃을 유지할지 판단한다.
2. 라벨된 한국어 CS 발화로 다음을 비교한다. 부분 발화 접두사를 포함한 300개 이상을 쓴다.
   - JEV 1.14와 1.13
   - instructions를 영어로 쓴 경우와 한국어로 쓴 경우
   - KURE-v1, bge-m3, e5-small
3. 플랜 B는 numpy로 직접 구현한다. semantic-router는 쓰지 않고, sklearn 보정을 붙인다.
4. TypeSafe 콘솔에서 rate limit을 확인한다. 부분 발화 호출 빈도와 동시 통화 수를 곱해 필요한 RPS를 산정한다.

---

## 출처
- TypeSafe 공식: https://docs.typesafe.ai/api.md , https://docs.typesafe.ai/models.md , https://docs.typesafe.ai/primitives/choice.md , https://docs.typesafe.ai/primitives/noul.md , https://docs.typesafe.ai/confidence.md , https://docs.typesafe.ai/api/errors.md , https://docs.typesafe.ai/changelog.md , https://docs.typesafe.ai/model-jaggedness.md , https://docs.typesafe.ai/sdk/python/api/clients/async.md
- 커뮤니티 가이드: https://typesafe-jev.com/en/guides/api/ (비공식)
- Pydantic AI: https://pydantic.dev/docs/ai/models/typesafe/
- innFactory 분석: https://innfactory.ai/en/blog/jev-system-one-model-classifier-not-llm/
- 지연 관련 2차 자료: https://www.eesel.ai/ja/blog/jev-ultrafast , https://pasqualepillitteri.it/en/news/16460/typesafe-jev-chatless-ai-beats-claude
- 임베딩: https://huggingface.co/nlpai-lab/KURE-v1 , https://huggingface.co/nlpai-lab/KURE-v2 , https://huggingface.co/nlpai-lab/KoE5 , https://huggingface.co/BAAI/bge-m3
- semantic-router: https://github.com/aurelio-labs/semantic-router , https://pypi.org/pypi/semantic-router/json
- OpenAI 임베딩 지연(서드파티): https://openrouter.ai/openai/text-embedding-3-small , https://www.orcarouter.ai/compare/gpt-5.4-pro-vs-text-embedding-3-small
