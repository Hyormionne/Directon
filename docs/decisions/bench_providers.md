# Provider 실측 (AG-T04)

- 측정: 2026-11-19 17:40~18:30 · `scripts/bench_providers.py` · Windows 개발 PC(한국 가정용 회선 추정) · `GEMINI_API_KEY` **무료 티어**
- 단위: 초. p50/p95는 성공한 시도 기준. 원시 로그는 남기지 않았다(스크립트를 다시 돌리면 이 파일을 덮어쓴다. 아래 표는 여러 번의 실행을 손으로 합친 것).
- **측정 조건의 한계**: 같은 키를 다른 에이전트(테스트 음성 합성 등)가 동시에 쓰고 있었고, Gemini Flash 계열은 503(high demand)이 잦았다. TTS는 무료 티어 한도(모델당 하루 10요청)를 이미 소진해 Gemini TTS 표본이 매우 적다.

## 요약

| 구간 | 권장 | p50 / p95 | 비고 |
|---|---|---|---|
| STT | `gemini-3.5-transcribe-live` (live) | finalize→최종 텍스트 **0.43 / 0.47** | CER 0(10/10). batch는 2.2~10.9초 + 503 잦음 → 폴백 전용 |
| TTS | **edge-tts `ko-KR-SunHiNeural`** (AD6) | TTFB **0.47 / 0.67** | Gemini TTS는 무료 티어 하루 10요청 → 실시간 경로 불가. 캐시 사전 합성에만 |
| LLM | **`gemini-3.5-flash-lite`** (thinking MINIMAL) | 첫 이벤트 **0.78 / 2.96**, 툴 결정 1.28 / 3.06 | 정확도 7/7, 툴 결과 왕복 OK(0.77s) |

예상 본 응답 지연(말 끝 → 본 응답 첫 소리): STT finalize 0.43 + LLM 첫 문장 ~0.8~1.3 + TTS TTFB 0.47 ≈ **1.7~2.2초(p50)**. 1.5초 목표는 캐시된 첫 마디로 채우고, 본 응답은 첫 마디 재생(1~2초) 뒤에 이어 붙이는 구조가 필요하다.

## STT (한국어 합성 음성 5문장, 실시간 스트리밍)

지연 = 마지막 음성 프레임 송신 → `finalize()` 반환(최종 텍스트). 송신은 100ms 묶음, finalize는 `audio_stream_end`. 측정 중에도 무음 프레임을 계속 보냈다(평가기와 같은 조건). batch는 이 시간에 전사 요청 전체가 들어간다.

| 방식 | 모델 | final p50 | final p95 | 평균 | 성공 | 첫 부분결과 p50(말 시작부터) | 연결 p50 | CER 평균 |
|---|---|---|---|---|---|---|---|---|
| live | `gemini-3.5-transcribe-live` | **0.43** | **0.47** | 0.41 | 10/10 | 1.32 | 0.37 | **0.000** |
| batch | `gemini-3.5-transcribe` | 2.49 | 10.37 | 5.25 | 2/5 (503 3회) | - | - | 성공분 0.000 |

- 테스트 음성: Gemini TTS 한도 소진으로 일부는 edge-tts로 합성(`gemini-3.8-flash-lite-tts:Puck`, `edge:ko-KR-SunHiNeural`, `gemini-3.8-flash-lite-tts:Charon`, `edge:ko-KR-SunHiNeural`, `edge:ko-KR-InJoonNeural`). 깨끗한 합성음이라 실제 통화(잡음·μ-law)보다 쉽다. 통화 환경 정확도는 AG-T09 테스트셋으로 다시 본다.
- 사전 실측(숫자): "주문번호 이육일일 일오공 이삼일 배송 언제 와요?" → `주문 번호 2611150231 배송 언제 와요?` (stream_end 후 0.37~0.46초). 숫자는 아라비아 숫자로 나온다.
- stream_end 없이 무음만 보내면 서버 VAD가 약 **1.0초** 뒤에 확정했다 → finalize를 꼭 부를 것.
- 같은 세션에서 stream_end 뒤에도 오디오를 계속 보낼 수 있다(턴마다 재연결 불필요).
- batch(`gemini-3.5-transcribe`)는 응답이 `parts[].audio_transcription.text`에 온다(`resp.text`는 None). 6초 음성에 서버 처리만 8.3초였다.

<details><summary>전사 결과</summary>

- live: `주문한 운동화가 아직 안 왔는데 언제 도착하나요?` (CER 0.00)
- live: `어제 산 바지를 교환하고 싶어요.` (CER 0.00)
- live: `배송지를 회사 주소로 바꿀 수 있을까요?` (CER 0.00)
- live: `환불은 보통 며칠 걸리나요?` (CER 0.00)
- live: `상담원이랑 직접 통화하고 싶습니다.` (CER 0.00)
- live: `주문한 운동화가 아직 안 왔는데 언제 도착하나요?` (CER 0.00)
- live: `어제 산 바지를 교환하고 싶어요.` (CER 0.00)
- live: `배송지를 회사 주소로 바꿀 수 있을까요?` (CER 0.00)
- live: `환불은 보통 며칠 걸리나요?` (CER 0.00)
- live: `상담원이랑 직접 통화하고 싶습니다.` (CER 0.00)
- batch: `` (CER 1.00)
- batch: `` (CER 1.00)
- batch: `배송지를 회사 주소로 바꿀 수 있을까요?` (CER 0.00)
- batch: `` (CER 1.00)
- batch: `상담원이랑 직접 통화하고 싶습니다.` (CER 0.00)

</details>

## TTS (짧은 문장 11~20자)

edge: `Communicate.stream()` MP3 → ffmpeg 파이프 → 16k. Gemini: `generate_content_stream` + 24k→16k soxr 스트림. TTFB = 요청 → 첫 16k PCM 덩어리.

| 모델 | TTFB p50 | TTFB p95 | 전체 p50 | 전체 p95 | 성공 | 16k 직접 출력 |
|---|---|---|---|---|---|---|
| `edge:ko-KR-SunHiNeural` | **0.47** | **0.67** | 0.72 | 0.89 | 10/10 | 해당 없음(MP3 고정) |
| `edge:ko-KR-InJoonNeural` | 0.59 | 1.41 | 0.86 | 1.72 | 10/10 | 해당 없음 |
| `gemini-3.8-flash-tts` | 1.10 (n=1) | - | 1.58 (n=1) | - | 1/1, 이후 **한도 소진** | **미지원**(아래) |
| `gemini-3.8-flash-lite-tts` | 1.05 (n=1) | - | 1.32 (n=1) | - | 1/1, 이후 **한도 소진** | 미확인(한도 소진) |
| `gemini-2.5-flash-preview-tts` | 2.56 | 3.27 | 2.56 | 3.27 | 3/4, 이후 **한도 소진** | **미지원**(generateContent 400) |

- n=1 수치는 같은 날 사전 실측(`"안녕하세요, 무엇을 도와드릴까요?"`). 2.5-flash-preview-tts는 전체를 한 덩어리로 준다(스트리밍 효과 없음, TTFB=전체).
- **Gemini TTS 무료 티어 한도**: `GenerateRequestsPerDayPerProjectPerModel-FreeTier` = **모델당 하루 10요청**(429 응답의 quotaValue). 상담원 실시간 TTS로 쓸 수 없다 → AD6(개발 중 edge-tts).
- **16kHz 직접 출력 미지원**: Interactions API `response_format.sample_rate: 16000`(문서에 있음) → 400 invalid_request(스트림·단건 모두, 8000도 400). generateContent `response_mime_type: audio/l16;rate=16000` → 400(허용 mime은 text/json 계열만). 서버는 `audio/l16; rate=24000`만 준다 → soxr 변환 유지.
- **Interactions API 스트림은 느리다**: 같은 문장 첫 덩어리 3.59초 vs `generate_content_stream` 1.10초 → Provider는 generateContent 스트림을 쓴다.
- edge-tts: 서비스가 `raw-16khz-16bit-mono-pcm`, `raw-24khz-...`, `audio-16khz-32kbitrate-mono-mp3`, webm/opus를 모두 거부했다(라이브러리 출력 형식 문자열을 바꿔 시험). MP3 24k만 된다 → ffmpeg 디코드. ffmpeg 기동은 edge 연결과 병렬.
- 스타일 지시(`스타일: 본문` 프롬프트)가 음성으로 읽히는지는 한도 소진으로 확인하지 못했다. `GEMINI_TTS_STYLE` 기본값은 비워 둔다.

## LLM (가짜 툴 14개, 질문 7개: 툴 5 + 툴 없음 2, 스트리밍)

첫 이벤트 = 요청 → 첫 텍스트 또는 툴 호출. 툴 결정 = 요청 → 첫 tool_call(툴 질문만). 정확도 = 기대 함수 포함(툴 없음 질문은 호출하지 않아야 정답). 질문당 1회(무료 티어 한도 때문에 10회 반복 대신 7문항 × 1).

| 모델 | 첫 이벤트 p50 | p95 | 툴 결정 p50 | p95 | 정확도 | 성공 | thinking(자동 선택) | 툴 결과 왕복 |
|---|---|---|---|---|---|---|---|---|
| `gemini-3.5-flash-lite` | **0.78** | **2.96** | **1.28** | **3.06** | **7/7** | 7/7 | MINIMAL | OK 0.77s |
| `gemini-3.1-flash-lite` | 3.81 | 5.51 | 4.22 | 5.85 | 5/7 | 7/7 | MINIMAL | OK 1.69s |
| `gemini-3.8-flash` | - | - | - | - | - | 0/7 (30초 시간 초과 6, 429 1; 재시도 503·429) | MINIMAL 불가(400) → budget 0 | - |
| `gemini-3.5-flash` | - | - | - | - | - | 0/7 (시간 초과 3, 503 3; 재시도 시간 초과 3) | - | - |

- 사전 실측(같은 날, 툴 1개): `gemini-3.8-flash` 첫 응답 **337초**, `gemini-3.5-flash` 10~73초, `gemini-3.1-flash-lite` 2.3~3.0초, `gemini-3.5-flash-lite` 8.9초(그 시점 혼잡). Flash 계열은 지금 무료 티어에서 사실상 쓸 수 없다.
- thinking 허용 범위: 3.8-flash는 `thinking_level=MINIMAL` 400·`thinking_budget=0` 허용, 3.5-flash-lite는 MINIMAL 허용·budget 0 400, 3.1-flash-lite·3.5-flash는 둘 다 허용. `GeminiLLM`은 MINIMAL → budget 0 → LOW → 미지정 순으로 자동 시도하고 모델별로 기억한다.
- 툴 결과 왕복(`thought_signature`): 첫 응답의 서명을 `GeminiLLM`이 호출 id별로 기억했다가 다음 요청의 function_call 파트에 붙인다. 두 lite 모델 모두 400 없이 왕복했다.
- 3.1-flash-lite 오답 2건: 재고 질문에 상품명 되묻기, 상담원 연결 요청에 사유 되묻기(함수 미호출).

<details><summary>툴 호출 상세</summary>

- `gemini-3.5-flash-lite` 주문번호 2611150231 배송 어디쯤 왔어요? → ['track_delivery'] [{'order_id': '2611150231'}] OK  ''
- `gemini-3.5-flash-lite` 제 번호 01012345678로 주문한 거 목록 좀 보여주세요. → ['get_order_list'] [{'phone_number': '01012345678'}] OK  ''
- `gemini-3.5-flash-lite` 주문번호 2611150231 취소해 주세요. 단순 변심이에요. → ['cancel_order'] [{'reason': '단순 변심', 'order_id': '2611150231'}] OK  ''
- `gemini-3.5-flash-lite` 검정 후드티 L 사이즈 재고 있어요? → ['check_stock'] [{'product_name': '후드티', 'option': '검정 L'}] OK  ''
- `gemini-3.5-flash-lite` 그냥 상담원 연결해 주세요. → ['transfer_to_agent'] [{'reason': '상담원 연결 요청'}] OK  ''
- `gemini-3.5-flash-lite` 영업시간이 어떻게 되나요? → [] [] OK  '저희 디렉트몰 고객센터 영업시간은 평일 오전 9시부터 오후 6시까지입니다'
- `gemini-3.5-flash-lite` 반품하고 싶은데요. → [] [] OK  '반품을 도와드리기 위해, 주문하신 분의 휴대폰 번호와 주문번호를 말씀해 '
- `gemini-3.1-flash-lite` 주문번호 2611150231 배송 어디쯤 왔어요? → ['track_delivery'] [{'order_id': '2611150231'}] OK  ''
- `gemini-3.1-flash-lite` 제 번호 01012345678로 주문한 거 목록 좀 보여주세요. → ['get_order_list'] [{'phone_number': '01012345678'}] OK  ''
- `gemini-3.1-flash-lite` 주문번호 2611150231 취소해 주세요. 단순 변심이에요. → ['cancel_order'] [{'order_id': '2611150231', 'reason': '단순 변심'}] OK  ''
- `gemini-3.1-flash-lite` 검정 후드티 L 사이즈 재고 있어요? → [] [] X  '네, 고객님. 정확한 재고 확인을 위해 상품명을 말씀해 주시겠어요?'
- `gemini-3.1-flash-lite` 그냥 상담원 연결해 주세요. → [] [] X  '죄송합니다만, 어떤 문제로 상담원 연결을 원하시는지 말씀해 주시겠어요? '
- `gemini-3.1-flash-lite` 영업시간이 어떻게 되나요? → [] [] OK  '저희 디렉트몰 고객센터 운영 시간은 평일 오전 9시부터 오후 6시까지입니'
- `gemini-3.1-flash-lite` 반품하고 싶은데요. → [] [] OK  '반품 접수를 도와드리겠습니다. 주문하신 번호를 알려주시겠어요?'

</details>

## 권장

1. **STT: `gemini-3.5-transcribe-live`** (factory 기본, 연결 실패 시 batch 폴백). 턴 감지기가 말 끝을 잡으면 즉시 `finalize()`(약 0.43초). 세션 시작 시 미리 연결(연결 0.37초)해 두는 것이 좋다. Soniox 키가 생기면 같은 테스트로 비교.
2. **TTS: edge-tts `ko-KR-SunHiNeural`** (factory 기본 `DIRECTON_TTS_PROVIDER=edge`, AD6). TTFB p50 0.47초. Gemini TTS는 유료 티어 전환 전까지 캐시 문구 사전 합성에만 쓰고, 하루 10요청 한도를 테스트 음성 합성과 나눠 써야 한다.
3. **LLM: `gemini-3.5-flash-lite`** (factory 기본 `GEMINI_LLM_MODEL`). 첫 이벤트 p50 0.78초, 툴 정확도 7/7. p95가 3초로 길어 첫 마디 캐시 + 필러가 필수. OpenAI 키가 생기면 `gpt-5.4-mini`(reasoning none)와 같은 질문으로 비교.
4. 유료 티어(Tier 1) 전환을 사용자에게 권한다: Gemini TTS 한도·Flash 503·LLM 일일 한도(3.8-flash에서 429 발생)가 모두 무료 티어 문제다.
