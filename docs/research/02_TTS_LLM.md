# 02. TTS / LLM 리서치 (Gemini TTS, OpenAI LLM)

- 작성일: 2026-11-19
- 목적: 한국어 음성 상담 에이전트(Python asyncio, WebSocket으로 16kHz/16-bit/mono PCM 송신, "발화 종료 → 첫 유성 오디오" ≤1.5s 목표, HTTP Mock API 툴 ~14개 function calling)의 TTS·LLM 선택 근거 정리
- 표기: **[확인]** = 공식 문서에서 직접 확인, **[미검증]** = 서드파티/커뮤니티 출처이거나 문서에 명시가 없어 실측 필요
- 주의: 2026-11 시점 모델 라인업이 초안(Gemini 2.5 Flash TTS + GPT-5.4 mini)을 작성할 때와 크게 달라졌음. 아래 1절 "요약"부터 먼저 볼 것.

---

## 0. 핵심 요약

| 항목 | 초안 | 현재(2026-11) 상황 | 권고 |
|---|---|---|---|
| TTS | Gemini 2.5 Flash TTS (`gemini-2.5-flash-preview-tts`) | Gemini API 문서의 현행 TTS 모델은 **`gemini-3.8-flash-tts`, `gemini-3.8-flash-lite-tts`**(GA로 보임). 2.5 Flash preview TTS는 가격표엔 남아 있으나 speech-generation 문서에서 빠짐 | **`gemini-3.8-flash-tts`** (또는 지연이 더 중요하면 `-lite-`)로 교체 검토. **`sample_rate: 16000`을 직접 요청할 수 있으므로 리샘플링을 생략할 수 있음** |
| TTS API 형태 | `generate_content` + `response_modalities=["AUDIO"]` | 공식 예제는 모두 새 **Interactions API**(`client.interactions.create(..., stream=True)`) 기반. 오디오 청크 스트리밍 지원 | Interactions API + `stream=True` 사용, 첫 청크 도착 시간(TTFA) 실측 필수 |
| LLM | GPT-5.4 mini | **`gpt-5.4-mini`는 존재**(스냅샷 `gpt-5.4-mini-2026-03-17`), 셧다운 예정 없음. 단 더 새로운 **GPT-6 계열**(`gpt-6-luna`, `gpt-6.1-terra`, `gpt-6.2-sol` 등)이 출시됨 | `gpt-5.4-mini`(reasoning `none`이 기본값이라 TTFT 유리)를 기본으로, **`gpt-6-luna`(훨씬 저렴)**를 A/B 후보로 둘 것 |
| 오디오 출력 | - | Gemini TTS 기본 24kHz s16le mono. 16k로 직접 받을 수 있음(문서 명시) | 16k 직접 요청 → 안 되면 `soxr` 스트리밍 리샘플러 |

---

## A. Gemini TTS

### A-1. 모델 ID, 제공 상태, 한국어, 보이스, 스타일

**Gemini API (AI Studio 키) – speech-generation 문서 기준 [확인]**
출처: https://aistudio.google.com/docs/speech-generation (구 URL https://ai.google.dev/gemini-api/docs/speech-generation 에서 301 리다이렉트)

| 모델 ID | 상태 | 비고 |
|---|---|---|
| `gemini-3.8-flash-tts` | 현행(Preview 접미사 없음) | 130개 이상 언어, 한국어 지원 |
| `gemini-3.8-flash-lite-tts` | 현행 | 100개 이상 언어, 한국어 지원. 마이그레이션 가이드상 `gemini-3.1-flash-tts-preview`의 후속 |
| `gemini-3.1-flash-tts-preview` | 이전 preview (문서에 "supported"로 남아 있음) | 2026-04 공개 |
| `gemini-2.5-pro-preview-tts` | 이전 preview | 무료 티어 없음 |
| `gemini-2.5-flash-preview-tts` | **speech-generation 문서에서 언급 사라짐**, 가격표에는 남아 있음 | 신규 설계에 쓰지 말 것 [상태 미검증] |

- OpenRouter 등에 `gemini-3.8-flash-tts-20260922` 스냅샷 표기가 있어 출시일은 2026-09-22 전후로 추정 [미검증].
- 신규 프로젝트 제한: 문서에 신규 프로젝트 사용 제한 언급은 없음. 무료 티어 사용 가능(가격표 기준). 단 일부 프로젝트는 verification 필요(rate-limits 문서) [확인].
- **한국어**: 3.8 Flash / Flash-Lite TTS 모두 한국어 지원, 언어 자동 감지 [확인].
- **보이스(30종 prebuilt)** [확인]: Zephyr, Puck, Charon, Kore, Fenrir, Leda, Orus, Aoede, Callirrhoe, Autonoe, Enceladus, Iapetus, Umbriel, Algieba, Despina, Erinome, Algenib, Rasalgethi, Laomedeia, Achernar, Alnilam, Schedar, Gacrux, Pulcherrima, Achird, Zubenelgenubi, Vindemiatrix, Sadachbia, Sadaltager, Sulafat.
  - 추가로 Extended Voice Library(`GET /v1beta/voices`), Voice design / replication / remixing 커스텀 보이스(`voice_...` 상태형 프로젝트당 200개·1년 TTL, `voicekey_...` 무상태형 7일 TTL) 존재.
  - 상담원 톤 후보: Kore(차분/단단), Aoede, Leda, Achernar 등 → 한국어로 직접 청취 비교 필요 [미검증, 주관].
- **스타일 제어** [확인]:
  - 턴 단위: 텍스트 항목의 `annotations`에 `speech_metadata.style` (예: `"친절하고 차분한 상담원 톤, 약간 빠르게"`).
  - 인라인 태그: `<laugh>`, `<sigh>`, `<short pause>`, `<long pause>` 등. **비영어 텍스트에서도 태그는 영어로 유지** 권장.
  - 대문자 강조(한국어에는 무의미), 백채널 `|oh really?|`.
- Cloud Text-to-Speech(Gemini-TTS) 쪽 모델 ID는 별도 체계 [확인]: `gemini-3.1-flash-tts-preview`, `gemini-2.5-flash-tts`, `gemini-2.5-flash-lite-preview-tts`, `gemini-2.5-pro-tts`. 배너에는 3.8 Flash/Flash-Lite TTS GA 언급이 있으나 모델 표·코드 예제에는 아직 없음 [미검증]. `ko-KR` GA.
  출처: https://docs.cloud.google.com/text-to-speech/docs/gemini-tts

### A-2. 출력 포맷과 리샘플링

**Gemini API 출력 [확인]**
- Unary(비스트리밍): 기본 `audio/wav` (24kHz, mono, s16le, 44바이트 RIFF 헤더 포함).
- Streaming: 기본 `audio/l16` = **헤더 없는 raw PCM 24kHz mono s16le**.
- `response_format.mime_type`: `audio/wav`, `audio/l16`, `audio/mulaw`, `audio/alaw` (μ/A-law는 8kHz).
- **`response_format.sample_rate`(선택, 기본 24000; 문서 예시로 16000, 8000 언급)** → `{"type":"audio","mime_type":"audio/l16","sample_rate":16000}`로 요청하면 **서버에서 바로 16k PCM**을 받을 수 있음. 실제 동작은 실측 확인 필요 [문서 확인 / 동작 미검증].

**리샘플링이 필요할 경우 (24000 → 16000 = 2/3)**
- 일괄(문장 단위 버퍼, 캐시 문구 사전 합성): `scipy.signal.resample_poly(x, up=2, down=3)` — 1초 오디오 기준 수 ms 이하(CPU 의존) [일반 지식, 미측정].
- **스트리밍 청크 단위**: 청크마다 `resample_poly`를 독립 호출하면 경계에서 필터 과도응답(클릭/틱)이 생김. → 상태를 유지하는 스트리밍 리샘플러 권장:
  - `soxr` (pip `soxr`)의 `soxr.ResampleStream(24000, 16000, 1, dtype='int16', quality='HQ')` → `resample_chunk(chunk)`; 마지막에 `last=True`로 flush. 추가 지연은 수 ms 수준 [미검증, 실측 권장].
  - 대안: `scipy.signal.upfirdn` + 필터 상태 직접 관리(복잡).
- 청크 바이트 정렬 주의: s16le이므로 홀수 바이트 청크가 오면 1바이트를 다음 청크로 이월.

```python
import numpy as np, soxr

class Resampler24to16:
    def __init__(self):
        self._rs = soxr.ResampleStream(24000, 16000, 1, dtype="int16", quality="HQ")
        self._carry = b""
    def feed(self, pcm24: bytes, last: bool = False) -> bytes:
        buf = self._carry + pcm24
        n = len(buf) // 2 * 2
        self._carry = buf[n:]
        x = np.frombuffer(buf[:n], dtype=np.int16)
        return self._rs.resample_chunk(x, last=last).tobytes()

# 일괄 변환(캐시용)
from scipy.signal import resample_poly
def to16k(pcm24: bytes) -> bytes:
    x = np.frombuffer(pcm24, dtype=np.int16).astype(np.float32)
    y = resample_poly(x, 2, 3)
    return np.clip(y, -32768, 32767).astype(np.int16).tobytes()
```

### A-3. 스트리밍

**Gemini API (Interactions API) [확인]**
- `client.interactions.create(model=..., input=[...], response_format={"type":"audio"}, generation_config={"speech_config":[{"voice":"Kore"}]}, stream=True)` → 이벤트 이터러블.
- `event.event_type == "step.delta"` 이고 `event.delta.type == "audio"`이면 `base64.b64decode(event.delta.data)`가 PCM 청크.
- **텍스트 입력 스트리밍(LLM 토큰을 흘려 넣기)은 미지원** — 요청 1건 = 완결된 텍스트. 실시간 양방향은 Live API로 안내됨. → **문장/구 단위로 잘라 여러 요청**을 보내는 구조가 필요.
- `generateContent`: speech-generation 문서에는 예제가 사라졌고, 마이그레이션 노트에 "generateContent에서 `AUDIO_L16`으로 raw PCM 요청 가능" 정도만 언급 [확인]. Cloud 문서의 Vertex AI 예제는 `generate_content_stream` + `SpeechConfig`로 inline audio를 누적 [확인].

**Cloud Text-to-Speech `StreamingSynthesize` (Gemini-TTS) [확인]**
- 양방향 스트리밍: 첫 요청에 config(voice name, language_code, model_name), 이후 텍스트 청크(LLM 제너레이터에서 비동기 송신 가능).
- **"클라이언트가 Half-Close를 보내야 합성 시작"** → 즉, LLM 토큰을 흘려 넣어도 입력이 끝나야 합성 시작이므로, 실질 지연 이득이 없음. 문장 단위로 스트림을 새로 여는 편이 낫다.
- `prompt`(스타일)는 **첫 입력 청크에만** 유효.
- 스트리밍 출력 포맷: PCM(기본), ALAW, MULAW, OGG_OPUS. 샘플레이트 16k 지정 가능 여부는 문서에 없음 [미검증].
- 제한: 입력 8,192 토큰, text/prompt 각 4,000바이트(합 8,000), 출력 약 655초.
  출처: https://docs.cloud.google.com/text-to-speech/docs/gemini-tts

**TTFA(첫 오디오까지 시간) 측정치**
- 공식 수치 없음. 3.8 TTS에 대한 공개 TTFA 벤치마크도 찾지 못함 [미검증].
- 커뮤니티 보고(3.1 Flash TTS preview, Tier 2 유료 키): 자체 키로 **첫 청크 10~20초**, AI Studio 기본 키로 2초 미만, 일시 개선 후 재악화. "청크가 점진적으로 오지 않고 오래 침묵 후 한꺼번에 도착 → 서버에서 전체 생성 후 스트리밍하는 것 같다", "Interactions API가 generateContent stream보다 상당히 느렸다"는 관찰.
  출처: https://discuss.ai.google.dev/t/3-1-flash-tts-preview-streaming-latency/176050
- **시사점**: Gemini TTS는 LLM 기반이라 TTFA 변동이 클 수 있음. **1.5s 목표에 Gemini TTS를 실시간 경로로 쓰려면 반드시 p50/p95 실측**하고, 폴백 TTS를 준비해야 함.

### A-4. 가격과 Rate limit

**가격 (Gemini API, 1M 토큰당, Standard 유료) [확인]** 출처: https://aistudio.google.com/docs/pricing

| 모델 | 무료 티어 | 입력(텍스트) / 출력(오디오) | 비고 |
|---|---|---|---|
| `gemini-3.8-flash-tts` | 있음 | $0.50 / $9.00 → **2027-01-01부터 $1.00 / $18.00** | 10초 오디오 ≈ $0.00225 (2027: $0.0045). Priority $0.90/$16.20 |
| `gemini-3.8-flash-lite-tts` | 있음 | $0.50 / $6.00 → 2027: $1.00 / $12.00 | |
| `gemini-3.1-flash-tts-preview` | 있음 | $1.00 / $20.00 | |
| `gemini-2.5-flash-preview-tts` | 있음 | $0.50 / $10.00 | |
| `gemini-2.5-pro-preview-tts` | 없음 | $1.00 / $20.00 | |

- 오디오 토큰은 초당 25토큰 [미검증: digitalapplied.com 기사]. 10초 ≈ 250토큰 ≈ $0.00225로 위 공식 환산치와 일치.
- 비용 감각: 395케이스 × 케이스당 응답 오디오 60초 가정 → 약 23,700초 ≈ 59만 오디오 토큰 ≈ **$5.3** (Flash TTS 2026 가격). 비용은 문제 아님.

**Rate limit [확인/미검증 혼재]** 출처: https://aistudio.google.com/docs/rate-limits
- rate-limits 문서에 TTS 모델의 RPM/TPM/RPD 수치가 **더 이상 표로 공개되어 있지 않음** ("AI Studio의 rate limit 페이지에서 프로젝트별 확인"). 배치 큐 한도만 표기(2.5 Flash TTS Tier1 100k 토큰 등).
- 티어 조건: Free(활성 프로젝트), Tier 1(결제 계정 연결), Tier 2(누적 $100 + 첫 결제 후 3일), Tier 3(누적 $1,000 + 30일).
- 과거(2025) 2.5 Flash preview TTS 무료 티어는 RPM 3·RPD 15 수준으로 매우 낮았던 것으로 기억 [미검증, 현재 수치 아님]. → **무료 티어로는 395케이스 × 동시성 4 평가를 견디기 어렵다고 가정**하고 **결제 연결(Tier 1 이상) 필수**, AI Studio 콘솔에서 해당 프로젝트의 TTS RPM/RPD를 직접 확인할 것.
- 동시성 4 × 문장 분할(응답당 2~4요청) → 순간 최대 ~10~16 RPS가 아니라, 대략 **분당 수십~100 요청** 수준. Tier 1 한도로 충분한지 확인 필요.
- **캐시 문구 사전 합성**은 서버 기동 시 매번 하지 말고 **빌드 타임에 1회 합성 → 파일(16k PCM)로 저장**해 RPD 소모를 없앨 것.

### A-5. Python 예제 (google-genai SDK, Interactions API, 스트리밍 PCM)

> 공식 예제 구조를 바탕으로 재작성한 코드. Interactions API의 비동기 클라이언트(`client.aio.interactions`) 존재 여부와 정확한 필드명은 SDK 버전에서 확인 필요 [미검증]. 동기 SDK라면 `asyncio.to_thread` + 큐로 감쌀 것.

```python
import asyncio, base64, os, time
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                      http_options={"api_version": "v1beta"})

STYLE = "친절하고 차분한 한국어 고객센터 상담원 톤, 또렷하고 약간 빠르게"

def _tts_stream_sync(text: str, q: asyncio.Queue, loop: asyncio.AbstractEventLoop,
                     voice: str = "Kore"):
    stream = client.interactions.create(
        model="gemini-3.8-flash-tts",
        input=[{
            "type": "user_input",
            "content": [{
                "type": "text",
                "text": text,
                "annotations": [{"type": "speech_metadata", "style": STYLE}],
            }],
        }],
        response_format={"type": "audio", "mime_type": "audio/l16",
                         "sample_rate": 16000},   # 16k 직접 요청 (실측으로 확인)
        generation_config={"speech_config": [{"voice": voice}]},
        stream=True,
    )
    for ev in stream:
        if ev.event_type == "step.delta" and getattr(ev.delta, "type", None) == "audio":
            loop.call_soon_threadsafe(q.put_nowait, base64.b64decode(ev.delta.data))
    loop.call_soon_threadsafe(q.put_nowait, None)

async def tts_pcm16k(text: str):
    """16kHz s16le mono PCM 청크를 비동기로 yield"""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    t0 = time.perf_counter()
    fut = loop.run_in_executor(None, _tts_stream_sync, text, q, loop)
    first = True
    while (chunk := await q.get()) is not None:
        if first:
            print(f"TTFA {1000*(time.perf_counter()-t0):.0f} ms"); first = False
        yield chunk
    await fut
```

- 레거시 `generate_content` 방식(2.5 TTS용, 참고): `client.models.generate_content(model="gemini-2.5-flash-preview-tts", contents=text, config=types.GenerateContentConfig(response_modalities=["AUDIO"], speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")))))` → `resp.candidates[0].content.parts[0].inline_data.data`가 24kHz PCM. 현 문서에서 빠졌으므로 사용 비권장 [과거 문서 기준, 현재 미검증].

### A-6. 저지연 한국어 TTS 대안 비교

| 서비스 / 모델 | TTFB(첫 오디오) | 한국어 | 스트리밍 | 16k PCM 직접 출력 | 비고 |
|---|---|---|---|---|---|
| **Gemini 3.8 Flash / Flash-Lite TTS** | 공식 수치 없음, 변동 큼(커뮤니티: 2s~20s, 3.1 preview 기준) [미검증] | O (품질·표현력 우수 평가) | 출력 스트리밍 O, 입력 스트리밍 X | O (`sample_rate`) [문서 확인] | 스타일 프롬프트 강력. 2027-01 가격 2배 |
| **Google Cloud TTS Chirp 3: HD** | 공식 수치 없음 (일반적으로 수백 ms로 알려짐) [미검증] | O (`ko-KR`) | O (StreamingSynthesize, PCM/OGG_OPUS/MULAW/ALAW) | LINEAR16 기본, 스트리밍 시 PCM. sample_rate_hertz 지정 가능 여부 실측 필요 [미검증] | 비-LLM 계열이라 지연 안정적일 가능성. `asia-northeast1` 리전 GA. 출처: https://docs.cloud.google.com/text-to-speech/docs/chirp3-hd |
| **ElevenLabs Flash v2.5** (`eleven_flash_v2_5`) | 모델 ~75–100ms(네트워크 제외), 동북아 WebSocket 150–200ms, 독립 측정 197–255ms [공식+서드파티] | O (`ko`) | O (HTTP stream, WebSocket 입력 스트리밍) | O (`output_format=pcm_16000`) [문서 일부 확인] | 가장 검증된 저지연 옵션. `eleven_v4_turbo`(~100ms), `eleven_v3_conversational`(~165ms)도 한국어 지원. 출처: https://elevenlabs.io/docs/overview/models , https://elevenlabs.io/docs/eleven-api/guides/how-to/best-practices/latency-optimization |
| **OpenAI `gpt-4o-mini-tts` / `tts-1`** | 수치 없음 | O (Whisper 언어 기준; 보이스는 영어 최적화) | O (chunked) | X (PCM 24kHz 고정 → 리샘플 필요) | **2027-01-06 셧다운 예정**, 후속은 Realtime `gpt-realtime-2.1-mini`. 신규 채택 비권장. 출처: https://developers.openai.com/api/docs/guides/text-to-speech , https://developers.openai.com/api/docs/deprecations |
| **Typecast** (`ssfm-v31` 등) | 수치 없음 | O (`kor`) | O (`POST /v1/text-to-speech/stream`) | X (WAV 32kHz 또는 MP3 44.1kHz → 리샘플 필요) | 한국 업체, 한국어 감정 표현 강점. 출처: https://typecast.ai/docs/ko/api-reference/text-to-speech/streaming-text-to-speech.md |
| **Supertone** (Supertonic 3 / API) | 호스티드 API 수치 없음 (로컬 서버 first-byte 450–650ms 보고) [미검증] | O | 미확인 | 미확인 (로컬 44.1kHz) | 문서 부족. 출처: https://www.supertone.ai/ko/api |

---

## B. OpenAI LLM

### B-1. "GPT-5.4 mini" 존재 여부와 사양

**존재함 [확인]** 출처: https://developers.openai.com/api/docs/models/gpt-5.4-mini

| 항목 | 값 |
|---|---|
| 모델 ID | `gpt-5.4-mini` (스냅샷 `gpt-5.4-mini-2026-03-17`) |
| 컨텍스트 | 400K (입력 최대 272K, 출력 최대 128K), 지식 컷오프 2025-08-31 |
| 가격(1M 토큰) | 입력 $0.75 / 캐시 입력 $0.075 / 출력 $4.50 |
| Fast mode(구 Priority) | 2배: $1.50 / $0.15 / $9.00, p50 목표 >100 tok/s(일부 로케일 50) [https://openai.com/api-priority-processing/] |
| reasoning.effort | **`none`(기본값)**, `low`, `medium`, `high`, `xhigh` |
| 엔드포인트 | Chat Completions, Responses, Batch (Realtime 미지원) |
| 기능 | 스트리밍, Structured Outputs, function calling, 프롬프트 캐싱 |
| Rate limit | Build 5,000 RPM / 2M TPM, Launch 10,000 RPM, Grow 30,000 RPM |
| 수명 | 셧다운 예정 없음. 단 **EU 데이터 레지던시 + fast/priority 조합은 2027-01-11 종료** (우리는 해당 없을 가능성 높음) |
| TTFT | 공식 수치 없음 [미검증]. "GPT-5 mini 대비 2배 이상 빠름"(출시 보도) |

**더 새로운 후보 (2026-11 기준 모델 목록) [확인]** 출처: https://developers.openai.com/api/docs/models
- `gpt-6-luna`: "가장 효율적" 모델. 입력 $0.10 / 캐시 $0.01 / 출력 $0.50 (gpt-5.4-mini의 약 1/7~1/9). 컨텍스트 1.05M. reasoning `none|low|medium(기본)|high|xhigh|max`. **Chat Completions에서 function calling은 reasoning `none`일 때만, 그 외엔 Responses API 필요.** `gpt-5.4-nano`의 후속(2027-04-01 셧다운). 출처: https://developers.openai.com/api/docs/models/gpt-6-luna
- `gpt-6.1-terra`: "빠르고 효율적", $1 / $0.05 / $5, 기본 reasoning `medium`, **툴 호출은 Responses API 필수**. 출처: https://developers.openai.com/api/docs/models/gpt-6.1-terra
- `gpt-6.2-sol`, `gpt-6.2-astra`: 상위 모델(우리 용도엔 과함).
- 주의: GPT-6 계열은 **기본 reasoning이 `medium`** → 반드시 `none`(또는 `low`)으로 명시해야 TTFT가 짧아짐.

### B-2. Responses API vs Chat Completions — 스트리밍 + function calling

출처: https://developers.openai.com/api/docs/guides/function-calling

| 비교 | Responses API (`/v1/responses`) | Chat Completions |
|---|---|---|
| 권장 여부 | OpenAI 신규 권장, GPT-6 계열 툴 호출 필수 | 계속 지원. gpt-5.4-mini는 지원 |
| 툴 정의 | `{"type":"function","name","description","parameters","strict"}` (평평한 구조) | `{"type":"function","function":{...}}` |
| 툴 결과 반환 | `{"type":"function_call_output","call_id","output"}` | `role:"tool", tool_call_id` |
| 상태 | `previous_response_id`로 체이닝 가능 (또는 `store=False`로 전체 입력 재전송) | 매번 전체 메시지 재전송 |
| reasoning 항목 | reasoning 모델이면 reasoning item을 툴 결과와 함께 되돌려야 함 (`previous_response_id` 쓰면 자동) | - |
| 스트리밍 이벤트 | `response.output_text.delta`, `response.output_item.added`(function_call 시작, name/call_id), `response.function_call_arguments.delta/done`, `response.output_item.done`, `response.completed` | `choices[0].delta.tool_calls[i].function.arguments` 누적 |
| WebSocket mode | 지원: 소켓 유지, 매 턴 새 입력 + `previous_response_id`만 전송, 툴 다회 루프에서 최대 ~40% 단축 [https://developers.openai.com/api/docs/guides/websocket-mode] | 없음 |

- **parallel_tool_calls**: 기본 허용. `parallel_tool_calls=False`면 턴당 0~1개. 우리 시나리오(주문조회 후 콜백 요청 등)는 대부분 순차 의존이지만, 독립 조회가 동시에 필요하면 병렬 허용 + `asyncio.gather`로 실행.
- **Strict 모드** 요건: 모든 object에 `additionalProperties: false`, `properties`의 모든 필드를 `required`에 나열, 선택 필드는 `"type": ["string","null"]`. 위반 시 요청 거부. Responses에서 `strict` 생략 시 자동 정규화 시도.

**최소 async 예제 (Responses API, 스트리밍 텍스트 + 툴 루프)**

```python
import asyncio, json, httpx
from openai import AsyncOpenAI

client = AsyncOpenAI()
http = httpx.AsyncClient(base_url="http://mock-api", timeout=3.0)

TOOLS = [{
    "type": "function",
    "name": "get_order_info",
    "description": "주문번호로 주문 상태/배송 정보를 조회한다.",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "주문번호"},
            "phone_last4": {"type": ["string", "null"]},
        },
        "required": ["order_id", "phone_last4"],
        "additionalProperties": False,
    },
}]

async def call_tool(name: str, args: dict) -> str:
    r = await http.post(f"/{name}", json=args)
    return r.text

async def run_turn(history: list, on_text):
    """history: Responses input items. on_text: 텍스트 델타 콜백(→ 문장 분할 → TTS)"""
    while True:
        calls = {}           # output_index -> {"name","call_id","args"}
        stream = await client.responses.create(
            model="gpt-5.4-mini",
            instructions=SYSTEM_PROMPT,           # 고정 프리픽스 → 프롬프트 캐시
            input=history,
            tools=TOOLS,
            reasoning={"effort": "none"},
            text={"verbosity": "low"},
            max_output_tokens=300,
            store=False,
            stream=True,
        )
        async for ev in stream:
            t = ev.type
            if t == "response.output_text.delta":
                await on_text(ev.delta)
            elif t == "response.output_item.added" and ev.item.type == "function_call":
                calls[ev.output_index] = {"name": ev.item.name,
                                          "call_id": ev.item.call_id, "args": ""}
            elif t == "response.function_call_arguments.delta":
                calls[ev.output_index]["args"] += ev.delta
            elif t == "response.completed":
                history.extend(o.model_dump(exclude_none=True)
                               for o in ev.response.output)   # assistant/function_call/reasoning 항목 보존
        if not calls:
            return
        results = await asyncio.gather(*(call_tool(c["name"], json.loads(c["args"]))
                                         for c in calls.values()))
        for c, out in zip(calls.values(), results):
            history.append({"type": "function_call_output",
                            "call_id": c["call_id"], "output": out})
        # 루프: 툴 결과를 넣고 다시 생성
```

- 툴 호출 전 "확인해 드릴게요, 잠시만요" 같은 **필러 음성을 캐시 오디오로 즉시 송출**하면 툴 왕복 시간이 첫 유성 오디오 지연에 들어가지 않음.

### B-3. TTFT 최소화 팁

출처: https://developers.openai.com/api/docs/guides/latency-optimization , https://openai.com/api-priority-processing/

1. **reasoning.effort = `none`** (gpt-5.4-mini 기본값이지만 명시). GPT-6 계열은 기본 `medium`이므로 반드시 명시. `reasoning_tokens`가 0인지 usage로 확인.
2. **프롬프트 캐싱**: 시스템 프롬프트 → 툴 정의 → (예시) 순서로 **고정 프리픽스**를 앞에 두고 대화만 뒤에 덧붙임. `cached_tokens`로 히트 확인. 캐시 입력은 90% 할인 + TTFT 단축.
3. **입력/출력 줄이기**: 툴 description 간결화(14개 툴 스키마가 매 요청 프리픽스), `text.verbosity="low"`, `max_output_tokens` 제한, "1~2문장으로 답하라" 지시.
4. **Fast mode** (`service_tier="fast"` 또는 `"priority"`, 2배 가격): 처리량 목표 >100 tok/s. 커뮤니티에서 mini/nano가 priority로 처리되지 않거나 default로 조용히 강등된 사례 보고 → 응답의 `service_tier` 필드 기록 [미검증].
5. **연결 재사용**: `AsyncOpenAI` 클라이언트 1개 재사용(HTTP keep-alive), 서버 기동 시 워밍업 요청. 툴 루프가 길면 Responses **WebSocket mode** 검토.
6. **LLM 생략**: 인사/정형 안내/확인 멘트는 규칙 기반 + 캐시 오디오로 처리.
7. 리전: 문서에 리전 선택 TTFT 가이드 없음. 데이터 레지던시 엔드포인트는 +10% 가격. 서버를 미국 리전(OpenAI·Google 엔드포인트 근접)에 두는 것이 유리할 수 있으나 평가 서버 위치와의 RTT도 고려 [미검증].

### B-4. OpenAI Realtime API (speech-to-speech) 대안

출처: https://developers.openai.com/api/docs/models , https://developers.openai.com/api/docs/models/gpt-realtime-2.1-mini , https://developers.openai.com/api/docs/guides/realtime-conversations

- 모델: `gpt-realtime-2.1`, **`gpt-realtime-2.1-mini`**(툴 사용 가능한 경량 reasoning 모델), `gpt-live-1`, `gpt-realtime-2`, `gpt-realtime-1.5`. `gpt-realtime`(2025-08)은 2027-01-20 셧다운.
- `gpt-realtime-2.1-mini` 가격(1M 토큰): 오디오 입력 $10 / 캐시 $0.30 / 출력 $20, 텍스트 $0.60 / $2.40. 128K 컨텍스트. Function calling, 프롬프트 캐싱 지원. 영숫자 인식 개선(주문번호·전화번호에 유리).
- 연결: WebSocket(서버), WebRTC, SIP.
- 오디오 포맷: `{"type":"audio/pcm","rate":24000}` (문서 예시는 24kHz mono). `audio/pcmu`(8k μ-law)도 있음. **16kHz PCM 직접 입출력은 문서상 불가** → 입력 16k→24k 업샘플, 출력 24k→16k 다운샘플 필요 [16k 미지원은 미검증이나 문서에 24k만 표기].
- 함수 호출 흐름: `session.tools` 등록 → 출력 item `function_call`(name, arguments, call_id) → 서버에서 실행 → `conversation.item.create`(`function_call_output`) → `response.create`.
- 한국어: 문서에 언어별 안내 없음 [미검증]. 실사용상 한국어 대화 가능하나 품질·발음은 직접 평가 필요.
- 장점: STT→LLM→TTS 3단 파이프라인 지연 제거, 한 소켓에서 바지인/턴감지 처리. 단점: 우리 프로젝트는 **STT 결과/발화 종료 판정 기준이 평가기 쪽에 있을 수 있음**, 툴 결과 기반 정확한 멘트 제어·로깅이 어려움, 오디오 토큰 비용이 텍스트 파이프라인보다 큼(395케이스 × 수 분이면 수십 달러 수준으로 감당 가능), 보이스 선택(marin/cedar 등)은 영어 최적화.
- 결론: **1차 아키텍처는 파이프라인(STT → gpt-5.4-mini → TTS)**, Realtime은 TTS 지연이 목표를 못 맞출 때의 플랜 B로 PoC만.

---

## 설계에 주는 시사점 (권고)

1. **TTS 모델 교체**: 초안의 `gemini-2.5-flash-preview-tts` 대신 **`gemini-3.8-flash-tts`**(품질) 또는 **`gemini-3.8-flash-lite-tts`**(지연/비용)로 설계. 둘 다 한국어 지원, 30 보이스, 스타일 프롬프트. 첫 주에 두 모델 × 보이스 3~4개(Kore/Aoede/Leda/Achernar 등)로 한국어 청취 비교 + TTFA 측정.
2. **16k 직접 요청**: `response_format={"type":"audio","mime_type":"audio/l16","sample_rate":16000}`로 리샘플링 제거 시도. 안 되면 `soxr.ResampleStream`(스트리밍), 캐시 문구는 `resample_poly(2,3)` 일괄 변환. 청크 경계 홀수 바이트 이월 처리.
3. **Gemini TTS TTFA 리스크를 전제로 설계**: 공식 수치가 없고 3.1 preview에서 10~20초 사례가 있었음. 따라서
   - (a) 응답 첫 문장은 **짧게**(10~20자) 만들어 첫 TTS 요청을 작게,
   - (b) LLM 스트림을 **문장/구 단위로 분할**해 TTS 요청을 파이프라인으로 연속 발행(Interactions API는 입력 텍스트 스트리밍 불가),
   - (c) **캐시 필러**("네, 확인해 드리겠습니다." 등)를 발화 종료 즉시 송출해 1.5s 점수 확보,
   - (d) TTFA p95가 1.2s를 넘으면 **ElevenLabs Flash v2.5(`pcm_16000`, ~200ms)** 또는 **Chirp 3 HD** 로 실시간 경로 폴백. 폴백 인터페이스(`async def synth(text) -> AsyncIterator[bytes]`)를 처음부터 추상화.
4. **캐시 문구는 빌드 타임 사전 합성 → 16k PCM 파일 저장**. 서버 기동/평가 중 TTS 호출을 실시간 응답에만 쓰도록 해 rate limit 소모 최소화. 프로젝트 결제 연결(Tier 1+) 필수, AI Studio에서 TTS RPM/RPD 실제 값 확인. 429 대비 지수 백오프 + 폴백 TTS.
5. **LLM**: `gpt-5.4-mini` + **Responses API** + `reasoning.effort="none"` + `stream=True` + `strict` 툴 스키마. 14개 툴 정의와 시스템 프롬프트를 고정 프리픽스로 둬서 프롬프트 캐싱. `gpt-6-luna`(reasoning `none` 명시)를 같은 평가셋으로 A/B — 툴 선택 정확도가 비슷하면 비용 1/7.
6. **TTFT 목표 배분(예시)**: 발화종료 판정 ~300ms + LLM TTFT ~300–500ms + 첫 문장 완성 ~150ms + TTS TTFA ~300–600ms + 전송 ~50ms ≈ 1.1–1.6s. 빠듯하므로 (3c) 캐시 필러가 사실상 필수. 툴 호출 턴은 필러로 가리고 툴 결과 후 본 응답.
7. **Fast mode**는 측정 후 선택. TTFT 개선보다 처리량 개선 성격이며, mini에서 강등 사례가 보고됨 → `service_tier` 응답값 로깅.
8. **Realtime API(`gpt-realtime-2.1-mini`)**는 플랜 B. 24kHz 고정이라 16k 양방향 리샘플 필요, 한국어 품질·툴 정확도 직접 검증 후 판단.
9. **가격 일정 주의**: Gemini 3.8 TTS/Flash는 **2027-01-01부터 2배**, OpenAI `gpt-4o-mini-tts`는 **2027-01-06 셧다운**(폴백 후보에서 제외).

## 미검증/실측 필요 목록

- Gemini 3.8 TTS `sample_rate: 16000` 실제 동작, TTFA p50/p95 (서울/미국 리전별), Interactions vs generateContent 스트림 TTFA 차이
- google-genai SDK의 Interactions API 비동기 지원 여부, 정확한 이벤트 필드명
- Gemini TTS 모델별 RPM/RPD (Tier 1 기준) — AI Studio 콘솔에서 확인
- gpt-5.4-mini / gpt-6-luna 실제 TTFT (reasoning none, 툴 14개 프리픽스, 캐시 히트 시)
- Chirp 3 HD 스트리밍 16k 출력 가능 여부와 TTFB, ElevenLabs 한국어 상담 톤 품질
- Realtime API 한국어 품질 및 16k 지원 여부

## 출처 목록

- Gemini speech generation: https://aistudio.google.com/docs/speech-generation
- Gemini pricing: https://aistudio.google.com/docs/pricing
- Gemini rate limits: https://aistudio.google.com/docs/rate-limits
- Cloud TTS Gemini-TTS: https://docs.cloud.google.com/text-to-speech/docs/gemini-tts
- Cloud TTS Chirp 3 HD: https://docs.cloud.google.com/text-to-speech/docs/chirp3-hd
- 3.1 Flash TTS 지연 포럼: https://discuss.ai.google.dev/t/3-1-flash-tts-preview-streaming-latency/176050
- Gemini 3.8 Flash TTS 가격 기사(서드파티): https://www.digitalapplied.com/blog/gemini-3-8-flash-tts-voice-cloning-price-doubles-january
- OpenAI gpt-5.4-mini: https://developers.openai.com/api/docs/models/gpt-5.4-mini
- OpenAI 모델 목록: https://developers.openai.com/api/docs/models
- OpenAI gpt-6-luna: https://developers.openai.com/api/docs/models/gpt-6-luna
- OpenAI gpt-6.1-terra: https://developers.openai.com/api/docs/models/gpt-6.1-terra
- OpenAI deprecations: https://developers.openai.com/api/docs/deprecations
- OpenAI function calling: https://developers.openai.com/api/docs/guides/function-calling
- OpenAI latency optimization: https://developers.openai.com/api/docs/guides/latency-optimization
- OpenAI WebSocket mode: https://developers.openai.com/api/docs/guides/websocket-mode
- OpenAI Fast mode(Priority): https://openai.com/api-priority-processing/
- OpenAI priority 지연 커뮤니티: https://community.openai.com/t/gpt-5-4-nano-priority-service-tier-inconsistent-latency/1377270
- OpenAI TTS: https://developers.openai.com/api/docs/guides/text-to-speech
- OpenAI Realtime: https://developers.openai.com/api/docs/guides/realtime , https://developers.openai.com/api/docs/guides/realtime-conversations , https://developers.openai.com/api/docs/models/gpt-realtime-2.1-mini
- ElevenLabs models: https://elevenlabs.io/docs/overview/models
- ElevenLabs latency: https://elevenlabs.io/docs/eleven-api/guides/how-to/best-practices/latency-optimization
- Typecast streaming: https://typecast.ai/docs/ko/api-reference/text-to-speech/streaming-text-to-speech.md
- Supertone API: https://www.supertone.ai/ko/api
