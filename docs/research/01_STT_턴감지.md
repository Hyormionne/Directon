# 01. STT(Soniox) 연동과 턴 종료 감지 리서치

- 작성일: 2026-11-19
- 범위: Soniox 실시간 STT, 로컬 VAD(Silero/webrtcvad), Pipecat Smart Turn, LiveKit 턴 감지기, 한국어 발화 종료의 함정
- 전제: 평가기가 16kHz/16-bit/mono PCM을 20ms 프레임(640바이트)으로 WebSocket에 보내고, 고객 발화가 끝난 뒤에도 무음을 계속 보낸다. 속도 점수는 "실제 발화 종료 → 우리 첫 유성 음성"까지의 시간이다. 1.5초 이내면 만점이고, 발화 종료 전에 말하면 0점이다.
- 표기: **[확인]**은 공식 문서에서 직접 확인한 내용, **[미확인]**은 2차 출처이거나 추정이라 직접 검증해야 하는 내용이다.

---

## 1. Soniox 실시간 WebSocket API

### 1.1 엔드포인트와 인증 [확인]
- URL: `wss://stt-rt.soniox.com/transcribe-websocket`
- 인증: WebSocket 업그레이드 요청의 `Authorization: Bearer <SONIOX_API_KEY>` 헤더나 WebSocket subprotocols 목록 중 **한 곳에만** 키를 넣는다. 두 곳에 다 넣거나 형식이 잘못되면 400 에러 프레임이 오고 연결이 끊긴다.
  - Python `websockets`에서는 `websockets.connect(url, additional_headers={"Authorization": f"Bearer {key}"})`를 쓴다.
- 시작 메시지의 `api_key` 필드는 **deprecated**다. **2027-01-15부터** 키를 이 필드로만 보내는 연결은 거부된다. 처음부터 헤더 방식으로 구현해야 한다.
- 브라우저용 임시 키(`snx_temp_` 접두사)도 있지만, 우리는 서버 측에서 연결하므로 필요 없다.
- 출처: https://docs.soniox.com/stt/api-reference/websocket-api , https://soniox.com/docs/guides/websocket-authentication , https://soniox.com/docs/guides/migrate-websocket-authentication

### 1.2 모델 ID [확인]
| 모델 | 상태 |
|---|---|
| `stt-rt-v6` | **현재 최신 실시간 모델** (Pipecat Soniox 서비스의 기본값도 v6) |
| `stt-rt-v5` | Deprecated. **2026-12-15에 제거**되며 이후 요청은 v6로 라우팅된다 |
| `stt-rt-v4` | v5를 가리키는 alias |

- 지원 언어는 60개 이상이며, 한국어(`ko`)는 실시간 API에서 지원된다.
- 모델 페이지에는 v6의 한국어 전용 수치(WER, 지연)가 없다. **[미확인]**
- 출처: https://docs.soniox.com/stt/models , https://docs.soniox.com/stt/concepts/supported-languages

### 1.3 설정(시작) 메시지 필드 [확인]
연결 직후 JSON 텍스트 프레임으로 한 번 보낸다.

| 필드 | 필수 | 비고 |
|---|---|---|
| `model` | O | `"stt-rt-v6"` |
| `audio_format` | O | 원시 PCM이면 `"pcm_s16le"` (`"auto"`는 컨테이너 포맷을 자동 감지) |
| `sample_rate` | 원시 PCM이면 O | `16000` |
| `num_channels` | 원시 PCM이면 O | `1` |
| `language_hints` | X | `["ko"]` (중복 불가) |
| `language_hints_strict` | X | 힌트 언어로만 인식을 제한한다. 일부 모델만 지원 |
| `context` | X | `general`(key/value 목록), `text`(배경 설명), `terms`(용어 목록), `translation_terms`. 최대 8000 토큰 |
| `enable_endpoint_detection` | X | 의미 기반 엔드포인트 감지. 켜면 `<end>` 토큰이 온다 |
| `max_endpoint_delay_ms` | X | 500~3000, 기본 2000. "모든 모델이 지원하지는 않음"이라고 되어 있어 **v6 지원 여부는 실측 필요 [미확인]** |
| `endpoint_sensitivity` | X | -1.0~1.0, 기본 0.0 (v5, v6 지원) |
| `endpoint_latency_adjustment_level` | X | 0~3, 기본 0 (v5, v6 지원) |
| `enable_speaker_diarization` | X | 엔드포인트 감지와 같이 쓰면 화자분리 정확도가 떨어진다. 끄는 것을 권장 |
| `enable_language_identification` | X | 필요 없음 |
| `enable_agent_text_input` | X | v6 전용. 에이전트가 한 말을 `{"type":"agent_text",...}`로 알려줄 수 있다. 문맥 개선용 |
| `client_reference_id` | X | 최대 256자. 사용량 로그에 기록된다 |

상담 도메인 용어(상품명, 요금제명 등)는 `context.terms`에 넣고 `context.general`에 `{"key":"domain","value":"고객센터 상담"}`을 넣는 식으로 인식률을 보정할 수 있다.

### 1.4 응답 토큰 형식 [확인]
```json
{
  "tokens": [
    {"text": "환불", "start_ms": 1200, "end_ms": 1480, "confidence": 0.97, "is_final": true},
    {"text": "하고", "start_ms": 1480, "end_ms": 1700, "confidence": 0.88, "is_final": false}
  ],
  "final_audio_proc_ms": 1480,
  "total_audio_proc_ms": 1760
}
```
- **non-final 토큰**은 매 응답마다 새로 교체되는 가설이다. 바뀌거나 사라질 수 있다.
- **final 토큰**은 한 번만 오고 다시 바뀌지 않는다. 누적해서 쓴다.
- `final_audio_proc_ms`는 final 토큰으로 확정된 오디오 길이, `total_audio_proc_ms`는 non-final까지 포함해 처리된 오디오 길이다. 둘의 차이가 "아직 확정되지 않은 꼬리"다.
- 특수 토큰:
  - `<end>`: 엔드포인트 감지 시 세그먼트 끝에 한 번 온다. 항상 `is_final: true`다.
  - `<fin>`: 수동 finalize가 끝났음을 알린다. 항상 final이다.
  - 둘 다 텍스트에 포함하지 말고 걸러내야 한다.
- 스트림 종료 응답: `{"tokens": [], "final_audio_proc_ms":..., "total_audio_proc_ms":..., "finished": true}`. 이후 서버가 연결을 닫는다.

### 1.5 제어 메시지 [확인]
| 목적 | 메시지 | 비고 |
|---|---|---|
| 수동 확정 | `{"type": "finalize"}` | 그 시점까지의 오디오를 모두 final로 만들고 `<fin>`을 보낸다. 세션 중 여러 번 호출할 수 있지만 너무 잦으면 연결이 끊길 수 있다("몇 초에 한 번" 정도는 괜찮다고 함). **발화 종료 후 약 200ms의 무음을 보낸 다음** 호출하라고 권장한다. 더 일찍 부르면 정확도가 떨어진다 |
| 연결 유지 | `{"type": "keepalive"}` | 오디오를 보내지 않는 동안 **20초 안에** 최소 한 번 보내야 한다(5~10초 간격 권장). 20초 동안 아무것도 없으면 408로 종료된다. 시작 메시지 전 대기 중이면 40초 한도이고, 어기면 1001로 닫힌다 |
| 정상 종료 | 빈 **텍스트** 프레임 `""` | 빈 **바이너리** 프레임은 빈 오디오로 취급되어 종료되지 않는다 |

- 과금은 처리된 오디오가 아니라 **스트림 전체 시간** 기준이다. 우리는 평가기가 무음을 계속 보내므로 어차피 계속 오디오를 전달하게 되고, keepalive는 보조 수단이다.
- 출처: https://docs.soniox.com/stt/rt/manual-finalization , https://docs.soniox.com/stt/rt/connection-keepalive

### 1.6 에러 코드 [확인]
에러 응답에는 `error_code`(HTTP 상태), `error_type`(분기용 고정 식별자), `error_message`, `request_id`가 담긴다. 서버는 에러를 보낸 직후 연결을 닫는다.

| 코드 | error_type | 대응 |
|---|---|---|
| 400 | `invalid_request`, `model_not_available` | 설정 수정. "No audio received" 같은 메시지 |
| 401 | `unauthenticated` | 키 확인 |
| 402 | `organization_balance_exhausted` 등 | 잔액, 예산 확인 |
| 403 | `permission_denied`, `temp_api_key_session_expired` | |
| 408 | `request_timeout` | 재시도. 첫 오디오 대기 타임아웃 포함 |
| 413 | `max_duration_reached` | 새 연결 생성 |
| 429 | `limit_exceeded`, `max_concurrent_connections_reached` | 동시성, RPM 초과 |
| 500 | `internal_error` | 재시도 가능 |
| 503 | `service_unavailable` | 백오프 후 재시도 |

### 1.7 세션 한도와 동시성 [확인]
- 스트림당 최대 오디오 300분(고정이며 늘릴 수 없음)
- 기본 **동시 스트림 10개**, **분당 요청 100개**. 콘솔에서 증설을 요청할 수 있다.
- 동시 연결 수(유휴 포함)는 동시 요청 한도의 배수다.
- 평가가 여러 세션을 병렬로 돌린다면 10개 한도에 걸릴 수 있으니 미리 확인해야 한다.
- 비공식 메모에 "1시간 후 재연결"이 언급되지만 공식 문서에서는 확인되지 않았다. **[미확인]**
- 출처: https://docs.soniox.com/stt/rt/limits-and-quotas , https://soniox.com/docs/guides/concurrency-limits

### 1.8 가격 [확인]
- 실시간: **$0.12/시간** (약 $0.002/분). 비동기는 $0.10/시간.
- 토큰 단가(실시간): 입력 오디오 $2.00/1M, 입력 텍스트(context) $4.00/1M, 출력 텍스트 $4.00/1M. 오디오 1시간은 약 30,000 토큰이다.
- 무료 크레딧은 페이지에 명시가 없다. **[미확인]**
- 출처: https://soniox.com/pricing

### 1.9 Python SDK [확인]
- 공식 패키지는 `pip install soniox`다. 최신 2.11.0(2026-11-03 업로드)이며 Python 3.10 이상이 필요하다.
- 실시간 API: `client.realtime.stt.connect(config=...)`, `session.send_bytes()`, `session.receive_events()` 등. PyPI 설명에는 동기 예제만 있다. 비동기 클라이언트도 있다고 하지만 구체적인 시그니처는 **[미확인]**이다.
- Python 3.13.6에는 ssl 회귀가 있어 실시간 STT가 멈출 수 있으니 피하라는 경고가 있다.
- 우리 서버는 asyncio 기반이고 제어(finalize 타이밍)를 세밀하게 해야 하므로 **SDK 대신 `websockets`로 직접 구현하는 것을 권장**한다. 프로토콜이 단순하다.
- 서드파티 `soniox-pro-sdk`, `soniox-sdk`는 공식이 아니다.
- 출처: https://pypi.org/project/soniox , https://github.com/soniox/soniox-python

### 1.10 우리 서버에 맞춘 최소 예제 (asyncio + websockets)
평가기에서 받은 640바이트 PCM 프레임을 그대로 전달한다. 턴 감지기가 `finalize()`를 호출할 수 있게 해 둔다.

```python
import asyncio, json, os
import websockets

SONIOX_URL = "wss://stt-rt.soniox.com/transcribe-websocket"

class SonioxStream:
    def __init__(self, on_final_text, on_end, on_fin, terms=None):
        self.on_final_text = on_final_text   # (text, start_ms, end_ms) 콜백
        self.on_end = on_end                  # <end> 수신 콜백
        self.on_fin = on_fin                  # <fin> 수신 콜백
        self.terms = terms or []
        self.ws = None
        self.audio_q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=500)
        self.nonfinal_text = ""

    async def run(self):
        async with websockets.connect(
            SONIOX_URL,
            additional_headers={"Authorization": f"Bearer {os.environ['SONIOX_API_KEY']}"},
            max_size=2**22,
            ping_interval=20,
        ) as ws:
            self.ws = ws
            await ws.send(json.dumps({
                "model": "stt-rt-v6",
                "audio_format": "pcm_s16le",
                "sample_rate": 16000,
                "num_channels": 1,
                "language_hints": ["ko"],
                "enable_endpoint_detection": True,
                # 낮은 지연 시작점(공식 권장): level 2, sensitivity 0.3, max 1500
                "endpoint_latency_adjustment_level": 2,
                "endpoint_sensitivity": 0.3,
                "max_endpoint_delay_ms": 1000,   # v6 지원 여부 실측 필요
                "context": {
                    "general": [{"key": "domain", "value": "고객센터 상담"}],
                    "terms": self.terms,
                },
            }))
            await asyncio.gather(self._sender(), self._receiver())

    async def push_audio(self, frame: bytes):
        """평가기 WS 핸들러에서 20ms 프레임마다 호출한다. 큐가 가득 차면 오래된 것을 버린다."""
        try:
            self.audio_q.put_nowait(frame)
        except asyncio.QueueFull:
            _ = self.audio_q.get_nowait(); self.audio_q.put_nowait(frame)

    async def finalize(self):
        if self.ws:
            await self.ws.send(json.dumps({"type": "finalize"}))

    async def close(self):
        await self.audio_q.put(None)

    async def _sender(self):
        while True:
            frame = await self.audio_q.get()
            if frame is None:
                await self.ws.send("")          # 빈 텍스트 프레임 = 정상 종료
                return
            await self.ws.send(frame)           # 바이너리 프레임으로 원시 PCM 전송

    async def _receiver(self):
        async for msg in self.ws:
            res = json.loads(msg)
            if res.get("error_code"):
                raise RuntimeError(f"Soniox {res['error_code']} {res.get('error_type')}: {res.get('error_message')}")
            nonfinal = []
            for t in res.get("tokens", []):
                txt = t["text"]
                if txt == "<end>":
                    await self.on_end(); continue
                if txt == "<fin>":
                    await self.on_fin(); continue
                if t.get("is_final"):
                    await self.on_final_text(txt, t.get("start_ms"), t.get("end_ms"))
                else:
                    nonfinal.append(txt)
            self.nonfinal_text = "".join(nonfinal)   # 미리 LLM에 넣어볼 때 사용
            if res.get("finished"):
                return
```
메모:
- 평가기가 무음을 계속 보내므로 오디오를 끊기지 않게 계속 전달하면 keepalive는 필요 없다. 혹시 전달을 멈추는 구간이 있다면 5~10초마다 `{"type":"keepalive"}`를 보낸다.
- Soniox 토큰은 서브워드 단위이고 공백까지 토큰 텍스트에 들어 있으므로, 이어붙일 때 별도 구분자 없이 `"".join`으로 합친다. 한국어 띄어쓰기 처리 방식은 실측으로 확인해야 한다. **[미확인]**
- Soniox 연결은 평가 세션이 시작될 때 미리 열어 둔다. 연결과 TLS 핸드셰이크 지연을 첫 턴에서 빼기 위해서다.

---

## 2. Soniox 지연 특성 (한국어)
- 공식 문서에는 **한국어 수치(TTFT, 확정 지연)가 없다.** 한국어 voice-agent 페이지도 "말하는 중에 텍스트를 반환", "깨지기 쉬운 무음 타이머 없이 end 이벤트 제공" 같은 마케팅 문구뿐이다. **[미확인]**
- 구조적으로 보면 non-final 토큰은 거의 즉시 오고, final 토큰은 모델이 문맥을 더 본 뒤 지연되어 온다. 이 지연은 `total_audio_proc_ms - final_audio_proc_ms`로 측정할 수 있다.
- **finalize의 왕복 지연(finalize 전송 → `<fin>` 수신)이 우리 예산에서 핵심 수치**인데 공개 수치가 없다. 개발 초기에 반드시 측정해야 한다(예상 수십~수백 ms). **[미확인]**
- 출처: https://soniox.com/speech-to-text/use-cases/voice-agents/korean

---

## 3. Soniox 내장 엔드포인트 감지

### 3.1 동작 [확인]
- 침묵만 보는 게 아니라 쉼, 억양, 발화 패턴, 대화 문맥을 보는 **의미 기반(semantic) 엔드포인팅**이다.
- 엔드포인트로 판단하면 세그먼트의 모든 토큰을 final로 확정하고 `<end>`를 한 번 붙여 보낸다.
- 권장 사용 패턴: non-final은 자막용으로만 쓰고, 행동(LLM 호출)은 `<end>` 이후 final 텍스트로 한다.

### 3.2 파라미터 [확인]
| 파라미터 | 범위 | 기본 | 효과 |
|---|---|---|---|
| `endpoint_latency_adjustment_level` | 0~3 | 0 | 높을수록 빠르고 잦은 엔드포인트. 긴 발화를 쪼갤 수 있고 정확도가 약간 떨어진다. 의미 기반이라 계속 말할 것 같으면 여전히 기다린다 |
| `endpoint_sensitivity` | -1~1 | 0 | 높을수록 엔드포인트가 잘 난다. 음수는 느린 화자용 |
| `max_endpoint_delay_ms` | 500~3000 | 2000 | 발화 종료 후 엔드포인트를 낼 때까지의 **하드 상한** |

- 공식 "저지연 시작점"은 level 2, sensitivity 0.3, max 1500ms다.
- 음수 sensitivity와 level 1 이상을 함께 쓰는 것은 비권장이다(서로 상충).
- 출처: https://docs.soniox.com/stt/rt/endpoint-detection

### 3.3 우리 턴 종료 신호로 쓸 수 있는가?
- **장점**: 한국어 문맥(어미)을 볼 수 있는 유일한 신호다. 별도 모델 없이 공짜로 쓸 수 있다.
- **단점**:
  1. 기본 상한 2000ms, 저지연 권장 1500ms 모두 **우리 1.5s 예산을 혼자 다 쓴다.** 최소값 500ms로 해도 그건 "상한"일 뿐이고 실제 분포는 모른다. 여기에 네트워크 왕복이 더해진다.
  2. 엔드포인트 지연 분포가 공개되지 않아 예측이 어렵다.
  3. 외부 서비스 장애나 지연 스파이크가 그대로 턴 종료 지연이 된다.
- **Pipecat의 선택**: 기본 모드(`vad_force_turn_endpoint=True`)에서는 **로컬 VAD/Smart Turn이 턴을 결정**한다. VAD가 정지를 감지하면 Soniox에 `finalize`를 보내 전사를 빨리 받고, Soniox `<end>`는 턴 결정에 쓰지 않는다. 이 모드에서는 엔드포인트 설정이 무시된다. 업계 참고 구현도 "턴 결정은 로컬, Soniox는 빠른 확정 전사"로 역할을 나눈다는 뜻이다.
  - 출처: https://reference-server.pipecat.ai/en/latest/_modules/pipecat/services/soniox/stt.html
- **결론**: `<end>`는 **보조 신호**(이른 확정, 교차 검증)로 쓰고, 1차 트리거는 로컬 VAD 무음 타이머와 Smart Turn으로 하는 것을 권장한다.

---

## 4. 로컬 VAD (Python, 16kHz)

### 4.1 Silero VAD [확인]
- PyPI `silero-vad` 최신은 **6.2.3**(2026-09-23)이고 Python 3.8 이상, MIT 라이선스다. 기본 의존성은 torch이고 `silero-vad[onnx-cpu]` extra로 onnxruntime을 쓴다.
- 입력: **16kHz에서는 정확히 512샘플(32ms) 청크만 허용**한다. 내부적으로 앞 청크의 64샘플 컨텍스트를 이어붙인다. 다른 길이는 에러다.
  - → 우리 20ms(320샘플) 프레임은 **링버퍼에 모아 512샘플 단위로 잘라** 넣어야 한다. 판단 해상도는 32ms다.
- 지연: 청크당 **1ms 미만**(CPU 단일 스레드). 충분히 가볍다.
- 6000개 이상 언어로 학습되어 언어 독립적이다. 한국어 전용 평가 자료는 없다.
- `VADIterator` 기본값: `threshold=0.5`, `neg_threshold=threshold-0.15`(=0.35), `min_silence_duration_ms=100`, `speech_pad_ms=30`.
- 출처: https://github.com/snakers4/silero-vad , https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/utils_vad.py , https://pypi.org/project/silero-vad/

권장 사용(우리 상황):
- `VADIterator`의 start/end 이벤트에 의존하기보다 **청크별 확률을 직접 받아 우리 상태기계를 돌리는 편이 낫다.**
  - 발화 시작: prob ≥ 0.5가 연속 2~3청크(64~96ms)
  - 무음: prob < 0.35인 청크가 이어진 시간을 누적한다. 마지막 유성 청크의 끝 시각을 "추정 발화 종료 시각"으로 기록한다.
- 평가기의 무음이 **디지털 제로**라면 에너지(RMS) 게이트만으로도 매우 정확하다. RMS < 임계값을 보조 판정으로 함께 두면 Silero 오탐을 막을 수 있다. 실제 무음이 제로인지 잡음인지는 평가기 샘플로 확인해야 한다. **[미확인]**

### 4.2 webrtcvad
- 8/16/32/48kHz, 16-bit mono, **10/20/30ms 프레임**을 지원하므로 우리 20ms 프레임을 그대로 넣을 수 있다. aggressiveness는 0~3이다.
- 원 저장소(wiseman/py-webrtcvad)는 마지막 버전이 2.0.10이고 활발히 관리되지 않는 것으로 보인다. 최신 파이썬용 휠은 포크인 `webrtcvad-wheels`를 흔히 쓴다. **[미확인: 포크 상태]**
- GMM 기반이라 Silero보다 잡음과 숨소리에 오탐이 많다. 보조 수단이나 폴백 정도로 적합하다.
- 출처: https://github.com/wiseman/py-webrtcvad

---

## 5. Pipecat Smart Turn

### 5.1 개요 [확인]
- 최신 **v3.2**. HF `pipecat-ai/smart-turn-v3` 저장소에 `smart-turn-v3.2-cpu.onnx`(int8, 약 8MB)와 `smart-turn-v3.2-gpu.onnx`(fp32, 약 32MB)가 있다. v3.0, v3.1 파일도 있고, 저장소 갱신일은 2026-01-07이다.
- **오디오 기반**(전사 불필요)이다. Whisper Tiny 인코더 + 선형 분류기 구조로 약 8M 파라미터다.
- **한국어 지원**: 23개 언어에 포함된다.
  - v3.2-cpu 벤치마크에서 한국어 **정확도 96.96%, FPR 2.25%, FNR 0.79%**(889샘플). 전체 평균 정확도는 92.63%다.
  - v3 시점 한국어는 96.85%, FP 1.12%, FN 2.02%였다.
  - 여기서 FPR(미완료인데 완료로 판정)이 **우리에게 0점 위험**이다.
- 라이선스: **BSD-2-Clause** (상용 가능)
- 출처: https://github.com/pipecat-ai/smart-turn , https://huggingface.co/pipecat-ai/smart-turn-v3 , https://huggingface.co/pipecat-ai/smart-turn-v3/raw/main/benchmarks/smart-turn-v3.2-cpu.md

### 5.2 입력 형식 [확인]
- 16kHz mono float32 파형, **최대 8초**. 8초보다 길면 앞부분을 잘라 끝 8초를 남기고, 짧으면 **앞쪽을 0으로 패딩**한다.
- 입력은 "현재 턴의 전체 오디오"이고, 이전 턴은 필요 없다.
- 출력은 sigmoid 확률이며, 0.5 초과면 "완료"다.

### 5.3 CPU 지연 [확인/일부 미확인]
- Daily 블로그(v3) 측정, 전처리 + 추론 기준: AWS c7a.2xlarge 12.6ms, c8g.2xlarge 15.2ms, t3.2xlarge 33.8ms, c8g.medium 59.8ms, t3.medium 94.8ms. Pipecat Cloud 1x 인스턴스에서는 약 65ms다.
- v3.2 공식 지연 수치는 없다. 서드파티 ONNX 포트에서는 Apple M5 Pro 2스레드 기준 약 36ms였다. **[미확인]**
- 주의: `WhisperFeatureExtractor`(transformers)의 멜 변환 시간이 전처리에 포함된다. transformers 의존성이 무겁다면 멜 계산을 numpy로 대체할 수 있다(검증 필요).
- 출처: https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/

### 5.4 Pipecat 없이 단독 실행 [확인]
저장소 `inference.py`가 이미 독립 실행용이다. 의존성은 `onnxruntime`, `transformers`(WhisperFeatureExtractor), `numpy`다.
```python
import numpy as np, onnxruntime as ort
from transformers import WhisperFeatureExtractor

so = ort.SessionOptions()
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
so.inter_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession("smart-turn-v3.2-cpu.onnx", sess_options=so)
fe = WhisperFeatureExtractor(chunk_length=8)

def predict_endpoint(pcm16: bytes) -> float:
    audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    audio = audio[-8 * 16000:]                                   # 끝 8초 유지
    if len(audio) < 8 * 16000:                                   # 앞쪽 0 패딩
        audio = np.pad(audio, (8 * 16000 - len(audio), 0))
    feats = fe(audio, sampling_rate=16000, return_tensors="np", padding="max_length",
               max_length=8 * 16000, truncation=True, do_normalize=True).input_features
    feats = feats.astype(np.float32)                             # shape (1, 80, 800)
    return float(sess.run(None, {"input_features": feats})[0][0].item())  # 완료 확률
```
- asyncio 이벤트 루프를 막지 않도록 `loop.run_in_executor`(스레드 1개 전용 풀)로 실행한다.
- 서버 시작 시 더미 입력으로 한 번 워밍업해서 첫 호출 지연을 없앤다.

### 5.5 Pipecat의 권장 결합 방식 [확인]
- Silero VAD `stop_secs=0.2`: 200ms 무음이면 "정지"로 보고 그 시점에 Smart Turn을 실행한다.
- Smart Turn이 "완료"면 즉시 턴을 종료한다. "미완료"면 `SmartTurnParams.stop_secs`(기본 **3.0초**) 무음까지 기다렸다가 강제 종료한다.
- 사용자가 다시 말하기 시작하면 이후 다시 정지할 때 턴 전체 오디오로 재추론한다.
- `pre_speech_ms`: 문서에는 0, 소스에는 500으로 되어 있어 불일치한다.
- **우리에게 3.0초 폴백은 너무 길다.** 1.5s 예산 안에서는 폴백 상한을 1.0초 안팎으로 줄여야 한다(아래 시사점 참고).
- 출처: https://docs.pipecat.ai/server/utilities/smart-turn , https://docs.pipecat.ai/server/utilities/turn-management/user-turn-strategies

---

## 6. LiveKit 턴 감지기 (설계 참고)
- **오디오 턴 감지기**(권장, 신규): `TurnDetector` `v1`(LiveKit Inference 클라우드) / `v1-mini`(로컬 CPU, 인스턴스당 140~250MB). 14개 언어에 **한국어가 포함**된다. 라이선스는 LiveKit Model License(오픈소스가 아님)이고 LiveKit Agents 밖에서 쓰기 어렵다.
- **텍스트 턴 감지기**(`MultilingualModel`): Qwen2.5-0.5B 기반, 396MB, 턴당 50~160ms. **한국어를 지원**하고 STT 전사를 입력으로 받는다. 단 **deprecated**(Agents 2.0에서 제거 예정)다.
- **확률 → 지연 매핑 방식** (우리 설계에 차용할 만함):
  - 모델이 "턴 완료" 확률을 낸다. 확률이 `unlikely_threshold`(언어별 보정값) 이상이면 `min_delay`, 미만이면 `max_delay`만큼 무음을 기다린 뒤 턴을 종료한다.
  - 오디오 감지기를 쓸 때 기본값은 `min_delay=0.3s`, `max_delay=2.5s`다(기존 텍스트 기반은 0.5 / 3.0초, 구버전은 0.5 / 6.0초).
  - `dynamic` 모드(Python)는 세션의 쉼 통계(EMA α=0.9)로 범위 안에서 지연을 조정한다.
  - 추론이 약 1초를 넘으면 그냥 턴을 확정한다(타임아웃 시 확률 1.0으로 간주).
- 알려진 함정: "네" 같은 **짧은 단답**을 미완료로 판정해 `max_delay`까지 기다리는 문제가 커뮤니티와 공식 블로그에 보고되어 있다. 상담에서는 "네", "아니요" 단답이 매우 흔하므로 주의해야 한다.
- 출처: https://docs.livekit.io/agents/build/turns/turn-detector/ , https://docs.livekit.io/reference/agents/turn-handling-options/ , https://livekit.com/blog/short-utterances , https://livekit.com/blog/using-a-transformer-to-improve-end-of-turn-detection

---

## 7. 한국어 발화 종료의 함정
한국어 턴 종료 감지만 다룬 공개 벤치마크나 문헌은 거의 찾지 못했다. 아래는 언어학 자료와 일반 원리를 바탕으로 정리한 것이고, 일부는 **추론**이다.

1. **연결어미로 끝나는 쉼**: "~했는데요", "~거든요", "~해서요", "~하고요"는 문장이 끝난 것처럼 들리지만, 상담 맥락에서는 뒤에 본론("…환불 가능한가요?")이 이어지는 경우가 많다. 특히 "~는데(요)"는 배경을 제시하는 어미라 **미완료 신호일 가능성이 높다.** 반대로 "~거든요"는 이유를 제시하는 것으로 턴이 끝나는 경우도 흔해 애매하다. (추론)
2. **종결어미와 억양의 불일치**: 한국어는 같은 어미(-어/아, -요, -지)가 억양에 따라 평서, 의문, 확인 등으로 달라진다. Yun(2017)에 따르면 -어/아는 문장 유형과 억양이 비교적 잘 대응하지만 -지는 그렇지 않다. 텍스트만 보는 판정기는 "-지요"류를 오판할 수 있다.
   - 출처: https://linguistics.stonybrook.edu/jiwonyun/presentation/2017_ICKL_handout_Yun.pdf , 오승신(2006, 서울대) 대화체 문말 경계억양 분류 https://s-space.snu.ac.kr/handle/10371/29665
3. **간투사와 머뭇거림**: "어…", "그…", "음…", "저기요…" 뒤의 쉼은 길어도(500ms 이상) 미완료다. 텍스트(STT)에 간투사만 있으면 턴 종료를 보류하는 규칙이 유효하다.
4. **짧은 단답**: "네", "아니요", "맞아요", "괜찮아요"는 완료다. 단어가 짧아서 오디오 모델이 망설일 수 있으니 **텍스트 규칙으로 즉시 완료 처리**하는 것이 좋다.
5. **숫자와 정보 나열**: 전화번호, 주문번호, 생년월일을 부를 때 "010에… 1234에…"처럼 끊어 말한다. 숫자 나열 중 쉼은 미완료일 가능성이 높다. 자릿수(예: 전화번호 11자리)가 다 채워졌는지 보는 도메인 규칙을 둘 수 있다.
6. **TTS로 생성된 평가 음성일 가능성**: 평가기 고객 음성이 TTS라면 문장 내부 쉼은 보통 짧고(200~400ms), 문장 사이 쉼은 조금 길다(300~700ms). **여러 문장으로 된 발화에서 문장 사이 쉼을 턴 종료로 오인하는 것이 최대 위험**이다(0점). 평가 샘플로 문장 간 쉼의 분포를 반드시 측정해야 한다. **[미확인: 평가기 음성의 특성]**

---

## 설계에 주는 시사점

### A. 예산 분해 (목표: 실제 발화 종료부터 첫 유성 음성까지 1.5s, 실무 목표 ≤ 1.2s)
| 구간 | 목표 |
|---|---|
| 턴 종료 판정(무음 대기) | 400~600ms (최악 900ms) |
| 전사 확정(`finalize` → `<fin>`) | 판정과 **병렬**. 무음 200ms 시점에 이미 보냄 |
| LLM 첫 토큰 | 무음 200~300ms 시점에 **추측 실행(speculative)**으로 시작해 숨긴다 |
| TTS 첫 오디오 | 200~400ms (스트리밍 TTS, 첫 구절을 짧게) |
| 네트워크/버퍼 | 50~100ms |

→ 판정에 쓸 수 있는 시간은 사실상 **최대 ~700ms**다. Soniox `<end>` 단독(상한 500~2000ms, 분포 미공개)에 맡기기에는 위험하다.

### B. 신뢰할 신호의 우선순위 (권장 하이브리드)
1. **1차: 로컬 VAD 무음 타이머** (Silero 512샘플 청크 + RMS 보조). 결정적이고 지연 예측이 가능하며 네트워크와 무관하다.
2. **2차: Smart Turn v3.2 (CPU ONNX)**. 무음이 시작된 뒤 판정을 앞당기거나(완료 시) 늦추는(미완료 시) 용도다. 한국어 정확도 약 97%다.
3. **3차: Soniox 텍스트 신호** (`<end>`, `<fin>` 이후의 final 텍스트). 간투사나 연결어미("~는데", "그래서", "어…")로 끝나면 보류하고, 단답("네", "아니요")이면 즉시 완료하는 **텍스트 규칙**과 교차 검증한다. `<end>`가 VAD 타이머보다 먼저 오면 확정을 앞당기는 가속 신호로만 쓴다.

### C. 구체적인 상태기계와 기본 임계값 (초기값, 실측으로 튜닝)
- 발화 시작: Silero prob ≥ 0.5가 2청크(64ms) 연속. 시작 전 300ms 오디오를 프리롤로 보관한다.
- **무음 t=200ms**(prob < 0.35 연속, 마지막 유성 청크 기준):
  - Soniox에 `{"type":"finalize"}`를 보낸다(턴당 1회. 재개되면 다음 정지 때 다시 보냄).
  - 턴 전체 오디오로 Smart Turn을 추론한다(약 15~60ms).
  - 현재 텍스트(final + non-final)로 **LLM 추측 생성을 시작**한다. 출력은 버퍼에만 쌓고 재생하지 않는다.
- **확정 규칙**:
  - Smart Turn p ≥ 0.7 **그리고** 무음 ≥ **400ms** → 턴 종료하고 재생을 시작한다.
  - 0.5 ≤ p < 0.7 → 무음 ≥ **600ms**에서 종료한다.
  - p < 0.5(미완료) 또는 텍스트가 간투사/연결어미로 끝남 → 무음 ≥ **900ms**에서 강제 종료(하드 상한)한다. 3.0초 폴백(Pipecat 기본)은 예산을 초과하므로 쓰지 않는다.
  - 텍스트 단답 규칙(네/아니요 등) 일치 → 무음 ≥ 300~400ms에서 종료한다.
- **재개 처리**: 확정 전에 prob ≥ 0.5로 재개되면 추측 LLM을 취소하고 턴을 이어간다. **확정 후 재생 직전**에 재개가 감지되면 재생을 보류한다(0점보다는 지연이 낫다).
- Soniox 설정: `enable_endpoint_detection: true`, `endpoint_latency_adjustment_level: 2`, `endpoint_sensitivity: 0.3`, `max_endpoint_delay_ms: 1000`(v6 지원 시). `<end>`는 보조 신호로 쓰고, 턴 결정의 주체는 로컬이다.

### D. "발화 종료 전에 말하면 0점"에 대한 비대칭 위험
- 너무 이르면 0점이고, 늦어도 1.5s까지는 만점이다. 따라서 임계값은 **보수적인 쪽(약간 길게)**으로 시작하는 게 맞다. 무음 400ms 미만에서 확정하는 규칙은 두지 않는다.
- 단 1.5s 이후 점수가 어떻게 줄어드는지(선형 감점인지 0점인지)에 따라 하드 상한 900ms를 조정한다. 평가 규칙을 확인해야 한다. **[미확인]**
- "첫 유성 음성" 기준이므로 TTS 출력 앞부분의 무음 패딩은 지연으로 계산될 수 있다. TTS 첫 청크의 선행 무음을 잘라내는 것을 고려한다.
- 평가기의 "실제 발화 종료"는 아마 원본 음성 파일의 끝(또는 마지막 유성 샘플)일 것이다. 우리 VAD의 마지막 유성 청크 시각과 실제 종료는 최대 32ms(+ speech_pad) 정도 어긋날 수 있다. 측정 로그에 `last_voiced_ts`, `turn_commit_ts`, `first_audio_out_ts`를 모두 남겨 구간별 지연을 계측할 수 있게 한다.

### E. 구현 체크리스트
1. 평가기 세션 시작 시 Soniox WS를 미리 연결하고 Smart Turn과 Silero를 워밍업한다.
2. 20ms 프레임 → Soniox로 그대로 전달 + 512샘플 링버퍼 → Silero.
3. Smart Turn은 전용 단일 스레드 executor에서 돌려 이벤트 루프를 막지 않게 한다.
4. 초기에 반드시 실측할 항목: (a) `finalize` → `<fin>` 왕복 지연, (b) Soniox `<end>`의 실제 지연 분포(v6, `max_endpoint_delay_ms` 지원 여부), (c) 평가기 무음이 디지털 제로인지 여부, (d) 평가 음성의 문장 간 쉼 분포, (e) 동시 세션 수 대비 Soniox 동시성 한도(기본 10).
5. `api_key`는 Authorization 헤더로 보낸다(2027-01-15부터 필드 방식은 거부). 모델은 `stt-rt-v6`로 고정한다(v5는 2026-12-15 제거).

---

## 출처 목록
- Soniox WebSocket API: https://docs.soniox.com/stt/api-reference/websocket-api
- Soniox 엔드포인트 감지: https://docs.soniox.com/stt/rt/endpoint-detection
- Soniox 수동 finalize: https://docs.soniox.com/stt/rt/manual-finalization
- Soniox keepalive: https://docs.soniox.com/stt/rt/connection-keepalive
- Soniox 모델: https://docs.soniox.com/stt/models
- Soniox 지원 언어: https://docs.soniox.com/stt/concepts/supported-languages
- Soniox 한도: https://docs.soniox.com/stt/rt/limits-and-quotas , https://soniox.com/docs/guides/concurrency-limits
- Soniox WebSocket 인증: https://soniox.com/docs/guides/websocket-authentication
- Soniox 가격: https://soniox.com/pricing
- Soniox 한국어 페이지: https://soniox.com/speech-to-text/use-cases/voice-agents/korean
- Soniox Python SDK: https://pypi.org/project/soniox , https://github.com/soniox/soniox-python
- Pipecat Soniox 서비스 소스: https://reference-server.pipecat.ai/en/latest/_modules/pipecat/services/soniox/stt.html
- Silero VAD: https://github.com/snakers4/silero-vad , https://pypi.org/project/silero-vad/
- webrtcvad: https://github.com/wiseman/py-webrtcvad
- Smart Turn: https://github.com/pipecat-ai/smart-turn , https://huggingface.co/pipecat-ai/smart-turn-v3 , https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/
- Pipecat Smart Turn 문서: https://docs.pipecat.ai/server/utilities/smart-turn
- LiveKit 턴 감지기: https://docs.livekit.io/agents/build/turns/turn-detector/ , https://docs.livekit.io/reference/agents/turn-handling-options/ , https://livekit.com/blog/short-utterances
- 한국어 문말 억양: https://linguistics.stonybrook.edu/jiwonyun/presentation/2017_ICKL_handout_Yun.pdf , https://s-space.snu.ac.kr/handle/10371/29665
