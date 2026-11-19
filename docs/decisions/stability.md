# AG-T18 외부 접속·안정성

측정 당일에는 주최 측 노트북의 측정기가 우리 서버 주소(노트북 IP 또는 터널 URL)로 붙어 4통화를 동시에 보낸다.
측정 중에는 코드 수정·재시작이 불가하다. 그래서 (1) 세션 하나의 예외가 서버·다른 세션을 죽이지 않고,
(2) edge-tts(비공식 서비스)가 막혀도 말을 이어 가며, (3) 외부에서 붙을 주소를 바로 만들 수 있어야 한다.

## 결정

| 항목 | 결정 | 위치 |
|---|---|---|
| TTS 대체 | 문장 단위 `FallbackTTS(edge → Gemini TTS)`. 1차가 첫 덩어리 전에 실패·2.0초 무응답·빈 출력이면 같은 문장을 2차로 합성. 1차는 30초 쿨다운 동안 건너뜀. 첫 덩어리 뒤 실패는 반복 없이 올림(상위가 캐시 사과) | `agent/providers/fallback_tts.py` |
| TTS 조립 | 서버는 `make_tts_with_fallback`을 먼저 찾는다. 1차 edge + `GEMINI_API_KEY`(또는 주입 client)일 때만 감싼다. `make_tts`는 그대로(기존 테스트·BAZE 경로 불변). BAZE 키만으로는 대체를 붙이지 않는다(토큰 소진) | `agent/providers/factory.py` |
| 캐시 키 | `FallbackTTS`는 1차 기준으로 캐시 네임스페이스를 잡는다(`_cache_voice_key`가 `.primary`를 펼침). 실측: 감싼 뒤 재기동에서 `cache_synthesized=0, cache_missing=[]` | `agent/server.py` |
| 세션 격리 | `handle()`은 예외를 밖으로 내지 않는다. `Session.run`이 죽으면 비상 모드: 응답 중(turn.end 미송신)이었으면 즉시 캐시 사과 + transcript + turn.end. 그 뒤로는 연결이 끊길 때까지 에너지 기준(RMS 300, 발화 200ms 이상 뒤 무음 1000ms)으로 고객 말 끝을 잡아 매 턴 사과 + turn.end. 고객 발화가 끝나기 전에는 소리를 보내지 않는다. 정상 모드에서는 최근 프레임(약 10초)을 deque에 넣기만 하고 에너지는 비상 모드에서만 계산한다(Session이 벽시계로 송신 속도를 맞추므로 프레임마다 계산을 얹으면 타이밍이 1프레임 밀린다 — 실제로 `test_opener_session`에서 확인) | `agent/server.py` `GuardedWS`·`_degraded` |
| ping | `ping_interval=20, ping_timeout=60`(4세션 처리 중 루프가 잠깐 바쁘거나 터널을 거쳐도 끊지 않게) | `agent/server.py` `serve` |
| 로그 원문 가리기 | `DIRECTON_NO_AUDIO_LOG=1`이면 STT 숫자 보정 로그·OpenAI 함수 인자 파싱 실패 로그에 원문 대신 `<N자, 숫자 M개>` | `agent/providers/_privacy.py`, `stt_recheck.py`, `openai_llm.py` |
| 외부 주소 | 1순위: 같은 와이파이에서 `ws://<노트북IP>:9000`(방화벽 9000 허용). 2순위: Cloudflare quick tunnel `wss://xxxx.trycloudflare.com` | `scripts/tunnel.ps1`, `scripts/tunnel.sh` |
| 부하 도구 | `eval.stress`: N세션 동시 × M회, 실패·말 끝 전 발화·첫 소리 p50/p95·서버 RSS | `eval/stress.py` |

환경변수(모두 선택): `DIRECTON_TTS_FALLBACK`(gemini|none, 기본 gemini), `TTS_FALLBACK_TIMEOUT_S`(2.0), `TTS_FALLBACK_COOLDOWN_S`(30).
2차 Gemini TTS는 `GEMINI_TTS_MODEL`·`GEMINI_TTS_VOICE`(기본 `gemini-3.8-flash-tts`/Kore)를 그대로 쓴다.

### 이미 Session이 처리하던 실패(변경 없음, 테스트로 확인)
LLM 예외·TTS 예외·함수(Mock) 연결 실패·STT 열기/송신/확정 실패는 `session.py _respond_turn`이 캐시 사과(또는 되묻기) + turn.end로 끝낸다.
4세션 동시 + 그중 하나에 고장 주입 테스트에서 8종 모두 나머지 3세션 정상, 고장 세션도 turn.end 1회.

## 실측 (2026-11-20 00:13, 로컬 노트북, GEMINI_API_KEY만)

구성: STT gemini-3.5-transcribe-live / LLM gemini-3.5-flash-lite hedge(3.1-flash-lite) / TTS edge SunHi(+Gemini 대체) / 문항 T0021~T0028(범위 밖 문의, 8개 중 1개는 800ms 중간 쉼).

```
uv run python -m eval.stress --server ws://127.0.0.1:9190 --mock http://127.0.0.1:8190 \
  --sessions 4 --rounds 2 --offset 20 --out runs/stress/smoke_4x2.json
```

| 지표 | 값 |
|---|---|
| 세션 | 8 (4동시 × 2회) |
| 실패(오류·turn.end 없음·소리 없음) | 0 |
| 말 끝 전 발화 | 0 |
| 첫 소리 p50 / p95 / 최대 | 0.86s / 1.21s / 1.22s |
| turn.end p50 / p95 (speech_end 기준) | 3.26s / 3.59s |
| 서버 RSS 시작 → 최고 → 끝 | 159MB → 246MB → 236MB |
| 서버 로그 경고·오류 | 없음(예열 시 2.5-flash-lite 404 제외 경고만) |

- 8개 모두 내용 있는 응답(범위 밖 안내). TTS 대체는 발동하지 않았다(edge 정상).
- RSS +78MB는 첫 8세션의 지연 초기화(STT 연결·LLM 캐시·onnx 세션 버퍼) 범위로 보인다. 8세션으로는 누수 여부를 판단할 수 없다 → 당일 전 `--rounds 10` 이상으로 한 번 더 본다.
- TTS 대체 실측: edge를 강제로 실패시키면(ffmpeg 경로 오류) 같은 문장이 Gemini TTS로 첫 덩어리 1.04초, 3.28초 분량, 짝수 바이트, 레벨 정규화 후 음성 구간 RMS 2681(측정기 기준 300의 약 9배).

### 실측 중 발견
- **Windows 포트 중복 바인딩**: 다른 작업이 띄운 서버가 `127.0.0.1:9000`에 떠 있는데 새 서버가 `0.0.0.0:9000`으로 오류 없이 떴다. `127.0.0.1:9000` 연결은 먼저 떠 있던(더 구체적인) 쪽으로 갔다. 첫 스모크는 그 오래된 서버를 쟀고, 8개 모두 STT 빈 결과 → 되묻기였다. 같은 오디오를 새 코드로 돌리면 정상 인식. 당일에는 **측정 전에 9000을 쓰는 프로세스가 하나뿐인지 확인**한다(`Get-NetTCPConnection -LocalPort 9000 -State Listen`).
- 고객 숫자 원문 로그가 하나 더 있다: `agent/responder.py`의 `log.info("number arg filled ... %s=%r -> %s")`(함수 인자 숫자). responder 담당 작업에서 같은 `redact()`를 쓰도록 넘긴다.

## 당일 절차(요약)
1. `uv run python -m agent.server --port 9000` (예열 로그에서 `cache_missing=[]`, `stt_ok=True` 확인)
2. 같은 와이파이면 `ipconfig`의 IPv4로 `ws://<IP>:9000` 전달. 방화벽 인바운드 TCP 9000 허용.
3. 터널이 필요하면 `powershell -ExecutionPolicy Bypass -File scripts\tunnel.ps1` → 출력된 `wss://…trycloudflare.com` 전달. 창을 닫으면 주소가 사라지고, 다시 켜면 주소가 바뀐다.
4. 점검: `uv run python -m eval.stress --server <주소> --sessions 4 --rounds 1`(Mock 주소는 `--mock`).

cloudflared 설치(Windows, 계정 불필요): `winget install --id Cloudflare.cloudflared` → 새 터미널에서 `cloudflared --version`.
