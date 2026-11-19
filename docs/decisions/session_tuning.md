# 세션 튜닝 (AG-T10c)

- 측정: 2026-11-19 · 결정기 AG-T12 커밋(d06e3f0) `make_detector()` 기준 · STT gemini-3.5-transcribe-live / LLM gemini-3.5-flash-lite(폴백 3.1-flash-lite) / TTS edge SunHi
- 변경: `agent/session.py`, `agent/turn/vad.py`(래퍼 추가), `agent/config.py`, `agent/tts_cache.py`, 새 `eval/vad_noise_eval.py`
- 끄는 스위치(A/B용): `DIRECTON_PRECALL_REFRESH=0`(선호출 개선 전부), `DIRECTON_OPENER_BY_TEXT=0`(텍스트 첫 마디), `DIRECTON_VAD_NOISE_GATE=0`(잡음 VAD)

## 1. 무엇을 바꿨나

| 항목 | 내용 |
|---|---|
| finalize 후 선호출 재시작 | 침묵 200ms 선호출(부분 결과) 뒤 finalize가 확정 전에 끝나면 그 텍스트로 선호출을 다시 건다(정규화 텍스트가 다를 때만). 부분 결과가 비어 선호출을 못 했던 짧은 발화도 이때 선호출된다 |
| 정규화 비교 | `session.match_key`: 어절마다 `agent.normalize.spoken_digits`로 한글 숫자→아라비아 숫자, 공백·문장부호 제거, 소문자. 확정 뒤 finalize 텍스트와 선호출 텍스트를 이걸로 비교해 같으면 재사용 |
| 첫 문장 선합성 | 선호출 LLM 스트림에서 응답기와 같은 `SentenceSplitter` 규칙으로 첫 문장을 찾으면 TTS를 미리 받아 **버퍼에만** 둔다. 재사용되면 `_PresynthTTS`가 그 버퍼를 이어 보낸다. 확정 전 송신 0(테스트로 확인) |
| 결정기 입력 텍스트 | finalize가 끝났으면 `TurnSignals.text`에 finalize 텍스트(부분 결과 대신). 빈 텍스트는 그대로 결정기 규칙(짧은 발화 정상 경로) |
| 텍스트 기반 첫 마디 | `tts_cache.select_opener_for_text`: 불만 키워드 → "네, 불편을 드려 죄송합니다.", 조회·처리 키워드 또는 4자리 이상 숫자 → "네, 확인해 드릴게요.", 그 밖 → 기본("네, 고객님."). 캐시에 없으면 기본. 선호출이 있으면 그 선호출을 만들 때 고른 첫 마디를 쓴다. 시스템 프롬프트는 첫 마디별로 `build_system(info, opener)`를 다시 불러 만든다(세션 캐시). 조회 첫 마디 턴의 툴 필러는 "잠시만 기다려 주세요."(“확인해 드릴게요” 반복 방지). 세 문구는 `Settings.cached_phrases()`에 들어가 서버 시작 때 합성 |
| 턴 로그 | `precall_reused`, `precall_count`, `presynth_reused`, `opener`, `opener_ms`, `gap_ms`(= 본 응답 첫 오디오 송신 − (첫 마디 송신 시작 + 첫 마디 길이)), marks `opener_start`·`response_first_audio`, events `precall(source=partial/finalize)`·`finalize_early`·`presynth`·`precall_replaced` |
| 잡음 강건 VAD | `agent.turn.vad.NoiseGate` / `NoiseRobustVAD(inner)` / `NoiseRobustSileroVAD(SileroVAD)`, `make_vad()`(서버 기본). 아래 2절 |

## 2. 잡음 강건 VAD

### 설계
- 잡음 바닥: 100ms 평활 에너지의 최근 3초 최솟값. 바닥 > −60 dBFS면 잡음 있음.
- 엄격 판정은 잡음이 있고 (a) **출렁이는 잡음**(스트림 첫 200ms 프레임 에너지 표준편차 ≥ 1.5 dB) 또는 (b) **내부 VAD가 고른 잡음에 속음**(바닥+3dB 미만이 300ms 이상 이어진 프레임에서 p≥0.5 비율의 지수 평균 ≥ 0.4)일 때만 건다.
- 엄격 판정: 시작 = p ≥ 0.6 + 에너지 ≥ 바닥+6dB + 3프레임 연속, 유지 = p ≥ 0.5 + 에너지 ≥ 바닥+3dB.
- 그 밖에는 Silero 확률 그대로(잡음 없음·white·pink·hum은 동작 불변).
- 왜 출렁임 조건인가: 모든 잡음에 걸면(첫 시도) white/pink/hum에서 recall 0.92→0.64~0.77, 말 중간 긴 침묵 오인(≥600ms) 6→15건으로 늘었다. Silero는 고른 잡음은 이미 잘 거른다(꼬리 프레임 p≥0.5 비율 1~2%). 문제는 babble뿐(43%). 첫 200ms 에너지 표준편차: babble 1.2~27 dB(중앙 2.6), white ≤0.8, hum ≤0.9, pink 0.4~2.8.
- "바닥 근처에서 Silero가 속는지"만으로 babble을 찾는 방식은 babble이 바닥 근처에 오래 머물지 않아 실패(활성화 0~4/27).

### 오프라인 측정(`uv run python -m eval.vad_noise_eval`, 172문항, 결과 `runs/vad_noise_eval_final/`)
정답 쉼은 합성 캐시로 깨끗한 트랙을 다시 조립해 구했다(172/172 복원). post200 = 말 끝 → '비음성 200ms 연속' 도달(세션 선호출·결정 시작점). 오인 = GT 쉼이 없는 곳의 말 중간 긴 침묵.

| 방식 | 잡음 | post200 p50 / p90 / 최대(ms) | 파일 끝까지 도달 못 함 | 오인 ≥400 / ≥600 / ≥900 | recall |
|---|---|---|---|---|---|
| Silero ≥0.5(이전) | babble(27) | 420 / 1500 / 1500 | **9** | 2 / 1 / 1 | 0.92 |
| Silero ≥0.7 | babble | 320 / 1500 / 1500 | 7 | 4 / 1 / 2 | 0.88 |
| **NoiseGate** | babble | **200 / 356** / 1500 | **1** | 5 / 2 / 2 | 0.80 |
| Silero ≥0.5 | white / pink / hum | 300·320·300 / 346·386·340 | 0 | 1·2·1 / 2·1·1 / 2·0·1 | 0.92~0.94 |
| NoiseGate | white / pink / hum | 동일(pink 1문항만 출렁임 판정) | 0 | 1·2·1 / 2·1·1 / 2·1·1 | 0.92~0.93 |
| 둘 다 | none(62) | 280 / 300 / 360 | 0 | 3 / 1 / 0 | 0.94 |
| Silero ≥0.5 | 전체 | 300 / 400 / 1500 | 9 | 9 / 6 / 4 | 0.93 |
| NoiseGate | 전체 | 280 / 340 / 1500 | 1 | 12 / 7 / 6 | 0.91 |

- 격자 탐색(`--sweep`, margin·hold·onset·p_noisy·출렁임 임계 48조합, `runs/vad_noise_eval_sweep3.log`)에서 오인 ≥600이 7건으로 가장 적은 축 중 babble p90이 가장 짧은 조합을 골랐다. 임계 0.7은 0.6보다 오인이 1건 많았다.
- 대가: babble 저SNR(5~10dB) 문항에서 새 오인 5건(T0009·T0048 ≥400, T0087 ≥600, T0041·T0061 ≥900). babble이 쉼을 메워 주던 것이 사라져 쉼이 '침묵'이 되기 때문이다.

### 턴 감지 보정셋에서 false cut(`uv run python -m eval.vad_noise_eval --turnset`, eval.turn_sim + make_detector, 771발화)

| VAD | 잡음 | false cut | 지연 p50 / p95(ms) | miss(끝까지 결정 없음) |
|---|---|---|---|---|
| Silero | 전체 | 129 (16.7%) | 912 / 1106 | 8.6% |
| NoiseGate | 전체 | **142 (18.4%)** | 905 / 1091 | 5.3% |
| Silero | babble(193) | 21 (10.9%) | 882 / 1182 | **34.2%** |
| NoiseGate | babble | **30 (15.5%)** | 874 / 1090 | **21.2%** |
| Silero / Gate | none(232) | 44 / 44 | 902 / 902 | 0 |
| Silero / Gate | white·pink·hum | 23·26·15 / 24·28·16 | 비슷 | 0 |

**false cut이 babble에서 +9건(+4.6%p), 전체 +13건 늘었다.** 메인 세션이 우려한 대로 래퍼가 babble 속 쉼을 '침묵'으로 만들어 짧은 쉼 확정이 는다. babble의 false cut 비율(15.5%)은 잡음 없음(19.0%)과 비슷해진 것이라, 래퍼가 babble을 "깨끗한 음성처럼" 만든 결과다(결정기 쪽 쉼 처리가 같은 수준이 된다). 반대로 babble miss(말 끝 뒤 확정 못 함 → 실서버에선 첫 소리 ~2초)는 34%→21%.
→ **판단은 메인 세션에 넘긴다.** false cut은 그 문항 속도 0점이고 miss는 1.5초 초과 감점이라 문항당 손해는 false cut이 크다. 보정셋 babble 기준 +9 false cut vs −25 miss. E2E(아래 3절)에선 babble 12문항 중 false cut 1→1(T0171 해소, T0087 신규). 기본값은 켬으로 두었고 `DIRECTON_VAD_NOISE_GATE=0`으로 끌 수 있다. 결정기 보정(AG-T12)을 래퍼 확률로 다시 하면 둘 다 잡을 여지가 있다(보정셋 특징 캐시를 `eval.vad_noise_eval.gate_probs`로 바꿔 넣으면 된다).

## 3. E2E (mock 8102 · agent 9102, smoke 20 + babble 10, 같은 시간대 A/B)

- A = 세 스위치 끔(이전 동작, 결정기만 T12), B = 모두 켬. 원시 결과 `runs/session_tuning/{A1,B1}/`(smoke·babble 하위, 턴 로그 `logs/`). 스크립트는 같은 폴더의 runner/score 로그 참고.
- 주의: 측정 시간대에 gemini-3.5-flash-lite가 첫 이벤트 2.5초 제한을 자주 넘어 폴백(3.1-flash-lite)으로 넘어갔다(A 로그 74줄, B 27줄). 그래서 확정 → LLM 첫 이벤트가 ~4초로 평소(~1초)보다 훨씬 길고, 공백 절대값이 크다. A/B 상대 비교만 의미가 있다.

| 지표 | 베이스라인(T09a 스모크 / 172) | A (끔) | B (켬) |
|---|---|---|---|
| 선호출 재사용 | 2/20 | 10/30 | **24/30** |
| 첫 문장 선합성 재사용 | - | 0 | 14/30 |
| 공백 gap_ms p50 / p90 | ~900(추정) | 1912 / 2783 | **1692 / 2509** |
| 확정 → 본 응답 첫 오디오 p50 | 1858 | 4104 | 4143 |
| 첫 마디 분포 | 고정 | 네, 고객님. 30 | 확인해 드릴게요 20 · 고객님 9 · 죄송합니다 1 |
| smoke 속도 평균 / 0점 / 첫 소리 p50·p95 | 59.5 / 8 / 0.50·1.94 | 74.5 / 5 / 0.75·1.87 | **80.0 / 4 / 0.63·1.44** |
| smoke 함수 정확률 | 0.35 | 0.60 | 0.55 |
| babble 10문항 속도 / 첫 소리 p50·p95 | (172 babble p50 1.89) | 85.0 / 2.12·2.44 | 83.0 / **1.31**·2.39 |
| babble 10문항 함수 정확률 | | 0.80 | 0.90 |

babble 문항별 첫 소리(초, *는 말 끝 전 발화):

| case | SNR | 베이스라인 172 | A | B |
|---|---|---|---|---|
| T0003 | 5 | 1.93 | 1.81 | **0.82** |
| T0013 | 10 | 1.89 | 1.80 | **0.52** |
| T0019 | 10 | 1.94 | 1.83 | 1.60 |
| T0022 | 30 | 0.60 | 0.78 | 0.39 |
| T0048 | 10 | 1.91 | 2.22 | 2.39 |
| T0075 | 5 | 1.90 | 2.20 | 2.18 |
| T0087 | 10 | −4.60* | 2.44 | **−4.79\*** |
| T0135 | 10 | 1.93 | 2.12 | 2.01 |
| T0139 | 10 | 1.90 | 2.20 | **0.81** |
| T0142 | 20 | 1.88 | 2.21 | **1.31** |
| T0009(smoke) | 5 | 1.90 | 1.87 | **0.63** |
| T0171(smoke) | 20 | −1.32* | −1.35* | **0.54** |

- babble 12문항 중 7문항이 1.5초 안으로 들어왔고, 3문항(T0048·T0075·T0135, SNR 5~10)은 그대로 ~2초, T0087은 새로 말 끝 전 발화(오프라인 예측과 같은 문항). 비-babble 문항 첫 소리는 A/B 동일(래퍼 비활성 확인).
- 공백이 재사용률만큼 줄지 않은 이유: 재사용되는 선호출은 대부분 finalize 재시작분이라 확정 100~400ms 전에 시작되는데, 이번 측정의 LLM 첫 이벤트가 ~4초여서 절약분이 작다. 공백 감소(−220ms p50)의 상당 부분은 조회 첫 마디가 더 길어서(약 1.4초 vs 1.0초)이기도 하다. LLM이 평소 속도(~1초)면 선호출 재사용·선합성의 효과가 상대적으로 커진다.
- T0042(부상 불만)에서 사과 첫 마디 뒤 LLM이 상담원 연결을 "연결해 드릴까요?"로 묻기만 하고 `transfer_to_agent`를 안 불러 함수 점수가 떨어졌다(A에선 불렀다). 1건이라 LLM 편차와 구분이 안 된다. 관찰 필요.

## 4. 빈 텍스트 짧은 발화
- 세션은 `TurnSignals.text=""`여도 결정기를 그대로 부른다(테스트: 800ms 발화, 텍스트 없음 → 결정기 확정 → finalize도 비면 재질문 문구, LLM 호출 0).
- make_detector(T12): speech 800ms·텍스트 없음은 `noise_short_speech`가 아니라 정상 표 경로(≤2초 안에 끝 판정, 테스트).
- finalize가 확정 전에 끝나면 결정기에 그 텍스트를 준다(짧은 발화 부분 결과 ~1.1초 지연 보완).

## 5. 재현
```
uv run python -m eval.vad_noise_eval [--sweep] [--out runs/vad_noise_eval_final]
uv run python -m eval.vad_noise_eval --turnset
# E2E: mock 8102 / agent 9102. A는 DIRECTON_PRECALL_REFRESH=0 DIRECTON_OPENER_BY_TEXT=0 DIRECTON_VAD_NOISE_GATE=0
uv run python -m eval.runner --server ws://127.0.0.1:9102 --cases data/testset/smoke/cases.jsonl --audio data/testset/smoke/audio \
  --mock http://127.0.0.1:8102 --out runs/session_tuning/<X>/smoke --concurrency 2
uv run python -m eval.runner ... --cases data/testset/cases.jsonl --audio data/testset/audio \
  --only T0003,T0013,T0019,T0048,T0075,T0087,T0135,T0139,T0022,T0142 --out runs/session_tuning/<X>/babble
uv run python -m eval.score --run runs/session_tuning/<X>/smoke
```
