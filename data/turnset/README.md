# 턴 보정 데이터셋 (AG-T12, 블록 2·3 AG-T15b)

한국어 통화 음성에서 "말 중간 쉼"(미완료)과 "진짜 말 끝"(완료)을 구분하는 턴 결정기를 보정하기 위한 데이터.
테스트셋(`legacy/testset/`)과 문장이 겹치지 않는다(정규화 후 5자 이상 세그먼트 일치 금지, 테스트로 검사).

| 파일 | 내용 |
|---|---|
| `src/utterances.jsonl` | 원문(세그먼트 + 쉼 길이 + 쉼 유형). 템플릿 736(숫자 그룹 블록 48 포함) + Gemini 초안 35 |
| `src/gemini_drafts.json` | Gemini 초안 캐시(규칙 검사 전) |
| `src/stt_cache.jsonl` | 지점별 실제 STT 결과 캐시(`uid:render_key:지점번호` → 텍스트, 모델) |
| `labels.jsonl` | 발화별 라벨: 분할·음성·환경·세그먼트, `points`(쉼 시작/최종 끝 시각, 쉼 유형·길이, `text_gt`, `text_stt`) |
| `calibration.json`, `calibration_report.md` | `eval/turn_calibrate.py` 결과(T12) |
| `calibration_r2*.json`, `calibration_r2*_report.md` | `eval/turn_calibrate.py r2` 결과(AG-T15b, `_turnset`: turnset만, `_mix`: turnset + 172문항) |
| `testset172/` | 172문항 시뮬레이션용: `probs/<case>.npy`(Silero 프레임 확률), `stt_points.jsonl`(말 중간 침묵 지점 STT), `end_texts.json`(말 끝 텍스트) |
| `audio/` | git 제외: `<uid>.wav`(16k mono), `cache/`(세그먼트 합성 캐시), `feats/`(VAD·Smart Turn 특징 캐시) |

## 다시 만들기

```
uv run python scripts/gen_turnset.py sources [--gemini]   # 원문(결정적, Gemini 초안은 캐시 재사용)
uv run python scripts/gen_turnset.py audio                # 합성·조립·통화 환경·라벨(바뀐 발화만)
uv run python scripts/gen_turnset.py stt --models gemini-3.5-transcribe,gemini-3.1-flash-lite --rpm 8
uv run python -m eval.turn_sim features                   # VAD·Smart Turn 특징 캐시
uv run python -m eval.turn_calibrate                      # 보정 → calibration.json/report.md
uv run pytest -q tests/test_turnset.py tests/test_turn_sim.py
```

## 블록 2·3 (AG-T15b, 960발화: train 732 / test 228)

기존 771발화의 uid·음성·환경·렌더 키는 그대로 두고 끝에 붙였다(`build_sources`가 블록 2·3을 마지막에 더함, 블록마다 환경 난수 분리).
음성은 edge·SAPI만(Gemini TTS 0회). 분할은 패턴 단위로 직접 정했다(test 패턴은 표현이 train과 다름, template_id `b2:<패턴>`).

| 블록 | 내용 | 발화(train/test) | 쉼 |
|---|---|---|---|
| 2 | 자기 정정·잠깐 멈춤(`SELF_CORR_PATTERNS`, 유형 `self_correction`, 쉼 유형 `self_correction`): "취소요. 아 아니다," [쉼] "교환으로 해 주세요.", "삼일, 아니" [쉼] "사일이요.", "공일공 …, 다시 할게요" [쉼] …, "잠시만요, 번호 찾아볼게요" [쉼] … | 36 / 27 | 300~1500ms |
| 2 | 숫자 그룹 낭독 변형(`DIGIT_GROUP2_PATTERNS`, 유형 `digits`): 그룹마다 쉼, '에'로 이어 부르기("핸드폰 번호 공일공에" [쉼] "칠육구팔에" [쉼] "사오팔칠요.") | 24 / 15 | 300~900ms |
| 2 | 위 패턴을 쉼 없이 한 번에 말한 대조군(유형 `complete`) | 22 / 13 | - |
| 3 | 질문·번호·짧은 답 뒤 쉼 + 추가 정보(`TRAP2_PATTERNS`, 유형 `trap`): "배송지 회사로 바꿀 수 있어요?" [쉼] "서울 마포구…", "이육… …" [쉼] "이거 배송 언제 와요?", "라지요." [쉼] … | 24 / 16 | 300~1200ms |
| 3 | 쉼 없는 대조군 | 8 / 4 | - |

STT: 새 test 지점 146개 전부 `gemini-3.1-flash-lite`(받아쓰기 프롬프트)로 전사. 새 train 지점은 `text_stt=null`(정답 텍스트로 대체).
`check_segment`에 쉼 유형 `self_correction`(끝이 `SELF_CORR_MARKERS` 중 하나)을 더했다.

## 구성 (블록 0·1: 771발화, train 618 / test 153)

아래 세부 분포는 처음 723발화 기준. 이후 숫자 그룹 쉼 블록 48발화(블록 1, `DIGIT_GROUP_PATTERNS`, 기존 uid·환경 유지)를 끝에 붙였다.

- 유형(첫 쉼 기준): connective 122, particle 136, digits 95, filler 82, trap 82, complete(쉼 없음) 206(28.5%)
- 쉼 수: 0개 206, 1개 478, 2개 39. 쉼 556개, 길이 분포(속도 반영 후) <300ms 20 · 300~600 166 · 600~1000 175 ·
  1000~1100 34 · 1100~1500 109 · 1500~2000 48 · >2000 4 (1.1초 초과 28.6%)
- 존대 613 / 반말 110. 원문 템플릿 351종
- 쉼 유형 정의(규칙 검사 `check_segment`): connective(-서/-는데/-고/-면/-니까/-지만…), particle(조사로 끝남),
  digits(한글 숫자로 끝남), filler(간투사만), trap(종결어미로 끝난 완결 문장 뒤에 요청이 이어짐). 마지막 세그먼트는
  `check_final`(종결어미·단답·숫자+요). 아라비아 숫자 금지.

### 음성 (가상 화자 = 원 음성 × 피치 −3/0/+3 반음)

| 원 음성 | train | test |
|---|---|---|
| edge ko-KR-SunHiNeural (여) | −3, 0, +3 | |
| edge ko-KR-HyunsuMultilingualNeural (남) | −3, 0, +3 | |
| sapi Heami (Windows 내장, 여) | −3, 0 | +3 |
| edge ko-KR-InJoonNeural (남) | | −3, 0, +3 (원 음성 통째로 test) |

Gemini TTS는 2026-11-19 한도(무료 모델당 하루 10회 → 결제 후 Tier 1 분당 10회, 메인 세션 합성과 경합) 때문에 넣지 않았다.
`scripts/gen_turnset.py`의 `BASE_VOICES` 또는 `TEST_ONLY_VOICES`(예: `gemini:Puck`, `gemini:Leda`)에 더하고
`audio`·`stt`·`features`를 다시 돌리면 바뀐 발화만 다시 만든다(합성·전사·특징 캐시 재사용).

### 분할

원문 템플릿 fold(텍스트 해시 % 5 == 0 → test)를 먼저 정하고, test 발화에는 test 가상 화자만, train 발화에는
train 가상 화자만 배정한다. 같은 문장은 어느 유형에서 쓰이든 같은 쪽 fold다. 숫자 패턴은 패턴 id 단위.

### 통화 환경(3.3절 축, 균형 배분)

channel wideband 217 / phone 253 / phone_ulaw 253 · noise none 218 / babble 181 / white·pink·hum 108씩 ·
SNR 30/20/10/5 = 133/172/123/77 · gain 0/−12/+6 = 434/181/108 · packet_loss 0.02 = 108 · speed 0.85/1.0/1.2 = 145/434/144 ·
lead 200~1500ms 균등. babble 풀은 같은 split의 다른 발화 원음.

### 라벨

세그먼트를 각각 합성 → 앞뒤 무음 trim → (피치 변형) → 쉼을 정확히 끼워 조립 → `eval.augment.augment`.
쉼 시작·다음 세그먼트 시작은 예상 위치 주변 창에서 깨끗한 트랙(`AugResult.clean`)의 음성 경계로 다시 잡고,
최종 끝은 `AugResult.speech_end`. 사인파 버스트 가짜 TTS로 20ms 이내 정확도를 테스트한다(속도 0.85/1.2, 잡음, 채널, 패킷 손실 포함).

### STT 텍스트

각 지점(쉼 시작, 최종 끝) + 200ms까지의 오디오(세션은 침묵 200ms에 finalize)를 실제 STT로 전사.
`gemini-3.5-transcribe`(transcription config)와 `gemini-3.1-flash-lite`(받아쓰기 프롬프트)를 모델당 분당 8회로
나눠 썼고, test 지점을 먼저 처리했다(test 284/284, train 383/1119; transcribe는 95회 후 일일 한도). 전사가 없는 지점은 `text_stt=null` → 시뮬레이터가 `text_gt`로 대체한다.
