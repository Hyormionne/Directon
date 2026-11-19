# Jev 턴 종료 판정 (AG-T24)

> 2026-11-20 · 모듈 `agent/turn/jev_turn.py` · 스위치 `agent/turn/detector.py` `make_detector()`(`DIRECTON_JEV_TURN=1`) ·
> 평가 `eval/jev_turn_eval.py` · 원자료 `data/turnset/jev_cache.jsonl`(응답 캐시), `jev_eval_report.md`(AUC), `jev_sim.json`·`jev_sim_report.md`(결합 시뮬레이션) ·
> 테스트 `tests/test_jev_turn.py` · 사람 인계는 `jev_handoff.md`

## 1. 요약

- **판정 품질은 좋다.** 쉼 지점의 STT 텍스트만으로 "말 끝 vs 말 중간 쉼" AUC는 **0.916**이다(1164지점). 규칙 `end_score`는 0.826, Smart Turn은 0.57이다.
  - 172문항은 0.908, 보정셋 test는 0.878이다.
  - 규칙이 '끝 쪽'으로 보는 애매 구간에서도 0.827이다(규칙 0.695).
- **지연은 침묵 창 안에 들어온다.** 실서버와 같은 경로(SDK 비동기 클라이언트, 연결 재사용, 순차 40회)로 재면 p50 **176ms**, p95 **247ms**, 최대 325ms다.
  - 연결을 새로 맺는 첫 호출은 408ms다.
  - 침묵 200ms에 LLM 선호출과 같이 요청하면 p는 침묵 약 **450ms**(p95)에 도착한다. 표 결정기의 대기는 대부분 600~1300ms라 결정 전에 쓸 수 있다.
  - 이 도착 시각에서는 결합 결과가 지연 0ms를 가정했을 때와 같다(4절, 240ms 변형). 따라서 미리 받아 두는 신호로만 쓸 필요는 없고, 결정에 바로 쓸 수 있다.
- **그러나 보정된 표 결정기(`CALIBRATED_CONFIG`) 위에 얹은 이득은 작다.**
  - 대기 늘리기(finished ≤ 0.3이면 상한까지): false cut이 train 56 → 53, test 23 → 22, 172 13 → 12로 준다. 대가로 p95가 +12/+14/−1ms, train miss가 +1건이다.
  - 대기 줄이기(finished ≥ hi): 지연 p50을 170~280ms 줄이지만 false cut이 는다. 172에서 hi 0.9를 쓰면 13 → 23건이다.
  - 대기 ~1초까지는 속도 점수 손실이 없으므로(turn_calibration.md) 줄이기에는 점수 이득이 없다.
- **권고: 옵트인(기본 꺼짐) 유지, 기본 채택은 하지 않는다.** 켤 때의 기본 정책은 '늘리기만'(`JevTurnPolicy(hi=1.01, lo=0.3)`)이다. 이유는 다음과 같다.
  - 이득(테스트 1건 수준)이 외부 API 의존성(네트워크·장애)에 비해 작다.
  - 남은 false cut은 "완결 문장 + 쉼 + 다음 문장"이다. 텍스트로는 원리상 끝과 구별할 수 없고, Jev도 finished 0.9 이상을 준다(5절).
- **API 호출:** 총 1,456회(모두 jev-1.14.0).
  - 이번 작업: turnset train 정답 텍스트 지점 738회, 지연 측정 81회, live 테스트 4회.
  - 인계 평가 633회(3회 반복 × 211)는 `jev_handoff.md`에 있다.
  - 앞선 실험 캐시 1,066회(AUC 보고서)는 다시 쓰고 새로 부르지 않았다.

## 2. 설정

- **SDK:** `typesafe-sdk` 0.7.5(`uv add`)의 `AsyncTypeSafeClient`. 재시도는 없음(`RetryPolicy(max_retries=0)`), 시간 초과는 0.6초(`DIRECTON_JEV_TIMEOUT_S`)다.
  - 테스트는 SDK에 `httpx2.MockTransport`를 주입한다. 평가기의 대량 호출은 httpx로 같은 HTTP 본문을 보낸다.
- **모델:** `jev-1.14.0`으로 고정했다. 바꾸면 캐시 키가 달라지므로 평가를 다시 돌린다.
- **상태(state)**, `jev_turn.build_state`:
  - `situation`: 통화·침묵·STT 오류 가능성 설명.
  - `previous_conversation`: 상담원 직전 질문 포함, 172문항은 cases의 history.
  - `caller_already_identified`.
  - `customer_speech_so_far`: 침묵 200ms 시점의 STT 텍스트.
- 누적 음성 길이와 침묵 길이는 넣지 않았다. 요청 시점의 침묵은 늘 200ms라 정보가 없고, 음성 길이는 표 결정기가 이미 쓴다.
- **질문:** 한 요청에 Noul 세 개를 묻는다. 병렬로 평가되므로 지연은 질문 1개와 비슷하다.

| id | 질문(요지) | 쓰임 |
|---|---|---|
| `finished` | 고객이 이번 턴을 마쳐 상담원이 지금 답해야 하나. false = 문장·숫자 중간, 간투사·망설임·자기 정정, 뒤에 본론(요청·번호·정정값)이 오는 도입부 | **결정에 씀** |
| `cut_off` | 텍스트가 문장·구·숫자 중간에서 끝났나 | 로그·분석 |
| `more_expected` | 곧 더 말할 것 같은가(번호·변경 내용·정정값·추가 질문) | 로그·분석 |

## 3. 판정 품질 (`uv run python -m eval.jev_turn_eval report`, `data/turnset/jev_eval_report.md`)

| 묶음 | n(끝/쉼) | finished | 1−cut_off | 1−more_expected | 규칙 end_score |
|---|---|---|---|---|---|
| 전체 | 1164 (586/578) | **0.916** | 0.862 | 0.895 | 0.826 |
| turnset test | 430 | 0.878 | 0.788 | 0.863 | 0.748 |
| 172문항(대화 맥락 포함) | 351 | 0.908 | 0.823 | 0.900 | 0.793 |
| 애매 구간(short·question·dcomp·final·soft·weak·neutral·comma) | 743 | 0.827 | 0.721 | 0.815 | 0.695 |

쉼 유형별 finished 평균과 p ≥ 0.7 비율:

- 연결어미 0.22(5%), 숫자 그룹 0.31(3%), 간투사 0.15, 조사 0.13, 자기 정정 0.29(7%).
- 말 끝은 0.81(81%), 172 말 끝은 0.89(95%).
- 약점은 **함정(완결 문장 뒤 쉼)** 0.73(64%)과 **172 말 중간** 0.49(32%)다. 텍스트만 보면 끝이 맞는 지점이다.

## 4. 표 결정기와 결합 (`uv run python -m eval.jev_turn_eval sim --latency 0 240 400`)

- **방법:** `eval/turn_sim.simulate_fast`를 그대로 쓴다.
  - 침묵 구간마다 침묵 200ms 시점의 텍스트로 Jev finished를 캐시에서 찾는다.
  - 그 값을 200+지연 ms부터 결정기에 보인다(어댑터 `SimJevDetector`가 smart_turn_p 자리에 싣는다).
  - 결합 규칙 `combine_wait`(래퍼 `JevTurnDetector`와 같음): finished ≥ hi이면 대기 = min(표, fast_ms), finished ≤ lo이면 대기 = 상한. 상한(1.3초, 예외 묶음 2초)은 그대로다.
- **조건:** r2 보정과 같다. turnset 빈 텍스트 10%, 172는 0%, 172 문항별 expect_short.
- **기준선 재현:** train 56 / test 23 / 172 13건, p95 1298/1356/1292, miss 9.3/9.2/7.6%로 `turn_calibration_r2.md` 6절과 일치한다.
- **p가 붙은 침묵 구간:** train 1115/1319, test 360/424, 172 346/346이다. 나머지는 빈 텍스트이고, 캐시에 없는 지점은 0이다.
- **정책 격자:** hi {0.8, 0.85, 0.9, 0.93, 0.95, 끔} × lo {끔, 0.1, 0.15, 0.2, 0.3} × fast {300…700}, 130개.

지연 240ms(p95 실측, p 도착 = 침묵 440ms). 칸은 false cut / 지연 p50 / p95 / miss 건수다.

| 정책 | train (732) | test (228) | 172 |
|---|---|---|---|
| 기준선(표만, 현재 `CALIBRATED_CONFIG`) | 56 / 971 / 1298 / 68 | 23 / 1045 / 1356 / 21 | 13 / 816 / 1292 / 13 |
| **늘리기만 lo 0.3(켤 때 기본)** | **53** / 973 / 1310 / 69 | **22** / 1045 / 1370 / 21 | **12** / 812 / 1291 / 13 |
| 늘리기만 lo 0.2 | 55 / 973 / 1305 / 69 | 22 / 1045 / 1361 / 21 | 12 / 812 / 1291 / 13 |
| 줄이기 hi 0.95 fast 600 + 늘리기 lo 0.3 | 55 / 955 / 1305 / 69 | 23 / 1041 / 1371 / 21 | 13 / 734 / 1265 / 12 |
| 줄이기 hi 0.93 fast 500 + lo 0.3 | 57 / 916 / 1303 / 69 | 23 / 1009 / 1371 / 21 | 16 / 646 / 1237 / 12 |
| 줄이기 hi 0.9 fast 500 | 61 / 795 / 1276 / 63 | 24 / 995 / 1356 / 21 | 22 / 601 / 1090 / 12 |
| 줄이기 hi 0.85 + lo 0.3 | 61 / 674 / 1275 / 61 | 24 / 920 / 1371 / 21 | 25 / 533 / 816 / 12 |
| 줄이기 hi 0.8 fast 400 + lo 0.3 | 66 / 571 / 1253 / 61 | 27 / 638 / 1361 / 21 | 27 / 531 / 765 / 12 |

- **기계적 선택 결과:** train에서 "miss·p95를 기준선보다 나쁘게 하지 않고 false cut 최소(안전형)"와 "false cut·miss를 기준선 이하로 두고 p95 최소(속도형)"를 골랐다. 둘 다 **기준선(Jev 안 씀)**이 뽑혔다. 기준선을 모든 축에서 이기는 정책은 없다.
- **지연 0ms와 240ms는 결과가 같다.** 400ms(도착 600ms)에서는 줄이기 효과가 조금 줄어든다(hi 0.9: 172 p50 601 → 698). 늘리기 결과는 같다.
- **이득이 작은 이유:**
  - 표 결정기가 이미 텍스트 유형 × 음성 길이로 보정돼 있다. 연결어미·조사·숫자 미완처럼 Jev가 잘 가르는 쉼은 표가 이미 1.3~2초 기다린다.
  - 남은 false cut은 "완결 문장 + 쉼 + 다음 문장"이다(172의 T0038·T0039·T0040·T0101·T0120 등). 텍스트로는 끝과 같아서 Jev도 finished가 높다.

## 5. 남는 오류 (애매 구간, finished가 크게 틀린 예)

- "아니요, 없어요.", "배송비 얼마야?", "주소 변경 가능할까요?", "상담원 좀 바꿔 주세요."는 쉼 뒤에 말이 이어졌는데 finished가 0.93~0.97이다. 텍스트로는 끝이 맞다.
- 반대로 "통계 중에 하나는 최소한", "아자 아자…"(172 말 끝 STT 오인식)는 0.06이다. STT가 틀리면 Jev도 틀린다.

## 6. 연동

### 6.1 이번에 넣은 것(기본 꺼짐)

- **`make_detector()`:** `DIRECTON_JEV_TURN=1`이고 `TYPESAFE_API_KEY`(환경변수 또는 `.env`)가 있으면 `JevTurnDetector(TableTurnDetector(CALIBRATED_CONFIG), shared_judge())`를 돌려준다.
  - 꺼져 있거나 키가 없으면 이전과 똑같다(같은 타입·같은 설정).
- **`JevTurnDetector`:** 계약은 `TurnDetector.decide`다.
  - `decide`가 침묵 ≥ 200ms이고 텍스트가 있는 첫 프레임에서 실행 중인 이벤트 루프에 Jev 요청 작업을 띄운다. 그래서 **세션 수정 없이도 동작**한다.
  - 텍스트가 바뀌면(부분 결과 → finalize) 한 침묵 안에서 최대 2회 다시 요청한다. 음성이 다시 시작되면 결과를 버린다.
  - p는 같은 텍스트일 때만 쓴다. 응답 전·실패·시간 초과이면 표 결정기와 같은 결정을 낸다(테스트로 확인).
  - `set_jev(text, p)`로 바깥에서 확률을 넣을 수 있고, `set_context(history, caller_known)`로 대화 맥락을 넣는다.
- 클라이언트는 프로세스에서 하나를 공유한다(`shared_judge`). 첫 요청 때 만든다.

### 6.2 세션 연결안(적용하지 않음: `agent/session.py`는 다른 작업 중)

켜려면 대화 맥락만 넘기면 된다. 맥락 없이도 동작하지만, 172문항 평가는 맥락을 넣은 결과다. 위치는 2026-11-20 현재 줄 번호 기준이다.

```python
# agent/session.py  start 처리, `self.expect_short = _expect_short_from_history(self.info.history)` (546행) 바로 뒤
        if hasattr(self.detector, "set_context"):
            self.detector.set_context(self.info.history, caller_known=bool(self.info.caller))

# 턴 끝에서 결정기를 새로 만든 뒤, `self.detector = self.d.detector_factory()` (1218행) 바로 뒤
        if hasattr(self.detector, "set_context"):
            hist = list(self.info.history or []) + [
                {"role": "customer" if m.role == "user" else "agent", "text": m.content}
                for m in self.conversation if m.role in ("user", "assistant")]
            self.detector.set_context(hist, caller_known=bool(self.info.caller))

# (선택) 첫 호출 408ms → 연결 예열: 세션 시작 때 한 번
#   from agent.turn.jev_turn import shared_judge; j = shared_judge()
#   if j: asyncio.create_task(j.judge("네"))
```

- 세션이 결정기를 매 턴 새로 만들므로(1218행) 진행 중인 Jev 작업은 버려진다. 래퍼의 `close()`를 부르면 깔끔하지만 필수는 아니다(결과는 무시된다).

## 7. 다시 돌리기

```
uv run python -m eval.jev_turn_eval run --workers 4        # 캐시에 없는 지점만 호출(현재 0회)
uv run python -m eval.jev_turn_eval report                 # AUC → data/turnset/jev_eval_report.md
uv run python -m eval.jev_turn_eval sim --latency 0 240 400  # 결합 시뮬레이션(API 호출 없음, 약 10분)
uv run python -m eval.jev_turn_eval latency --n 40         # SDK 경로 지연(81회 호출)
uv run pytest -m live tests/test_jev_turn.py               # 실제 호출 2회
```
