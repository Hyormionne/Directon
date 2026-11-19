# Jev 사람 인계 판정 (AG-T24 범위 추가, 2026-11-20 사용자 요청)

> 모듈 `agent/handoff_jev.py`(스위치 `DIRECTON_JEV_HANDOFF=1`, 기본 꺼짐) · 평가 `eval/jev_handoff_eval.py` ·
> 보고서 `data/turnset/jev_handoff_report.md`(최종), `runs/jev_handoff_report_v1.md`·`_v2.md`(질문 수정 전) · 캐시 `runs/jev_handoff_cache.jsonl` ·
> 테스트 `tests/test_jev_handoff.py`

## 1. 요약

- **testset 172문항 결과:** 인계 정답 16건 중 Jev는 **16건을 잡았고 오탐은 0건**이다(정밀도 100%, 재현율 100%).
  - round1 실서버 LLM은 10/16(재현율 62%, 오탐 0)이었다.
  - LLM이 놓친 6건(T0037 소비자원 신고, T0038 책임자 요청, T0039 반품 기한 경과 불만, T0040 반복·상담원 요청, T0041 이중 결제, T0042 화상)을 모두 잡는다.
- **practice_health 40문항 결과:** 인계 4건 중 4건을 잡았고 오탐은 0건이다. 섭취 후 복통·두드러기·두근거림 3건은 health 0.97~0.99로 잡혔다.
- **영업시간 외 상담원 요청:** 코드 정책이 transfer 대신 callback으로 돌린다(T0045·T0046·T0047·H0040, 4/4). 콜백 정답 11건 중 transfer로 잘못 보낸 것은 0건이다.
- **지연은 LLM 선호출 안에 숨는다.** 질문 6개를 한 요청으로 보내고 SDK 순차 40회로 재면 p50 **180ms**, p95 **246ms**다.
  - LLM 선호출과 함께 침묵 200ms에 쏘면 확정(보통 침묵 600ms 이상) 전에 결과가 와 있다.
- **주의(과적합):** 위 수치는 같은 212문항을 보며 질문 문구를 두 번, 임계값을 두 번 고친 뒤의 **표본 안** 결과다.
  - v1(첫 문구)에서는 testset 정밀도 67%, 재현율 75%였다(4절).
  - 따로 떼어 둔 검증셋이 없으므로 실서버 round2에서 확인해야 한다.
- **권고: 켤 가치가 있다.** 연결 방식은 판정을 LLM 프롬프트 지시로 넘기는 것이다(6절 패치).
  - 오탐 비용(불필요한 인계)은 낮지 않으므로 임계값은 보수적(0.7~0.8)으로 둔다.
  - 스위치는 기본 꺼짐으로 두고, 세션·응답기 연결 뒤 실서버로 확인하고 켠다.

## 2. 설계

- **상태:**
  - `situation`: AI 상담, 건강기능식품 포함 쇼핑몰, 상담원 있음.
  - `recent_conversation`: history 최근 6개.
  - `agent_failed_to_understand_count`: 호출자가 센 이해 실패 횟수. 오프라인 평가는 0.
  - `caller_already_identified`.
  - `customer_utterance`: 확정 STT 텍스트.
- **질문:** Noul 6개를 한 요청에 묻는다(병렬 평가). 정책은 코드가 가진다. 기준별 임계값을 두고, 하나라도 넘으면 인계한다.

| id | 판정(요지) | 임계값 |
|---|---|---|
| `health` | 제품 섭취·사용 후 몸 불편·이상 반응·부상. 주의사항 안내 **후에도** 의학적 판단을 계속 요구 | 0.7 |
| `human_request` | 상담원·책임자 연결 요청 또는 상담원 연결 제안 수락. 상담 가능 시간만 묻는 것은 아님 | 0.8 |
| `abuse_anger` | 욕설·모욕·강한 격앙("이게 말이 돼요?") | 0.8 |
| `repeated_failure` | 여러 번 물었다·계속 못 알아듣는다, 또는 실패 횟수 ≥ 2 | 0.8 |
| `legal_dispute` | 소비자원·경찰 신고, 소송, 카드 결제 분쟁, 사기·도용 | 0.7 |
| `out_of_policy` | 매뉴얼로 못 정하는 예외: 기한 지난 반품, 보상, 대량·법인 주문, 이중 결제, 개인정보 사고 | 0.8 |

- **경로 `route(j, now)`:** 인계이고 영업시간(평일 09~18시, 점심 12~13시 제외, 토·일 휴무)이면 `transfer`, 영업시간 밖이면 `callback`이다. 시각을 해석할 수 없으면 `transfer`다.
  - 영업시간은 매뉴얼 3절에서 가져왔다. 공휴일은 모르지만 대회 날짜(11-20 금, 11-21 토)에는 공휴일이 없다.
- **실패 처리:** 시간 초과(기본 1초)나 오류이면 None을 돌려준다. 이때 강제 인계는 없고 LLM 판단을 그대로 쓴다.
  - 스위치가 꺼져 있으면 `maybe_judge`가 아예 부르지 않는다.

## 3. 결과 (`uv run python -m eval.jev_handoff_eval report`)

| 묶음 | n | 인계 | TP | FP | FN | 정밀도 | 재현율 |
|---|---|---|---|---|---|---|---|
| testset Jev(최종) | 172 | 16 | 16 | 0 | 0 | 100% | 100% |
| testset LLM(round1 실서버) | 172 | 16 | 10 | 0 | 6 | 100% | 62% |
| testset LLM ∪ Jev | 172 | 16 | 16 | 0 | 0 | 100% | 100% |
| practice_health Jev | 40 | 4 | 4 | 0 | 0 | 100% | 100% |

- **텍스트:** testset은 실서버 STT 최종 텍스트(`end_texts.json`)를 썼다. "Say in this style." 같은 TTS 지시문 오염이 그대로 들어 있다. practice_health는 원문이다.
- **기준별 AUC(인계 vs 비인계, 최대값 0.988):** human_request 0.92, out_of_policy 0.97, abuse_anger 0.91, health 0.64, legal 0.73, repeated 0.75.
  - 기준별 AUC가 낮은 것은 각 기준이 인계 사유의 일부만 맡기 때문이다. 비인계 문항의 평균 p는 0.01~0.07이다.
- **임계값 민감도:** 모든 기준을 같은 값으로 두면 0.7~0.8에서 20/20·FP 0이고, 0.5~0.6에서는 FP 2건, 0.9에서는 FN 2건이다. 0.7~0.8 사이는 평평하다.
- **비인계 문항의 최대 p:** out_of_policy 0.61(T0031), abuse_anger 0.67(T0164), legal 0.06, repeated 0.06. 임계값까지 여유가 있다.

## 4. 반복 기록(같은 데이터, 표본 안)

| 판 | 바꾼 것 | testset TP/FP/FN | health TP/FP/FN | 남은 오류 |
|---|---|---|---|---|
| v1 | 첫 문구, 임계값 0.7/0.7/0.8/0.8/0.7/0.9 | 12/6/4 | 4/2/0 | FP: 상담 가능 시간 질문 3건(T0005·T0012·T0017), 영업시간 외 상담원 요청 3건(→ 경로 정책으로 해결). FN: T0039·T0041·T0043·T0044 |
| v2 | human_request에서 '가능 시간 질문' 제외, out_of_policy에 이중 결제·개인정보 사고 명시, 영업시간 경로 추가 | 15/0/1 | 4/1/0 | FN T0044(out_of_policy 0.88 < 0.9), FP H0031("고혈압 약 먹는 사람도 먹어도 돼요?") |
| v3(최종) | health: 첫 의학 질문은 주의사항 안내 대상(매뉴얼 9·10절), out_of_policy 0.8, human_request 0.8(T0017 "지금 통화 돼요?" 0.70 → callback 오경로 제거) | 16/0/0 | 4/0/0 | - |

- H0031은 데이터 안에서 기준이 엇갈린다. 매뉴얼 10절 3항은 "계속 묻는다(주의사항 안내 후에도)"이고, tools.json 설명은 "의학적 판단 필요 질문"이다. 정답 라벨은 매뉴얼 쪽(주의사항 안내, 인계 아님)이라 질문을 매뉴얼에 맞췄다.
  - 대가로 "처음부터 약 복용 중인데 먹어도 되냐"는 질문은 인계하지 않는다. LLM이 주의사항을 안내하고, 다시 물으면 history로 health가 켜진다.

## 5. 지연·비용

- **SDK 순차:** p50 180ms, p95 246ms, 최대 280ms(40회). 평가 실행(동시 4)에서는 p50 181ms, p95 266ms다. 입력은 약 900 토큰이다.
- **API 호출:** 633회(v1·v2·v3 × 211). 지연 측정 40회는 `jev_turn.md`의 81회에 포함되고, live 테스트는 2회다.

## 6. 연결안(적용하지 않음: `agent/session.py`·`agent/responder.py`·`agent/server.py`는 다른 작업 중)

- **방식:** Jev가 인계라고 하면 그 턴의 LLM 시스템 프롬프트 끝에 `system_hint(j, now)`를 붙인다.
  - 내용은 "공감 한 문장 → transfer_to_agent 호출"이고, 영업시간 밖이면 "request_callback 접수", 건강 이상이면 "병원 진료 권유 한 문장"이 붙는다.
  - 함수 인자(reason, 콜백 전화번호)와 말투는 LLM이 그대로 맡는다. 판정만 Jev로 옮긴다.
- **시점:** LLM 선호출(침묵 200ms)과 같은 순간에 쏜다. 확정 시점(보통 침묵 600~1300ms)에는 이미 결과가 있다.
  - Jev가 인계라고 하면 힌트 없이 시작한 선호출을 버리고 힌트를 붙여 다시 부른다(인계 턴에만, 드묾). 첫 마디(opener)는 이미 나갔으므로 LLM 첫 토큰 지연은 opener 재생에 가려진다.
- **위치:** 2026-11-20 현재 줄 번호 기준.

```python
# agent/session.py
from agent.handoff_jev import JevHandoff, handoff_enabled, route, system_hint   # 상단 import

# SessionDeps(데이터클래스)에 필드 추가
    handoff: "JevHandoff | None" = None          # server build_deps: JevHandoff() if handoff_enabled() and 키 있음

# Session.__init__ (상태 초기화부)
        self._handoff_task: asyncio.Task | None = None
        self._handoff_key = ""
        self.failures = 0                          # 빈 텍스트 재질문(reprompt)·응답 실패마다 +1

    def _handoff_history(self) -> list[dict]:
        conv = [{"role": "customer" if m.role == "user" else "agent", "text": m.content}
                for m in self.conversation if m.role in ("user", "assistant")]
        return list(self.info.history or []) + conv

    def _fire_handoff(self, text: str) -> None:
        if self.d.handoff is None or not handoff_enabled() or not (text or "").strip():
            return
        k = self._key(text)
        if k == self._handoff_key:
            return
        if self._handoff_task is not None:
            self._handoff_task.cancel()
        self._handoff_key = k
        self._handoff_task = asyncio.create_task(
            self.d.handoff.judge(text, self._handoff_history(), self.failures, bool(self.info.caller)))

    async def _await_handoff(self, text: str):
        self._fire_handoff(text)                   # 최종 텍스트가 선호출 텍스트와 다르면 다시
        t = self._handoff_task
        if t is None:
            return None
        try:
            return await asyncio.wait_for(asyncio.shield(t), 0.3)   # 보통 이미 끝나 있다
        except Exception:  # noqa: BLE001
            return None

# _start_precall, `pf = r.prefetch(self._system_for(opener), ...)` (843행) 바로 앞
        self._fire_handoff(text)

# _respond_turn, `messages = self.conversation + [Message("user", user_text)]` (1167행) 바로 앞
                    system = self._system_for(self.turn_opener)
                    j = await self._await_handoff(user_text)
                    hint = system_hint(j, self.info.now)
                    if hint:
                        system += hint
                        if pc is not None:              # 힌트 없는 선호출은 버린다
                            pc.cancel()
                            pc = None
                            tts = _PresynthTTS(self.d.tts, None)
                        self.tlog.set("jev_handoff", {"route": route(j, self.info.now), "reasons": j.reasons,
                                                      "probs": j.probs, "ms": round(j.ms)})
# 그리고 respond(...) 호출의 첫 인자 `self._system_for(self.turn_opener)` (1170행)를 `system`으로 바꾼다.

# 턴 끝 초기화(1218행 `self.detector = self.d.detector_factory()` 근처)
        self._handoff_task, self._handoff_key = None, ""
```

```python
# agent/server.py build_deps (SessionDeps 생성부)
        from agent.handoff_jev import JevHandoff, handoff_enabled
        from agent.turn.jev_turn import api_key
        handoff = JevHandoff() if handoff_enabled() and api_key() else None
        # SessionDeps(..., handoff=handoff)
```

- **응답기(`agent/responder.py`)는 바꾸지 않아도 된다.** 프롬프트 지시로 충분하다.
  - 지시를 어길 때를 막으려면 선택적으로 보강한다: `Responder.respond`에 `required_tool: str | None` 인자를 두고, 첫 LLM 응답에 그 함수 호출이 없으면 `tool_choice`를 강제해 한 번 더 부른다.
  - round1에서 인계 6건을 놓친 원인은 판정(그 함수를 불러야 하는지 몰랐음)이었고, 지시를 받고 따르지 않은 것은 아니었다. 그래서 1차는 프롬프트만으로 둔다.
- **테스트 제안:** `tests/test_session.py`에 fake JevHandoff(즉시 인계 판정)를 넣는다. 응답기에 전달된 system에 "transfer_to_agent"가 들어가는지, 스위치를 끄면 judge를 부르지 않는지 확인한다.
