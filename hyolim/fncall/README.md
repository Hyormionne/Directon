# fncall — 함수 호출 정확도 + 조회 뒤 지연 최소화

LLM 에게 번호를 맡기지 않고, 코드가 번호를 확정·검증한 뒤 함수를 부르는 모듈.

| 파일 | 역할 |
|---|---|
| `numbers_ko.py` | STT 텍스트 → 전화번호·주문번호 후보. "공일공 사구오칠…", "이천이십육 영구이구 다시 영영삼삼사이팔", 이름이 붙은 숫자, 송장·계좌번호 제외 |
| `slots.py` | 번호 우선순위: 이번 발화 → history(호출 인수·조회 결과·손님 말) → 발신번호. 들은 번호가 발신번호와 한 자리만 다르면 발신번호 채택 |
| `planner.py` | `plan_lookups` 즉시 보낼 조회(애매하면 변형 최대 8개 병렬), `gate_commit` 접수 함수 게이트(번호를 코드 값으로 덮어씀, 조회 전·동의 전 접수 차단), `suggest_commit` 규칙 기반 접수 판단 |
| `verify.py` | `ToolClient`: X-Session-Id 자동 첨부, 같은 호출 중복 제거 |
| `templates.py` | 조회·접수 결과 → 바로 읽을 문장 (LLM 재호출 없음). 채움말 목록 |
| `pipeline.py` | `run_turn()`: 한 턴 전체 (조회 → 접수 판단 → 필요 시 조회 후 접수 → 템플릿 답) |
| `eval_fncall.py` | 오프라인 평가 (음성·LLM·키 불필요, 수 초) |

## 서버에 붙이기

```python
async with ToolClient(session["tools_url"], session["session_id"]) as tc:
    task = asyncio.create_task(run_turn(session, final_transcript, tc.call))   # 엔드포인트 직후 바로
    await send_pcm(FILLER_PCM["get_order_info"])     # 미리 합성한 채움말 → 첫 소리
    out = await task
    if out.speech:  await tts_stream(out.speech)     # 템플릿 답 → 조회 뒤 답 = TTS 첫 조각 시간뿐
    else:           await llm_answer(session, out.calls)   # 여러 주문, 반품 흐름 안내 등
```

LLM 이 접수 함수를 고르는 구조라면 `run_turn(..., llm_commit=(name, args))` 로 넘긴다. 번호는 게이트가 교체한다.

## 측정 (연습 80문항, `python3 -m fncall.eval_fncall --stt-noise 0.5`)

| 조건 | 함수 점수 |
|---|---|
| 규칙 1차 (문맥 규칙 전) | 87.5 |
| 글 그대로 | 98.8 |
| 한글 숫자로 읽은 STT 흉내 (2가지 seed) | 98.8 |
| + 50% 문항에 숫자 한 자리 누락 | 98.8 (후보 병렬 검증 전 90.0) |

- 함수를 부른 38문항 중 32문항이 템플릿 답 → 조회 뒤 LLM 재호출 없음. 순차 함수 왕복은 37문항 1회, 1문항 2회.
- 남은 실패 P0071-05: 주문번호가 사진에만 있어 조회 불가.
- 주의: 규칙을 이 80문항을 보며 다듬었으므로 시험 문항에서는 이보다 낮을 수 있다. 실제 Soniox 출력으로 다시 재 봐야 한다.
