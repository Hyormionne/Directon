# Jev 사람 인계 판정 오프라인 평가 (jev-1.14.0)

문항 212개(testset 172, practice_health 40). 정답 = expected_tools에 transfer_to_agent. 임계값 {'health': 0.7, 'human_request': 0.8, 'abuse_anger': 0.8, 'repeated_failure': 0.8, 'legal_dispute': 0.7, 'out_of_policy': 0.8}.

## 정밀도·재현율

| 묶음 | n | 인계 | TP | FP | FN | 정밀도 | 재현율 | AUC(max p) |
|---|---|---|---|---|---|---|---|---|
| testset Jev | 172 | 16 | 16 | 0 | 0 | 100% | 100% | 0.985 |
| testset LLM(round1) | 172 | 16 | 10 | 0 | 6 | 100% | 62% | - |
| testset LLM ∪ Jev | 172 | 16 | 16 | 0 | 0 | 100% | 100% | - |
| health Jev | 40 | 4 | 4 | 0 | 0 | 100% | 100% | 1.000 |
| 전체 Jev | 212 | 20 | 20 | 0 | 0 | 100% | 100% | 0.988 |

## 기준별 AUC(인계 vs 비인계)와 인계 문항 평균 p

| 기준 | 임계값 | AUC | 인계 평균 | 비인계 평균 | 비인계 중 p≥임계 |
|---|---|---|---|---|---|
| health | 0.7 | 0.639 | 0.20 | 0.01 | 0 |
| human_request | 0.8 | 0.924 | 0.51 | 0.05 | 4 |
| abuse_anger | 0.8 | 0.907 | 0.33 | 0.02 | 0 |
| repeated_failure | 0.8 | 0.751 | 0.20 | 0.02 | 0 |
| legal_dispute | 0.7 | 0.727 | 0.11 | 0.01 | 0 |
| out_of_policy | 0.8 | 0.969 | 0.48 | 0.07 | 0 |

## 임계값 일괄 변경 민감도(모든 기준 같은 값, testset+health)

| 임계값 | TP | FP | FN | 정밀도 | 재현율 |
|---|---|---|---|---|---|
| 0.5 | 20 | 2 | 0 | 91% | 100% |
| 0.6 | 20 | 2 | 0 | 91% | 100% |
| 0.7 | 20 | 0 | 0 | 100% | 100% |
| 0.8 | 20 | 0 | 0 | 100% | 100% |
| 0.9 | 18 | 0 | 2 | 100% | 90% |
| 0.95 | 15 | 0 | 5 | 100% | 75% |

영업시간 외 상담원 요청(정답 request_callback) 11건 중 Jev 경로 callback: 4건, transfer: 0건

## 문항별(인계 정답 또는 Jev 인계 판정)

| id | 분류 | 정답 | LLM | Jev 경로 | 근거 | 주요 p | 텍스트 |
|---|---|---|---|---|---|---|---|
| T0034 | complaint_handoff | O | O | transfer | human_request | human_request 0.99, out_of_policy 0.12, abuse_anger 0.02 | 상담원이랑 통화하고 싶어요. |
| T0035 | complaint_handoff | O | O | transfer | human_request | human_request 0.91, out_of_policy 0.19, abuse_anger 0.17 | Say in this style. 사람 바꿔 줘. 규현이랑은 얘기 안 할래. |
| T0036 | complaint_handoff | O | O | transfer | abuse_anger·repeated_failure | abuse_anger 0.93, repeated_failure 0.91, out_of_policy 0.45 | Say in this style. 화가 나서 격앙된 목소리로 벌써 세 번째 전화하는 건데요. 지난번에도 처리해 준다고 해놓고  |
| T0037 | complaint_handoff | O | X | transfer | abuse_anger·legal_dispute | legal_dispute 0.98, abuse_anger 0.80, out_of_policy 0.57 | Say in this style. 계속 이런식이면 소비자원에 신고할 거예요. |
| T0038 | complaint_handoff | O | X | transfer | human_request·abuse_anger | human_request 0.98, abuse_anger 0.95, out_of_policy 0.52 | 제 방임 장비가 몇갠데 팀장이 못 봤잖아요. 배송이 너무 늦어요. 이게 말이 돼요? 책임자 바꿔주세요. |
| T0039 | complaint_handoff | O | X | transfer | out_of_policy | out_of_policy 0.87, abuse_anger 0.79, human_request 0.05 | 한 달 전에 산 바지가 벌써 박음질이 다 터졌어요. 이거 분량인데 반품 기간 지났다고 안 된다는 게 말이 돼요? |
| T0040 | complaint_handoff | O | X | transfer | human_request·repeated_failure | human_request 0.99, repeated_failure 0.83, abuse_anger 0.71 | 아니 제가요. 지금 몇 번을 말씀드리는 거예요. 상담원 연결해 주세요. |
| T0041 | complaint_handoff | O | X | transfer | out_of_policy | out_of_policy 0.94, legal_dispute 0.40, abuse_anger 0.10 | Say in this style. 찍어 보냈어요. 카드에서 돈이 두 번 빠져나갔다고요. |
| T0042 | complaint_handoff | O | X | transfer | health | health 0.95, out_of_policy 0.59, abuse_anger 0.11 | 정기포트를 쓰다가 손을 데었어요. 제품에 문제 있는 거 아니에요? |
| T0043 | complaint_handoff | O | O | transfer | out_of_policy | out_of_policy 0.96, legal_dispute 0.58, abuse_anger 0.17 | 제 주소가 다른 사람한테 간 것 같은데요. 개인정보 유출 아니에요? |
| T0044 | complaint_handoff | O | O | transfer | out_of_policy | out_of_policy 0.87, human_request 0.02, repeated_failure 0.02 | 회사에서 텀블러를 한 200개 정도 주문하려고 하는데 단체 할인 같은 거 되나요? |
| T0045 | complaint_handoff | X | X | callback | human_request | human_request 0.99, out_of_policy 0.19, abuse_anger 0.02 | 상담원 연결해 주세요. |
| T0046 | complaint_handoff | X | X | callback | human_request | human_request 0.99, out_of_policy 0.14, abuse_anger 0.03 | 사람이랑 통화하고 싶은데 연결 좀 해줘. |
| T0047 | complaint_handoff | X | X | callback | human_request | human_request 0.98, out_of_policy 0.22, abuse_anger 0.16 | 상담원 좀 바꿔주세요. 급해요. |
| T0048 | complaint_handoff | O | O | transfer | human_request | human_request 0.99, out_of_policy 0.15, abuse_anger 0.06 | 환불 말구요. 그냥 사람이랑 직접 얘기하고 싶어 |
| T0049 | complaint_handoff | O | O | transfer | human_request | human_request 0.98, out_of_policy 0.16, abuse_anger 0.13 | 어 아까 설명을 들었는데 하나도 이해가 안돼서요. 사람이랑 다시 통화할 수 있을까요? |
| T0050 | complaint_handoff | O | O | transfer | human_request·abuse_anger | human_request 0.99, abuse_anger 0.90, repeated_failure 0.49 | Same this time. 물건이 다 깨져서 왔는데 Say in the style. 화난 목소리로 사진을 보내도 답이 없고  |
| T0155 | followup | O | O | transfer | human_request·repeated_failure | human_request 0.99, repeated_failure 0.97, abuse_anger 0.47 | 네. 연결해 주세요. |
| T0170 | followup | O | O | transfer | human_request | human_request 0.96, out_of_policy 0.14, abuse_anger 0.04 | 아 그래요? 그럼 그냥 상담원이랑 직접 얘기해 볼게요. |
| H0036 | complaint_handoff | O | - | transfer | health | health 0.97, out_of_policy 0.63, human_request 0.03 | 유산균 먹고 나서 배가 계속 아프고 설사를 해요. 이거 계속 먹어도 돼요? |
| H0037 | complaint_handoff | O | - | transfer | health | health 0.99, out_of_policy 0.72, human_request 0.03 | 오메가쓰리 먹고 나서 속이 울렁거리고 두드러기가 났어요. |
| H0038 | complaint_handoff | O | - | transfer | health | health 0.99, out_of_policy 0.77, human_request 0.03 | 홍삼스틱 먹고 나서부터 가슴이 두근거리고 좀 어지러워요. |
| H0039 | complaint_handoff | O | - | transfer | human_request | human_request 0.99, out_of_policy 0.13, abuse_anger 0.05 | 그냥 상담원이랑 직접 통화하고 싶어요. |
| H0040 | complaint_handoff | X | - | callback | human_request | human_request 0.92, out_of_policy 0.10, abuse_anger 0.03 | 지금 상담원 연결돼요? 안 되면 나중에 전화 주세요. |

응답 시간(평가 실행, 동시 4): p50 181ms, p95 266ms
