# 측정 보고서 — baseline_172_1838

배점(공식): 응대 50% / 함수 20% / 속도 30%. 속도 표는 Starter Kit score_demo.py와 같다.

## 전체 요약

| 구분 | n | 속도 | 속도0 | 첫소리p50 | 첫소리p95 | 함수 | 함수정확률 | 응대 | 추정(응대제외) | 추정총점 |
|---|---|---|---|---|---|---|---|---|---|---|
| 전체 | 172 | 67.4 | 55 | 0.51 | 1.90 | 56.4 | 0.52 | - | 63.0 | - |

- 말 끝 전 발화: 55건, 에러: 0건, 속도 0점: 55건

## 범주(category)별

| category | n | 속도 | 속도0 | 첫소리p50 | 첫소리p95 | 함수 | 함수정확률 | 응대 | 추정(응대제외) | 추정총점 |
|---|---|---|---|---|---|---|---|---|---|---|
| complaint_handoff | 17 | 34.7 | 11 | 0.48 | 1.91 | 11.8 | 0.12 | - | 25.5 | - |
| faq | 19 | 87.4 | 2 | 0.50 | 1.94 | 94.7 | 0.95 | - | 90.3 | - |
| followup | 30 | 70.0 | 9 | 0.59 | 0.79 | 46.7 | 0.40 | - | 60.7 | - |
| need_info | 19 | 92.1 | 1 | 0.58 | 1.93 | 71.1 | 0.68 | - | 83.7 | - |
| out_of_scope | 14 | 85.7 | 2 | 0.50 | 0.72 | 100.0 | 1.00 | - | 91.4 | - |
| tool_action | 34 | 52.9 | 16 | 0.48 | 0.70 | 45.6 | 0.35 | - | 50.0 | - |
| tool_lookup | 39 | 63.8 | 14 | 0.50 | 1.08 | 51.3 | 0.49 | - | 58.8 | - |

## 태그(tags)별

| tags | n | 속도 | 속도0 | 첫소리p50 | 첫소리p95 | 함수 | 함수정확률 | 응대 | 추정(응대제외) | 추정총점 |
|---|---|---|---|---|---|---|---|---|---|---|
| (없음) | 19 | 73.2 | 5 | 0.50 | 1.89 | 73.7 | 0.74 | - | 73.4 | - |
| banmal | 21 | 80.5 | 4 | 0.51 | 1.90 | 52.4 | 0.52 | - | 69.2 | - |
| caller_known | 55 | 66.7 | 18 | 0.50 | 1.91 | 53.6 | 0.42 | - | 61.5 | - |
| connective_end | 16 | 61.9 | 6 | 0.51 | 1.93 | 46.9 | 0.44 | - | 55.9 | - |
| digits | 54 | 61.1 | 21 | 0.53 | 1.08 | 38.0 | 0.35 | - | 51.9 | - |
| filler_words | 16 | 50.0 | 8 | 0.45 | 0.72 | 59.4 | 0.56 | - | 53.8 | - |
| long | 11 | 62.7 | 4 | 0.51 | 1.88 | 54.5 | 0.45 | - | 59.5 | - |
| mid_pause | 34 | 43.5 | 19 | 0.53 | 1.94 | 45.6 | 0.41 | - | 44.4 | - |
| negation | 9 | 76.7 | 2 | 0.55 | 1.91 | 61.1 | 0.56 | - | 70.4 | - |
| self_correction | 8 | 12.5 | 7 | 0.50 | 0.50 | 0.0 | 0.00 | - | 7.5 | - |
| short | 31 | 95.5 | 1 | 0.50 | 1.90 | 72.6 | 0.68 | - | 86.3 | - |

## 채널(channel)별

| channel | n | 속도 | 속도0 | 첫소리p50 | 첫소리p95 | 함수 | 함수정확률 | 응대 | 추정(응대제외) | 추정총점 |
|---|---|---|---|---|---|---|---|---|---|---|
| phone | 57 | 66.1 | 19 | 0.50 | 1.90 | 56.1 | 0.54 | - | 62.1 | - |
| phone_ulaw | 57 | 58.9 | 23 | 0.54 | 1.93 | 48.2 | 0.40 | - | 54.7 | - |
| wideband | 58 | 76.9 | 13 | 0.49 | 1.89 | 64.7 | 0.62 | - | 72.0 | - |

## 잡음(noise)별

| noise | n | 속도 | 속도0 | 첫소리p50 | 첫소리p95 | 함수 | 함수정확률 | 응대 | 추정(응대제외) | 추정총점 |
|---|---|---|---|---|---|---|---|---|---|---|
| babble | 27 | 66.3 | 8 | 1.89 | 1.94 | 57.4 | 0.56 | - | 62.7 | - |
| hum | 27 | 77.8 | 6 | 0.50 | 0.67 | 63.0 | 0.59 | - | 71.9 | - |
| none | 62 | 69.4 | 19 | 0.48 | 0.69 | 57.3 | 0.52 | - | 64.5 | - |
| pink | 28 | 67.9 | 9 | 0.51 | 1.24 | 53.6 | 0.50 | - | 62.1 | - |
| white | 28 | 53.6 | 13 | 0.50 | 1.03 | 50.0 | 0.46 | - | 52.1 | - |

## 실패·감점 문항 (89건)

| case_id | 사유 |
|---|---|
| T0088 | 말 끝 전 발화; 함수 불일치(누락: check_stock{"product_name": "스마트워치 밴드", "option": "네이비"} / 여분: check_stock{"option": "핑크", "product_name": "스마트워치 밴드"}) |
| T0034 | 함수 불일치(누락: transfer_to_agent{"reason": "고객이 상담원 연결 요청"} / 여분: -) |
| T0119 | 함수 불일치(누락: request_callback{"phone_number": "01012348765"} / 여분: request_callback{"phone_number": "01012348766"}) |
| T0151 | 함수 불일치(누락: change_shipping_address{"order_id": "2611190412", "new_address": "경기도 수원시 팔달구 효원로 299, 703호"} / 여분: change_shipping_address{"order_id": "2611194212"}) |
| T0089 | 말 끝 전 발화; 함수 불일치(누락: get_product_info{"product_name": "노트북 파우치"} / 여분: -) |
| T0152 | 말 끝 전 발화; 함수 불일치(누락: request_exchange{"order_id": "2611120384", "reason": "M 사이즈로 교환"}, check_stock{"product_name": "기모 후드티", "option": "M"} / 여분: -) |
| T0120 | 말 끝 전 발화; 함수 불일치(누락: change_shipping_address{"order_id": "2611170026", "new_address": "울산 남구 번영로 150"} / 여분: -) |
| T0090 | 말 끝 전 발화; 함수 불일치(누락: - / 여분: get_order_list{"phone_number": "01034567812"}) |
| T0153 | 함수 불일치(누락: get_coupons{"phone_number": "01076543210"} / 여분: get_coupons{"phone_number": "0107543210"}) |
| T0154 | 말 끝 전 발화; 함수 불일치(누락: get_refund_status{"order_id": "2611100678"} / 여분: -) |
| T0091 | 함수 불일치(누락: - / 여분: get_order_info{"order_id": "2611190412"}) |
| T0122 | 함수 불일치(누락: request_exchange{"order_id": "2611120384", "reason": "M 사이즈로 교환"} / 여분: request_exchange{"order_id": "25120384"}) |
| T0060 | 말 끝 전 발화; 함수 불일치(누락: get_refund_status{"order_id": "2611140835"} / 여분: -) |
| T0028 | 말 끝 전 발화 |
| T0155 | 말 끝 전 발화; 함수 불일치(누락: transfer_to_agent{"reason": "반복 이해 실패로 상담원 연결 요청"} / 여분: -) |
| T0092 | 함수 불일치(누락: - / 여분: get_order_info{"order_id": "2611180321"}) |
| T0061 | 말 끝 전 발화; 함수 불일치(누락: get_refund_status{"order_id": "2611050412"} / 여분: get_order_info{"order_id": "251505411"}) |
| T0093 | 함수 불일치(누락: cancel_order{"order_id": "2611150231"} / 여분: get_order_info{"order_id": "261110231"}) |
| T0062 | 말 끝 전 발화; 함수 불일치(누락: track_delivery{"order_id": "2611160457"} / 여분: -) |
| T0094 | 함수 불일치(누락: cancel_order{"order_id": "2611140498"} / 여분: get_order_info{"order_id": "2611140498"}) |
| T0063 | 말 끝 전 발화; 함수 불일치(누락: track_delivery{"order_id": "2611170026"} / 여분: -) |
| T0127 | 함수 불일치(누락: - / 여분: get_order_info{"order_id": "261115023"}) |
| T0033 | 말 끝 전 발화 |
| T0065 | 말 끝 전 발화; 함수 불일치(누락: check_stock{"product_name": "캠핑 의자", "option": "베이지"} / 여분: get_product_info{"product_name": "캠핑 의자"}) |
| T0159 | 함수 불일치(누락: - / 여분: get_order_list{"phone_number": "01078123456"}) |
| T0097 | 말 끝 전 발화; 함수 불일치(누락: request_exchange{"order_id": "2611110593", "reason": "270 사이즈로 교환"} / 여분: -) |
| T0035 | 말 끝 전 발화; 함수 불일치(누락: transfer_to_agent{"reason": "고객이 사람 상담원 연결 요청"} / 여분: -) |
| T0066 | 함수 불일치(누락: get_product_info{"product_name": "보조배터리 10000"} / 여분: get_product_info{"product_name": "보조배터리 만"}) |
| T0036 | 말 끝 전 발화; 함수 불일치(누락: transfer_to_agent{"reason": "같은 문제로 반복 문의, 강한 불만"} / 여분: -) |
| T0161 | 함수 불일치(누락: - / 여분: get_order_list{"phone_number": "01023456781"}) |
| T0067 | 함수 불일치(누락: get_refund_status{"order_id": "2611140498"} / 여분: get_refund_status{"order_id": "26140498"}) |
| T0130 | 함수 불일치(누락: - / 여분: get_product_info{"product_name": "무선 이어폰 에어핏"}) |
| T0099 | 말 끝 전 발화; 함수 불일치(누락: change_shipping_address{"order_id": "2611190871", "new_address": "경기도 성남시 분당구 대왕판교로 660"} / 여분: -) |
| T0037 | 말 끝 전 발화; 함수 불일치(누락: transfer_to_agent{"reason": "소비자원 신고 언급"} / 여분: -) |
| T0100 | 말 끝 전 발화; 함수 불일치(누락: change_shipping_address{"order_id": "2611180189", "new_address": "광주 북구 용봉로 77"} / 여분: get_order_info{"order_id": "26111802189"}) |
| T0068 | 함수 불일치(누락: check_stock{"product_name": "요가 매트", "option": "그레이"} / 여분: check_stock{"product_name": "세댁 요가 매트"}) |
| T0101 | 말 끝 전 발화 |
| T0164 | 함수 불일치(누락: cancel_order{"order_id": "2611190412"} / 여분: -) |
| T0038 | 말 끝 전 발화; 함수 불일치(누락: transfer_to_agent{"reason": "배송 지연 강한 불만, 책임자 요청"} / 여분: get_order_list{"phone_number": "01023456781"}, track_delivery{"order_id": "2611150231"}) |
| T0039 | 말 끝 전 발화; 함수 불일치(누락: transfer_to_agent{"reason": "수령 7일 지난 불량 문의"} / 여분: -) |
| T0102 | 말 끝 전 발화; 함수 불일치(누락: request_callback{"phone_number": "01081234567"} / 여분: -) |
| T0002 | 말 끝 전 발화 |
| T0040 | 말 끝 전 발화; 함수 불일치(누락: transfer_to_agent{"reason": "반복 설명에 불만, 상담원 연결 요청"} / 여분: -) |
| T0133 | 함수 불일치(누락: - / 여분: get_member_info{"phone_number": "01034578111"}) |
| T0166 | 말 끝 전 발화; 함수 불일치(누락: check_stock{"product_name": "노트북 파우치", "option": "15인치"} / 여분: -) |
| T0134 | 말 끝 전 발화 |
| T0103 | 말 끝 전 발화; 함수 불일치(누락: request_callback{"phone_number": "01065432109"} / 여분: get_member_info{"phone_number": "01065432108"}) |
| T0041 | 말 끝 전 발화; 함수 불일치(누락: transfer_to_agent{"reason": "이중 결제 금액 오류"} / 여분: -) |
| T0104 | 말 끝 전 발화; 함수 불일치(누락: cancel_order{"order_id": "2611200342", "reason": "다른 상품으로 재주문"} / 여분: -) |
| T0072 | 말 끝 전 발화; 함수 불일치(누락: track_delivery{"order_id": "2611130147"} / 여분: get_order_info{"order_id": "2611130147"}) |
| T0135 | 함수 불일치(누락: get_order_list{"phone_number": "01045678123"} / 여분: -) |
| T0167 | 말 끝 전 발화; 함수 불일치(누락: get_member_info{"phone_number": "01043210987"} / 여분: -) |
| T0010 | 말 끝 전 발화 |
| T0042 | 말 끝 전 발화; 함수 불일치(누락: transfer_to_agent{"reason": "제품 사용 중 부상"} / 여분: -) |
| T0168 | 함수 불일치(누락: get_refund_status{"order_id": "2611070586"} / 여분: -) |
| T0043 | 말 끝 전 발화; 함수 불일치(누락: transfer_to_agent{"reason": "개인정보 유출 의심"} / 여분: -) |
| T0074 | 함수 불일치(누락: get_product_info{"product_name": "극세사 이불"} / 여분: get_product_info{"product_name": "국세서 이불"}) |
| T0169 | 말 끝 전 발화; 함수 불일치(누락: get_order_info{"order_id": "2611110169"} / 여분: -) |
| T0044 | 함수 불일치(누락: transfer_to_agent{"reason": "대량 주문 문의"} / 여분: -) |
| T0107 | 말 끝 전 발화; 함수 불일치(누락: request_exchange{"order_id": "2611170026"} / 여분: get_order_list{"phone_number": "01076543210"}) |
| T0076 | 말 끝 전 발화; 함수 불일치(누락: get_coupons{"phone_number": "01023456781"} / 여분: -) |
| T0045 | 함수 불일치(누락: request_callback{"phone_number": "01065432109"} / 여분: transfer_to_agent{}) |
| T0108 | 말 끝 전 발화; 함수 불일치(누락: request_return{"order_id": "2611180956"} / 여분: get_order_list{"phone_number": "01043210987"}) |
| T0171 | 말 끝 전 발화; 함수 불일치(누락: - / 여분: get_order_list{"phone_number": "01076543210"}) |
| T0014 | 함수 불일치(누락: - / 여분: get_product_info{"product_name": "3만 원짜리 상품"}) |
| T0046 | 함수 불일치(누락: request_callback{"phone_number": "01056781234"} / 여분: transfer_to_agent{}) |
| T0172 | 말 끝 전 발화; 함수 불일치(누락: get_order_info{"order_id": "2611170905"} / 여분: -) |
| T0140 | 함수 불일치(누락: - / 여분: get_member_info{"phone_number": "01012345678"}) |
| T0047 | 말 끝 전 발화; 함수 불일치(누락: request_callback{"phone_number": "01087654321"} / 여분: -) |
| T0078 | 말 끝 전 발화; 함수 불일치(누락: get_order_list{"phone_number": "01011112222"} / 여분: get_member_info{"phone_number": "01011112222"}) |
| T0141 | 함수 불일치(누락: - / 여분: get_refund_status{"order_id": "2611090314"}) |
| T0079 | 말 끝 전 발화; 함수 불일치(누락: get_refund_status{"order_id": "2611090314"} / 여분: -) |
| T0112 | 말 끝 전 발화; 함수 불일치(누락: cancel_order{"order_id": "2611170648"} / 여분: -) |
| T0050 | 말 끝 전 발화; 함수 불일치(누락: transfer_to_agent{"reason": "파손 상품 처리 지연 강한 불만"} / 여분: -) |
| T0080 | 함수 불일치(누락: track_delivery{"order_id": "2611160712"} / 여분: track_delivery{"order_id": "2611116072"}) |
| T0081 | 함수 불일치(누락: get_order_info{"order_id": "2611180764"} / 여분: get_order_info{"order_id": "61180764"}) |
| T0144 | 함수 불일치(누락: cancel_order{"order_id": "2611200342"} / 여분: get_order_info{"order_id": "2611200342"}) |
| T0051 | 말 끝 전 발화; 함수 불일치(누락: track_delivery{"order_id": "2611150231"} / 여분: -) |
| T0145 | 함수 불일치(누락: - / 여분: get_order_list{"phone_number": "01023456781"}) |
| T0114 | 함수 불일치(누락: request_exchange{"order_id": "2611060227"} / 여분: request_exchange{"order_id": "261106027"}) |
| T0115 | 말 끝 전 발화; 함수 불일치(누락: request_callback{"phone_number": "01023456781", "preferred_time": "월요일 오전"} / 여분: -) |
| T0083 | 말 끝 전 발화 |
| T0116 | 말 끝 전 발화; 함수 불일치(누락: change_shipping_address{"order_id": "2611190871", "new_address": "서울 강서구 공항대로 200"} / 여분: get_order_info{"order_id": "2611190871"}, cancel_order{"order_id": "2611190871"}) |
| T0117 | 말 끝 전 발화; 함수 불일치(누락: get_order_list{"phone_number": "01045678123"}, request_return{"order_id": "2611130147", "reason": "단순 변심"} / 여분: get_product_info{"product_name": "무선 이어폰 에어핏"}) |
| T0118 | 말 끝 전 발화; 함수 불일치(누락: cancel_order{"order_id": "2611160457"} / 여분: get_member_info{"phone_number": "01012345678"}) |
| T0149 | 말 끝 전 발화; 함수 불일치(누락: get_refund_status{"order_id": "2611180042"} / 여분: get_refund_status{"order_id": "2611"}) |
| T0086 | 함수 불일치(누락: track_delivery{"order_id": "2611180956"} / 여분: -) |
| T0150 | 함수 불일치(누락: - / 여분: get_order_list{"phone_number": "01056781234"}) |
| T0087 | 말 끝 전 발화; 함수 불일치(누락: track_delivery{"order_id": "2611190233"} / 여분: get_order_list{"phone_number": "01023456781"}) |
