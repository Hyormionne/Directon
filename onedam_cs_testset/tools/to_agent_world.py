"""온담헬스 데이터셋 → 상담원(Directon) 당일 교체 형식으로 변환해, 실제 서버로 측정·채점할 수 있게 만든다.

사용(프로젝트 루트에서):
  uv run python onedam_cs_testset/tools/to_agent_world.py            # audio/에 있는 문항만
  uv run python onedam_cs_testset/tools/to_agent_world.py --all      # 녹음 없는 문항까지 cases.jsonl에 포함

출력 onedam_cs_testset/agent_world/:
  tools.json      practice_health와 같은 함수 15개(온담 상품명·12자리 주문번호로 설명만 교체)
  mock_data.json  Starter Kit 매핑 형식 {함수: [{arguments, result}]} — mock_api --data, DIRECTON_CATALOG_PATH
  manual.md       mock_db.json의 정책으로 만든 응대 매뉴얼 — DIRECTON_MANUAL
  cases.jsonl     eval.runner/eval.score 스키마(expected_tools는 위 함수 이름으로 변환, 정책 질문은 함수 없음)
  audio/          문항 음성 사본 + meta.json(speech_end)
  mapping_notes.json  변환하지 못한 기대 함수(데이터셋 가정 함수 중 상담원 함수에 없는 것)

주문번호: 데이터셋 ORD-20261123-0777 → 숫자 12자리 202611230777 (상담원 정규화가 숫자만 남기므로).
"""
import argparse, json, os, re, shutil, sys, wave
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.dirname(HERE)
ROOT = os.path.dirname(DS)
OUT = os.path.join(DS, "agent_world")
PRACTICE_TOOLS = os.path.join(ROOT, "data", "practice_health", "tools.json")
BRAND = "온담헬스"


def digits(s):
    return re.sub(r"\D", "", s or "")


def oid(s):
    return digits(s)


# ───────────── 함수 정의 ─────────────
def build_tools(products, sample_oid):
    tools = json.load(open(PRACTICE_TOOLS, encoding="utf-8"))
    ex_name = products["P02"]["name"]
    for t in tools:
        t["description"] = t["description"].replace("온결헬스", BRAND).replace("온결", "온담")
        for k, p in t["parameters"]["properties"].items():
            d = p.get("description", "").replace("온결헬스", BRAND)
            if k == "order_id":
                d = f"주문번호. 숫자 12자리(주문일 8자리 + 일련번호 4자리). 고객이 '대시'를 넣어 읽어도 숫자만. 예: {sample_oid}"
            elif k == "product_name":
                d = f"{BRAND} 상품명(정확한 이름). 예: {ex_name}"
            elif k == "option":
                d = "옵션명(맛). 예: 레몬맛. 생략하면 전체 옵션."
            p["description"] = d.replace("온결", "온담")
    return tools


# ───────────── Mock 데이터 ─────────────
STATUS = {"결제완료(출고전)": "결제완료", "출고완료": "배송중"}


def product_options(p):
    m = re.search(r"옵션:\s*([^,]+)", p.get("note", ""))
    return [o.strip() for o in m.group(1).split("/")] if m else ["기본"]


def build_mock(db):
    P, O, C = db["products"], db["orders"], db["customers"]
    data = {k: [] for k in ("get_member_info", "get_order_list", "get_order_info", "track_delivery",
                            "get_payment_info", "get_product_info", "check_stock", "get_refund_status",
                            "transfer_to_agent", "request_callback")}
    for pid, p in P.items():
        opts = product_options(p)
        desc = f"섭취 방법: {p['dose']}. 보관: {p['storage']}." + (f" 참고: {p['note']}." if p.get("note") else "")
        data["get_product_info"].append({"arguments": {"product_name": p["name"]}, "result": {"status": "ok", "data": {
            "name": p["name"], "price": p["price"], "volume": p["unit"], "how_to_take": p["dose"],
            "storage": p["storage"], "notes": p.get("note", ""), "options": opts, "description": desc,
            "shipping_policy": db["policies"]["shipping"]}}})
        restock = re.search(r"(\d+월 \d+일) 재입고", p.get("note", ""))
        def stock_row(o):
            r = {"option": o, "stock": p["stock"], "in_stock": p["stock"] > 0}
            if restock and p["stock"] == 0: r["restock_date"] = restock.group(1)
            return r
        data["check_stock"].append({"arguments": {"product_name": p["name"]}, "result": {"status": "ok", "data": {
            "product_name": p["name"], "options": [stock_row(o) for o in opts]}}})
        for o in opts:
            data["check_stock"].append({"arguments": {"product_name": p["name"], "option": o}, "result": {
                "status": "ok", "data": {"product_name": p["name"], "options": [stock_row(o)]}}})

    def items(o):
        out = []
        for it in o["items"]:          # [상품ID, 수량] 또는 [상품ID, 수량, 옵션]
            row = {"product_name": P[it[0]]["name"], "quantity": it[1], "unit_price": P[it[0]]["price"]}
            if len(it) > 2: row["option"] = it[2]
            out.append(row)
        return out

    by_phone = {}
    for k, o in O.items():
        i = oid(k)
        st = STATUS.get(o["status"], o["status"])
        ordered = f"{i[:4]}-{i[4:6]}-{i[6:8]}"
        info = {"order_id": i, "ordered_at": ordered, "status": st, "items": items(o), "total_amount": o["amount"],
                "payment_method": o["pay"]}
        if o.get("ship_to"): info["shipping_address"] = o["ship_to"]
        for extra in ("note", "delivery_note", "received_item_note", "coupon_used", "ship_plan"):
            if o.get(extra): info[extra] = o[extra]
        data["get_order_info"].append({"arguments": {"order_id": i}, "result": {"status": "ok", "data": info}})
        tr = {"order_id": i, "status": st, "courier": "CJ대한통운"}
        if o.get("tracking"): tr["tracking_number"] = digits(o["tracking"])
        if o.get("eta"): tr["estimated_delivery"] = o["eta"]
        if o.get("delivered"): tr["delivered_at"] = o["delivered"]
        msg = {"배송중": f"배송중입니다. {o.get('eta', '')} 도착 예정입니다.".replace("  ", " "),
               "배송완료": f"{o.get('delivered', '')} 배송 완료되었습니다."}.get(st, f"현재 {st} 상태입니다.")
        data["track_delivery"].append({"arguments": {"order_id": i}, "result": {"status": "ok", "message": msg, "data": tr}})
        pay = {"order_id": i, "method": o["pay"], "amount": o["amount"], "status": "결제완료"}
        if o.get("paid_at"): pay["paid_at"] = o["paid_at"]
        data["get_payment_info"].append({"arguments": {"order_id": i}, "result": {"status": "ok", "data": pay}})
        if st in ("결제완료", "상품준비중"):
            cres = {"status": "ok", "message": "주문이 취소되었습니다. 결제 수단으로 전액 환불됩니다.",
                    "data": {"order_id": i, "status": "취소완료", "refund_amount": o["amount"]}}
        else:
            cres = {"status": "error", "error": "not_allowed",
                    "message": f"{st} 상태라 취소할 수 없습니다. 받으신 뒤 반품으로 접수해 주세요."}
        data.setdefault("cancel_order", []).append({"arguments": {"order_id": i}, "result": cres})
        if o.get("return"):
            data["get_refund_status"].append({"arguments": {"order_id": i}, "result": {"status": "ok", "data": {
                "order_id": i, **{k2: v for k2, v in o["return"].items()}}}})
        phone = digits(C[o["customer"]]["phone"]) if o.get("customer") in C else digits(o.get("guest_phone"))
        if phone:
            by_phone.setdefault(phone, []).append({"order_id": i, "ordered_at": ordered, "status": st,
                                                   "product_name": items(o)[0]["product_name"], "total_amount": o["amount"]})
    # 상담원 함수에는 정기구독 전용 함수가 없어서, 구독 정보는 회원 조회 결과에 넣는다
    subs_by_cust = {}
    for sid, s in db.get("subscriptions", {}).items():
        subs_by_cust.setdefault(s["customer"], []).append(
            {"subscription_id": sid, "products": [P[x]["name"] for x in s["items"]],
             **{k: v for k, v in s.items() if k not in ("customer", "items")}})
    for cid, c in C.items():
        ph = digits(c["phone"])
        data["get_member_info"].append({"arguments": {"phone_number": ph}, "result": {"status": "ok", "data": {
            "phone_number": ph, "name": c["name"], "grade": c["grade"], "points": c.get("points"),
            "coupons": c.get("coupons", []), "address": c.get("address"),
            "subscriptions": subs_by_cust.get(cid, [])}}})
        data["request_callback"].append({"arguments": {"phone_number": ph}, "result": {"status": "ok",
            "message": "콜백 요청이 접수되었습니다. 영업시간(평일 09:00~18:00) 중 상담원이 연락드립니다."}})
    for ph, lst in by_phone.items():
        name = next((c["name"] for c in C.values() if digits(c["phone"]) == ph), None)
        data["get_order_list"].append({"arguments": {"phone_number": ph}, "result": {"status": "ok",
                                                                                      "data": {"name": name, "orders": lst}}})
    data["transfer_to_agent"].append({"arguments": {}, "result": {"status": "ok",
        "message": "상담원에게 연결합니다. 예상 대기 시간은 약 2분입니다."}})
    return data


# ───────────── 매뉴얼 ─────────────
def build_manual(db):
    pol, P = db["policies"], db["products"]
    plist = "\n".join(f"- {p['name']} — {p['price']:,}원, {p['unit']}" for p in P.values())
    return f"""# {BRAND} 고객센터 음성 응대 매뉴얼 (가상, 측정용)

> `onedam_cs_testset/tools/to_agent_world.py`가 `mock_db.json` 정책으로 만든 파일. **{BRAND}은 실제 회사가 아닌 가상의 건강기능식품 쇼핑몰이다.**
> 함수·데이터는 같은 폴더의 `tools.json`·`mock_data.json`. 사용: `DIRECTON_MANUAL=onedam_cs_testset/agent_world/manual.md`.

## 0. 업무 원칙

- **확인되지 않은 정보는 안내하지 않는다.** 주문·결제·배송·환불 정보는 함수 조회 결과로만 말한다. 상품 정보는 `get_product_info` 결과에 있는 내용만 말한다.
- **필요한 정보를 받기 전에 업무를 실행하지 않는다.** 취소·반품·교환·배송지 변경은 주문번호(또는 본인 확인된 전화번호로 찾은 주문)와 고객의 분명한 요청이 있어야 실행한다.
- **이 매뉴얼에 없는 정책은 만들지 않는다.** 모르는 것은 상담원 연결이나 콜백을 제안한다.
- **건강기능식품은 의약품이 아니다.** 질병 예방·치료 효과를 말하지 않고, 질환·복용 약·증상에 대한 의학적 판단을 하지 않는다.

## 1. 인사와 호칭

- 첫 응답: "안녕하세요, {BRAND} 고객센터입니다." 뒤에 바로 용건에 답한다.
- 고객 이름을 알면 "OOO 고객님", 모르면 "고객님".
- 고객이 용건 없이 "다시 걸게요"처럼 끊으려 하면 붙잡지 말고 짧게 끝인사한다.

## 2. 본인 확인

- 주문·결제·환불처럼 개인 정보가 필요한 요청은 주문번호 또는 휴대폰 번호로 확인한다. 비회원 주문도 주문번호 + 주문자 휴대폰 번호로 확인한다.
- **통화 정보(caller)에 휴대폰 번호가 있으면 다시 묻지 않는다.** 그 번호로 `get_order_list`를 조회한다. 발신번호가 없거나 일반전화면 주문번호나 주문자 휴대폰 번호를 요청한다.
- 주문번호는 **숫자 12자리**(주문일 8자리 + 일련번호 4자리, 예: 202611230777). 고객이 "이공이육 일일이삼 대시 공칠칠칠"처럼 읽으면 숫자만 이어 붙인다.

## 3. 영업시간

- {pol['hours']}

## 4. 배송

- {pol['shipping']}
- 배송 문의는 `track_delivery`. 송장번호는 읽지 않고 문자로 보내 드린다고 안내한다.

## 5. 주문 취소

- {pol['cancel']}

## 6. 반품·교환·환불

- {pol['return']}
- {pol['refund']}

## 7. 정기구독

- {pol['subscription']}
- 고객의 구독 상품·주기·다음 결제일·금액·결제카드는 `get_member_info`(휴대폰 번호) 결과의 `subscriptions`로 확인한다. 쿠폰·포인트도 같은 결과에 있다.

## 8. 결제·포인트·증빙

- 결제 수단: {pol['payment']}
- 무통장 입금 계좌는 주문마다 다르다. 주문번호나 주문자 휴대폰 번호로 주문을 확인한 뒤 계좌를 문자로 다시 보내 드린다고 안내한다.
- 포인트: {pol['points']}
- 증빙: {pol['documents']}
- 카드 번호·비밀번호는 묻지도 말하지도 않는다.

## 9. 상품

- 가격·용량·섭취 방법·보관·주의사항은 `get_product_info`, 재고는 `check_stock`(품절이면 재입고 예정일).
- 유통기한: {pol['expiry']}
- 상품명이 모호하면 아래 이름으로 확인해 조회한다("오메가3" → 온담 rTG 오메가3, "철분제" → 온담 철분 엽산).
{plist}
- 어린이 섭취 문의는 해당 상품 정보(`get_product_info`)의 대상 연령을 확인해 안내하고, 소아과·약사 상담을 권한다. 어린이 용량을 임의로 말하지 않는다.
- 할인·프로모션: {pol['promotion']}

## 10. 매장·회사

- {pol['store']}

## 11. 이상 반응과 상담원 연결

- {pol['adverse']}
- 다음이면 `transfer_to_agent`: 섭취 후 몸이 불편하다는 문의(가장 우선), 고객의 직접 요청, 의학적 판단이 필요한 질문의 반복, 욕설·강한 불만, 세 번 이상 이해 실패, 매뉴얼로 처리할 수 없는 예외(대량 구매 B2B 포함).
- 운영시간 외 상담원 요청은 `request_callback`.

## 12. 응답 스타일 (음성)

- **한 번에 2~3문장**, 핵심부터. 목록·표·이모지 금지.
- 금액 "삼만 이천 원", 날짜 "십일월 이십오일"처럼 읽는다.
- {BRAND} 업무가 아닌 질문(잘못 건 전화, 날씨 등)은 {BRAND} 고객센터임을 밝히고 정중히 범위 밖임을 알린다.
"""


# ───────────── 문항 ─────────────
def map_tools(rec, P, C):
    out, unmapped = [], []
    for t in rec["expected_tools"]:
        n, a = t["name"], t["args"]
        phone = digits(C[a["customer_id"]]["phone"]) if a.get("customer_id") in C else digits(a.get("phone"))
        if n in ("get_benefits", "get_subscription") and phone:
            out.append({"name": "get_member_info", "arguments": {"phone_number": phone}})  # 쿠폰·포인트·구독은 회원 조회에
        elif n == "lookup_customer" and phone:
            out.append({"name": "get_member_info", "arguments": {"phone_number": phone}})
        elif n == "list_orders" and phone:
            out.append({"name": "get_order_list", "arguments": {"phone_number": phone}})
        elif n == "get_product_info":
            out.append({"name": "get_product_info", "arguments": {"product_name": P[a["product_id"]]["name"]}})
        elif n == "check_stock":
            out.append({"name": "check_stock", "arguments": {"product_name": P[a["product_id"]]["name"]}})
        elif n == "track_shipment":
            out.append({"name": "track_delivery", "arguments": {"order_id": oid(a["order_id"])}})
        elif n == "get_order":
            out.append({"name": "get_order_info", "arguments": {"order_id": oid(a["order_id"])}})
        elif n == "get_refund_status" and a.get("order_id"):
            out.append({"name": "get_refund_status", "arguments": {"order_id": oid(a["order_id"])}})
        elif n == "cancel_order":
            out.append({"name": "cancel_order", "arguments": {"order_id": oid(a["order_id"])}})
        elif n == "request_return":
            out.append({"name": "request_return", "arguments": {"order_id": oid(a["order_id"])}})
        elif n == "transfer_to_human":
            out.append({"name": "transfer_to_agent", "arguments": {}})
        elif n == "schedule_callback":
            out.append({"name": "request_callback", "arguments": {}})
        elif n == "report_adverse_event":
            out.append({"name": "transfer_to_agent", "arguments": {}})  # 상담원 매뉴얼: 이상 반응은 상담원 연결
        elif n in ("get_policy", "send_link", "update_subscription") or (n == "lookup_customer" and "order_id" in a):
            # 정책은 매뉴얼로 답함 / 주문번호로 고객 조회는 track_delivery·get_order_info에 포함
            # 문자 링크·구독 변경 함수는 상담원 쪽에 없음 → 함수 없이 안내(응대 심사로만 평가)
            pass
        else:
            unmapped.append(t)
    seen, uniq = set(), []
    for t in out:
        k = json.dumps(t, sort_keys=True, ensure_ascii=False)
        if k not in seen: seen.add(k); uniq.append(t)
    return uniq, unmapped


def split_points(text):
    """기대 응답을 심사 항목으로 나눈다. ' / '뿐 아니라 '쉼표+공백'으로 묶인 내용도 따로 채점해
    하나만 빠져도 항목 전체가 0이 되지 않게 한다. 괄호 안 쉼표와 '6,000' 같은 숫자 쉼표는 나누지 않는다."""
    out = []
    for chunk in text.split(" / "):
        parts, buf, depth = [], "", 0
        for i, ch in enumerate(chunk):
            depth += ch == "(" ; depth -= ch == ")"
            if ch == "," and depth == 0 and chunk[i + 1:i + 2] == " ":
                parts.append(buf); buf = ""
            else:
                buf += ch
        parts.append(buf)
        out += [p.strip() for p in parts if p.strip()]
    return out


def speech_end(pcm):
    x = np.frombuffer(pcm, np.int16)
    n = len(x) // 320
    db = 20 * np.log10(np.sqrt((x[: n * 320].reshape(n, 320).astype(np.float32) ** 2).mean(1)) / 32768 + 1e-9)
    thr = min(max(np.percentile(db, 10) + 10, np.percentile(db, 90) - 30), np.percentile(db, 90) - 6)
    v = np.where(db >= thr)[0]
    return round(float((v[-1] + 1) * 0.02), 3) if len(v) else round(n * 0.02, 3)


def build_cases(recs, db, audio_dir, include_all):
    P, C = db["products"], db["customers"]
    cases, meta, notes = [], {}, {}
    os.makedirs(os.path.join(OUT, "audio"), exist_ok=True)
    for r in recs:
        wav = os.path.join(audio_dir, f"{r['id']}.wav")
        has_audio = os.path.exists(wav)
        if not has_audio and not include_all:
            continue
        exp, unmapped = map_tools(r, P, C)
        if unmapped: notes[r["id"]] = unmapped
        cal = r["session_start"]["caller"]
        prof = None
        if cal.get("customer_id"):
            prof = {"name": cal["name"], "phone": digits(cal["phone"]), "email": None}
        elif cal.get("phone") and digits(cal["phone"]).startswith("010"):
            prof = {"name": None, "phone": digits(cal["phone"]), "email": None}
        # 매뉴얼 2절: 발신번호가 있으면 주문번호를 묻지 않고 get_order_list로 주문을 찾는다.
        # 고객이 주문번호를 말하지 않았고 대화 이력도 없으면(=주문을 아직 특정하지 않음) 그 조회도 기대 함수로 둔다.
        order_tools = {"track_delivery", "get_order_info", "cancel_order", "request_return", "get_refund_status"}
        if (prof and prof.get("phone") and not r["session_start"]["history"]
                and any(t["name"] in order_tools for t in exp) and not re.search(r"\d{4}|이공이", r["utterance"])
                and not any(t["name"] == "get_order_list" for t in exp)):
            exp = [{"name": "get_order_list", "arguments": {"phone_number": prof["phone"]}}] + exp
        ts = r["session_start"]["timestamp"]
        tags = list(r["tags"]) + (["tools_unmapped"] if unmapped else [])
        cases.append({
            "case_id": r["id"], "now": ts[:10] + " " + ts[11:16], "history": r["session_start"]["history"],
            "customer_profile": prof, "customer_utterance": r["utterance"],
            "segments": [{"text": r["utterance"], "pause_after_ms": 0}],
            "voice": f"human:{r['recording']['speaker_profile']}",
            "env": {"channel": "wideband", "noise": "none"},
            "category": r["category"], "tags": tags, "expected_tools": exp,
            "expected_points": split_points(r["expected_response"]),
            "must_not": [s.strip() for s in re.split(r"[,/]", r["forbidden"]) if s.strip()],
            "expected_route": r["expected_route"], "pauses": r["pauses"]})
        if has_audio:
            shutil.copyfile(wav, os.path.join(OUT, "audio", f"{r['id']}.wav"))
            with wave.open(wav) as w:
                pcm = w.readframes(w.getnframes())
            meta[r["id"]] = {"speech_end": speech_end(pcm), "duration": round(len(pcm) / 32000, 3),
                             "voice": f"human:{r['recording']['speaker_profile']}", "env": {"channel": "wideband", "noise": "none"}}
    return cases, meta, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="녹음 없는 문항도 cases.jsonl에 포함")
    a = ap.parse_args()
    db = json.load(open(os.path.join(DS, "mock_db.json"), encoding="utf-8"))
    recs = [json.loads(l) for l in open(os.path.join(DS, "metadata.jsonl"), encoding="utf-8")]
    if os.path.isdir(os.path.join(OUT, "audio")):
        shutil.rmtree(os.path.join(OUT, "audio"))
    os.makedirs(OUT, exist_ok=True)
    sample = oid(next(iter(db["orders"])))
    w = lambda name, obj: json.dump(obj, open(os.path.join(OUT, name), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    w("tools.json", build_tools(db["products"], sample))
    w("mock_data.json", build_mock(db))
    open(os.path.join(OUT, "manual.md"), "w", encoding="utf-8").write(build_manual(db))
    cases, meta, notes = build_cases(recs, db, os.path.join(DS, "audio"), a.all)
    with open(os.path.join(OUT, "cases.jsonl"), "w", encoding="utf-8") as f:
        for c in cases: f.write(json.dumps(c, ensure_ascii=False) + "\n")
    w(os.path.join("audio", "meta.json"), meta)
    w("mapping_notes.json", notes)
    print(f"cases {len(cases)} (음성 {len(meta)}) · 기대 함수 변환 못 한 문항 {len(notes)} → {OUT}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
