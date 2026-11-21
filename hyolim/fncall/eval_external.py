"""외부 테스트셋(팀원이 만든 data/testset/cases.jsonl)으로 번호 인식만 평가한다.
  python3 fncall/eval_external.py <cases.jsonl>
브랜드·함수·주문번호 형식이 달라서, 전화번호 추출과 "숫자 읽기→숫자열" 변환만 본다."""
import json, sys, re, collections
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fncall.numbers_ko import find_number_spans, extract_ids, parse_now
from fncall.slots import resolve
dg = lambda v: re.sub(r"\D", "", str(v or ""))
cs = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
st = collections.Counter(); fails = []
for c in cs:
    utt = c["customer_utterance"]; caller = dg((c.get("customer_profile") or {}).get("phone"))
    spans_obj = find_number_spans(utt, min_digits=3)
    spans = [s.digits for s in spans_obj]
    alts = [s.alt for s in spans_obj if s.alt]
    joined = "".join(spans)
    now = parse_now(c.get("now"))
    ids = extract_ids(utt, now)
    for t in c["expected_tools"]:
        for k in ("phone_number", "order_id"):
            e = dg(t["arguments"].get(k))
            if not e:
                continue
            spoken = e in joined or any(e in s for s in spans) or bool(re.search(r"[공영일이삼사오육칠팔구]{2}", utt))
            if e == caller and not any(e[-4:] in s for s in spans):
                st[f"{k}: 발신번호(말하지 않음)"] += 1
                s = resolve({"history": c["history"], "now": c.get("now"), "caller": c.get("customer_profile")}, utt)
                st[f"{k}: 발신번호 → 슬롯이 고름"] += (s.phone == e) if k == "phone_number" else 0
                continue
            # 정답 번호의 앞 4자리도 발화에 없으면, 번호를 말로 부르지 않은 문항(주소 숫자 등)으로 본다
            if not any(e[:4] in x for x in spans + alts):
                st[f"{k}: 발화에 번호 없음(이력·주소 숫자 등)"] += 1
                continue
            st[f"{k}: 말로 부름"] += 1
            exact_span = any(x == e for x in spans + alts)   # '…공사이요' 처럼 대안 읽기로 맞은 것 포함
            st[f"{k}: B 숫자열 정확"] += exact_span
            if k == "phone_number":
                top = next((x.value for x in ids if x.kind == "phone"), None)
                anyc = [x.value for x in ids if x.kind == "phone"]
                st["phone_number: A 1순위 정확"] += (top == e)
                st["phone_number: A 후보 안에 있음(조회 검증으로 복구 가능)"] += (e in anyc)
                if top != e:
                    fails.append((c["case_id"], k, e, utt[:70], spans, anyc[:3]))
            elif not exact_span:
                fails.append((c["case_id"], k, e, utt[:70], spans, []))
for k in sorted(st):
    print(f"{st[k]:4d}  {k}")
print("\n틀린 것:")
for f in fails:
    print(" ", f)
