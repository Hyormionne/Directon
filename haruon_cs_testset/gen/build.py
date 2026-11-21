"""하루온 음성 CS 녹음 테스트셋 생성.
사용: python gen/build.py [출력 폴더=haruon_cs_testset] [--pdf]
- 공식 search_products 기록은 반출 금지 데이터라 저장소에 넣지 않는다. 공식 폴더가 있으면 빌드 때 mock_db.json에 합친다.
"""
import collections, csv, datetime, html, json, os, re, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from world import CUSTOMERS, ORDERS, RETURNS, SPECIAL, P  # noqa: E402
from items import ITEMS  # noqa: E402

ROOT = os.path.dirname(HERE)
PROJECT = os.path.dirname(ROOT)
OFFICIAL = os.path.join(PROJECT, "directon-2026-voice-cs", "data")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT = args[0] if args else ROOT
MAKE_PDF = "--pdf" in sys.argv
os.makedirs(OUT, exist_ok=True)

N_TOTAL, PER_SPK = 200, 40
SPEAKERS = {"S1": ("신소연", "F"), "S2": ("손효림", "F"), "S3": ("윤진", "M"), "S4": ("최준호", "M"), "S5": ("신호진", "M")}
GENDER_KO = {"F": "여성", "M": "남성"}
SYL_PER_SEC = {"느림": 4.5, "보통": 6.0, "빠름": 7.5}
DOW = "월화수목금토일"

# ───────── mock DB ─────────
def order_rows(o):
    rows = []
    for i, (pk, opt, qty) in enumerate(o["items"], 1):
        name = P[pk]
        paid = o["status"] not in ("입금전", "입금전취소")
        at = datetime.datetime.strptime(o["at"], "%Y-%m-%d %H:%M:%S")
        rows.append({
            "주문번호": o["oid"], "상세주문번호": f"{o['oid']}-{i:02d}", "주문일자": o["at"], "제품명": name,
            "옵션": f"옵션 선택={name} {opt}", "수량": qty, "결제수단": o["pay"], "결제여부": "T" if paid else "F",
            "결제일자": (o["paid_at"] or (at + datetime.timedelta(minutes=3)).strftime("%Y-%m-%d %H:%M:%S")) if paid else None,
            "배송상태": o["status"], "송장번호": o["invoice"], "출고예정일자": o["ship_due"], "배송완료일자": o["done"],
        })
    return rows


def ok_orders(rows):
    return {"status": "ok", "data": {"orders": rows}, "message": "주문 조회 완료"}


by_phone = collections.defaultdict(list)
for o in sorted(ORDERS, key=lambda x: x["at"], reverse=True):
    by_phone[o["phone"]] += order_rows(o)
DB = collections.defaultdict(list)
for k, (name, ph, *_rest) in CUSTOMERS.items():
    DB["get_order_info"].append({"arguments": {"order_id": None, "buyer_cellphone": ph}, "result": ok_orders(by_phone.get(ph, []))})
for o in ORDERS:
    DB["get_order_info"].append({"arguments": {"order_id": o["oid"], "buyer_cellphone": None}, "result": ok_orders(order_rows(o))})
for ph, rs in RETURNS.items():
    DB["get_return_info"].append({"arguments": {"customer_phone_number": ph}, "result": {"status": "ok", "data": {"반품정보": [
        {"상품명": P[r["prod"]], "송장번호": r["invoice"], "주문번호": r["oid"], "현재상태": r["status"],
         "반품요청일자": r["req"], "휴대폰결제여부": r["phone_pay"]} for r in rs]}, "message": "반품 조회 완료"}})
for fn, recs in SPECIAL.items():
    for a, r in recs:
        DB[fn].append({"arguments": a, "result": r})
NOT_FOUND = {"status": "not_found", "message": "조회된 기록이 없습니다."}


def digits(v):
    return re.sub(r"\D", "", str(v or ""))


def lookup(name, a):
    """mock_api.server와 같은 규칙의 간이 조회(history의 tool_result 생성용)."""
    if name == "get_order_info":
        oid, ph = digits(a.get("order_id")), digits(a.get("buyer_cellphone"))
        for key in ("order_id", "buyer_cellphone"):
            want = oid if key == "order_id" else ph
            if not want:
                continue
            for e in DB[name]:
                if digits(e["arguments"].get(key)) == want:
                    return e["result"]
        return NOT_FOUND
    if name == "get_return_info":
        for e in DB[name]:
            if digits(e["arguments"]["customer_phone_number"]) == digits(a.get("customer_phone_number")):
                return e["result"]
        return {"status": "not_found", "message": "반품 상품이 없습니다."}
    for e in DB.get(name, []):
        if all(digits(a.get(k)) == digits(v) for k, v in e["arguments"].items() if k in ("order_number", "order_id")):
            return e["result"]
    return {"status": "ok"}


# ───────── 대본 파싱 ─────────
NUM = re.compile(r"\{\{([^|{}]+)\|([^}]*(?:\{[\d.]+\}[^}]*)*)\}\}")
PAUSE = re.compile(r"\{(\d+(?:\.\d+)?)\}")


def ws(s):
    return re.sub(r"\s+", " ", s).strip()


def parse(script):
    utter = ws(PAUSE.sub("", NUM.sub(lambda m: m.group(1), script)))
    read = NUM.sub(lambda m: m.group(2), script)
    pauses, clean, pos = [], "", 0
    for m in PAUSE.finditer(read):
        clean += read[pos:m.start()]
        pos = m.end()
        c = ws(clean)
        pauses.append({"after": c[-12:], "char_offset": len(c), "sec": float(m.group(1))})
    read_text = ws(clean + read[pos:])
    display = ws(PAUSE.sub(lambda m: f"[{m.group(1)}초 쉼]", read))
    return utter, read_text, display, pauses


def fmt_now(s):
    return s


def default_now(i):
    # 11/25(수)~11/27(금) 10:00~16:50, 문항마다 9분씩
    slot = i * 9
    day, minute = divmod(slot, 410)
    d = datetime.datetime(2026, 11, 25 + day % 3, 10, 0) + datetime.timedelta(minutes=minute)
    return f"{d:%Y-%m-%d} ({DOW[d.weekday()]}) {d:%H:%M}"


# ───────── 화자 배정 ─────────
assert len(ITEMS) == N_TOTAL, len(ITEMS)
load = {s: 0 for s in SPEAKERS}
assign = {}
cust_count = collections.Counter(it["who"] for it in ITEMS if it["who"] != "U")
cust_spk = {}
for cid, n in sorted(cust_count.items(), key=lambda x: -x[1]):
    g = CUSTOMERS[cid][2]
    s = min((s for s in SPEAKERS if SPEAKERS[s][1] == g), key=lambda s: (load[s], s))
    cust_spk[cid] = s
    load[s] += n
unknown = [i for i, it in enumerate(ITEMS) if it["who"] == "U"]
# 성별을 맞추되 40개를 넘기지 않게. g가 "F!"/"M!"(내용상 성별 고정)인 문항을 먼저 배정
unknown.sort(key=lambda i: not str(ITEMS[i]["g"]).endswith("!"))
for i in unknown:
    g = (ITEMS[i]["g"] or "").rstrip("!") or None
    pool = [s for s in SPEAKERS if load[s] < PER_SPK and (g is None or SPEAKERS[s][1] == g)]
    if not pool:
        pool = [s for s in SPEAKERS if load[s] < PER_SPK]
        ITEMS[i]["gender_override"] = True
    s = min(pool, key=lambda s: (load[s], s))
    assign[i] = s
    load[s] += 1
assert all(v == PER_SPK for v in load.values()), load

# ───────── 레코드 ─────────
records = []
for i, it in enumerate(ITEMS):
    hid = f"h{i + 1:03d}"
    utter, read_text, display, pauses = parse(it["script"])
    who = it["who"]
    spk = cust_spk[who] if who != "U" else assign[i]
    caller = None
    tags = list(it["tags"])
    if who != "U":
        name, ph, g, old = CUSTOMERS[who]
        caller = {"name": name, "phone": ph, "email": None}
        if old and "elderly" not in tags:
            tags.append("elderly")
    else:
        tags.append("caller_null")
    hist = []
    for h in it["hist"]:
        if h[0] == "c":
            hist.append({"role": "customer", "text": h[1]})
        elif h[0] == "a":
            hist.append({"role": "agent", "text": h[1]})
        else:
            hist.append({"role": "tool_call", "name": h[1], "arguments": h[2]})
            hist.append({"role": "tool_result", "name": h[1], "result": lookup(h[1], h[2])})
    if hist and "has_history" not in tags:
        tags.append("has_history")
    if any(p["sec"] >= 1.0 for p in pauses) and "long_pause" not in tags:
        tags.append("long_pause")
    if not it["tools"]:
        tags.append("no_tool")
    if it.get("gender_override"):
        tags.append("voice_gender_free")
    syl = len(re.findall(r"[가-힣]", read_text)) + len(re.findall(r"[A-Za-z]+", read_text)) * 2
    est = round(syl / SYL_PER_SEC[it["speed"]] + sum(p["sec"] for p in pauses), 1)
    now = it["now"] or default_now(i)
    ss = {"type": "session.start", "session_id": f"{hid}-rec", "case_id": hid, "now": now, "history": hist,
          "caller": caller, "tools_url": "http://localhost:8200",
          "audio": {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1}}
    exp_tools = [{"name": n, "arguments": {k: v for k, v in a.items() if v is not None}} for n, a in it["tools"]]
    records.append({
        "id": hid, "scenario": it["cat"], "intent": it["intent"], "session_start": ss, "utterance": utter,
        "recording": {"file": f"audio/{hid}.wav", "script": display, "read_text": read_text, "speaker": spk,
                      "speaker_profile": SPEAKERS[spk][0], "emotion": it["emo"], "speed": it["speed"], "est_duration_sec": est},
        "pauses": pauses,
        "reference": {"tool_calls": exp_tools,
                      # search_products 결과는 공식 데이터라 적지 않는다(mock_db.json에만, 빌드 때 복사)
                      "tool_results": [{"status": "ok", "note": "공식 search_products 기록 중 가장 가까운 검색어의 결과"}
                                       if t["name"] == "search_products" else lookup(t["name"], t["arguments"]) for t in exp_tools],
                      "answer_points": it["resp"], "forbidden": it["forbid"]},
        "tags": tags,
    })

# 번호가 공식 데이터와 겹치지 않는지
if os.path.isdir(OFFICIAL):
    blob = open(os.path.join(OFFICIAL, "mock_db.json"), encoding="utf-8").read() + \
        open(os.path.join(OFFICIAL, "practice_cases.jsonl"), encoding="utf-8").read()
    blob_d = re.sub(r"[\s\-]", "", blob)
    mine = [c[1] for c in CUSTOMERS.values()] + [o["oid"].replace("-", "") for o in ORDERS]
    clash = [m for m in mine if m in blob_d]
    assert not clash, f"공식 데이터와 번호 겹침: {clash}"

# ───────── 파일 쓰기 ─────────
def jdump(obj, path, indent=2):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


with open(f"{OUT}/metadata.jsonl", "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
# 공식 practice_cases.jsonl과 같은 모양(측정기 --cases용). customer_utterance = STT 정답
with open(f"{OUT}/cases.jsonl", "w", encoding="utf-8") as f:
    for r in records:
        ss = r["session_start"]
        f.write(json.dumps({"case_id": r["id"], "room_id": r["id"], "now": ss["now"], "history": ss["history"],
                            "customer_utterance": r["utterance"],
                            "reference": {"tool_calls": r["reference"]["tool_calls"], "tool_results": r["reference"]["tool_results"],
                                          "answer": r["reference"]["answer_points"]},
                            "has_image": False, "customer_profile": ss["caller"]}, ensure_ascii=False) + "\n")
os.makedirs(f"{OUT}/session_start", exist_ok=True)
for r in records:
    jdump(r["session_start"], f"{OUT}/session_start/{r['id']}.json")

db_out = {k: list(v) for k, v in DB.items()}
for fn in ("search_products", "cancel_order", "return_order", "return_order_item", "exchange_order", "request_callback"):
    db_out.setdefault(fn, [])
if os.path.isdir(OFFICIAL):  # 상품 정보는 공식 기록을 그대로 쓴다(저장소에는 안 올림)
    off = json.load(open(os.path.join(OFFICIAL, "mock_db.json"), encoding="utf-8"))
    db_out["search_products"] = off.get("search_products", [])
jdump(db_out, f"{OUT}/mock_db.json", indent=1)

with open(f"{OUT}/metadata.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "scenario", "intent", "now", "caller_name", "caller_phone", "history_turns", "utterance", "script",
                "speaker", "emotion", "speed", "est_sec", "max_pause_sec", "expected_tools", "answer_points", "forbidden", "tags"])
    for r in records:
        ss, rc = r["session_start"], r["recording"]
        w.writerow([r["id"], r["scenario"], r["intent"], ss["now"], (ss["caller"] or {}).get("name", ""),
                    (ss["caller"] or {}).get("phone", ""), len(ss["history"]), r["utterance"], rc["script"], rc["speaker"],
                    rc["emotion"], rc["speed"], rc["est_duration_sec"], max([p["sec"] for p in r["pauses"]], default=0),
                    json.dumps(r["reference"]["tool_calls"], ensure_ascii=False), r["reference"]["answer_points"],
                    r["reference"]["forbidden"], ",".join(r["tags"])])


def last_agent(r):
    for h in reversed(r["session_start"]["history"]):
        if h["role"] == "agent":
            return h["text"]
    return ""


by_spk = collections.defaultdict(list)
for r in records:
    by_spk[r["recording"]["speaker"]].append(r)

with open(f"{OUT}/recording_script.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "file", "speaker", "profile", "caller_name", "emotion", "speed", "script", "est_sec", "context(직전 상담원 말)", "done"])
    for s in sorted(by_spk):
        for r in by_spk[s]:
            rc = r["recording"]
            w.writerow([r["id"], f"{r['id']}.wav", s, rc["speaker_profile"], (r["session_start"]["caller"] or {}).get("name", ""),
                        rc["emotion"], rc["speed"], rc["script"], rc["est_duration_sec"], last_agent(r), ""])

RULES = [
    ("형식", "WAV, 16kHz · 16-bit · mono (측정기 입력과 동일). 다른 형식으로 녹음했다면 <code>tools/check_audio.py --fix</code>로 변환"),
    ("앞뒤 무음", "시작 전 0.3초 이내, 끝난 뒤 0.5초 이내로 자르기. 말 끝난 뒤 무음은 측정기가 붙여 줌"),
    ("[n초 쉼]", "말 중간에 실제로 그만큼 멈추기(±0.2초). 말 끝 판단(엔드포인팅) 함정 테스트용이라 가장 중요한 지시. 1초 이상 쉼은 굵게 표시"),
    ("문맥", "<span class='ctx-tag'>상담원 (읽지 않음)</span> 인용 블록은 직전에 상담원이 한 말. <b>녹음하지 않고</b> 그 말에 대답한다고 생각하며 <span class='cust'>고객 (녹음)</span> 줄만 읽기"),
    ("숫자", "전화번호·주문번호·계좌번호는 <b>적힌 읽기 그대로</b> 발음('공일공'과 '영일영'을 섞어 둠). 띄어 쓴 곳에서 살짝 끊어 읽고, 붙여 쓴 곳은 이어서 빠르게"),
    ("말버릇", "'어…', '그…', '아 아니다' 같은 말버릇과 말 바꾸기는 그대로 살리기"),
    ("화자 5명 × 40문항", "한 화자는 한 사람이 끝까지 녹음. 같은 고객의 문항은 같은 화자에 묶여 있어 목소리가 일관됨. 대본 순서(문항 번호 순)대로 이어 읽으면 <code>tools/split_recording.py</code>로 자동 분할 가능"),
    ("(고령 고객)", "목소리를 늙게 흉내 내지 말고 <b>속도·쉼·말버릇</b>만 살리기. 과장된 연기는 STT·엔드포인팅 테스트를 오히려 왜곡함"),
    ("검사", "녹음 후 <code>python tools/check_audio.py audio/ metadata.jsonl</code>로 형식·길이·중간 쉼을 검사"),
]


def customers_of(s):
    names = []
    for r in by_spk[s]:
        c = r["session_start"]["caller"]
        if c and c["name"] not in names:
            names.append(c["name"])
    nu = sum(1 for r in by_spk[s] if not r["session_start"]["caller"])
    return ", ".join(names) + (f" + 미등록 {nu}" if nu else "")


def bold_long(s, md):
    def f(m):
        t = f"[{m.group(1)}초 쉼]"
        if float(m.group(1)) >= 1.0:
            return f"**{t}**" if md else f"<b class='lp'>{t}</b>"
        return t if md else f"<span class='sp'>{t}</span>"
    return re.sub(r"\[(\d+(?:\.\d+)?)초 쉼\]", f, s)


cat_count = collections.Counter(r["scenario"] for r in records)
title = f"녹음 대본 — 하루온 음성 CS 고객 발화 {len(records)}문항"
intro = "가상 건강기능식품 브랜드 <b>하루온</b> 고객센터에 전화한 고객의 한 마디씩입니다. 한 파일 = 한 문항(고객 한 턴). 공식 상담 시나리오 10개와 브랜드 정보를 바탕으로 만들었습니다."
total_sec = round(sum(r["recording"]["est_duration_sec"] for r in records))

# Markdown
L = [f"# {title}", "", re.sub(r"</?b>", "**", intro), "", "## 녹음 규칙", ""]
for k, v in RULES:
    v = re.sub(r"<code>(.*?)</code>", r"`\1`", v)
    v = re.sub(r"<span class='ctx-tag'>(.*?)</span>", r"`\1`", v)
    v = re.sub(r"<span class='cust'>(.*?)</span>", r"**\1**", v)
    v = re.sub(r"</?b>", "**", v)
    L.append(f"- **{k}**: {v}")
L += ["", "## 화자별 분량", "", "| 화자 | 녹음자 | 성별 | 문항 수 | 예상 총 길이 | 맡은 고객 |", "|---|---|---|---|---|---|"]
for s in sorted(by_spk):
    L.append(f"| {s} | {SPEAKERS[s][0]} | {GENDER_KO[SPEAKERS[s][1]]} | {len(by_spk[s])} | "
             f"{round(sum(r['recording']['est_duration_sec'] for r in by_spk[s]))}초 | {customers_of(s)} |")
L += ["", f"총 예상 녹음 길이 약 {total_sec // 60}분 {total_sec % 60}초 (쉼 포함, 재녹음 제외).", "",
      "## 시나리오 분포", "", " · ".join(f"{k} {v}" for k, v in cat_count.items()), ""]
for s in sorted(by_spk):
    L.append(f"## {s} — {SPEAKERS[s][0]} ({GENDER_KO[SPEAKERS[s][1]]}, {len(by_spk[s])}문항)\n")
    for r in by_spk[s]:
        rc = r["recording"]
        who = (r["session_start"]["caller"] or {}).get("name") or "미등록 고객"
        if "elderly" in r["tags"]:
            who += " **(고령 고객)**"
        L.append(f"### {r['id']} · {who}\n")
        L.append(f"*{r['scenario']} · {rc['emotion']} · 속도 {rc['speed']} · 약 {rc['est_duration_sec']}초*\n")
        ctx = last_agent(r)
        if ctx:
            L.append(f"> **상담원 (읽지 않음)** — {ctx}\n")
        L.append(f"**고객 (녹음)** — {bold_long(rc['script'], True)}\n")
open(f"{OUT}/recording_script.md", "w", encoding="utf-8").write("\n".join(L))

# HTML (PDF용) — onedam 대본 PDF와 같은 모양
E = html.escape
H = [f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>{E(title)}</title><style>
@page {{ size: A4; margin: 16mm 15mm; }}
body {{ font-family: 'Malgun Gothic','Noto Sans KR',sans-serif; color:#1d1d1f; font-size:10.5pt; line-height:1.55; }}
h1 {{ font-size:19pt; margin:0 0 6px; }} h2 {{ font-size:14pt; border-bottom:2px solid #222; padding-bottom:5px; margin:22px 0 10px; }}
h3 {{ font-size:11pt; margin:0 0 1px; }} code {{ font-family:Consolas,monospace; font-size:9pt; background:#f1f1f1; padding:0 3px; border-radius:3px; }}
ul {{ padding-left:20px; }} li {{ margin:3px 0; }}
table {{ border-collapse:collapse; width:100%; font-size:9.5pt; }} th,td {{ border:1px solid #d4d4d4; padding:5px 7px; text-align:left; }} th {{ background:#f3f3f3; }} td:nth-child(-n+5), th {{ white-space:nowrap; }}
.item {{ border-bottom:1px solid #ddd; padding:9px 0 10px; break-inside:avoid; }}
.meta {{ color:#666; font-style:italic; font-size:9pt; margin-bottom:4px; }}
.ctx {{ background:#f4f4f5; border-left:3px solid #aaa; color:#555; font-size:9.5pt; padding:5px 9px; margin:4px 0 5px; }}
.ctx-tag {{ font-size:9pt; color:#555; background:#eee; padding:0 3px; }}
.cust {{ color:#0b5cad; font-weight:700; }} .lp {{ color:#b42318; }} .sp {{ color:#555; }}
.old {{ color:#b45309; font-weight:700; font-size:9.5pt; }} .num {{ letter-spacing:.2px; }}
.spk {{ break-before:page; }} .dist {{ font-size:9.5pt; color:#333; }}
</style></head><body>""", f"<h1>{E(title)}</h1><p>{intro}</p>", "<h2>녹음 규칙</h2><ul>"]
H += [f"<li><b>{E(k)}</b>: {v}</li>" for k, v in RULES]
H.append("</ul><h2>화자별 분량</h2><table><tr><th>화자</th><th>녹음자</th><th>성별</th><th>문항 수</th><th>예상 총 길이</th><th>맡은 고객</th></tr>")
for s in sorted(by_spk):
    H.append(f"<tr><td>{s}</td><td>{SPEAKERS[s][0]}</td><td>{GENDER_KO[SPEAKERS[s][1]]}</td><td>{len(by_spk[s])}</td>"
             f"<td>{round(sum(r['recording']['est_duration_sec'] for r in by_spk[s]))}초</td><td>{E(customers_of(s))}</td></tr>")
H.append(f"</table><p class='dist'>총 예상 녹음 길이 약 {total_sec // 60}분 {total_sec % 60}초 (쉼 포함). "
         f"시나리오 분포: {' · '.join(f'{E(k)} {v}' for k, v in cat_count.items())}</p>")
for s in sorted(by_spk):
    H.append(f"<h2 class='spk'>{s} — {SPEAKERS[s][0]} ({GENDER_KO[SPEAKERS[s][1]]}, {len(by_spk[s])}문항)</h2>")
    for r in by_spk[s]:
        rc = r["recording"]
        who = E((r["session_start"]["caller"] or {}).get("name") or "미등록 고객")
        old = " <span class='old'>(고령 고객)</span>" if "elderly" in r["tags"] else ""
        ctx = last_agent(r)
        H.append(f"<div class='item'><h3>{r['id']} · {who}{old}</h3>"
                 f"<div class='meta'>{E(r['scenario'])} · {rc['emotion']} · 속도 {rc['speed']} · 약 {rc['est_duration_sec']}초</div>"
                 + (f"<div class='ctx'><b>상담원 (읽지 않음)</b> — {E(ctx)}</div>" if ctx else "")
                 + f"<div><span class='cust'>고객 (녹음)</span> — {bold_long(E(rc['script']), False)}</div></div>")
H.append("</body></html>")
html_path = os.path.join(OUT, "recording_script.html")
open(html_path, "w", encoding="utf-8").write("\n".join(H))

if MAKE_PDF:
    browsers = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Google\Chrome\Application\chrome.exe", shutil.which("chromium") or "", shutil.which("google-chrome") or ""]
    exe = next((b for b in browsers if b and os.path.exists(b)), None)
    if exe:
        pdf = os.path.abspath(os.path.join(OUT, "recording_script.pdf"))
        subprocess.run([exe, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                        "file:///" + os.path.abspath(html_path).replace("\\", "/")], check=True, capture_output=True, timeout=120)
        print("PDF:", pdf)
    else:
        print("PDF 건너뜀: Edge/Chrome 없음")

st = {"total": len(records),
      "by_scenario": cat_count,
      "by_speaker": collections.Counter(r["recording"]["speaker"] for r in records),
      "with_history": sum(1 for r in records if r["session_start"]["history"]),
      "caller_null": sum(1 for r in records if r["session_start"]["caller"] is None),
      "with_tool_call": sum(1 for r in records if r["reference"]["tool_calls"]),
      "by_tool": collections.Counter(t["name"] for r in records for t in r["reference"]["tool_calls"]),
      "with_long_pause": sum(1 for r in records if "long_pause" in r["tags"]),
      "total_est_audio_sec": total_sec,
      "tags": collections.Counter(t for r in records for t in r["tags"])}
jdump(st, f"{OUT}/stats.json")
print(json.dumps(st, ensure_ascii=False, indent=1))
