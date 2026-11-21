import csv, json, re, os, sys, datetime, collections
sys.path.insert(0, os.path.dirname(__file__))
from mockdb import *
from items import ITEMS

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
TOOLS_URL = "http://localhost:8000/tools"   # 측정기가 실제 값으로 채워 보냄

# 화자 5명 × 20문항. 같은 고객은 항상 같은 화자(목소리 일관성), 고객 성별과 화자 성별 일치
SPEAKERS = {"S1": "신소연", "S2": "손효림", "S3": "윤진", "S4": "최준호", "S5": "신호진"}
SPEAKER_GENDER = {"S1": "여성", "S2": "여성", "S3": "남성", "S4": "남성", "S5": "남성"}
CALLER_SPK = {"C004": "S1", "C001": "S1", "C014": "S1",
              "C003": "S2", "C011": "S2", "C013": "S2", "C009": "S2", "C015": "S2", "C005": "S2",
              "C010": "S3", "C007": "S3", "C012": "S3", "C008": "S3",
              "C002": "S4", "C006": "S4"}
# 미등록 발신자 문항: 분량 균형용 일부만 S1·S3·S4에, 나머지 20개는 S5
UNKNOWN_SPK = {51: "S1", 68: "S1", 65: "S3", 94: "S3", 54: "S4"}
SYL_PER_SEC = {"느림": 4.5, "보통": 6.0, "빠름": 7.5}
TOK = re.compile(r"\{(\d+(?:\.\d+)?)\}|\^")

def parse(script):
    clean, pauses, reveal = "", [], None
    pos = 0
    for m in TOK.finditer(script):
        clean += script[pos:m.start()]
        pos = m.end()
        c = re.sub(r"\s+", " ", clean)
        if m.group(1):
            pauses.append({"after": c.strip()[-12:], "char_offset": len(c.strip()), "sec": float(m.group(1))})
        else:
            reveal = len(c)
    clean += script[pos:]
    clean = re.sub(r"\s+", " ", clean).strip()
    # reveal: ^ 바로 뒤 단어 끝까지
    raw_no_pause = re.sub(r"\{\d+(?:\.\d+)?\}", "", script)
    raw_no_pause = re.sub(r"\s+", " ", raw_no_pause).strip()
    i = raw_no_pause.index("^")
    rest = raw_no_pause[i + 1:]
    word = re.match(r"\S+", rest).group(0)
    prefix = (raw_no_pause[:i] + word).strip()
    word_clean = word.rstrip(".,?!")
    return clean, pauses, word_clean, prefix, round(len(prefix) / len(clean), 2)

def display(script):
    s = re.sub(r"\{(\d+(?:\.\d+)?)\}", lambda m: f"[{m.group(1)}초 쉼]", script)
    return re.sub(r"\s+", " ", s.replace("^", "")).strip()

def caller_obj(cid):
    if cid in CUSTOMERS:
        c = CUSTOMERS[cid]
        return {"phone": c["phone"], "customer_id": cid, "name": c["name"], "grade": c["grade"]}
    return {"phone": UNKNOWN_CALLERS[cid]["phone"], "customer_id": None, "name": None, "grade": None}

base = datetime.datetime(2026, 11, 24, 9, 5, tzinfo=datetime.timezone(datetime.timedelta(hours=9)))
records = []
assert len(ITEMS) == 100, len(ITEMS)
for n, it in enumerate(ITEMS, 1):
    qid = f"q{n:03d}"
    clean, pauses, rword, prefix, ratio = parse(it["script"])
    ts = it["ts"] or (base + datetime.timedelta(minutes=5 * (n - 1))).isoformat()
    spk = CALLER_SPK.get(it["caller"]) or UNKNOWN_SPK.get(n, "S5")
    tags = list(it["tags"])
    if it["hist"] and "has_history" not in tags: tags.append("has_history")
    if any(p["sec"] >= 1.0 for p in pauses) and "long_pause" not in tags: tags.append("long_pause")
    if it["caller"] not in CUSTOMERS and not any(t.startswith("caller_") for t in tags):
        tags.append("caller_unmatched")
    syl = len(re.findall(r"[가-힣]", clean)) + len(re.findall(r"[A-Za-z]+", clean)) * 2
    est = round(syl / SYL_PER_SEC[it["speed"]] + sum(p["sec"] for p in pauses), 1)
    history = [{"role": "customer" if r == "c" else "agent", "text": t} for r, t in it["hist"]]
    rec = {
        "id": qid,
        "category": it["cat"],
        "intent": it["intent"],
        "expected_route": it["route"],
        "session_start": {
            "type": "session.start",
            "session_id": f"sess_{qid}",
            "timestamp": ts,
            "history": history,
            "caller": caller_obj(it["caller"]),
            "tools_url": TOOLS_URL,
        },
        "utterance": clean,
        "recording": {
            "file": f"audio/{qid}.wav",
            "script": display(it["script"]),
            "speaker": spk,
            "speaker_profile": SPEAKERS[spk],
            "emotion": it["emo"],
            "speed": it["speed"],
            "est_duration_sec": est,
        },
        "pauses": pauses,
        "intent_reveal": {"word": rword, "prefix": prefix, "ratio": ratio},
        "expected_tools": [{"name": t, "args": a} for t, a in it["tools"]],
        "expected_response": it["resp"],
        "forbidden": it["forbid"],
        "tags": tags,
    }
    records.append(rec)

with open(f"{OUT}/metadata.jsonl", "w", encoding="utf-8") as f:
    for r in records: f.write(json.dumps(r, ensure_ascii=False) + "\n")

os.makedirs(f"{OUT}/session_start", exist_ok=True)
for r in records:
    with open(f"{OUT}/session_start/{r['id']}.json", "w", encoding="utf-8") as f:
        json.dump(r["session_start"], f, ensure_ascii=False, indent=2)

cols = ["id", "category", "intent", "expected_route", "caller_id", "caller_phone", "timestamp", "history_turns",
        "utterance", "script", "speaker", "emotion", "speed", "est_duration_sec", "pauses", "max_pause_sec",
        "reveal_word", "reveal_ratio", "expected_tool_names", "expected_tools_json", "expected_response", "forbidden", "tags"]
with open(f"{OUT}/metadata.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(cols)
    for r in records:
        s = r["session_start"]
        w.writerow([r["id"], r["category"], r["intent"], r["expected_route"], s["caller"]["customer_id"] or "",
                    s["caller"]["phone"] or "", s["timestamp"], len(s["history"]), r["utterance"], r["recording"]["script"],
                    r["recording"]["speaker"], r["recording"]["emotion"], r["recording"]["speed"], r["recording"]["est_duration_sec"],
                    ";".join(f"{p['char_offset']}:{p['sec']}" for p in r["pauses"]),
                    max([p["sec"] for p in r["pauses"]], default=0),
                    r["intent_reveal"]["word"], r["intent_reveal"]["ratio"],
                    ",".join(t["name"] for t in r["expected_tools"]),
                    json.dumps(r["expected_tools"], ensure_ascii=False), r["expected_response"], r["forbidden"], ",".join(r["tags"])])

# 녹음 대본 (화자별 묶음)
with open(f"{OUT}/recording_script.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "file", "speaker", "profile", "caller_name", "emotion", "speed", "script", "est_sec", "context(직전 상담원 말)", "done"])
    for r in sorted(records, key=lambda x: (x["recording"]["speaker"], x["id"])):
        h = r["session_start"]["history"]
        w.writerow([r["id"], f"{r['id']}.wav", r["recording"]["speaker"], r["recording"]["speaker_profile"],
                    r["session_start"]["caller"]["name"] or "", r["recording"]["emotion"], r["recording"]["speed"],
                    r["recording"]["script"], r["recording"]["est_duration_sec"], h[-1]["text"] if h else "", ""])

SPK_DESC = SPEAKERS
by_spk = collections.defaultdict(list)
for r in records: by_spk[r["recording"]["speaker"]].append(r)
L = ["# 녹음 대본 — 건강기능식품 CS 고객 발화 100문항", "",
     "가상 브랜드 **온담헬스** 고객센터에 전화한 고객의 한 마디씩입니다. 한 파일 = 한 문항(고객 한 턴).", "",
     "## 녹음 규칙", "",
     "- **형식**: WAV, 16kHz · 16-bit · mono (측정기 입력과 동일). 다른 형식으로 녹음했다면 `tools/check_audio.py --fix`로 변환",
     "- **앞뒤 무음**: 시작 전 0.3초 이내, 끝난 뒤 0.5초 이내로 자르기. 말 끝난 뒤 무음은 측정기가 붙여 줌",
     "- **[n초 쉼]**: 말 중간에 실제로 그만큼 멈추기(±0.2초). 엔드포인팅 함정 테스트용이라 가장 중요한 지시. 1초 이상 쉼은 굵게 표시",
     "- **문맥**: `상담원 (읽지 않음)` 인용 블록은 직전에 상담원이 한 말. **녹음하지 않고** 그 말에 대답한다고 생각하며 **고객 (녹음)** 줄만 읽기. 상담원 말을 먼저 소리 내어 읽어 두면 억양이 자연스러워짐",
     "- 대본의 '어…', '그…' 같은 말버릇은 그대로 살리고, 숫자는 적힌 읽기(예: '이공이육')대로 발음",
     "- **화자 5명 × 20문항**. 한 화자는 한 사람이 끝까지 녹음. 같은 고객의 문항은 같은 화자에 묶여 있어 목소리가 일관됨",
     "- 화자 성별은 발신자 이름·대화 이력과 맞춰 둠. 나이는 신경 쓰지 않아도 됨(모두 20대여도 OK)",
     "- **(고령 고객)** 표시 문항: 목소리를 늙게 흉내 내지 말고 **속도·쉼·말버릇**만 살리기. 과장된 연기는 STT·엔드포인팅 테스트를 오히려 왜곡함",
     "- 녹음 후 `python tools/check_audio.py audio/ metadata.jsonl`로 형식·길이·중간 쉼을 검사", ""]
L.append("## 화자별 분량\n")
L.append("| 화자 | 녹음자 | 성별 | 문항 수 | 예상 총 길이 | 맡은 고객 |\n|---|---|---|---|---|---|")
for s in sorted(by_spk):
    names = sorted({r["session_start"]["caller"]["name"] for r in by_spk[s] if r["session_start"]["caller"]["name"]})
    nu = sum(1 for r in by_spk[s] if not r["session_start"]["caller"]["name"])
    L.append(f"| {s} | {SPK_DESC[s]} | {SPEAKER_GENDER[s]} | {len(by_spk[s])} | {round(sum(r['recording']['est_duration_sec'] for r in by_spk[s]))}초 | {' + '.join(x for x in [', '.join(names), f'미등록 {nu}' if nu else ''] if x)} |")
L.append("")
for s in sorted(by_spk):
    L.append(f"## {s} — {SPK_DESC[s]} ({SPEAKER_GENDER[s]}, {len(by_spk[s])}문항)\n")
    for r in by_spk[s]:
        rec = r["recording"]
        line = re.sub(r"\[(\d+(?:\.\d+)?)초 쉼\]", lambda m: f"**[{m.group(1)}초 쉼]**" if float(m.group(1)) >= 1.0 else f"[{m.group(1)}초 쉼]", rec["script"])
        who = r["session_start"]["caller"]["name"] or "미등록 고객"
        if "elderly" in r["tags"]: who += " **(고령 고객)**"
        L.append(f"### {r['id']} · {who}\n")
        L.append(f"*{rec['emotion']} · 속도 {rec['speed']} · 약 {rec['est_duration_sec']}초*\n")
        h = r["session_start"]["history"]
        if h: L.append(f"> **상담원 (읽지 않음)** — {h[-1]['text']}\n")
        L.append(f"**고객 (녹음)** — {line}\n")
open(f"{OUT}/recording_script.md", "w", encoding="utf-8").write("\n".join(L))

# 목업 DB / 함수 목록
json.dump({"brand": BRAND, "products": PRODUCTS, "policies": POLICIES, "customers": CUSTOMERS, "orders": ORDERS,
           "subscriptions": SUBSCRIPTIONS, "unknown_callers": UNKNOWN_CALLERS},
          open(f"{OUT}/mock_db.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(TOOLS, open(f"{OUT}/tools_assumed.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# 통계
st = {"total": len(records),
      "by_category": collections.Counter(r["category"] for r in records),
      "by_route": collections.Counter(r["expected_route"] for r in records),
      "by_speaker": collections.Counter(r["recording"]["speaker"] for r in records),
      "with_history": sum(1 for r in records if r["session_start"]["history"]),
      "with_long_pause": sum(1 for r in records if "long_pause" in r["tags"]),
      "caller_unmatched_or_anonymous": sum(1 for r in records if r["session_start"]["caller"]["customer_id"] is None),
      "no_tool_call": sum(1 for r in records if not r["expected_tools"]),
      "total_est_audio_sec": round(sum(r["recording"]["est_duration_sec"] for r in records)),
      "tags": collections.Counter(t for r in records for t in r["tags"])}
json.dump(st, open(f"{OUT}/stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(st, ensure_ascii=False, indent=1))
