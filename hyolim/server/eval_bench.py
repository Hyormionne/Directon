"""음성합성 없이 '실제 음성 → Soniox 받아쓰기 → 함수 처리 → GPT 답' 을 돌리고 응대·함수·받아쓰기 정확도를 채점한다.

키는 베이즈 하나로 된다(.env 의 BAZE_API_KEY, BAZE_WS_URL). OpenAI 키가 없으면 GPT·채점도 베이즈 게이트웨이를 거친다.

  python3 -m server.eval_bench                     # 연습 80 + 하루온 200 둘 다
  python3 -m server.eval_bench --data practice     # 연습 80만
  python3 -m server.eval_bench --data haruon       # 하루온 200만
  python3 -m server.eval_bench --limit 3           # 각 데이터 앞 3문항만 (시험용)

데이터별로 할 수 있는 것
- 연습 80 (data/): 정답이 있어 대회 채점기와 같은 방식으로 응대·함수 점수, 원문이 있어 받아쓰기 정확도까지 낸다.
- 하루온 200 (haruon_bench/): 정답·원문이 없다. 응대는 참고 답 없이 시나리오·브랜드 정보로 판정, 함수는 채점 AI 판정,
  받아쓰기 정확도는 낼 수 없다. 팀 비교용 공식 점수가 아니라 참고용이다. haruon_bench 폴더의 파일은 읽기만 한다.

크레딧 절약: 받아쓴 글은 eval/heard_soniox_<데이터>.json 에 저장하고, 다시 돌리면 Soniox 로 보내지 않는다(--redo-stt 로 다시).
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import statistics
import time
import wave

import httpx

from . import config as C
from . import llm
from .answer import guarded, plan_answer
from .app import FILLER_TEXT, pick_filler
from fncall.eval_stt_digits import hit
from fncall.eval_stt_soniox import transcribe
from fncall.numbers_ko import parse_now
from fncall.pipeline import run_turn

ROOT = C.ROOT
HARUON = os.environ.get("HARUON_DIR", os.path.join(os.path.dirname(ROOT), "haruon_bench"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


score = _load("score", os.path.join(ROOT, "eval", "score.py"))
mock_server = _load("mock_server", os.path.join(ROOT, "mock_tools", "mock_server.py"))
score_live = _load("score_live", os.path.join(ROOT, "demo", "score_live.py"))
dg = lambda v: re.sub(r"\D", "", str(v or ""))


def datasets(which: str):
    out = []
    if which in ("practice", "both"):
        out.append({"name": "practice", "cases": os.path.join(ROOT, "data", "practice_cases.jsonl"),
                    "audio": os.path.join(ROOT, "data", "audio", "practice"), "db": os.path.join(ROOT, "data", "mock_db.json"),
                    "has_ref": True})
    if which in ("haruon", "both"):
        out.append({"name": "haruon", "cases": os.path.join(HARUON, "cases_public.jsonl"),
                    "audio": os.path.join(HARUON, "audio"), "db": os.path.join(HARUON, "mock_db.json"), "has_ref": False})
    return out


# ───────────────────────── 받아쓰기 (Soniox) ─────────────────────────
async def stt_all(ds, cases, redo):
    path = os.path.join(ROOT, "eval", f"heard_soniox_{ds['name']}.json")
    heard = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    sec, sem = 0.0, asyncio.Semaphore(3)

    async def one(c):
        nonlocal sec
        cid = c["case_id"]
        if heard.get(cid) and not redo:
            return
        wav = os.path.join(ds["audio"], cid + ".wav")
        async with sem:
            w = wave.open(wav)
            sec += w.getnframes() / w.getframerate()
            try:
                t = await transcribe(wav)
            except Exception as e:
                print(f"  {cid}: 받아쓰기 실패 {str(e)[:100]}", flush=True)
                return
        if t:
            heard[cid] = t
            json.dump(heard, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    await asyncio.gather(*[one(c) for c in cases])
    print(f"  [{ds['name']}] Soniox 로 보낸 음성 {sec / 60:.2f}분 · 저장 {os.path.relpath(path, ROOT)}", flush=True)
    return heard


# ───────────────────────── 채점 AI 호출 (게이트웨이 대응) ─────────────────────────
async def ask_json(http, prompt, schema_name, schema):
    body = {"model": C.JUDGE_MODEL, "messages": [{"role": "user", "content": prompt}], "reasoning_effort": "low",
            "response_format": {"type": "json_schema", "json_schema": {"name": schema_name, "strict": True, "schema": schema}}}
    for attempt in range(4):
        r = await http.post(C.JUDGE_URL, json=body, headers={"Authorization": f"Bearer {C.OPENAI_API_KEY}"}, timeout=180)
        if r.status_code == 200:
            txt = r.json()["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", txt, re.S)
            return json.loads(m.group(0) if m else txt)
        if r.status_code == 400 and attempt == 0:          # 게이트웨이가 일부 옵션을 모르면 빼고 다시
            body.pop("reasoning_effort", None)
            continue
        if r.status_code == 400 and attempt == 1:
            body.pop("response_format", None)
            body["messages"][0]["content"] += "\n\nJSON 으로만 답하라."
            continue
        await asyncio.sleep(2)
    raise RuntimeError(f"채점 실패 {r.status_code} {r.text[:150]}")


VERDICT = {"type": "object", "additionalProperties": False, "required": ["verdict", "reason"],
           "properties": {"verdict": {"type": "string", "enum": list(score.PTS)}, "reason": {"type": "string"}}}
TOOLV = {"type": "object", "additionalProperties": False, "required": ["ok", "reason"],
         "properties": {"ok": {"type": "boolean"}, "reason": {"type": "string"}}}


def judge_prompt(case, said, calls, ctx, has_ref):
    if has_ref:
        ref = case["reference"]
        ref_txt = "".join(f"({t['name']} 함수 호출, 인수: {json.dumps(t['arguments'], ensure_ascii=False)})\n" for t in ref["tool_calls"])
        ref_txt += "".join(f"({t['name']} 함수 결과: {json.dumps(t.get('result'), ensure_ascii=False)[:1500]})\n" for t in ref.get("tool_results", []))
        ref_txt += ref.get("answer") or ""
        utter = case["customer_utterance"]
    else:
        ref_txt = "(참고 답 없음: 시나리오·브랜드 정보·함수 결과만으로 판단하라. 함수 호출 여부는 응대 판정에서 보지 마라.)"
        utter = case.get("_heard", "")
    ag = "".join(f"({t['name']} 함수 호출, 인수: {json.dumps(t['arguments'], ensure_ascii=False)}, 조회 성공: {t['matched']})\n" for t in calls)
    ag += "말한 내용: " + (said or "(아무 말도 하지 않음)")
    return score.JUDGE_PROMPT.format(scenarios=ctx["scenarios"], info=ctx["info"], now=case["now"],
                                     caller=json.dumps(case.get("customer_profile"), ensure_ascii=False),
                                     history=score.fmt_history(case["history"]), utterance=utter, reference=ref_txt, rubric="", agent=ag)


# ───────────────────────── 받아쓰기 정확도 ─────────────────────────
def cer(ref: str, hyp: str) -> float:
    a = re.sub(r"[\s\W_]", "", ref or "")
    b = re.sub(r"[\s\W_]", "", hyp or "")
    if not a:
        return 0.0
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[-1] / len(a)


def spoken_ids(case):
    ud = dg(case["customer_utterance"])
    want = set()
    for t in case["reference"]["tool_calls"]:
        for k in ("buyer_cellphone", "customer_phone_number", "phone_number", "order_id", "order_number"):
            v = dg(t["arguments"].get(k))
            if v and len(v) >= 10 and v in ud:
                want.add(v)
    return want


# ───────────────────────── 한 데이터 전체 ─────────────────────────
async def run_dataset(ds, a, ctx):
    meta = json.load(open(os.path.join(ds["audio"], "meta.json"), encoding="utf-8"))
    cases = [json.loads(l) for l in open(ds["cases"], encoding="utf-8")]
    cases = [c for c in cases if c["case_id"] in meta][: a.limit or None]
    print(f"\n== {ds['name']}: {len(cases)}문항", flush=True)
    heard = await stt_all(ds, cases, a.redo_stt)
    store = mock_server.Store(ds["db"], os.path.join(ROOT, "data", "brand", "tools.json"))
    sem = asyncio.Semaphore(4)
    rows = []

    async with httpx.AsyncClient(timeout=120) as http:
        async def one(c):
            cid = c["case_id"]
            utt = heard.get(cid)
            if not utt:
                return None
            c["_heard"] = utt
            log = []

            async def call(n, args):                        # 측정기 중계처럼 보낸 호출을 모두 기록
                res = store.lookup(n, args)[0]
                log.append({"name": n, "arguments": args, "matched": res.get("status") == "ok"})
                return res
            session = {"history": c["history"], "now": c.get("now"), "caller": c.get("customer_profile"), "case_id": cid}
            async with sem:
                out = await run_turn(session, utt, call)
                plan = plan_answer(session, utt, out)
                fk = pick_filler(session, utt)
                said = ["네, 고객님."] + ([FILLER_TEXT[fk]] if fk and fk != "ack" else [])
                try:
                    if plan.mode in ("safe", "template"):
                        said.append(plan.text)
                    else:
                        async for s in guarded(plan, llm.stream_answer(http, plan.messages), ack_played=True):
                            said.append(s)
                except Exception as e:
                    said.append(f"(답 생성 실패: {str(e)[:80]})")
                said_txt = " ".join(said)
                row = {"case_id": cid, "heard": utt, "said": said_txt, "how": plan.mode, "risk": plan.risk.level,
                       "tool_calls": log}
                if a.no_judge:
                    rows.append(row)
                    return
                try:
                    v = await ask_json(http, judge_prompt(c, said_txt, log, ctx, ds["has_ref"]), "v", VERDICT)
                except Exception as e:
                    v = {"verdict": "틀림", "reason": f"채점 실패: {str(e)[:80]}"}
                row.update(verdict=v["verdict"], reason=v["reason"])
                if ds["has_ref"]:
                    ok, why = score.tool_match(c["reference"]["tool_calls"], log, c["history"])
                    row.update(tool_ok=ok, tool_why=why, utterance=c["customer_utterance"],
                               cer=cer(c["customer_utterance"], utt))
                    want = spoken_ids(c)
                    if want:
                        row["number_heard_ok"] = hit(utt, want, parse_now(c.get("now")))
                else:
                    rec = {"now": c["now"], "caller": c.get("customer_profile"), "history": c["history"], "heard": utt,
                           "tool_calls": [{**x, "status": "ok" if x["matched"] else "not_found"} for x in log]}
                    calls_txt = "\n".join(f"- {x['name']} {json.dumps(x['arguments'], ensure_ascii=False)} → {x['status']}"
                                          for x in rec["tool_calls"]) or "(호출 없음)"
                    prompt = score_live.TOOL_PROMPT.format(now=rec["now"], caller=json.dumps(rec["caller"], ensure_ascii=False),
                                                           history=score.fmt_history(rec["history"]) or "(없음)",
                                                           truth="(원문 없음 — 받아쓴 말을 손님 말로 본다)", heard=utt, calls=calls_txt)
                    try:
                        t = await ask_json(http, prompt, "t", TOOLV)
                    except Exception as e:
                        t = {"ok": False, "reason": f"채점 실패: {str(e)[:80]}"}
                    row.update(tool_ok=t["ok"], tool_why=t["reason"])
                rows.append(row)
                mark = {"맞음": "O", "괜찮음": "△", "틀림": "X"}.get(row.get("verdict"), "?")
                print(f"  {mark} {'함수O' if row.get('tool_ok') else '함수X'} {cid} {utt[:40]!r}", flush=True)
        await asyncio.gather(*[one(c) for c in cases])

    out_dir = os.path.join(ROOT, "eval", "runs", time.strftime(f"bench_{ds['name']}_%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "scored.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summ = {"data": ds["name"], "cases": len(rows), "transcribed": sum(1 for c in cases if heard.get(c["case_id"]))}
    if rows and not a.no_judge:
        summ["response"] = round(statistics.mean(score.PTS[r["verdict"]] for r in rows) * 100, 1)
        summ["function"] = round(statistics.mean(1.0 if r["tool_ok"] else 0.0 for r in rows) * 100, 1)
        summ["verdicts"] = {k: sum(r["verdict"] == k for r in rows) for k in score.PTS}
    if ds["has_ref"] and rows:
        summ["stt_cer"] = round(statistics.mean(r.get("cer", 0) for r in rows) * 100, 1)
        nums = [r["number_heard_ok"] for r in rows if "number_heard_ok" in r]
        summ["number_accuracy"] = f"{sum(nums)}/{len(nums)}"
    json.dump(summ, open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    summ["out"] = os.path.relpath(out_dir, ROOT)
    return summ


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="both", choices=["both", "practice", "haruon"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-judge", action="store_true", help="채점 없이 받아쓰기·답만")
    ap.add_argument("--redo-stt", action="store_true", help="저장된 받아쓴 글이 있어도 Soniox 로 다시")
    a = ap.parse_args()
    if not C.SONIOX_API_KEY:
        raise SystemExit("키가 없습니다. .env 에 BAZE_API_KEY=... 와 BAZE_WS_URL=... 를 넣어 주세요.")
    brand = os.path.join(ROOT, "data", "brand")
    info = json.load(open(os.path.join(brand, "information.json"), encoding="utf-8"))
    ctx = {"scenarios": open(os.path.join(brand, "scenarios.md"), encoding="utf-8").read(),
           "info": "\n".join(f"{i['tag']}\n{i['content']}" for i in info)}
    print(f"받아쓰기 {C.STT_MODEL} · 답 {C.LLM_MODEL} · 채점 {C.JUDGE_MODEL} · 음성합성 없음"
          + (" · 베이즈 게이트웨이" if C.USE_BAZE else ""))
    results = [await run_dataset(ds, a, ctx) for ds in datasets(a.data)]
    print("\n================ 결과 ================")
    for s in results:
        line = f"[{s['data']}] {s['cases']}문항"
        if "response" in s:
            line += f" · 응대 {s['response']} · 함수 {s['function']} {s['verdicts']}"
        if "stt_cer" in s:
            line += f" · 받아쓰기 글자 오류율 {s['stt_cer']}% · 번호 맞힘 {s['number_accuracy']}"
        if s["data"] == "haruon":
            line += " (정답 없는 데이터라 참고용)"
        print(line + f"\n   저장: {s['out']}")


if __name__ == "__main__":
    asyncio.run(main())
