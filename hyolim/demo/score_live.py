"""직접 통화한 기록을 채점기와 같은 기준으로 평가한다 (OPENAI_API_KEY 필요).

  python3 demo/score_live.py                       # 가장 최근 통화 기록
  python3 demo/score_live.py demo/sessions/live_1121_1300.jsonl

- 응대: 채점기(eval/score.py)의 판정 프롬프트·모델 그대로. 단, 직접 통화엔 '실제 상담 답'이 없어서
  시나리오·브랜드 정보·함수 결과로만 판단하게 한다. 손님 말은 화면에 적은 '실제로 한 말'이 있으면 그것을 쓴다.
- 함수: 정답 호출표가 없으므로 같은 채점 AI 가 SCORING.md 의 함수 규칙대로 판정한다
  (맞는 기능을, 손님이 실제로 말한 번호로 불렀나 / 필요 없는 접수 함수를 부르지 않았나).
- 속도: 채점기와 같은 점수표 (첫 소리 1.5초 이하 만점, 조회 뒤 답 3.5초 이하 만점, 끼어들면 0점).
"""
from __future__ import annotations

import asyncio
import glob
import importlib.util
import json
import os
import statistics
import sys

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
from server import config as C  # noqa: E402  (.env 읽기)

_spec = importlib.util.spec_from_file_location("score", os.path.join(ROOT, "eval", "score.py"))
score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score)

TOOL_PROMPT = """너는 쇼핑몰 전화 상담 AI 의 '함수 호출'을 채점한다. 아래 규칙만 본다.
- 손님의 기록을 조회하거나 바꾸는 함수만 본다: 조회(get_order_info, get_return_info), 접수(cancel_order, return_order, return_order_item, exchange_order, request_callback). search_products 는 보지 않는다.
- 이번 턴에 꼭 해야 했던 호출을 모두 했고, 번호(전화번호·주문번호)가 손님이 '실제로 말한' 번호(또는 대화 이력·발신자 정보에 있는 맞는 번호)와 같으면 맞음. 하이픈·공백 차이는 무시.
- 조회 함수를 더 부른 것은 감점하지 않는다. 같은 조회 결과가 이력에 이미 있으면 다시 조회하지 않아도 된다.
- 손님이 요청하지 않았거나 아직 필요한 확인(동의·사유 등)을 하지 않았는데 접수 함수를 부르면 틀림.
- 필요한 정보(번호)가 아직 없어서 묻는 것이 맞는 상황이면, 함수를 부르지 않은 것이 맞음.
JSON 으로만 답한다: {{"ok": true|false, "reason": "한 문장"}}

[현재 시각] {now}
[발신자 정보] {caller}
[대화 이력]
{history}
[손님이 실제로 한 말] {truth}
[AI 가 받아쓴 말] {heard}
[AI 의 함수 호출과 결과]
{calls}
"""


async def judge_tool(c, rec, truth):
    calls = "\n".join(f"- {x['name']} {json.dumps(x['arguments'], ensure_ascii=False)} → {x['status']}" for x in rec["tool_calls"]) or "(호출 없음)"
    prompt = TOOL_PROMPT.format(now=rec["now"], caller=json.dumps(rec["caller"], ensure_ascii=False),
                                history=score.fmt_history(rec["history"]) or "(없음)", truth=truth, heard=rec["heard"], calls=calls)
    body = {"model": score.JUDGE_MODEL, "messages": [{"role": "user", "content": prompt}], "reasoning_effort": "low",
            "response_format": {"type": "json_schema", "json_schema": {"name": "t", "strict": True, "schema": {
                "type": "object", "additionalProperties": False, "required": ["ok", "reason"],
                "properties": {"ok": {"type": "boolean"}, "reason": {"type": "string"}}}}}}
    r = await c.post("https://api.openai.com/v1/chat/completions", json=body, headers={"Authorization": "Bearer " + C.OPENAI_API_KEY})
    r.raise_for_status()
    return json.loads(r.json()["choices"][0]["message"]["content"])


async def main():
    path = sys.argv[1] if len(sys.argv) > 1 else max(glob.glob(os.path.join(HERE, "sessions", "live_*.jsonl")), key=os.path.getmtime)
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    brand = os.path.join(ROOT, "data", "brand")
    info = json.load(open(os.path.join(brand, "information.json"), encoding="utf-8"))
    ctx = {"scenarios": open(os.path.join(brand, "scenarios.md"), encoding="utf-8").read(),
           "info": "\n".join(f"{i['tag']}\n{i['content']}" for i in info), "rubric": {}}
    out = []
    async with httpx.AsyncClient(timeout=180) as c:
        for rec in rows:
            truth = rec.get("truth") or rec["heard"]
            case = {"case_id": rec["session_id"], "now": rec["now"], "customer_profile": rec["caller"], "history": rec["history"],
                    "customer_utterance": truth,
                    "reference": {"tool_calls": [], "tool_results": [],
                                  "answer": "(직접 통화라 실제 상담 답이 없다. 시나리오·브랜드 정보·함수 결과만으로 판단하라. 함수 호출 여부는 응대 판정에서 보지 마라.)"}}
            v = await score.judge(c, case, rec["said"], [{**x, "matched": x["status"] == "ok"} for x in rec["tool_calls"]], ctx)
            t = await judge_tool(c, rec, truth)
            inf = float("inf")
            first = rec["first_sound"] if rec["first_sound"] is not None and not rec["spoke_before_end"] else inf
            sp = score.speed_points(first)
            if rec["tool_calls"]:
                after = rec["after_tool_sound"] if rec["after_tool_sound"] is not None else inf
                sp = (sp + score.speed_points(after, score.AFTER_TOOL_TABLE)) / 2
            out.append({**rec, "verdict": v["verdict"], "reason": v["reason"], "tool_ok": t["ok"], "tool_reason": t["reason"],
                        "speed_points": sp, "truth_given": bool(rec.get("truth"))})
    resp = statistics.mean(score.PTS[o["verdict"]] for o in out) * 100
    tool = statistics.mean(1.0 if o["tool_ok"] else 0.0 for o in out) * 100
    speed = statistics.mean(o["speed_points"] for o in out)
    total = resp * score.W_RESP + tool * score.W_TOOL + speed * score.W_SPEED
    lines = [f"# 직접 통화 채점 — {os.path.basename(path)}", "",
             f"턴 {len(out)}개 · 응대 {resp:.1f} × 0.5 + 함수 {tool:.1f} × 0.2 + 속도 {speed:.1f} × 0.3 = **총점 {total:.1f}** (행사 반영 {total / 5:.1f}/20)", ""]
    for o in out:
        f = lambda x: "—" if x is None else f"{x:.2f}초"
        lines += [f"## 턴 {o['turn']}",
                  f"- 실제로 한 말: {o.get('truth') or '(적지 않음 — 받아쓴 말로 채점)'}",
                  f"- 받아쓴 말: {o['heard']}",
                  f"- AI: {o['said']}",
                  f"- 함수: {', '.join(x['name'] + json.dumps({k: v for k, v in x['arguments'].items() if k not in ('reason', 'refund_account')}, ensure_ascii=False) + '→' + str(x['status']) for x in o['tool_calls']) or '없음'}",
                  f"- 응대 **{o['verdict']}** — {o['reason']}",
                  f"- 함수 **{'맞음' if o['tool_ok'] else '틀림'}** — {o['tool_reason']}",
                  f"- 속도 {o['speed_points']:.0f}점 (첫 소리 {f(o['first_sound'])}, 조회 뒤 답 {f(o['after_tool_sound'])}{', 말 도중 끼어듦' if o['spoke_before_end'] else ''})", ""]
    report = path.replace(".jsonl", "_report.md")
    open(report, "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines[:3]))
    for o in out:
        print(f"턴 {o['turn']}: 응대 {o['verdict']} / 함수 {'O' if o['tool_ok'] else 'X'} / 속도 {o['speed_points']:.0f} | {o['heard'][:40]!r}")
    print(f"\n자세한 보고서: {os.path.relpath(report, ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
