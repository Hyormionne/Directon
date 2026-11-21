"""로컬 측정기 흉내 — 공식 측정기 아님. 규격 문서(11/21)가 나오면 메시지 필드·함수 호출 형식을 맞춰 고칠 것.

동작: 문항마다 WebSocket 연결 → session.start 전송 → 고객 음성을 20ms(640B)씩 실시간 속도로 전송
      → 말이 끝난 뒤에도 무음 조각을 계속 전송 → turn.end 수신 또는 25초 초과 시 종료.
      tools_url은 이 스크립트가 띄운 목업 함수 서버(mock_db.json 기반)를 가리킴.

사용: python tools/simulate.py ws://localhost:9000/ws --audio audio/ --meta metadata.jsonl [--only q001,q002] [--out results/]
필요: pip install websockets numpy
"""
import argparse, asyncio, json, os, threading, time, wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import websockets

CHUNK = 640
SILENCE = b"\x00" * CHUNK
CALLS = {}   # session_id -> [ {name, args, t} ]
DB = {}

def tool_result(name, a):
    o, s, c, p = DB["orders"], DB["subscriptions"], DB["customers"], DB["products"]
    if name == "lookup_customer":
        if a.get("order_id") in o: return {"order": o[a["order_id"]]}
        for cid, v in c.items():
            if v["phone"] == a.get("phone"): return {"customer_id": cid, **v}
        return {"found": False}
    if name == "list_orders":
        return {"orders": {k: v for k, v in o.items() if v.get("customer") == a.get("customer_id")}}
    if name in ("get_order", "track_shipment", "get_refund_status"):
        oid = a.get("order_id")
        if not oid and a.get("return_id"):
            oid = next((k for k, v in o.items() if v.get("return", {}).get("id") == a["return_id"]), None)
        return o.get(oid, {"found": False})
    if name == "get_subscription":
        return {"subscriptions": {k: v for k, v in s.items() if v["customer"] == a.get("customer_id")}}
    if name in ("get_product_info", "check_stock"): return p.get(a.get("product_id"), {"found": False})
    if name == "get_benefits":
        v = c.get(a.get("customer_id"), {}); return {k: v.get(k) for k in ("grade", "points", "coupons")}
    if name == "get_policy": return {"policy": DB["policies"].get(a.get("topic"), "해당 정책 없음")}
    return {"ok": True, "echo": a}   # 쓰기 계열은 성공으로 응답

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        sid = self.path.rstrip("/").split("/")[-1]
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        name = body.get("name") or body.get("tool") or body.get("function")
        args = body.get("arguments") or body.get("args") or body.get("parameters") or {}
        if isinstance(args, str): args = json.loads(args or "{}")
        CALLS.setdefault(sid, []).append({"name": name, "args": args, "t": time.time()})
        out = json.dumps({"result": tool_result(name, args)}, ensure_ascii=False).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(out)
    def log_message(self, *a): pass

def speech_end_offset(pcm_bytes):
    x = np.frombuffer(pcm_bytes, dtype=np.int16)
    n = len(x) // 320
    db = 20 * np.log10(np.sqrt((x[: n * 320].reshape(n, 320).astype(np.float32) ** 2).mean(1)) / 32768 + 1e-9)
    thr = min(max(np.percentile(db, 10) + 10, np.percentile(db, 90) - 30), np.percentile(db, 90) - 6)
    v = np.where(db >= thr)[0]
    return float((v[-1] + 1) * 0.02) if len(v) else n * 0.02

async def run_item(url, rec, audio_dir, out_dir, tools_base):
    with wave.open(os.path.join(audio_dir, f"{rec['id']}.wav")) as w:
        pcm = w.readframes(w.getnframes())
    ss = dict(rec["session_start"]); ss["tools_url"] = f"{tools_base}/{ss['session_id']}"
    speech_end_rel = speech_end_offset(pcm)
    res = {"id": rec["id"], "barge_in": False, "first_audio_after_end": None, "turn_end_after_end": None,
           "timeout": False, "resp_audio_sec": 0.0}
    resp = bytearray()
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps(ss, ensure_ascii=False))
        t0 = time.monotonic(); t_end = t0 + speech_end_rel
        done = asyncio.Event()

        async def sender():
            i = 0
            while not done.is_set():
                chunk = pcm[i * CHUNK:(i + 1) * CHUNK] if i * CHUNK < len(pcm) else SILENCE
                if len(chunk) < CHUNK: chunk = chunk + b"\x00" * (CHUNK - len(chunk))
                await ws.send(chunk); i += 1
                await asyncio.sleep(max(0, t0 + i * 0.02 - time.monotonic()))
                if time.monotonic() > t_end + 25 and not done.is_set():
                    res["timeout"] = True; done.set()

        async def receiver():
            async for m in ws:
                now = time.monotonic()
                if isinstance(m, bytes):
                    if res["first_audio_after_end"] is None:
                        res["first_audio_after_end"] = round(float(now - t_end), 3)
                        if now < t_end: res["barge_in"] = True
                    resp.extend(m)
                else:
                    msg = json.loads(m)
                    if msg.get("type") == "turn.end":
                        res["turn_end_after_end"] = round(float(now - t_end), 3); done.set(); return

        tasks = [asyncio.create_task(sender()), asyncio.create_task(receiver())]
        await done.wait()
        for t in tasks: t.cancel()
    res["resp_audio_sec"] = round(len(resp) / 32000, 2)
    with wave.open(os.path.join(out_dir, f"{rec['id']}_resp.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(bytes(resp))
    called = CALLS.get(ss["session_id"], [])
    want = {t["name"] for t in rec["expected_tools"]}
    got = {c["name"] for c in called}
    res.update({"tools_called": called, "tools_expected": sorted(want),
                "tool_recall": round(len(want & got) / len(want), 2) if want else None,
                "tool_extra": sorted(got - want), "expected_route": rec["expected_route"]})
    return res

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url"); ap.add_argument("--audio", default="audio"); ap.add_argument("--meta", default="metadata.jsonl")
    ap.add_argument("--db", default="mock_db.json"); ap.add_argument("--out", default="results")
    ap.add_argument("--only"); ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    DB.update(json.load(open(a.db, encoding="utf-8")))
    os.makedirs(a.out, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    recs = [json.loads(l) for l in open(a.meta, encoding="utf-8")]
    if a.only: recs = [r for r in recs if r["id"] in a.only.split(",")]
    results = []
    for r in recs:
        if not os.path.exists(os.path.join(a.audio, f"{r['id']}.wav")): continue
        try:
            x = await run_item(a.url, r, a.audio, a.out, f"http://127.0.0.1:{a.port}/tools")
        except Exception as e:
            x = {"id": r["id"], "error": repr(e)}
        results.append(x)
        print(r["id"], {k: x.get(k) for k in ("barge_in", "first_audio_after_end", "turn_end_after_end", "timeout", "tool_recall", "error")})
    with open(os.path.join(a.out, "results.jsonl"), "w", encoding="utf-8") as f:
        for x in results: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    ok = [x for x in results if "error" not in x]
    lat = sorted(x["first_audio_after_end"] for x in ok if x["first_audio_after_end"] is not None and not x["barge_in"])
    print(f"\n문항 {len(results)} · 끼어듦 {sum(x['barge_in'] for x in ok)} · 타임아웃 {sum(x['timeout'] for x in ok)} · 오류 {len(results) - len(ok)}")
    if lat: print(f"첫 응답 지연(말 끝 기준) 중앙값 {lat[len(lat)//2]:.2f}s · p90 {lat[round(0.9 * (len(lat) - 1))]:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
