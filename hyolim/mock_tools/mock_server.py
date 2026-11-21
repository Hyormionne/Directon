"""모의 함수 서버 뼈대 — 쇼핑몰 시스템 대신 함수(주문 조회·반품 접수 등)에 응답하는 서버를 여러분이 완성한다.

HTTP 주소 처리는 되어 있고, 응답을 고르는 lookup() 은 비어 있다.
호출 기록과 응답 지연은 측정기(run_eval.py)의 함수 중계가 맡으므로 여기서 할 필요가 없다.
이 파일을 고쳐 써도 되고, 다른 언어로 새로 만들어도 된다. 지켜야 할 규칙은 SPEC.md 의 「모의 함수 서버」에 있다.

실행:  python3 mock_server.py --port 8100 [--db ../data/mock_db.json]

  GET  /tools                 함수 목록 (OpenAI tools 형식, data/brand/tools.json)
  POST /tools/{함수명}         body = 인수 JSON. 헤더 X-Session-Id 에 session.start 로 받은 session_id 가 온다.
  GET  /calls?session_id=...  (참고용) 이 서버가 받은 호출 기록. 채점에는 쓰지 않는다.
"""
import argparse, json, os, re, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))


class Store:
    def __init__(self, db_path, tools_path):
        self.db = json.load(open(db_path, encoding="utf-8")) if os.path.exists(db_path) else {}   # {함수명: [{"arguments", "result"}]}
        self.tools = json.load(open(tools_path, encoding="utf-8"))
        self.calls = []
        self.lock = threading.Lock()

    def lookup(self, name, args):
        """TODO: 여러분이 구현할 부분. (응답 JSON, 기록과 맞았는지 bool) 을 돌려준다.

        self.db[name] 에 실제 상담에서 나온 (인수, 결과) 쌍이 들어 있다. 들어온 인수에 맞는 결과를 골라 돌려준다.
        - 조회 함수는 전화번호·주문번호가 맞는 기록을 돌려준다. 하이픈·공백 같은 표기 차이는 무시한다.
        - 맞는 기록이 없으면 {"status": "not_found", ...}, 필수 값이 빠졌으면 {"status": "invalid", ...}
        - 접수 함수(취소·반품·교환·전화 요청)는 기록이 없어도 필수 값이 다 있으면 {"status": "ok", ...}
        - 상품 검색은 검색어가 정확히 같지 않아도 가장 가까운 기록을 고른다.
        """
        dg = lambda v: re.sub(r"\D", "", str(v or ""))
        tool = next((t for t in self.tools if t["name"] == name), {})
        required = (tool.get("parameters") or {}).get("required") or []
        if any(args.get(k) in (None, "", []) for k in required):
            return {"status": "invalid", "message": "필수 값이 빠졌습니다."}, False
        recs = self.db.get(name, [])
        id_keys = ("order_id", "order_number", "buyer_cellphone", "customer_phone_number", "phone_number")

        if name in ("get_order_info", "get_return_info"):
            want = {dg(args.get(k)) for k in id_keys if dg(args.get(k))}
            if not want:
                return {"status": "invalid", "message": "주문번호 또는 전화번호가 필요합니다."}, False
            for r in recs:
                have = {dg(r["arguments"].get(k)) for k in id_keys if dg(r["arguments"].get(k))}
                # 조회 결과 안의 주문번호로도 맞춰 본다 (전화번호로 기록된 주문을 주문번호로 찾는 경우)
                data = r["result"].get("data") if isinstance(r["result"].get("data"), dict) else {}
                for row in (data.get("orders") or data.get("반품정보") or []):
                    have.add(dg(row.get("주문번호")))
                if want & have:
                    return r["result"], True
            return {"status": "not_found", "message": "조회 결과가 없습니다."}, False

        if name == "search_products":
            q = re.sub(r"\s", "", str(args.get("query") or ""))
            def sim(a, b):
                a, b = set(a), set(b)
                return len(a & b) / max(1, len(a | b))
            if recs:
                best = max(recs, key=lambda r: sim(q, re.sub(r"\s", "", str(r["arguments"].get("query") or ""))))
                return best["result"], True
            return {"status": "not_found", "message": "검색 결과가 없습니다."}, False

        # 접수 함수: 기록에 같은 번호가 있으면 그 결과(already_requested 등), 없으면 ok
        want = {dg(args.get(k)) for k in id_keys if dg(args.get(k))}
        if name == "return_order_item":
            want |= {dg(it.get("order_item_code")) for it in args.get("return_items") or []}
        for r in recs:
            have = {dg(r["arguments"].get(k)) for k in id_keys if dg(r["arguments"].get(k))}
            have |= {dg(it.get("order_item_code")) for it in r["arguments"].get("return_items") or []}
            if want & have:
                return r["result"], True
        return {"status": "ok", "message": "접수되었습니다."}, False


def make_handler(store):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/tools":
                return self._send(200, [{"type": "function", "function": t} for t in store.tools])
            if u.path == "/calls":
                cid = parse_qs(u.query).get("session_id", [None])[0]
                with store.lock:
                    return self._send(200, [c for c in store.calls if cid is None or c["session_id"] == cid])
            self._send(404, {"error": "not found"})

        def do_POST(self):
            m = re.fullmatch(r"/tools/(\w+)", urlparse(self.path).path)
            if not m or m.group(1) not in {t["name"] for t in store.tools}:
                return self._send(404, {"error": "unknown function"})
            name = m.group(1)
            t0 = time.time()
            try:
                args = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except json.JSONDecodeError:
                return self._send(400, {"error": "body must be JSON"})
            result, matched = store.lookup(name, args)
            # 참고용 호출 기록 (디버깅용)
            rec = {"session_id": self.headers.get("X-Session-Id"), "name": name, "arguments": args, "matched": matched,
                   "t_request": t0, "t_response": time.time()}
            with store.lock:
                store.calls.append(rec)
            self._send(200, result)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--db", default=os.path.join(HERE, "..", "data", "mock_db.json"))
    ap.add_argument("--tools", default=os.path.join(HERE, "..", "data", "brand", "tools.json"))
    a = ap.parse_args()
    store = Store(a.db, a.tools)
    print(f"mock tools on :{a.port}  functions={len(store.tools)}  records={sum(map(len, store.db.values()))}",
          file=sys.stderr)
    ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(store)).serve_forever()


if __name__ == "__main__":
    main()
