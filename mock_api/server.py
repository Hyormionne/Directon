"""공식 규격 모의 함수 서버 (AG-T30). 규칙: `DIRECT_ON 2026 Starter Kit/SPEC.md` 「모의 함수 서버」.

    python3 -m mock_api.server --port 8100 --db data/mock_db.json [--tools data/brand/tools.json]

- GET  /tools              함수 목록 [{"type":"function","function":{name,description,parameters}}]
- POST /tools/{함수명}      body = 인수 JSON → 결과 JSON. 함수 정의에 없는 함수는 404.
- GET  /calls?session_id=  참고용 호출 기록(채점에는 측정기 중계의 기록을 쓴다).

데이터는 `{함수명: [{"arguments": {...}, "result": {...}}]}` 모양이다. 응답을 고르는 규칙:
- 조회(get_order_info, get_return_info): 주문번호·전화번호를 **숫자만** 남겨 비교한다(하이픈·공백 무시).
  주문번호가 맞는 기록이 먼저이고, 그다음 전화번호가 맞는 기록이다. 없으면 {"status": "not_found"}.
  get_order_info는 필수 인수가 없지만 주문번호와 전화번호가 둘 다 비면 {"status": "invalid"}.
- 접수(cancel_order, return_order, return_order_item, exchange_order, request_callback): 필수 값이 빠지면
  {"status": "invalid"}. 다 있으면 번호가 맞는 기록의 결과(already_requested 등), 기록이 없으면 {"status": "ok"}.
- 검색(search_products): 검색어가 가장 가까운 기록의 결과.

데이터·함수 정의 경로: `--db`(=`--data`)/`--tools` → 환경변수 DIRECTON_MOCK_DATA/DIRECTON_MOCK_TOOLS →
저장소의 `data/`(제출 묶음) → `DIRECT_ON 2026 Starter Kit/data/`(개발). 상태를 바꾸지 않는다(같은 입력 → 같은 출력).
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
KIT_DIRS = (ROOT / "data", ROOT / "DIRECT_ON 2026 Starter Kit" / "data")
DATA_ENV = "DIRECTON_MOCK_DATA"
TOOLS_ENV = "DIRECTON_MOCK_TOOLS"

ORDER_KEYS = ("order_id", "order_number")
PHONE_KEYS = ("buyer_cellphone", "customer_phone_number", "phone_number")
LOOKUP_FUNCS = frozenset({"get_order_info", "get_return_info"})
SEARCH_FUNCS = frozenset({"search_products"})


def digits(v) -> str:
    return re.sub(r"\D", "", str(v)) if v is not None else ""


def _compact(s) -> str:
    return re.sub(r"\s+", "", str(s or "")).lower()


def _empty(v) -> bool:
    return v is None or (isinstance(v, (str, list, dict)) and len(v) == 0) or (isinstance(v, str) and not v.strip())


def normalize_tools(raw) -> list[dict]:
    """함수 정의 목록 → [{name, description, parameters}]. OpenAI 래핑·{"tools": [...]}도 푼다."""
    if isinstance(raw, dict) and isinstance(raw.get("tools"), list):
        raw = raw["tools"]
    if not isinstance(raw, list):
        raise ValueError("tools 파일은 함수 정의 목록이어야 합니다")
    out = []
    for t in raw:
        if isinstance(t, dict) and isinstance(t.get("function"), dict):
            t = t["function"]
        if not isinstance(t, dict) or not isinstance(t.get("name"), str):
            raise ValueError(f"함수 정의 형식 오류: {t!r}"[:200])
        t = dict(t)
        t.setdefault("description", "")
        t.setdefault("parameters", {"type": "object", "properties": {}})
        out.append(t)
    return out


def _ids(args: dict) -> tuple[set[str], set[str]]:
    """인수에서 (주문번호 숫자 집합, 전화번호 숫자 집합). return_items의 상세주문번호도 주문번호로 본다."""
    orders = {digits(args.get(k)) for k in ORDER_KEYS} - {""}
    phones = {digits(args.get(k)) for k in PHONE_KEYS} - {""}
    items = args.get("return_items")
    if isinstance(items, list):
        orders |= {digits(i.get("order_item_code")) for i in items if isinstance(i, dict)} - {""}
    return orders, phones


class Store:
    def __init__(self, db: dict, tools: list[dict]):
        self.db = db if isinstance(db, dict) else {}
        self.tools = normalize_tools(tools)
        self._schemas = {t["name"]: t.get("parameters") or {} for t in self.tools}
        self.calls: list[dict] = []
        self.lock = threading.Lock()

    @property
    def tool_names(self) -> set[str]:
        return set(self._schemas)

    def _records(self, name: str) -> list[dict]:
        return [r for r in self.db.get(name) or [] if isinstance(r, dict) and isinstance(r.get("arguments"), dict)]

    def _missing(self, name: str, args: dict) -> list[str]:
        return [k for k in self._schemas.get(name, {}).get("required") or [] if _empty(args.get(k))]

    def _match(self, name: str, args: dict) -> dict | None:
        """번호가 맞는 기록. 주문번호 일치가 전화번호 일치보다 앞서고, 같으면 인수 구성이 같은 기록, 그다음 파일 순서."""
        orders, phones = _ids(args)
        given = {k for k, v in args.items() if not _empty(v)}
        best, best_score = None, 0.0
        for r in self._records(name):
            r_orders, r_phones = _ids(r["arguments"])
            score = 2.0 * bool(orders & r_orders) + 1.0 * bool(phones & r_phones)
            if score <= 0:
                continue
            if given == {k for k, v in r["arguments"].items() if not _empty(v)}:
                score += 0.5
            if score > best_score:
                best, best_score = r, score
        return best

    def _search(self, args: dict) -> tuple[dict, bool]:
        q = _compact(args.get("query"))
        recs = self._records("search_products")
        if not q:
            return {"status": "invalid", "message": "검색어가 없습니다."}, False
        if not recs:
            return {"status": "not_found", "message": "검색 결과가 없습니다."}, False

        def score(r: dict) -> float:
            rq = _compact(r["arguments"].get("query"))
            base = difflib.SequenceMatcher(None, q, rq).ratio()
            return base + (0.5 if rq and (q in rq or rq in q) else 0.0)

        best = max(recs, key=score)        # 동점이면 파일에서 먼저 나온 기록
        return best.get("result", {"status": "ok"}), True

    def lookup(self, name: str, args: dict) -> tuple[dict, bool]:
        """(응답 JSON, 기록과 맞았는지). name은 tool_names 안에 있다고 가정한다."""
        if name in SEARCH_FUNCS:
            return self._search(args)
        missing = self._missing(name, args)
        if name == "get_order_info" and not any(_ids(args)):
            missing = ["order_id 또는 buyer_cellphone"]
        if missing:
            return {"status": "invalid", "message": f"필수 값이 없습니다: {', '.join(missing)}"}, False
        rec = self._match(name, args)
        if rec is not None:
            return rec.get("result", {"status": "ok"}), True
        if name in LOOKUP_FUNCS:
            return {"status": "not_found", "message": "조회된 기록이 없습니다."}, False
        return {"status": "ok", "message": "접수되었습니다."}, False


def _read_json(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def default_paths() -> tuple[Path | None, Path | None]:
    """(mock_db.json, tools.json) 기본 위치: 제출 묶음의 data/ → 개발용 공식 키트."""
    db = next((d / "mock_db.json" for d in KIT_DIRS if (d / "mock_db.json").is_file()), None)
    tools = next((d / "brand" / "tools.json" for d in KIT_DIRS if (d / "brand" / "tools.json").is_file()), None)
    return db, tools


def load_store(db_path: Path | str | None = None, tools_path: Path | str | None = None, env=None) -> Store:
    env = os.environ if env is None else env
    d_default, t_default = default_paths()
    db = db_path or env.get(DATA_ENV) or d_default
    tools = tools_path or env.get(TOOLS_ENV) or t_default
    if not tools or not Path(tools).is_file():
        raise FileNotFoundError(f"함수 정의 파일이 없습니다: {tools} (--tools 또는 {TOOLS_ENV})")
    if not db or not Path(db).is_file():
        raise FileNotFoundError(f"모의 함수 데이터 파일이 없습니다: {db} (--db 또는 {DATA_ENV})")
    return Store(_read_json(db), _read_json(tools))


class MockServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, store: Store):
        super().__init__(addr, Handler)
        self.store = store


class Handler(BaseHTTPRequestHandler):
    server: MockServer

    def log_message(self, fmt, *args):
        pass

    def send_obj(self, obj, code=200):
        buf = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(buf)))
        self.end_headers()
        self.wfile.write(buf)

    def do_GET(self):
        u = urlparse(self.path)
        store = self.server.store
        if u.path == "/tools":
            return self.send_obj([{"type": "function", "function": t} for t in store.tools])
        if u.path == "/calls":
            sid = parse_qs(u.query).get("session_id", [None])[0]
            with store.lock:
                rows = [c for c in store.calls if sid is None or c["session_id"] == sid]
            return self.send_obj(rows)
        return self.send_obj({"error": "not found"}, 404)

    def do_POST(self):
        store = self.server.store
        path = urlparse(self.path).path
        name = path.removeprefix("/tools/")
        if not path.startswith("/tools/") or name not in store.tool_names:
            return self.send_obj({"error": "unknown function"}, 404)
        t0 = time.time()
        try:
            args = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        except (ValueError, TypeError):
            return self.send_obj({"error": "body must be JSON"}, 400)
        if not isinstance(args, dict):
            return self.send_obj({"error": "body must be a JSON object"}, 400)
        result, matched = store.lookup(name, args)
        with store.lock:
            store.calls.append({"session_id": self.headers.get("X-Session-Id"), "name": name, "arguments": args,
                                "matched": bool(matched), "t_request": t0, "t_response": time.time()})
        return self.send_obj(result)


def create_server(port: int = 8100, host: str = "0.0.0.0", store: Store | None = None) -> MockServer:
    """서버를 만들어 돌려준다(serve_forever는 호출자가). port=0이면 임의 포트."""
    return MockServer((host, port), store or load_store())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="공식 규격 모의 함수 서버")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--db", "--data", dest="db", default=None, help=f"mock_db.json 경로(기본 {DATA_ENV} → data/)")
    p.add_argument("--tools", default=None, help=f"tools.json 경로(기본 {TOOLS_ENV} → data/brand/)")
    return p


def main(argv: list[str] | None = None):
    a = build_parser().parse_args(argv)
    try:
        store = load_store(a.db, a.tools)
    except (FileNotFoundError, ValueError) as e:
        print(f"모의 함수 서버를 시작할 수 없습니다: {e}", file=sys.stderr)
        raise SystemExit(2)
    srv = create_server(a.port, a.host, store=store)
    print(f"mock tools on http://{a.host}:{srv.server_address[1]}  functions={len(store.tools)}  "
          f"records={sum(len(v) for v in store.db.values() if isinstance(v, list))}", file=sys.stderr, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
