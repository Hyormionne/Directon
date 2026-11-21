"""공식 규격 모의 함수 서버(AG-T36, directon-2026-voice-cs SPEC.md 「모의 함수 서버」).

- GET  /tools                 함수 목록(OpenAI tools 형식). 내용은 공식 `data/brand/tools.json` 사본(`mock_api/data/tools.json`).
- POST /tools/{함수명}         body = 인수 JSON, 응답 = 결과 JSON. 헤더 X-Session-Id를 호출 기록에 남긴다.
- GET  /calls?session_id=...  (디버깅용) 받은 호출 기록. 채점은 측정기 중계 기록으로 한다.

데이터는 `--db`(연습 `data/mock_db.json`, 최종 측정 `mock_db_test.json`), 모양 `{함수명: [{"arguments", "result"}]}`.
공식 데이터는 반출 금지라 저장소에 넣지 않는다(AD14). 기본 경로는 환경변수 DIRECTON_MOCK_DB → 공식 폴더.
응답 지연은 측정기 중계가 넣는다. 이 서버는 지연을 넣지 않고 상태도 바꾸지 않는다(같은 입력 → 같은 출력).

응답 규칙:
- 조회(get_order_info, get_return_info): 번호 인자가 맞는 기록의 result. 하이픈·공백 등 표기 차이는 무시.
  get_order_info는 주문번호·전화번호가 둘 다 맞는 기록 → 주문번호 → 전화번호 순. 없으면 not_found,
  번호가 하나도 없으면 invalid.
- 접수(cancel_order, return_order, return_order_item, exchange_order, request_callback): 필수 값이 빠지면 invalid.
  다 있으면 번호 인자가 맞는 기록의 result(예: already_requested), 기록이 없으면 {"status": "ok"}.
- 검색(search_products): 검색어가 가장 가까운 기록의 result. 기록이 없으면 not_found.
- 함수 정의에 없는 함수는 404.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOOLS_PATH = Path(__file__).resolve().parent / "data" / "tools.json"
OFFICIAL_DB_PATH = ROOT / "directon-2026-voice-cs" / "data" / "mock_db.json"
DB_ENV = "DIRECTON_MOCK_DB"

LOOKUP_FUNCS = {"get_order_info", "get_return_info"}
SEARCH_FUNCS = {"search_products"}
# 번호 인자: 표기 차이를 무시하고 비교한다. 기록 매칭에 쓴다.
ID_ARGS = ("order_id", "order_number", "buyer_cellphone", "customer_phone_number", "phone_number")
ORDER_ARGS = ("order_id", "order_number")
_SEP = re.compile(r"[\s\-_./()]")


def norm_id(v) -> str:
    return _SEP.sub("", str(v)).lower() if v not in (None, "") else ""


def _norm_text(v) -> str:
    return re.sub(r"[\s\W_]+", "", str(v or "")).lower()


def _filled(v) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, (list, dict)):
        return bool(v)
    return True


def _item_codes(args: dict) -> set[str]:
    items = args.get("return_items")
    if not isinstance(items, list):
        return set()
    return {norm_id(i.get("order_item_code")) for i in items if isinstance(i, dict) and i.get("order_item_code")}


def invalid(missing: list[str]) -> dict:
    return {"status": "invalid", "message": f"필수 값이 빠졌습니다: {', '.join(missing)}"}


NOT_FOUND = {"status": "not_found", "message": "조회된 기록이 없습니다."}


class Store:
    def __init__(self, db: dict | None, tools: list[dict]):
        self.db: dict[str, list[dict]] = {k: v for k, v in (db or {}).items() if isinstance(v, list)}
        self.tools = tools
        self.schemas = {t["name"]: t.get("parameters") or {} for t in tools}
        self.calls: list[dict] = []
        self.lock = threading.Lock()

    @property
    def tool_names(self) -> set[str]:
        return set(self.schemas)

    def _rows(self, name: str) -> list[tuple[dict, dict]]:
        out = []
        for r in self.db.get(name, []):
            if isinstance(r, dict) and isinstance(r.get("arguments"), dict):
                out.append((r["arguments"], r.get("result", {"status": "ok"})))
        return out

    def lookup(self, name: str, args: dict) -> tuple[dict, bool]:
        """(결과 JSON, 기록과 맞았는지)."""
        args = args if isinstance(args, dict) else {}
        if name in LOOKUP_FUNCS:
            return self._lookup(name, args)
        if name in SEARCH_FUNCS:
            return self._search(name, args)
        return self._write(name, args)

    def _missing(self, name: str, args: dict) -> list[str]:
        req = self.schemas.get(name, {}).get("required") or []
        return [k for k in req if not _filled(args.get(k))]

    def _lookup(self, name: str, args: dict) -> tuple[dict, bool]:
        given = {k: norm_id(args.get(k)) for k in ID_ARGS if norm_id(args.get(k))}
        if not given:
            props = list(self.schemas.get(name, {}).get("properties") or [])
            return invalid([k for k in props if k in ID_ARGS] or props), False
        rows = self._rows(name)

        def hits(rargs: dict) -> set[str]:
            return {k for k, v in given.items() if norm_id(rargs.get(k)) == v}

        best, best_rank = None, 0
        for rargs, result in rows:
            h = hits(rargs)
            if not h:
                continue
            rank = 3 if h == set(given) and len(given) > 1 else (2 if h & set(ORDER_ARGS) else 1)
            if rank > best_rank:
                best, best_rank = result, rank
        if best is None:
            return dict(NOT_FOUND), False
        return best, True

    def _write(self, name: str, args: dict) -> tuple[dict, bool]:
        missing = self._missing(name, args)
        if missing:
            return invalid(missing), False
        given = {k: norm_id(args.get(k)) for k in ID_ARGS if norm_id(args.get(k))}
        codes = _item_codes(args)
        for rargs, result in self._rows(name):
            if given and all(norm_id(rargs.get(k)) == v for k, v in given.items()):
                return result, True
            if codes and codes == _item_codes(rargs):
                return result, True
        return {"status": "ok"}, False

    def _search(self, name: str, args: dict) -> tuple[dict, bool]:
        missing = self._missing(name, args) or ([] if _filled(args.get("query")) else ["query"])
        if missing:
            return invalid(missing), False
        q = _norm_text(args.get("query"))
        best, best_score = None, -1.0
        for rargs, result in self._rows(name):
            c = _norm_text(rargs.get("query"))
            score = difflib.SequenceMatcher(None, q, c).ratio()
            if q and c and (q in c or c in q):
                score += 1.0
            if score > best_score:
                best, best_score = result, score
        if best is None:
            return dict(NOT_FOUND), False
        return best, True


def _read_json(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_tools(path: Path | str = DEFAULT_TOOLS_PATH) -> list[dict]:
    raw = _read_json(path)
    if isinstance(raw, dict) and isinstance(raw.get("tools"), list):
        raw = raw["tools"]
    return [t["function"] if isinstance(t, dict) and isinstance(t.get("function"), dict) else t for t in raw]


def default_db_path(env=None) -> Path:
    env = os.environ if env is None else env
    return Path(env.get(DB_ENV) or OFFICIAL_DB_PATH)


def load_store(db_path: Path | str | None = None, tools_path: Path | str | None = None) -> Store:
    """파일이 없으면 빈 DB로 띄운다(조회는 not_found, 접수는 필수 값만 본다)."""
    p = Path(db_path) if db_path else default_db_path()
    db = _read_json(p) if p.exists() else {}
    return Store(db, load_tools(tools_path or DEFAULT_TOOLS_PATH))


class MockServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, store: Store):
        self.store = store
        super().__init__(addr, Handler)


class Handler(BaseHTTPRequestHandler):
    server: MockServer

    def log_message(self, *a):
        pass

    def _send(self, code: int, obj) -> None:
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        store = self.server.store
        u = urlparse(self.path)
        if u.path == "/tools":
            return self._send(200, [{"type": "function", "function": t} for t in store.tools])
        if u.path == "/calls":
            sid = parse_qs(u.query).get("session_id", [None])[0]
            with store.lock:
                return self._send(200, [c for c in store.calls if sid is None or c["session_id"] == sid])
        self._send(404, {"error": "not found"})

    def do_POST(self):
        store = self.server.store
        m = re.fullmatch(r"/tools/(\w+)", urlparse(self.path).path)
        if not m or m.group(1) not in store.tool_names:
            return self._send(404, {"error": "unknown function"})
        name = m.group(1)
        t0 = time.time()
        try:
            args = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._send(400, {"error": "body must be JSON"})
        result, matched = store.lookup(name, args)
        rec = {"session_id": self.headers.get("X-Session-Id"), "name": name, "arguments": args,
               "matched": matched, "t_request": t0, "t_response": time.time()}
        with store.lock:
            store.calls.append(rec)
        self._send(200, result)


def create_server(port: int = 8100, host: str = "127.0.0.1", store: Store | None = None) -> MockServer:
    """서버를 만들어 돌려준다(serve_forever는 호출자가). port=0이면 임의 포트."""
    return MockServer((host, port), store or load_store())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="공식 규격 모의 함수 서버")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--db", default=None, help=f"mock_db.json 경로(기본 {DB_ENV} → 공식 폴더)")
    p.add_argument("--tools", default=None, help="tools.json 경로(기본 mock_api/data/tools.json)")
    return p


def main(argv: list[str] | None = None):
    a = build_parser().parse_args(argv)
    db = Path(a.db) if a.db else default_db_path()
    store = load_store(db, a.tools)
    srv = create_server(a.port, a.host, store=store)
    print(f"mock tools on http://{a.host}:{srv.server_address[1]} functions={len(store.tools)} "
          f"records={sum(map(len, store.db.values()))} db={db}{'' if db.exists() else ' (없음, 빈 DB)'}",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
