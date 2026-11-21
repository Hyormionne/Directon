"""가상 쇼핑몰 '다이렉트몰' Mock API 서버.

Starter Kit `mock_tools/mock_api_template.py`와 같은 경로·응답 형식:
- GET  /tools                 -> [{"type":"function","function":{name,description,parameters}}]
- POST /tools/{name}          -> 함수 결과 JSON (X-Session-Id 헤더로 세션 구분)
- GET  /calls?session_id=...  -> [{session_id,name,arguments,matched,t_response}]

matched 규칙은 `legacy/shop_mock/README.md` 참고. 상태를 바꾸지 않는다(같은 입력 → 같은 출력).

AG-T17 당일 교체: `--data <mock_data.json> --tools <tools.json>`(또는 환경변수 DIRECTON_MOCK_DATA,
DIRECTON_MOCK_TOOLS, 인자가 우선). 둘 다 없으면 `legacy/shop_mock/`. 데이터 형식은 두 가지:
- 쇼핑몰 형식(members·orders·products·refunds·coupons): 함수별 업무 규칙(_fn_*)으로 답한다.
- 매핑 형식(Starter Kit `demo_mock_data.json`): {함수명: [{"arguments": {...}, "result": {...}}]}.
  항목의 인자가 모두 같으면 그 result(matched=true), 없으면 not_found(matched=false).
  자유 텍스트 인자(FREE_TEXT_ARGS)는 정확 일치 항목이 없을 때 비교에서 뺀다.
함수 정의는 [{name, description, parameters}] 또는 OpenAI 형식 [{"type":"function","function":{...}}] 둘 다 읽는다.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

DATA_DIR = Path(__file__).resolve().parent.parent / "legacy" / "shop_mock"

# matched 판정에 쓰는 키 인자. 이 밖의 인자(reason, new_address, preferred_time)는 자유 텍스트로 판정에서 제외한다.
KEY_ARGS = {"phone_number", "order_id", "product_name", "option"}

PRE_SHIP = ("결제완료", "상품준비중", "배송준비")

DATA_ENV = "DIRECTON_MOCK_DATA"
TOOLS_ENV = "DIRECTON_MOCK_TOOLS"
FREE_TEXT_ARGS = {"reason", "new_address", "preferred_time", "memo", "note", "comment", "detail", "message"}
_MAP_ARG_KEYS = ("arguments", "args", "input", "params")
_MAP_RESULT_KEYS = ("result", "output", "response")
NOT_FOUND = {"status": "not_found", "message": "정의되지 않은 입력입니다."}


def is_mapping_data(data) -> bool:
    """Starter Kit 매핑 형식인가: 모든 값이 [{arguments, result}] 목록."""
    if not isinstance(data, dict) or not data:
        return False
    for rows in data.values():
        if not isinstance(rows, list):
            return False
        for r in rows:
            if not isinstance(r, dict) or not any(isinstance(r.get(k), dict) for k in _MAP_ARG_KEYS):
                return False
    return True


def _entry_args(r: dict) -> dict:
    return next(r[k] for k in _MAP_ARG_KEYS if isinstance(r.get(k), dict))


def _entry_result(r: dict):
    return next((r[k] for k in _MAP_RESULT_KEYS if k in r), {"status": "ok"})


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


def ok(data=None, message=None):
    out = {"status": "ok"}
    if message is not None:
        out["message"] = message
    if data is not None:
        out["data"] = data
    return out


def fail(status, message, **extra):
    return {"status": status, "message": message, **extra}


@dataclass
class Shop:
    tools: list[dict]
    data: dict
    _members: dict = field(init=False)
    _orders: dict = field(init=False)
    _products: dict = field(init=False)
    _refunds: dict = field(init=False)
    _schemas: dict = field(init=False)

    mapping: bool = field(init=False, default=False)

    def __post_init__(self):
        self._schemas = {t["name"]: t.get("parameters") or {} for t in self.tools}
        self.mapping = is_mapping_data(self.data)
        if self.mapping:
            self._members = self._orders = self._products = self._refunds = {}
            return
        self._members = {m["phone_number"]: m for m in self.data["members"]}
        self._orders = {o["order_id"]: o for o in self.data["orders"]}
        self._products = {p["name"]: p for p in self.data["products"]}
        self._refunds = {r["order_id"]: r for r in self.data["refunds"]}

    @property
    def tool_names(self) -> set[str]:
        return set(self._schemas)

    # ---------- 진입점 ----------
    def lookup(self, name: str, args: dict) -> tuple[dict, bool]:
        """(응답 JSON, matched). name은 tool_names 안에 있다고 가정한다."""
        err = self._validate(name, args)
        if err:
            return fail("invalid_arguments", err), False
        if self.mapping:
            return self._map_lookup(name, args)
        fn = getattr(self, f"_fn_{name}", None)
        if fn is None:                       # 함수 정의만 바뀌고 업무 규칙이 없는 함수
            return dict(NOT_FOUND), False
        return fn(args)

    def _map_lookup(self, name, args):
        rows = [r for r in self.data.get(name, []) if isinstance(r, dict)]
        for strict in (True, False):
            for r in rows:
                want = _entry_args(r)
                keys = [k for k in want if strict or k not in FREE_TEXT_ARGS]
                if all(args.get(k) == want[k] for k in keys) and (strict or keys):
                    return json.loads(json.dumps(_entry_result(r))), True
        return dict(NOT_FOUND), False

    def _validate(self, name, args):
        schema = self._schemas[name]
        props = schema.get("properties") or {}
        required = schema.get("required") or []
        unknown = [k for k in args if k not in props]
        if unknown:
            return f"정의되지 않은 인자입니다: {', '.join(unknown)}"
        missing = [k for k in required if k not in args]
        if missing:
            return f"필수 인자가 없습니다: {', '.join(missing)}"
        for k, v in args.items():
            ptype = (props.get(k) or {}).get("type", "string")
            if ptype == "string" and not isinstance(v, str):
                return f"인자 {k}는 문자열이어야 합니다."
            if k in required and isinstance(v, str) and not v.strip():
                return f"인자 {k}가 비어 있습니다."
        return None

    # ---------- 공통 조회 ----------
    def _member(self, args):
        m = self._members.get(args["phone_number"])
        if m is None:
            return None, fail("not_found", "해당 전화번호로 가입된 회원을 찾을 수 없습니다.")
        return m, None

    def _order(self, args):
        o = self._orders.get(args["order_id"])
        if o is None:
            return None, fail("not_found", "해당 주문번호의 주문을 찾을 수 없습니다.")
        return o, None

    @staticmethod
    def _summary(o):
        it = o["items"][0]
        return {"order_id": o["order_id"], "ordered_at": o["ordered_at"], "status": o["status"],
                "product_name": it["product_name"], "option": it["option"], "quantity": it["quantity"],
                "total_amount": o["total_amount"]}

    # ---------- 회원 ----------
    def _fn_get_member_info(self, a):
        m, e = self._member(a)
        if e:
            return e, False
        return ok({k: m[k] for k in ("phone_number", "name", "grade", "points", "address", "joined_at")}), True

    def _fn_get_order_list(self, a):
        m, e = self._member(a)
        if e:
            return e, False
        rows = sorted((o for o in self.data["orders"] if o["phone_number"] == m["phone_number"]),
                      key=lambda o: o["ordered_at"], reverse=True)
        return ok({"name": m["name"], "orders": [self._summary(o) for o in rows]}), True

    def _fn_get_coupons(self, a):
        m, e = self._member(a)
        if e:
            return e, False
        return ok({"name": m["name"], "coupons": self.data["coupons"].get(m["phone_number"], [])}), True

    def _fn_request_callback(self, a):
        m, e = self._member(a)
        if e:
            return e, False
        return ok({"callback_id": "CB" + m["phone_number"][3:], "phone_number": m["phone_number"]},
                  "콜백 요청이 접수되었습니다. 영업시간(평일 09:00~18:00) 중 상담원이 연락드립니다."), True

    def _fn_transfer_to_agent(self, a):
        return ok({"queue": "일반상담", "estimated_wait_minutes": 2},
                  "상담원에게 연결합니다. 예상 대기 시간은 약 2분입니다."), True

    # ---------- 주문 ----------
    def _fn_get_order_info(self, a):
        o, e = self._order(a)
        if e:
            return e, False
        keys = ("order_id", "ordered_at", "status", "items", "product_amount", "shipping_fee", "total_amount",
                "payment_method", "shipping_address", "expected_ship_date", "cancelled_at", "requested_at")
        return ok({k: o[k] for k in keys if k in o}), True

    def _fn_track_delivery(self, a):
        o, e = self._order(a)
        if e:
            return e, False
        d = {"order_id": o["order_id"], "status": o["status"]}
        for k in ("courier", "tracking_number", "shipped_at", "current_location", "estimated_delivery", "delivered_at"):
            if k in o:
                d[k] = o[k]
        st = o["status"]
        if st in PRE_SHIP:
            d["expected_ship_date"] = o["expected_ship_date"]
            msg = "아직 출고 전입니다. 출고되면 송장번호를 문자로 보내드립니다."
        elif st == "배송중":
            msg = f"배송중입니다. 현재 위치는 {o['current_location']}이고 {o['estimated_delivery']} 도착 예정입니다."
        elif st == "취소완료":
            msg = "취소된 주문이라 배송 정보가 없습니다."
        else:
            msg = f"{o['delivered_at']}에 배송 완료되었습니다."
        return ok(d, msg), True

    def _fn_cancel_order(self, a):
        o, e = self._order(a)
        if e:
            return e, False
        st, oid = o["status"], o["order_id"]
        if st in ("결제완료", "상품준비중"):
            return ok({"order_id": oid, "status": "취소완료", "refund_amount": o["total_amount"],
                       "refund_method": o["payment_method"], "refund_eta": "카드 3~5영업일, 그 외 1~3영업일"},
                      "주문이 취소되었습니다. 결제 수단으로 전액 환불됩니다."), True
        if st == "배송준비":
            return ok({"order_id": oid, "status": "취소요청", "refund_amount": o["total_amount"],
                       "refund_method": o["payment_method"]},
                      "출고 직전 주문이라 취소 요청으로 접수되었습니다. 출고를 막으면 1영업일 안에 취소 확정 문자를 드립니다."), True
        msgs = {
            "배송중": "이미 배송이 시작되어 취소할 수 없습니다. 상품을 받으신 뒤 반품을 신청해 주세요.",
            "배송완료": "배송이 완료된 주문은 취소할 수 없습니다. 수령 후 7일 이내라면 반품을 신청해 주세요.",
            "취소완료": "이미 취소된 주문입니다.",
            "반품접수": "이미 반품이 접수된 주문이라 취소할 수 없습니다.",
            "교환접수": "교환이 접수된 주문이라 취소할 수 없습니다.",
        }
        return fail("not_allowed", msgs[st], data={"order_id": oid, "status": st}), True

    def _return_like(self, a, kind):
        """kind: '반품' 또는 '교환'"""
        o, e = self._order(a)
        if e:
            return e, False
        st, oid = o["status"], o["order_id"]
        base = {"order_id": oid, "status": st}
        if st in PRE_SHIP:
            return fail("not_allowed", f"아직 출고 전 주문이라 {kind} 대신 주문 취소가 가능합니다.", data=base), True
        if st == "배송중":
            return fail("not_allowed", f"배송중인 주문입니다. 상품을 받으신 뒤 7일 이내에 {kind}을 신청해 주세요.", data=base), True
        if st == "취소완료":
            return fail("not_allowed", "취소된 주문이라 반품이나 교환을 할 수 없습니다.", data=base), True
        if st in ("반품접수", "교환접수"):
            return fail("not_allowed", f"이미 {st[:2]}이 접수된 주문입니다.", data=base), True
        if o.get("return_window_expired"):
            return fail("not_allowed", f"수령 후 7일이 지나 {kind} 접수 기간이 끝났습니다(기한 {o['return_deadline']}).",
                        data={**base, "return_deadline": o["return_deadline"]}), True
        new_status = "반품접수" if kind == "반품" else "교환접수"
        d = {"order_id": oid, "status": new_status, "pickup": "1~2영업일 안에 택배 기사가 회수 방문",
             "shipping_fee_policy": "단순변심은 왕복 배송비 6,000원 고객 부담, 불량·오배송은 무료"}
        if kind == "반품":
            d["refund_amount_before_fee"] = o["product_amount"]
            msg = "반품이 접수되었습니다. 회수 확인 후 환불되며, 카드는 3~5영업일이 걸립니다."
        else:
            msg = "교환이 접수되었습니다. 회수 확인 후 새 상품을 보내드립니다."
        return ok(d, msg), True

    def _fn_request_return(self, a):
        return self._return_like(a, "반품")

    def _fn_request_exchange(self, a):
        return self._return_like(a, "교환")

    def _fn_change_shipping_address(self, a):
        o, e = self._order(a)
        if e:
            return e, False
        st, oid = o["status"], o["order_id"]
        if st in ("결제완료", "상품준비중"):
            return ok({"order_id": oid, "status": st}, "배송지가 변경되었습니다. 변경된 주소로 출고됩니다."), True
        msg = "이미 출고 준비가 끝나 배송지를 바꿀 수 없습니다." if st in ("배송준비", "배송중") \
            else f"{st} 상태의 주문이라 배송지를 바꿀 수 없습니다."
        return fail("not_allowed", msg, data={"order_id": oid, "status": st}), True

    def _fn_get_refund_status(self, a):
        o, e = self._order(a)
        if e:
            return e, False
        r = self._refunds.get(o["order_id"])
        if r is None:
            return fail("no_refund", "이 주문에는 환불 내역이 없습니다.",
                        data={"order_id": o["order_id"], "status": o["status"]}), True
        return ok(dict(r)), True

    # ---------- 상품 ----------
    def _product(self, args):
        p = self._products.get(args["product_name"])
        if p is None:
            return None, fail("not_found", "해당 이름의 상품을 찾을 수 없습니다.")
        return p, None

    def _fn_get_product_info(self, a):
        p, e = self._product(a)
        if e:
            return e, False
        fee = 0 if p["price"] >= self.data["shop"]["free_shipping_threshold"] else self.data["shop"]["base_shipping_fee"]
        return ok({"name": p["name"], "price": p["price"], "category": p["category"], "description": p["description"],
                   "options": [o["name"] for o in p["options"]], "shipping_fee": fee,
                   "shipping_policy": "5만원 이상 무료배송, 결제 후 1~3영업일 출고"}), True

    def _fn_check_stock(self, a):
        p, e = self._product(a)
        if e:
            return e, False
        opts = p["options"]
        if "option" in a:
            opts = [o for o in opts if o["name"] == a["option"]]
            if not opts:
                return fail("not_found", "해당 상품에 그런 옵션이 없습니다.",
                            data={"available_options": [o["name"] for o in p["options"]]}), False
        rows = [{"option": o["name"], "stock": o["stock"], "in_stock": o["stock"] > 0,
                 **({"restock_date": o["restock_date"]} if "restock_date" in o else {})} for o in opts]
        return ok({"product_name": p["name"], "options": rows}), True


def _read_json(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_shop(data_dir: Path | str = DATA_DIR, *, data_path: Path | str | None = None,
              tools_path: Path | str | None = None) -> Shop:
    """data_path·tools_path를 주면 그 파일, 아니면 data_dir의 mock_data.json·tools.json."""
    d = Path(data_dir)
    return Shop(tools=normalize_tools(_read_json(tools_path or d / "tools.json")),
                data=_read_json(data_path or d / "mock_data.json"))


class MockServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, shop: Shop):
        super().__init__(addr, Handler)
        self.shop = shop
        self.calls: list[dict] = []
        self.lock = threading.Lock()


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
        if u.path == "/tools":
            return self.send_obj([{"type": "function", "function": t} for t in self.server.shop.tools])
        if u.path == "/calls":
            sid = parse_qs(u.query).get("session_id", [None])[0]
            with self.server.lock:
                rows = [c for c in self.server.calls if sid is None or c["session_id"] == sid]
            return self.send_obj(rows)
        return self.send_obj({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        name = path.removeprefix("/tools/")
        if not path.startswith("/tools/") or name not in self.server.shop.tool_names:
            return self.send_obj({"error": "unknown function"}, 404)
        try:
            args = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        except (ValueError, TypeError):
            return self.send_obj({"error": "invalid JSON"}, 400)
        if not isinstance(args, dict):
            return self.send_obj({"error": "invalid JSON"}, 400)
        result, matched = self.server.shop.lookup(name, args)
        with self.server.lock:
            self.server.calls.append({"session_id": self.headers.get("X-Session-Id"), "name": name,
                                      "arguments": args, "matched": bool(matched), "t_response": time.time()})
        return self.send_obj(result)


def create_server(port: int = 8100, host: str = "127.0.0.1", shop: Shop | None = None) -> MockServer:
    """서버를 만들어 돌려준다(serve_forever는 호출자가). port=0이면 임의 포트."""
    return MockServer((host, port), shop or load_shop())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="다이렉트몰 Mock API")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--data", default=None, help=f"mock_data.json 경로(기본 {DATA_ENV} → legacy/shop_mock/)")
    p.add_argument("--tools", default=None, help=f"tools.json 경로(기본 {TOOLS_ENV} → legacy/shop_mock/)")
    return p


def resolve_paths(a: argparse.Namespace, env=None) -> tuple[Path, Path]:
    env = os.environ if env is None else env
    data = a.data or env.get(DATA_ENV) or DATA_DIR / "mock_data.json"
    tools = a.tools or env.get(TOOLS_ENV) or DATA_DIR / "tools.json"
    return Path(data), Path(tools)


def shop_from_args(argv: list[str] | None = None, env=None) -> Shop:
    data, tools = resolve_paths(build_parser().parse_args(argv), env)
    return load_shop(data_path=data, tools_path=tools)


def main(argv: list[str] | None = None):
    a = build_parser().parse_args(argv)
    data, tools = resolve_paths(a)
    srv = create_server(a.port, a.host, shop=load_shop(data_path=data, tools_path=tools))
    mode = "mapping" if srv.shop.mapping else "shop"
    print(f"Directmall mock API on http://{a.host}:{srv.server_address[1]} ({len(srv.shop.tools)} tools, "
          f"{mode}) data={data} tools={tools}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
