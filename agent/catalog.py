"""카탈로그 기준 상품명·옵션 정규화 (AG-T10d, 표준 라이브러리만).

Mock API(와 공식 채점)는 product_name·option을 데이터 표기와 문자 그대로 비교한다. LLM이 고객 말
그대로("엑스라지", "회색", "후드티", "스마트 워치 밴드") 넘기면 not_found가 되므로 카탈로그 표기로 맞춘다.

해석 순서(앞 단계에서 정해지면 끝. 어느 단계든 후보가 둘 이상이면 None = 원문 유지):
1. 공백·대소문자 무시 정확 일치
2. 별칭(옵션만): 사이즈(엠→M, 엑스라지→XL …)·색상(회색↔그레이 …) 동의어 묶음, 한글 수 읽기("이백칠십"→270,
   "십오 인치"→15인치). 끝의 "사이즈", "색", "으로", "요" 같은 말은 떼어 보고 다시 맞춘다.
   별칭 묶음에 있는 말인데 그 상품에 해당 옵션이 없으면(예: 후드티에 "XXL") 바로 None(부분 일치로 넘기지 않음).
3. 부분 일치: 고객 표현이 카탈로그 이름의 부분 문자열이거나 그 반대(2글자 이상). 숫자만인 표현은
   "15"→"15인치"처럼 숫자 뒤에 단위만 붙은 옵션에만 맞춘다("28"→"280" 금지).
   상품명은 추가로 낱말 일치(카탈로그 이름의 2글자 이상 낱말이 표현에 몇 개 들었나, 최댓값이 유일할 때).
4. 문자 유사도(difflib) 0.75 이상이고 2등과 0.1 이상 차이 날 때.

카탈로그 출처: 환경변수 DIRECTON_CATALOG_PATH → DIRECTON_MOCK_DATA(AG-T17 당일 교체) → mock_api/data/mock_data.json.
파일이 없거나 형식이 다르면 None(정규화 생략). Starter Kit 형식({함수명: [{arguments, result}]})이면
arguments의 product_name·option에서 상품·옵션을 모은다.

번호 자릿수(AG-T17): order_id_lengths()·phone_lengths()가 같은 데이터에서 주문번호·전화번호 길이를 모은다
(키 이름이 order_id/phone을 포함하는 값). 모르면 기존 가정(주문번호 10자리, 전화번호 11자리).
"""
from __future__ import annotations

import difflib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from agent.normalize import spoken_digits

__all__ = ["Catalog", "load_catalog", "default_catalog", "reset_default_catalog", "catalog_enabled",
           "DEFAULT_CATALOG_PATH", "data_path", "order_id_lengths", "phone_lengths", "order_id_example",
           "DEFAULT_ORDER_LENGTHS", "DEFAULT_PHONE_LENGTHS"]

# AG-T38: 기본 파일 없음. 공식 측정에서 상담 서버가 mock 데이터(mock_db_test.json)를 읽으면 안 된다(README 7장 금지 사항).
# 옛 체계 도구는 DIRECTON_CATALOG_PATH / DIRECTON_MOCK_DATA로 legacy mock_data.json을 준다.
DEFAULT_CATALOG_PATH: Path | None = None
LEGACY_CATALOG_PATH = Path(__file__).resolve().parent.parent / "mock_api" / "data" / "mock_data.json"

_SIM_THRESHOLD = 0.75
_SIM_MARGIN = 0.1

# 같은 옵션을 가리키는 표현 묶음(비교는 공백 제거·소문자).
_OPTION_GROUPS: list[set[str]] = [
    # 사이즈
    {"xs", "엑스스몰", "엑스에스"},
    {"s", "에스", "스몰", "small"},
    {"m", "엠", "미디엄", "미디움", "미듐", "medium"},
    {"l", "엘", "라지", "large"},
    {"xl", "엑스라지", "엑스엘", "엑스트라라지", "extralarge"},
    {"xxl", "2xl", "투엑스라지", "투엑스엘", "투엑스", "이엑스라지", "2엑스라지", "엑스엑스라지", "엑스엑스엘"},
    {"xxxl", "3xl", "쓰리엑스라지", "쓰리엑스엘", "삼엑스라지", "3엑스라지"},
    {"free", "프리", "프리사이즈", "원사이즈", "onesize"},
    # 색상
    {"그레이", "회색", "그레이색", "쥐색", "gray", "grey"},
    {"블랙", "검정", "검정색", "검은색", "검은", "까만색", "까망", "black"},
    {"화이트", "흰색", "하얀색", "하양", "흰", "white"},
    {"네이비", "남색", "곤색", "navy"},
    {"레드", "빨강", "빨간색", "빨강색", "적색", "red"},
    {"블루", "파랑", "파란색", "파랑색", "청색", "blue"},
    {"핑크", "분홍", "분홍색", "핑크색", "pink"},
    {"베이지", "베이지색", "beige"},
    {"카키", "카키색", "국방색", "khaki"},
    {"퍼플", "보라", "보라색", "purple"},
    {"실버", "은색", "silver"},
    {"골드", "금색", "gold"},
    {"그린", "초록", "초록색", "녹색", "green"},
    {"옐로", "옐로우", "노랑", "노란색", "노랑색", "yellow"},
    {"브라운", "갈색", "brown"},
    {"오렌지", "주황", "주황색", "orange"},
    {"아이보리", "아이보리색", "ivory"},
    {"민트", "민트색", "mint"},
]
_ALIAS: dict[str, int] = {w: i for i, g in enumerate(_OPTION_GROUPS) for w in g}

# 옵션 표현 끝에서 떼어 볼 말(긴 것 먼저)
_OPT_SUFFIXES = ("이에요", "사이즈", "색상", "색깔", "컬러", "칼라", "으로", "이요", "예요", "에요",
                 "요", "로", "색", "은", "는", "도")
# 상품명 표현 끝에서 떼어 볼 조사
_PROD_SUFFIXES = ("이에요", "이요", "예요", "에요", "요", "을", "를", "은", "는", "이", "가", "도")

_SINO_RUN = re.compile(r"^([영공일이삼사오육칠팔구십백천]+)(.*)$")


def _compact(s: str) -> str:
    return re.sub(r"\s+", "", s or "").casefold()


def _strip_variants(s: str, suffixes: Iterable[str]) -> list[str]:
    """s와, 끝말을 하나씩 떼어 낸 형태들(2글자 이상 남을 때만은 아님: 1글자 사이즈 'm'도 남긴다)."""
    out = [s]
    cur = s
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if cur.endswith(suf) and len(cur) > len(suf):
                cur = cur[: -len(suf)]
                out.append(cur)
                changed = True
                break
    return out


def _unique(items: list[str]) -> str | None:
    items = list(dict.fromkeys(items))
    return items[0] if len(items) == 1 else None


def _similar(text: str, names: dict[str, str]) -> str | None:
    if len(text) < 2:
        return None
    scored = sorted(((difflib.SequenceMatcher(None, text, k).ratio(), v) for k, v in names.items()),
                    reverse=True)
    if not scored or scored[0][0] < _SIM_THRESHOLD:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < _SIM_MARGIN:
        return None
    return scored[0][1]


class Catalog:
    """상품명 → 옵션명 목록. resolve_*는 카탈로그 표기를 돌려주고, 애매하거나 모르면 None."""

    def __init__(self, products: dict[str, Iterable[str]]):
        self.products: dict[str, list[str]] = {str(k): [str(o) for o in (v or [])] for k, v in products.items()}
        self._by_compact = {_compact(name): name for name in self.products}

    # ── 상품명 ─────────────────────────────────────
    def resolve_product(self, text: str) -> str | None:
        if not isinstance(text, str):
            return None
        base = _compact(text)
        if not base:
            return None
        variants = _strip_variants(base, _PROD_SUFFIXES)
        for v in variants:
            if v in self._by_compact:
                return self._by_compact[v]
        # 부분 일치
        if len(base) >= 2:
            inner = [n for k, n in self._by_compact.items() if k in base]           # 카탈로그 ⊂ 표현
            outer = [n for k, n in self._by_compact.items()
                     if any(len(v) >= 2 and v in k for v in variants)]               # 표현 ⊂ 카탈로그
            cands = list(dict.fromkeys(inner + outer))
            if cands:
                if len(cands) == 1:
                    return cands[0]
                if inner:  # 표현 안에 이름이 통째로 든 것 중 가장 긴 것이 유일하면
                    longest = max(len(_compact(n)) for n in inner)
                    top = [n for n in inner if len(_compact(n)) == longest]
                    if len(top) == 1 and not outer:
                        return top[0]
                return None
        # 낱말 일치
        scores: dict[str, int] = {}
        for name in self.products:
            toks = [_compact(t) for t in name.split() if len(_compact(t)) >= 2]
            scores[name] = sum(1 for t in toks if t in base)
        best = max(scores.values(), default=0)
        if best > 0:
            return _unique([n for n, s in scores.items() if s == best])
        return _similar(base, self._by_compact)

    # ── 옵션 ───────────────────────────────────────
    def resolve_option(self, product: str, text: str) -> str | None:
        if not isinstance(product, str) or not isinstance(text, str):
            return None
        name = self._by_compact.get(_compact(product))
        if name is None:
            return None
        options = self.products[name]
        if not options:
            return None
        by_compact = {_compact(o): o for o in options}
        base = _compact(text)
        if not base:
            return None
        variants = _strip_variants(base, _OPT_SUFFIXES)
        # 한글 수 읽기 → 아라비아 숫자("이백칠십"→"270", "십오인치"→"15인치")
        for v in list(variants):
            m = _SINO_RUN.match(v)
            if m:
                d = spoken_digits(m.group(1))
                if d:
                    variants.extend(_strip_variants(d + m.group(2), _OPT_SUFFIXES))
        variants = list(dict.fromkeys(variants))
        # 1. 정확 일치
        for v in variants:
            if v in by_compact:
                return by_compact[v]
        # 2. 별칭
        for v in variants:
            if v in _ALIAS:
                group = _OPTION_GROUPS[_ALIAS[v]]
                return _unique([o for k, o in by_compact.items() if k in group])
        # 3. 부분 일치
        for v in variants:
            if v.isdigit():
                cands = [o for k, o in by_compact.items()
                         if k.startswith(v) and k[len(v):] and not k[len(v)].isdigit()]
            elif len(v) >= 2:
                cands = [o for k, o in by_compact.items() if (len(k) >= 2 and k in v) or v in k]
            else:
                continue
            if cands:
                return _unique(cands)
        # 4. 유사도(숫자가 섞이면 하지 않음)
        if not any(ch.isdigit() for ch in base):
            return _similar(base, by_compact)
        return None


# ── 로드 ───────────────────────────────────────────

def _option_names(raw: Any) -> list[str] | None:
    if raw is None:
        return []
    if not isinstance(raw, list):
        return None
    out = []
    for o in raw:
        if isinstance(o, str):
            out.append(o)
        elif isinstance(o, dict) and isinstance(o.get("name", o.get("option")), str):
            out.append(o.get("name", o.get("option")))
        else:
            return None
    return out


_ARG_KEYS = ("arguments", "args", "input", "params")


def _mapping_entries(data: Any) -> list[dict] | None:
    """Starter Kit 형식이면 모든 항목의 인자 dict 목록, 아니면 None."""
    if not isinstance(data, dict) or not data:
        return None
    out = []
    for rows in data.values():
        if not isinstance(rows, list):
            return None
        for r in rows:
            if not isinstance(r, dict):
                return None
            a = next((r[k] for k in _ARG_KEYS if isinstance(r.get(k), dict)), None)
            if a is None:
                return None
            out.append(a)
    return out


def _parse_mapping(data: Any) -> dict[str, list[str]] | None:
    entries = _mapping_entries(data)
    if not entries:
        return None
    products: dict[str, list[str]] = {}
    for a in entries:
        name = a.get("product_name")
        if not isinstance(name, str) or not name.strip():
            continue
        opts = products.setdefault(name, [])
        opt = a.get("option")
        if isinstance(opt, str) and opt.strip() and opt not in opts:
            opts.append(opt)
    return products or None


def _parse(data: Any) -> dict[str, list[str]] | None:
    if isinstance(data, dict) and "products" not in data:
        return _parse_mapping(data)
    if isinstance(data, dict):
        data = data.get("products")
    products: dict[str, list[str]] = {}
    if isinstance(data, dict):
        for name, opts in data.items():
            names = _option_names(opts.get("options") if isinstance(opts, dict) else opts)
            if not isinstance(name, str) or names is None:
                return None
            products[name] = names
    elif isinstance(data, list):
        for p in data:
            if not isinstance(p, dict):
                return None
            name = p.get("name", p.get("product_name"))
            names = _option_names(p.get("options"))
            if not isinstance(name, str) or not name.strip() or names is None:
                return None
            products[name] = names
    else:
        return None
    return products or None


def load_catalog(path: str | os.PathLike | None = None) -> Catalog | None:
    """JSON 카탈로그를 읽는다. path가 없으면 DIRECTON_CATALOG_PATH → mock_data.json. 실패하면 None.

    형식: {"products": [{"name": ..., "options": [{"name": ...} | "..."]}]}, 최상위 리스트,
    또는 {"products": {"상품명": [옵션...]}}.
    """
    if path is None:
        path = data_path()
    if path is None:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    products = _parse(data)
    return Catalog(products) if products else None


_cache: dict[str, Catalog | None] = {}


def catalog_enabled() -> bool:
    return os.environ.get("DIRECTON_CATALOG_NORMALIZE", "1").strip().lower() not in ("0", "false", "off", "no")


def default_catalog() -> Catalog | None:
    """기본 카탈로그(지연 로드, 경로별 캐시). DIRECTON_CATALOG_NORMALIZE=0이면 None."""
    if not catalog_enabled():
        return None
    dp = data_path()
    if dp is None:
        return None
    key = str(dp)
    if key not in _cache:
        _cache[key] = load_catalog(key)
    return _cache[key]


def reset_default_catalog() -> None:
    _cache.clear()
    _ids_cache.clear()


# ── 번호 자릿수(AG-T17) ───────────────────────────────

DEFAULT_ORDER_LENGTHS = frozenset({15, 16})   # 공식 주문번호 YYYYMMDD-NNNNNNN(15), 일부 16자리 숫자(AG-T38)
DEFAULT_PHONE_LENGTHS = frozenset({11})


def _env_paths() -> list[str]:
    return [p for p in (os.environ.get("DIRECTON_CATALOG_PATH"), os.environ.get("DIRECTON_MOCK_DATA")) if p]


def data_path() -> Path | None:
    """카탈로그·번호 형식을 읽을 데이터 파일: DIRECTON_CATALOG_PATH → DIRECTON_MOCK_DATA. 없으면 None(공식 기본값)."""
    env = _env_paths()
    return Path(env[0]) if env else DEFAULT_CATALOG_PATH


def _is_order_key(k: str) -> bool:
    k = k.lower()
    return k in ("order_id", "orderid", "order_no", "order_number") or k.endswith("order_id")


def _is_phone_key(k: str) -> bool:
    return "phone" in k.lower()


def _collect_ids(data: Any) -> dict[str, list[str]]:
    """데이터 전체를 돌며 주문번호·전화번호 값을 모은다(공백·하이픈 제거, 처음 나온 순서)."""
    found: dict[str, list[str]] = {"order": [], "phone": []}

    def add(kind: str, v: Any) -> None:
        if isinstance(v, bool) or not isinstance(v, (str, int)):
            return
        s = re.sub(r"[\s\-]", "", str(v))
        if s and s not in found[kind]:
            found[kind].append(s)

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(k, str) and _is_order_key(k):
                    add("order", v)
                elif isinstance(k, str) and _is_phone_key(k):
                    add("phone", v)
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(data)
    return found


_ids_cache: dict[tuple[str, float], dict[str, list[str]]] = {}


def _ids(path: str | os.PathLike) -> dict[str, list[str]]:
    p = Path(path)
    try:
        key = (str(p.resolve()), p.stat().st_mtime)
    except OSError:
        return {"order": [], "phone": []}
    if key not in _ids_cache:
        try:
            _ids_cache[key] = _collect_ids(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            _ids_cache[key] = {"order": [], "phone": []}
    return _ids_cache[key]


def _values(kind: str, path: str | os.PathLike | None) -> list[str]:
    cands = [path] if path is not None else (_env_paths() or ([DEFAULT_CATALOG_PATH] if DEFAULT_CATALOG_PATH else []))
    for c in cands:
        vals = _ids(c)[kind]
        if vals:
            return vals
    return []


def order_id_lengths(path: str | os.PathLike | None = None,
                     default: frozenset[int] = DEFAULT_ORDER_LENGTHS) -> frozenset[int]:
    """데이터에 있는 주문번호 길이들. path가 없으면 DIRECTON_CATALOG_PATH → DIRECTON_MOCK_DATA → 기본 데이터.
    주문번호를 찾지 못하면 default(공식 15·16자리). 응답기·프롬프트의 자릿수 규칙은 이 값을 쓴다."""
    vals = _values("order", path)
    return frozenset(len(v) for v in vals) or default


def phone_lengths(path: str | os.PathLike | None = None,
                  default: frozenset[int] = DEFAULT_PHONE_LENGTHS) -> frozenset[int]:
    """데이터에 있는 전화번호 길이들(모르면 default, 기존 가정 11자리)."""
    vals = _values("phone", path)
    return frozenset(len(v) for v in vals) or default


def order_id_example(path: str | os.PathLike | None = None) -> str | None:
    """가장 흔한 길이의 첫 주문번호(프롬프트 되읽기 예시용). 없으면 None."""
    vals = _values("order", path)
    if not vals:
        return None
    lens = [len(v) for v in vals]
    common = max(set(lens), key=lambda n: (lens.count(n), n))
    return next(v for v in vals if len(v) == common)
