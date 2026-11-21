"""LLM 함수 호출 인자 정규화 (AG-T08, 표준 라이브러리만).

STT/LLM이 낸 인자(한국어 숫자 낭독, 하이픈 섞인 전화번호, 영문 철자 읽기, 상대 날짜)를
Mock API 정의값과 정확히 맞도록 바꾼다.

정한 규칙(애매한 경우):
- 숫자 낭독: 공/영/빵=0, 일~구(륙=6). 띄어쓰기·쉼표·마침표·하이픈·"+"·"다시"·숫자가 아닌 낱말은
  무시하고 숫자 토큰만 이어 붙인다. 토큰 끝 조사("에", "요", "이요", "이에요", "입니다", "번" 등)는 떼어낸다.
  "이요"·"이에요"는 앞 글자에 받침이 있을 때만 조사로 본다("팔이요"=8, "사이요"=42).
- 자릿값 읽기("천이백삼십사")는 토큰마다 정수로 바꾸고, 4자리 미만이어도 앞을 0으로 채우지 않는다
  ("공일공 이백삼십사 ..." → "010234..."). 토큰 앞의 공/영은 그대로 0으로 붙인다.
- 영문 철자: 알파벳 이름(에이, 비, 씨, 디, 이, ... 더블유, 제트)을 대문자로. "지"=G, "제트/젯/제드"=Z.
  "이"(E/2)와 "오"(O/5)처럼 숫자로도 읽히는 토큰은 앞 토큰의 종류(문자/숫자)를 따르고,
  앞이 없으면 뒤 토큰, 문맥이 전혀 없으면 숫자로 본다.
- 날짜: now 기준. "이번 주"는 달력 주(월~일)라 지난 날일 수도 있다. 요일만 말하면 오늘 이후(오늘 제외)
  가장 가까운 그 요일. "D일"만 말하면 이번 달(지났으면 다음 달), "M월 D일"은 올해(지났으면 내년).
- 시간: 오전/아침/새벽=오전, 오후/저녁/밤/낮/점심=오후. 오전·오후 표시가 없고 1~6시면 오후(영업시간 가정).
- 출력 형식: 날짜 "YYYY-MM-DD", 시간 "HH:MM", 둘 다 있으면 "YYYY-MM-DD HH:MM", 시간만 있으면 "HH:MM".
"""
from __future__ import annotations

import copy
import re
from datetime import date, datetime, timedelta
from typing import Any

__all__ = [
    "spoken_digits",
    "normalize_phone",
    "normalize_order_id",
    "normalize_date",
    "normalize_time",
    "normalize_datetime",
    "normalize_args",
    "digits_for_speech",
    "group_digits",
]

# ------------------------------------------------------------------ 숫자

_DIGIT = {
    "공": "0", "영": "0", "빵": "0",
    "일": "1", "이": "2", "삼": "3", "사": "4", "오": "5",
    "육": "6", "륙": "6", "칠": "7", "팔": "8", "구": "9",
}
_DIGIT.update({str(i): str(i) for i in range(10)})
_UNIT = {"십": 10, "백": 100, "천": 1000}
_MAN = "만"
_ZERO_CHARS = set("공영빵0")
_SPEECH_DIGIT = "공일이삼사오육칠팔구"

_SPLIT_RE = re.compile(r"[\s,.\-+/·~()\[\]:;!?\"']+")

# (조사, 앞 글자 받침 필요 여부)
_SUFFIXES = [
    ("입니다", True), ("이에요", True), ("이예요", True), ("이요", True), ("이고", True),
    ("예요", False), ("에요", False), ("요", False), ("에", False),
    ("은", False), ("는", False), ("을", False), ("를", False), ("번", False),
]


def _has_batchim(ch: str) -> bool:
    if ch.isdigit():
        return True
    code = ord(ch) - 0xAC00
    return 0 <= code <= 11171 and code % 28 != 0


def _strip_suffix(tok: str) -> str:
    changed = True
    while changed and tok:
        changed = False
        for suf, need_batchim in _SUFFIXES:
            if tok.endswith(suf):
                rest = tok[: -len(suf)]
                if rest and need_batchim and not _has_batchim(rest[-1]):
                    continue
                tok = rest
                changed = True
                break
    return tok


def _tokens(text: str) -> list[str]:
    out = []
    for raw in _SPLIT_RE.split(text or ""):
        tok = _strip_suffix(raw)
        if tok:
            out.append(tok)
    return out


def _sino_to_int(s: str) -> int | None:
    """자릿값 한자어 수(천이백삼십사 등) → 정수. 형식이 아니면 None."""
    total = section = 0
    cur: int | None = None
    for ch in s:
        if ch in _UNIT:
            section += (1 if cur is None else cur) * _UNIT[ch]
            cur = None
        elif ch == _MAN:
            section += cur or 0
            total += (section or 1) * 10000
            section, cur = 0, None
        elif ch in _DIGIT and _DIGIT[ch] != "0":
            if cur is not None:
                return None
            cur = int(_DIGIT[ch])
        else:
            return None
    return total + section + (cur or 0)


def _parse_number_token(tok: str) -> str | None:
    """한 토큰을 숫자열로. 숫자 토큰이 아니면 None."""
    if not tok or any(ch not in _DIGIT and ch not in _UNIT and ch != _MAN for ch in tok):
        return None
    if not any(ch in _UNIT or ch == _MAN for ch in tok):
        return "".join(_DIGIT[ch] for ch in tok)
    i = 0
    while i < len(tok) and tok[i] in _ZERO_CHARS:
        i += 1
    n = _sino_to_int(tok[i:])
    if n is None:
        return None
    return "0" * i + str(n)


def spoken_digits(text: str) -> str | None:
    """한국어 숫자 낭독·아라비아 숫자 혼합 텍스트 → 숫자열. 숫자가 없으면 None."""
    parts = [d for d in (_parse_number_token(t) for t in _tokens(text)) if d is not None]
    return "".join(parts) if parts else None


def normalize_phone(text: str) -> str:
    """전화번호 → 숫자열(하이픈 없음). 11자리가 아니어도 숫자만 남긴다. 숫자가 없으면 원문(strip)."""
    digits = spoken_digits(text)
    if not digits:
        return (text or "").strip()
    if digits.startswith("82") and len(digits) in (11, 12):
        digits = "0" + digits[2:]
    return digits


# ------------------------------------------------------------------ 영문 철자

_LETTER_NAMES = {
    "A": ["에이"], "B": ["비", "삐"], "C": ["씨", "시"], "D": ["디"], "E": ["이"],
    "F": ["에프"], "G": ["지", "쥐"], "H": ["에이치", "에치"], "I": ["아이"], "J": ["제이"],
    "K": ["케이"], "L": ["엘"], "M": ["엠"], "N": ["엔"], "O": ["오"], "P": ["피"],
    "Q": ["큐"], "R": ["알"], "S": ["에스"], "T": ["티"], "U": ["유"], "V": ["브이", "뷔"],
    "W": ["더블유", "더블류"], "X": ["엑스"], "Y": ["와이"], "Z": ["제트", "젯", "제드"],
}
_NAME_TO_LETTER = {name: letter for letter, names in _LETTER_NAMES.items() for name in names}
_NAMES_BY_LEN = sorted(_NAME_TO_LETTER, key=len, reverse=True)
_LETTER_SPEECH = {letter: names[0] for letter, names in _LETTER_NAMES.items()}


def _segment_letters(tok: str) -> str | None:
    if not tok:
        return ""
    for name in _NAMES_BY_LEN:
        if tok.startswith(name):
            rest = _segment_letters(tok[len(name):])
            if rest is not None:
                return _NAME_TO_LETTER[name] + rest
    return None


def normalize_order_id(text: str) -> str:
    """주문번호·코드 → 숫자열 또는 대문자 영숫자. 알아볼 것이 없으면 원문(strip)."""
    items: list[tuple[str | None, str, str | None]] = []  # (class, letters, digits)
    for tok in _tokens(text):
        if re.fullmatch(r"[A-Za-z0-9]+", tok):
            items.append(("L" if re.search(r"[A-Za-z]", tok) else "D", tok.upper(), tok))
            continue
        letters = _segment_letters(tok)
        digits = _parse_number_token(tok)
        if letters is not None and digits is not None:
            items.append((None, letters, digits))
        elif letters is not None:
            items.append(("L", letters, None))
        elif digits is not None:
            items.append(("D", digits, digits))
    if not items:
        return (text or "").strip()
    out = []
    for i, (cls, letters, digits) in enumerate(items):
        if cls is None:
            prev = next((c for c, _, _ in reversed(items[:i]) if c), None)
            nxt = next((c for c, _, _ in items[i + 1:] if c), None)
            cls = prev or nxt or "D"
            # 앞에서 정한 종류를 뒤 모호 토큰이 이어받게 기록
            items[i] = (cls, letters, digits)
        out.append(letters if cls == "L" else digits)
    return "".join(out)


# ------------------------------------------------------------------ 날짜·시간

_WEEKDAY = {ch: i for i, ch in enumerate("월화수목금토일")}
_SINO = "일이삼사오육칠팔구십"
_NUM = rf"(?:\d{{1,4}}|[{_SINO}백천]+)"
_NATIVE_HOUR = {
    "열두": 12, "열한": 11, "열": 10, "한": 1, "두": 2, "세": 3, "네": 4,
    "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9,
}
_AM = {"오전", "아침", "새벽"}
_PM = {"오후", "저녁", "밤", "낮", "점심"}
_MERIDIEM_RE = "|".join(sorted(_AM | _PM, key=len, reverse=True))
_HOUR_RE = r"\d{1,2}|" + "|".join(sorted(_NATIVE_HOUR, key=len, reverse=True)) + rf"|[{_SINO}]{{1,3}}"
_TIME_RE = re.compile(
    rf"(?:({_MERIDIEM_RE})\s*|(?<![가-힣\d]))({_HOUR_RE})\s*시(?!간)"
    rf"(?:\s*(반|(\d{{1,2}}|[{_SINO}]+)\s*분))?"
)
_CLOCK_RE = re.compile(rf"(?:({_MERIDIEM_RE})\s*)?\b(\d{{1,2}}):(\d{{2}})\b")


def _num(s: str) -> int | None:
    if s.isdigit():
        return int(s)
    return _sino_to_int(s)


def _to_datetime(now: Any) -> datetime:
    if now is None:
        return datetime.now()
    if isinstance(now, datetime):
        return now
    if isinstance(now, date):
        return datetime(now.year, now.month, now.day)
    return datetime.fromisoformat(str(now).strip())


def _apply_meridiem(hour: int, mer: str | None) -> int:
    if mer in _PM:
        return hour + 12 if hour < 12 else hour
    if mer in _AM:
        return 0 if hour == 12 else hour
    return hour + 12 if 1 <= hour <= 6 else hour


def _find_time(text: str) -> tuple[str, tuple[int, int]] | None:
    if "정오" in text:
        return "12:00", _span(text, "정오")
    if "자정" in text:
        return "00:00", _span(text, "자정")
    m = _CLOCK_RE.search(text)
    if m:
        hour = _apply_meridiem(int(m.group(2)), m.group(1))
        minute = int(m.group(3))
        if hour < 24 and minute < 60:
            return f"{hour:02d}:{minute:02d}", m.span()
    for m in _TIME_RE.finditer(text):
        mer, h, half, mins = m.groups()
        hour = _NATIVE_HOUR.get(h)
        if hour is None:
            hour = _num(h)
        if hour is None or hour > 24:
            continue
        minute = 30 if half == "반" else (_num(mins) if mins else 0)
        if minute is None or minute >= 60:
            continue
        hour = _apply_meridiem(hour, mer) % 24
        return f"{hour:02d}:{minute:02d}", m.span()
    return None


def _span(text: str, word: str) -> tuple[int, int]:
    i = text.index(word)
    return i, i + len(word)


def normalize_time(text: str) -> str | None:
    """시간 표현 → "HH:MM". 없으면 None."""
    found = _find_time(text or "")
    return found[0] if found else None


def _add_month(d: date, months: int, day: int) -> date | None:
    y, m = divmod(d.month - 1 + months, 12)
    try:
        return date(d.year + y, m + 1, day)
    except ValueError:
        return None


def _find_date(text: str, today: date) -> date | None:
    t = text
    m = re.search(r"(\d{4})[-./](\d{1,2})[-./](\d{1,2})", t)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    m = re.search(rf"({_NUM}|일주일)\s*일?\s*(?:뒤|후)", t)
    if m:
        n = 7 if m.group(1) == "일주일" else _num(m.group(1))
        if n is not None:
            return today + timedelta(days=n)
    for word, delta in (("그저께", -2), ("그제", -2), ("어제", -1), ("오늘", 0),
                        ("내일모레", 2), ("내일", 1), ("모레", 2), ("글피", 3)):
        if word in t:
            return today + timedelta(days=delta)
    m = re.search(r"(?:(다다음|다음|이번|지난|저번)\s*주\s*)?([월화수목금토일])요일", t)
    if m:
        target = _WEEKDAY[m.group(2)]
        which = m.group(1)
        if which is None:
            ahead = (target - today.weekday()) % 7 or 7
            return today + timedelta(days=ahead)
        week = {"이번": 0, "다음": 1, "다다음": 2, "지난": -1, "저번": -1}[which]
        monday = today - timedelta(days=today.weekday())
        return monday + timedelta(days=7 * week + target)
    m = re.search(
        rf"(?:(\d{{4}}|[{_SINO}천백]+)\s*년\s*)?(\d{{1,2}}|[{_SINO}]+|시|유)\s*월\s*(\d{{1,2}}|[{_SINO}]+)\s*일", t
    )
    if m:
        month = {"시": 10, "유": 6}.get(m.group(2)) or _num(m.group(2))
        day = _num(m.group(3))
        if month and day:
            year = _num(m.group(1)) if m.group(1) else today.year
            try:
                d = date(year, month, day)
            except ValueError:
                return None
            if not m.group(1) and d < today:
                d = date(year + 1, month, day)
            return d
    m = re.search(rf"(이번|다음|다다음)\s*달\s*(\d{{1,2}}|[{_SINO}]+)\s*일", t)
    if m:
        day = _num(m.group(2))
        if day:
            return _add_month(today, {"이번": 0, "다음": 1, "다다음": 2}[m.group(1)], day)
    m = re.search(rf"(?<![가-힣\d])(\d{{1,2}}|[{_SINO}]+)\s*일(?!요)", t)
    if m:
        day = _num(m.group(1))
        if day and 1 <= day <= 31:
            d = _add_month(today, 0, day)
            if d is None or d < today:
                d = _add_month(today, 1, day)
            return d
    return None


def normalize_date(text: str, now: Any = None) -> str | None:
    """날짜 표현 → "YYYY-MM-DD"(now 기준). 없으면 None."""
    d = _find_date(text or "", _to_datetime(now).date())
    return d.isoformat() if d else None


def normalize_datetime(text: str, now: Any = None) -> str | None:
    """날짜·시간 → "YYYY-MM-DD HH:MM" / "YYYY-MM-DD" / "HH:MM". 아무것도 없으면 None."""
    text = text or ""
    today = _to_datetime(now).date()
    found = _find_time(text)
    rest = text
    if found:
        s, e = found[1]
        rest = text[:s] + " " + text[e:]
    d = _find_date(rest, today)
    if d and found:
        return f"{d.isoformat()} {found[0]}"
    if d:
        return d.isoformat()
    return found[0] if found else None


# ------------------------------------------------------------------ 인자

def _properties(schema: Any) -> dict:
    if not isinstance(schema, dict):
        return {}
    if isinstance(schema.get("function"), dict):
        schema = schema["function"]
    if isinstance(schema.get("parameters"), dict):
        schema = schema["parameters"]
    props = schema.get("properties")
    return props if isinstance(props, dict) else {}


def _kind(key: str) -> str:
    k = key.lower()
    if "phone" in k:
        return "phone"
    if "order_id" in k or k.endswith("_id") or k == "id" or k == "order_number":
        return "id"
    if "date" in k or "time" in k:
        return "datetime"
    return "plain"


def _coerce(value: Any, ptype: Any) -> Any:
    if ptype == "integer" and isinstance(value, str):
        s = value.strip()
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        digits = spoken_digits(s)
        return int(digits) if digits else value
    if ptype == "number" and isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            digits = spoken_digits(value)
            return float(digits) if digits else value
    if ptype == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return value


_PRODUCT_KEYS = ("product_name",)
_OPTION_KEYS = ("option",)
_PRODUCT_TOOLS = frozenset({"get_product_info", "check_stock"})


def _pick_catalog(tool_name: str, props: dict, catalog: Any) -> Any:
    """catalog: 객체면 그대로, False면 끔, None이면 기본 카탈로그(AG-T10d).

    기본 카탈로그는 상품 도구이거나 스키마에 상품명 인자가 있을 때만 쓴다(그 밖은 기존 동작).
    """
    if catalog is False:
        return None
    if catalog is not None:
        return catalog
    if tool_name not in _PRODUCT_TOOLS and not any(k in props for k in _PRODUCT_KEYS):
        return None
    try:
        from agent.catalog import default_catalog
        return default_catalog()
    except Exception:  # noqa: BLE001
        return None


def _apply_catalog(out: dict, cat: Any) -> None:
    """product_name·option을 카탈로그 표기로. 해석 못 하거나 오류면 값을 그대로 둔다."""
    product = None
    for key in _PRODUCT_KEYS:
        v = out.get(key)
        if isinstance(v, str) and v:
            try:
                resolved = cat.resolve_product(v)
            except Exception:  # noqa: BLE001
                resolved = None
            if resolved:
                out[key] = resolved
            product = out[key]
    if product is None:
        return
    for key in _OPTION_KEYS:
        v = out.get(key)
        if isinstance(v, str) and v:
            try:
                resolved = cat.resolve_option(product, v)
            except Exception:  # noqa: BLE001
                resolved = None
            if resolved:
                out[key] = resolved


def normalize_args(tool_name: str, args: dict, schema: Any = None, now: Any = None,
                   catalog: Any = None) -> dict:
    """파라미터 이름 패턴으로 인자를 정규화한 새 dict를 돌려준다(원본은 바꾸지 않는다).

    - *phone*        → 전화번호 숫자열 (정수로 와서 앞 0이 빠졌으면 복원)
    - *order_id*, *_id, id → 숫자열 또는 대문자 영숫자
    - *date*, *time* → normalize_datetime, 못 알아보면 strip한 원문
    - 그 외 문자열   → strip
    - product_name / option → 카탈로그 표기(agent.catalog, AG-T10d). catalog=None이면 기본 카탈로그를
      상품 도구(get_product_info, check_stock)나 스키마에 product_name이 있을 때만 쓰고,
      DIRECTON_CATALOG_NORMALIZE=0이면 끈다. catalog=False면 끔, 객체를 주면 그 카탈로그를 쓴다.
    schema(도구 정의 또는 JSON Schema)의 type이 integer/number면 숫자로, string이면 문자열로 바꾼다.
    """
    props = _properties(schema)
    out: dict = {}
    for key, value in copy.deepcopy(args or {}).items():
        kind = _kind(key)
        is_num = isinstance(value, int) and not isinstance(value, bool)
        if kind == "phone" and is_num:
            s = str(value)
            value = "0" + s if len(s) == 10 and s.startswith("10") else s
        elif isinstance(value, str):
            if kind == "phone":
                value = normalize_phone(value)
            elif kind == "id":
                value = normalize_order_id(value)
            elif kind == "datetime":
                value = normalize_datetime(value, now) or value.strip()
            else:
                value = value.strip()
        ptype = props.get(key, {}).get("type") if isinstance(props.get(key), dict) else None
        out[key] = _coerce(value, ptype)
    cat = _pick_catalog(tool_name, props, catalog)
    if cat is not None:
        _apply_catalog(out, cat)
    return out


# ------------------------------------------------------------------ TTS 역변환

def _read_digits(s: str) -> str:
    return "".join(_SPEECH_DIGIT[int(c)] for c in s)


def _chunk(s: str, sizes: list[int]) -> list[str]:
    out, i = [], 0
    for n in sizes:
        out.append(s[i:i + n])
        i += n
    return [p for p in out if p]


def _chunk4(s: str) -> list[str]:
    return [s[i:i + 4] for i in range(0, len(s), 4)]


def _digit_groups(s: str, kind: str) -> list[str]:
    if kind == "phone" and len(s) == 11:
        return _chunk(s, [3, 4, 4])
    if kind == "phone" and len(s) == 10:
        return _chunk(s, [3, 3, 4])
    if kind == "order" and len(s) == 10:
        return _chunk(s, [4, 3, 3])
    if kind == "order" and len(s) == 15:            # 공식 YYYYMMDD-NNNNNNN: 날짜 4·4 + 뒤 3·4(AG-T39)
        return _chunk(s, [4, 4, 3, 4])
    return _chunk4(s)


def group_digits(s: str, kind: str = "digits") -> str:
    """숫자열을 묶음 단위로 끊어 아라비아 숫자 그대로 ("2611, 150, 231"). 읽기는 TTS에 맡긴다."""
    s = re.sub(r"[\s\-]", "", s or "")
    if not s.isdigit():
        return s
    return ", ".join(_digit_groups(s, kind))


def digits_for_speech(s: str, kind: str = "digits") -> str:
    """번호를 되읽기 쉬운 문자열로. kind: "phone"(3-4-4, 10자리 3-3-4), "order"(10자리 4-3-3), 그 외 4자리씩.

    영문자는 알파벳 이름(디 이 엠 오)으로 읽고, 묶음 사이는 ", "로 잇는다.
    """
    s = re.sub(r"[\s\-]", "", s or "").upper()
    if not s:
        return ""
    if s.isdigit():
        return ", ".join(_read_digits(g) for g in _digit_groups(s, kind))
    parts = []
    for run in re.findall(r"[A-Z]+|\d+|[^A-Z\d]+", s):
        if run.isdigit():
            parts.extend(_read_digits(g) for g in _chunk4(run))
        elif run.isalpha() and run.isascii():
            parts.append(" ".join(_LETTER_SPEECH[c] for c in run))
        else:
            parts.append(run)
    return ", ".join(parts)
