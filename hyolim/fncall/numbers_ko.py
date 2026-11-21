"""한국어 받아쓰기 결과에서 전화번호·주문번호를 뽑는다.

STT 가 숫자를 "010 1234 5678" 처럼 줄 때도, "공일공 일이삼사 오육칠팔" 이나
"이천이십육 영구이구 다시 영영삼삼사이팔" 처럼 한글로 줄 때도 같은 숫자열로 바꾼다.

오인식을 줄이기 위한 원칙
- 숫자 덩어리(아라비아 숫자 / 한글 숫자 단어)가 구분자(공백·하이픈·쉼표·마침표·"다시")로만
  이어진 구간을 하나로 묶고, 그 구간의 숫자가 7자리 이상일 때만 번호로 본다.
  → "사이", "오일", "9월28일" 같은 일반 단어·날짜는 건드리지 않는다.
- 결과는 후보 목록으로 돌려준다. 어느 후보가 맞는지는 조회 함수로 검증한다(verify.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

DIGIT = {"공": "0", "영": "0", "빵": "0", "일": "1", "이": "2", "삼": "3", "사": "4",
         "오": "5", "육": "6", "륙": "6", "칠": "7", "팔": "8", "구": "9"}
SMALL_UNIT = {"십": 10, "백": 100, "천": 1000}
BIG_UNIT = {"만": 10_000, "억": 100_000_000}
NUM_CHARS = set(DIGIT) | set(SMALL_UNIT) | set(BIG_UNIT)

# 숫자 단어 뒤에 붙는 조사·어미. 길이가 긴 것부터 벗긴다. ("이" 단독은 숫자 2 라서 벗기지 않는다)
SUFFIXES = sorted(["입니다", "이에요", "이예요", "예요", "에요", "이요", "이고요", "이고", "이랑", "하고",
                   "이구요", "으로", "번이요", "번", "요", "고", "랑", "로", "은", "는", "을", "를", "에"],
                  key=len, reverse=True)
SEP_WORDS = {"다시", "대시", "하이픈", "빼기", "에"}   # 하이픈을 읽은 말

TOKEN_RE = re.compile(r"\d+|[가-힣]+|[\s\-–—,.·/]+|.", re.S)


def sino_to_int(word: str) -> int | None:
    """'이천이십육' → 2026, '삼만삼천사백이십팔' → 33428. 단위가 없는 숫자열은 None."""
    if not any(c in SMALL_UNIT or c in BIG_UNIT for c in word):
        return None
    total, section, num = 0, 0, None
    for c in word:
        if c in DIGIT:
            num = int(DIGIT[c])
        elif c in SMALL_UNIT:
            section += (1 if num is None else num) * SMALL_UNIT[c]
            num = None
        elif c in BIG_UNIT:
            section += num or 0
            total += (section or 1) * BIG_UNIT[c]
            section, num = 0, None
        else:
            return None
    return total + section + (num or 0)


def _hangul_number(word: str) -> str | None:
    """한글 단어가 숫자 단어면 숫자열을, 아니면 None 을 돌려준다."""
    if word in SUFFIXES:
        return None
    for suf in SUFFIXES:
        if word.endswith(suf) and len(word) > len(suf) and all(c in NUM_CHARS for c in word[:-len(suf)]):
            word = word[:-len(suf)]
            break
    if not word or not all(c in NUM_CHARS for c in word):
        return None
    units = [i for i, c in enumerate(word) if c in SMALL_UNIT or c in BIG_UNIT]
    if not units:
        return "".join(DIGIT[c] for c in word)
    # '이천이십육공구이구' 처럼 자릿수 읽기 뒤에 한 자리씩 읽기가 붙은 경우를 나눈다
    cut = units[-1] + 1
    if cut < len(word) and word[cut] in DIGIT:
        cut += 1
    v = sino_to_int(word[:cut])
    if v is None:
        return None
    return str(v) + "".join(DIGIT[c] for c in word[cut:])


def _ambiguous_i(word: str) -> bool:
    """'오사삼이요'처럼 끝의 '이'가 숫자 2인지 '-이요' 어미인지 모호한 경우."""
    for suf in ("이요", "이에요", "이예요", "이고요", "이고", "이구요", "이랑"):
        if word.endswith(suf) and len(word) > len(suf) and all(c in DIGIT for c in word[:-len(suf)]):
            return True
    return False


# 번호를 부르다 고쳐 말할 때 쓰는 말: "이일공팔, 아 아니 이일공구요", "공사삼, 아니다, 공사이요"
CORRECTION_WORDS = {"아", "아니", "아니다", "아니요", "아뇨", "아니고", "아니라", "잠깐", "잠깐만", "잠깐만요", "잠시만요",
                    "정정", "다시", "어", "음", "그러니까", "그게", "죄송", "죄송해요", "미안", "미안해요", "말고"}
CORRECTION_TRIGGER = {"아니", "아니다", "아니요", "아뇨", "아니고", "아니라", "잠깐", "잠깐만", "잠깐만요", "잠시만요", "정정", "말고"}


def _glued_prefix(tok: str, cur: list) -> int:
    """덜 끝난 번호(11자리 미만) 뒤에 숫자 글자 2개 이상이 붙어 있으면 그 길이, 아니면 0."""
    have = sum(len(g) for g, _, _ in cur)
    k = 0
    while k < len(tok) and tok[k] in DIGIT:
        k += 1
    return k if k >= 2 and have < 11 and have + k <= 11 else 0


@dataclass
class NumberSpan:
    digits: str                 # 구간 전체를 이은 숫자열
    groups: list[str]           # 말로 끊어 읽은 덩어리별 숫자열
    start: int                  # 원문 위치
    end: int
    raw: str
    context: str = ""           # 앞쪽 12글자 (송장번호인지 등 판단용)
    alt: str | None = None      # 끝의 '이'를 숫자 2로 읽은 대안 ('오사삼이요' → …5432)
    corrected: bool = False     # 고쳐 말한 번호를 반영했는지


def find_number_spans(text: str, min_digits: int = 7) -> list[NumberSpan]:
    spans: list[NumberSpan] = []
    cur: list[tuple[str, int, int]] = []      # (digits, start, end)
    pending_sep = False
    correcting = 0          # 0: 평소, 1: 고쳐 말하는 말이 나옴(다음 숫자가 마지막 덩어리를 대신함), 2: 대신하는 중
    corrected = False
    ambig_end = False

    def flush():
        nonlocal cur, correcting, corrected, ambig_end
        if cur:
            d = "".join(g for g, _, _ in cur)
            if len(d) >= min_digits:
                s, e = cur[0][1], cur[-1][2]
                sp = NumberSpan(d, [g for g, _, _ in cur], s, e, text[s:e], text[max(0, s - 12):s], corrected=corrected)
                if ambig_end:
                    sp.alt = d + "2"
                spans.append(sp)
        cur = []
        correcting, corrected, ambig_end = 0, False, False

    def add(g, s, e):
        nonlocal correcting, corrected
        if correcting == 1 and cur:
            cur.pop()                       # 고쳐 말한 첫 덩어리가 직전 덩어리를 대신한다
            correcting, corrected = 2, True
        cur.append((g, s, e))

    for m in TOKEN_RE.finditer(text):
        tok = m.group(0)
        if tok.isdigit():
            add(tok, m.start(), m.end())
            ambig_end = False
            pending_sep = False
        elif re.fullmatch(r"[가-힣]+", tok):
            hn = _hangul_number(tok)
            if hn is not None:
                add(hn, m.start(), m.end())
                ambig_end = _ambiguous_i(tok)
                pending_sep = False
            elif cur and tok in CORRECTION_WORDS:
                if tok in CORRECTION_TRIGGER:
                    correcting = 1          # "아 아니", "아니다", "잠깐만요" → 다음 숫자가 고친 번호
                pending_sep = True
            elif tok in SEP_WORDS and cur:
                pending_sep = True
            elif cur and tok[0] in DIGIT and _glued_prefix(tok, cur):
                # '공이사칠성건경' 처럼 숫자 끝에 이름이 붙어 받아쓰인 경우: 앞쪽 숫자 글자만 떼어 낸다
                k = _glued_prefix(tok, cur)
                cur.append(("".join(DIGIT[c] for c in tok[:k]), m.start(), m.start() + k))
                flush()
                pending_sep = False
            else:
                # 숫자 뒤에 바로 붙은 조사("01012345678이요")는 구간을 닫기만 한다
                flush()
                pending_sep = False
        elif re.fullmatch(r"[\s\-–—,.·/]+", tok):
            # 줄바꿈 두 번 이상이면 다른 문단으로 본다
            if tok.count("\n") >= 2:
                flush()
            pending_sep = True
        else:
            flush()
    flush()
    return spans


# ---------------------------------------------------------------- 분류

PHONE_RE = re.compile(r"01[016789]\d{7,8}")
ORDER_RE = re.compile(r"(20\d{6})(\d{7})")
INVOICE_CTX = re.compile(r"송장|운송장|운송|택배번호|등기")
ACCOUNT_CTX = re.compile(r"계좌|은행|예금주|입금자")


@dataclass
class Cand:
    value: str
    kind: str          # "phone" | "order"
    conf: float        # 0~1, 클수록 그대로 믿을 만함
    source: str        # "utterance" | "history" | "caller" ...
    note: str = ""

    def __hash__(self):
        return hash((self.kind, self.value))


def _valid_date(yyyymmdd: str, now: datetime | None) -> bool:
    try:
        d = datetime.strptime(yyyymmdd, "%Y%m%d")
    except ValueError:
        return False
    if now is None:
        return 2020 <= d.year <= 2030
    return now - timedelta(days=400) <= d <= now + timedelta(days=1)


def phone_variants(d: str) -> list[tuple[str, float, str]]:
    """숫자열 하나에서 전화번호 후보를 만든다. (값, 신뢰도, 메모)"""
    out = []
    if re.fullmatch(r"01[016789]\d{8}", d):
        out.append((d, 0.95, "exact"))
    elif re.fullmatch(r"01[016789]\d{7}", d):
        out.append((d, 0.75, "10-digit"))            # 옛 번호(011-xxx-xxxx) 또는 한 자리 누락
        for i in range(3, len(d)):                   # 같은 숫자 두 번 읽은 걸 하나로 들은 경우
            out.append((d[:i + 1] + d[i:], 0.3, "dup-digit"))
    elif re.fullmatch(r"1[016789]\d{8}", d):
        out.append(("0" + d, 0.7, "missing-0"))
    elif re.fullmatch(r"\d{8}", d) and not _valid_date(d, None) and d[:2] not in ("15", "16", "18"):
        out.append(("010" + d, 0.55, "no-010"))                 # 1600-0000 같은 대표번호는 제외
    elif re.fullmatch(r"(050|[1-9]10)\d{8}", d):       # 앞자리 010 을 '050'·'오일공(510)' 으로 받아씀
        out.append(("010" + d[3:], 0.6, "prefix-fix"))
    elif re.fullmatch(r"0\d{9,10}", d) and not d.startswith("01"):   # 0으로 시작: 유선(02·031…)일 수도, 010 을 잘못 들었을 수도
        out.append((d, 0.55, "landline"))
        if len(d) == 11:
            out.append(("010" + d[3:], 0.5, "prefix-010"))   # 휴대폰이면 010 + 뒤 8자리
    elif re.fullmatch(r"01[016789]\d{9}", d):        # 한 자리 더 들림
        for i in range(3, len(d)):
            if d[i] == d[i - 1]:
                out.append((d[:i] + d[i + 1:], 0.45, "extra-dup"))
        if not out:
            out.append((d[:11], 0.3, "trim"))
    return out


def order_variants(d: str, now: datetime | None) -> list[tuple[str, float, str]]:
    out = []
    m = re.fullmatch(r"(20\d{6})(\d{1,8})", d)
    if m and _valid_date(m.group(1), now):
        date, rest = m.groups()
        if len(rest) == 7:
            out.append((f"{date}-{rest}", 0.95, "exact"))
        elif len(rest) < 7:                           # "영영삼삼사이팔" 을 "삼만삼천…"으로 읽어 앞 0 이 빠진 경우
            out.append((f"{date}-{rest.zfill(7)}", 0.6, "zero-pad"))
        elif len(rest) == 8:
            out.append((f"{date}-{rest[1:]}", 0.35, "trim-head"))
            out.append((f"{date}-{rest[:7]}", 0.3, "trim-tail"))
    elif re.fullmatch(r"\d{14,18}", d) and not PHONE_RE.fullmatch(d):
        out.append((d, 0.6, "long-id"))              # 외부몰 주문번호 등 (예: 16자리)
    return out


def extract_ids(text: str, now: datetime | None = None, source: str = "utterance") -> list[Cand]:
    """텍스트에서 전화번호·주문번호 후보를 신뢰도 순으로 뽑는다."""
    cands: dict[tuple[str, str], Cand] = {}

    def add(kind, val, conf, note):
        key = (kind, val)
        if key not in cands or cands[key].conf < conf:
            cands[key] = Cand(val, kind, conf, source, note)

    for sp in find_number_spans(text):
        if INVOICE_CTX.search(sp.context) or ACCOUNT_CTX.search(sp.context):
            continue                                   # 송장번호·계좌번호는 ID 로 쓰지 않는다
        readings = [(sp, 1.0)]
        if sp.alt:                                    # '…삼이요': 끝의 '이'를 2로 읽은 쪽도 후보로 (완성된 번호가 되면 그쪽이 이긴다)
            readings.append((NumberSpan(sp.alt, sp.groups, sp.start, sp.end, sp.raw, sp.context), 1.0))
        for rsp, scale in readings:
            for kind, d in _segment(rsp, now):
                fn = phone_variants if kind == "phone" else (lambda x: order_variants(x, now))
                for v, c, note in fn(d):
                    add(kind, v, c * scale, note)
    return sorted(cands.values(), key=lambda c: -c.conf)


def _segment(sp: NumberSpan, now) -> list[tuple[str, str]]:
    """한 구간 안에 주문번호와 전화번호가 붙어 들어온 경우를 나눈다."""
    d = sp.digits
    out: list[tuple[str, str]] = []
    # 1) 덩어리 경계를 따라 나눠 본다 (말로 끊어 읽은 곳)
    acc, chunks = "", []
    for g in sp.groups:
        acc += g
        chunks.append(acc)
    # 주문번호(15자리)로 시작하면 떼어 낸다
    m = ORDER_RE.match(d)
    if m and _valid_date(m.group(1), now):
        out.append(("order", d[:15]))
        rest = d[15:]
        if rest:
            out.append(("phone", rest))
        return out
    # 전화번호가 구간 중간에 있으면 그 앞뒤를 나눈다
    pm = PHONE_RE.search(d)
    if pm and (len(d) > 12):
        before, phone, after = d[:pm.start()], d[pm.start():pm.start() + 11], d[pm.start() + 11:]
        if before:
            out.append(("order", before))
        out.append(("phone", phone))
        if after:
            out.append(("order", after))
        return out
    if re.fullmatch(r"0?1[016789]\d{7,9}|\d{8}|(050|[1-9]10)\d{8}|0\d{9,10}", d) and not (d.startswith("20") and _valid_date(d[:8], now)):
        out.append(("phone", d))
    else:
        out.append(("order", d))
    return out


def to_spoken_digits(d: str, grouping: tuple[int, ...] | None = None) -> str:
    """숫자열을 TTS 가 한 자리씩 읽도록 한글로 바꾼다. 01012345678 → '공일공, 일이삼사, 오육칠팔'."""
    rd = {"0": "공", "1": "일", "2": "이", "3": "삼", "4": "사", "5": "오", "6": "육", "7": "칠", "8": "팔", "9": "구"}
    d = re.sub(r"\D", "", d)
    if grouping is None:
        grouping = (3, 4, 4) if len(d) == 11 else (3, 3, 4) if len(d) == 10 else (len(d),)
    parts, i = [], 0
    for g in grouping:
        parts.append("".join(rd[c] for c in d[i:i + g]))
        i += g
    if i < len(d):
        parts.append("".join(rd[c] for c in d[i:]))
    return ", ".join(p for p in parts if p)


def parse_now(now: str | None) -> datetime | None:
    if not now:
        return None
    m = re.match(r"(\d{4}-\d{2}-\d{2})(?:\s*\([^)]*\))?\s*(\d{1,2}:\d{2})?", now)
    if not m:
        return None
    return datetime.strptime(f"{m.group(1)} {m.group(2) or '00:00'}", "%Y-%m-%d %H:%M")


# ───────────────────────── 음성 인식 오류 보정 후보 (2차 조회용) ─────────────────────────
# 실제 음성 측정에서 본 오류: 이웃한 두 자리 바뀜(9850→9580), 비슷한 소리 숫자 혼동(공↔오, 일↔이·칠, 육↔구),
# 앞자리 010 을 '오일공'·'050' 으로 받아씀. 한 번에 '한 가지 오류'만 고친 후보를 만든다(너무 먼 번호는 남의 번호일 수 있다).
CONFUSABLE = {"0": "5", "5": "0", "1": "27", "2": "1", "7": "1", "6": "9", "9": "6"}


def _one_edit(d: str, start: int) -> list[str]:
    out = []
    for i in range(start, len(d) - 1):                 # 이웃한 두 자리 바뀜
        if d[i] != d[i + 1]:
            out.append(d[:i] + d[i + 1] + d[i] + d[i + 2:])
    for i in range(start, len(d)):                     # 비슷한 소리 숫자 하나 바뀜
        for c in CONFUSABLE.get(d[i], ""):
            out.append(d[:i] + c + d[i + 1:])
    return out


def stt_variants(value: str, kind: str, limit: int = 40) -> list[str]:
    """받아쓴 번호에서 '한 가지 오류'를 고친 후보들. 조회해서 맞는 기록이 딱 하나일 때만 쓴다."""
    d = re.sub(r"\D", "", value or "")
    out: list[str] = []
    if kind == "phone":
        bases = [d]
        if len(d) == 11 and d[1:3] == "10" and d[0] != "0":
            bases.append("0" + d[1:])                  # '오일공' → 010
        if len(d) == 11 and d.startswith("050"):
            bases.append("010" + d[3:])                # '050' → 010
        if len(d) == 11 and not d.startswith("010"):
            bases.append("010" + d[3:])                # 휴대폰은 010 으로 시작한다: 앞 3자리는 믿지 않고 뒤 8자리만
        if len(d) == 10 and not d.startswith("010"):
            bases.append("010" + d[2:])                # 앞자리가 한 자리 빠진 경우
        if len(d) == 12 and d.startswith("010"):
            bases += [d[:i] + d[i + 1:] for i in range(3, 12)]   # 한 자리 더 들림
        for b in bases:
            if b != d:
                out.append(b)
            out += _one_edit(b, 3) if len(b) == 11 and b.startswith("010") else []
        out = [x for x in out if re.fullmatch(r"01[016789]\d{7,8}", x)]
    elif kind == "order":
        m = re.fullmatch(r"(\d{8})(\d{7})", d)
        if m:
            out = [f"{m.group(1)}-{x}" for x in _one_edit(m.group(2), 0)]
    seen, res = {d}, []
    for x in out:
        k = re.sub(r"\D", "", x)
        if k not in seen:
            seen.add(k)
            res.append(x)
    return res[:limit]
