"""AG-T12 턴 보정 데이터셋 생성: 원문 → 음성(합성·통화 환경) → 라벨 → STT.

단계(각 단계는 결과를 재사용하므로 다시 돌려도 바뀐 것만 만든다):
  sources : 원문 생성 → data/turnset/src/utterances.jsonl
            직접 쓴 템플릿 + (선택) Gemini 초안(data/turnset/src/gemini_drafts.json 캐시). 규칙 검사로 거른다.
  audio   : 분할·음성·환경 배정 → 세그먼트 합성(캐시) → 피치 변형 → 조립 → eval.augment 통화 환경
            → data/turnset/audio/<uid>.wav + data/turnset/labels.jsonl(쉼 시작·최종 끝, 깨끗한 트랙 기준)
  stt     : 각 지점(쉼 시작·최종 끝)+200ms까지의 오디오를 실제 STT로 전사 → labels.jsonl의 text_stt
            (캐시 data/turnset/src/stt_cache.jsonl, test 먼저. 한도에 걸리면 멈추고 나머지는 text_stt=null)

  uv run python scripts/gen_turnset.py sources [--gemini]
  uv run python scripts/gen_turnset.py audio
  uv run python scripts/gen_turnset.py stt [--limit N]

음성(원 음성 × 피치 = 가상 화자): BASE_VOICES × PITCHES. Gemini TTS는 무료 한도(모델당 하루 10회) 때문에 지금은
빼 두었다 — 결제 후 BASE_VOICES에 "gemini:Puck" 등을 더하고(또는 --voices) audio·stt를 다시 돌리면
기존 합성·전사 캐시는 그대로 쓰고 바뀐 발화만 다시 만든다.
분할: 원문 템플릿 fold(해시 % 5 == 0 → test)와 가상 화자(TEST_VOICES)를 맞춰 배정한다. 원 음성 하나(InJoon)는
test 전용이다.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, NamedTuple, Sequence

import numpy as np
import soxr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import augment as aug  # noqa: E402

SR = 16000
TURNSET = ROOT / "data" / "turnset"
SRC_DIR = TURNSET / "src"
UTTS = SRC_DIR / "utterances.jsonl"
DRAFTS = SRC_DIR / "gemini_drafts.json"
STT_CACHE = SRC_DIR / "stt_cache.jsonl"
LABELS = TURNSET / "labels.jsonl"
AUDIO_DIR = TURNSET / "audio"
SYNTH_CACHE = AUDIO_DIR / "cache"
TESTSET = ROOT / "data" / "testset" / "cases.jsonl"
RENDER_VERSION = 1
STT_TAIL_S = 0.2              # 세션은 침묵 200ms에 finalize → 그때까지의 오디오
SEED = 20261119

# ───────────────────────── 음성 설정 ─────────────────────────
BASE_VOICES: tuple[str, ...] = (
    "edge:ko-KR-SunHiNeural",
    "edge:ko-KR-InJoonNeural",
    "edge:ko-KR-HyunsuMultilingualNeural",
    "sapi:Heami",                      # Windows 내장 SAPI 한국어(Microsoft Heami Desktop)
    # 결제 후 추가: "gemini:Puck", "gemini:Charon", "gemini:Fenrir", "gemini:Aoede", "gemini:Leda", ...
)
PITCHES: tuple[int, ...] = (-3, 0, 3)   # 반음
# test 전용 추가 음성(피치 변형 없음). 예: ("gemini:Puck", "gemini:Leda").
# 2026-11-19 시도: Gemini TTS Tier 1 한도가 모델당 분당 10회이고 메인 세션 테스트셋 합성과 겹쳐 429 → 이번엔 비워 둠.
TEST_ONLY_VOICES: tuple[str, ...] = ()
# test 전용 가상 화자: InJoon 전부 + Heami +3(여성 화자 포함)
TEST_VOICES: frozenset[str] = frozenset({
    "edge:ko-KR-InJoonNeural|-3", "edge:ko-KR-InJoonNeural", "edge:ko-KR-InJoonNeural|+3", "sapi:Heami|+3",
    *TEST_ONLY_VOICES,
})
TEST_FOLD = 0
N_FOLDS = 5

# 3.3절 통화 환경 축(가중치)
ENV_AXES = {
    "channel": [("wideband", 3), ("phone", 3.5), ("phone_ulaw", 3.5)],
    "noise": [("none", 3), ("white", 1.5), ("pink", 1.5), ("babble", 2.5), ("hum", 1.5)],
    "snr_db": [(30, 2.5), (20, 3.5), (10, 2.5), (5, 1.5)],
    "gain_db": [(0, 6), (-12, 2.5), (6, 1.5)],
    "packet_loss": [(0.0, 8.5), (0.02, 1.5)],
    "speed": [(0.85, 2), (1.0, 6), (1.2, 2)],
}


def voice_id(base: str, pitch: int) -> str:
    return base if pitch == 0 else f"{base}|{pitch:+d}"


# ───────────────────────── 텍스트 규칙 ─────────────────────────
DIGIT_SYL = "공영일이삼사오육칠팔구"
_PUNCT = "…·.,!?~ \"'"
CONNECTIVE_ENDS = ("는데", "은데", "인데", "던데", "건데", "ㄴ데", "서", "고", "면", "니까", "지만", "다가", "려고", "는지",
                   "거나", "도록", "길래", "든지")
PARTICLE_ENDS = ("에서", "으로", "한테", "에게", "부터", "까지", "이랑", "랑", "가", "이", "를", "을", "은", "는", "에",
                 "로", "도", "만", "의")
FILLER_WORDS = frozenset({"어", "음", "그", "저", "저기", "아", "에", "흠", "그러니까", "그니까", "그게", "뭐지", "뭐더라",
                          "있잖아요", "있잖아", "어어", "음음", "저기요", "그러니깐", "뭐냐", "이게"})
FINAL_CHARS = set("요다까죠야지줘해돼와네래게어아자")
SHORT_FINALS = frozenset({"네", "예", "넵", "응", "아니요", "아뇨", "아니", "맞아요", "맞아", "그래요", "좋아요",
                          "괜찮아요", "됐어요", "알겠습니다", "네네", "예예", "어어", "아니에요"})


# 자기 정정·잠깐 멈춤 표현(블록 2, AG-T15b). 쉼 앞 세그먼트가 이것으로 끝나면 cut="self_correction"
SELF_CORR_MARKERS = ("아니", "아 아니", "아니다", "아 아니다", "아니야", "아 아니야", "아니고요", "아니고", "아니에요",
                     "다시 할게요", "다시 말할게요", "잘못 말했어요", "잘못 말했다", "틀렸다", "아 틀렸다",
                     "그게 아니라", "아니 그게 아니라", "잠깐만요", "아 잠깐만요", "잠시만요", "찾아볼게요",
                     "번호 찾아볼게요", "확인해 볼게요")


def has_arabic(text: str) -> bool:
    return bool(re.search(r"[0-9]", text))


def _last_word(text: str) -> str:
    t = text.strip().strip(_PUNCT)
    return t.split()[-1] if t.split() else ""


def _jong(ch: str) -> int:
    o = ord(ch) - 0xAC00
    return o % 28 if 0 <= o < 11172 else -1


def check_final(text: str) -> bool:
    """마지막 세그먼트: 완결된 발화로 끝나는가(종결어미·단답·숫자+요)."""
    if has_arabic(text) or not text.strip():
        return False
    t = text.strip().rstrip(_PUNCT)
    if t in SHORT_FINALS:
        return True
    if t and all(c in DIGIT_SYL for c in t.replace(" ", "")) and len(t.replace(" ", "")) >= 3:
        return True                                        # "일오공 이삼일"(숫자를 끝까지 부르고 멈춤)
    w = _last_word(t)
    if not w:
        return False
    if w.endswith("니까"):
        return len(w) >= 3 and _jong(w[-3]) == 17          # 합니까/됩니까
    if any(w.endswith(e) for e in ("는데", "은데", "인데", "지만", "려고")):
        return False
    return w[-1] in FINAL_CHARS


def check_segment(text: str, cut: str) -> bool:
    """쉼 앞 세그먼트의 끝이 라벨 유형과 맞는가."""
    if has_arabic(text) or not text.strip():
        return False
    raw = text.strip()
    w = _last_word(raw)
    if not w:
        return False
    if cut == "trap":
        return check_final(raw)
    if cut == "self_correction":
        t = re.sub(r"\s+", " ", raw.rstrip(_PUNCT)).strip()
        return any(t == m or t.endswith(" " + m) or t.endswith("," + m) for m in SELF_CORR_MARKERS)
    if "?" in raw[-2:]:
        return False
    if cut == "filler":
        return w in FILLER_WORDS
    if w.endswith("요"):
        return False
    if cut == "digits":
        core = re.sub(r"(에|이고|하고|이랑|다음에)$", "", w)
        return len(core) >= 2 and all(c in DIGIT_SYL for c in core)
    if cut == "connective":
        if w in FILLER_WORDS or w.endswith("에서"):
            return False
        return any(w.endswith(e) for e in CONNECTIVE_ENDS)
    if cut == "particle":
        if all(c in DIGIT_SYL for c in re.sub(r"에$", "", w)):
            return False
        return len(w) >= 2 and any(w.endswith(e) for e in PARTICLE_ENDS)
    raise ValueError(cut)


def norm_text(text: str) -> str:
    return re.sub(r"[^가-힣a-zA-Z0-9]", "", text or "")


def testset_texts(path: Path = TESTSET) -> set[str]:
    out: set[str] = set()
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        out.add(norm_text(c.get("customer_utterance", "")))
        for s in c.get("segments", []):
            out.add(norm_text(s["text"]))
    out.discard("")
    return out


# ───────────────────────── 템플릿 ─────────────────────────
# 주제별: 연결어미로 끝나는 앞부분(cn), 조사로 끝나는 앞부분(pt), 마무리(존대 fy / 반말 fb), 완결 문장(cp)
TOPICS: dict[str, dict[str, list[str]]] = {
    "delivery": {
        "cn": ["지난주에 시킨 게 아직 안 와서", "배송 조회를 해 봤는데", "택배가 계속 배송 준비 중이라고 떠서",
               "오늘 도착 예정이라고 했는데", "문자로는 배송 완료라고 왔는데", "제가 집에 없을 때 와서",
               "선물로 보내는 거라서", "급하게 써야 되는 거라서", "출고가 됐다고 들었는데",
               "기사님이 연락이 없으시고", "경비실에도 없고", "주말에 받아야 되면"],
        "pt": ["제가 시킨 운동화가", "어제 주문한 거를", "택배 기사님 연락처를", "배송 예정일이", "송장번호를",
               "그 택배 상자가", "지난 금요일에 산 거는", "배송 기사님한테"],
        "fy": ["언제쯤 받을 수 있을까요?", "지금 어디쯤 있는지 알 수 있나요?", "확인 좀 해 주시겠어요?",
               "언제 도착하는지 알려 주세요.", "오늘 안에 받을 수 있어요?", "배송 상태 좀 봐 주세요.",
               "기사님 연락처 알 수 있을까요?", "빨리 좀 받고 싶어요."],
        "fb": ["언제 와?", "지금 어디야?", "확인 좀 해 줘.", "오늘 오는 거 맞아?", "빨리 좀 보내 줘."],
        "cp": ["배송이 왜 이렇게 늦어요?", "택배가 아직 안 왔어요.", "배송 조회가 안 돼요.",
               "배송 완료로 나오는데 물건이 없어요.", "오늘 출발한 거 맞나요?", "언제 도착하는지 궁금해요.",
               "배송지 앞에 놓고 가 달라고 전해 주세요.", "택배 언제 와?", "아직도 배송 준비 중이야?",
               "주문한 물건이 다른 집으로 간 것 같아요.", "기사님이 전화를 안 받으세요."],
    },
    "cancel": {
        "cn": ["실수로 같은 걸 두 번 주문해서", "다른 데서 더 싸게 팔길래", "생각해 보니까 필요가 없어서",
               "아직 출고 전이면", "결제를 잘못 해서", "옵션을 잘못 골라서", "마음이 바뀌어서",
               "주문하고 나서 보니까", "배송이 너무 오래 걸린다고 해서", "카드를 바꾸려고 하는데"],
        "pt": ["방금 주문한 거를", "어제 결제한 주문을", "두 개 중에 하나는", "그 주문 건은", "취소 버튼이",
               "남편이 시킨 거를", "제 주문 내역에서"],
        "fy": ["취소할 수 있을까요?", "취소해 주세요.", "주문 취소 부탁드려요.", "전부 취소하고 싶어요.",
               "취소 가능한가요?", "취소 처리 좀 해 주시겠어요?", "환불까지 얼마나 걸려요?"],
        "fb": ["취소해 줘.", "취소돼?", "그냥 다 취소할래.", "취소 좀 해 줄래?"],
        "cp": ["주문 취소하고 싶어요.", "방금 주문한 거 취소돼요?", "취소 버튼이 안 눌려요.",
               "취소 신청했는데 처리가 안 됐어요.", "주문 하나만 취소할 수 있나요?", "주문 취소해 줘.",
               "취소하면 쿠폰은 돌려받을 수 있나요?", "결제 취소는 언제 돼요?", "부분 취소도 되나요?"],
    },
    "return": {
        "cn": ["받아 보니까 사진이랑 너무 달라서", "옷을 입어 봤는데", "포장을 뜯었는데", "상자가 찌그러져서 왔고",
               "한 번도 안 썼는데", "색이 화면이랑 달라서", "생각보다 너무 커서", "불량인 것 같아서",
               "반품하면", "택을 뗐는데", "받은 지 일주일 됐는데"],
        "pt": ["반품 신청을", "어제 받은 가방을", "반품 택배는", "반품 배송비가", "제가 산 원피스를",
               "불량 제품은", "회수 기사님이"],
        "fy": ["반품하고 싶어요.", "반품 가능한가요?", "반품 접수해 주세요.", "배송비는 누가 내요?",
               "언제 회수해 가세요?", "반품 방법 좀 알려 주세요.", "환불은 언제 되나요?"],
        "fb": ["반품할래.", "반품돼?", "회수 언제 와?", "반품 접수 좀 해 줘."],
        "cp": ["반품하고 싶어요.", "반품 신청은 어떻게 하나요?", "반품 회수가 아직 안 왔어요.",
               "반품 배송비는 얼마예요?", "단순 변심도 반품되나요?", "반품 접수해 줘.",
               "불량이라서 반품하려고 합니다.", "반품했는데 환불이 안 들어왔어요."],
    },
    "exchange": {
        "cn": ["사이즈가 좀 작아서", "다른 색으로 바꾸고 싶은데", "한 치수 큰 걸로 하려고 하는데",
               "신어 봤는데 발이 아파서", "재고가 있으면", "같은 상품인데", "교환 신청을 했는데",
               "옵션을 잘못 선택해서", "선물 받은 건데"],
        "pt": ["검은색 말고 흰색으로", "이 바지를", "교환 상품은", "두 사이즈 큰 거로", "같은 모델의 다른 색상을",
               "교환 신청서를"],
        "fy": ["교환 가능할까요?", "교환해 주세요.", "교환 신청하고 싶어요.", "재고 있는지 봐 주세요.",
               "언제 다시 보내 주세요?", "교환 배송비가 있나요?"],
        "fb": ["바꿔 줘.", "교환돼?", "큰 걸로 바꿀래.", "재고 있어?"],
        "cp": ["사이즈 교환하고 싶어요.", "다른 색상으로 교환되나요?", "교환 신청은 어디서 해요?",
               "교환 상품이 아직 안 왔어요.", "교환할 때 배송비 내야 돼요?", "교환해 줘.",
               "같은 상품으로 교환 부탁드립니다."],
    },
    "coupon": {
        "cn": ["쿠폰을 쓰려고 했는데", "결제할 때 할인이 안 돼서", "생일 쿠폰을 받았는데", "적립금이 사라져서",
               "이벤트 쿠폰이 있다고 해서", "쿠폰 코드를 넣었는데", "첫 구매 할인을 받으려면"],
        "pt": ["제 쿠폰이", "할인 쿠폰을", "적립금은", "이번 달 쿠폰은", "포인트를"],
        "fy": ["왜 적용이 안 되나요?", "쿠폰 좀 확인해 주세요.", "언제까지 쓸 수 있어요?", "다시 넣어 주실 수 있나요?",
               "제가 가진 쿠폰 알려 주세요."],
        "fb": ["왜 안 돼?", "쿠폰 다시 줘.", "언제까지야?", "내 쿠폰 뭐 있어?"],
        "cp": ["쿠폰이 적용이 안 돼요.", "제 쿠폰 목록 좀 알려 주세요.", "쿠폰 유효기간이 언제까지예요?",
               "적립금은 어디서 확인해요?", "쿠폰 두 장 같이 쓸 수 있나요?", "쿠폰 있어?",
               "할인 쿠폰 다시 받을 수 있을까요?"],
    },
    "callback": {
        "cn": ["지금은 통화가 어려워서", "제가 운전 중이라서", "회의 들어가야 돼서", "상담원이랑 직접 얘기하고 싶은데",
               "점심시간이 지나면", "퇴근하고 나서", "자세히 설명을 들어야 해서"],
        "pt": ["오후 세 시쯤에", "내일 오전에", "제 휴대폰으로", "상담원분이", "저녁 여섯 시 이후에"],
        "fy": ["전화 좀 주세요.", "다시 연락 주실 수 있나요?", "콜백 예약해 주세요.", "상담원 연결해 주세요.",
               "연락 부탁드릴게요."],
        "fb": ["전화 줘.", "다시 연락해 줘.", "상담원 바꿔 줘."],
        "cp": ["상담원 연결해 주세요.", "나중에 다시 전화 주실 수 있나요?", "콜백 신청할게요.",
               "사람이랑 통화하고 싶어요.", "상담원 바꿔 줘.", "내일 오전에 전화 부탁드립니다."],
    },
    "address": {
        "cn": ["이사를 가게 돼서", "주소를 잘못 적어서", "회사로 받으려고 하는데", "동 호수를 빼먹어서",
               "출고 전이면", "부모님 댁으로 보내려고 하는데"],
        "pt": ["배송지를", "새 주소는", "아파트 동 호수가", "받는 사람 이름을", "주문서의 주소를"],
        "fy": ["주소 바꿀 수 있나요?", "배송지 변경해 주세요.", "주소 좀 수정해 주세요.", "지금 변경 가능한가요?"],
        "fb": ["주소 바꿔 줘.", "배송지 바꿀 수 있어?", "주소 고쳐 줘."],
        "cp": ["배송지 변경하고 싶어요.", "주소를 잘못 입력했어요.", "주소 바꿀 수 있어?",
               "이미 출고됐으면 주소 변경 안 되나요?", "받는 분 연락처도 바꿀 수 있나요?"],
    },
}
# 완결 대조군(쉼 없음)
SHORT_COMPLETES = ["네", "예", "아니요", "아뇨", "네 맞아요", "맞습니다", "네 알겠습니다", "아니요 괜찮아요", "그래요",
                   "좋아요", "네네", "됐어요", "응", "아니", "맞아", "괜찮아요", "네 그렇게 해 주세요.", "아니요 그건 아니에요.",
                   "네 감사합니다.", "그걸로 할게요.", "네 그거 맞아요.", "아니요 다른 거예요."]
LONG_COMPLETES = [
    "지난주 화요일에 주문한 원피스가 아직 배송 준비 중으로 나오는데 언제쯤 출고되는지 알 수 있을까요?",
    "아이 선물로 산 장난감이 부서진 채로 와서 교환하거나 환불받고 싶은데 어떻게 하면 되나요?",
    "결제는 카드로 했는데 취소하면 카드사 승인 취소까지 보통 며칠 정도 걸리는지 궁금해요.",
    "쿠폰을 적용해서 결제하려고 했는데 계속 오류가 나서 그냥 결제했거든요 차액 돌려받을 수 있나요?",
    "회사 주소로 배송지를 바꾸고 싶은데 이미 출고가 됐으면 택배사에 직접 연락해야 하는 건가요?",
    "반품 회수 기사님이 오신다고 했는데 이틀째 안 오셔서요 다시 접수해 주실 수 있을까요?",
    "같은 상품을 두 개 주문했는데 하나만 왔어요 나머지 하나는 따로 오는 건지 확인 부탁드려요.",
    "상담원분이랑 직접 통화하고 싶은데 지금 연결이 안 되면 오후에 전화 좀 주실 수 있나요?",
    "사이즈가 생각보다 커서 한 치수 작은 걸로 바꾸고 싶은데 교환 배송비는 제가 내야 하나요?",
    "택배 상자는 멀쩡한데 안에 든 컵이 깨져 있었어요 사진 찍어 뒀는데 어디로 보내면 될까요?",
    "어제 밤에 주문한 거 아직 결제 완료로만 나오는데 오늘 출고되는 거 맞지?",
    "생일 쿠폰이 이번 달 말까지라고 들었는데 제 계정에서는 안 보이거든요 확인해 주세요.",
]
# 숫자: (앞, 쉼, 뒤) 패턴 — {a},{b}는 한글 숫자 낭독으로 채운다
DIGIT_PATTERNS = [
    ("phone_a", "제 번호는 공일공에 {a}", "{b}이요."),
    ("phone_b", "공일공 {a}", "{b}입니다."),
    ("phone_c", "전화번호가 공일공", "{a}에 {b}이에요."),
    ("phone_d", "휴대폰 번호 공일공에 {a}", "{b}요."),
    ("phone_e", "연락처는 공일공 {a}", "{b}로 해 주세요."),
    ("order_a", "주문번호 {o1}", "{o2}이요."),
    ("order_b", "주문번호가 {o1}", "{o2}인데요 확인돼요?"),
    ("order_c", "송장번호는 {o1}", "{o2}이에요."),
    ("order_d", "주문번호 불러 드릴게요 {o1}", "{o2}입니다."),
    ("order_e", "번호가 {o1} {o3}", "{o4}요."),
    ("order_f", "주문번호 {o1}", "{o2} 맞아요?"),
    ("phone_f", "공일공 다시 {a}", "{b} 맞지?"),
]
DIGIT_COMPLETES = [("dc_phone", "공일공 {a} {b}이요."), ("dc_order", "{o1}{o2}이요."),
                   ("dc_short", "{a}요."), ("dc_phone2", "공일공에 {a}에 {b}요"), ("dc_order2", "주문번호는 {o1} {o2}입니다.")]
FILLER_PREFIXES = ["어…", "음…", "음 그러니까", "저기", "그…", "아 그게", "저기요 그", "어어", "음 뭐더라", "그러니까",
                   "아", "저 그게"]
TRAP_SECONDS = {  # 완결처럼 들리는 앞 문장 뒤에 이어지는 요청(존대·반말)
    "fy": ["그리고 주소도 바꿀 수 있나요?", "아 그리고 쿠폰도 같이 봐 주세요.", "근데 언제쯤 받을 수 있어요?",
           "그거 말고 다른 것도 하나 있는데요 반품도 되나요?", "그리고 환불은 언제 돼요?", "혹시 오늘 안에 처리되나요?",
           "아 그리고 상담원 연결도 해 주세요.", "그다음에 교환도 하고 싶어요.", "그런데 배송비는 어떻게 돼요?"],
    "fb": ["그리고 주소도 바꿔 줘.", "근데 언제 와?", "아 그리고 쿠폰도 봐 줘.", "그리고 환불은 언제 돼?"],
}
TRAP_FIRSTS = {
    "delivery": ["배송 조회하려고요.", "택배가 아직 안 왔어요.", "주문한 거 확인하려고요.", "배송 문의드려요."],
    "cancel": ["주문 취소하려고요.", "취소 문의드리려고 전화했어요.", "주문 하나 취소했는데요."],
    "return": ["반품 신청했거든요.", "반품하려고요.", "물건 받았어요."],
    "exchange": ["교환하려고요.", "사이즈가 안 맞아요.", "교환 신청했어요."],
    "coupon": ["쿠폰 문의드려요.", "쿠폰 받았어요.", "결제했어요."],
    "callback": ["상담원이랑 통화하고 싶어요.", "전화 좀 받으려고요."],
    "address": ["주소 변경하려고요.", "이사했어요.", "배송지 확인했어요."],
}
_D = ["공", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]


def kor_digits(rng: random.Random, n: int, zero: str = "공") -> str:
    s = "".join(rng.choice(_D) for _ in range(n))
    return s.replace("공", zero) if zero != "공" else s


def _pause_ms(rng: random.Random) -> int:
    r = rng.random()
    if r < 0.30:
        v = rng.uniform(300, 600)
    elif r < 0.65:
        v = rng.uniform(600, 1000)
    elif r < 0.92:
        v = rng.uniform(1000, 1500)
    else:
        v = rng.uniform(1500, 2000)
    return int(round(v / 10) * 10)


def fold_of(key: str) -> int:
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % N_FOLDS


def _utt(uid_key: str, typ: str, segs: list[tuple[str, str]], template_id: str, register: str, source: str,
         rng: random.Random, fold: int) -> dict:
    out = []
    for i, (text, cut) in enumerate(segs):
        d = {"text": text, "pause_after_ms": 0 if i == len(segs) - 1 else _pause_ms(rng)}
        if i < len(segs) - 1:
            d["cut"] = cut
        out.append(d)
    return {"key": uid_key, "type": typ, "segments": out, "template_id": template_id, "register": register,
            "source": source, "fold": fold}


def _fill_digits(tpl: str, rng: random.Random) -> str:
    zero = rng.choice(["공", "공", "영"])
    vals = {"a": kor_digits(rng, 4, zero), "b": kor_digits(rng, 4, zero), "o1": "이육" + kor_digits(rng, 2, zero),
            "o2": kor_digits(rng, 6, zero), "o3": kor_digits(rng, 3, zero), "o4": kor_digits(rng, 3, zero)}
    return tpl.format(**vals)


def _phrase_lists():
    """(fold별) 구문 목록. 같은 fold끼리만 조합해 train/test 원문이 겹치지 않게 한다."""
    return TOPICS


def template_sources(rng: random.Random, counts: dict[str, int]) -> list[dict]:
    T = _phrase_lists()
    utts: list[dict] = []

    def split_fold(key: str) -> int:
        """원문 텍스트(또는 패턴 id) 단위 fold. 같은 문장은 어느 유형에서든 같은 쪽에 간다."""
        return TEST_FOLD if fold_of(key.split(":", 1)[1] if key[:2] in ("p:", "c:", "t:", "f:", "s:") else key) == TEST_FOLD else 1

    def pick_final(topic: str, fold: int, reg: str) -> str | None:
        pool = [s for s in T[topic][reg] if split_fold("f:" + s) == fold]
        return rng.choice(pool) if pool else None

    # 연결어미·조사: 앞부분 × 같은 fold의 마무리
    for typ, field_ in (("connective", "cn"), ("particle", "pt")):
        cands = []
        for topic, d in T.items():
            for s in d[field_]:
                f = split_fold("p:" + s)
                for reg in ("fy", "fy", "fb"):
                    fin = pick_final(topic, f, reg)
                    if fin:
                        cands.append((topic, s, fin, reg, f))
        rng.shuffle(cands)
        seen = set()
        for topic, s, fin, reg, f in cands:
            if len([u for u in utts if u["type"] == typ]) >= counts[typ]:
                break
            if (s, fin) in seen:
                continue
            seen.add((s, fin))
            utts.append(_utt(f"{typ}:{s}|{fin}", typ, [(s, typ), (fin, "")], f"{field_}:{s}",
                             "banmal" if reg == "fb" else "jondae", "template", rng, f))
    # 두 번 쉼: 간투사 + 연결/조사 + 마무리, 조사 + 숫자 + 숫자
    n2 = 0
    for topic, d in T.items():
        for s in d["cn"][:3] + d["pt"][:2]:
            if n2 >= counts.get("two_pause", 0):
                break
            f = split_fold("p:" + s)
            fin = pick_final(topic, f, "fy")
            if fin is None:
                continue
            fil = rng.choice(FILLER_PREFIXES)
            typ = "connective" if s in d["cn"] else "particle"
            utts.append(_utt(f"2:{fil}|{s}|{fin}", typ, [(fil, "filler"), (s, typ), (fin, "")], f"{('cn' if typ == 'connective' else 'pt')}:{s}",
                             "jondae", "template", rng, f))
            n2 += 1
    # 숫자
    nd = 0
    while nd < counts["digits"]:
        pid, a, b = DIGIT_PATTERNS[nd % len(DIGIT_PATTERNS)]
        f = split_fold("d:" + pid)
        sa, sb = _fill_digits(a, rng), _fill_digits(b, rng)
        if pid.startswith("order") and nd % 3 == 0:          # 조사 + 숫자 두 번 쉼
            digits_only = " ".join(w for w in sa.split() if all(c in DIGIT_SYL for c in w))
            utts.append(_utt(f"digits:{sa}|{sb}|2", "digits", [("주문번호가", "particle"), (digits_only, "digits"), (sb, "")],
                             f"d:{pid}", "jondae", "template", rng, f))
        else:
            utts.append(_utt(f"digits:{sa}|{sb}", "digits", [(sa, "digits"), (sb, "")], f"d:{pid}",
                             "banmal" if sb.endswith(("지?", "야.")) else "jondae", "template", rng, f))
        nd += 1
    # 간투사: 간투사 + 완결 문장
    allcp = [(topic, s) for topic, d in T.items() for s in d["cp"]]
    rng.shuffle(allcp)
    for k in range(counts["filler"]):
        topic, s = allcp[k % len(allcp)]
        fil = FILLER_PREFIXES[k % len(FILLER_PREFIXES)]
        f = split_fold("c:" + s)
        utts.append(_utt(f"filler:{fil}|{s}", "filler", [(fil, "filler"), (s, "")], f"c:{s}",
                         "banmal" if not s.rstrip(".?").endswith(("요", "다", "까")) else "jondae", "template", rng, f))
    # 함정: 완결 문장 + 이어지는 요청
    traps = []
    for topic, firsts in TRAP_FIRSTS.items():
        for s in firsts:
            f = split_fold("t:" + s)
            for reg in ("fy", "fb"):
                for x in TRAP_SECONDS[reg]:
                    if split_fold("s:" + x) == f or f == TEST_FOLD:
                        traps.append((s, x, reg, f))
    rng.shuffle(traps)
    seen = set()
    for s, sec, reg, f in traps:
        if len([u for u in utts if u["type"] == "trap"]) >= counts["trap"]:
            break
        if (s, sec) in seen:
            continue
        seen.add((s, sec))
        utts.append(_utt(f"trap:{s}|{sec}", "trap", [(s, "trap"), (sec, "")], f"t:{s}",
                         "banmal" if reg == "fb" else "jondae", "template", rng, f))
    # 완결 대조군
    comp = ([("short", s) for s in SHORT_COMPLETES] + [("long", s) for s in LONG_COMPLETES]
            + [("cp", s) for _, s in allcp]
            + [("fin", s) for d in T.values() for reg in ("fy", "fb") for s in d[reg]]
            + [("trap1", s) for v in TRAP_FIRSTS.values() for s in v] + [("trap2", s) for v in TRAP_SECONDS.values() for s in v])
    k = 0
    for kind, s in comp:
        f = split_fold("c:" + s)
        utts.append(_utt(f"complete:{s}", "complete", [(s, "")], f"c:{s}", "jondae", "template", rng, f))
        k += 1
    while k < counts["complete"]:
        pid, tpl = DIGIT_COMPLETES[k % len(DIGIT_COMPLETES)]
        s = _fill_digits(tpl, rng)
        if not check_final(s):
            s = s + "요."
        f = split_fold("d:" + pid)
        utts.append(_utt(f"complete:{s}", "complete", [(s, "")], f"d:{pid}", "jondae", "template", rng, f))
        k += 1
    return utts


# 블록 1(나중에 추가, 기존 발화의 uid·환경을 바꾸지 않도록 끝에 붙인다): 숫자를 그룹으로 끊어 읽는 쉼
DIGIT_GROUP_PATTERNS = [
    ("g3_order_a", [("주문번호 {o1}", "digits"), ("{o3}", "digits"), ("{o4}이요.", "")]),
    ("g3_order_b", [("주문번호는 {o1}", "digits"), ("{o3}", "digits"), ("{o4}입니다.", "")]),
    ("g3_order_c", [("{o1}", "digits"), ("{o3}", "digits"), ("{o4}", "")]),
    ("g3_order_d", [("송장번호 {o1}", "digits"), ("{o3}", "digits"), ("{o4}요.", "")]),
    ("g3_phone_a", [("공일공", "digits"), ("{a}", "digits"), ("{b}이요.", "")]),
    ("g3_phone_b", [("제 번호 공일공", "digits"), ("{a}", "digits"), ("{b}", "")]),
    ("g3_phone_c", [("연락처 공일공에", "digits"), ("{a}에", "digits"), ("{b}입니다.", "")]),
    ("g2_order_a", [("주문번호 {o1}", "digits"), ("{o3} {o4}", "")]),
    ("g2_order_b", [("{o1} {o3}", "digits"), ("{o4}", "")]),
    ("g2_phone_a", [("공일공 {a}", "digits"), ("{b}", "")]),
    ("g2_order_c", [("주문번호가 {o1}", "digits"), ("{o3} {o4}이요.", "")]),
    ("g2_phone_b", [("번호는 공일공 {a}", "digits"), ("{b}요.", "")]),
]
N_DIGIT_GROUP = 48


def digit_group_sources(rng: random.Random) -> list[dict]:
    out = []
    for k in range(N_DIGIT_GROUP):
        pid, segs = DIGIT_GROUP_PATTERNS[k % len(DIGIT_GROUP_PATTERNS)]
        filled = [(_fill_digits(t, rng), c) for t, c in segs]
        f = TEST_FOLD if fold_of("d:" + pid) == TEST_FOLD else 1
        u = _utt("dg:" + "|".join(t for t, _ in filled), "digits", filled, f"d:{pid}", "jondae", "template", rng, f)
        u["block"] = 1
        out.append(u)
    return out


# 블록 2(AG-T15b, 끝에 추가): 자기 정정(정정 표현 뒤 쉼)과 숫자 그룹 낭독 변형(그룹 사이 쉼 300~900ms, '에'로 이어 부르기).
# 분할은 패턴 단위로 직접 정한다(test 패턴은 표현이 train과 다르다). 쉼 없는 같은 패턴 발화는 complete 대조군.
SC = "self_correction"
SELF_CORR_PATTERNS: list[tuple[str, bool, list[tuple[str, str]]]] = [
    # (패턴 id, test 여부, [(세그먼트, 쉼 유형)...])
    ("sc_digit_a", False, [("주문번호 {o1} {o3}, 아니", SC), ("{o4} {o3}이요.", "")]),
    ("sc_digit_b", False, [("공일공 {a}에 {b}, 아 아니다", SC), ("{b2}요.", "")]),
    ("sc_cancel", False, [("취소해 주세요. 아 아니다", SC), ("반품으로 해 주세요.", "")]),
    ("sc_time", False, [("내일 오전에, 아니고요", SC), ("모레 오후에 전화 주세요.", "")]),
    ("sc_redo", False, [("주문번호가 {o1} {o3} 아 잠깐만요 다시 할게요", SC), ("{o1} {o4} {o3}이요.", "")]),
    ("sc_size", False, [("사이즈는 이백육십, 아니", SC), ("이백칠십으로 바꿔 주세요.", "")]),
    ("sc_addr", False, [("주소가 부산 해운대구, 아 잘못 말했어요", SC), ("수영구 광안로예요.", "")]),
    ("sc_color", False, [("검은색으로, 아니 그게 아니라", SC), ("흰색으로 교환해 주세요.", "")]),
    ("sc_stall_a", False, [("잠시만요, 주문번호 찾아볼게요", SC), ("{o1} {o3} {o4}이요.", "")]),
    ("sc_stall_b", False, [("아 잠깐만요", SC), ("번호가 공일공 {a} {b}이에요.", "")]),
    ("sc_mistake", False, [("교환해 주세요. 아니에요", SC), ("그냥 환불해 주세요.", "")]),
    ("sc_banmal", False, [("취소해 줘. 아 아니다", SC), ("교환으로 해 줘.", "")]),
    ("sc_date", False, [("어제, 아니", SC), ("그저께 주문한 거 취소돼요?", "")]),
    ("sc_check", False, [("확인해 볼게요", SC), ("주문번호 {o1} {o3} {o4}요.", "")]),
    ("sc_redo2", False, [("공일공 {a} {b} 아니 다시 말할게요", SC), ("공일공 {a} {b2}요.", "")]),
    ("sc_two", False, [("반품하려고 하는데, 아 아니", SC), ("교환으로 바꿔 주시겠어요?", "")]),
    ("sc_wrong", False, [("송장번호 {o1} {o3}, 아 틀렸다", SC), ("{o1} {o4} {o3}이요.", "")]),
    ("sc_3seg", False, [("주문번호 {o1}", "digits"), ("{o3}, 아니", SC), ("{o4} {o3}이요.", "")]),
    ("sc_t_digit", True, [("이육{x} {o3}, 아 아니다", SC), ("{o4} {o3}이요.", "")]),
    ("sc_t_cancel", True, [("취소요. 아 아니다", SC), ("교환으로 해 주세요.", "")]),
    ("sc_t_day", True, [("삼일, 아니", SC), ("사일이요.", "")]),
    ("sc_t_redo", True, [("공일공 {a} {b}, 다시 할게요", SC), ("공일공 {a2} {b}입니다.", "")]),
    ("sc_t_notthat", True, [("배송지 바꿔 주세요. 아니고요", SC), ("그냥 주문 취소로 할게요.", "")]),
    ("sc_t_stall", True, [("잠시만요, 번호 찾아볼게요", SC), ("주문번호 이육{x} {o3} {o4}이에요.", "")]),
    ("sc_t_wrong", True, [("반품, 아 잘못 말했다", SC), ("교환 신청할게요.", "")]),
    ("sc_t_banmal", True, [("환불해 줘. 아 아니야", SC), ("그냥 교환해 줘.", "")]),
    ("sc_t_3seg", True, [("주문번호가", "particle"), ("이육{x} {o3}, 아니", SC), ("{o4} {o3}이요.", "")]),
]
DIGIT_GROUP2_PATTERNS: list[tuple[str, bool, list[tuple[str, str]]]] = [
    ("dg2_p_a", False, [("공일공에", "digits"), ("{a}에", "digits"), ("{b}이요.", "")]),
    ("dg2_p_b", False, [("번호 공일공", "digits"), ("{a}", "digits"), ("{b}요.", "")]),
    ("dg2_o_a", False, [("이육{x}", "digits"), ("{o3}", "digits"), ("{o4}요.", "")]),
    ("dg2_o_b", False, [("주문번호는 이육{x}에", "digits"), ("{o3}에", "digits"), ("{o4}입니다.", "")]),
    ("dg2_p_c", False, [("제 연락처 공일공 {a}", "digits"), ("{b}", "")]),
    ("dg2_o_c", False, [("송장번호가 이육{x} {o3}", "digits"), ("{o4}", "")]),
    ("dg2_t_p", True, [("핸드폰 번호 공일공에", "digits"), ("{a}에", "digits"), ("{b}요.", "")]),
    ("dg2_t_o", True, [("주문한 거 번호가 이육{x}", "digits"), ("{o3}", "digits"), ("{o4}이요.", "")]),
    ("dg2_t_p2", True, [("공일공 {a}에", "digits"), ("{b}", "")]),
]
DIGIT_COMPLETE2 = [("dg2_p_a", "공일공에 {a}에 {b}이요."), ("dg2_o_b", "주문번호는 이육{x}에 {o3}에 {o4}입니다."),
                   ("dg2_t_p", "핸드폰 번호 공일공에 {a}에 {b}요."), ("dg2_t_p2", "공일공 {a}에 {b}요.")]
N_SC_PER = {False: 2, True: 3}           # 패턴당 쉼 있는 발화 수(train / test)
N_DG2_PER = {False: 4, True: 5}


def _fill2(tpl: str, rng: random.Random) -> str:
    zero = rng.choice(["공", "공", "영"])
    vals = {k: kor_digits(rng, 4, zero) for k in ("a", "b", "a2", "b2")}
    vals.update({"o1": "이육" + kor_digits(rng, 2, zero), "x": kor_digits(rng, 2, zero),
                 "o3": kor_digits(rng, 3, zero), "o4": kor_digits(rng, 3, zero)})
    return tpl.format(**vals)


def _utt2(key: str, typ: str, segs: list[tuple[str, str]], pid: str, test: bool, rng: random.Random,
          pause_range: tuple[int, int]) -> dict:
    out = []
    for i, (text, cut) in enumerate(segs):
        last = i == len(segs) - 1
        d = {"text": text, "pause_after_ms": 0 if last else int(round(rng.uniform(*pause_range) / 10) * 10)}
        if not last:
            d["cut"] = cut
        out.append(d)
    return {"key": key, "type": typ, "segments": out, "template_id": f"b2:{pid}",
            "register": "banmal" if out[-1]["text"].rstrip(".?").endswith(("줘", "야")) else "jondae",
            "source": "template", "fold": TEST_FOLD if test else 1, "block": 2}


def block2_sources(rng: random.Random) -> list[dict]:
    out = []
    for pid, test, segs in SELF_CORR_PATTERNS:
        for _ in range(N_SC_PER[test]):
            # 하나의 rng 흐름으로 세그먼트를 채운다(같은 발화 안 번호 일관성은 필요 없음: 정정이므로 다를 수 있다)
            filled = [(_fill2(t, rng), c) for t, c in segs]
            typ = SC                                                   # 블록 2는 정정 쉼이 있으면 유형 self_correction
            out.append(_utt2("sc:" + "|".join(t for t, _ in filled), typ, filled, pid, test, rng, (300, 1500)))
        whole = _fill2(" ".join(t for t, _ in segs), rng)             # 쉼 없이 한 번에(대조군)
        out.append(_utt2("sc:c:" + whole, "complete", [(whole, "")], pid, test, rng, (300, 1500)))
    for pid, test, segs in DIGIT_GROUP2_PATTERNS:
        for _ in range(N_DG2_PER[test]):
            filled = [(_fill2(t, rng), c) for t, c in segs]
            out.append(_utt2("dg2:" + "|".join(t for t, _ in filled), "digits", filled, pid, test, rng, (300, 900)))
    test_pids = {pid for pid, test, _ in DIGIT_GROUP2_PATTERNS if test}
    for pid, tpl in DIGIT_COMPLETE2:
        for _ in range(2):
            s = _fill2(tpl, rng)
            out.append(_utt2("dg2:c:" + s, "complete", [(s, "")], pid, pid in test_pids, rng, (300, 900)))
    return out


# 블록 3(AG-T15b, 끝에 추가): 질문·번호·짧은 답 뒤 쉼 + 추가 정보/요청(round1 말 끝 전 확정 23건 중 9건이 이 꼴:
# "…바꿀 수 있어요? [쉼] 울산 남구…", "이육일일 일삼공 일사칠, [쉼] 배송 완료라고…"). 유형 trap, 쉼 300~1200ms.
TRAP2_PATTERNS: list[tuple[str, bool, list[tuple[str, str]]]] = [
    ("tq_addr", False, [("배송지 회사로 바꿀 수 있어요?", "trap"), ("서울 마포구 월드컵로 삼십이요.", "")]),
    ("tq_time", False, [("오늘 전화 다시 주실 수 있나요?", "trap"), ("오후 다섯 시 이후면 좋겠어요.", "")]),
    ("tq_order", False, [("반품 접수 지금 되나요?", "trap"), ("주문번호 {o1} {o3} {o4}요.", "")]),
    ("tq_size", False, [("재고 있는지 봐 주실래요?", "trap"), ("검은색 이백오십이요.", "")]),
    ("tn_req", False, [("{o1} {o3} {o4}", "trap"), ("이거 배송 언제 와요?", "")]),
    ("tn_req2", False, [("공일공 {a} {b}", "trap"), ("이 번호로 콜백 부탁드려요.", "")]),
    ("ts_short", False, [("라지요.", "trap"), ("라지 사이즈 남아 있나요?", "")]),
    ("ts_short2", False, [("네 맞아요.", "trap"), ("그리고 하나 더 여쭤볼게요.", "")]),
    ("tq_t_addr", True, [("주소 변경 가능할까요?", "trap"), ("대구 수성구 동대구로 백이요.", "")]),
    ("tq_t_cb", True, [("내일 연락 주실 수 있어요?", "trap"), ("오전 열 시쯤이요.", "")]),
    ("tn_t_req", True, [("이육{x} {o3} {o4}", "trap"), ("이 주문 취소돼요?", "")]),
    ("ts_t_short", True, [("스몰이요.", "trap"), ("스몰은 언제 입고돼요?", "")]),
]
N_TRAP2_PER = {False: 3, True: 4}


def block3_sources(rng: random.Random) -> list[dict]:
    out = []
    for pid, test, segs in TRAP2_PATTERNS:
        for _ in range(N_TRAP2_PER[test]):
            filled = [(_fill2(t, rng), c) for t, c in segs]
            u = _utt2("t2:" + "|".join(t for t, _ in filled), "trap", filled, pid, test, rng, (300, 1200))
            u["block"] = 3
            out.append(u)
        whole = _fill2(" ".join(t for t, _ in segs), rng)
        u = _utt2("t2:c:" + whole, "complete", [(whole, "")], pid, test, rng, (300, 1200))
        u["block"] = 3
        out.append(u)
    return out


TEMPLATE_COUNTS = {"connective": 115, "particle": 95, "digits": 95, "filler": 85, "trap": 95, "two_pause": 25,
                   "complete": 215}


def validate_utt(u: dict, test_texts: set[str]) -> str | None:
    segs = u["segments"]
    if not segs:
        return "empty"
    for s in segs:
        if has_arabic(s["text"]):
            return "arabic"
        n = norm_text(s["text"])
        if len(n) > 4 and n in test_texts:
            return "testset_overlap"
    for s in segs[:-1]:
        if not check_segment(s["text"], s.get("cut", "")):
            return f"cut_mismatch:{s.get('cut')}:{s['text']}"
    if not check_final(segs[-1]["text"]):
        return f"final_mismatch:{segs[-1]['text']}"
    return None


def drafts_to_utts(drafts: list[dict], rng: random.Random) -> list[dict]:
    out = []
    for d in drafts:
        segs = d.get("segments") or []
        typ = d.get("type", "")
        if not segs or typ not in ("connective", "particle", "digits", "filler", "trap", "complete"):
            continue
        texts = [str(s).strip() for s in segs if str(s).strip()]
        if typ == "complete":
            texts = [" ".join(texts)]
            pairs = [(texts[0], "")]
        else:
            if len(texts) < 2:
                continue
            texts = texts[:3]
            pairs = [(t, typ if i == len(texts) - 2 else ("filler" if check_segment(t, "filler") else typ))
                     for i, t in enumerate(texts[:-1])] + [(texts[-1], "")]
        key = "g:" + "|".join(texts)
        out.append(_utt(key, typ, pairs, "g:" + hashlib.sha1(key.encode()).hexdigest()[:10],
                        d.get("register", "jondae"), "gemini", rng, TEST_FOLD if fold_of(key) == TEST_FOLD else 1))
    return out


def build_sources(seed: int = SEED, gemini_drafts: list[dict] | None = None) -> list[dict]:
    """원문 목록(결정적). 규칙 검사·테스트셋 중복 검사를 통과한 것만, uid 부여."""
    rng = random.Random(seed)
    test_texts = testset_texts()
    utts = template_sources(rng, TEMPLATE_COUNTS)
    if gemini_drafts:
        utts += drafts_to_utts(gemini_drafts, random.Random(seed + 1))
    utts += digit_group_sources(random.Random(seed + 2))
    utts += block2_sources(random.Random(seed + 3))
    utts += block3_sources(random.Random(seed + 4))
    out, seen = [], set()
    for u in utts:
        k = norm_text(" ".join(s["text"] for s in u["segments"]))
        if u.get("block") in (2, 3):  # 블록 2·3은 같은 문장을 쉼 있음/없음·쉼 길이별로 함께 둔다
            k = (k, tuple(s["pause_after_ms"] for s in u["segments"]))
        if k in seen or validate_utt(u, test_texts):
            continue
        seen.add(k)
        out.append(u)
    for i, u in enumerate(out):
        u["uid"] = f"TS{i + 1:04d}"
    return out


# ───────────────────────── Gemini 초안 ─────────────────────────
GEMINI_TEXT_MODELS = ("gemini-3.8-flash", "gemini-3.5-flash")
DRAFT_PROMPT = """한국어 쇼핑몰 고객센터 전화에서 고객이 하는 말을 만들어라. 주제: 주문·배송·취소·반품·교환·쿠폰·콜백·주소 변경.
유형: {typ}. {desc}
조건: 숫자는 반드시 한글 낭독(예: 공일공, 이육일일 일오공)으로 쓰고 아라비아 숫자를 쓰지 않는다. 존댓말 70%, 반말 30%.
자연스러운 구어체. 서로 다른 문장 {n}개. JSON 배열만 출력: [{{"segments": ["...", "..."], "register": "jondae"|"banmal"}}]"""
DRAFT_DESC = {
    "connective": "첫 세그먼트는 연결어미(-서/-는데/-고/-면/-니까/-지만)로 끝나 말이 이어질 것이 분명하다. 둘째 세그먼트는 요청·질문으로 완결.",
    "particle": "첫 세그먼트는 조사(-가/-를/-은/-는/-에/-로)로 끝나는 명사구(예: '어제 주문한 거를'). 둘째 세그먼트에서 문장을 완결.",
    "digits": "첫 세그먼트는 전화번호(공일공…)나 열 자리 주문번호를 부르다 중간에서 끊긴다(한글 숫자로 끝남). 둘째 세그먼트는 나머지 숫자와 '-이요/-입니다' 등으로 완결.",
    "filler": "첫 세그먼트는 간투사 하나만('어', '음', '저기', '그러니까', '음 그러니까'). 둘째 세그먼트가 실제 요청 문장.",
    "trap": "첫 세그먼트는 그 자체로 완결된 문장(-요./-어요./-거든요.)이지만 뒤에 다른 요청이 이어진다. 둘째 세그먼트는 '그리고/근데/아 그리고'로 시작하는 추가 요청.",
    "complete": "쉼 없이 한 번에 끝나는 완결 발화. segments는 원소 1개. 단답(네/아니요), 숫자만 말하기, 긴 문장을 섞는다.",
}


def fetch_gemini_drafts(n_per_type: int = 40, path: Path = DRAFTS) -> list[dict]:
    """Gemini로 초안 생성(캐시). 한도/오류면 그때까지의 결과만."""
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"drafts": [], "calls": 0}
    have = {d["type"] for d in cache["drafts"]}
    from agent.providers._gemini import make_gemini_client
    client = make_gemini_client()
    for typ, desc in DRAFT_DESC.items():
        if typ in have:
            continue
        for model in GEMINI_TEXT_MODELS:
            try:
                cache["calls"] += 1
                resp = client.models.generate_content(model=model, contents=DRAFT_PROMPT.format(typ=typ, desc=desc, n=n_per_type))
                txt = resp.text or ""
                m = re.search(r"\[.*\]", txt, re.S)
                arr = json.loads(m.group(0)) if m else []
                for d in arr:
                    if isinstance(d, dict) and d.get("segments"):
                        cache["drafts"].append({"type": typ, "segments": d["segments"], "register": d.get("register", "jondae"),
                                                "model": model})
                print(f"gemini drafts {typ}: {len(arr)} ({model})", flush=True)
                break
            except Exception as e:  # noqa: BLE001
                print(f"gemini drafts {typ} {model} 실패: {str(e)[:200]}", flush=True)
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    break
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return cache["drafts"]


# ───────────────────────── 분할·음성·환경 배정 ─────────────────────────
def _balanced(rng: random.Random, axis: list[tuple], n: int) -> list:
    tot = sum(w for _, w in axis)
    vals = []
    for v, w in axis:
        vals += [v] * int(round(n * w / tot))
    while len(vals) < n:
        vals.append(rng.choice(axis)[0])
    vals = vals[:n]
    rng.shuffle(vals)
    return vals


def assign_split_voice_env(utts: list[dict], seed: int = SEED, base_voices: Sequence[str] = BASE_VOICES,
                           pitches: Sequence[int] = PITCHES, test_voices: frozenset[str] = TEST_VOICES,
                           test_only: Sequence[str] = TEST_ONLY_VOICES) -> list[dict]:
    rng = random.Random(seed + 7)
    all_v = [(b, p) for b in base_voices for p in pitches] + [(v, 0) for v in test_only]
    test_v = [v for v in all_v if voice_id(*v) in test_voices]
    train_v = [v for v in all_v if voice_id(*v) not in test_voices]
    rows = []
    cnt = {"train": 0, "test": 0}
    blocks = sorted({u.get("block", 0) for u in utts})
    envs: dict[int, dict] = {}
    for blk in blocks:
        idx = [i for i, u in enumerate(utts) if u.get("block", 0) == blk]
        brng = rng if blk == 0 else random.Random(seed + 7 + 1000 * blk)
        axes = {k: _balanced(brng, ax, len(idx)) for k, ax in ENV_AXES.items()}
        leads = [int(round(200 + 1300 * (j + 0.5) / len(idx))) for j in range(len(idx))]
        brng.shuffle(leads)
        for j, i in enumerate(idx):
            e = {k: axes[k][j] for k in ENV_AXES}
            e["lead_ms"] = leads[j]
            envs[i] = e
    for i, u in enumerate(utts):
        split = "test" if u["fold"] == TEST_FOLD else "train"
        pool = test_v if split == "test" else train_v
        b, p = pool[cnt[split] % len(pool)]
        cnt[split] += 1
        env = envs[i]
        if env["noise"] == "none":
            env["snr_db"] = None
        r = dict(u)
        r.update({"split": split, "base_voice": b, "pitch_st": p, "voice": voice_id(b, p), "env": env,
                  "seed": int(hashlib.sha1(u["uid"].encode()).hexdigest(), 16) % (2 ** 31)})
        r["render_key"] = render_key(r)
        rows.append(r)
    return rows


def render_key(r: dict) -> str:
    d = {"v": RENDER_VERSION, "segments": r["segments"], "voice": r["voice"], "env": r["env"], "seed": r["seed"]}
    return hashlib.sha256(json.dumps(d, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:20]


# ───────────────────────── 합성 ─────────────────────────
def pitch_shift(x: np.ndarray, semitones: float) -> np.ndarray:
    """피치만 바꾸고 길이는 유지(리샘플 + 시간 늘이기)."""
    if not semitones:
        return np.asarray(x, np.float32)
    r = 2 ** (semitones / 12)
    y = soxr.resample(np.asarray(x, np.float64), SR, SR / r, quality="HQ").astype(np.float32)
    y = aug.time_stretch(y, 1 / r)
    return np.asarray(y, np.float32)


class SapiTTS:
    """Windows SAPI(System.Speech) 일괄 합성. PowerShell 한 번에 여러 문장."""

    def __init__(self, voice_map: dict[str, str] | None = None):
        self.voice_map = voice_map or {"Heami": "Microsoft Heami Desktop"}

    def synth_batch(self, name: str, jobs: list[tuple[str, Path]]) -> None:
        if not jobs:
            return
        with tempfile.TemporaryDirectory() as td:
            spec = Path(td) / "jobs.json"
            spec.write_text(json.dumps([{"text": t, "path": str(p)} for t, p in jobs], ensure_ascii=False), encoding="utf-8")
            ps = Path(td) / "run.ps1"
            ps.write_text(
                "Add-Type -AssemblyName System.Speech\n"
                f"$jobs = Get-Content -Raw -Encoding UTF8 '{spec}' | ConvertFrom-Json\n"
                "$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, "
                "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)\n"
                "foreach ($j in $jobs) {\n"
                "  $s = New-Object System.Speech.Synthesis.SpeechSynthesizer\n"
                f"  $s.SelectVoice('{self.voice_map[name]}')\n"
                "  $s.SetOutputToWaveFile($j.path, $fmt)\n"
                "  $s.Speak($j.text)\n"
                "  $s.Dispose()\n"
                "}\n", encoding="utf-8-sig")
            subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps)],
                           check=True, capture_output=True)


class TurnsetSynth:
    """edge/gemini는 eval.synth.Synthesizer(캐시 재사용), sapi는 같은 캐시 폴더에 자체 해시 캐시."""

    def __init__(self, cache_dir: Path = SYNTH_CACHE, edge_concurrency: int = 6):
        from eval.synth import Synthesizer
        self.cache_dir = Path(cache_dir)
        self.syn = Synthesizer(cache_dir=self.cache_dir, max_edge_concurrency=edge_concurrency,
                               max_gemini_concurrency=2)
        self.sapi = SapiTTS()
        self._lock = threading.Lock()

    def sapi_path(self, text: str, name: str) -> Path:
        k = hashlib.sha256(json.dumps({"v": 1, "engine": "sapi", "voice": name, "text": text},
                                      ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:32]
        return self.cache_dir / f"sapi_{k}.pcm"

    def prefetch_sapi(self, name: str, texts: Sequence[str], batch: int = 40) -> None:
        from agent import audio as au
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        todo = sorted({t for t in texts if not self.sapi_path(t, name).exists()})
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            with tempfile.TemporaryDirectory() as td:
                jobs = [(t, Path(td) / f"{k}.wav") for k, t in enumerate(chunk)]
                self.sapi.synth_batch(name, jobs)
                for t, p in jobs:
                    pcm, sr = au.wav_read(p)
                    a = np.frombuffer(pcm, np.int16)
                    if sr != SR:
                        a = np.clip(np.round(soxr.resample(a.astype(np.float32), sr, SR)), -32768, 32767).astype(np.int16)
                    a.astype("<i2").tofile(self.sapi_path(t, name))
            print(f"sapi {name}: {min(i + batch, len(todo))}/{len(todo)}", flush=True)

    def synthesize(self, text: str, base_voice: str) -> np.ndarray:
        engine, _, name = base_voice.partition(":")
        if engine == "sapi":
            p = self.sapi_path(text, name)
            if not p.exists():
                self.prefetch_sapi(name, [text])
            return np.fromfile(p, dtype=np.int16)
        return self.syn.synthesize(text, base_voice)


# ───────────────────────── 조립·라벨 ─────────────────────────
class Rendered(NamedTuple):
    audio: np.ndarray                     # int16 최종(통화 환경 적용)
    clean: np.ndarray                     # float32 깨끗한 트랙(같은 길이)
    pauses: list[tuple[float, float]]     # (쉼 시작 초, 쉼 길이 ms) — 깨끗한 트랙 기준
    end_t: float                          # 최종 음성 끝(초)
    info: dict


def _bounds(x: np.ndarray, lo: int, hi: int) -> tuple[int, int] | None:
    lo, hi = max(0, int(lo)), min(len(x), int(hi))
    if hi - lo < 160:
        return None
    b = aug.speech_bounds(x[lo:hi])
    return None if b is None else (lo + b[0], lo + b[1])


def render(segments: Sequence[dict], synth_fn: Callable[[str], np.ndarray], env: dict, seed: int,
           pitch_st: float = 0.0, babble_pool: Sequence[np.ndarray] | None = None) -> Rendered:
    parts, bounds, pos = [], [], 0
    for i, seg in enumerate(segments):
        t = aug.trim_silence(np.asarray(synth_fn(seg["text"]), np.float32))
        if pitch_st:
            t = aug.trim_silence(pitch_shift(t, pitch_st))
        if len(t) == 0:
            raise RuntimeError(f"세그먼트 음성 없음: {seg['text']!r}")
        bounds.append((pos, pos + len(t)))
        parts.append(t.astype(np.float32))
        pos += len(t)
        if i < len(segments) - 1:
            z = int(round(int(seg.get("pause_after_ms", 0)) * SR / 1000))
            parts.append(np.zeros(z, np.float32))
            pos += z
    clean = np.concatenate(parts)
    e = {k: v for k, v in env.items() if v is not None}
    res = aug.augment(clean, e, seed=seed, babble_pool=babble_pool)
    speed = float(e.get("speed", 1.0))
    lead = int(round(int(e.get("lead_ms", 0)) * SR / 1000))
    m = lambda n: lead + n / speed  # noqa: E731
    pauses = []
    for i in range(len(segments) - 1):
        est_end, est_next = m(bounds[i][1]), m(bounds[i + 1][0])
        mid = (est_end + est_next) / 2
        b1 = _bounds(res.clean, m(bounds[i][0]) - 160, mid)
        b2 = _bounds(res.clean, mid, m(bounds[i + 1][1]) + 160)
        end = b1[1] if b1 else est_end
        nxt = b2[0] if b2 else est_next
        pauses.append((round(end / SR, 4), round((nxt - end) / SR * 1000, 1)))
    end_t = res.speech_end if res.speech_end is not None else m(bounds[-1][1]) / SR
    return Rendered(audio=res.audio, clean=res.clean, pauses=pauses, end_t=round(float(end_t), 4), info=res.info)


def label_row(r: dict, rd: Rendered, duration_s: float) -> dict:
    segs = r["segments"]
    pts = []
    for i, (t, dur) in enumerate(rd.pauses):
        pts.append({"t": t, "kind": "pause", "cut_type": segs[i]["cut"], "dur_ms": int(round(dur)),
                    "text_gt": " ".join(s["text"] for s in segs[: i + 1]), "text_stt": None})
    pts.append({"t": rd.end_t, "kind": "end", "text_gt": " ".join(s["text"] for s in segs), "text_stt": None})
    keep = ("uid", "template_id", "source", "type", "register", "split", "voice", "base_voice", "pitch_st", "env",
            "seed", "segments", "render_key", "fold")
    row = {k: r[k] for k in keep}
    row.update({"duration_s": round(duration_s, 3), "points": pts, "aug_info": rd.info})
    return row


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, path)


def cmd_sources(use_gemini: bool) -> None:
    drafts = []
    if use_gemini:
        drafts = fetch_gemini_drafts()
    elif DRAFTS.exists():
        drafts = json.loads(DRAFTS.read_text(encoding="utf-8"))["drafts"]
    rng = random.Random(SEED + 1)
    test_texts = testset_texts()
    rej: dict[str, int] = {}
    for u in drafts_to_utts(drafts, rng):
        why = validate_utt(u, test_texts)
        if why:
            rej[why.split(":")[0]] = rej.get(why.split(":")[0], 0) + 1
    utts = build_sources(SEED, drafts)
    write_jsonl(UTTS, utts)
    from collections import Counter
    print("utterances", len(utts), dict(Counter(u["type"] for u in utts)), dict(Counter(u["source"] for u in utts)))
    print("gemini drafts", len(drafts), "rejected", rej)


def cmd_audio(workers: int = 8, base_voices: Sequence[str] = BASE_VOICES) -> None:
    from agent import audio as au
    utts = read_jsonl(UTTS)
    rows = assign_split_voice_env(utts, SEED, base_voices=base_voices)
    syn = TurnsetSynth()
    # 1) 합성(세그먼트·원 음성 단위 캐시)
    need: dict[str, set[str]] = {}
    for r in rows:
        for s in r["segments"]:
            need.setdefault(r["base_voice"], set()).add(s["text"])
    for bv, texts in need.items():
        if bv.startswith("sapi:"):
            syn.prefetch_sapi(bv.split(":", 1)[1], sorted(texts))
    jobs = [(t, bv) for bv, ts in need.items() if not bv.startswith("sapi:") for t in sorted(ts)]
    done = [0]

    def synth_one(j):
        syn.synthesize(*j)
        done[0] += 1
        if done[0] % 200 == 0:
            print(f"synth {done[0]}/{len(jobs)} {syn.syn.stats}", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(synth_one, jobs))
    print("synth stats", syn.syn.stats, flush=True)
    # 2) babble 풀: split별 다른 발화의 깨끗한 원음
    pool = {"train": [], "test": []}
    for r in rows:
        if len(pool[r["split"]]) < 60 and len(r["segments"]) == 1:
            pool[r["split"]].append(syn.synthesize(r["segments"][0]["text"], r["base_voice"]).astype(np.float32))
    # 3) 조립·증강·라벨(기존 결과 재사용)
    old = {r["uid"]: r for r in read_jsonl(LABELS)}
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    def render_one(r):
        wav = AUDIO_DIR / f"{r['uid']}.wav"
        o = old.get(r["uid"])
        if o and o.get("render_key") == r["render_key"] and wav.exists():
            return o
        others = [p for p in pool[r["split"]]]
        rd = render(r["segments"], lambda t: syn.synthesize(t, r["base_voice"]), r["env"], r["seed"],
                    pitch_st=r["pitch_st"], babble_pool=others)
        au.wav_write(wav, rd.audio.tobytes())
        return label_row(r, rd, len(rd.audio) / SR)

    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(render_one, rows))
    write_jsonl(LABELS, out)
    from collections import Counter
    print("labels", len(out), dict(Counter(r["split"] for r in out)))


# ───────────────────────── STT ─────────────────────────
class QuotaExhausted(Exception):
    pass


_RETRY = re.compile(r"retryDelay'?\"?\s*:\s*'?\"?(\d+(?:\.\d+)?)s")


def cmd_stt(limit: int | None = None, splits: Sequence[str] = ("test", "train"),
            models: Sequence[str] = ("gemini-3.5-transcribe",), rpm: float = 8.0,
            deadline_min: float | None = None) -> None:
    """모델마다 워커 하나 + 분당 rpm 제한. 분당 한도 429는 retryDelay만큼 쉬고 다시, 일일 한도면 그 모델 중단.
    test 지점을 먼저 처리한다. deadline_min이 지나면 남은 지점은 비워 둔다(시뮬레이터가 정답 텍스트로 대체)."""
    import time
    from agent import audio as au
    from agent.providers.gemini_stt import GeminiBatchSTT
    rows = read_jsonl(LABELS)
    cache = {d["key"]: d for d in read_jsonl(STT_CACHE)}
    jobs = []
    for split in splits:
        for r in rows:
            if r["split"] != split:
                continue
            for k, p in enumerate(r["points"]):
                key = f"{r['uid']}:{r['render_key']}:{k}"
                if key not in cache:
                    jobs.append((key, r["uid"], p["t"]))
    if limit is not None:
        jobs = jobs[:limit]
    print(f"stt jobs {len(jobs)} (cached {len(cache)}) models={list(models)} rpm={rpm}", flush=True)
    stats = {m: {"calls": 0, "ok": 0, "429": 0, "err": 0} for m in models}
    f = STT_CACHE.open("a", encoding="utf-8")
    t_end = time.monotonic() + deadline_min * 60 if deadline_min else None

    async def main():
        queue: asyncio.Queue = asyncio.Queue()
        for j in jobs:
            queue.put_nowait(j)

        async def worker(model: str):
            stt = GeminiBatchSTT(model=model)
            await stt.open_stream()
            gap = 60.0 / rpm
            next_t = 0.0
            while not queue.empty():
                if t_end and time.monotonic() > t_end:
                    return
                key, uid, t = queue.get_nowait()
                now = time.monotonic()
                if now < next_t:
                    await asyncio.sleep(next_t - now)
                next_t = time.monotonic() + gap
                pcm, _ = au.wav_read(AUDIO_DIR / f"{uid}.wav")
                n = int(round((t + STT_TAIL_S) * SR)) * 2
                stats[model]["calls"] += 1
                try:
                    text = await asyncio.wait_for(stt.transcribe(pcm[:n]), 90)
                except Exception as e:  # noqa: BLE001
                    msg = str(e)
                    queue.put_nowait((key, uid, t))
                    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                        stats[model]["429"] += 1
                        if "PerDay" in msg:
                            print(f"{model}: 일일 한도 → 중단", flush=True)
                            return
                        m = _RETRY.search(msg)
                        await asyncio.sleep(float(m.group(1)) + 1 if m else 30)
                    else:
                        stats[model]["err"] += 1
                        if stats[model]["err"] > 20:
                            print(f"{model}: 오류 반복 → 중단 {msg[:200]}", flush=True)
                            return
                        await asyncio.sleep(5)
                    continue
                cache[key] = {"key": key, "text": text, "model": model}
                f.write(json.dumps(cache[key], ensure_ascii=False) + "\n")
                f.flush()
                stats[model]["ok"] += 1
                tot = sum(v["ok"] for v in stats.values())
                if tot % 50 == 0:
                    print(f"stt ok {tot}/{len(jobs)} {stats}", flush=True)

        await asyncio.gather(*(worker(m) for m in models))

    try:
        asyncio.run(main())
    finally:
        f.close()
    apply_stt(rows, cache)
    print("stt stats", stats, flush=True)


def apply_stt(rows: list[dict] | None = None, cache: dict | None = None) -> None:
    rows = rows if rows is not None else read_jsonl(LABELS)
    cache = cache if cache is not None else {d["key"]: d for d in read_jsonl(STT_CACHE)}
    n_have = 0
    for r in rows:
        for k, p in enumerate(r["points"]):
            d = cache.get(f"{r['uid']}:{r['render_key']}:{k}")
            p["text_stt"] = d["text"] if d else None
            p["stt_model"] = d.get("model", "gemini-3.5-transcribe") if d else None
            n_have += d is not None
    write_jsonl(LABELS, rows)
    print("points with stt", n_have, "/", sum(len(r["points"]) for r in rows), flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sources")
    s.add_argument("--gemini", action="store_true")
    a = sub.add_parser("audio")
    a.add_argument("--workers", type=int, default=8)
    a.add_argument("--voices", default=None, help="쉼표 구분 원 음성 목록(기본 BASE_VOICES)")
    t = sub.add_parser("stt")
    t.add_argument("--limit", type=int, default=None)
    t.add_argument("--splits", default="test,train")
    t.add_argument("--models", default="gemini-3.5-transcribe")
    t.add_argument("--rpm", type=float, default=8.0)
    t.add_argument("--deadline-min", type=float, default=None)
    args = ap.parse_args(argv)
    if args.cmd == "sources":
        cmd_sources(args.gemini)
    elif args.cmd == "audio":
        cmd_audio(args.workers, tuple(args.voices.split(",")) if args.voices else BASE_VOICES)
    elif args.cmd == "stt":
        cmd_stt(args.limit, tuple(args.splits.split(",")), tuple(args.models.split(",")), args.rpm,
                args.deadline_min)


if __name__ == "__main__":
    main()
