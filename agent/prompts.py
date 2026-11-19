"""시스템 프롬프트. 고정부(앞, 프롬프트 캐시 적중) + 세션 가변부(뒤)."""
from __future__ import annotations

from pathlib import Path

from agent.core.types import SessionInfo

FIXED = """너는 온라인 쇼핑몰 고객센터의 전화 상담원이다. 고객의 말은 음성 인식 결과라서 오탈자나 띄어쓰기 오류가 있을 수 있다. 뜻을 헤아려 응대한다.

## 음성 응답 규칙
- 답은 소리로 읽힌다. 한 번에 2~3문장, 짧고 정중한 존댓말로 말한다.
- 마크다운, 목록 기호, 이모지, 괄호 설명을 쓰지 않는다.
- 숫자는 듣기 쉽게 말한다. 금액은 "만 오천 원"처럼, 날짜는 "11월 21일"처럼, 주문번호·전화번호는 필요할 때만 말한다.
- 모르는 내용은 지어내지 않는다. 조회가 필요하면 함수를 호출하고 그 결과로만 답한다.
- 함수 인자는 정해진 형식을 지킨다. 전화번호는 하이픈 없는 숫자 11자리(예: 01012345678), 주문번호는 숫자만(예: 2611150231). 고객이 한국어로 읽은 숫자("공일공", "이육일일")는 숫자로 바꿔 넣는다.
- 취소·반품·교환·주소 변경처럼 되돌리기 어려운 처리는 고객이 요청했을 때만 한다.
- 필요한 정보가 없으면 한 번에 하나만 짧게 되묻는다."""

MANUAL_HEADER = "## 응대 매뉴얼"


def load_manual(path: str | Path | None) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        return p.read_text(encoding="utf-8").strip() if p.exists() else ""
    except OSError:
        return ""


def _history_lines(history: list[dict]) -> list[str]:
    out = []
    for h in history or []:
        if not isinstance(h, dict):
            out.append(f"- {h}")
            continue
        role = h.get("role", "")
        who = "상담원" if role in ("agent", "assistant", "bot") else "고객"
        text = h.get("text") or h.get("content") or ""
        out.append(f"- {who}: {text}")
    return out


def build_system(info: SessionInfo, opener: str, manual_path: str | Path | None = "cases/manual.md",
                 manual: str | None = None) -> str:
    manual_text = manual if manual is not None else load_manual(manual_path)
    parts = [FIXED]
    if manual_text:
        parts.append(f"{MANUAL_HEADER}\n{manual_text}")

    var = ["## 이번 통화"]
    if info.now:
        var.append(f"- 현재 시각: {info.now}")
    caller = info.caller or {}
    known = {k: v for k, v in caller.items() if v}
    if known:
        labels = {"name": "이름", "phone": "전화번호", "email": "이메일"}
        desc = ", ".join(f"{labels.get(k, k)} {v}" for k, v in known.items())
        var.append(f"- 발신 고객 정보: {desc}. 이 정보는 이미 확인된 것이니 고객에게 다시 묻지 말고 그대로 쓴다.")
    else:
        var.append("- 발신 고객 정보: 없음. 조회에 필요한 정보만 물어본다.")
    hist = _history_lines(info.history)
    if hist:
        var.append("- 이전 대화:")
        var.extend("  " + line for line in hist)
    if opener:
        var.append(f"- 너는 이미 '{opener}'라고 말했다. 같은 말을 반복하지 말고 그 뒤에 이어서 말하라.")
    parts.append("\n".join(var))
    return "\n\n".join(parts)
