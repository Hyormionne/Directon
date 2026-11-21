"""공식 브랜드 자료(scenarios.md·information.json·style.json) → 프롬프트용 매뉴얼·말투 텍스트.

agent.prompts.build_system이 DIRECTON_MANUAL·DIRECTON_STYLE 파일을 고정부의 `## 응대 매뉴얼`·`## 말투 가이드` 절로 넣는다.
그 절 안에서 구조가 섞이지 않게 원문 제목은 `###` 이하로 낮춘다. 출력은 official/(gitignore, 커밋하지 않음).

사용:
  uv run python scripts/build_manual.py [--brand <data/brand 폴더>] [--out official]
  export DIRECTON_MANUAL=official/manual.md DIRECTON_STYLE=official/style.md
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRAND = ROOT.parent / "02_대회자료" / "directon-2026-voice-cs" / "data" / "brand"
OUT = ROOT / "official"


def demote(md: str, levels: int = 2) -> str:
    """마크다운 제목을 levels 단계 낮춘다(# → ###)."""
    return re.sub(r"(?m)^(#+) ", lambda m: "#" * (len(m.group(1)) + levels) + " ", md)


def build_manual(brand: Path) -> str:
    scenarios = demote((brand / "scenarios.md").read_text(encoding="utf-8").strip())
    info = json.loads((brand / "information.json").read_text(encoding="utf-8"))
    lines = [scenarios, "", "### 하루온 브랜드 정보",
             "아래 항목이 정책·사실의 근거다. 항목에 없는 내용은 지어내지 않는다.", ""]
    for item in info:
        tag = item["tag"].strip().strip("[]").strip()
        lines += [f"#### {tag}", item["content"].strip(), ""]
    return "\n".join(lines).strip() + "\n"


def build_style(brand: Path) -> str:
    style = json.loads((brand / "style.json").read_text(encoding="utf-8"))
    return f"- 첫 인사: \"{style['greeting']}\"\n{style['style'].strip()}\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", type=Path, default=BRAND)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "manual.md").write_text(build_manual(a.brand), encoding="utf-8")
    (a.out / "style.md").write_text(build_style(a.brand), encoding="utf-8")
    print(f"wrote {a.out / 'manual.md'}, {a.out / 'style.md'}")


if __name__ == "__main__":
    main()
