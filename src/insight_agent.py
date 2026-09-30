"""선택형 LLM 연결부. 핵심 분석은 LLM 없이 실행됩니다."""

from __future__ import annotations
import json
from pathlib import Path


def build_insight_prompt(summary_path: str | Path) -> str:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    return (
        "리테일 분석 담당자로서 제공된 지표만 사용하세요. "
        "근거가 있는 실행 제안 세 가지, 위험 한 가지, 실험 한 가지를 한국어로 작성하세요. "
        "인과관계를 추측해 단정하지 마세요.\n\n지표:\n" + json.dumps(summary, indent=2, ensure_ascii=False)
    )
