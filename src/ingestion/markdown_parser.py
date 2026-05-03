"""
markdown_parser.py
==================
.md 파일 파서.

- '#', '##', '###' heading 단위로 ParsedSection 분리.
- heading 텍스트를 section_title 로 사용.
- uploaded_category 가 "slack" 이면 slack_manual_parser 위임.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

from src.logger import get_logger
from src.utils.encoding_utils import read_text_safely

log = get_logger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def parse_markdown(
    path: Path, document_id: str, uploaded_category: str = "misc"
) -> List[Dict[str, Any]]:
    if uploaded_category == "slack":
        from src.ingestion.slack_manual_parser import parse_slack_manual
        return parse_slack_manual(path, document_id)

    try:
        text, used_enc = read_text_safely(path)
    except UnicodeDecodeError:
        log.error("Markdown 디코딩 실패: %s", path.name)
        raise

    text = text.strip()
    if not text:
        return []

    lines = text.splitlines()
    sections: List[Dict[str, Any]] = []
    current_title = path.stem
    current_level = 0
    buffer: List[str] = []

    def flush():
        body = "\n".join(buffer).strip()
        if body:
            sections.append({
                "section_title": current_title,
                "content_type": "guide",
                "content": body,
                "metadata": {
                    "encoding": used_enc,
                    "heading_level": current_level,
                },
            })

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            flush()
            current_level = len(m.group(1))
            current_title = m.group(2).strip() or path.stem
            buffer = []
        else:
            buffer.append(line)
    flush()

    if not sections:
        # heading 이 하나도 없는 경우 전체를 단일 섹션으로
        sections.append({
            "section_title": path.stem,
            "content_type": "guide",
            "content": text,
            "metadata": {"encoding": used_enc, "heading_level": 0},
        })

    log.info("Markdown 파싱 완료: %s -> %d section", path.name, len(sections))
    return sections
