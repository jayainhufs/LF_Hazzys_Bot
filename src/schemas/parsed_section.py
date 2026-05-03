"""
ParsedSection
=============
파서가 문서에서 추출한 의미 단위 (문단/표/대화/시트 등).

content_type:
    text / table / conversation / guide / note /
    excel_summary / excel_raw_table
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


@dataclass
class ParsedSection:
    section_id: str
    document_id: str
    section_title: Optional[str] = None
    content_type: str = "text"
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
