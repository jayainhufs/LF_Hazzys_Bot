"""
ExcelSummary
============
Excel 시트/표 영역 1건에 대한 한국어 업무 요약 메타데이터.
실제 본문(Markdown)은 summary_markdown_path 파일에 저장된다.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


@dataclass
class ExcelSummary:
    summary_id: str
    document_id: str
    file_name: str
    sheet_name: str
    table_range: Optional[str]
    raw_table_hash: str
    summary_text: str
    summary_markdown_path: str
    source_raw_path: str
    created_at: str
    model_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
