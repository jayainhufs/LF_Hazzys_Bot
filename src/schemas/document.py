"""
Document
========
원본 파일 1개에 대응하는 metadata 객체.

source_type:
    slack_manual / guide / kakao / excel / word / txt / markdown / misc
uploaded_category:
    사용자가 업로드 시 선택한 카테고리. (slack / guide / kakao / excel / misc)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict


@dataclass
class Document:
    document_id: str
    source_type: str
    uploaded_category: str
    file_name: str
    file_path: str
    file_hash: str
    title: str = ""
    created_at: str = ""
    ingested_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
