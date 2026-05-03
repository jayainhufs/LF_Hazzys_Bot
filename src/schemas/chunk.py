"""
Chunk / RetrievedChunk
======================
임베딩/검색의 최소 단위.

- Chunk           : 색인 시 만들어 ChromaDB에 적재되는 객체
- RetrievedChunk  : 검색 결과로 반환되는 객체 (score 포함)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    source_type: str
    uploaded_category: str
    file_name: str
    content: str
    clean_content: str
    embedding_text: str
    parent_chunk_id: Optional[str] = None
    section_title: Optional[str] = None
    content_type: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    file_name: str
    source_type: str
    uploaded_category: str
    content_type: str
    content: str
    score: float
    final_score: float
    parent_chunk_id: Optional[str] = None
    section_title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
