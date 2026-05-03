"""
QALog
=====
업무 질문/답변 1건을 저장하는 객체.
storage/qa_logs 폴더에 JSON 파일로 저장된다.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class QALog:
    question: str
    answer: str
    model_name: str
    embedding_provider: str
    created_at: str
    rewritten_query: Optional[str] = None
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
