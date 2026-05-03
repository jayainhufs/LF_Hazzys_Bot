"""
qa_eval.py
==========
QA 품질 평가용 placeholder.

Future:
- answer groundedness (LLM-judge)
- source coverage
- 사용자 피드백 (좋음/나쁨) 누적 기반 통계
"""
from __future__ import annotations

from typing import Iterable

from src.schemas import RetrievedChunk


def source_coverage(answer: str, chunks: Iterable[RetrievedChunk]) -> float:
    """
    답변 안에 chunk 의 file_name 이 등장한 비율.
    매우 거친 휴리스틱이지만 빠르게 직관 확인용으로 사용.
    """
    chunks = list(chunks)
    if not chunks:
        return 0.0
    answer_lc = (answer or "").lower()
    hit = sum(1 for c in chunks if (c.file_name or "").lower() in answer_lc)
    return hit / len(chunks)
