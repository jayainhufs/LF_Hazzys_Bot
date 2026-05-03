"""retriever 의 reranker 로직만 단위 테스트.

VectorStore / Embedder 는 모킹한다.
"""
from __future__ import annotations

from src.rag.reranker import rerank_simple
from src.schemas import RetrievedChunk


def _rc(score: float, sw: float, category: str = "guide", content_type: str = "text") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"c_{score}_{category}",
        document_id="d",
        file_name="f",
        source_type="guide",
        uploaded_category=category,
        section_title=None,
        content_type=content_type,
        content="dummy",
        score=score,
        final_score=score,
        metadata={"source_weight": sw},
    )


def test_rerank_orders_by_final_score():
    a = _rc(score=0.6, sw=1.0, category="guide", content_type="text")
    b = _rc(score=0.5, sw=1.1, category="excel", content_type="excel_summary")
    out = rerank_simple([a, b], category_boost={"excel": 1.05, "guide": 1.02})
    # b 의 final_score 가 a 보다 커야 함 (sw + content_type boost + category boost)
    assert out[0].chunk_id == b.chunk_id
    assert out[0].final_score >= out[1].final_score
