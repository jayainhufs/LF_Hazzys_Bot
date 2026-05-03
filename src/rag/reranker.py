"""
reranker.py
===========
1차 MVP rule-based reranker.

final_score = similarity_score * source_weight * category_boost * recency_score

추후 Gemini 기반 LLM reranker / cross-encoder reranker 를 붙일 수 있도록
인터페이스만 미리 정의해 둔다.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.logger import get_logger
from src.schemas import RetrievedChunk

log = get_logger(__name__)


def _recency_score(created_at_iso: Optional[str]) -> float:
    """오래된 chunk 의 가중치를 살짝 깎는다 (0.85~1.0)."""
    if not created_at_iso:
        return 1.0
    try:
        dt = datetime.fromisoformat(created_at_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days = max(0.0, (now - dt).total_seconds() / 86400.0)
    except Exception:  # noqa: BLE001
        return 1.0
    if days <= 7:
        return 1.0
    if days <= 90:
        return 0.97
    if days <= 365:
        return 0.93
    return 0.85


def _content_type_boost(content_type: str) -> float:
    """Excel summary 는 검색 단계에서 약간 더 우대."""
    if content_type == "excel_summary":
        return 1.05
    if content_type == "excel_raw_table":
        return 1.0
    return 1.0


def rerank_simple(
    candidates: List[RetrievedChunk],
    category_boost: Optional[Dict[str, float]] = None,
) -> List[RetrievedChunk]:
    """
    Parameters
    ----------
    candidates : VectorStore.search 결과
    category_boost : uploaded_category 별 부스트

    Returns
    -------
    final_score 내림차순으로 정렬된 동일 객체 리스트.
    """
    cat_boost = category_boost or {}
    for c in candidates:
        sim = float(c.score or 0.0)
        sw = float(c.metadata.get("source_weight") or 0.5)
        cat = c.uploaded_category or c.metadata.get("uploaded_category") or "misc"
        cb = float(cat_boost.get(cat, 1.0))
        ct_b = _content_type_boost(c.content_type)
        rs = _recency_score(c.metadata.get("created_at"))
        c.final_score = max(0.0, sim) * sw * cb * ct_b * rs
    candidates.sort(key=lambda x: x.final_score, reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# 미래 확장용 인터페이스 (현 MVP 미사용)
# ---------------------------------------------------------------------------
def rerank_with_llm(
    query: str, candidates: List[RetrievedChunk], top_k: int = 8
) -> List[RetrievedChunk]:
    """
    TODO: Gemini 또는 cross-encoder 를 활용한 reranker.
    현재 MVP 에서는 호출하지 않는다.
    """
    log.info("rerank_with_llm 은 아직 구현되지 않았습니다. rerank_simple 결과를 반환합니다.")
    candidates.sort(key=lambda x: x.final_score, reverse=True)
    return candidates[:top_k]
