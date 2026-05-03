"""
retrieval_eval.py
=================
검색 품질 평가용 최소 placeholder.

Future:
- ground-truth chunk 기준 hit rate / MRR / nDCG
- 카테고리별 hit rate
- 임베딩 모델 A/B
"""
from __future__ import annotations

from typing import Iterable, List

from src.schemas import RetrievedChunk


def hit_rate(retrieved: Iterable[RetrievedChunk], ground_truth_chunk_ids: List[str]) -> float:
    """검색 결과에 ground truth chunk_id 가 하나라도 있으면 1, 아니면 0."""
    if not ground_truth_chunk_ids:
        return 0.0
    ids = {c.chunk_id for c in retrieved}
    return 1.0 if any(g in ids for g in ground_truth_chunk_ids) else 0.0


def reciprocal_rank(retrieved: List[RetrievedChunk], ground_truth_chunk_ids: List[str]) -> float:
    for i, c in enumerate(retrieved, start=1):
        if c.chunk_id in ground_truth_chunk_ids:
            return 1.0 / i
    return 0.0
