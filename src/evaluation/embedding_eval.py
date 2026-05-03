"""
embedding_eval.py
=================
임베딩 모델 비교용 placeholder.

Future:
- 동일 query/document 쌍에 대해 cosine sim 분포 비교
- 한국어 retrieval 품질 차이 측정
"""
from __future__ import annotations

import math
from typing import Iterable, List


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
