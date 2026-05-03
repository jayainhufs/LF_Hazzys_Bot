"""
deduplicator.py
===============
임베딩 호출 비용을 줄이기 위해 거의 동일한 chunk 를 임베딩 전에 제거.

정책:
- chunk_hash 기준 완전 중복은 곧바로 제거.
- 짧은 안내 문구가 반복되는 경우 (예: 표/시그니처 같은 행) 은 빈도 기준으로 1번만 남긴다.
"""
from __future__ import annotations

from typing import List

from src.logger import get_logger
from src.schemas import Chunk

log = get_logger(__name__)


def deduplicate_chunks(chunks: List[Chunk]) -> List[Chunk]:
    seen: set[str] = set()
    out: List[Chunk] = []
    dropped = 0
    for c in chunks:
        h = c.metadata.get("chunk_hash")
        if not h:
            out.append(c)
            continue
        if h in seen:
            dropped += 1
            continue
        seen.add(h)
        out.append(c)
    if dropped:
        log.info("Deduplicate: %d chunks 제거 (총 %d → %d)", dropped, len(chunks), len(out))
    return out
