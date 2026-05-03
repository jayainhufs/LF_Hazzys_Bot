"""
cost_utils.py
=============
정확한 과금 계산이 아닌, 대략적인 API 호출량 모니터링.

용도:
- embedding / generation / Excel summary 호출 수 집계
- prompt 길이, retrieved chunk 수, cache hit 여부 로깅
- 월 ~$20 한도 운영을 보조하기 위한 가벼운 추적기

streamlit 세션과 별개로 프로세스 메모리에서 카운트한다.
필요 시 storage/qa_logs 로 내보낼 수 있다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict


@dataclass
class CostTracker:
    embedding_calls: int = 0
    embedding_chunks: int = 0
    generation_calls: int = 0
    excel_summary_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    last_prompt_chars: int = 0
    last_retrieved_chunks: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def add_embedding(self, n_chunks: int = 1) -> None:
        with self._lock:
            self.embedding_calls += 1
            self.embedding_chunks += int(n_chunks)

    def add_generation(self, prompt_chars: int = 0) -> None:
        with self._lock:
            self.generation_calls += 1
            self.last_prompt_chars = int(prompt_chars)

    def add_excel_summary(self) -> None:
        with self._lock:
            self.excel_summary_calls += 1

    def add_cache_hit(self) -> None:
        with self._lock:
            self.cache_hits += 1

    def add_cache_miss(self) -> None:
        with self._lock:
            self.cache_misses += 1

    def set_retrieved_chunks(self, n: int) -> None:
        with self._lock:
            self.last_retrieved_chunks = int(n)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "embedding_calls": self.embedding_calls,
                "embedding_chunks": self.embedding_chunks,
                "generation_calls": self.generation_calls,
                "excel_summary_calls": self.excel_summary_calls,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "last_prompt_chars": self.last_prompt_chars,
                "last_retrieved_chunks": self.last_retrieved_chunks,
                "extra": dict(self.extra),
            }

    def reset(self) -> None:
        with self._lock:
            self.embedding_calls = 0
            self.embedding_chunks = 0
            self.generation_calls = 0
            self.excel_summary_calls = 0
            self.cache_hits = 0
            self.cache_misses = 0
            self.last_prompt_chars = 0
            self.last_retrieved_chunks = 0
            self.extra.clear()


# 프로세스 단위 글로벌 트래커
tracker = CostTracker()
