"""
retriever.py
============
질문 → 임베딩 → ChromaDB 검색 → metadata-aware reranking → parent-child 보강.

reranker.py 의 rule-based reranker 를 사용한다.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.config import settings
from src.logger import get_logger
from src.rag.embedder import Embedder, get_default_embedder
from src.rag.reranker import rerank_simple
from src.schemas import RetrievedChunk
from src.storage.vector_store import VectorStore
from src.utils.cost_utils import tracker

log = get_logger(__name__)


CATEGORY_BOOST: Dict[str, float] = {
    "excel": 1.05,
    "guide": 1.02,
    "slack": 1.0,
    "kakao": 0.95,
    "misc": 0.9,
}


class Retriever:
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None,
    ) -> None:
        self.embedder = embedder or get_default_embedder()
        self.vector_store = vector_store or VectorStore()

    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        uploaded_category: Optional[str] = None,
        with_parent_children: bool = True,
        max_children_per_parent: int = 2,
    ) -> List[RetrievedChunk]:
        """
        Parameters
        ----------
        uploaded_category : "all" 또는 None 이면 필터 미적용.
                            'slack' / 'guide' / 'kakao' / 'excel' 만 허용.
        """
        if not query or not query.strip():
            return []

        k = int(top_k or settings.top_k)
        # 후처리에서 자르기 위해 약간 더 넉넉히 가져옴
        raw_top_k = max(k * 2, k + 4)

        filters = self._build_filters(uploaded_category)

        q_vec = self.embedder.embed_query(query)
        if not q_vec:
            log.warning("Query embedding 결과가 비어 있습니다.")
            return []

        candidates = self.vector_store.search(q_vec, top_k=raw_top_k, filters=filters)
        if not candidates:
            return []

        ranked = rerank_simple(candidates, category_boost=CATEGORY_BOOST)

        # parent-child 보강: Excel summary 가 검색되면 같은 sheet 의 raw_table 일부를 추가
        if with_parent_children:
            ranked = self._augment_with_children(ranked, max_per_parent=max_children_per_parent)

        out = ranked[:k]
        tracker.set_retrieved_chunks(len(out))
        return out

    # ------------------------------------------------------------------
    def _build_filters(self, uploaded_category: Optional[str]) -> Optional[Dict[str, Any]]:
        if not uploaded_category or uploaded_category in {"all", "전체"}:
            return None
        # Streamlit 한글 라벨 호환
        mapping = {
            "Slack 대화": "slack",
            "가이드": "guide",
            "카톡 대화": "kakao",
            "Excel": "excel",
            "기타": "misc",
        }
        cat = mapping.get(uploaded_category, uploaded_category)
        return {"uploaded_category": cat}

    def _augment_with_children(
        self, ranked: List[RetrievedChunk], max_per_parent: int = 2
    ) -> List[RetrievedChunk]:
        existing_ids = {c.chunk_id for c in ranked}
        out = list(ranked)
        for parent in list(ranked):
            if parent.content_type != "excel_summary":
                continue
            sheet = parent.metadata.get("sheet_name")
            children = self.vector_store.get_children(
                parent.chunk_id, sheet_name=sheet, limit=max_per_parent
            )
            for ch in children:
                if ch.chunk_id in existing_ids:
                    continue
                # 자식은 보조 근거이므로 final_score 를 parent 의 90% 정도로 둔다.
                ch.final_score = float(parent.final_score) * 0.9
                ch.score = ch.final_score
                out.append(ch)
                existing_ids.add(ch.chunk_id)
        # final_score 기준 재정렬
        out.sort(key=lambda x: x.final_score, reverse=True)
        return out
