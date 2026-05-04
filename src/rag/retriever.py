"""
retriever.py
============
질문 → 임베딩 → ChromaDB 검색 → metadata-aware reranking →
threshold 필터 → 파일별 cap → MMR(다양성) → parent-child 보강.

reranker.py 의 rule-based reranker / diversity penalty 를 사용한다.

Score 해석 주의
---------------
ChromaDB 는 cosine **distance** (낮을수록 좋음) 를 반환하므로,
`vector_store.search` 안에서 이미 `score = 1 - distance` 형태의 similarity 로 변환한다.
따라서 `RetrievedChunk.score` 는 "값이 높을수록 좋은" 유사도 점수로 해석한다.
(임베딩이 정규화되어 있다면 대체로 [0.0, 1.0] 범위에 들어온다.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.config import settings
from src.logger import get_logger
from src.rag.embedder import Embedder, get_default_embedder
from src.rag.reranker import (
    DEFAULT_CATEGORY_BOOST,
    apply_diversity_penalty,
    classify_query,
    extract_query_metadata,
    rerank_simple,
)
from src.schemas import RetrievedChunk
from src.storage.vector_store import VectorStore
from src.utils.cost_utils import tracker

log = get_logger(__name__)


# 호환을 위한 export (기존에 import 하던 코드용)
CATEGORY_BOOST: Dict[str, float] = DEFAULT_CATEGORY_BOOST


# ---------------------------------------------------------------------------
# 검색 결과 진단 객체
# ---------------------------------------------------------------------------
@dataclass
class RetrievalDetails:
    """검색 테스트 페이지 / qa_pipeline 에서 모두 사용하는 진단용 결과 객체."""
    passed: List[RetrievedChunk] = field(default_factory=list)
    candidates: List[RetrievedChunk] = field(default_factory=list)  # 탈락 포함 전부
    summary: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
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
        min_similarity: Optional[float] = None,
        min_final: Optional[float] = None,
        max_per_file: Optional[int] = None,
        use_mmr: Optional[bool] = None,
        enable_date_filter: Optional[bool] = None,
    ) -> List[RetrievedChunk]:
        """기존 호환 인터페이스. threshold 통과 + MMR 적용된 chunk 만 반환."""
        details = self.retrieve_with_details(
            query=query,
            top_k=top_k,
            uploaded_category=uploaded_category,
            with_parent_children=with_parent_children,
            max_children_per_parent=max_children_per_parent,
            min_similarity=min_similarity,
            min_final=min_final,
            max_per_file=max_per_file,
            use_mmr=use_mmr,
            enable_date_filter=enable_date_filter,
        )
        return details.passed

    # ------------------------------------------------------------------
    def retrieve_with_details(
        self,
        query: str,
        top_k: Optional[int] = None,
        uploaded_category: Optional[str] = None,
        with_parent_children: bool = True,
        max_children_per_parent: int = 2,
        min_similarity: Optional[float] = None,
        min_final: Optional[float] = None,
        max_per_file: Optional[int] = None,
        use_mmr: Optional[bool] = None,
        enable_date_filter: Optional[bool] = None,
    ) -> RetrievalDetails:
        """
        검색 + threshold + MMR + parent-child 까지 적용한 진단용 결과.

        Returns
        -------
        RetrievalDetails
            - passed     : 최종 사용 가능한 chunk (top_k 까지)
            - candidates : 모든 후보 (탈락 포함, passed_threshold/filter_reason 채워짐)
            - summary    : 카운트 / 임계값 / 옵션 메타
        """
        # query metadata 추출 (date / topics / intent)
        query_meta = extract_query_metadata(query or "")

        details = RetrievalDetails(summary={
            "candidate_count": 0,
            "passed_count": 0,
            "dropped_count": 0,
            "min_similarity": float(min_similarity if min_similarity is not None else settings.min_similarity_score),
            "min_final": float(min_final if min_final is not None else settings.min_final_score),
            "max_per_file": int(max_per_file if max_per_file is not None else settings.max_chunks_per_file),
            "use_mmr": bool(use_mmr if use_mmr is not None else settings.use_mmr),
            "mmr_lambda": float(settings.mmr_lambda),
            "top_k": int(top_k or settings.top_k),
            "uploaded_category_filter": uploaded_category or "all",
            "score_interpretation": "score = 1 - cosine_distance (높을수록 유사)",
            "query_class": query_meta.get("query_class") or classify_query(query or ""),
            # date / topic 진단
            "query_date": query_meta.get("query_date"),
            "query_date_text": query_meta.get("query_date_text"),
            "query_topics": query_meta.get("query_topics") or [],
            "query_intent": query_meta.get("query_intent") or [],
            "enable_date_filter": bool(
                enable_date_filter if enable_date_filter is not None else settings.enable_date_filter
            ),
            "date_exact_match_boost": float(settings.date_exact_match_boost),
            "date_mismatch_penalty": float(settings.date_mismatch_penalty),
            "topic_match_boost": float(settings.topic_match_boost),
            "topic_mismatch_penalty": float(settings.topic_mismatch_penalty),
            # 비식별화 상태 (UI 노출용)
            "anonymize_output": bool(settings.anonymize_output),
            "show_raw_content": bool(settings.show_raw_content),
        })

        if not query or not query.strip():
            return details

        k = int(top_k or settings.top_k)
        # 후처리에서 자르기 위해 약 3배 넉넉히 가져옴
        raw_top_k = max(k * 3, k + 6)
        min_sim = float(details.summary["min_similarity"])
        min_fin = float(details.summary["min_final"])
        per_file = int(details.summary["max_per_file"])
        do_mmr = bool(details.summary["use_mmr"])
        do_date_filter = bool(details.summary["enable_date_filter"])
        q_date = query_meta.get("query_date")

        filters = self._build_filters(uploaded_category)

        q_vec = self.embedder.embed_query(query)
        if not q_vec:
            log.warning("Query embedding 결과가 비어 있습니다.")
            return details

        candidates = self.vector_store.search(q_vec, top_k=raw_top_k, filters=filters)
        if not candidates:
            return details

        # 1) rerank (final_score 채움 + 진단 metadata 기록)
        ranked = rerank_simple(
            candidates,
            category_boost=CATEGORY_BOOST,
            query=query,
            query_metadata=query_meta,
        )

        # 2) parent-child 보강 (Excel summary -> raw_table)
        if with_parent_children:
            ranked = self._augment_with_children(ranked, max_per_parent=max_children_per_parent)

        # 3) threshold 필터 + 파일별 cap (+ optional date filter)
        per_file_count: Dict[str, int] = {}
        passed: List[RetrievedChunk] = []
        for c in ranked:
            sim = float(c.score or 0.0)
            fin = float(c.final_score or 0.0)
            chunk_date = c.metadata.get("document_date")

            if do_date_filter and q_date and chunk_date and chunk_date != q_date:
                c.passed_threshold = False
                c.filter_reason = (
                    f"date_filter_excluded(query_date={q_date}, doc_date={chunk_date})"
                )
                continue
            if sim < min_sim:
                c.passed_threshold = False
                c.filter_reason = f"similarity({sim:.3f}) < min_similarity({min_sim:.2f})"
                continue
            if fin < min_fin:
                c.passed_threshold = False
                c.filter_reason = f"final_score({fin:.3f}) < min_final({min_fin:.2f})"
                continue
            file_key = c.file_name or c.document_id or ""
            cnt = per_file_count.get(file_key, 0)
            if per_file > 0 and cnt >= per_file:
                c.passed_threshold = False
                c.filter_reason = f"file_cap_exceeded(max_per_file={per_file})"
                continue
            c.passed_threshold = True
            c.filter_reason = None
            per_file_count[file_key] = cnt + 1
            passed.append(c)

        # 4) MMR 비슷한 다양성 보정 (passed 안에서만 적용 → 더 다양한 chunk 가 위로)
        if do_mmr and len(passed) > 1:
            passed = apply_diversity_penalty(
                passed,
                mmr_lambda=float(settings.mmr_lambda),
            )

        # 5) top_k 절단
        passed = passed[:k]

        # 6) 결과 패킹
        details.candidates = ranked
        details.passed = passed
        details.summary["candidate_count"] = len(ranked)
        details.summary["passed_count"] = len(passed)
        details.summary["dropped_count"] = len(ranked) - len(passed)

        tracker.set_retrieved_chunks(len(passed))
        return details

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
                ch.score = max(float(ch.score or 0.0), float(parent.score or 0.0) * 0.9)
                ch.metadata["mmr_diversity_bonus"] = 1.0
                ch.metadata["category_boost"] = parent.metadata.get("category_boost")
                ch.metadata["content_type_boost"] = parent.metadata.get("content_type_boost")
                ch.metadata["source_weight"] = ch.metadata.get("source_weight") or parent.metadata.get("source_weight")
                ch.metadata["similarity_score"] = ch.score
                out.append(ch)
                existing_ids.add(ch.chunk_id)
        # final_score 기준 재정렬
        out.sort(key=lambda x: x.final_score, reverse=True)
        return out
