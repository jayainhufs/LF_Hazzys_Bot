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
from typing import Any, Dict, List, Optional, Tuple

from src.config import settings
from src.logger import get_logger
from src.rag.embedder import Embedder, get_default_embedder
from src.rag.reranker import (
    DEFAULT_CATEGORY_BOOST,
    apply_diversity_penalty,
    apply_normalized_document_priority,
    classify_query,
    extract_query_metadata,
    is_normalized_document_chunk,
    rerank_simple,
)
# legacy alias — 기존 코드 호환
apply_knowledge_card_priority = apply_normalized_document_priority
is_knowledge_card_chunk = is_normalized_document_chunk
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

        # MVP 2차 Step 1 (Retrieval Diagnostics 강화): 첫 topic 을 별도로 노출.
        # query_topics 가 list 이므로 단수형 ``query_topic`` (대표 topic) 도 함께 제공해
        # Slack debug / Streamlit 카드 표시에서 바로 쓸 수 있게 한다.
        _qtopics: List[str] = list(query_meta.get("query_topics") or [])
        _qtopic_single: Optional[str] = _qtopics[0] if _qtopics else None

        details = RetrievalDetails(summary={
            "candidate_count": 0,
            # MVP 2차 Step 1: candidate_count 의 alias.
            # Slack/Streamlit diagnostics 가 통일된 이름 (retrieved_count) 으로 읽을 수 있게 둔다.
            "retrieved_count": 0,
            "passed_count": 0,
            "dropped_count": 0,
            "min_similarity": float(min_similarity if min_similarity is not None else settings.min_similarity_score),
            "min_final": float(min_final if min_final is not None else settings.min_final_score),
            "max_per_file": int(max_per_file if max_per_file is not None else settings.max_chunks_per_file),
            "use_mmr": bool(use_mmr if use_mmr is not None else settings.use_mmr),
            "mmr_lambda": float(settings.mmr_lambda),
            # MVP 2차 Step 6: Hybrid Retrieval / BM25 diagnostics.
            # HYBRID_RETRIEVAL_ENABLED=false is the default, so vector-only
            # behavior remains the baseline.
            "hybrid_retrieval_enabled": bool(settings.hybrid_retrieval_enabled),
            "bm25_candidate_count": 0,
            "vector_candidate_count": 0,
            "hybrid_merged_candidate_count": 0,
            "bm25_only_candidate_count": 0,
            "vector_only_candidate_count": 0,
            "overlap_candidate_count": 0,
            "hybrid_rrf_k": int(settings.hybrid_rrf_k),
            "hybrid_vector_weight": float(settings.hybrid_vector_weight),
            "hybrid_bm25_weight": float(settings.hybrid_bm25_weight),
            "top_k": int(top_k or settings.top_k),
            "uploaded_category_filter": uploaded_category or "all",
            "score_interpretation": "score = 1 - cosine_distance (높을수록 유사)",
            "query_class": query_meta.get("query_class") or classify_query(query or ""),
            # date / topic 진단
            "query_date": query_meta.get("query_date"),
            "query_date_text": query_meta.get("query_date_text"),
            "query_topics": _qtopics,
            # MVP 2차 Step 1: list 의 첫 topic 을 단수로 노출. 비어있으면 None.
            "query_topic": _qtopic_single,
            "query_intent": query_meta.get("query_intent") or [],
            "enable_date_filter": bool(
                enable_date_filter if enable_date_filter is not None else settings.enable_date_filter
            ),
            "date_exact_match_boost": float(settings.date_exact_match_boost),
            "date_mismatch_penalty": float(settings.date_mismatch_penalty),
            "topic_match_boost": float(settings.topic_match_boost),
            "topic_mismatch_penalty": float(settings.topic_mismatch_penalty),
            # MVP 2차 Step 1: topic mismatch / candidate 구성 진단 카운트.
            # 후처리에서 채워진다 (default 0).
            "topic_mismatch_count": 0,
            "normalized_document_candidate_count": 0,
            "raw_candidate_count": 0,
            # MVP 2차 Step 2: topic-aware 격하 진단.
            # apply_normalized_document_priority 가 명확한 topic mismatch chunk 를
            # primary_card 로 승격하지 않고 raw_fallback 으로 격하한 건수.
            "topic_mismatch_demoted_count": 0,
            # Normalized Document 우선 retrieval 진단 (Task 6)
            # 신규 표준 키
            "prioritize_normalized_documents": bool(settings.prioritize_knowledge_cards),
            "normalized_document_content_boost": float(settings.knowledge_card_content_boost),
            "normalized_document_count": 0,
            # legacy 호환 키 (기존 UI / 테스트 호환)
            "prioritize_knowledge_cards": bool(settings.prioritize_knowledge_cards),
            "knowledge_card_content_boost": float(settings.knowledge_card_content_boost),
            "knowledge_card_count": 0,
            "raw_evidence_boost": float(settings.raw_evidence_boost),
            "enable_parent_raw_evidence": bool(settings.enable_parent_raw_evidence),
            "parent_raw_evidence_top_k": int(settings.parent_raw_evidence_top_k),
            "raw_evidence_count": 0,
            "raw_fallback_count": 0,
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
        hybrid_summary: Dict[str, Any] = {
            "vector_candidate_count": len(candidates),
            "bm25_candidate_count": 0,
            "hybrid_merged_candidate_count": len(candidates),
            "bm25_only_candidate_count": 0,
            "vector_only_candidate_count": len(candidates),
            "overlap_candidate_count": 0,
        }
        if details.summary["hybrid_retrieval_enabled"]:
            candidates, hybrid_summary = self._merge_hybrid_candidates(
                query=query,
                vector_candidates=candidates,
                top_k=raw_top_k,
                filters=filters,
            )
        if not candidates:
            details.summary.update(hybrid_summary)
            return details

        # 1) rerank (final_score 채움 + 진단 metadata 기록)
        ranked = rerank_simple(
            candidates,
            category_boost=CATEGORY_BOOST,
            query=query,
            query_metadata=query_meta,
        )

        # 1-b) Normalized Document 우선 retrieval (Task 6)
        # PRIORITIZE_NORMALIZED_DOCUMENTS=true 면 Normalized Document chunk 가
        # raw chunk 보다 위로 올라온다. (legacy: PRIORITIZE_KNOWLEDGE_CARDS)
        ranked = apply_normalized_document_priority(
            ranked,
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

        # 6) parent_raw_evidence 진단 처리 (Task 6)
        # ENABLE_PARENT_RAW_EVIDENCE=false 면 Normalized Document 의 parent_raw_chunk_ids 를
        # metadata 에서 비워둔다.
        if not settings.enable_parent_raw_evidence:
            for c in passed:
                if is_normalized_document_chunk(c):
                    md = c.metadata or {}
                    md["parent_raw_chunk_ids"] = []
                    c.metadata = md

        # 7) role 별 카운트 (요약용)
        kc_count = 0
        raw_evidence_count = 0
        raw_fallback_count = 0
        for c in passed:
            md = c.metadata or {}
            role = md.get("retrieval_role")
            if role == "primary_card":
                kc_count += 1
            elif role == "raw_evidence":
                raw_evidence_count += 1
            elif role == "raw_fallback":
                raw_fallback_count += 1
            else:
                # role 이 비어 있으면 추정값으로 보정
                if is_normalized_document_chunk(c):
                    kc_count += 1
                else:
                    raw_fallback_count += 1

        # MVP 2차 Step 1 (Retrieval Diagnostics):
        # 어떤 chunk 가 검색됐는지 / 왜 mismatch 가 났는지 더 잘 보이도록
        # candidate 전체에 대한 진단 카운트도 함께 채운다. (검색 정책은 변경하지 않음)
        nd_candidate_count = 0
        raw_candidate_count = 0
        topic_mismatch_count = 0
        # MVP 2차 Step 2: topic-aware 격하 건수
        topic_mismatch_demoted_count = 0
        for c in ranked:
            md = c.metadata or {}
            if is_normalized_document_chunk(c):
                nd_candidate_count += 1
            else:
                raw_candidate_count += 1
            if md.get("topic_match") == "mismatch":
                topic_mismatch_count += 1
            if md.get("topic_mismatch_demoted"):
                topic_mismatch_demoted_count += 1

        # 8) 결과 패킹
        details.candidates = ranked
        details.passed = passed
        details.summary["candidate_count"] = len(ranked)
        # MVP 2차 Step 1: candidate_count 와 동일한 의미의 alias.
        details.summary["retrieved_count"] = len(ranked)
        details.summary["passed_count"] = len(passed)
        details.summary["dropped_count"] = len(ranked) - len(passed)
        # MVP 2차 Step 1: 진단 카운트
        details.summary["topic_mismatch_count"] = topic_mismatch_count
        details.summary["normalized_document_candidate_count"] = nd_candidate_count
        details.summary["raw_candidate_count"] = raw_candidate_count
        # MVP 2차 Step 2: topic-aware 격하 건수
        details.summary["topic_mismatch_demoted_count"] = topic_mismatch_demoted_count
        # 신규 표준 키
        details.summary["normalized_document_count"] = kc_count
        # legacy 호환 키
        details.summary["knowledge_card_count"] = kc_count
        details.summary["raw_evidence_count"] = raw_evidence_count
        details.summary["raw_fallback_count"] = raw_fallback_count
        details.summary.update(hybrid_summary)
        details.summary["hybrid_merged_candidate_count"] = len(candidates)

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

    def _merge_hybrid_candidates(
        self,
        *,
        query: str,
        vector_candidates: List[RetrievedChunk],
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> Tuple[List[RetrievedChunk], Dict[str, Any]]:
        """
        Merge vector and BM25 candidates with reciprocal rank fusion.

        This is only called when HYBRID_RETRIEVAL_ENABLED=true. If the injected
        vector store does not support BM25, the vector-only candidates are
        returned unchanged with diagnostics that make the fallback visible.
        """
        bm25_search = getattr(self.vector_store, "search_bm25", None)
        if bm25_search is None:
            return list(vector_candidates), {
                "vector_candidate_count": len(vector_candidates),
                "bm25_candidate_count": 0,
                "hybrid_merged_candidate_count": len(vector_candidates),
                "bm25_only_candidate_count": 0,
                "vector_only_candidate_count": len(vector_candidates),
                "overlap_candidate_count": 0,
            }

        bm25_top_k = int(settings.hybrid_bm25_top_k or top_k)
        bm25_candidates = list(
            bm25_search(query=query, top_k=bm25_top_k, filters=filters)
        )

        vector_by_id: Dict[str, RetrievedChunk] = {
            c.chunk_id: c for c in vector_candidates
        }
        bm25_by_id: Dict[str, RetrievedChunk] = {
            c.chunk_id: c for c in bm25_candidates
        }
        vector_ranks = {c.chunk_id: idx for idx, c in enumerate(vector_candidates, start=1)}
        bm25_ranks = {c.chunk_id: idx for idx, c in enumerate(bm25_candidates, start=1)}

        rrf_k = max(1, int(settings.hybrid_rrf_k or 60))
        vector_weight = float(settings.hybrid_vector_weight or 1.0)
        bm25_weight = float(settings.hybrid_bm25_weight or 1.0)

        merged: List[RetrievedChunk] = []
        for chunk_id in set(vector_by_id) | set(bm25_by_id):
            chunk = vector_by_id.get(chunk_id) or bm25_by_id[chunk_id]
            md = dict(chunk.metadata or {})
            sources: List[str] = []
            vector_rank = vector_ranks.get(chunk_id)
            bm25_rank = bm25_ranks.get(chunk_id)

            hybrid_rrf = 0.0
            if vector_rank is not None:
                sources.append("vector")
                hybrid_rrf += vector_weight / float(rrf_k + vector_rank)
            if bm25_rank is not None:
                sources.append("bm25")
                hybrid_rrf += bm25_weight / float(rrf_k + bm25_rank)

            bm25_chunk = bm25_by_id.get(chunk_id)
            vector_chunk = vector_by_id.get(chunk_id)
            bm25_score = float(bm25_chunk.score or 0.0) if bm25_chunk else 0.0
            vector_score = float(vector_chunk.score or 0.0) if vector_chunk else 0.0

            md["retrieval_sources"] = sources
            if vector_rank is not None:
                md["vector_rank"] = int(vector_rank)
            if bm25_rank is not None:
                md["bm25_rank"] = int(bm25_rank)
                if bm25_chunk is not None:
                    md["bm25_score"] = bm25_chunk.metadata.get("bm25_score")
                    md["bm25_normalized_score"] = bm25_chunk.metadata.get(
                        "bm25_normalized_score", round(bm25_score, 6)
                    )
            md["hybrid_rrf_score"] = round(float(hybrid_rrf), 6)

            chunk.metadata = md
            chunk.score = max(vector_score, bm25_score)
            chunk.final_score = float(chunk.score)
            merged.append(chunk)

        merged.sort(
            key=lambda c: (
                float((c.metadata or {}).get("hybrid_rrf_score") or 0.0),
                float(c.score or 0.0),
            ),
            reverse=True,
        )

        vector_ids = set(vector_by_id)
        bm25_ids = set(bm25_by_id)
        overlap = vector_ids & bm25_ids
        summary = {
            "vector_candidate_count": len(vector_candidates),
            "bm25_candidate_count": len(bm25_candidates),
            "hybrid_merged_candidate_count": len(merged),
            "bm25_only_candidate_count": len(bm25_ids - vector_ids),
            "vector_only_candidate_count": len(vector_ids - bm25_ids),
            "overlap_candidate_count": len(overlap),
        }
        return merged, summary

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
