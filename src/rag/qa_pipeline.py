"""
qa_pipeline.py
==============
업무 QA 전체 파이프라인.

Steps:
1) 질문 정제
2) (옵션) query rewrite
3) retriever 로 chunk 검색 + threshold + MMR + parent-child 보강 (retriever 내부)
4) 근거 부족(threshold 통과 chunk < MIN_RETRIEVED_CHUNKS) 이면 Gemini generation **호출하지 않고** 안내 메시지 반환
5) 충분한 근거가 있을 때만 prompt_builder 로 프롬프트 구성 → Gemini generation
6) qa_logs 에 JSON 저장 (generation_skipped 여부 포함)
7) (answer, retrieved_chunks, meta) 반환
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import settings
from src.logger import get_logger
from src.preprocessing.anonymizer import anonymize_text
from src.rag.embedder import Embedder, get_default_embedder
from src.rag.generator import Generator
from src.rag.prompt_builder import (
    build_normalized_document_answer_prompt,
    build_qa_prompt,
    split_chunks_by_retrieval_role,
)
from src.rag.query_rewriter import rewrite_query_if_enabled
from src.rag.retriever import Retriever
from src.schemas import QALog, RetrievedChunk
from src.utils.path_utils import ensure_dir
from src.utils.time_utils import now_compact, now_iso

log = get_logger(__name__)


INSUFFICIENT_EVIDENCE_MESSAGE = (
    "현재 적재된 자료만으로는 질문에 직접 답할 수 있는 근거가 부족합니다.\n"
    "검색된 후보 문서는 있었지만 관련도 기준을 통과하지 못했습니다.\n"
    "검색어를 더 구체화하거나, 관련 가이드/Slack 스레드/Excel 파일을 추가로 업로드한 뒤 다시 시도해 주세요."
)


class QAPipeline:
    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        generator: Optional[Generator] = None,
        embedder: Optional[Embedder] = None,
    ) -> None:
        self.embedder = embedder or get_default_embedder()
        self.retriever = retriever or Retriever(embedder=self.embedder)
        self.generator = generator or Generator()

    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
        uploaded_category: Optional[str] = None,
        enable_query_rewrite: Optional[bool] = None,
        save_log: bool = True,
        min_retrieved_chunks: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not question or not question.strip():
            return {
                "answer": "질문이 비어 있습니다.",
                "retrieved_chunks": [],
                "used_chunks": [],
                "model_name": "",
                "embedding_provider": self.embedder.provider,
                "embedding_model": self.embedder.model_name,
                "rewritten_query": None,
                "generation_skipped": True,
                "skip_reason": "empty_question",
                "retrieval_summary": {},
                # Task 7: Normalized Document 중심 답변 진단 (호환용 default)
                "answer_mode": "insufficient_evidence",
                "answer_format_label": "default",
                "primary_normalized_document_count": 0,
                "primary_normalized_documents": [],
                "normalized_document_answer_template_version": (
                    settings.knowledge_card_answer_template_version
                ),
                # legacy 호환 키
                "primary_card_count": 0,
                "raw_evidence_count": 0,
                "raw_fallback_count": 0,
                "primary_cards": [],
                "raw_evidence": [],
                "raw_fallback": [],
                "knowledge_card_answer_template_version": (
                    settings.knowledge_card_answer_template_version
                ),
                # MVP 2차 Step 1: Retrieval Diagnostics 강화 — empty 케이스에도 default 값을 노출.
                "query_topic": None,
                "query_topics": [],
                "query_intent": [],
                "query_date": None,
                "retrieved_count": 0,
                "passed_count": 0,
                "topic_mismatch_count": 0,
                "normalized_document_candidate_count": 0,
                "raw_candidate_count": 0,
            }

        # 1) 정제
        q = question.strip()

        # 2) query rewrite
        rewritten = rewrite_query_if_enabled(q, enable=enable_query_rewrite)
        search_query = rewritten or q

        # 3) 검색 (진단 정보 포함)
        from src.rag.retriever import RetrievalDetails
        try:
            details = self.retriever.retrieve_with_details(
                search_query,
                top_k=top_k,
                uploaded_category=uploaded_category,
            )
        except Exception as e:  # noqa: BLE001
            log.error("검색 실패: %s", e)
            details = RetrievalDetails()

        passed: List[RetrievedChunk] = list(details.passed)
        candidates: List[RetrievedChunk] = list(details.candidates)
        summary = dict(details.summary)

        # 4) 근거 부족 판정
        threshold_n = int(
            min_retrieved_chunks
            if min_retrieved_chunks is not None
            else settings.min_retrieved_chunks
        )
        generation_skipped = False
        skip_reason: Optional[str] = None
        used_chunks: List[RetrievedChunk] = []
        used_model = ""
        prompt_chars = 0
        answer_format_label = "default"

        # Task 7: retrieval_role 기반 그룹 카운트 (Normalized Document 중심 답변용)
        groups = split_chunks_by_retrieval_role(passed)
        primary_card_count = len(groups["primary_cards"])
        raw_evidence_count = len(groups["raw_evidence"])
        raw_fallback_count = len(groups["raw_fallback"])

        if len(passed) < threshold_n:
            generation_skipped = True
            skip_reason = (
                "no_candidates"
                if not candidates
                else "below_threshold"
            )
            answer_mode = "insufficient_evidence"
            answer = INSUFFICIENT_EVIDENCE_MESSAGE
            log.info(
                "QA: generation skipped (passed=%d / candidates=%d / threshold=%d)",
                len(passed), len(candidates), threshold_n,
            )
        else:
            # 5) prompt + generation
            if (
                settings.answer_with_knowledge_cards
                and primary_card_count > 0
            ):
                prompt, used_chunks, answer_format_label = (
                    build_normalized_document_answer_prompt(
                        question=q,
                        chunks=passed,
                        rewritten_query=rewritten,
                    )
                )
                # answer_mode 라벨은 기존 ("knowledge_card") 을 유지해 외부 호환을 보장한다.
                # Streamlit / 로그 / 테스트가 이미 이 라벨을 비교하기 때문이다.
                answer_mode = "knowledge_card"
            else:
                prompt, used_chunks = build_qa_prompt(
                    q, passed, rewritten_query=rewritten
                )
                answer_mode = "raw_fallback"
            prompt_chars = len(prompt or "")
            answer, used_model = self.generator.generate(prompt)

        # 6) log
        def _preview_for(c: RetrievedChunk) -> str:
            md = c.metadata or {}
            if settings.anonymize_output:
                src = md.get("sanitized_content") or anonymize_text(c.content or "")
            else:
                src = c.content or ""
            return src[:600]

        log_obj = QALog(
            question=q,
            rewritten_query=rewritten,
            answer=answer,
            retrieved_chunks=[
                {
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "file_name": c.file_name,
                    "section_title": c.section_title,
                    "content_type": c.content_type,
                    "uploaded_category": c.uploaded_category,
                    "source_type": c.source_type,
                    "score": c.score,
                    "final_score": c.final_score,
                    "passed_threshold": c.passed_threshold,
                    "filter_reason": c.filter_reason,
                    # 비식별화된 preview 만 로그에 남긴다 (원문은 raw 파일에 그대로 보존됨)
                    "preview": _preview_for(c),
                    "metadata": c.metadata,
                }
                for c in (used_chunks or passed or candidates)
            ],
            model_name=used_model or "",
            embedding_provider=self.embedder.provider,
            created_at=now_iso(),
            metadata={
                "embedding_model": self.embedder.model_name,
                "uploaded_category_filter": uploaded_category or "all",
                "top_k": int(top_k or settings.top_k),
                "n_retrieved": len(candidates),
                "n_passed": len(passed),
                "n_dropped": max(0, len(candidates) - len(passed)),
                "min_similarity": summary.get("min_similarity"),
                "min_final": summary.get("min_final"),
                "max_per_file": summary.get("max_per_file"),
                "use_mmr": summary.get("use_mmr"),
                "min_retrieved_chunks": threshold_n,
                "generation_skipped": generation_skipped,
                "skip_reason": skip_reason,
                "prompt_chars": prompt_chars,
                "used_chunks": len(used_chunks),
                "query_class": summary.get("query_class"),
                "query_date": summary.get("query_date"),
                "query_topic": summary.get("query_topic"),
                "query_topics": summary.get("query_topics"),
                "query_intent": summary.get("query_intent"),
                "enable_date_filter": summary.get("enable_date_filter"),
                "anonymize_output": summary.get("anonymize_output"),
                # MVP 2차 Step 1: Retrieval Diagnostics 강화
                "retrieved_count": summary.get("retrieved_count"),
                "topic_mismatch_count": summary.get("topic_mismatch_count"),
                "normalized_document_candidate_count": summary.get(
                    "normalized_document_candidate_count"
                ),
                "raw_candidate_count": summary.get("raw_candidate_count"),
                # Task 7: Normalized Document 중심 답변 진단
                "answer_mode": answer_mode,
                "answer_format_label": answer_format_label,
                # 신규 진단 키
                "primary_normalized_document_count": primary_card_count,
                "answer_with_normalized_documents": bool(
                    settings.answer_with_knowledge_cards
                ),
                "normalized_document_answer_template_version": (
                    settings.knowledge_card_answer_template_version
                ),
                # legacy 호환 키
                "primary_card_count": primary_card_count,
                "raw_evidence_count": raw_evidence_count,
                "raw_fallback_count": raw_fallback_count,
                "answer_with_knowledge_cards": bool(
                    settings.answer_with_knowledge_cards
                ),
                "knowledge_card_answer_template_version": (
                    settings.knowledge_card_answer_template_version
                ),
            },
        )
        if save_log:
            try:
                self._save_log(log_obj)
            except Exception as e:  # noqa: BLE001
                log.warning("QA 로그 저장 실패: %s", e)

        return {
            "answer": answer,
            "retrieved_chunks": passed if passed else candidates,
            "all_candidates": candidates,
            "used_chunks": used_chunks,
            "model_name": used_model,
            "embedding_provider": self.embedder.provider,
            "embedding_model": self.embedder.model_name,
            "rewritten_query": rewritten,
            "generation_skipped": generation_skipped,
            "skip_reason": skip_reason,
            "retrieval_summary": summary,
            "prompt_chars": prompt_chars,
            # Task 7: Normalized Document 중심 답변 진단
            "answer_mode": answer_mode,
            "answer_format_label": answer_format_label,
            # 신규 진단 키
            "primary_normalized_document_count": primary_card_count,
            "primary_normalized_documents": list(groups["primary_cards"]),
            "normalized_document_answer_template_version": (
                settings.knowledge_card_answer_template_version
            ),
            # legacy 호환 키
            "primary_card_count": primary_card_count,
            "raw_evidence_count": raw_evidence_count,
            "raw_fallback_count": raw_fallback_count,
            "primary_cards": list(groups["primary_cards"]),
            "raw_evidence": list(groups["raw_evidence"]),
            "raw_fallback": list(groups["raw_fallback"]),
            "knowledge_card_answer_template_version": (
                settings.knowledge_card_answer_template_version
            ),
            # ----------------------------------------------------------------
            # MVP 2차 Step 1: Retrieval Diagnostics 강화
            #
            # qa_pipeline 반환 dict 의 top-level 에도 retrieval diagnostics 를
            # 명시적으로 노출한다. (retrieval_summary 안에도 같은 값이 있지만,
            # Streamlit / Slack adapter / 테스트에서 평탄한 dict 로 바로 읽을 수
            # 있도록 중복 노출한다.)
            #
            # 이 Step 에서는 검색 점수/필터/penalty 정책을 바꾸지 않는다 — 단지
            # 왜 그런 결과가 나왔는지 더 잘 보이게 한다.
            # ----------------------------------------------------------------
            "query_topic": summary.get("query_topic"),
            "query_topics": list(summary.get("query_topics") or []),
            "query_intent": list(summary.get("query_intent") or []),
            "query_date": summary.get("query_date"),
            "retrieved_count": int(
                summary.get("retrieved_count")
                if summary.get("retrieved_count") is not None
                else summary.get("candidate_count") or 0
            ),
            "passed_count": int(summary.get("passed_count") or 0),
            "topic_mismatch_count": int(summary.get("topic_mismatch_count") or 0),
            "normalized_document_candidate_count": int(
                summary.get("normalized_document_candidate_count") or 0
            ),
            "raw_candidate_count": int(summary.get("raw_candidate_count") or 0),
        }

    # ------------------------------------------------------------------
    def _save_log(self, log_obj: QALog) -> Path:
        ensure_dir(settings.qa_log_dir)
        path = settings.qa_log_dir / f"qa_{now_compact()}.json"
        path.write_text(
            json.dumps(log_obj.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path
