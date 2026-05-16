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
from src.rag.reranker import _GENERIC_TOPICS, is_clear_topic_mismatch
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


# ---------------------------------------------------------------------------
# MVP 2차 Step 4: Raw Fallback 오남용 방지 — diagnostics helper
# ---------------------------------------------------------------------------
# weak_evidence_warning 정책 임계값. 너무 강하게 잡지 않도록 보수적으로 둔다.
# - raw_fallback chunk 가 최소 2개 이상이고,
# - 그중 query_topic 과 명확히 다른 topic 비율이 0.7 이상일 때 weak 로 본다.
_WEAK_EVIDENCE_MIN_RAW_FALLBACK = 2
_WEAK_EVIDENCE_TOPIC_MISMATCH_RATIO = 0.7


def _summarize_raw_fallback_policy(
    *,
    primary_count: int,
    raw_evidence_count: int,
    raw_fallback_chunks: List[RetrievedChunk],
    normalized_document_candidate_count: int,
    query_topics: List[str],
    generation_skipped: bool,
) -> Dict[str, Any]:
    """
    Step 4 진단 필드를 계산해 dict 로 반환한다.

    출력 키:
        raw_fallback_only / raw_fallback_only_reason
        raw_fallback_topic_mismatch_count / raw_fallback_topic_mismatch_ratio
        primary_evidence_available / normalized_document_available
        evidence_strength / weak_evidence_warning

    정책:
    - ``raw_fallback_only`` = primary normalized document 가 0개이고 raw_fallback 이 있을 때.
    - ``primary_evidence_available`` = primary normalized document 가 있거나 raw_evidence 가 있을 때.
    - ``normalized_document_available`` = retrieval candidate 단계에서 normalized document 가
      한 건이라도 존재했는지. (이번 검색에서 정규화 문서가 후보로라도 잡혔는지 여부)
    - ``raw_fallback_topic_mismatch_count`` = raw_fallback chunk 중 query_topic 과
      명확히 mismatch 인 chunk 수.
        - query_topic 이 비어 있거나 chunk topic 이 generic / unknown 이면 mismatch 로 잡지 않는다.
    - ``evidence_strength``:
        - generation_skipped 또는 모든 카운트가 0 → "insufficient"
        - primary normalized document 가 있음 → "strong"
        - raw_evidence 가 있음 또는 raw_fallback 이 모두 topic match → "medium"
        - raw_fallback_only 인데 topic mismatch 비율이 높음 → "insufficient"
        - 그 외 raw_fallback_only → "weak"
    - ``weak_evidence_warning`` =
        raw_fallback_only AND query_topic 명확 AND
        raw_fallback_count >= 2 AND mismatch_ratio >= 0.7
    """
    raw_fallback_count = len(raw_fallback_chunks)
    raw_fallback_only = (primary_count == 0) and (raw_fallback_count > 0)

    # query_topic 이 명확한지 여부 (None 이면 mismatch warning 을 강하게 켜지 않음)
    query_topic_clear = bool(query_topics)

    # raw_fallback chunk 중 명확한 topic mismatch 개수.
    # ratio 는 "topic 정보가 명확한 (non-generic) raw_fallback" 만을 분모로 둔다.
    # common / general / unknown 같은 generic topic chunk 가 분모에 섞이면 mismatch
    # 강도가 희석되기 때문이다.
    # 예) 시나리오 A:
    #   raw_fallback = [kakao, kakao, common]
    #   non-generic = 2 (kakao 두 개), mismatch = 2 → ratio = 1.0  (warning 켜짐)
    # 시나리오 C (query_topic 미지정 등):
    #   non-generic = 0 → ratio = 0.0 (warning 안 켜짐)
    mismatch_count = 0
    non_generic_count = 0
    if query_topic_clear and raw_fallback_chunks:
        for c in raw_fallback_chunks:
            try:
                md = getattr(c, "metadata", {}) or {}
                topics = list(md.get("topic_tags") or [])
                primary_topic = md.get("primary_topic")
                if primary_topic and primary_topic not in topics:
                    topics = topics + [primary_topic]
                topics_lower = [
                    str(t).strip().lower() for t in topics if t
                ]
                non_generic = [
                    t for t in topics_lower if t not in _GENERIC_TOPICS
                ]
                if not non_generic:
                    # topic 정보가 없거나 generic 만 있으면 분모에 포함하지 않는다.
                    # (mismatch 로도 잡지 않는다 — is_clear_topic_mismatch 동일.)
                    continue
                non_generic_count += 1
                if is_clear_topic_mismatch(c, query_topics=query_topics):
                    mismatch_count += 1
            except Exception:  # noqa: BLE001
                # 진단 보조 함수의 어떤 예외도 답변 흐름을 막지 않는다.
                continue

    mismatch_ratio = (
        float(mismatch_count) / float(non_generic_count)
        if non_generic_count > 0
        else 0.0
    )

    primary_evidence_available = (primary_count > 0) or (raw_evidence_count > 0)
    normalized_document_available = normalized_document_candidate_count > 0

    # raw_fallback_only_reason — 디버깅 / Slack debug 표시용
    if not raw_fallback_only:
        raw_fallback_only_reason: Optional[str] = None
    elif normalized_document_available:
        # 정규화 문서가 후보로는 잡혔지만 primary 로 승격되지 못함 (예: topic mismatch demote)
        raw_fallback_only_reason = "no_primary_normalized_document"
    else:
        raw_fallback_only_reason = "no_normalized_document_candidate"

    # weak_evidence_warning — 보수적인 조건
    weak_evidence_warning = bool(
        raw_fallback_only
        and query_topic_clear
        and raw_fallback_count >= _WEAK_EVIDENCE_MIN_RAW_FALLBACK
        and mismatch_ratio >= _WEAK_EVIDENCE_TOPIC_MISMATCH_RATIO
    )

    # evidence_strength 분류
    if generation_skipped:
        evidence_strength = "insufficient"
    elif primary_count > 0:
        evidence_strength = "strong"
    elif raw_evidence_count > 0:
        evidence_strength = "medium"
    elif raw_fallback_only:
        # raw_fallback 만 있는 경우 — mismatch 비율로 weak / insufficient 구분
        if query_topic_clear and mismatch_ratio >= _WEAK_EVIDENCE_TOPIC_MISMATCH_RATIO:
            evidence_strength = "insufficient"
        elif raw_fallback_count > 0 and mismatch_count == 0 and query_topic_clear:
            # 모두 topic match (또는 generic) 인 raw_fallback 만 있으면 medium 으로 본다.
            evidence_strength = "medium"
        else:
            evidence_strength = "weak"
    else:
        evidence_strength = "insufficient"

    return {
        "raw_fallback_only": raw_fallback_only,
        "raw_fallback_only_reason": raw_fallback_only_reason,
        "raw_fallback_topic_mismatch_count": int(mismatch_count),
        "raw_fallback_topic_mismatch_ratio": round(float(mismatch_ratio), 6),
        "primary_evidence_available": bool(primary_evidence_available),
        "normalized_document_available": bool(normalized_document_available),
        "weak_evidence_warning": bool(weak_evidence_warning),
        "evidence_strength": evidence_strength,
    }


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
                # MVP 2차 Step 2: topic-aware 격하 진단
                "topic_mismatch_demoted_count": 0,
                # MVP 2차 Step 4: Raw Fallback 오남용 방지 진단 — empty 케이스 default.
                "raw_fallback_only": False,
                "raw_fallback_only_reason": None,
                "raw_fallback_topic_mismatch_count": 0,
                "raw_fallback_topic_mismatch_ratio": 0.0,
                "primary_evidence_available": False,
                "normalized_document_available": False,
                "weak_evidence_warning": False,
                "evidence_strength": "insufficient",
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
        raw_fallback_chunks: List[RetrievedChunk] = list(groups["raw_fallback"])

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

        # MVP 2차 Step 4: Raw Fallback 오남용 방지 진단 계산.
        # 이번 Step 은 답변 흐름을 막지 않고 진단 가시성만 강화한다.
        step4_diag = _summarize_raw_fallback_policy(
            primary_count=primary_card_count,
            raw_evidence_count=raw_evidence_count,
            raw_fallback_chunks=raw_fallback_chunks,
            normalized_document_candidate_count=int(
                summary.get("normalized_document_candidate_count") or 0
            ),
            query_topics=list(summary.get("query_topics") or []),
            generation_skipped=generation_skipped,
        )

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
                # MVP 2차 Step 2: topic-aware 격하 진단
                "topic_mismatch_demoted_count": summary.get(
                    "topic_mismatch_demoted_count"
                ),
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
                # MVP 2차 Step 4: Raw Fallback 오남용 방지 진단
                "raw_fallback_only": step4_diag["raw_fallback_only"],
                "raw_fallback_only_reason": step4_diag["raw_fallback_only_reason"],
                "raw_fallback_topic_mismatch_count": step4_diag[
                    "raw_fallback_topic_mismatch_count"
                ],
                "raw_fallback_topic_mismatch_ratio": step4_diag[
                    "raw_fallback_topic_mismatch_ratio"
                ],
                "primary_evidence_available": step4_diag[
                    "primary_evidence_available"
                ],
                "normalized_document_available": step4_diag[
                    "normalized_document_available"
                ],
                "weak_evidence_warning": step4_diag["weak_evidence_warning"],
                "evidence_strength": step4_diag["evidence_strength"],
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
            # MVP 2차 Step 2: topic-aware 격하 진단
            "topic_mismatch_demoted_count": int(
                summary.get("topic_mismatch_demoted_count") or 0
            ),
            # ----------------------------------------------------------------
            # MVP 2차 Step 4: Raw Fallback 오남용 방지 진단
            #
            # raw_fallback 만으로 답변하거나 topic mismatch raw_fallback 이 과도하게
            # 사용되는 상황을 외부 (Slack debug / Streamlit) 에서 명확히 볼 수
            # 있도록 top-level 에 노출한다. 답변 자체를 막지는 않는다 — 약한
            # 근거 상태를 사용자가 알아보고 신중하게 해석하도록 돕는 용도.
            # ----------------------------------------------------------------
            "raw_fallback_only": step4_diag["raw_fallback_only"],
            "raw_fallback_only_reason": step4_diag["raw_fallback_only_reason"],
            "raw_fallback_topic_mismatch_count": step4_diag[
                "raw_fallback_topic_mismatch_count"
            ],
            "raw_fallback_topic_mismatch_ratio": step4_diag[
                "raw_fallback_topic_mismatch_ratio"
            ],
            "primary_evidence_available": step4_diag["primary_evidence_available"],
            "normalized_document_available": step4_diag[
                "normalized_document_available"
            ],
            "weak_evidence_warning": step4_diag["weak_evidence_warning"],
            "evidence_strength": step4_diag["evidence_strength"],
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
