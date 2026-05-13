"""
src.slack_bot.qa_adapter
========================
Slack Bot 과 기존 ``src.rag.qa_pipeline.QAPipeline`` 사이의 얇은 adapter.

Slack handler 가 qa_pipeline 의 내부 구조 (RetrievedChunk dataclass, 진단
필드, anonymizer 동작 등) 를 직접 알지 않도록 한 곳에서 변환한다. Slack 쪽은
이 모듈이 반환하는 dict 만 사용한다.

Slack Bot 안에 retriever / reranker / prompt_builder 로직을 새로 만들지
않는다. 이 모듈은 항상 기존 ``QAPipeline.ask`` 를 호출한다.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.config import settings
from src.logger import get_logger
from src.preprocessing.anonymizer import anonymize_text
from src.schemas import RetrievedChunk

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# QAPipeline 인스턴스 캐시
# ---------------------------------------------------------------------------
# QAPipeline 은 embedder/retriever/generator 를 무겁게 들고 있어 매 메시지마다
# 새로 만들면 비용이 크다. 프로세스 단위로 1 개만 유지한다.
_PIPELINE_SINGLETON: Optional[Any] = None


def _get_pipeline() -> Any:
    """
    기본 ``QAPipeline`` 인스턴스를 lazy 하게 생성/캐시한다.

    import 자체를 함수 안에서 한다 — Slack Bot 이 비활성화된 환경에서도
    이 모듈만 import 되는 경우가 있어 무거운 의존성 (chromadb 등) 로딩을
    실제 사용 시점까지 미룬다.
    """
    global _PIPELINE_SINGLETON
    if _PIPELINE_SINGLETON is None:
        from src.rag.qa_pipeline import QAPipeline  # 지연 import

        log.info("Slack QA adapter: QAPipeline 인스턴스를 새로 생성합니다.")
        _PIPELINE_SINGLETON = QAPipeline()
    return _PIPELINE_SINGLETON


def reset_pipeline_for_tests() -> None:
    """테스트 격리용. 캐시된 ``QAPipeline`` 을 비운다."""
    global _PIPELINE_SINGLETON
    _PIPELINE_SINGLETON = None


# ---------------------------------------------------------------------------
# RetrievedChunk → dict 변환 (Slack formatter 가 사용)
# ---------------------------------------------------------------------------
def _chunk_label(chunk: RetrievedChunk) -> str:
    """채널에 노출할 안전한 chunk 라벨 — 파일명/섹션 위주."""
    md = chunk.metadata or {}
    parts: List[str] = []
    if chunk.file_name:
        parts.append(str(chunk.file_name))
    if chunk.section_title:
        parts.append(str(chunk.section_title))
    if not parts:
        # 최소한 카테고리/타입 정보라도 노출
        cat = chunk.uploaded_category or chunk.source_type or "unknown"
        parts.append(str(cat))
    label = " · ".join(parts)
    # 정규화 문서면 type 도 함께 표시
    nd_type = (
        md.get("normalized_document_type")
        or md.get("card_type")
    )
    if nd_type:
        label = f"{label} ({nd_type})"
    return label


def _chunk_safe_preview(chunk: RetrievedChunk, max_chars: int = 240) -> str:
    """
    Slack 메시지에 노출 가능한 짧은 preview.

    raw 원문은 직접 노출하지 않는다. 항상 anonymize_output 정책을 통과한
    sanitized_content 를 우선 사용하고, 없으면 anonymize_text 로 마스킹한 뒤
    잘라낸다. raw 원문 전체를 그대로 복사하지 않는다.
    """
    md = chunk.metadata or {}
    if settings.anonymize_output:
        text = md.get("sanitized_content") or anonymize_text(chunk.content or "")
    else:
        text = chunk.content or ""
    text = (text or "").strip()
    if not text:
        return ""
    text = " ".join(text.split())
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def _serialize_chunks(chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
    """
    Slack formatter 가 사용할 source dict 로 직렬화한다.

    MVP 2차 Step 1 (Retrieval Diagnostics 강화):
    Slack 기본 출력에는 영향을 주지 않지만, ``--debug`` 모드에서 source 별
    진단 정보를 짧게 노출할 수 있도록 필드를 보강한다 (검색 정책은 변경하지
    않음).
    """
    out: List[Dict[str, Any]] = []
    for c in chunks or []:
        md = c.metadata or {}
        # similarity_score 는 rerank_simple 가 metadata 에 채워둔 진단값.
        similarity_score = md.get("similarity_score")
        if similarity_score is None:
            similarity_score = c.score
        out.append(
            {
                "label": _chunk_label(c),
                "preview": _chunk_safe_preview(c),
                "file_name": c.file_name,
                "section_title": c.section_title,
                "content_type": c.content_type,
                "source_type": c.source_type,
                # source_category 는 source_type 의 사용자 친화적 표현으로
                # uploaded_category 와 동일한 값을 alias 로 둔다.
                "source_category": c.uploaded_category,
                "uploaded_category": c.uploaded_category,
                "score": c.score,
                "similarity_score": similarity_score,
                "final_score": c.final_score,
                "normalized_document_type": (
                    md.get("normalized_document_type") or md.get("card_type")
                ),
                "primary_topic": md.get("primary_topic"),
                "topic_tags": md.get("topic_tags") or [],
                # 검색 진단 metadata (UI/debug 용)
                "retrieval_role": md.get("retrieval_role"),
                "topic_match": md.get("topic_match"),
                "date_match": md.get("date_match"),
                "normalized_document_boost": (
                    md.get("normalized_document_boost")
                    or md.get("knowledge_card_boost")
                ),
                "topic_boost": md.get("topic_boost"),
                "date_boost": md.get("date_boost"),
                # MVP 2차 Step 2: topic-aware 격하 여부
                "topic_mismatch_demoted": bool(md.get("topic_mismatch_demoted")),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Slack용 메인 진입점
# ---------------------------------------------------------------------------
def answer_slack_question(
    question: str,
    user_id: Optional[str] = None,
    channel_id: Optional[str] = None,
    *,
    pipeline: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Slack 에서 들어온 질문을 기존 QA Pipeline 에 전달하고, Slack formatter 가
    바로 사용할 수 있는 dict 를 반환한다.

    Parameters
    ----------
    question : str
        bot mention 을 제거한, 사용자가 실제로 던진 질문 텍스트.
    user_id : str, optional
        Slack user id (감사 로그/디버깅 용도, qa_pipeline 동작에는 영향 없음).
    channel_id : str, optional
        Slack channel id (감사 로그/디버깅 용도).
    pipeline : QAPipeline, optional
        테스트에서 mock 을 주입하기 위한 hook. 지정하지 않으면 모듈 캐시된
        기본 인스턴스를 사용한다.

    Returns
    -------
    dict
        Slack formatter 가 사용하는 표준 키:
            - ``answer``                            (str)
            - ``sources``                           (list[dict])
            - ``diagnostics``                       (dict)
            - ``answer_mode``                       (str)
            - ``primary_normalized_document_count`` (int)
            - ``raw_evidence_count``                (int)
            - ``raw_fallback_count``                (int)
    """
    log.info(
        "Slack QA adapter: question received "
        "(user_id=%s, channel_id=%s, len=%d)",
        user_id, channel_id, len(question or ""),
    )

    pipe = pipeline if pipeline is not None else _get_pipeline()
    raw = pipe.ask(question)

    # qa_pipeline 이 반환하는 dict 의 키를 그대로 활용 (없으면 안전 default).
    answer = raw.get("answer", "") or ""
    answer_mode = raw.get("answer_mode") or "insufficient_evidence"

    primary_chunks: List[RetrievedChunk] = list(
        raw.get("primary_normalized_documents")
        or raw.get("primary_cards")
        or []
    )
    raw_evidence_chunks: List[RetrievedChunk] = list(raw.get("raw_evidence") or [])
    raw_fallback_chunks: List[RetrievedChunk] = list(raw.get("raw_fallback") or [])

    primary_count = int(
        raw.get("primary_normalized_document_count")
        or raw.get("primary_card_count")
        or len(primary_chunks)
    )
    raw_evidence_count = int(
        raw.get("raw_evidence_count") or len(raw_evidence_chunks)
    )
    raw_fallback_count = int(
        raw.get("raw_fallback_count") or len(raw_fallback_chunks)
    )

    sources: Dict[str, List[Dict[str, Any]]] = {
        "primary_normalized_documents": _serialize_chunks(primary_chunks),
        "raw_evidence": _serialize_chunks(raw_evidence_chunks),
        "raw_fallback": _serialize_chunks(raw_fallback_chunks),
    }

    # ----------------------------------------------------------------------
    # MVP 2차 Step 1: Retrieval Diagnostics 강화
    #
    # qa_pipeline 이 top-level 로 노출하기 시작한 진단 필드를 그대로 전달한다.
    # Slack 기본 답변에는 영향이 없고, ``--debug`` 모드에서만 formatter 가
    # 이 dict 의 값을 짧게 보여준다.
    # ----------------------------------------------------------------------
    retrieval_summary: Dict[str, Any] = raw.get("retrieval_summary") or {}

    def _first_or_none(value: Any) -> Any:
        if isinstance(value, list) and value:
            return value[0]
        return value if value not in (None, []) else None

    diagnostics: Dict[str, Any] = {
        "answer_mode": answer_mode,
        "primary_normalized_document_count": primary_count,
        "raw_evidence_count": raw_evidence_count,
        "raw_fallback_count": raw_fallback_count,
        "generation_skipped": bool(raw.get("generation_skipped")),
        "skip_reason": raw.get("skip_reason"),
        "model_name": raw.get("model_name") or "",
        "embedding_provider": raw.get("embedding_provider") or "",
        "embedding_model": raw.get("embedding_model") or "",
        "rewritten_query": raw.get("rewritten_query"),
        "answer_format_label": raw.get("answer_format_label"),
        # MVP 2차 Step 1: query / candidate 진단 (Slack debug 표시용)
        "query_topic": raw.get("query_topic") or _first_or_none(
            retrieval_summary.get("query_topics")
        ),
        "query_topics": list(raw.get("query_topics") or retrieval_summary.get("query_topics") or []),
        "query_intent": list(raw.get("query_intent") or retrieval_summary.get("query_intent") or []),
        "query_date": raw.get("query_date") or retrieval_summary.get("query_date"),
        "retrieved_count": int(
            raw.get("retrieved_count")
            or retrieval_summary.get("retrieved_count")
            or retrieval_summary.get("candidate_count")
            or 0
        ),
        "passed_count": int(
            raw.get("passed_count") or retrieval_summary.get("passed_count") or 0
        ),
        "topic_mismatch_count": int(
            raw.get("topic_mismatch_count")
            or retrieval_summary.get("topic_mismatch_count")
            or 0
        ),
        "normalized_document_candidate_count": int(
            raw.get("normalized_document_candidate_count")
            or retrieval_summary.get("normalized_document_candidate_count")
            or 0
        ),
        "raw_candidate_count": int(
            raw.get("raw_candidate_count")
            or retrieval_summary.get("raw_candidate_count")
            or 0
        ),
        # MVP 2차 Step 2: topic-aware 격하 진단
        "topic_mismatch_demoted_count": int(
            raw.get("topic_mismatch_demoted_count")
            or retrieval_summary.get("topic_mismatch_demoted_count")
            or 0
        ),
    }

    return {
        "answer": answer,
        "sources": sources,
        "diagnostics": diagnostics,
        "answer_mode": answer_mode,
        "primary_normalized_document_count": primary_count,
        "raw_evidence_count": raw_evidence_count,
        "raw_fallback_count": raw_fallback_count,
    }
