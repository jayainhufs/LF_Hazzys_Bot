"""
pipeline_integration.py
=======================
LLM 기반 KnowledgeCard 정규화를 ingest pipeline 에 연결하는 helper 모듈.

설계 의도
---------
- ``src/pipeline.py`` 안에 길게 분기를 늘리지 않고, branch 진입/실행을
  helper 함수로 분리해 단위 테스트가 쉬운 형태로 둔다.
- ``ENABLE_LLM_NORMALIZATION`` 이 false 일 때는 어떤 함수도 호출되지 않으며,
  기존 raw ingest 흐름과 완전히 동일하게 동작해야 한다 (Task 4 제약).
- LLM / vector store / file 저장이 어디에서 실패해도 raw indexing 흐름이
  죽지 않도록, ``run_normalization_branch`` 는 raise 하지 않고 결과 dict
  를 반환한다. 모든 실패 경로는 log.warning 으로만 남긴다.
- retrieval / QA / Streamlit UI 변경은 다른 Task 에서 처리한다. 이번 Task 에서는
  단지 KnowledgeCard 가 chunk 형태로 저장될 수 있는 통로만 만든다.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from src.config import settings as default_settings
from src.logger import get_logger
from src.normalization.guide_normalizer import GuideKnowledgeNormalizer
from src.normalization.normalization_store import NormalizationStore
from src.normalization.slack_normalizer import SlackThreadKnowledgeNormalizer
from src.schemas import Chunk, KnowledgeCard
from src.utils.hash_utils import short_hash

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Branch decision
# ---------------------------------------------------------------------------
_GUIDE_CATEGORIES = {"guide"}
_GUIDE_SOURCE_TYPES = {"guide"}
_SLACK_CATEGORIES = {"slack", "slack_manual"}
_SLACK_SOURCE_TYPES = {"slack", "slack_manual"}


def should_normalize_file(
    *,
    uploaded_category: Optional[str],
    source_type: Optional[str],
    settings_obj: Any = None,  # noqa: ARG001 - 향후 카테고리 확장 시 사용
) -> Optional[str]:
    """파일이 normalization 대상인지 판정한다.

    Returns
    -------
    "guide" : Guide normalizer 사용
    "slack" : Slack normalizer 사용
    None    : normalization skip (kakao / excel / word / etc.)
    """
    cat = (uploaded_category or "").strip().lower()
    src = (source_type or "").strip().lower()
    if cat in _GUIDE_CATEGORIES or src in _GUIDE_SOURCE_TYPES:
        return "guide"
    if cat in _SLACK_CATEGORIES or src in _SLACK_SOURCE_TYPES:
        return "slack"
    return None


# ---------------------------------------------------------------------------
# Input extraction
# ---------------------------------------------------------------------------
def extract_normalization_inputs(
    parsed_sections: Iterable[Any],
) -> Dict[str, Any]:
    """parsed_sections 에서 normalizer 에 보낼 텍스트 + 메타데이터를 모은다.

    - Slack parser v2 가 채워준 ``sanitized_content`` / ``topic_tags`` /
      ``todo_phase`` / ``parser_format`` / ``document_date`` / ``display_date``
      를 우선 사용하고, 없으면 raw content 로 fallback 한다.
    - Guide 등 sanitized_content 가 없는 source 는 그냥 ``ps.content`` 가 사용된다.
    """
    texts: List[str] = []
    topic_tags: List[str] = []
    seen_tags: set = set()
    todo_phase: Optional[str] = None
    parser_format: Optional[str] = None
    document_date: Optional[str] = None
    display_date: Optional[str] = None

    for ps in parsed_sections or []:
        md = getattr(ps, "metadata", None) or {}
        content = getattr(ps, "content", "") or ""
        sanitized = md.get("sanitized_content") if isinstance(md, dict) else None
        text = sanitized if (isinstance(sanitized, str) and sanitized.strip()) else content
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())

        if isinstance(md, dict):
            for tag in md.get("topic_tags") or []:
                if isinstance(tag, str) and tag.strip() and tag not in seen_tags:
                    seen_tags.add(tag)
                    topic_tags.append(tag)
            if not todo_phase and md.get("todo_phase"):
                todo_phase = str(md.get("todo_phase"))
            if not parser_format and md.get("parser_format"):
                parser_format = str(md.get("parser_format"))
            if not document_date and md.get("document_date"):
                document_date = str(md.get("document_date"))
            if not display_date and md.get("display_date"):
                display_date = str(md.get("display_date"))

    return {
        "text": "\n\n".join(texts).strip(),
        "topic_tags": topic_tags,
        "todo_phase": todo_phase,
        "parser_format": parser_format,
        "document_date": document_date,
        "display_date": display_date,
    }


# ---------------------------------------------------------------------------
# Parent-child link
# ---------------------------------------------------------------------------
def attach_parent_raw_chunk_ids(
    cards: List[KnowledgeCard],
    *,
    raw_chunks: List[Any],
    top_k: int = 3,
) -> None:
    """raw_chunks 의 앞쪽 top_k 개 chunk_id 를 모든 card.parent_raw_chunk_ids 에 채워 넣는다.

    Task 4 단계에서는 정확한 semantic 매칭이 아니라 "같은 파일의 앞쪽 raw 청크" 를
    parent 로 단순 연결한다. 정교한 매칭은 이후 Task 에서 다룬다.

    이미 채워진 카드의 parent_raw_chunk_ids 는 덮어쓰지 않는다.
    """
    if not cards or not raw_chunks:
        return
    k = max(int(top_k), 0)
    if k <= 0:
        return
    parent_ids: List[str] = []
    for c in raw_chunks[:k]:
        cid = getattr(c, "chunk_id", None)
        if isinstance(cid, str) and cid:
            parent_ids.append(cid)
    if not parent_ids:
        return
    for card in cards:
        if not card.parent_raw_chunk_ids:
            card.parent_raw_chunk_ids = list(parent_ids)


# ---------------------------------------------------------------------------
# KnowledgeCard → Chunk
# ---------------------------------------------------------------------------
def knowledge_cards_to_chunks(
    cards: List[KnowledgeCard],
    *,
    document_id: str,
    settings_obj: Any = None,
) -> List[Chunk]:
    """KnowledgeCard 리스트를 ChromaDB 에 적재 가능한 Chunk 리스트로 변환한다.

    metadata 는 Task 6 의 retrieval 우선순위 적용 시 사용할 수 있는 형태로
    정리한다 (이번 Task 에서는 boost / 우선순위 변경 자체는 적용하지 않는다).
    """
    s = settings_obj or default_settings
    out: List[Chunk] = []
    for idx, card in enumerate(cards or []):
        if card is None:
            continue
        try:
            if not card.validate_minimum():
                log.warning(
                    "KnowledgeCard validate_minimum 실패 → chunk 변환 skip: card_id=%s",
                    getattr(card, "card_id", "?"),
                )
                continue
        except Exception as e:  # noqa: BLE001
            log.warning("KnowledgeCard 검증 중 예외 (skip): %s", e)
            continue
        body = (card.sanitized_markdown or card.to_markdown() or "").strip()
        if not body:
            continue

        topic_tags_str = ",".join([t for t in (card.topic_tags or []) if isinstance(t, str) and t])
        parent_str = ",".join(
            [p for p in (card.parent_raw_chunk_ids or []) if isinstance(p, str) and p]
        )
        card_meta = card.metadata if isinstance(card.metadata, dict) else {}

        chunk_id_suffix = (card.card_id or f"card{idx:03d}").replace(" ", "_")[-24:]
        chunk = Chunk(
            chunk_id=f"chunk_{document_id}_norm_{idx:04d}_{chunk_id_suffix}",
            document_id=document_id,
            chunk_index=idx,
            source_type="llm_normalized",
            uploaded_category=card.source_category or "",
            file_name=card.source_file_name or "",
            content=body,
            clean_content=body,
            embedding_text=body,
            parent_chunk_id=None,
            section_title=card.title or None,
            content_type="knowledge_card",
            metadata={
                "card_id": card.card_id or "",
                "card_type": card.card_type or "",
                "primary_topic": card.primary_topic or "",
                "topic_tags": topic_tags_str,
                "task_type": card.task_type or "",
                "document_date": card.document_date or "",
                "display_date": card.display_date or "",
                "source_file_hash": card.source_file_hash or "",
                "parent_raw_chunk_ids": parent_str,
                "source_weight": float(getattr(s, "normalization_card_source_weight", 1.25)),
                "normalized": True,
                "prompt_version": str(card_meta.get("prompt_version", "")),
                "model_name": str(card_meta.get("model_name", "")),
                "todo_phase": str(card_meta.get("todo_phase", "") or ""),
                "parser_format": str(card_meta.get("parser_format", "") or ""),
                "section_title": card.title or "",
            },
        )
        out.append(chunk)
    return out


# ---------------------------------------------------------------------------
# Normalizer dispatch
# ---------------------------------------------------------------------------
def normalize_document_for_pipeline(
    *,
    kind: str,
    text: str,
    file_name: str,
    file_hash_value: str,
    source_category: str,
    source_type: str,
    document_date: Optional[str] = None,
    display_date: Optional[str] = None,
    topic_tags: Optional[List[str]] = None,
    todo_phase: Optional[str] = None,
    parser_format: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    gemini_client: Any,
    store: NormalizationStore,
    settings_obj: Any = None,
) -> List[KnowledgeCard]:
    """kind ('guide' | 'slack') 에 따라 적절한 normalizer 를 호출한다.

    raise 가 발생하면 ``run_normalization_branch`` 에서 잡아 raw ingest 가
    계속 진행되도록 한다.
    """
    s = settings_obj or default_settings
    file_hash_short = short_hash(file_hash_value, length=10) if file_hash_value else "unknown"

    if kind == "guide":
        normalizer = GuideKnowledgeNormalizer(
            gemini_client=gemini_client, store=store, settings=s
        )
        return normalizer.normalize_guide_text(
            text=text,
            file_name=file_name,
            file_hash=file_hash_value,
            file_hash_short=file_hash_short,
            source_category=source_category,
            source_type=source_type,
            document_date=document_date,
            display_date=display_date,
            metadata=metadata,
        )

    if kind == "slack":
        normalizer = SlackThreadKnowledgeNormalizer(
            gemini_client=gemini_client, store=store, settings=s
        )
        return normalizer.normalize_slack_thread_text(
            text=text,
            file_name=file_name,
            file_hash=file_hash_value,
            file_hash_short=file_hash_short,
            source_category=source_category,
            source_type=source_type,
            document_date=document_date,
            display_date=display_date,
            topic_tags=topic_tags,
            todo_phase=todo_phase,
            parser_format=parser_format,
            metadata=metadata,
        )

    raise ValueError(f"Unknown normalization kind: {kind}")


# ---------------------------------------------------------------------------
# Branch entry
# ---------------------------------------------------------------------------
def run_normalization_branch(
    *,
    document: Any,
    parsed_sections: List[Any],
    raw_chunks: List[Any],
    embedder: Any,
    vector_store: Any,
    document_store: Any,
    gemini_client: Any = None,
    normalization_store: Optional[NormalizationStore] = None,
    settings_obj: Any = None,
) -> Dict[str, Any]:
    """ingest_file 의 LLM normalization branch entry.

    - 절대 raise 하지 않는다. 어떤 단계에서 실패해도 result dict 만 반환한다.
    - cache hit 이면 LLM 호출 없이 KnowledgeCard 를 복원해 chunk 화한다.
    - vector store / document store 저장 실패도 raw indexing 을 멈추지 않는다.

    Returns
    -------
    dict
        {
            "kind": "guide" | "slack" | None,
            "card_count": int,
            "chunks_added": int,
            "skipped_reason": Optional[str],
        }
    """
    s = settings_obj or default_settings
    result: Dict[str, Any] = {
        "kind": None,
        "card_count": 0,
        "chunks_added": 0,
        "skipped_reason": None,
    }

    kind = should_normalize_file(
        uploaded_category=getattr(document, "uploaded_category", None),
        source_type=getattr(document, "source_type", None),
        settings_obj=s,
    )
    if not kind:
        result["skipped_reason"] = "category/source_type 이 normalization 대상이 아닙니다."
        return result
    result["kind"] = kind

    try:
        store_obj = normalization_store or NormalizationStore()
    except Exception as e:  # noqa: BLE001
        log.warning("NormalizationStore 생성 실패 (skip): %s", e)
        result["skipped_reason"] = f"store init 실패: {e}"
        return result

    if gemini_client is None:
        try:
            from src.rag.gemini_client import get_default_client

            gemini_client = get_default_client()
        except Exception as e:  # noqa: BLE001
            log.warning("Gemini client 초기화 실패 (skip): %s", e)
            result["skipped_reason"] = f"gemini client 초기화 실패: {e}"
            return result

    inputs = extract_normalization_inputs(parsed_sections)
    text = inputs["text"]
    if not text or not text.strip():
        result["skipped_reason"] = "정규화할 텍스트가 비어 있습니다."
        return result

    try:
        cards = normalize_document_for_pipeline(
            kind=kind,
            text=text,
            file_name=getattr(document, "file_name", "") or "",
            file_hash_value=getattr(document, "file_hash", "") or "",
            source_category=getattr(document, "uploaded_category", "") or "",
            source_type=getattr(document, "source_type", "") or "",
            document_date=inputs["document_date"],
            display_date=inputs["display_date"],
            topic_tags=inputs["topic_tags"] if kind == "slack" else None,
            todo_phase=inputs["todo_phase"] if kind == "slack" else None,
            parser_format=inputs["parser_format"] if kind == "slack" else None,
            metadata={
                "document_id": getattr(document, "document_id", "") or "",
            },
            gemini_client=gemini_client,
            store=store_obj,
            settings_obj=s,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("LLM normalization 실패 (raw ingest 는 계속 진행): %s", e)
        result["skipped_reason"] = f"LLM normalization 실패: {e}"
        return result

    result["card_count"] = len(cards or [])
    if not cards:
        result["skipped_reason"] = "LLM 결과 카드가 0개입니다."
        return result

    try:
        attach_parent_raw_chunk_ids(
            cards,
            raw_chunks=raw_chunks,
            top_k=int(getattr(s, "normalization_parent_raw_top_k", 3)),
        )
    except Exception as e:  # noqa: BLE001
        log.warning("parent raw chunk 연결 실패 (계속 진행): %s", e)

    norm_chunks = knowledge_cards_to_chunks(
        cards,
        document_id=getattr(document, "document_id", "") or "unknown_doc",
        settings_obj=s,
    )
    if not norm_chunks:
        result["skipped_reason"] = "knowledge_card → chunk 변환 결과가 0개입니다."
        return result

    try:
        norm_inputs = [c.embedding_text or c.content or "" for c in norm_chunks]
        norm_embeddings = embedder.embed_documents(norm_inputs)
    except Exception as e:  # noqa: BLE001
        log.warning("normalized chunk embedding 실패 (raw ingest 는 계속 진행): %s", e)
        result["skipped_reason"] = f"embedding 실패: {e}"
        return result

    if not norm_embeddings or len(norm_embeddings) != len(norm_chunks):
        log.warning(
            "normalized chunk embedding 결과 길이 불일치 (chunks=%d, embeddings=%d) → skip",
            len(norm_chunks),
            len(norm_embeddings or []),
        )
        result["skipped_reason"] = "embedding 결과 길이 불일치"
        return result

    try:
        n_added = vector_store.add_chunks(
            norm_chunks, norm_embeddings, skip_existing=True
        )
    except Exception as e:  # noqa: BLE001
        log.warning("normalized chunk vstore 저장 실패 (raw ingest 는 계속 진행): %s", e)
        result["skipped_reason"] = f"vstore add 실패: {e}"
        return result

    try:
        document_store.save_chunks(
            f"{getattr(document, 'document_id', '') or 'unknown_doc'}_norm",
            norm_chunks,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("normalized chunk JSONL 저장 실패: %s", e)

    result["chunks_added"] = int(n_added)
    log.info(
        "[normalized] %s | kind=%s, cards=%d, chunks_added=%d",
        getattr(document, "file_name", ""),
        kind,
        len(cards),
        int(n_added),
    )
    return result
