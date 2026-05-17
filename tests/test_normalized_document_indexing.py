from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.normalization.pipeline_integration import normalized_documents_to_chunks
from src.rag.retriever import Retriever
from src.schemas import Chunk, NormalizedDocument, RetrievedChunk
from src.storage.vector_store import _metadata_for_chunk


class FakeEmbedder:
    provider = "fake"
    model_name = "fake-model"

    def embed_query(self, text: str) -> List[float]:  # noqa: ARG002
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    def __init__(self, results: List[RetrievedChunk]) -> None:
        self._results = results

    def search(
        self,
        query_embedding: List[float],  # noqa: ARG002
        top_k: int = 8,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        out = list(self._results)
        if filters:
            cat = filters.get("uploaded_category")
            if cat:
                out = [c for c in out if c.uploaded_category == cat]
        return out[: int(top_k)]

    def get_children(
        self,
        parent_ids: List[str],  # noqa: ARG002
        limit: int = 20,  # noqa: ARG002
    ) -> List[RetrievedChunk]:
        return []


def _make_normalized_document(
    *,
    doc_type: str = "action_item",
    answer_use_cases: Optional[List[str]] = None,
) -> NormalizedDocument:
    doc = NormalizedDocument(
        card_id="nd_meta_action_001",
        card_type=doc_type,
        title="메타 피드광고 후속 조치",
        summary="세팅 이후 확인해야 할 일을 정리한다.",
        source_file_name="meta_thread.txt",
        source_file_hash="f" * 64,
        source_category="slack",
        source_type="slack",
        document_date="2026-05-17",
        display_date="해당 업무일",
        primary_topic="meta",
        topic_tags=["meta", "issue"],
        task_type="setup",
        steps=["소재 확인", "URL 확인"],
        answer_use_cases=list(answer_use_cases or ["checklist", "history_lookup"]),
        metadata={"prompt_version": "slack_thread_v1_5", "model_name": "fake"},
        parent_raw_chunk_ids=["chunk_raw_001"],
    )
    doc.sanitized_markdown = doc.to_markdown()
    return doc


def _retrieved_from_chunk(chunk: Chunk, *, score: float = 0.95) -> RetrievedChunk:
    md = _metadata_for_chunk(chunk)
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        document_id=str(md.get("document_id", "")),
        file_name=str(md.get("file_name", "")),
        source_type=str(md.get("source_type", "")),
        uploaded_category=str(md.get("uploaded_category", "")),
        section_title=md.get("section_title"),
        content_type=str(md.get("content_type", "text")),
        content=chunk.content,
        score=score,
        final_score=score,
        parent_chunk_id=md.get("parent_chunk_id"),
        metadata=dict(md),
    )


def test_normalized_document_chunk_metadata_is_indexable_as_normalized_document() -> None:
    doc = _make_normalized_document(doc_type="action_item")

    chunks = normalized_documents_to_chunks([doc], document_id="doc_meta_thread")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.content_type == "normalized_document"
    assert chunk.source_type == "llm_normalized"
    assert chunk.metadata["content_type"] == "normalized_document"
    assert chunk.metadata["source_type"] == "llm_normalized"
    assert chunk.metadata["normalized_document_type"] == "action_item"
    assert chunk.metadata["card_type"] == "action_item"
    assert chunk.metadata["answer_use_cases"] == "checklist,history_lookup"
    assert chunk.metadata["primary_topic"] == "meta"
    assert chunk.metadata["task_type"] == "setup"
    assert chunk.metadata["source_file_name"] == "meta_thread.txt"
    assert chunk.metadata["source_category"] == "slack"


def test_vector_store_metadata_preserves_chunk_top_level_content_type() -> None:
    chunk = Chunk(
        chunk_id="chunk_doc_norm_0001",
        document_id="doc_meta",
        chunk_index=0,
        source_type="llm_normalized",
        uploaded_category="slack",
        file_name="meta_thread.txt",
        content="normalized body",
        clean_content="normalized body",
        embedding_text="normalized body",
        parent_chunk_id=None,
        section_title="메타 피드광고",
        content_type="normalized_document",
        metadata={"normalized_document_type": "status_update", "card_type": "status_update"},
    )

    md = _metadata_for_chunk(chunk)

    assert md["content_type"] == "normalized_document"
    assert md["source_type"] == "llm_normalized"
    assert md["document_id"] == "doc_meta"
    assert md["file_name"] == "meta_thread.txt"
    assert md["uploaded_category"] == "slack"
    assert md["normalized_document_type"] == "status_update"
    assert md["card_type"] == "status_update"


def test_retriever_counts_indexed_normalized_document_candidate() -> None:
    doc = _make_normalized_document(doc_type="status_update", answer_use_cases=["summary"])
    chunk = normalized_documents_to_chunks([doc], document_id="doc_meta_thread")[0]
    retrieved = _retrieved_from_chunk(chunk)
    retriever = Retriever(embedder=FakeEmbedder(), vector_store=FakeVectorStore([retrieved]))

    details = retriever.retrieve_with_details(
        "메타 피드광고 핵심만 요약해줘",
        top_k=3,
        with_parent_children=False,
        use_mmr=False,
        min_similarity=0.0,
        min_final=0.0,
    )

    assert details.summary["normalized_document_candidate_count"] == 1
    assert details.summary["raw_candidate_count"] == 0
    assert details.summary["normalized_document_count"] >= 1
    assert details.passed[0].content_type == "normalized_document"
    assert details.passed[0].metadata["retrieval_role"] == "primary_card"


def test_raw_text_chunk_remains_raw_candidate() -> None:
    raw = RetrievedChunk(
        chunk_id="chunk_raw_001",
        document_id="doc_raw",
        file_name="guide.txt",
        source_type="guide",
        uploaded_category="guide",
        section_title="raw",
        content_type="text",
        content="raw body",
        score=0.95,
        final_score=0.95,
        parent_chunk_id=None,
        metadata={"content_type": "text", "source_type": "guide", "primary_topic": "meta"},
    )
    retriever = Retriever(embedder=FakeEmbedder(), vector_store=FakeVectorStore([raw]))

    details = retriever.retrieve_with_details(
        "메타 피드광고 셋팅 방법",
        top_k=3,
        with_parent_children=False,
        use_mmr=False,
        min_similarity=0.0,
        min_final=0.0,
    )

    assert details.summary["normalized_document_candidate_count"] == 0
    assert details.summary["raw_candidate_count"] == 1
    assert details.passed[0].metadata["retrieval_role"] != "primary_card"
