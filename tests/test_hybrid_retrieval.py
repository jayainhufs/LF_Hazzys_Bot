from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.config import settings
from src.rag.retriever import Retriever
from src.schemas import RetrievedChunk
from src.slack_bot import formatter, qa_adapter


class FakeEmbedder:
    provider = "fake"
    model_name = "fake-model"

    def embed_query(self, text: str) -> List[float]:  # noqa: ARG002
        return [0.1, 0.2, 0.3]


class FakeHybridVectorStore:
    def __init__(
        self,
        vector_results: List[RetrievedChunk],
        bm25_results: Optional[List[RetrievedChunk]] = None,
    ) -> None:
        self.vector_results = vector_results
        self.bm25_results = bm25_results or []
        self.search_calls = 0
        self.bm25_calls = 0

    def search(
        self,
        query_embedding: List[float],  # noqa: ARG002
        top_k: int = 8,
        filters: Optional[Dict[str, Any]] = None,  # noqa: ARG002
    ) -> List[RetrievedChunk]:
        self.search_calls += 1
        return list(self.vector_results)[: int(top_k)]

    def search_bm25(
        self,
        query: str,  # noqa: ARG002
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,  # noqa: ARG002
    ) -> List[RetrievedChunk]:
        self.bm25_calls += 1
        return list(self.bm25_results)[: int(top_k)]

    def get_children(
        self,
        parent_chunk_id: str,  # noqa: ARG002
        sheet_name: Optional[str] = None,  # noqa: ARG002
        limit: int = 3,  # noqa: ARG002
    ) -> List[RetrievedChunk]:
        return []


def _chunk(
    chunk_id: str,
    *,
    content: str,
    score: float,
    file_name: str = "guide.md",
    topic: str = "meta",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        file_name=file_name,
        source_type="guide",
        uploaded_category="guide",
        section_title="section",
        content_type="text",
        content=content,
        score=float(score),
        final_score=float(score),
        metadata={
            "source_weight": 1.0,
            "file_name": file_name,
            "source_type": "guide",
            "uploaded_category": "guide",
            "content_type": "text",
            "primary_topic": topic,
            "topic_tags": [topic],
        },
    )


def test_hybrid_disabled_keeps_vector_only_flow(monkeypatch):
    monkeypatch.setattr(settings, "hybrid_retrieval_enabled", False)
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    vector = [_chunk("vector_1", content="semantic match", score=0.6)]
    bm25 = [_chunk("bm25_1", content="ASC exact match", score=1.0)]
    store = FakeHybridVectorStore(vector, bm25)
    retriever = Retriever(embedder=FakeEmbedder(), vector_store=store)

    details = retriever.retrieve_with_details("ASC", top_k=5, max_per_file=10)

    assert store.search_calls == 1
    assert store.bm25_calls == 0
    assert [c.chunk_id for c in details.candidates] == ["vector_1"]
    assert details.summary["hybrid_retrieval_enabled"] is False
    assert details.summary["vector_candidate_count"] == 1
    assert details.summary["bm25_candidate_count"] == 0


def test_hybrid_enabled_merges_bm25_only_candidate(monkeypatch):
    monkeypatch.setattr(settings, "hybrid_retrieval_enabled", True)
    monkeypatch.setattr(settings, "hybrid_bm25_top_k", 5)
    monkeypatch.setattr(settings, "hybrid_rrf_k", 60)
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    vector = [_chunk("shared", content="ASC ROAS vector", score=0.4)]
    bm25 = [
        _chunk("shared", content="ASC ROAS vector", score=0.9),
        _chunk("bm25_only", content="T&D 캠페인보드 exact keyword", score=1.0),
    ]
    store = FakeHybridVectorStore(vector, bm25)
    retriever = Retriever(embedder=FakeEmbedder(), vector_store=store)

    details = retriever.retrieve_with_details("T&D 캠페인보드", top_k=5, max_per_file=10)
    ids = {c.chunk_id for c in details.candidates}

    assert store.bm25_calls == 1
    assert "shared" in ids
    assert "bm25_only" in ids
    assert details.summary["hybrid_retrieval_enabled"] is True
    assert details.summary["vector_candidate_count"] == 1
    assert details.summary["bm25_candidate_count"] == 2
    assert details.summary["hybrid_merged_candidate_count"] == 2
    assert details.summary["bm25_only_candidate_count"] == 1
    assert details.summary["overlap_candidate_count"] == 1

    by_id = {c.chunk_id: c for c in details.candidates}
    assert by_id["bm25_only"].metadata["retrieval_sources"] == ["bm25"]
    assert by_id["bm25_only"].metadata["bm25_rank"] == 2
    assert by_id["shared"].metadata["retrieval_sources"] == ["vector", "bm25"]
    assert by_id["shared"].metadata["vector_rank"] == 1
    assert by_id["shared"].metadata["bm25_rank"] == 1
    assert by_id["shared"].metadata["hybrid_rrf_score"] > 0


def test_slack_adapter_forwards_hybrid_diagnostics():
    chunk = _chunk("bm25_only", content="ASC exact", score=1.0)
    chunk.metadata.update({
        "retrieval_sources": ["bm25"],
        "bm25_rank": 1,
        "hybrid_rrf_score": 0.016393,
    })

    class FakePipeline:
        def ask(self, question: str) -> Dict[str, Any]:  # noqa: ARG002
            return {
                "answer": "[fake answer]",
                "primary_normalized_documents": [],
                "raw_evidence": [],
                "raw_fallback": [chunk],
                "answer_mode": "raw_fallback",
                "retrieval_summary": {
                    "hybrid_retrieval_enabled": True,
                    "vector_candidate_count": 1,
                    "bm25_candidate_count": 2,
                    "hybrid_merged_candidate_count": 2,
                    "bm25_only_candidate_count": 1,
                    "vector_only_candidate_count": 0,
                    "overlap_candidate_count": 1,
                },
            }

    out = qa_adapter.answer_slack_question("ASC 알려줘", pipeline=FakePipeline())

    diagnostics = out["diagnostics"]
    assert diagnostics["hybrid_retrieval_enabled"] is True
    assert diagnostics["bm25_candidate_count"] == 2
    assert diagnostics["bm25_only_candidate_count"] == 1
    source = out["sources"]["raw_fallback"][0]
    assert source["retrieval_sources"] == ["bm25"]
    assert source["bm25_rank"] == 1
    assert source["hybrid_rrf_score"] == 0.016393


def test_slack_debug_shows_hybrid_diagnostics_only_in_debug():
    result = {
        "answer": "## 1. 결론\nASC 기준입니다.",
        "sources": {
            "primary_normalized_documents": [],
            "raw_evidence": [],
            "raw_fallback": [{
                "file_name": "guide.md",
                "section_title": "ASC",
                "content_type": "text",
                "retrieval_role": "raw_fallback",
                "final_score": 0.8,
                "retrieval_sources": ["bm25"],
                "bm25_rank": 1,
                "hybrid_rrf_score": 0.016393,
                "preview": "ASC exact keyword",
            }],
        },
        "diagnostics": {
            "answer_mode": "raw_fallback",
            "hybrid_retrieval_enabled": True,
            "vector_candidate_count": 1,
            "bm25_candidate_count": 2,
            "hybrid_merged_candidate_count": 2,
            "bm25_only_candidate_count": 1,
            "vector_only_candidate_count": 0,
            "overlap_candidate_count": 1,
        },
        "answer_mode": "raw_fallback",
        "primary_normalized_document_count": 0,
        "raw_evidence_count": 0,
        "raw_fallback_count": 1,
    }

    debug_text = formatter.format_qa_result(result, debug=True)
    default_text = formatter.format_qa_result(result, debug=False)

    assert "*Hybrid Retrieval*" in debug_text
    assert "bm25_candidate: 2" in debug_text
    assert "bm25_rank=`1`" in debug_text
    assert "hybrid_rrf_score=`0.016`" in debug_text
    assert "*Hybrid Retrieval*" not in default_text
