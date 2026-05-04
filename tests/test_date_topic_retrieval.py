"""
test_date_topic_retrieval.py
============================
질문에서 추출된 query_date / query_topics / query_intent 가
retriever / reranker 에서 어떻게 boost / penalty 로 작용하는지 검증한다.

외부 Gemini API / ChromaDB 호출 없이 단위 테스트 가능하도록
Embedder / VectorStore 를 fake 객체로 주입한다.

검증:
1. extract_query_metadata 가 "4월 29일 메타 캠페인 세팅..." 에서
   query_date=2026-04-29, query_topics 에 meta 가 포함되도록 추출한다.
2. 질문 날짜와 다른 chunk 는 final_score 가 낮아진다 (date_mismatch_penalty).
3. exact date + topic match chunk 가 다른 날짜 slack chunk 보다 위에 온다.
4. ENABLE_DATE_FILTER=True 이면 다른 날짜 chunk 가 결과에서 제거된다.
5. query_intent 에 "todo_lookup" / "explanation" 이 잘 잡힌다.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.config import settings
from src.rag.reranker import extract_query_metadata, rerank_simple
from src.rag.retriever import Retriever
from src.schemas import RetrievedChunk


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeEmbedder:
    provider = "fake"
    model_name = "fake-model"

    def embed_query(self, text: str) -> List[float]:  # noqa: D401
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    def __init__(self, results: List[RetrievedChunk]) -> None:
        self._results = results

    def search(
        self,
        query_embedding: List[float],
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
        self, parent_chunk_id: str, sheet_name: Optional[str] = None, limit: int = 3
    ) -> List[RetrievedChunk]:
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_chunk(
    *,
    chunk_id: str,
    file_name: str,
    score: float,
    uploaded_category: str = "slack",
    source_type: str = "slack_manual",
    content_type: str = "conversation",
    document_date: Optional[str] = None,
    topic_tags: Optional[List[str]] = None,
    primary_topic: Optional[str] = None,
    section_title: Optional[str] = None,
) -> RetrievedChunk:
    sw = settings.category_source_weight.get(source_type, 0.7)
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        file_name=file_name,
        source_type=source_type,
        uploaded_category=uploaded_category,
        section_title=section_title,
        content_type=content_type,
        content=f"[content for {chunk_id}]",
        score=float(score),
        final_score=float(score),
        metadata={
            "source_weight": float(sw),
            "chunk_index": 0,
            "uploaded_category": uploaded_category,
            "file_name": file_name,
            "source_type": source_type,
            "content_type": content_type,
            "section_title": section_title,
            "document_date": document_date,
            "topic_tags": topic_tags or [],
            "primary_topic": primary_topic,
        },
    )


def _make_retriever(chunks: List[RetrievedChunk]) -> Retriever:
    return Retriever(embedder=FakeEmbedder(), vector_store=FakeVectorStore(chunks))


# ---------------------------------------------------------------------------
# 1. extract_query_metadata
# ---------------------------------------------------------------------------
def test_extract_query_metadata_finds_date_and_topic():
    meta = extract_query_metadata("4월 29일 메타 캠페인 세팅에서 놓치면 안 되는 건 뭐였어?")
    # 4/29 만으로도 default_year=2026 (현재 시점) 으로 잡힘.
    # 해마다 default_year 가 달라질 수 있으므로 month-day 만 검증한다.
    assert meta["query_date"] is not None
    assert meta["query_date"].endswith("-04-29")
    assert "meta" in meta["query_topics"]
    assert "todo_lookup" in meta["query_intent"]


def test_extract_query_metadata_dash_form():
    meta = extract_query_metadata("2026-04-30 카카오 발송 어땠어?")
    assert meta["query_date"] == "2026-04-30"
    assert "kakao" in meta["query_topics"]


def test_extract_query_metadata_no_date():
    meta = extract_query_metadata("BAU랑 ASC 성과 차이는 어떻게 설명했어?")
    assert meta["query_date"] is None
    assert "meta" in meta["query_topics"]
    assert "explanation" in meta["query_intent"]


# ---------------------------------------------------------------------------
# 2. date mismatch penalty (rerank only)
# ---------------------------------------------------------------------------
def test_rerank_applies_date_mismatch_penalty():
    same_date = _make_chunk(
        chunk_id="same",
        file_name="[2026년 4월 29일 TODO].txt",
        score=0.70,
        document_date="2026-04-29",
        topic_tags=["meta"],
        primary_topic="meta",
    )
    diff_date = _make_chunk(
        chunk_id="diff",
        file_name="[2026년 4월 30일 TODO].txt",
        score=0.70,
        document_date="2026-04-30",
        topic_tags=["meta"],
        primary_topic="meta",
    )
    qm = extract_query_metadata("4월 29일 메타 캠페인 세팅 알려줘")
    ranked = rerank_simple([same_date, diff_date], query="4월 29일 메타 캠페인 세팅 알려줘", query_metadata=qm)

    # final_score 비교
    f_same = next(c for c in ranked if c.chunk_id == "same").final_score
    f_diff = next(c for c in ranked if c.chunk_id == "diff").final_score
    assert f_same > f_diff, f"same date 가 mismatch 보다 final_score 가 커야 한다 (same={f_same}, diff={f_diff})"
    # date_match label
    same_label = next(c for c in ranked if c.chunk_id == "same").metadata.get("date_match")
    diff_label = next(c for c in ranked if c.chunk_id == "diff").metadata.get("date_match")
    assert same_label == "exact"
    assert diff_label == "mismatch"


# ---------------------------------------------------------------------------
# 3. exact + topic match chunk 가 다른 날짜 chunk 보다 위
# ---------------------------------------------------------------------------
def test_retriever_prefers_exact_date_topic_match():
    chunks = [
        # 다른 날짜 + meta topic (similarity 가 약간 더 높음)
        _make_chunk(
            chunk_id="apr30",
            file_name="[2026년 4월 30일 TODO].txt",
            score=0.78,
            document_date="2026-04-30",
            topic_tags=["meta"],
            primary_topic="meta",
        ),
        _make_chunk(
            chunk_id="may4",
            file_name="[2026년 5월 4일 TODO].txt",
            score=0.77,
            document_date="2026-05-04",
            topic_tags=["meta", "kakao"],
            primary_topic="meta",
        ),
        # 같은 날짜 (4/29) + meta topic — similarity 살짝 낮지만 boost 로 1순위가 되어야 함
        _make_chunk(
            chunk_id="apr29",
            file_name="[2026년 4월 29일 TODO].txt",
            score=0.72,
            document_date="2026-04-29",
            topic_tags=["meta"],
            primary_topic="meta",
        ),
    ]
    r = _make_retriever(chunks)
    details = r.retrieve_with_details(
        "4월 29일 메타 캠페인 세팅에서 놓치면 안 되는 건 뭐였어?",
        top_k=5,
        max_per_file=10,
        min_similarity=0.30,
        min_final=0.10,
        with_parent_children=False,
    )
    assert details.passed
    assert details.passed[0].chunk_id == "apr29", (
        f"4/29 chunk 가 1순위여야 한다. 실제: {details.passed[0].chunk_id} "
        f"(final={details.passed[0].final_score:.4f})"
    )
    assert details.summary.get("query_date", "").endswith("-04-29")
    assert "meta" in details.summary.get("query_topics", [])


# ---------------------------------------------------------------------------
# 4. ENABLE_DATE_FILTER=True 이면 다른 날짜 chunk 제거
# ---------------------------------------------------------------------------
def test_enable_date_filter_excludes_other_dates():
    chunks = [
        _make_chunk(
            chunk_id="apr29", file_name="[2026년 4월 29일 TODO].txt",
            score=0.72, document_date="2026-04-29", topic_tags=["meta"], primary_topic="meta",
        ),
        _make_chunk(
            chunk_id="apr30", file_name="[2026년 4월 30일 TODO].txt",
            score=0.85, document_date="2026-04-30", topic_tags=["meta"], primary_topic="meta",
        ),
        _make_chunk(
            chunk_id="may4", file_name="[2026년 5월 4일 TODO].txt",
            score=0.83, document_date="2026-05-04", topic_tags=["meta"], primary_topic="meta",
        ),
    ]
    r = _make_retriever(chunks)
    details = r.retrieve_with_details(
        "4월 29일 메타 캠페인 세팅 알려줘",
        top_k=10,
        max_per_file=10,
        min_similarity=0.30,
        min_final=0.10,
        enable_date_filter=True,
        with_parent_children=False,
    )
    passed_ids = [c.chunk_id for c in details.passed]
    assert passed_ids == ["apr29"], (
        f"ENABLE_DATE_FILTER 이면 4/29 만 통과해야 한다. 실제: {passed_ids}"
    )
    excluded = [
        c for c in details.candidates
        if not c.passed_threshold and (c.filter_reason or "").startswith("date_filter_excluded")
    ]
    assert len(excluded) == 2


# ---------------------------------------------------------------------------
# 5. topic_match 라벨이 정확히 채워지는지
# ---------------------------------------------------------------------------
def test_topic_match_labels():
    chunks = [
        _make_chunk(
            chunk_id="meta_1", file_name="meta.txt", score=0.7,
            topic_tags=["meta"], primary_topic="meta",
        ),
        _make_chunk(
            chunk_id="kakao_1", file_name="kakao.txt", score=0.7,
            topic_tags=["kakao"], primary_topic="kakao",
        ),
        _make_chunk(
            chunk_id="empty_1", file_name="other.txt", score=0.7,
            topic_tags=[], primary_topic=None,
        ),
    ]
    qm = extract_query_metadata("메타 캠페인 셋팅")
    ranked = rerank_simple(list(chunks), query="메타 캠페인 셋팅", query_metadata=qm)
    labels = {c.chunk_id: c.metadata.get("topic_match") for c in ranked}
    assert labels["meta_1"] == "match"
    assert labels["kakao_1"] == "mismatch"
    assert labels["empty_1"] == "none"


# ---------------------------------------------------------------------------
# 6. guide chunk + procedure 질문은 topic 부재여도 약하게만 페널티
# ---------------------------------------------------------------------------
def test_guide_chunk_not_strongly_penalized_for_procedure_query():
    guide = _make_chunk(
        chunk_id="guide_1",
        file_name="LF 정산 가이드.txt",
        score=0.70,
        uploaded_category="guide",
        source_type="guide",
        content_type="text",
        topic_tags=[],
        primary_topic=None,
    )
    qm = extract_query_metadata("정산 프로세스 어떻게 진행해?")
    ranked = rerank_simple([guide], query="정산 프로세스 어떻게 진행해?", query_metadata=qm)
    boost = ranked[0].metadata.get("topic_boost") or 0.0
    # procedure + guide + topic_tags 비어있음 → 0.95 weak penalty
    assert 0.9 <= boost <= 1.0
