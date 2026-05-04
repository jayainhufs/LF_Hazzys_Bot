"""
test_retrieval_precision.py
===========================
검색 정밀도 (retriever / reranker / qa_pipeline) 단위 테스트.

외부 Gemini API / ChromaDB 호출 없이 단위 테스트만 가능하도록
Embedder / VectorStore / Generator 를 모두 fake 객체로 주입한다.

검증 시나리오:
1. guide chunk 와 slack chunk 가 섞여 있을 때 "정산 프로세스" 질문은 guide 가 상위에 와야 한다.
2. "4/29 메타 TODO" 질문은 slack_manual 이 상위에 와야 한다.
3. 낮은 점수 chunk 는 threshold 에서 탈락해야 한다.
4. 같은 파일에서 MAX_CHUNKS_PER_FILE 이상 나오지 않아야 한다.
5. retrieved chunk 가 없으면 qa_pipeline 이 generation 을 skip 해야 한다.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from src.config import settings
from src.rag.qa_pipeline import QAPipeline
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


class FakeGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, **_kwargs):  # noqa: D401
        self.calls += 1
        return ("[fake answer]", "fake-model")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_chunk(
    *,
    chunk_id: str,
    file_name: str,
    uploaded_category: str,
    source_type: str,
    content_type: str = "text",
    score: float = 0.7,
    source_weight: Optional[float] = None,
    section_title: Optional[str] = None,
    chunk_index: int = 0,
    document_id: Optional[str] = None,
    content: str = "",
) -> RetrievedChunk:
    sw = source_weight
    if sw is None:
        sw = settings.category_source_weight.get(source_type, 0.7)
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id or f"doc_{file_name}",
        file_name=file_name,
        source_type=source_type,
        uploaded_category=uploaded_category,
        section_title=section_title,
        content_type=content_type,
        content=content or f"[content for {chunk_id}]",
        score=float(score),
        final_score=float(score),
        metadata={
            "source_weight": float(sw),
            "chunk_index": chunk_index,
            "uploaded_category": uploaded_category,
            "file_name": file_name,
            "source_type": source_type,
            "content_type": content_type,
            "section_title": section_title,
        },
    )


def _make_retriever(chunks: List[RetrievedChunk]) -> Retriever:
    return Retriever(embedder=FakeEmbedder(), vector_store=FakeVectorStore(chunks))


# ---------------------------------------------------------------------------
# 1) guide vs slack: "정산 프로세스" → guide 우선
# ---------------------------------------------------------------------------
def test_settlement_query_prefers_guide_over_slack():
    chunks = [
        # slack 이 임베딩 유사도는 더 높지만,
        # guide 는 settlement keyword boost + 더 높은 source_weight 로 final_score 가 더 커야 함.
        _make_chunk(
            chunk_id="slack_1",
            file_name="[2026년 4월 29일 TODO].txt",
            uploaded_category="slack",
            source_type="slack_manual",
            content_type="conversation",
            score=0.78,
        ),
        _make_chunk(
            chunk_id="guide_1",
            file_name="LF 정산 가이드.txt",
            uploaded_category="guide",
            source_type="guide",
            content_type="text",
            score=0.72,
            section_title="정산 프로세스",
        ),
    ]
    r = _make_retriever(chunks)
    details = r.retrieve_with_details(
        "정산 프로세스는 어떤 순서로 진행해야 해?",
        top_k=5,
        with_parent_children=False,
    )
    assert details.passed, "통과 chunk 가 있어야 한다"
    assert details.passed[0].chunk_id == "guide_1", (
        f"정산 질문에서는 guide 가 1순위여야 한다. "
        f"실제 1순위: {details.passed[0].chunk_id} "
        f"(final_score={details.passed[0].final_score:.4f})"
    )
    assert details.summary["query_class"]["is_settlement"] is True


# ---------------------------------------------------------------------------
# 2) "4/29 메타 TODO" → slack_manual 우선
# ---------------------------------------------------------------------------
def test_todo_query_prefers_slack_over_guide():
    chunks = [
        # guide 임베딩 유사도가 약간 더 높지만,
        # TODO keyword boost 로 slack final_score 가 더 커야 함.
        _make_chunk(
            chunk_id="guide_1",
            file_name="LF 정산 가이드.txt",
            uploaded_category="guide",
            source_type="guide",
            content_type="text",
            score=0.74,
        ),
        _make_chunk(
            chunk_id="slack_1",
            file_name="[2026년 4월 29일 TODO].txt",
            uploaded_category="slack",
            source_type="slack_manual",
            content_type="conversation",
            score=0.70,
            section_title="Slack 대화 블록 1",
        ),
    ]
    r = _make_retriever(chunks)
    details = r.retrieve_with_details(
        "4월 29일 메타 캠페인 세팅에서 놓치면 안 되는 게 뭐였어? 오늘 TODO 알려줘",
        top_k=5,
        with_parent_children=False,
    )
    assert details.passed
    assert details.passed[0].chunk_id == "slack_1", (
        f"TODO 질문에서는 slack 이 1순위여야 한다. "
        f"실제 1순위: {details.passed[0].chunk_id} "
        f"(final_score={details.passed[0].final_score:.4f})"
    )
    qc = details.summary["query_class"]
    assert qc["is_todo"] is True


# ---------------------------------------------------------------------------
# 3) 낮은 점수 chunk threshold 탈락
# ---------------------------------------------------------------------------
def test_low_similarity_chunk_drops_with_threshold():
    chunks = [
        _make_chunk(
            chunk_id="ok_1",
            file_name="LF 정산 가이드.txt",
            uploaded_category="guide",
            source_type="guide",
            score=0.70,
        ),
        _make_chunk(
            chunk_id="too_low",
            file_name="LF 정산 가이드.txt",
            uploaded_category="guide",
            source_type="guide",
            score=0.10,  # MIN_SIMILARITY_SCORE 0.35 보다 낮음
            chunk_index=1,
        ),
    ]
    r = _make_retriever(chunks)
    details = r.retrieve_with_details(
        "정산 가이드 알려줘",
        top_k=5,
        min_similarity=0.35,
        min_final=0.30,
        max_per_file=10,  # 파일 cap 영향 배제
        with_parent_children=False,
    )
    passed_ids = [c.chunk_id for c in details.passed]
    dropped = [c for c in details.candidates if not c.passed_threshold]
    assert "too_low" not in passed_ids, "낮은 similarity chunk 는 통과하면 안 된다"
    assert any(c.chunk_id == "too_low" for c in dropped)
    bad = next(c for c in dropped if c.chunk_id == "too_low")
    assert bad.filter_reason and "similarity" in bad.filter_reason


# ---------------------------------------------------------------------------
# 4) 같은 파일에서 MAX_CHUNKS_PER_FILE 이상 나오지 않음
# ---------------------------------------------------------------------------
def test_max_chunks_per_file_enforced():
    chunks = [
        _make_chunk(
            chunk_id=f"g_{i}",
            file_name="LF 정산 가이드.txt",
            uploaded_category="guide",
            source_type="guide",
            score=0.80 - i * 0.01,
            chunk_index=i,
        )
        for i in range(5)
    ]
    r = _make_retriever(chunks)
    details = r.retrieve_with_details(
        "정산 가이드",
        top_k=10,
        max_per_file=2,
        min_similarity=0.30,
        min_final=0.10,
        with_parent_children=False,
    )
    same_file_pass = [c for c in details.passed if c.file_name == "LF 정산 가이드.txt"]
    assert len(same_file_pass) <= 2, (
        f"파일당 최대 2개 chunk 만 통과해야 함. 실제: {len(same_file_pass)}"
    )
    file_cap_dropped = [
        c for c in details.candidates
        if not c.passed_threshold and (c.filter_reason or "").startswith("file_cap_exceeded")
    ]
    assert file_cap_dropped, "file cap 으로 탈락한 chunk 가 있어야 한다"


# ---------------------------------------------------------------------------
# 5) qa_pipeline: 근거 없으면 generation skip
# ---------------------------------------------------------------------------
def test_qa_pipeline_skips_generation_when_no_passing_chunks():
    # 모든 chunk 의 similarity 가 임계값 미만 → 통과 0개
    chunks = [
        _make_chunk(
            chunk_id="x_1",
            file_name="random.txt",
            uploaded_category="misc",
            source_type="misc",
            score=0.05,
        ),
    ]
    fake_gen = FakeGenerator()
    pipeline = QAPipeline(
        retriever=_make_retriever(chunks),
        generator=fake_gen,  # type: ignore[arg-type]
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
    )
    result = pipeline.ask(
        "관련 없는 질문입니다",
        top_k=5,
        save_log=False,
    )
    assert result["generation_skipped"] is True
    assert fake_gen.calls == 0, "근거 부족 시 Gemini Generation 호출이 0이어야 한다"
    assert "근거가 부족" in result["answer"]


def test_qa_pipeline_calls_generator_when_chunks_pass():
    # high similarity → 통과 → generation 호출되어야 함
    chunks = [
        _make_chunk(
            chunk_id="g_1",
            file_name="LF 정산 가이드.txt",
            uploaded_category="guide",
            source_type="guide",
            score=0.80,
            section_title="정산 프로세스",
        ),
    ]
    fake_gen = FakeGenerator()
    pipeline = QAPipeline(
        retriever=_make_retriever(chunks),
        generator=fake_gen,  # type: ignore[arg-type]
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
    )
    result = pipeline.ask(
        "정산 프로세스 알려줘",
        top_k=5,
        save_log=False,
    )
    assert result["generation_skipped"] is False
    assert fake_gen.calls == 1
    assert result["answer"] == "[fake answer]"


# ---------------------------------------------------------------------------
# 6) MMR diversity: 같은 파일 chunk 가 연속으로 위에 오지 않게 분산되는지
# ---------------------------------------------------------------------------
def test_mmr_reduces_same_file_repetition():
    # 같은 파일 chunk 3개 + 다른 파일 1개. file cap=10 (영향 배제).
    chunks = [
        _make_chunk(
            chunk_id="a_0", file_name="A.txt", uploaded_category="guide",
            source_type="guide", score=0.80, chunk_index=0,
        ),
        _make_chunk(
            chunk_id="a_1", file_name="A.txt", uploaded_category="guide",
            source_type="guide", score=0.79, chunk_index=1,
        ),
        _make_chunk(
            chunk_id="a_2", file_name="A.txt", uploaded_category="guide",
            source_type="guide", score=0.78, chunk_index=2,
        ),
        _make_chunk(
            chunk_id="b_0", file_name="B.txt", uploaded_category="guide",
            source_type="guide", score=0.74, chunk_index=0,
        ),
    ]
    r = _make_retriever(chunks)
    details = r.retrieve_with_details(
        "정산 관련", top_k=4,
        max_per_file=10,
        min_similarity=0.10,
        min_final=0.10,
        use_mmr=True,
        with_parent_children=False,
    )
    # 통과 chunk 4개. MMR 적용으로 B.txt 가 2번째 안에 들어와야 (다양성 보정)
    # (lambda=0.7 이면 강한 보정은 아니지만 충분히 작용해야 한다)
    top_files = [c.file_name for c in details.passed[:2]]
    assert "B.txt" in top_files or details.passed[1].file_name == "B.txt", (
        f"MMR 적용 시 상위 2개에 다른 파일이 섞여야 한다. 실제: {top_files}"
    )
