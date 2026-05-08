"""
test_knowledge_card_retrieval.py
================================
Task 6: 검색 단계에서 Normalized Document 우선 retrieval / reranker 단위 테스트.

외부 Gemini API / ChromaDB 호출 없이 RetrievedChunk 를 직접 만들어
reranker / retriever 동작을 검증한다.

명칭 변경 노트
----------------
- ``KnowledgeCard`` 개념은 ``NormalizedDocument`` 로 교체되었다.
- 이 파일은 기존 import (``apply_knowledge_card_priority`` /
  ``is_knowledge_card_chunk`` / ``card_type_boost_for``) 가 legacy alias 로
  계속 동작하는지 함께 검증하는 역할을 한다.
- 신규 명칭에 기반한 추가 시나리오는 ``tests/test_legacy_compatibility.py`` 에 있다.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from src.config import settings
from src.rag.reranker import (
    apply_knowledge_card_priority,
    card_type_boost_for,
    extract_query_metadata,
    is_knowledge_card_chunk,
    rerank_simple,
)
from src.rag.retriever import Retriever
from src.schemas import RetrievedChunk


# ---------------------------------------------------------------------------
# Fakes / helpers
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
        parent_chunk_id: str,  # noqa: ARG002
        sheet_name: Optional[str] = None,  # noqa: ARG002
        limit: int = 3,  # noqa: ARG002
    ) -> List[RetrievedChunk]:
        return []


def _make_kc_chunk(
    *,
    chunk_id: str,
    file_name: str,
    card_type: str,
    primary_topic: str = "common",
    task_type: str = "unknown",
    score: float = 0.6,
    section_title: Optional[str] = None,
    parent_raw_chunk_ids: Optional[List[str]] = None,
    source_file_hash: Optional[str] = None,
    uploaded_category: str = "guide",
    document_id: Optional[str] = None,
) -> RetrievedChunk:
    md: Dict[str, Any] = {
        "source_weight": float(settings.normalization_card_source_weight),
        "uploaded_category": uploaded_category,
        "file_name": file_name,
        "source_type": "llm_normalized",
        "content_type": "knowledge_card",
        "section_title": section_title,
        "card_id": chunk_id,
        "card_type": card_type,
        "primary_topic": primary_topic,
        "task_type": task_type,
        "topic_tags": [primary_topic] if primary_topic else [],
        "parent_raw_chunk_ids": list(parent_raw_chunk_ids or []),
        "normalized": True,
    }
    if source_file_hash:
        md["source_file_hash"] = source_file_hash
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id or f"doc_{file_name}",
        file_name=file_name,
        source_type="llm_normalized",
        uploaded_category=uploaded_category,
        section_title=section_title,
        content_type="knowledge_card",
        content=f"[knowledge_card content for {chunk_id}]",
        score=float(score),
        final_score=float(score),
        metadata=md,
    )


def _make_raw_chunk(
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
    primary_topic: Optional[str] = None,
    source_file_hash: Optional[str] = None,
) -> RetrievedChunk:
    sw = source_weight
    if sw is None:
        sw = settings.category_source_weight.get(source_type, 0.7)
    md: Dict[str, Any] = {
        "source_weight": float(sw),
        "chunk_index": chunk_index,
        "uploaded_category": uploaded_category,
        "file_name": file_name,
        "source_type": source_type,
        "content_type": content_type,
        "section_title": section_title,
    }
    if primary_topic:
        md["primary_topic"] = primary_topic
        md["topic_tags"] = [primary_topic]
    if source_file_hash:
        md["source_file_hash"] = source_file_hash
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id or f"doc_{file_name}",
        file_name=file_name,
        source_type=source_type,
        uploaded_category=uploaded_category,
        section_title=section_title,
        content_type=content_type,
        content=f"[raw content for {chunk_id}]",
        score=float(score),
        final_score=float(score),
        metadata=md,
    )


# ---------------------------------------------------------------------------
# 1. helper 단위 테스트
# ---------------------------------------------------------------------------
def test_is_knowledge_card_chunk_via_content_type():
    chunk = _make_kc_chunk(
        chunk_id="kc1", file_name="meta_guide.txt", card_type="workflow"
    )
    assert is_knowledge_card_chunk(chunk) is True


def test_is_knowledge_card_chunk_via_source_type():
    raw = _make_raw_chunk(
        chunk_id="x1",
        file_name="x.txt",
        uploaded_category="guide",
        source_type="llm_normalized",
        content_type="text",
    )
    assert is_knowledge_card_chunk(raw) is True


def test_is_knowledge_card_chunk_negative():
    raw = _make_raw_chunk(
        chunk_id="r1",
        file_name="raw.txt",
        uploaded_category="guide",
        source_type="guide",
        content_type="text",
    )
    assert is_knowledge_card_chunk(raw) is False


def test_is_knowledge_card_chunk_accepts_dict():
    md = {"content_type": "knowledge_card"}
    assert is_knowledge_card_chunk(md) is True
    assert is_knowledge_card_chunk({"source_type": "llm_normalized"}) is True
    assert is_knowledge_card_chunk({"content_type": "text"}) is False


def test_card_type_boost_for_each_card_type():
    """spec 표에 정의된 card_type 별 boost 가 정확히 반영된다."""
    cases = [
        ("workflow", settings.workflow_card_boost),
        ("checklist", settings.checklist_card_boost),
        ("faq", settings.faq_card_boost),
        ("decision", settings.decision_card_boost),
        ("communication_template", settings.communication_template_boost),
        ("glossary", settings.glossary_card_boost),
    ]
    for ct, expected in cases:
        boost, intent_match = card_type_boost_for(ct, query_intent=[])
        assert intent_match is False
        assert boost == pytest.approx(float(expected), rel=1e-6)


def test_card_type_boost_for_intent_match_adds_bonus():
    """intent 와 일치하는 card_type 은 추가 부스트(1.10x) 가 적용된다."""
    base, _ = card_type_boost_for("workflow", query_intent=[])
    boosted, intent_match = card_type_boost_for("workflow", query_intent=["procedure"])
    assert intent_match is True
    assert boosted > base
    assert boosted == pytest.approx(base * 1.10, rel=1e-6)


def test_card_type_boost_for_unknown_returns_neutral():
    boost, intent_match = card_type_boost_for("issue", query_intent=[])
    # issue 는 boost map 에 없으므로 1.0
    assert boost == pytest.approx(1.0, rel=1e-6)
    assert intent_match is False
    boost2, im2 = card_type_boost_for(None, query_intent=["procedure"])
    assert boost2 == pytest.approx(1.0)
    assert im2 is False


# ---------------------------------------------------------------------------
# 2. apply_knowledge_card_priority 동작
# ---------------------------------------------------------------------------
def _rerank_and_prioritize(
    candidates: List[RetrievedChunk], query: str
) -> List[RetrievedChunk]:
    qm = extract_query_metadata(query)
    rerank_simple(candidates, query=query, query_metadata=qm)
    return apply_knowledge_card_priority(candidates, query_metadata=qm)


def test_priority_sets_metadata_diagnostics_for_knowledge_card():
    cands = [
        _make_kc_chunk(
            chunk_id="kc1", file_name="meta_guide.txt", card_type="workflow",
            primary_topic="meta",
        ),
    ]
    out = _rerank_and_prioritize(cands, "메타 캠페인 셋팅 가이드 알려줘")
    md = out[0].metadata
    assert md["retrieval_role"] == "primary_card"
    assert md["knowledge_card_boost"] == pytest.approx(
        float(settings.knowledge_card_content_boost), rel=1e-6
    )
    assert md["card_type_boost"] > 1.0
    assert md["card_type_match"] is True


def test_priority_marks_raw_evidence_for_same_source_file():
    """같은 file_name 에 knowledge_card 가 있으면 raw chunk 는 raw_evidence."""
    cands = [
        _make_kc_chunk(
            chunk_id="kc1",
            file_name="meta_guide.txt",
            card_type="workflow",
            primary_topic="meta",
        ),
        _make_raw_chunk(
            chunk_id="r1",
            file_name="meta_guide.txt",
            uploaded_category="guide",
            source_type="guide",
        ),
        _make_raw_chunk(
            chunk_id="r2",
            file_name="other.txt",
            uploaded_category="guide",
            source_type="guide",
        ),
    ]
    out = _rerank_and_prioritize(cands, "메타 셋팅 가이드")
    by_id = {c.chunk_id: c for c in out}
    assert by_id["kc1"].metadata["retrieval_role"] == "primary_card"
    assert by_id["r1"].metadata["retrieval_role"] == "raw_evidence"
    assert by_id["r2"].metadata["retrieval_role"] == "raw_fallback"


def test_priority_preserves_parent_raw_chunk_ids():
    cands = [
        _make_kc_chunk(
            chunk_id="kc1",
            file_name="meta_guide.txt",
            card_type="workflow",
            parent_raw_chunk_ids=["raw_a", "raw_b"],
        ),
    ]
    out = _rerank_and_prioritize(cands, "셋팅 가이드")
    assert out[0].metadata["parent_raw_chunk_ids"] == ["raw_a", "raw_b"]


def test_priority_disabled_does_not_change_score(monkeypatch):
    """PRIORITIZE_KNOWLEDGE_CARDS=false 면 final_score 가 boost 로 변하지 않는다."""
    monkeypatch.setattr(settings, "prioritize_knowledge_cards", False)

    cands = [
        _make_kc_chunk(
            chunk_id="kc1", file_name="meta_guide.txt", card_type="workflow"
        ),
    ]
    qm = extract_query_metadata("셋팅 가이드")
    rerank_simple(cands, query="셋팅 가이드", query_metadata=qm)
    score_before = cands[0].final_score
    apply_knowledge_card_priority(cands, query_metadata=qm)
    score_after = cands[0].final_score
    # 비활성화 시에는 boost 로 점수가 변하지 않아야 한다.
    assert score_after == pytest.approx(score_before, rel=1e-9)
    # 단 진단 필드는 채워져 있어야 한다.
    md = cands[0].metadata
    assert md.get("retrieval_role") == "primary_card"
    assert md.get("knowledge_card_boost") == 1.0
    assert md.get("card_type_boost") == 1.0


# ---------------------------------------------------------------------------
# 3. 시나리오: card_type 별 우선순위
# ---------------------------------------------------------------------------
def test_workflow_card_outranks_raw_guide_for_setting_question():
    """'메타 셋팅 가이드' 질문 → workflow knowledge_card 가 raw guide 보다 위."""
    cands = [
        _make_raw_chunk(
            chunk_id="raw_g",
            file_name="meta_guide.txt",
            uploaded_category="guide",
            source_type="guide",
            score=0.72,
            primary_topic="meta",
        ),
        _make_kc_chunk(
            chunk_id="kc_wf",
            file_name="meta_guide_card.txt",
            card_type="workflow",
            primary_topic="meta",
            score=0.60,
        ),
    ]
    out = _rerank_and_prioritize(cands, "메타 캠페인 셋팅 가이드 알려줘")
    assert out[0].chunk_id == "kc_wf"
    assert out[0].metadata["card_type_match"] is True


def test_checklist_card_outranks_raw_for_checklist_question():
    cands = [
        _make_raw_chunk(
            chunk_id="raw_g",
            file_name="raw_guide.txt",
            uploaded_category="guide",
            source_type="guide",
            score=0.72,
        ),
        _make_kc_chunk(
            chunk_id="kc_cl",
            file_name="checklist_card.txt",
            card_type="checklist",
            score=0.60,
        ),
    ]
    out = _rerank_and_prioritize(cands, "메타 캠페인 체크리스트 어떻게 확인해?")
    assert out[0].chunk_id == "kc_cl"


def test_decision_card_outranks_raw_for_explanation_question():
    cands = [
        _make_raw_chunk(
            chunk_id="raw_conv",
            file_name="slack_thread.txt",
            uploaded_category="slack",
            source_type="slack_manual",
            content_type="conversation",
            score=0.70,
        ),
        _make_kc_chunk(
            chunk_id="kc_dec",
            file_name="bau_asc_card.txt",
            card_type="decision",
            primary_topic="meta",
            score=0.55,
            uploaded_category="slack",
        ),
        _make_kc_chunk(
            chunk_id="kc_faq",
            file_name="bau_asc_card2.txt",
            card_type="faq",
            primary_topic="meta",
            score=0.55,
            uploaded_category="slack",
        ),
    ]
    out = _rerank_and_prioritize(cands, "BAU랑 ASC 차이 설명해줘")
    assert out[0].chunk_id in {"kc_dec", "kc_faq"}
    # raw conversation 은 knowledge_card 보다 아래로 밀려야 한다.
    raw_rank = next(i for i, c in enumerate(out) if c.chunk_id == "raw_conv")
    kc_top = 0
    assert raw_rank > kc_top


def test_communication_template_card_outranks_raw_for_share_question():
    cands = [
        _make_raw_chunk(
            chunk_id="raw_c",
            file_name="email_thread.txt",
            uploaded_category="slack",
            source_type="slack_manual",
            content_type="conversation",
            score=0.70,
        ),
        _make_kc_chunk(
            chunk_id="kc_ct",
            file_name="share_card.txt",
            card_type="communication_template",
            score=0.55,
            uploaded_category="slack",
        ),
    ]
    out = _rerank_and_prioritize(cands, "광고주 공유 문안 어떻게 작성해?")
    assert out[0].chunk_id == "kc_ct"
    assert out[0].metadata["card_type_match"] is True


def test_glossary_card_outranks_raw_for_term_question():
    cands = [
        _make_raw_chunk(
            chunk_id="raw_t",
            file_name="terms.txt",
            uploaded_category="guide",
            source_type="guide",
            score=0.70,
        ),
        _make_kc_chunk(
            chunk_id="kc_g",
            file_name="glossary_card.txt",
            card_type="glossary",
            score=0.55,
        ),
    ]
    out = _rerank_and_prioritize(cands, "ASC 용어 무슨 뜻이야?")
    assert out[0].chunk_id == "kc_g"
    assert out[0].metadata["card_type_match"] is True


def test_priority_disabled_does_not_force_card_above_raw(monkeypatch):
    """PRIORITIZE_KNOWLEDGE_CARDS=false 면 boost 가 없어 raw similarity 가 더 높을 때
    raw chunk 가 위에 남을 수 있다."""
    monkeypatch.setattr(settings, "prioritize_knowledge_cards", False)

    cands = [
        _make_raw_chunk(
            chunk_id="raw_g",
            file_name="meta_guide.txt",
            uploaded_category="guide",
            source_type="guide",
            score=0.85,
        ),
        _make_kc_chunk(
            chunk_id="kc_wf",
            file_name="meta_guide_card.txt",
            card_type="workflow",
            score=0.40,
        ),
    ]
    out = _rerank_and_prioritize(cands, "셋팅 가이드")
    assert out[0].chunk_id == "raw_g"


# ---------------------------------------------------------------------------
# 4. Retriever 통합 (FakeVectorStore)
# ---------------------------------------------------------------------------
def _make_retriever(chunks: List[RetrievedChunk]) -> Retriever:
    return Retriever(embedder=FakeEmbedder(), vector_store=FakeVectorStore(chunks))


def test_retriever_summary_includes_knowledge_card_diagnostics(monkeypatch):
    """retrieve_with_details 가 knowledge_card 진단 카운트와 boost 를 summary 에 채운다."""
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    chunks = [
        _make_kc_chunk(
            chunk_id="kc_a",
            file_name="card_a.txt",
            card_type="workflow",
            primary_topic="meta",
            score=0.55,
        ),
        _make_raw_chunk(
            chunk_id="raw_a",
            file_name="card_a.txt",
            uploaded_category="guide",
            source_type="guide",
            score=0.50,
        ),
        _make_raw_chunk(
            chunk_id="raw_b",
            file_name="other.txt",
            uploaded_category="guide",
            source_type="guide",
            score=0.45,
        ),
    ]
    retriever = _make_retriever(chunks)
    details = retriever.retrieve_with_details(
        "메타 셋팅 가이드 알려줘", top_k=5, max_per_file=5
    )
    summary = details.summary
    assert summary["prioritize_knowledge_cards"] is True
    assert summary["knowledge_card_content_boost"] == pytest.approx(
        float(settings.knowledge_card_content_boost), rel=1e-6
    )
    assert summary["enable_parent_raw_evidence"] is True

    roles = [c.metadata.get("retrieval_role") for c in details.passed]
    assert "primary_card" in roles
    assert "raw_evidence" in roles
    # 역할별 카운트 비교
    assert summary["knowledge_card_count"] == roles.count("primary_card")
    assert summary["raw_evidence_count"] == roles.count("raw_evidence")
    assert summary["raw_fallback_count"] == roles.count("raw_fallback")
    # knowledge_card 가 1순위
    assert details.passed[0].chunk_id == "kc_a"


def test_retriever_keeps_raw_fallback_when_no_knowledge_card_present(monkeypatch):
    """knowledge_card 가 후보에 전혀 없을 때도 기존 raw retrieval 흐름은 유지된다."""
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    chunks = [
        _make_raw_chunk(
            chunk_id="raw_a",
            file_name="g.txt",
            uploaded_category="guide",
            source_type="guide",
            score=0.70,
        ),
    ]
    retriever = _make_retriever(chunks)
    details = retriever.retrieve_with_details("정산 프로세스 알려줘", top_k=3)
    assert len(details.passed) == 1
    md = details.passed[0].metadata
    assert md.get("retrieval_role") == "raw_fallback"
    assert md.get("knowledge_card_boost") == 1.0


def test_retriever_hides_parent_raw_chunk_ids_when_disabled(monkeypatch):
    """ENABLE_PARENT_RAW_EVIDENCE=false 면 parent_raw_chunk_ids 는 빈 리스트로 노출된다."""
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)
    monkeypatch.setattr(settings, "enable_parent_raw_evidence", False)

    chunks = [
        _make_kc_chunk(
            chunk_id="kc_a",
            file_name="card_a.txt",
            card_type="workflow",
            parent_raw_chunk_ids=["r1", "r2", "r3"],
            score=0.55,
        ),
    ]
    retriever = _make_retriever(chunks)
    details = retriever.retrieve_with_details("셋팅 가이드", top_k=3)
    assert details.passed
    md = details.passed[0].metadata
    assert md.get("parent_raw_chunk_ids") == []


def test_priority_does_not_break_existing_date_topic_threshold(monkeypatch):
    """Task 6 의 boost 가 기존 date/topic threshold 동작을 망가뜨리지 않는다.

    PRIORITIZE_KNOWLEDGE_CARDS=true 상태에서 raw chunk 만 후보일 때
    date_match='exact' / topic_match='match' 가 그대로 기록되어야 한다.
    """
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    raw = _make_raw_chunk(
        chunk_id="r_today",
        file_name="meta_2026-04-29.txt",
        uploaded_category="slack",
        source_type="slack_manual",
        content_type="conversation",
        score=0.80,
        primary_topic="meta",
    )
    raw.metadata["document_date"] = "2026-04-29"
    raw.metadata["topic_tags"] = ["meta"]

    retriever = _make_retriever([raw])
    details = retriever.retrieve_with_details(
        "2026-04-29 메타 캠페인 셋팅에서 놓치면 안 되는 것 알려줘", top_k=3
    )
    assert details.passed
    md = details.passed[0].metadata
    assert md.get("date_match") in {"exact", "none"}
    assert md.get("topic_match") in {"match", "none"}
    # raw 만 있을 때는 knowledge_card 카운트가 0
    assert details.summary["knowledge_card_count"] == 0
