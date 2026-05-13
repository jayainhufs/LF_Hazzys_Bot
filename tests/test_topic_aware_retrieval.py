"""
test_topic_aware_retrieval.py
=============================
MVP 2차 Step 2 — Topic-aware Retrieval / Penalty 강화 단위 테스트.

검증 항목:

1) `_topic_match_factor` 라벨링
   - query/chunk 가 같은 topic → ``match``
   - query/chunk 가 다른 명확한 topic → ``mismatch``
   - chunk 가 generic (common/general/etc) 만 → ``neutral`` (mismatch 아님)
   - query_topics 비어 있음 → ``none`` (penalty 미적용)
   - chunk_topics 비어 있음 → ``none`` (weak)

2) ``is_clear_topic_mismatch`` helper
   - generic / 빈 topic / 비어 있는 query 는 False
   - 명확한 다른 topic 만 True

3) Normalized Document mismatch 격하
   - kakao normalized chunk 는 query=meta 에서 primary_card 로 승격되지 않는다.
   - retrieval_role 이 raw_fallback 으로 격하되고 ``topic_mismatch_demoted=True``.
   - kc_boost / type_boost 가 적용되지 않는다.

4) Raw chunk mismatch
   - 같은 source_file 의 raw chunk 라도 topic mismatch 면 raw_evidence 승격 X.

5) meta 질문 시나리오
   - meta workflow normalized vs kakao normalized vs common 후보 중
     meta 가 1순위, kakao 가 primary_card 가 아니고, common 은 그대로 penalty 없이 남는다.

6) query_topic 이 None 이면 penalty 가 과도하게 걸리지 않는다.

7) Slack ``--debug`` 출력에 ``topic_mismatch_demoted_count`` 가 노출된다 (>0 일 때).
   기본 출력에는 노출되지 않는다.

외부 Gemini API / ChromaDB 호출 없이 fake retriever / fake generator / fake
chunk 로 검증한다.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from src.config import settings
from src.rag.reranker import (
    _GENERIC_TOPICS,
    _topic_match_factor,
    apply_normalized_document_priority,
    extract_query_metadata,
    is_clear_topic_mismatch,
    rerank_simple,
)
from src.rag.qa_pipeline import QAPipeline
from src.rag.retriever import Retriever
from src.schemas import RetrievedChunk
from src.slack_bot import formatter


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------
class FakeEmbedder:
    provider = "fake"
    model_name = "fake-model"

    def embed_query(self, text: str) -> List[float]:  # noqa: D401, ARG002
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


class FakeGenerator:
    def __init__(self, answer: str = "[fake answer]") -> None:
        self.calls = 0
        self.last_prompt: str = ""
        self._answer = answer

    def generate(self, prompt: str, **_kwargs):  # noqa: D401
        self.calls += 1
        self.last_prompt = prompt
        return (self._answer, "fake-model")


def _make_kc(
    *,
    chunk_id: str,
    file_name: str,
    primary_topic: str,
    card_type: str = "workflow",
    topic_tags: Optional[List[str]] = None,
    score: float = 0.55,
    uploaded_category: str = "guide",
) -> RetrievedChunk:
    md: Dict[str, Any] = {
        "source_weight": float(settings.normalization_card_source_weight),
        "uploaded_category": uploaded_category,
        "file_name": file_name,
        "source_type": "llm_normalized",
        "content_type": "knowledge_card",
        "card_id": chunk_id,
        "card_type": card_type,
        "normalized_document_type": card_type,
        "primary_topic": primary_topic,
        "topic_tags": list(topic_tags or [primary_topic]),
        "parent_raw_chunk_ids": [],
    }
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc_{file_name}",
        file_name=file_name,
        source_type="llm_normalized",
        uploaded_category=uploaded_category,
        section_title=None,
        content_type="knowledge_card",
        content=f"[knowledge_card content for {chunk_id}]",
        score=float(score),
        final_score=float(score),
        metadata=md,
    )


def _make_raw(
    *,
    chunk_id: str,
    file_name: str,
    primary_topic: Optional[str],
    topic_tags: Optional[List[str]] = None,
    score: float = 0.6,
    uploaded_category: str = "slack",
    source_type: str = "slack_manual",
    content_type: str = "conversation",
) -> RetrievedChunk:
    sw = settings.category_source_weight.get(source_type, 0.7)
    md: Dict[str, Any] = {
        "source_weight": float(sw),
        "chunk_index": 0,
        "uploaded_category": uploaded_category,
        "file_name": file_name,
        "source_type": source_type,
        "content_type": content_type,
        "section_title": None,
    }
    if primary_topic:
        md["primary_topic"] = primary_topic
    if topic_tags is not None:
        md["topic_tags"] = list(topic_tags)
    elif primary_topic:
        md["topic_tags"] = [primary_topic]
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc_{file_name}",
        file_name=file_name,
        source_type=source_type,
        uploaded_category=uploaded_category,
        section_title=None,
        content_type=content_type,
        content=f"[raw content for {chunk_id}]",
        score=float(score),
        final_score=float(score),
        metadata=md,
    )


def _make_retriever(chunks: List[RetrievedChunk]) -> Retriever:
    return Retriever(embedder=FakeEmbedder(), vector_store=FakeVectorStore(chunks))


def _make_pipeline(chunks: List[RetrievedChunk]) -> tuple:
    gen = FakeGenerator()
    pipe = QAPipeline(
        retriever=_make_retriever(chunks),
        generator=gen,  # type: ignore[arg-type]
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
    )
    return pipe, gen


# ===========================================================================
# 1) _topic_match_factor 라벨링
# ===========================================================================
def test_topic_match_factor_match():
    factor, label = _topic_match_factor(
        ["meta"], "guide", ["meta"], [], boost=1.20, penalty=0.80
    )
    assert label == "match"
    assert factor == pytest.approx(1.20, rel=1e-6)


def test_topic_match_factor_mismatch():
    factor, label = _topic_match_factor(
        ["kakao"], "slack_manual", ["meta"], [], boost=1.20, penalty=0.80
    )
    assert label == "mismatch"
    assert factor == pytest.approx(0.80, rel=1e-6)


def test_topic_match_factor_generic_chunk_is_neutral():
    """chunk 가 generic topic (common/general/etc) 만 가지면 mismatch 가 아니라 neutral."""
    for generic in sorted(_GENERIC_TOPICS):
        factor, label = _topic_match_factor(
            [generic], "guide", ["meta"], [], boost=1.20, penalty=0.80
        )
        assert label == "neutral", f"generic 토픽 {generic!r} 은 neutral 이어야 함"
        assert factor == pytest.approx(1.0, rel=1e-6)


def test_topic_match_factor_no_query_topics_returns_none():
    factor, label = _topic_match_factor(
        ["meta"], "guide", [], [], boost=1.20, penalty=0.80
    )
    assert label == "none"
    assert factor == pytest.approx(1.0)


def test_topic_match_factor_empty_chunk_topics_falls_back_to_weak():
    """chunk topic 이 비어 있고 query topic 만 있으면 약한 페널티 (none)."""
    factor, label = _topic_match_factor(
        [], "slack_manual", ["meta"], [], boost=1.20, penalty=0.80
    )
    assert label == "none"
    # weak penalty = penalty + (1.0 - penalty) * 0.5 = 0.80 + 0.10 = 0.90
    assert 0.85 <= factor <= 0.95


# ===========================================================================
# 2) is_clear_topic_mismatch helper
# ===========================================================================
def test_is_clear_topic_mismatch_basic_cases():
    # 명확한 mismatch
    chunk_kakao = _make_kc(chunk_id="kc", file_name="x.txt", primary_topic="kakao")
    assert is_clear_topic_mismatch(chunk_kakao, query_topics=["meta"]) is True

    # 같은 topic — mismatch 아님
    chunk_meta = _make_kc(chunk_id="kc2", file_name="x.txt", primary_topic="meta")
    assert is_clear_topic_mismatch(chunk_meta, query_topics=["meta"]) is False

    # query_topics 비어있으면 항상 False
    assert is_clear_topic_mismatch(chunk_kakao, query_topics=[]) is False


def test_is_clear_topic_mismatch_generic_chunk_is_not_mismatch():
    """primary_topic=common 같은 generic chunk 는 mismatch 가 아니다."""
    chunk_common = _make_kc(
        chunk_id="kc_common",
        file_name="x.txt",
        primary_topic="common",
        topic_tags=["common"],
    )
    assert is_clear_topic_mismatch(chunk_common, query_topics=["meta"]) is False


def test_is_clear_topic_mismatch_empty_chunk_topics_is_not_mismatch():
    """topic 정보가 없는 chunk 도 명확한 mismatch 로 보지 않는다."""
    chunk_no_topic = _make_raw(
        chunk_id="r1",
        file_name="x.txt",
        primary_topic=None,
        topic_tags=[],
        uploaded_category="guide",
        source_type="guide",
    )
    assert is_clear_topic_mismatch(chunk_no_topic, query_topics=["meta"]) is False


def test_is_clear_topic_mismatch_uses_primary_topic_when_tags_missing():
    """topic_tags 가 비어있어도 primary_topic 만으로도 mismatch 판단 가능."""
    chunk = _make_raw(
        chunk_id="r1",
        file_name="x.txt",
        primary_topic="kakao",
        topic_tags=[],
    )
    assert is_clear_topic_mismatch(chunk, query_topics=["meta"]) is True


def test_is_clear_topic_mismatch_accepts_dict_metadata():
    md = {"topic_tags": ["kakao"]}
    assert is_clear_topic_mismatch(md, query_topics=["meta"]) is True
    md_generic = {"topic_tags": ["common", "general"]}
    assert is_clear_topic_mismatch(md_generic, query_topics=["meta"]) is False


# ===========================================================================
# 3) Normalized Document mismatch 격하
# ===========================================================================
def test_priority_demotes_topic_mismatch_normalized_document_to_raw_fallback():
    """kakao normalized chunk 는 meta 질문에서 primary_card 로 승격되지 않는다."""
    cands = [
        _make_kc(
            chunk_id="kc_meta",
            file_name="meta_guide.txt",
            primary_topic="meta",
            card_type="workflow",
        ),
        _make_kc(
            chunk_id="kc_kakao",
            file_name="kakao_notice.txt",
            primary_topic="kakao",
            card_type="workflow",
        ),
    ]
    qm = extract_query_metadata("메타 피드광고 셋팅 방법 알려줘")
    rerank_simple(cands, query="메타 피드광고 셋팅 방법 알려줘", query_metadata=qm)
    out = apply_normalized_document_priority(cands, query_metadata=qm)

    by_id = {c.chunk_id: c for c in out}
    # 명확한 meta normalized → primary_card 유지
    assert by_id["kc_meta"].metadata["retrieval_role"] == "primary_card"
    assert by_id["kc_meta"].metadata["topic_mismatch_demoted"] is False
    # 명확한 mismatch (kakao) → primary_card 로 승격되지 않고 raw_fallback 으로 격하
    assert by_id["kc_kakao"].metadata["retrieval_role"] == "raw_fallback"
    assert by_id["kc_kakao"].metadata["topic_mismatch_demoted"] is True
    # 격하된 chunk 는 kc_boost / type_boost 가 적용되지 않는다.
    assert by_id["kc_kakao"].metadata["knowledge_card_boost"] == 1.0
    assert by_id["kc_kakao"].metadata["card_type_boost"] == 1.0


def test_priority_keeps_generic_normalized_document_as_primary_card():
    """primary_topic=common 인 normalized document 는 mismatch 가 아니므로 primary_card 유지."""
    cands = [
        _make_kc(
            chunk_id="kc_common",
            file_name="common_card.txt",
            primary_topic="common",
            topic_tags=["common"],
            card_type="workflow",
        ),
    ]
    qm = extract_query_metadata("메타 피드광고 셋팅 방법 알려줘")
    rerank_simple(cands, query="메타 피드광고 셋팅 방법 알려줘", query_metadata=qm)
    out = apply_normalized_document_priority(cands, query_metadata=qm)
    md = out[0].metadata
    assert md["retrieval_role"] == "primary_card"
    assert md["topic_mismatch_demoted"] is False
    # kc_boost 가 적용되어 있다 (기존 동작 유지)
    assert md["knowledge_card_boost"] == pytest.approx(
        float(settings.knowledge_card_content_boost), rel=1e-6
    )


def test_priority_does_not_demote_when_query_topic_is_empty():
    """query_topic 이 없으면 mismatch 격하가 발생하지 않는다."""
    cands = [
        _make_kc(
            chunk_id="kc_kakao",
            file_name="kakao_card.txt",
            primary_topic="kakao",
            card_type="workflow",
        ),
    ]
    qm = extract_query_metadata("그냥 가이드 알려줘")  # topic keyword 없음
    rerank_simple(cands, query="그냥 가이드 알려줘", query_metadata=qm)
    out = apply_normalized_document_priority(cands, query_metadata=qm)
    md = out[0].metadata
    assert md["retrieval_role"] == "primary_card"
    assert md["topic_mismatch_demoted"] is False


# ===========================================================================
# 4) Raw chunk mismatch — 같은 source_file 의 raw evidence 격하
# ===========================================================================
def test_priority_does_not_promote_topic_mismatch_raw_to_raw_evidence():
    """
    같은 file 안에 normalized document 가 있어도, raw chunk 가 명확한 topic
    mismatch 면 raw_evidence 로 승격하지 않는다 (raw_fallback 유지).
    """
    cands = [
        _make_kc(
            chunk_id="kc_meta",
            file_name="shared_doc.txt",
            primary_topic="meta",
            card_type="workflow",
        ),
        # 같은 파일의 raw chunk 인데 topic 이 kakao 로 잘못 잡혀 있는 케이스
        _make_raw(
            chunk_id="raw_kakao_same_file",
            file_name="shared_doc.txt",
            primary_topic="kakao",
            topic_tags=["kakao"],
            uploaded_category="guide",
            source_type="guide",
            content_type="text",
        ),
    ]
    qm = extract_query_metadata("메타 캠페인 셋팅 알려줘")
    rerank_simple(cands, query="메타 캠페인 셋팅 알려줘", query_metadata=qm)
    out = apply_normalized_document_priority(cands, query_metadata=qm)
    by_id = {c.chunk_id: c for c in out}
    # raw chunk 가 같은 파일임에도 topic mismatch 라 raw_fallback 으로 유지된다.
    assert by_id["raw_kakao_same_file"].metadata["retrieval_role"] == "raw_fallback"
    assert by_id["raw_kakao_same_file"].metadata["topic_mismatch_demoted"] is True


# ===========================================================================
# 5) 시나리오: meta 질문 + kakao 후보 + common 후보
# ===========================================================================
def test_meta_question_kakao_chunk_demoted_below_meta(monkeypatch):
    """
    실제 문제 시나리오:
        질문: "메타 피드광고 셋팅 방법 알려줘"
        - A: primary_topic=meta, normalized_document (workflow)
        - B: primary_topic=kakao, normalized_document (workflow), similarity 약간 더 높음
        - C: primary_topic=common, normalized_document (workflow)
    기대:
        - meta(A) 가 1순위
        - kakao(B) 는 primary_card 가 아니다
        - topic_mismatch_count >= 1
        - common(C) 은 mismatch 가 아님 (primary_card 유지)
    """
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    chunks = [
        _make_kc(
            chunk_id="A_meta",
            file_name="meta_workflow.md",
            primary_topic="meta",
            card_type="workflow",
            score=0.55,
        ),
        _make_kc(
            chunk_id="B_kakao",
            file_name="kakao_notice.md",
            primary_topic="kakao",
            card_type="workflow",
            score=0.70,  # similarity 가 높아도 mismatch 면 primary 가 되면 안 됨
        ),
        _make_kc(
            chunk_id="C_common",
            file_name="common_tips.md",
            primary_topic="common",
            topic_tags=["common"],
            card_type="workflow",
            score=0.45,
        ),
    ]
    retriever = _make_retriever(chunks)
    details = retriever.retrieve_with_details(
        "메타 피드광고 셋팅 방법 알려줘", top_k=5, max_per_file=10
    )
    assert details.passed
    by_id = {c.chunk_id: c for c in details.passed}
    roles = {cid: c.metadata.get("retrieval_role") for cid, c in by_id.items()}

    # meta 는 primary_card
    assert roles["A_meta"] == "primary_card"
    # kakao 는 primary_card 가 아니어야 한다 (raw_fallback 으로 격하)
    assert roles["B_kakao"] == "raw_fallback"
    # common 은 primary_card 유지
    assert roles["C_common"] == "primary_card"

    # final_score: kakao 가 meta 보다 위에 있으면 안 됨
    meta_score = by_id["A_meta"].final_score
    kakao_score = by_id["B_kakao"].final_score
    assert meta_score > kakao_score, (
        f"meta final_score 가 kakao final_score 보다 커야 함 "
        f"(meta={meta_score:.4f}, kakao={kakao_score:.4f})"
    )

    # 진단 카운트
    assert details.summary["topic_mismatch_count"] >= 1
    assert details.summary["topic_mismatch_demoted_count"] >= 1
    assert details.summary["query_topic"] == "meta"


def test_meta_question_kakao_chunk_final_score_dropped_by_penalty(monkeypatch):
    """kakao chunk 의 final_score 가 topic_mismatch_penalty 만큼 작아진다."""
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    chunks = [
        _make_kc(
            chunk_id="kakao_kc",
            file_name="kakao_notice.md",
            primary_topic="kakao",
            card_type="workflow",
            score=0.70,
        ),
    ]
    # 동일 chunk 를 query topic 이 없는 경우와 비교한다.
    r1 = _make_retriever([_make_kc(
        chunk_id="kakao_kc",
        file_name="kakao_notice.md",
        primary_topic="kakao",
        card_type="workflow",
        score=0.70,
    )])
    d1 = r1.retrieve_with_details("메타 피드광고 셋팅 방법 알려줘", top_k=3)
    r2 = _make_retriever([_make_kc(
        chunk_id="kakao_kc",
        file_name="kakao_notice.md",
        primary_topic="kakao",
        card_type="workflow",
        score=0.70,
    )])
    d2 = r2.retrieve_with_details("그냥 가이드 알려줘", top_k=3)  # topic 없음

    meta_q_score = d1.passed[0].final_score
    no_q_score = d2.passed[0].final_score
    # 메타 질문에서는 mismatch penalty 와 kc_boost 미적용으로 점수가 더 낮아야 함.
    assert meta_q_score < no_q_score, (
        f"meta 질문 시 kakao chunk 점수는 더 낮아져야 함 "
        f"(meta={meta_q_score:.4f}, no-topic={no_q_score:.4f})"
    )


# ===========================================================================
# 6) query_topic 이 None 이면 과도하게 penalty 가 걸리지 않는다
# ===========================================================================
def test_no_query_topic_does_not_apply_topic_penalty(monkeypatch):
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    chunks = [
        _make_kc(
            chunk_id="kc_kakao",
            file_name="kakao_card.md",
            primary_topic="kakao",
            card_type="workflow",
            score=0.60,
        ),
    ]
    retriever = _make_retriever(chunks)
    details = retriever.retrieve_with_details("그냥 운영 가이드 알려줘", top_k=3)
    assert details.passed
    md = details.passed[0].metadata
    # query topic 이 없을 때 mismatch penalty 가 걸리면 안 됨
    assert md.get("topic_match") in {"none", "neutral"}
    assert md.get("topic_mismatch_demoted") is False
    assert details.summary["topic_mismatch_demoted_count"] == 0


# ===========================================================================
# 7) qa_pipeline 통합 — top-level diagnostics 에 demoted_count 노출
# ===========================================================================
def test_qa_pipeline_exposes_topic_mismatch_demoted_count(monkeypatch):
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    chunks = [
        _make_kc(
            chunk_id="A_meta",
            file_name="meta_workflow.md",
            primary_topic="meta",
            card_type="workflow",
            score=0.55,
        ),
        _make_kc(
            chunk_id="B_kakao",
            file_name="kakao_notice.md",
            primary_topic="kakao",
            card_type="workflow",
            score=0.70,
        ),
    ]
    pipeline, _gen = _make_pipeline(chunks)
    result = pipeline.ask(
        "메타 피드광고 셋팅 방법 알려줘", top_k=5, save_log=False
    )

    assert result["query_topic"] == "meta"
    # 격하 1건 이상
    assert result["topic_mismatch_demoted_count"] >= 1
    # primary_card 카운트는 meta 만 1개
    assert result["primary_normalized_document_count"] >= 1
    # kakao chunk 는 primary 가 아님
    primary_ids = {c.chunk_id for c in result["primary_normalized_documents"]}
    assert "B_kakao" not in primary_ids
    assert "A_meta" in primary_ids


def test_qa_pipeline_empty_question_default_includes_demoted_count():
    pipeline, _gen = _make_pipeline([])
    result = pipeline.ask("   ", save_log=False)
    assert result["topic_mismatch_demoted_count"] == 0


# ===========================================================================
# 8) Slack --debug 진단 출력
# ===========================================================================
def _slack_result(*, demoted_count: int) -> Dict[str, Any]:
    return {
        "answer": "## 1. 결론\n메타 피드광고 셋팅 절차 요약.\n",
        "answer_mode": "knowledge_card",
        "primary_normalized_document_count": 1,
        "raw_evidence_count": 0,
        "raw_fallback_count": 1,
        "sources": {
            "primary_normalized_documents": [
                {
                    "label": "meta_workflow.md · 셋팅 절차",
                    "preview": "1) 광고주 확인 ...",
                    "file_name": "meta_workflow.md",
                    "content_type": "knowledge_card",
                    "primary_topic": "meta",
                    "retrieval_role": "primary_card",
                    "final_score": 0.812,
                    "topic_match": "match",
                    "topic_mismatch_demoted": False,
                }
            ],
            "raw_evidence": [],
            "raw_fallback": [
                {
                    "label": "kakao_notice.md · 카카오 메시지 주의사항",
                    "preview": "카카오 메시지 발송 시 ...",
                    "file_name": "kakao_notice.md",
                    "content_type": "knowledge_card",
                    "primary_topic": "kakao",
                    "retrieval_role": "raw_fallback",
                    "final_score": 0.42,
                    "topic_match": "mismatch",
                    "topic_mismatch_demoted": True,
                }
            ],
        },
        "diagnostics": {
            "answer_mode": "knowledge_card",
            "primary_normalized_document_count": 1,
            "raw_evidence_count": 0,
            "raw_fallback_count": 1,
            "generation_skipped": False,
            "model_name": "fake-model",
            "query_topic": "meta",
            "query_intent": ["procedure"],
            "query_date": None,
            "retrieved_count": 5,
            "passed_count": 2,
            "topic_mismatch_count": 1,
            "topic_mismatch_demoted_count": demoted_count,
            "normalized_document_candidate_count": 2,
            "raw_candidate_count": 3,
        },
    }


def test_slack_debug_shows_topic_mismatch_demoted_count():
    text = formatter.format_qa_result(_slack_result(demoted_count=1), debug=True)
    assert "*진단*" in text
    assert "topic_mismatch_demoted_count: 1" in text
    # 기존 step1 진단 라벨은 그대로 유지
    assert "query_topic" in text
    assert "`meta`" in text
    assert "retrieved_count: 5" in text


def test_slack_debug_omits_demote_line_when_count_is_zero():
    text = formatter.format_qa_result(_slack_result(demoted_count=0), debug=True)
    # demoted_count = 0 일 때는 라인이 나오지 않아야 한다 (debug noise 최소화).
    assert "topic_mismatch_demoted_count" not in text
    # 다른 진단 라벨은 여전히 표시
    assert "query_topic" in text


def test_slack_default_output_hides_topic_mismatch_demoted_count():
    text = formatter.format_qa_result(_slack_result(demoted_count=1))
    # 기본 출력은 답변만 — 진단 라벨이 나오면 안 된다.
    assert "topic_mismatch_demoted_count" not in text
    assert "*진단*" not in text
    assert "참고 근거 (debug)" not in text
