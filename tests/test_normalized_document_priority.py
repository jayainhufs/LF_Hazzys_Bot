"""
test_normalized_document_priority.py
====================================
MVP 2차 Step 3 — Normalized Document 우선순위 점검 테스트.

이 Step 은 검색 정책을 **새로** 추가하지 않는다. Step 1 (Retrieval Diagnostics)
+ Step 2 (Topic-aware Retrieval / Penalty) 의 결과로 다음이 일관되게
동작하는지 **점검 + 회귀 방어** 한다.

검증 관점:

1) Normalized Document **인식**
   - 신규 ``content_type="normalized_document"``
   - legacy ``content_type="knowledge_card"``
   - ``source_type="llm_normalized"``
   - 일반 raw chunk 는 ``is_normalized_document_chunk == False``

2) Document type 해석 (`get_normalized_document_type`)
   - 신규 ``normalized_document_type`` 우선
   - legacy ``card_type`` fallback

3) Match Normalized Document **primary 승격**
   - query_topic 과 match 인 normalized document → ``retrieval_role="primary_card"``
   - raw chunk 보다 final_score 가 높다.

4) Mismatch Normalized Document **demote** (Step 2 정책과 충돌 없음)
   - query_topic 과 mismatch 인 normalized document → ``retrieval_role="raw_fallback"``
   - ``primary_normalized_documents`` 목록에 포함되지 않는다.

5) ``primary_normalized_document_count`` / ``primary_normalized_documents`` 집계 정확성

6) Legacy compatibility
   - ``content_type="knowledge_card"`` chunk 도 동일하게 primary 로 승격
   - ``card_id`` / ``card_type`` legacy metadata 키 유지
   - ``answer_mode="knowledge_card"`` legacy 라벨 유지

7) 실제 문제 케이스 (meta workflow vs meta raw vs kakao normalized)

8) Slack ``--debug`` 진단 노출 / 기본 출력 진단 숨김
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from src.config import settings
from src.rag.prompt_builder import split_chunks_by_retrieval_role
from src.rag.qa_pipeline import QAPipeline
from src.rag.reranker import (
    apply_normalized_document_priority,
    extract_query_metadata,
    get_normalized_document_type,
    is_knowledge_card_chunk,
    is_normalized_document_chunk,
    rerank_simple,
)
from src.rag.retriever import Retriever
from src.schemas import RetrievedChunk
from src.slack_bot import formatter
from src.slack_bot.qa_adapter import answer_slack_question


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------
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


def _make_nd(
    *,
    chunk_id: str,
    file_name: str,
    primary_topic: str,
    document_type: str = "workflow",
    content_type: str = "knowledge_card",
    use_legacy_card_keys: bool = True,
    topic_tags: Optional[List[str]] = None,
    score: float = 0.55,
    uploaded_category: str = "guide",
) -> RetrievedChunk:
    """Normalized Document chunk 를 만든다.

    use_legacy_card_keys=True (기본): legacy ``card_id`` / ``card_type`` 키도 함께 채움.
    """
    md: Dict[str, Any] = {
        "source_weight": float(settings.normalization_card_source_weight),
        "uploaded_category": uploaded_category,
        "file_name": file_name,
        "source_type": "llm_normalized",
        "content_type": content_type,
        "normalized_document_id": chunk_id,
        "normalized_document_type": document_type,
        "primary_topic": primary_topic,
        "topic_tags": list(topic_tags or [primary_topic]),
        "parent_raw_chunk_ids": [],
    }
    if use_legacy_card_keys:
        md["card_id"] = chunk_id
        md["card_type"] = document_type
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc_{file_name}",
        file_name=file_name,
        source_type="llm_normalized",
        uploaded_category=uploaded_category,
        section_title=None,
        content_type=content_type,
        content=f"[normalized_document content for {chunk_id}]",
        score=float(score),
        final_score=float(score),
        metadata=md,
    )


def _make_raw(
    *,
    chunk_id: str,
    file_name: str,
    primary_topic: Optional[str] = None,
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


def _make_pipeline(chunks: List[RetrievedChunk]) -> tuple:
    gen = FakeGenerator()
    pipe = QAPipeline(
        retriever=Retriever(embedder=FakeEmbedder(), vector_store=FakeVectorStore(chunks)),
        generator=gen,  # type: ignore[arg-type]
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
    )
    return pipe, gen


# ===========================================================================
# 1) Normalized Document 인식
# ===========================================================================
def test_recognizes_chunk_with_new_content_type_normalized_document():
    nd = _make_nd(
        chunk_id="nd1",
        file_name="meta_workflow.md",
        primary_topic="meta",
        content_type="normalized_document",
    )
    assert is_normalized_document_chunk(nd) is True
    # legacy alias 도 같은 결과
    assert is_knowledge_card_chunk(nd) is True


def test_recognizes_chunk_with_legacy_content_type_knowledge_card():
    nd = _make_nd(
        chunk_id="nd1",
        file_name="meta_workflow.md",
        primary_topic="meta",
        content_type="knowledge_card",  # legacy
    )
    assert is_normalized_document_chunk(nd) is True


def test_recognizes_chunk_via_source_type_only():
    """content_type 이 다른 값이어도 source_type=llm_normalized 면 인식."""
    nd = _make_nd(
        chunk_id="nd1",
        file_name="meta_workflow.md",
        primary_topic="meta",
        content_type="text",  # 의도적으로 generic content_type
    )
    # source_type=llm_normalized 가 이미 metadata 에 들어가 있다.
    assert is_normalized_document_chunk(nd) is True


def test_does_not_recognize_plain_raw_chunk():
    raw = _make_raw(
        chunk_id="r1",
        file_name="meta_guide.txt",
        primary_topic="meta",
        uploaded_category="guide",
        source_type="guide",
        content_type="text",
    )
    assert is_normalized_document_chunk(raw) is False
    assert is_knowledge_card_chunk(raw) is False


def test_recognizes_dict_metadata():
    """RetrievedChunk 객체 외에 dict-like metadata 도 받는다."""
    md_new = {"content_type": "normalized_document"}
    md_legacy = {"content_type": "knowledge_card"}
    md_other = {"content_type": "text", "source_type": "llm_normalized"}
    md_raw = {"content_type": "text", "source_type": "guide"}
    assert is_normalized_document_chunk(md_new) is True
    assert is_normalized_document_chunk(md_legacy) is True
    assert is_normalized_document_chunk(md_other) is True
    assert is_normalized_document_chunk(md_raw) is False


# ===========================================================================
# 2) Document type 해석
# ===========================================================================
def test_get_normalized_document_type_prefers_new_key():
    nd = _make_nd(
        chunk_id="nd1",
        file_name="x.md",
        primary_topic="meta",
        document_type="workflow",
    )
    assert get_normalized_document_type(nd) == "workflow"


def test_get_normalized_document_type_falls_back_to_legacy_card_type():
    md = {"card_type": "checklist"}  # legacy key only
    assert get_normalized_document_type(md) == "checklist"


def test_get_normalized_document_type_returns_empty_when_missing():
    md = {"content_type": "knowledge_card"}
    assert get_normalized_document_type(md) == ""
    assert get_normalized_document_type(None) == ""


# ===========================================================================
# 3) Match Normalized Document primary 승격 & raw chunk 보다 우선
# ===========================================================================
def test_match_normalized_document_promoted_to_primary_card():
    """query_topic 과 match 인 normalized document 는 primary_card 로 승격."""
    cands = [
        _make_nd(
            chunk_id="nd_meta",
            file_name="meta_workflow.md",
            primary_topic="meta",
            document_type="workflow",
        ),
    ]
    qm = extract_query_metadata("메타 피드광고 셋팅 방법 알려줘")
    rerank_simple(cands, query="메타 피드광고 셋팅 방법 알려줘", query_metadata=qm)
    out = apply_normalized_document_priority(cands, query_metadata=qm)
    md = out[0].metadata
    assert md["retrieval_role"] == "primary_card"
    assert md["topic_mismatch_demoted"] is False
    # Normalized Document content boost 적용
    assert md["normalized_document_boost"] == pytest.approx(
        float(settings.knowledge_card_content_boost), rel=1e-6
    )


def test_match_normalized_document_outranks_raw_chunk(monkeypatch):
    """match Normalized Document 는 같은 topic 의 raw chunk 보다 final_score 가 더 높다."""
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    nd = _make_nd(
        chunk_id="nd_meta",
        file_name="meta_workflow.md",
        primary_topic="meta",
        document_type="workflow",
        score=0.55,
    )
    raw = _make_raw(
        chunk_id="r_meta",
        file_name="meta_guide.txt",
        primary_topic="meta",
        uploaded_category="guide",
        source_type="guide",
        content_type="text",
        score=0.55,
    )
    retriever = Retriever(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore([nd, raw]),
    )
    details = retriever.retrieve_with_details(
        "메타 피드광고 셋팅 방법 알려줘", top_k=5
    )
    by_id = {c.chunk_id: c for c in details.passed}
    # NormalizedDocument 가 raw 보다 final_score 가 높아야 한다.
    assert by_id["nd_meta"].final_score > by_id["r_meta"].final_score
    # role 확인
    assert by_id["nd_meta"].metadata["retrieval_role"] == "primary_card"
    assert by_id["r_meta"].metadata["retrieval_role"] in {
        "raw_evidence",
        "raw_fallback",
    }


# ===========================================================================
# 4) Mismatch Normalized Document 는 Step 2 정책대로 demote 유지
# ===========================================================================
def test_mismatch_normalized_document_remains_demoted_after_step2_policy():
    """Step 2 에서 도입한 demote 가 Step 3 동작과 상충하지 않음을 확인."""
    cands = [
        _make_nd(
            chunk_id="nd_meta",
            file_name="meta_workflow.md",
            primary_topic="meta",
            document_type="workflow",
            score=0.55,
        ),
        _make_nd(
            chunk_id="nd_kakao",
            file_name="kakao_notice.md",
            primary_topic="kakao",
            document_type="workflow",
            score=0.70,
        ),
    ]
    qm = extract_query_metadata("메타 피드광고 셋팅 방법 알려줘")
    rerank_simple(cands, query="메타 피드광고 셋팅 방법 알려줘", query_metadata=qm)
    out = apply_normalized_document_priority(cands, query_metadata=qm)

    by_id = {c.chunk_id: c for c in out}
    assert by_id["nd_meta"].metadata["retrieval_role"] == "primary_card"
    # mismatch normalized 는 primary 가 되면 안 된다.
    assert by_id["nd_kakao"].metadata["retrieval_role"] == "raw_fallback"
    assert by_id["nd_kakao"].metadata["topic_mismatch_demoted"] is True

    # split_chunks_by_retrieval_role 가 mismatch normalized 를 primary 그룹에서 제외하는지.
    groups = split_chunks_by_retrieval_role(out)
    primary_ids = {c.chunk_id for c in groups["primary_cards"]}
    fallback_ids = {c.chunk_id for c in groups["raw_fallback"]}
    assert "nd_meta" in primary_ids
    assert "nd_kakao" not in primary_ids
    assert "nd_kakao" in fallback_ids


# ===========================================================================
# 5) primary_normalized_document_count / primary_normalized_documents 집계
# ===========================================================================
def test_pipeline_primary_normalized_document_count_increases_for_match(monkeypatch):
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    chunks = [
        _make_nd(
            chunk_id="nd_meta",
            file_name="meta_workflow.md",
            primary_topic="meta",
            document_type="workflow",
            score=0.60,
        ),
        _make_nd(
            chunk_id="nd_kakao",
            file_name="kakao_notice.md",
            primary_topic="kakao",
            document_type="workflow",
            score=0.70,
        ),
    ]
    pipeline, _gen = _make_pipeline(chunks)
    result = pipeline.ask(
        "메타 피드광고 셋팅 방법 알려줘", top_k=5, save_log=False
    )

    # 1개 (nd_meta) 만 primary 로 잡혀야 한다.
    assert result["primary_normalized_document_count"] == 1
    # legacy 호환 키도 같은 값
    assert result["primary_card_count"] == 1
    # 목록에 nd_meta 만, nd_kakao 는 없어야 한다.
    primary_ids = {c.chunk_id for c in result["primary_normalized_documents"]}
    assert "nd_meta" in primary_ids
    assert "nd_kakao" not in primary_ids


def test_pipeline_primary_normalized_documents_have_id_metadata(monkeypatch):
    """primary_normalized_documents 에 normalized_document_id (또는 legacy card_id) 가 포함."""
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    chunks = [
        _make_nd(
            chunk_id="nd_meta_id",
            file_name="meta_workflow.md",
            primary_topic="meta",
            document_type="workflow",
            score=0.60,
            use_legacy_card_keys=True,
        ),
    ]
    pipeline, _gen = _make_pipeline(chunks)
    result = pipeline.ask(
        "메타 피드광고 셋팅 방법 알려줘", top_k=5, save_log=False
    )
    primary = result["primary_normalized_documents"]
    assert len(primary) == 1
    md = primary[0].metadata
    # 신규 키 / legacy 키 모두 노출
    assert md.get("normalized_document_id") == "nd_meta_id"
    assert md.get("card_id") == "nd_meta_id"
    assert md.get("normalized_document_type") == "workflow"
    assert md.get("card_type") == "workflow"


def test_pipeline_answer_mode_is_knowledge_card_for_legacy_compat(monkeypatch):
    """answer_mode 는 legacy 호환을 위해 ``knowledge_card`` 문자열을 유지한다."""
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    chunks = [
        _make_nd(
            chunk_id="nd_meta",
            file_name="meta_workflow.md",
            primary_topic="meta",
            document_type="workflow",
            score=0.60,
        ),
    ]
    pipeline, _gen = _make_pipeline(chunks)
    result = pipeline.ask("메타 피드광고 셋팅 방법 알려줘", top_k=3, save_log=False)
    assert result["answer_mode"] == "knowledge_card"


def test_pipeline_falls_back_to_raw_when_only_raw_chunks_present(monkeypatch):
    """raw chunk 만 있으면 answer_mode 는 raw_fallback 으로 유지된다."""
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    chunks = [
        _make_raw(
            chunk_id="r1",
            file_name="meta_guide.txt",
            primary_topic="meta",
            uploaded_category="guide",
            source_type="guide",
            content_type="text",
            score=0.65,
        ),
    ]
    pipeline, _gen = _make_pipeline(chunks)
    result = pipeline.ask("메타 피드광고 셋팅 방법 알려줘", top_k=3, save_log=False)
    assert result["answer_mode"] == "raw_fallback"
    assert result["primary_normalized_document_count"] == 0
    assert result["primary_card_count"] == 0


# ===========================================================================
# 6) Legacy compatibility: content_type="knowledge_card" 도 primary 로 승격
# ===========================================================================
def test_legacy_content_type_knowledge_card_chunk_still_promoted(monkeypatch):
    """과거 색인된 ``content_type="knowledge_card"`` chunk 도 일관되게 primary 로 승격."""
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    chunks = [
        _make_nd(
            chunk_id="legacy_nd",
            file_name="meta_workflow.md",
            primary_topic="meta",
            document_type="workflow",
            content_type="knowledge_card",  # legacy 명칭
            score=0.55,
        ),
    ]
    pipeline, _gen = _make_pipeline(chunks)
    result = pipeline.ask("메타 피드광고 셋팅 방법 알려줘", top_k=3, save_log=False)
    assert result["primary_normalized_document_count"] == 1
    assert result["answer_mode"] == "knowledge_card"
    primary = result["primary_normalized_documents"]
    assert len(primary) == 1
    # 격하 안 됨
    assert primary[0].metadata["retrieval_role"] == "primary_card"
    assert primary[0].metadata["topic_mismatch_demoted"] is False


def test_legacy_knowledge_card_boost_keys_are_filled():
    """legacy 진단 필드 ``knowledge_card_boost`` / ``card_type_boost`` 가 유지된다."""
    cands = [
        _make_nd(
            chunk_id="nd_meta",
            file_name="meta_workflow.md",
            primary_topic="meta",
            document_type="workflow",
        ),
    ]
    qm = extract_query_metadata("메타 셋팅")
    rerank_simple(cands, query="메타 셋팅", query_metadata=qm)
    out = apply_normalized_document_priority(cands, query_metadata=qm)
    md = out[0].metadata
    # 신규 / legacy 키 모두 채워진다.
    assert md["normalized_document_boost"] == md["knowledge_card_boost"]
    assert md["normalized_document_type_boost"] == md["card_type_boost"]
    assert md["normalized_document_type_match"] == md["card_type_match"]


# ===========================================================================
# 7) 실제 문제 케이스: meta workflow vs meta raw vs kakao normalized
# ===========================================================================
def test_real_world_scenario_meta_workflow_vs_meta_raw_vs_kakao_normalized(
    monkeypatch,
):
    """
    질문: "메타 피드광고 셋팅 방법 알려줘"
    후보:
      A. primary_topic=meta, content_type=normalized_document  (workflow)
      B. primary_topic=meta, content_type=raw (guide)
      C. primary_topic=kakao, content_type=normalized_document (workflow)

    기대:
      - A 는 primary_card.
      - B 는 raw_evidence 또는 raw_fallback (primary 가 아님).
      - C 는 Step 2 정책으로 raw_fallback 으로 demote.
      - primary_normalized_document_count == 1 (A 만).
      - answer_mode == "knowledge_card".
    """
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    chunks = [
        _make_nd(
            chunk_id="A_meta_nd",
            file_name="meta_workflow.md",
            primary_topic="meta",
            document_type="workflow",
            content_type="normalized_document",
            score=0.55,
        ),
        _make_raw(
            chunk_id="B_meta_raw",
            file_name="meta_extra_notes.txt",
            primary_topic="meta",
            uploaded_category="guide",
            source_type="guide",
            content_type="text",
            score=0.60,
        ),
        _make_nd(
            chunk_id="C_kakao_nd",
            file_name="kakao_notice.md",
            primary_topic="kakao",
            document_type="workflow",
            content_type="normalized_document",
            score=0.70,
        ),
    ]
    pipeline, _gen = _make_pipeline(chunks)
    result = pipeline.ask(
        "메타 피드광고 셋팅 방법 알려줘", top_k=5, save_log=False
    )

    primary_ids = {c.chunk_id for c in result["primary_normalized_documents"]}
    raw_ev_ids = {c.chunk_id for c in result["raw_evidence"]}
    raw_fb_ids = {c.chunk_id for c in result["raw_fallback"]}

    # A 만 primary
    assert primary_ids == {"A_meta_nd"}, f"primary={primary_ids}"
    # B 는 raw_evidence 또는 raw_fallback
    assert "B_meta_raw" in (raw_ev_ids | raw_fb_ids)
    # C 는 mismatch 로 demote 됨 → raw_fallback
    assert "C_kakao_nd" in raw_fb_ids

    # 집계
    assert result["primary_normalized_document_count"] == 1
    assert result["answer_mode"] == "knowledge_card"
    # Step 2 진단도 유지
    assert result["topic_mismatch_demoted_count"] >= 1


# ===========================================================================
# 8) Slack --debug 진단 노출 / 기본 출력 진단 숨김
# ===========================================================================
class _StubPipeline:
    """qa_adapter 가 호출하는 pipeline.ask 만 충족하는 stub."""

    def __init__(self, result: Dict[str, Any]) -> None:
        self._result = result

    def ask(
        self,
        question: str,  # noqa: ARG002
        top_k: Optional[int] = None,  # noqa: ARG002
        save_log: bool = True,  # noqa: ARG002
    ) -> Dict[str, Any]:
        return dict(self._result)


def _build_pipeline_result_with_match_normalized() -> Dict[str, Any]:
    """answer_slack_question 이 받아들이는 형식의 pipeline result 를 만든다."""
    nd_meta = _make_nd(
        chunk_id="nd_meta",
        file_name="meta_workflow.md",
        primary_topic="meta",
        document_type="workflow",
        content_type="knowledge_card",
        score=0.60,
    )
    nd_meta.final_score = 0.95
    nd_meta.metadata.update(
        {
            "retrieval_role": "primary_card",
            "topic_match": "match",
            "date_match": "none",
            "normalized_document_boost": 1.35,
            "topic_boost": 1.20,
            "date_boost": 1.0,
            "topic_mismatch_demoted": False,
        }
    )

    nd_kakao = _make_nd(
        chunk_id="nd_kakao",
        file_name="kakao_notice.md",
        primary_topic="kakao",
        document_type="workflow",
        content_type="knowledge_card",
        score=0.70,
    )
    nd_kakao.final_score = 0.42
    nd_kakao.metadata.update(
        {
            "retrieval_role": "raw_fallback",
            "topic_match": "mismatch",
            "date_match": "none",
            "normalized_document_boost": 1.0,
            "topic_boost": 0.80,
            "date_boost": 1.0,
            "topic_mismatch_demoted": True,
        }
    )

    return {
        "question": "메타 피드광고 셋팅 방법 알려줘",
        "answer": "## 1. 결론\n메타 피드광고 셋팅 절차 요약.\n",
        "model": "fake-model",
        "passed": [nd_meta, nd_kakao],
        "candidates": [nd_meta, nd_kakao],
        "answer_mode": "knowledge_card",
        "answer_format_label": "default",
        "primary_normalized_document_count": 1,
        "primary_normalized_documents": [nd_meta],
        "primary_card_count": 1,
        "raw_evidence_count": 0,
        "raw_fallback_count": 1,
        "primary_cards": [nd_meta],
        "raw_evidence": [],
        "raw_fallback": [nd_kakao],
        "retrieval_summary": {
            "query_topic": "meta",
            "query_topics": ["meta"],
            "query_intent": ["procedure"],
            "query_date": None,
            "retrieved_count": 2,
            "passed_count": 2,
            "topic_mismatch_count": 1,
            "topic_mismatch_demoted_count": 1,
            "normalized_document_candidate_count": 2,
            "raw_candidate_count": 0,
        },
        "query_topic": "meta",
        "query_topics": ["meta"],
        "query_intent": ["procedure"],
        "query_date": None,
        "retrieved_count": 2,
        "passed_count": 2,
        "topic_mismatch_count": 1,
        "topic_mismatch_demoted_count": 1,
        "normalized_document_candidate_count": 2,
        "raw_candidate_count": 0,
    }


def test_slack_debug_shows_primary_normalized_document_count():
    result = answer_slack_question(
        question="메타 피드광고 셋팅 방법 알려줘",
        pipeline=_StubPipeline(_build_pipeline_result_with_match_normalized()),
    )
    text = formatter.format_qa_result(result, debug=True)

    # Step 1 / Step 2 진단 모두 유지
    assert "*진단*" in text
    assert "query_topic" in text
    assert "primary_normalized_document_count" in text
    assert "topic_mismatch_demoted_count" in text
    # primary normalized document 가 1건이라는 진단
    assert "primary_normalized_document_count: 1" in text


def test_slack_default_output_hides_primary_normalized_document_count():
    result = answer_slack_question(
        question="메타 피드광고 셋팅 방법 알려줘",
        pipeline=_StubPipeline(_build_pipeline_result_with_match_normalized()),
    )
    text = formatter.format_qa_result(result)
    # 기본 출력은 답변만 — 진단 라벨이 나오면 안 된다.
    assert "primary_normalized_document_count" not in text
    assert "*진단*" not in text
    assert "참고 근거 (debug)" not in text
