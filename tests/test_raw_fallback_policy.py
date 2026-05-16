"""
test_raw_fallback_policy.py
===========================
MVP 2차 Step 4 — Raw Fallback 오남용 방지 단위 테스트.

이 Step 은 raw_fallback 자체를 막지 않는다 — 정규화되지 않은 초기 데이터에서는
raw chunk 가 유일한 근거일 수 있기 때문이다. 대신 raw_fallback 기반 답변의
신뢰도를 외부에서 진단할 수 있도록 ``raw_fallback_only`` / ``evidence_strength``
/ ``weak_evidence_warning`` 등 진단 필드를 ``qa_pipeline`` / Slack adapter /
Slack formatter / Streamlit caption 에 노출한다.

검증 관점:

1) Evidence strength 분류
   - primary Normalized Document 존재 → ``strong``
   - raw_evidence 존재 → ``medium`` (or higher)
   - raw_fallback 만 존재 + 거의 모두 topic match / generic → ``medium``
   - raw_fallback 만 존재 + topic mismatch 다수 → ``weak`` 또는 ``insufficient``

2) raw_fallback_only 판정
   - primary normalized document 가 0개이고 raw_fallback 이 1건 이상이면 True
   - query_topic 이 None 이어도 raw_fallback_only 자체는 정상적으로 True 가 될 수 있음

3) weak_evidence_warning 정책
   - query_topic 이 명확하고
   - raw_fallback_count >= 2 이고
   - raw_fallback_topic_mismatch_ratio >= 0.7
   - 위 조건을 모두 만족할 때만 True. 보수적 정책.

4) Topic mismatch 처리
   - query_topic 이 None 이면 mismatch warning 이 과도하게 켜지지 않는다.
   - chunk topic 이 generic / common / unknown 이면 mismatch 로 보지 않는다.

5) 기존 answer_mode / Streamlit / Slack 호환성 유지
   - answer_mode 문자열 (knowledge_card / raw_fallback / insufficient_evidence) 유지.
   - 기존 진단 필드 (Step 1/2/3) 가 깨지지 않는다.

6) Slack debug 통합
   - ``--debug`` 모드에서 raw_fallback_only / evidence_strength / weak_evidence_warning 노출.
   - 기본 출력에서는 노출되지 않는다.

7) 실제 문제 케이스 시나리오 A/B/C 방어.

외부 Gemini API / ChromaDB 호출 없이 fake retriever / fake generator / fake
chunk 로 검증한다.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from src.config import settings
from src.rag.qa_pipeline import QAPipeline
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


def _make_nd(
    *,
    chunk_id: str,
    file_name: str,
    primary_topic: str,
    document_type: str = "workflow",
    content_type: str = "knowledge_card",
    topic_tags: Optional[List[str]] = None,
    score: float = 0.55,
    uploaded_category: str = "guide",
) -> RetrievedChunk:
    md: Dict[str, Any] = {
        "source_weight": float(settings.normalization_card_source_weight),
        "uploaded_category": uploaded_category,
        "file_name": file_name,
        "source_type": "llm_normalized",
        "content_type": content_type,
        "normalized_document_id": chunk_id,
        "normalized_document_type": document_type,
        "card_id": chunk_id,
        "card_type": document_type,
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
        retriever=Retriever(
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
            vector_store=FakeVectorStore(chunks),
        ),
        generator=gen,  # type: ignore[arg-type]
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
    )
    return pipe, gen


# ===========================================================================
# 1) Evidence strength 분류
# ===========================================================================
def test_evidence_strength_strong_when_primary_normalized_document_exists(monkeypatch):
    """primary Normalized Document 가 있으면 evidence_strength == 'strong'."""
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    chunks = [
        _make_nd(
            chunk_id="nd_meta",
            file_name="meta_workflow.md",
            primary_topic="meta",
            score=0.60,
        ),
    ]
    pipeline, _gen = _make_pipeline(chunks)
    result = pipeline.ask(
        "메타 피드광고 셋팅 방법 알려줘", top_k=5, save_log=False
    )
    assert result["evidence_strength"] == "strong"
    assert result["raw_fallback_only"] is False
    assert result["weak_evidence_warning"] is False
    assert result["primary_evidence_available"] is True
    # answer_mode 호환 유지
    assert result["answer_mode"] == "knowledge_card"


def test_evidence_strength_medium_when_only_raw_evidence_exists(monkeypatch):
    """primary normalized 0 + raw_evidence(같은 파일 raw chunk) 가 있으면 medium 이상."""
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    # 같은 파일에 normalized document + raw chunk 가 있어 raw chunk 가
    # raw_evidence 로 승격되도록 한다. 단, normalized document 의 topic 을
    # query 와 mismatch 로 두어 primary_card 승격은 막는다.
    chunks = [
        _make_nd(
            chunk_id="nd_kakao",
            file_name="shared_doc.md",
            primary_topic="kakao",
            score=0.50,
        ),
        _make_raw(
            chunk_id="r_meta",
            file_name="shared_doc.md",
            primary_topic="meta",
            uploaded_category="guide",
            source_type="guide",
            content_type="text",
            score=0.65,
        ),
    ]
    pipeline, _gen = _make_pipeline(chunks)
    result = pipeline.ask(
        "메타 피드광고 셋팅 방법 알려줘", top_k=5, save_log=False
    )
    # primary normalized 가 없고 raw_evidence 가 있어야 한다.
    assert result["primary_normalized_document_count"] == 0
    assert result["raw_evidence_count"] >= 1
    # raw_evidence 가 존재하므로 evidence_strength 는 medium 이상.
    assert result["evidence_strength"] in {"medium", "strong"}
    # primary_evidence_available 은 raw_evidence 만 있어도 True (Step 4 정의).
    assert result["primary_evidence_available"] is True
    # 격하된 nd_kakao 가 raw_fallback 으로 남을 수 있으므로 raw_fallback_only 자체는
    # 사용자 정의 ("primary_normalized_document_count == 0 and raw_fallback_count > 0")
    # 에 따라 True 일 수 있다. weak_evidence_warning 만 False 인지 확인한다.
    assert result["weak_evidence_warning"] is False


def test_evidence_strength_weak_when_only_topic_match_raw_fallback(monkeypatch):
    """raw_fallback 만 있고 모두 query_topic 과 매칭되면 medium 으로 분류 (강한 weak 아님)."""
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    chunks = [
        _make_raw(
            chunk_id="r_meta1",
            file_name="meta_guide.txt",
            primary_topic="meta",
            uploaded_category="guide",
            source_type="guide",
            content_type="text",
            score=0.65,
        ),
        _make_raw(
            chunk_id="r_meta2",
            file_name="meta_extra.txt",
            primary_topic="meta",
            uploaded_category="guide",
            source_type="guide",
            content_type="text",
            score=0.60,
        ),
    ]
    pipeline, _gen = _make_pipeline(chunks)
    result = pipeline.ask(
        "메타 피드광고 셋팅 방법 알려줘", top_k=5, save_log=False
    )
    # raw_fallback 만 있다.
    assert result["primary_normalized_document_count"] == 0
    assert result["raw_evidence_count"] == 0
    assert result["raw_fallback_count"] >= 1
    assert result["raw_fallback_only"] is True
    # 모두 topic match → medium 으로 분류 (insufficient/weak 아님)
    assert result["evidence_strength"] == "medium"
    assert result["weak_evidence_warning"] is False
    # answer_mode 호환 유지
    assert result["answer_mode"] == "raw_fallback"


def test_evidence_strength_weak_or_insufficient_when_mostly_mismatch_raw_fallback(
    monkeypatch,
):
    """raw_fallback 만 있고 대부분 topic mismatch 면 weak / insufficient 로 표시."""
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    # 시나리오 A 와 유사: query=meta, raw_fallback=kakao 2개 + common 1개
    chunks = [
        _make_raw(
            chunk_id="r_kakao1",
            file_name="kakao_notice1.txt",
            primary_topic="kakao",
            uploaded_category="kakao",
            source_type="kakao",
            content_type="conversation",
            score=0.65,
        ),
        _make_raw(
            chunk_id="r_kakao2",
            file_name="kakao_notice2.txt",
            primary_topic="kakao",
            uploaded_category="kakao",
            source_type="kakao",
            content_type="conversation",
            score=0.60,
        ),
        _make_raw(
            chunk_id="r_common",
            file_name="common_tips.txt",
            primary_topic="common",
            topic_tags=["common"],
            uploaded_category="guide",
            source_type="guide",
            content_type="text",
            score=0.55,
        ),
    ]
    pipeline, _gen = _make_pipeline(chunks)
    result = pipeline.ask(
        "메타 피드광고 셋팅 방법 알려줘", top_k=5, save_log=False
    )
    assert result["primary_normalized_document_count"] == 0
    assert result["raw_fallback_count"] >= 1
    assert result["raw_fallback_only"] is True
    # mismatch count 는 kakao 만 (common 은 generic → mismatch 아님).
    assert result["raw_fallback_topic_mismatch_count"] >= 2
    # mismatch ratio 가 0.7 이상이라 weak_evidence_warning 이 켜져야 한다.
    assert result["weak_evidence_warning"] is True
    assert result["evidence_strength"] in {"weak", "insufficient"}


# ===========================================================================
# 2) raw_fallback_only 판정
# ===========================================================================
def test_raw_fallback_only_true_when_no_primary(monkeypatch):
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
            score=0.60,
        ),
    ]
    pipeline, _gen = _make_pipeline(chunks)
    result = pipeline.ask("메타 피드광고 셋팅 방법 알려줘", top_k=3, save_log=False)
    assert result["raw_fallback_only"] is True
    assert result["raw_fallback_only_reason"] in {
        "no_primary_normalized_document",
        "no_normalized_document_candidate",
    }


def test_raw_fallback_only_false_when_primary_exists(monkeypatch):
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    chunks = [
        _make_nd(
            chunk_id="nd_meta",
            file_name="meta_workflow.md",
            primary_topic="meta",
            score=0.55,
        ),
    ]
    pipeline, _gen = _make_pipeline(chunks)
    result = pipeline.ask("메타 피드광고 셋팅 방법 알려줘", top_k=3, save_log=False)
    assert result["raw_fallback_only"] is False
    assert result["raw_fallback_only_reason"] is None


# ===========================================================================
# 3) weak_evidence_warning 정책
# ===========================================================================
def test_weak_evidence_warning_not_triggered_when_query_topic_is_none(monkeypatch):
    """query_topic 이 None 이면 mismatch warning 이 과도하게 켜지지 않는다."""
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    # raw_fallback chunk 가 여러 개 (다양한 topic) 있지만 query_topic 이 없는 케이스.
    chunks = [
        _make_raw(
            chunk_id="r1",
            file_name="kakao_notice.txt",
            primary_topic="kakao",
            uploaded_category="kakao",
            source_type="kakao",
            content_type="conversation",
            score=0.60,
        ),
        _make_raw(
            chunk_id="r2",
            file_name="other.txt",
            primary_topic="settlement",
            uploaded_category="guide",
            source_type="guide",
            content_type="text",
            score=0.55,
        ),
    ]
    pipeline, _gen = _make_pipeline(chunks)
    result = pipeline.ask("그냥 알려줘", top_k=3, save_log=False)

    # query_topic 이 None 인 케이스
    assert result["query_topic"] is None
    # raw_fallback_only 자체는 True 가 될 수 있다 (정상).
    assert result["raw_fallback_only"] is True
    # 하지만 weak warning 은 강하게 켜지지 않아야 한다.
    assert result["weak_evidence_warning"] is False
    # mismatch count 도 0 (query_topic 이 없어서 판단 불가).
    assert result["raw_fallback_topic_mismatch_count"] == 0


def test_common_topic_raw_fallback_not_counted_as_mismatch(monkeypatch):
    """primary_topic=common / unknown 인 raw_fallback 은 강한 mismatch 로 보지 않는다."""
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    chunks = [
        _make_raw(
            chunk_id="r_common1",
            file_name="common1.txt",
            primary_topic="common",
            topic_tags=["common"],
            uploaded_category="guide",
            source_type="guide",
            content_type="text",
            score=0.60,
        ),
        _make_raw(
            chunk_id="r_unknown1",
            file_name="common2.txt",
            primary_topic="unknown",
            topic_tags=["unknown"],
            uploaded_category="guide",
            source_type="guide",
            content_type="text",
            score=0.55,
        ),
    ]
    pipeline, _gen = _make_pipeline(chunks)
    result = pipeline.ask(
        "메타 피드광고 셋팅 방법 알려줘", top_k=3, save_log=False
    )
    # raw_fallback_only 자체는 True 가 될 수 있지만, generic topic 이므로 mismatch 0.
    assert result["raw_fallback_only"] is True
    assert result["raw_fallback_topic_mismatch_count"] == 0
    assert result["weak_evidence_warning"] is False


# ===========================================================================
# 4) 기존 answer_mode 호환성
# ===========================================================================
def test_raw_fallback_answer_mode_compat_when_raw_fallback_only(monkeypatch):
    """raw_fallback_only 상태에서도 answer_mode=raw_fallback 호환 유지."""
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
    # answer_mode 라벨은 기존 호환 유지.
    assert result["answer_mode"] == "raw_fallback"
    # Step 4 진단 필드와 함께 노출된다.
    assert result["raw_fallback_only"] is True


def test_insufficient_evidence_keeps_existing_threshold_behavior():
    """근거 부족 시 기존 insufficient_evidence answer_mode 가 유지된다."""
    # 후보가 전혀 없는 케이스 → insufficient_evidence
    pipeline, _gen = _make_pipeline([])
    result = pipeline.ask("메타 셋팅", top_k=3, save_log=False)
    assert result["generation_skipped"] is True
    assert result["answer_mode"] == "insufficient_evidence"
    # Step 4 진단 필드 default
    assert result["evidence_strength"] == "insufficient"
    assert result["raw_fallback_only"] is False
    assert result["weak_evidence_warning"] is False


def test_empty_question_returns_step4_default_diagnostics():
    pipeline, _gen = _make_pipeline([])
    result = pipeline.ask("   ", save_log=False)
    # empty 케이스에도 Step 4 default 가 채워져야 한다.
    assert result["evidence_strength"] == "insufficient"
    assert result["raw_fallback_only"] is False
    assert result["raw_fallback_only_reason"] is None
    assert result["raw_fallback_topic_mismatch_count"] == 0
    assert result["raw_fallback_topic_mismatch_ratio"] == 0.0
    assert result["primary_evidence_available"] is False
    assert result["normalized_document_available"] is False
    assert result["weak_evidence_warning"] is False


# ===========================================================================
# 5) Slack debug 통합 — adapter / formatter
# ===========================================================================
class _StubPipeline:
    """qa_adapter 가 호출하는 ``ask`` 만 만족시키는 stub."""

    def __init__(self, result: Dict[str, Any]) -> None:
        self._result = result

    def ask(self, question: str, **_kwargs) -> Dict[str, Any]:  # noqa: ARG002
        return dict(self._result)


def _scenario_a_result() -> Dict[str, Any]:
    """
    시나리오 A: 질문 'meta', raw_fallback 이 kakao×2 + common×1.
    weak_evidence_warning=True, evidence_strength=weak 또는 insufficient.
    """
    r_kakao1 = _make_raw(
        chunk_id="r_kakao1",
        file_name="kakao1.txt",
        primary_topic="kakao",
        uploaded_category="kakao",
        source_type="kakao",
        content_type="conversation",
    )
    r_kakao1.metadata["retrieval_role"] = "raw_fallback"
    r_kakao1.metadata["topic_match"] = "mismatch"

    r_kakao2 = _make_raw(
        chunk_id="r_kakao2",
        file_name="kakao2.txt",
        primary_topic="kakao",
        uploaded_category="kakao",
        source_type="kakao",
        content_type="conversation",
    )
    r_kakao2.metadata["retrieval_role"] = "raw_fallback"
    r_kakao2.metadata["topic_match"] = "mismatch"

    r_common = _make_raw(
        chunk_id="r_common",
        file_name="common.txt",
        primary_topic="common",
        topic_tags=["common"],
        uploaded_category="guide",
        source_type="guide",
        content_type="text",
    )
    r_common.metadata["retrieval_role"] = "raw_fallback"
    r_common.metadata["topic_match"] = "neutral"

    return {
        "answer": "## 1. 결론\n메타 피드광고 셋팅 절차 요약.\n",
        "answer_mode": "raw_fallback",
        "answer_format_label": "default",
        "primary_normalized_documents": [],
        "primary_normalized_document_count": 0,
        "primary_cards": [],
        "primary_card_count": 0,
        "raw_evidence": [],
        "raw_evidence_count": 0,
        "raw_fallback": [r_kakao1, r_kakao2, r_common],
        "raw_fallback_count": 3,
        "generation_skipped": False,
        "skip_reason": None,
        "model_name": "fake-model",
        "embedding_provider": "fake",
        "embedding_model": "fake-model",
        "rewritten_query": None,
        # MVP 2차 Step 1
        "query_topic": "meta",
        "query_topics": ["meta"],
        "query_intent": ["procedure"],
        "query_date": None,
        "retrieved_count": 3,
        "passed_count": 3,
        "topic_mismatch_count": 2,
        "normalized_document_candidate_count": 0,
        "raw_candidate_count": 3,
        # MVP 2차 Step 2
        "topic_mismatch_demoted_count": 0,
        # MVP 2차 Step 4
        "raw_fallback_only": True,
        "raw_fallback_only_reason": "no_normalized_document_candidate",
        "raw_fallback_topic_mismatch_count": 2,
        "raw_fallback_topic_mismatch_ratio": 0.6666,
        "primary_evidence_available": False,
        "normalized_document_available": False,
        "weak_evidence_warning": True,
        "evidence_strength": "weak",
        "retrieval_summary": {
            "query_topic": "meta",
            "query_topics": ["meta"],
            "query_intent": ["procedure"],
            "retrieved_count": 3,
            "candidate_count": 3,
            "passed_count": 3,
            "topic_mismatch_count": 2,
            "topic_mismatch_demoted_count": 0,
            "normalized_document_candidate_count": 0,
            "raw_candidate_count": 3,
        },
    }


def test_adapter_forwards_step4_diagnostics_to_slack_diag():
    out = answer_slack_question(
        question="메타 피드광고 셋팅 방법 알려줘",
        pipeline=_StubPipeline(_scenario_a_result()),
    )
    diag = out["diagnostics"]
    # 진단 dict 에 Step 4 필드가 그대로 전달된다.
    assert diag["evidence_strength"] == "weak"
    assert diag["raw_fallback_only"] is True
    assert diag["raw_fallback_only_reason"] == "no_normalized_document_candidate"
    assert diag["raw_fallback_topic_mismatch_count"] == 2
    assert 0.6 < diag["raw_fallback_topic_mismatch_ratio"] < 0.7
    assert diag["weak_evidence_warning"] is True
    assert diag["primary_evidence_available"] is False
    assert diag["normalized_document_available"] is False


def test_slack_debug_shows_step4_diagnostics():
    out = answer_slack_question(
        question="메타 피드광고 셋팅 방법 알려줘",
        pipeline=_StubPipeline(_scenario_a_result()),
    )
    text = formatter.format_qa_result(out, debug=True)
    # debug 모드에서 신규 진단이 노출된다.
    assert "*진단 요약*" in text
    assert "evidence_strength" in text
    assert "`weak`" in text
    assert "raw_fallback_only" in text
    assert "weak_evidence_warning" in text
    # mismatch count 라인 노출 (count=2)
    assert "raw_fallback_topic_mismatch: 2 / 3" in text
    # 기존 Step1/2 진단도 깨지지 않는다.
    assert "query_topic" in text
    assert "`meta`" in text


def test_slack_default_output_hides_step4_diagnostics():
    out = answer_slack_question(
        question="메타 피드광고 셋팅 방법 알려줘",
        pipeline=_StubPipeline(_scenario_a_result()),
    )
    text = formatter.format_qa_result(out)
    # 기본 출력에는 Step 4 진단 라벨이 노출되지 않는다.
    assert "evidence_strength" not in text
    assert "raw_fallback_only" not in text
    assert "weak_evidence_warning" not in text
    assert "*진단 요약*" not in text


def test_slack_debug_omits_warning_when_not_active():
    """weak_evidence_warning=False / mismatch=0 인 경우 그 라인들이 노출되지 않는다."""
    base = _scenario_a_result()
    base["raw_fallback_only"] = False
    base["raw_fallback_only_reason"] = None
    base["raw_fallback_topic_mismatch_count"] = 0
    base["raw_fallback_topic_mismatch_ratio"] = 0.0
    base["weak_evidence_warning"] = False
    base["evidence_strength"] = "strong"
    base["primary_evidence_available"] = True
    base["normalized_document_available"] = True

    out = answer_slack_question(
        question="메타 피드광고 셋팅 방법 알려줘",
        pipeline=_StubPipeline(base),
    )
    text = formatter.format_qa_result(out, debug=True)
    # evidence_strength 와 weak_evidence_warning 은 요약에서 항상 노출되고,
    # raw_fallback_only / mismatch 상세 라인만 필요할 때 표시된다.
    assert "evidence_strength: `strong`" in text
    assert "weak_evidence_warning: `False`" in text
    assert "raw_fallback_only" not in text
    assert "raw_fallback_topic_mismatch: 0 / 3" in text


# ===========================================================================
# 6) 실제 문제 케이스 시나리오 A/B/C 방어
# ===========================================================================
def test_scenario_a_meta_question_kakao_raw_fallback_triggers_weak_warning(
    monkeypatch,
):
    """
    시나리오 A:
        질문: "메타 피드광고 셋팅 방법 알려줘"
        후보 (모두 raw chunk):
          - kakao raw 2개
          - common raw 1개
        기대:
          - raw_fallback_only=True
          - raw_fallback_topic_mismatch_count >= 2
          - weak_evidence_warning=True
          - evidence_strength in {"weak", "insufficient"}
    """
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    chunks = [
        _make_raw(
            chunk_id="r_kakao1",
            file_name="kakao1.txt",
            primary_topic="kakao",
            uploaded_category="kakao",
            source_type="kakao",
            content_type="conversation",
            score=0.62,
        ),
        _make_raw(
            chunk_id="r_kakao2",
            file_name="kakao2.txt",
            primary_topic="kakao",
            uploaded_category="kakao",
            source_type="kakao",
            content_type="conversation",
            score=0.58,
        ),
        _make_raw(
            chunk_id="r_common",
            file_name="common.txt",
            primary_topic="common",
            topic_tags=["common"],
            uploaded_category="guide",
            source_type="guide",
            content_type="text",
            score=0.55,
        ),
    ]
    pipeline, _gen = _make_pipeline(chunks)
    result = pipeline.ask(
        "메타 피드광고 셋팅 방법 알려줘", top_k=5, save_log=False
    )
    assert result["raw_fallback_only"] is True
    assert result["raw_fallback_topic_mismatch_count"] >= 2
    assert result["weak_evidence_warning"] is True
    assert result["evidence_strength"] in {"weak", "insufficient"}
    # Slack debug 에서도 확인 가능해야 한다.
    out = answer_slack_question(
        "메타 피드광고 셋팅 방법 알려줘",
        pipeline=_StubPipeline(result),
    )
    text = formatter.format_qa_result(out, debug=True)
    assert "evidence_strength" in text
    assert "weak_evidence_warning" in text
    assert "raw_fallback_only" in text


def test_scenario_b_meta_question_primary_meta_keeps_strong_evidence(monkeypatch):
    """
    시나리오 B:
        질문: "메타 피드광고 셋팅 방법 알려줘"
        후보:
          - primary normalized_document (primary_topic=meta)
          - raw_fallback (primary_topic=kakao)
        기대:
          - raw_fallback_only=False
          - evidence_strength=strong
          - weak_evidence_warning=False
          - primary Normalized Document 가 답변의 중심 (primary_normalized_document_count >= 1)
    """
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    chunks = [
        _make_nd(
            chunk_id="nd_meta",
            file_name="meta_workflow.md",
            primary_topic="meta",
            score=0.55,
        ),
        _make_raw(
            chunk_id="r_kakao",
            file_name="kakao_notice.txt",
            primary_topic="kakao",
            uploaded_category="kakao",
            source_type="kakao",
            content_type="conversation",
            score=0.60,
        ),
    ]
    pipeline, _gen = _make_pipeline(chunks)
    result = pipeline.ask(
        "메타 피드광고 셋팅 방법 알려줘", top_k=5, save_log=False
    )
    assert result["raw_fallback_only"] is False
    assert result["weak_evidence_warning"] is False
    assert result["evidence_strength"] == "strong"
    assert result["primary_normalized_document_count"] >= 1
    primary_ids = {c.chunk_id for c in result["primary_normalized_documents"]}
    assert "nd_meta" in primary_ids


def test_scenario_c_unclear_query_topic_with_raw_fallback_only_no_strong_warning(
    monkeypatch,
):
    """
    시나리오 C:
        질문 topic 이 명확하지 않음 (예: "그냥 운영 가이드 알려줘")
        raw_fallback 만 존재
        기대:
          - raw_fallback_only=True
          - weak_evidence_warning=False (mismatch 판단 불가)
          - evidence_strength != "weak" 강제로 잡지 않음 (weak 도 허용은 되나 정책상 보수적)
    """
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    chunks = [
        _make_raw(
            chunk_id="r_misc1",
            file_name="misc1.txt",
            primary_topic=None,
            uploaded_category="guide",
            source_type="guide",
            content_type="text",
            score=0.62,
        ),
        _make_raw(
            chunk_id="r_misc2",
            file_name="misc2.txt",
            primary_topic="kakao",  # 다양한 topic 이라도 query_topic 이 없으면 mismatch 잡지 않음
            uploaded_category="kakao",
            source_type="kakao",
            content_type="conversation",
            score=0.55,
        ),
    ]
    pipeline, _gen = _make_pipeline(chunks)
    result = pipeline.ask("그냥 운영 가이드 알려줘", top_k=5, save_log=False)

    assert result["query_topic"] is None
    assert result["raw_fallback_only"] is True
    # query_topic 이 없으니 mismatch 판단 불가 → warning 안 켜짐.
    assert result["weak_evidence_warning"] is False
    assert result["raw_fallback_topic_mismatch_count"] == 0
    # evidence_strength 는 "weak" 가능 (mismatch 0, query topic 없음).
    assert result["evidence_strength"] in {"weak", "medium"}


# ===========================================================================
# 7) Step 1/2/3 호환성
# ===========================================================================
def test_step1_diagnostics_still_exposed(monkeypatch):
    """Step 4 진단 추가가 Step 1 진단 필드를 덮어쓰지 않는다."""
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    chunks = [
        _make_nd(
            chunk_id="nd_meta",
            file_name="meta_workflow.md",
            primary_topic="meta",
            score=0.55,
        ),
        _make_raw(
            chunk_id="r_meta",
            file_name="meta_guide.txt",
            primary_topic="meta",
            uploaded_category="guide",
            source_type="guide",
            content_type="text",
            score=0.50,
        ),
    ]
    pipeline, _gen = _make_pipeline(chunks)
    result = pipeline.ask("메타 피드광고 셋팅 방법 알려줘", top_k=5, save_log=False)
    # Step 1 진단 그대로 노출
    assert result["query_topic"] == "meta"
    assert "retrieved_count" in result
    assert "passed_count" in result
    assert "topic_mismatch_count" in result
    assert "normalized_document_candidate_count" in result
    assert "raw_candidate_count" in result
    # Step 2 진단
    assert "topic_mismatch_demoted_count" in result
    # Step 4 진단 신규
    assert "evidence_strength" in result
    assert "raw_fallback_only" in result
    assert "weak_evidence_warning" in result


def test_step2_kakao_demoted_to_raw_fallback_can_still_trigger_weak_warning(
    monkeypatch,
):
    """Step 2 에서 격하된 normalized document 는 raw_fallback 으로 분류되고,
    이 상태에서 query_topic 명확 + mismatch 다수면 weak_evidence_warning 가 켜진다."""
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    chunks = [
        _make_nd(
            chunk_id="nd_kakao1",
            file_name="kakao_notice1.md",
            primary_topic="kakao",
            score=0.70,
        ),
        _make_nd(
            chunk_id="nd_kakao2",
            file_name="kakao_notice2.md",
            primary_topic="kakao",
            score=0.65,
        ),
    ]
    pipeline, _gen = _make_pipeline(chunks)
    result = pipeline.ask(
        "메타 피드광고 셋팅 방법 알려줘", top_k=5, save_log=False
    )
    # primary 가 없고 raw_fallback (격하된 normalized document) 만 존재.
    assert result["primary_normalized_document_count"] == 0
    assert result["raw_fallback_count"] >= 2
    assert result["raw_fallback_only"] is True
    # 격하된 normalized document 가 모두 kakao topic 이므로 mismatch ratio 1.0.
    assert result["raw_fallback_topic_mismatch_count"] >= 2
    assert result["weak_evidence_warning"] is True
    # Step 2 의 topic_mismatch_demoted_count 도 유지.
    assert result["topic_mismatch_demoted_count"] >= 2
