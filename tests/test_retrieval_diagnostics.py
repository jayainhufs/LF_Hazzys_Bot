"""
test_retrieval_diagnostics.py
=============================
MVP 2차 Step 1 — Retrieval Diagnostics 강화 단위 테스트.

이 단계의 목표는 검색 정책을 바꾸는 것이 아니라, 검색 결과가 왜 그렇게
나왔는지 외부에서 더 잘 보이도록 ``retrieval_summary`` / ``qa_pipeline``
return dict / Slack ``--debug`` 출력 / Slack adapter sources 에 진단
필드를 추가하는 것이다.

검증 항목:
1. ``Retriever.retrieve_with_details`` summary 에 신규 진단 필드가 채워진다.
2. ``QAPipeline.ask`` 반환 dict 의 top-level 에 진단 필드가 노출된다.
3. ``QAPipeline.ask`` empty question 케이스에서도 default 진단 필드가 있다.
4. ``qa_adapter.answer_slack_question`` diagnostics 에 진단 필드가 들어간다.
5. ``qa_adapter`` 가 source dict 에 retrieval_role / topic_match 등을 채운다.
6. Slack ``formatter.format_qa_result(debug=True)`` 출력에 ``query_topic`` 과
   top source 의 진단 라인이 포함된다.
7. Slack 기본 출력 (``debug=False``) 에는 진단 라벨이 노출되지 않는다.

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
from src.slack_bot import formatter, qa_adapter


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


def _make_kc_chunk(
    *,
    chunk_id: str,
    file_name: str,
    score: float = 0.55,
    card_type: str = "workflow",
    primary_topic: str = "meta",
    topic_tags: Optional[List[str]] = None,
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
        "topic_tags": topic_tags or [primary_topic],
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
        content=f"[normalized document content for {chunk_id}]",
        score=float(score),
        final_score=float(score),
        metadata=md,
    )


def _make_raw_chunk(
    *,
    chunk_id: str,
    file_name: str,
    score: float = 0.6,
    uploaded_category: str = "guide",
    source_type: str = "guide",
    content_type: str = "text",
    primary_topic: Optional[str] = None,
    topic_tags: Optional[List[str]] = None,
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
        md["topic_tags"] = topic_tags
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


def _make_pipeline(chunks: List[RetrievedChunk]) -> QAPipeline:
    return QAPipeline(
        retriever=_make_retriever(chunks),
        generator=FakeGenerator(),  # type: ignore[arg-type]
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# 1. Retriever.retrieve_with_details summary 진단 필드
# ---------------------------------------------------------------------------
def test_retriever_summary_includes_step1_diagnostic_fields(monkeypatch):
    """summary 에 retrieved_count, query_topic, candidate 구성 카운트가 채워진다."""
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    chunks = [
        _make_kc_chunk(
            chunk_id="kc_meta",
            file_name="meta_card.txt",
            primary_topic="meta",
            topic_tags=["meta"],
            score=0.55,
        ),
        _make_raw_chunk(
            chunk_id="raw_meta",
            file_name="meta_card.txt",
            primary_topic="meta",
            topic_tags=["meta"],
            score=0.50,
        ),
        # 명백히 잘못 매칭된 kakao chunk (topic mismatch 케이스)
        _make_raw_chunk(
            chunk_id="raw_kakao",
            file_name="kakao_notice.txt",
            uploaded_category="kakao",
            source_type="kakao",
            content_type="conversation",
            primary_topic="kakao",
            topic_tags=["kakao"],
            score=0.45,
        ),
    ]
    r = _make_retriever(chunks)
    details = r.retrieve_with_details(
        "메타 피드광고 셋팅 방법 알려줘", top_k=5, max_per_file=10
    )
    summary = details.summary

    # query 진단
    assert summary.get("query_topic") == "meta"
    assert "meta" in (summary.get("query_topics") or [])

    # candidate / passed 카운트
    assert summary.get("retrieved_count") == len(details.candidates)
    assert summary.get("retrieved_count") == summary.get("candidate_count")
    assert summary.get("passed_count") == len(details.passed)

    # candidate 구성 카운트
    assert summary.get("normalized_document_candidate_count") >= 1
    assert summary.get("raw_candidate_count") >= 2

    # topic mismatch 카운트 — kakao chunk 가 mismatch 로 잡혀야 한다.
    assert summary.get("topic_mismatch_count") >= 1


def test_retriever_summary_query_topic_none_when_no_topic_keyword():
    """질문에 topic keyword 가 없으면 query_topic 은 None."""
    chunks = [_make_raw_chunk(chunk_id="r1", file_name="x.txt", score=0.6)]
    r = _make_retriever(chunks)
    details = r.retrieve_with_details("뭔가 알려줘", top_k=3, max_per_file=5)
    assert details.summary.get("query_topic") is None
    assert details.summary.get("query_topics") == []


# ---------------------------------------------------------------------------
# 2. QAPipeline.ask 반환 dict 의 top-level 진단 필드
# ---------------------------------------------------------------------------
def test_qa_pipeline_exposes_diagnostics_at_top_level(monkeypatch):
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    chunks = [
        _make_kc_chunk(
            chunk_id="kc_a",
            file_name="meta_guide.txt",
            primary_topic="meta",
            topic_tags=["meta"],
            score=0.55,
        ),
        _make_raw_chunk(
            chunk_id="raw_a",
            file_name="meta_guide.txt",
            primary_topic="meta",
            topic_tags=["meta"],
            score=0.50,
        ),
    ]
    pipeline = _make_pipeline(chunks)
    result = pipeline.ask("메타 피드광고 셋팅 방법 알려줘", top_k=5, save_log=False)

    # 신규 진단 필드 (MVP 2차 Step 1)
    assert result["query_topic"] == "meta"
    assert "meta" in result["query_topics"]
    assert isinstance(result["query_intent"], list)
    assert "retrieved_count" in result
    assert "passed_count" in result
    assert "topic_mismatch_count" in result
    assert "normalized_document_candidate_count" in result
    assert "raw_candidate_count" in result
    # 검색 정책을 바꾸지 않았으므로 카운트는 일관되어야 함.
    assert result["retrieved_count"] >= result["passed_count"]
    assert result["normalized_document_candidate_count"] >= 1


def test_qa_pipeline_empty_question_returns_default_diagnostics():
    pipeline = _make_pipeline([])
    result = pipeline.ask("   ", save_log=False)
    assert result["generation_skipped"] is True
    # MVP 2차 Step 1: empty 케이스에도 진단 필드 default 가 있어야 한다.
    assert result["query_topic"] is None
    assert result["query_topics"] == []
    assert result["query_intent"] == []
    assert result["query_date"] is None
    assert result["retrieved_count"] == 0
    assert result["passed_count"] == 0
    assert result["topic_mismatch_count"] == 0
    assert result["normalized_document_candidate_count"] == 0
    assert result["raw_candidate_count"] == 0


# ---------------------------------------------------------------------------
# 3. qa_adapter.answer_slack_question diagnostics
# ---------------------------------------------------------------------------
class _FakePipeline:
    """Slack adapter 통합 테스트용 fake pipeline."""

    def __init__(self, ask_result: Dict[str, Any]) -> None:
        self._ask_result = ask_result
        self.last_question: Optional[str] = None

    def ask(self, question: str, *_args, **_kwargs) -> Dict[str, Any]:
        self.last_question = question
        return self._ask_result


def _fake_pipeline_result(
    *,
    primary_chunks: List[RetrievedChunk],
    raw_evidence: List[RetrievedChunk],
    raw_fallback: List[RetrievedChunk],
    query_topic: str,
    retrieved_count: int,
    passed_count: int,
    topic_mismatch_count: int,
) -> Dict[str, Any]:
    return {
        "answer": "[fake answer body]",
        "answer_mode": "knowledge_card",
        "primary_normalized_documents": primary_chunks,
        "primary_normalized_document_count": len(primary_chunks),
        "raw_evidence": raw_evidence,
        "raw_evidence_count": len(raw_evidence),
        "raw_fallback": raw_fallback,
        "raw_fallback_count": len(raw_fallback),
        "primary_cards": primary_chunks,
        "primary_card_count": len(primary_chunks),
        "generation_skipped": False,
        "skip_reason": None,
        "model_name": "fake-model",
        "embedding_provider": "fake",
        "embedding_model": "fake-model",
        "rewritten_query": None,
        "answer_format_label": "default",
        # MVP 2차 Step 1 진단 필드
        "query_topic": query_topic,
        "query_topics": [query_topic],
        "query_intent": ["procedure"],
        "query_date": None,
        "retrieved_count": retrieved_count,
        "passed_count": passed_count,
        "topic_mismatch_count": topic_mismatch_count,
        "normalized_document_candidate_count": len(primary_chunks),
        "raw_candidate_count": len(raw_evidence) + len(raw_fallback),
        "retrieval_summary": {
            "query_topic": query_topic,
            "query_topics": [query_topic],
            "query_intent": ["procedure"],
            "candidate_count": retrieved_count,
            "retrieved_count": retrieved_count,
            "passed_count": passed_count,
            "topic_mismatch_count": topic_mismatch_count,
        },
    }


def test_adapter_diagnostics_contains_step1_fields():
    primary = [
        _make_kc_chunk(
            chunk_id="kc_meta", file_name="meta_card.txt", primary_topic="meta"
        )
    ]
    raw_ev = [
        _make_raw_chunk(
            chunk_id="raw_meta", file_name="meta_card.txt", primary_topic="meta"
        )
    ]
    raw_fb = [
        _make_raw_chunk(
            chunk_id="raw_kakao",
            file_name="kakao_notice.txt",
            uploaded_category="kakao",
            source_type="kakao",
            content_type="conversation",
            primary_topic="kakao",
            topic_tags=["kakao"],
        )
    ]
    # rerank 가 raw_kakao 에 topic_match=mismatch 를 채웠다고 가정.
    raw_fb[0].metadata["topic_match"] = "mismatch"
    raw_fb[0].metadata["retrieval_role"] = "raw_fallback"
    primary[0].metadata["retrieval_role"] = "primary_card"
    raw_ev[0].metadata["retrieval_role"] = "raw_evidence"

    ask_result = _fake_pipeline_result(
        primary_chunks=primary,
        raw_evidence=raw_ev,
        raw_fallback=raw_fb,
        query_topic="meta",
        retrieved_count=3,
        passed_count=3,
        topic_mismatch_count=1,
    )
    fake_pipe = _FakePipeline(ask_result)
    out = qa_adapter.answer_slack_question(
        "메타 피드광고 셋팅 방법 알려줘", pipeline=fake_pipe
    )

    diag = out["diagnostics"]
    assert diag["query_topic"] == "meta"
    assert diag["retrieved_count"] == 3
    assert diag["passed_count"] == 3
    assert diag["topic_mismatch_count"] == 1
    assert diag["normalized_document_candidate_count"] == 1
    assert diag["raw_candidate_count"] == 2

    # sources 에 진단 필드 (retrieval_role / content_type / primary_topic / topic_match) 포함
    src = out["sources"]
    p0 = src["primary_normalized_documents"][0]
    assert p0["retrieval_role"] == "primary_card"
    assert p0["primary_topic"] == "meta"
    assert p0["file_name"] == "meta_card.txt"
    assert "content_type" in p0
    assert "final_score" in p0
    e0 = src["raw_evidence"][0]
    assert e0["retrieval_role"] == "raw_evidence"
    f0 = src["raw_fallback"][0]
    assert f0["topic_match"] == "mismatch"


# ---------------------------------------------------------------------------
# 4. Slack formatter --debug 출력에 진단 / source 진단 라인 포함
# ---------------------------------------------------------------------------
def _slack_result_for_formatter() -> Dict[str, Any]:
    """formatter 단위 테스트용 dict (qa_adapter 의 반환 shape 와 일치)."""
    return {
        "answer": "## 1. 결론\n메타 피드광고 셋팅 절차 요약.\n",
        "answer_mode": "knowledge_card",
        "primary_normalized_document_count": 1,
        "raw_evidence_count": 1,
        "raw_fallback_count": 1,
        "sources": {
            "primary_normalized_documents": [
                {
                    "label": "meta_guide.md · 셋팅 절차",
                    "preview": "1) 광고주 계정 확인 2) 캠페인 생성",
                    "file_name": "meta_guide.md",
                    "content_type": "knowledge_card",
                    "primary_topic": "meta",
                    "retrieval_role": "primary_card",
                    "final_score": 0.812,
                    "topic_match": "match",
                }
            ],
            "raw_evidence": [
                {
                    "label": "meta_guide.md · 화면 캡처 메모",
                    "preview": "실제 운영 화면 ...",
                    "file_name": "meta_guide.md",
                    "content_type": "text",
                    "primary_topic": "meta",
                    "retrieval_role": "raw_evidence",
                    "final_score": 0.65,
                }
            ],
            "raw_fallback": [
                {
                    "label": "kakao_notice.txt · 카카오 메시지 주의사항",
                    "preview": "카카오 메시지 발송 시 ...",
                    "file_name": "kakao_notice.txt",
                    "content_type": "conversation",
                    "primary_topic": "kakao",
                    "retrieval_role": "raw_fallback",
                    "final_score": 0.42,
                    "topic_match": "mismatch",
                }
            ],
        },
        "diagnostics": {
            "answer_mode": "knowledge_card",
            "primary_normalized_document_count": 1,
            "raw_evidence_count": 1,
            "raw_fallback_count": 1,
            "generation_skipped": False,
            "model_name": "fake-model",
            "query_topic": "meta",
            "query_intent": ["procedure"],
            "query_date": None,
            "retrieved_count": 8,
            "passed_count": 3,
            "topic_mismatch_count": 1,
            "normalized_document_candidate_count": 2,
            "raw_candidate_count": 6,
        },
    }


def test_formatter_debug_includes_query_topic_and_top_sources():
    text = formatter.format_qa_result(_slack_result_for_formatter(), debug=True)

    # 답변 본문은 정상적으로 표시되어야 한다.
    assert "*1. 결론*" in text

    # 진단 블록에 신규 필드들이 노출되어야 한다.
    assert "*진단*" in text
    assert "query_topic" in text
    assert "`meta`" in text
    assert "retrieved_count: 8" in text
    assert "passed_count: 3" in text
    assert "topic_mismatch_count: 1" in text
    assert "normalized_document=2" in text or "normalized_document_candidate" in text
    assert "raw=6" in text or "raw_candidate" in text

    # source 진단 라인 — top3 정도까지 표시되며 file_name / content_type /
    # primary_topic / retrieval_role / final_score 가 등장한다.
    assert "참고 근거 (debug)" in text
    assert "meta_guide.md" in text
    assert "primary_card" in text
    # final_score 가 소수점 3자리로 표시.
    assert "final=`0.812`" in text


def test_formatter_default_output_has_no_diagnostics_labels():
    """기본 출력 (debug=False) 에는 새 진단 라벨이 노출되지 않는다."""
    text = formatter.format_qa_result(_slack_result_for_formatter())
    # 답변 본문은 보인다.
    assert "*1. 결론*" in text
    # 진단 / source 진단 라인이 노출되면 안 된다.
    assert "*진단*" not in text
    assert "query_topic" not in text
    assert "retrieved_count" not in text
    assert "topic_mismatch_count" not in text
    assert "참고 근거 (debug)" not in text
    # source 진단 라인에서 사용한 라벨도 들어가면 안 된다.
    assert "primary_card" not in text
    assert "final=`" not in text


# ---------------------------------------------------------------------------
# 5. Slack adapter empty case — 진단 default 값
# ---------------------------------------------------------------------------
def test_adapter_handles_pipeline_without_step1_fields_gracefully():
    """qa_pipeline 이 step1 필드를 채우지 않은 옛 응답을 그대로 받아도 깨지지 않는다."""
    legacy_result = {
        "answer": "(answer)",
        "answer_mode": "knowledge_card",
        "primary_normalized_documents": [],
        "primary_normalized_document_count": 0,
        "raw_evidence": [],
        "raw_evidence_count": 0,
        "raw_fallback": [],
        "raw_fallback_count": 0,
        "generation_skipped": False,
        "skip_reason": None,
        "model_name": "fake-model",
        # 의도적으로 query_topic / retrieved_count / topic_mismatch_count 미포함
    }
    fake_pipe = _FakePipeline(legacy_result)
    out = qa_adapter.answer_slack_question("질문", pipeline=fake_pipe)
    diag = out["diagnostics"]
    # 누락된 필드는 안전한 default (None / 0) 로 채워져야 한다.
    assert diag["query_topic"] is None
    assert diag["retrieved_count"] == 0
    assert diag["passed_count"] == 0
    assert diag["topic_mismatch_count"] == 0
    assert diag["normalized_document_candidate_count"] == 0
    assert diag["raw_candidate_count"] == 0
