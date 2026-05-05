"""
test_knowledge_card_qa.py
=========================
Task 7: KnowledgeCard 중심 QA prompt / qa_pipeline answer_mode 단위 테스트.

외부 Gemini API 호출 없이 fake generator/embedder/vector store 로 검증한다.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from src.config import settings
from src.rag.prompt_builder import (
    COMMUNICATION_ANSWER_FORMAT,
    DEFAULT_ANSWER_FORMAT,
    GLOSSARY_ANSWER_FORMAT,
    KNOWLEDGE_CARD_PRINCIPLES,
    KNOWLEDGE_CARD_SYSTEM_INSTRUCTION,
    build_knowledge_card_answer_prompt,
    build_qa_prompt,
    select_answer_format,
    split_chunks_by_retrieval_role,
)
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
    """Gemini 호출 대신 마지막 prompt 와 호출 횟수를 기록한다."""

    def __init__(self, answer: str = "[fake answer]") -> None:
        self.calls = 0
        self.last_prompt: str = ""
        self._answer = answer

    def generate(self, prompt: str, **_kwargs):  # noqa: D401
        self.calls += 1
        self.last_prompt = prompt
        return (self._answer, "fake-model")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_kc_chunk(
    *,
    chunk_id: str,
    file_name: str,
    card_type: str,
    primary_topic: str = "common",
    task_type: str = "unknown",
    score: float = 0.6,
    title: Optional[str] = None,
    parent_raw_chunk_ids: Optional[List[str]] = None,
    uploaded_category: str = "guide",
    sanitized_content: Optional[str] = None,
) -> RetrievedChunk:
    md: Dict[str, Any] = {
        "source_weight": float(settings.normalization_card_source_weight),
        "uploaded_category": uploaded_category,
        "file_name": file_name,
        "source_type": "llm_normalized",
        "content_type": "knowledge_card",
        "section_title": None,
        "card_id": chunk_id,
        "card_type": card_type,
        "primary_topic": primary_topic,
        "task_type": task_type,
        "topic_tags": [primary_topic] if primary_topic else [],
        "parent_raw_chunk_ids": list(parent_raw_chunk_ids or []),
        "title": title,
        "retrieval_role": "primary_card",
        "knowledge_card_boost": float(settings.knowledge_card_content_boost),
        "card_type_boost": 1.30,
        "card_type_match": True,
    }
    if sanitized_content:
        md["sanitized_content"] = sanitized_content
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc_{file_name}",
        file_name=file_name,
        source_type="llm_normalized",
        uploaded_category=uploaded_category,
        section_title=None,
        content_type="knowledge_card",
        content=sanitized_content
        or f"# {title or chunk_id}\n\n{card_type} card body for {chunk_id}",
        score=float(score),
        final_score=float(score) * 1.5,
        metadata=md,
    )


def _make_raw_chunk(
    *,
    chunk_id: str,
    file_name: str,
    uploaded_category: str = "guide",
    source_type: str = "guide",
    content_type: str = "text",
    score: float = 0.6,
    section_title: Optional[str] = None,
    chunk_index: int = 0,
    sanitized_content: Optional[str] = None,
    role: str = "raw_evidence",
) -> RetrievedChunk:
    md: Dict[str, Any] = {
        "source_weight": float(
            settings.category_source_weight.get(source_type, 0.7)
        ),
        "chunk_index": chunk_index,
        "uploaded_category": uploaded_category,
        "file_name": file_name,
        "source_type": source_type,
        "content_type": content_type,
        "section_title": section_title,
        "retrieval_role": role,
        "knowledge_card_boost": 1.0,
        "card_type_boost": 1.0,
        "card_type_match": False,
    }
    if sanitized_content:
        md["sanitized_content"] = sanitized_content
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc_{file_name}",
        file_name=file_name,
        source_type=source_type,
        uploaded_category=uploaded_category,
        section_title=section_title,
        content_type=content_type,
        content=sanitized_content or f"raw content {chunk_index} from {file_name}",
        score=float(score),
        final_score=float(score),
        metadata=md,
    )


def _make_pipeline(chunks: List[RetrievedChunk]) -> tuple:
    fake_gen = FakeGenerator()
    pipeline = QAPipeline(
        retriever=Retriever(
            embedder=FakeEmbedder(), vector_store=FakeVectorStore(chunks)
        ),
        generator=fake_gen,  # type: ignore[arg-type]
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
    )
    return pipeline, fake_gen


# ---------------------------------------------------------------------------
# 1. split_chunks_by_retrieval_role
# ---------------------------------------------------------------------------
def test_split_chunks_by_retrieval_role_separates_three_groups():
    cards = [
        _make_kc_chunk(
            chunk_id="kc_a", file_name="a_card.txt", card_type="workflow"
        )
    ]
    raw_ev = [
        _make_raw_chunk(
            chunk_id="r_e", file_name="x.txt", role="raw_evidence", chunk_index=0
        )
    ]
    raw_fb = [
        _make_raw_chunk(
            chunk_id="r_f", file_name="y.txt", role="raw_fallback", chunk_index=0
        )
    ]
    groups = split_chunks_by_retrieval_role(cards + raw_ev + raw_fb)
    assert [c.chunk_id for c in groups["primary_cards"]] == ["kc_a"]
    assert [c.chunk_id for c in groups["raw_evidence"]] == ["r_e"]
    assert [c.chunk_id for c in groups["raw_fallback"]] == ["r_f"]


def test_split_chunks_recognizes_card_via_content_type_only():
    """retrieval_role 이 비어 있어도 content_type=knowledge_card 면 primary_card."""
    chunk = _make_kc_chunk(
        chunk_id="kc_b", file_name="b.txt", card_type="checklist"
    )
    chunk.metadata.pop("retrieval_role", None)
    groups = split_chunks_by_retrieval_role([chunk])
    assert len(groups["primary_cards"]) == 1


def test_split_chunks_recognizes_card_via_source_type_only():
    """source_type=llm_normalized 만 있어도 primary_card."""
    chunk = _make_raw_chunk(
        chunk_id="kc_c",
        file_name="c.txt",
        uploaded_category="guide",
        source_type="llm_normalized",
        content_type="text",
    )
    chunk.metadata.pop("retrieval_role", None)
    groups = split_chunks_by_retrieval_role([chunk])
    assert len(groups["primary_cards"]) == 1


def test_split_chunks_handles_empty_list():
    groups = split_chunks_by_retrieval_role([])
    assert groups == {"primary_cards": [], "raw_evidence": [], "raw_fallback": []}


# ---------------------------------------------------------------------------
# 2. select_answer_format
# ---------------------------------------------------------------------------
def test_select_answer_format_glossary_for_glossary_card():
    cards = [_make_kc_chunk(chunk_id="kc1", file_name="g.txt", card_type="glossary")]
    fmt, label = select_answer_format(cards, "ASC 용어 무슨 뜻이야?")
    assert label == "glossary"
    assert fmt == GLOSSARY_ANSWER_FORMAT


def test_select_answer_format_communication_for_template_card():
    cards = [
        _make_kc_chunk(
            chunk_id="kc1", file_name="c.txt", card_type="communication_template"
        )
    ]
    fmt, label = select_answer_format(cards, "광고주 공유 문안 작성해줘")
    assert label == "communication_template"
    assert fmt == COMMUNICATION_ANSWER_FORMAT


def test_select_answer_format_default_for_workflow_card():
    cards = [_make_kc_chunk(chunk_id="kc1", file_name="w.txt", card_type="workflow")]
    fmt, label = select_answer_format(cards, "메타 셋팅 가이드 알려줘")
    assert label == "default"
    assert fmt == DEFAULT_ANSWER_FORMAT


def test_select_answer_format_keyword_overrides_when_no_card_type():
    """질문에 '용어' 키워드만 있어도 glossary 형식 선택."""
    cards = [_make_kc_chunk(chunk_id="kc1", file_name="x.txt", card_type="workflow")]
    fmt, label = select_answer_format(cards, "BAU 용어 정의 알려줘")
    assert label == "glossary"
    assert fmt == GLOSSARY_ANSWER_FORMAT


# ---------------------------------------------------------------------------
# 3. build_knowledge_card_answer_prompt — 구조 검증
# ---------------------------------------------------------------------------
def test_prompt_has_primary_card_section_before_raw_evidence():
    cards = [
        _make_kc_chunk(
            chunk_id="kc_wf",
            file_name="meta_card.txt",
            card_type="workflow",
            sanitized_content="# 메타 캠페인 셋팅\n절차: ...",
        ),
    ]
    raw = [
        _make_raw_chunk(
            chunk_id="raw_a",
            file_name="meta_card.txt",
            role="raw_evidence",
            sanitized_content="실제 화면에서는 ...",
        ),
    ]
    prompt, used, label = build_knowledge_card_answer_prompt(
        question="메타 캠페인 셋팅 가이드 알려줘",
        chunks=cards + raw,
    )
    assert label == "default"
    # 섹션 헤더 (markdown ##) 기준으로 위치 비교
    assert "## 주 근거 (KnowledgeCard)" in prompt
    assert "## 보조 근거 (Raw Evidence)" in prompt
    primary_idx = prompt.index("## 주 근거 (KnowledgeCard)")
    evidence_idx = prompt.index("## 보조 근거 (Raw Evidence)")
    assert primary_idx < evidence_idx, (
        "주 근거 섹션이 보조 근거 섹션보다 먼저 등장해야 한다"
    )
    # used 에는 card 와 raw evidence 가 모두 들어간다.
    chunk_ids = {c.chunk_id for c in used}
    assert "kc_wf" in chunk_ids
    assert "raw_a" in chunk_ids


def test_prompt_includes_anonymization_principles():
    cards = [_make_kc_chunk(chunk_id="kc1", file_name="x.txt", card_type="workflow")]
    prompt, _used, _label = build_knowledge_card_answer_prompt(
        question="셋팅 가이드", chunks=cards
    )
    assert KNOWLEDGE_CARD_SYSTEM_INSTRUCTION.split(".")[0] in prompt
    assert "사람 실명" in prompt
    assert "@멘션" in prompt
    assert "정확한 시간" in prompt
    assert "역할 표현" in prompt or "역할" in prompt


def test_prompt_uses_default_format_for_workflow_card():
    cards = [
        _make_kc_chunk(chunk_id="kc_wf", file_name="g.txt", card_type="workflow")
    ]
    prompt, _used, label = build_knowledge_card_answer_prompt(
        question="메타 셋팅 가이드", chunks=cards
    )
    assert label == "default"
    assert "업무 처리 순서 또는 핵심 체크포인트" in prompt
    assert "체크리스트" in prompt
    assert COMMUNICATION_ANSWER_FORMAT not in prompt


def test_prompt_uses_communication_format_for_template_card():
    cards = [
        _make_kc_chunk(
            chunk_id="kc_ct",
            file_name="share.txt",
            card_type="communication_template",
        )
    ]
    prompt, _used, label = build_knowledge_card_answer_prompt(
        question="광고주 공유 문안 작성해줘", chunks=cards
    )
    assert label == "communication_template"
    assert "바로 사용할 수 있는 초안" in prompt
    assert "문안 작성 포인트" in prompt


def test_prompt_uses_glossary_format_for_glossary_card():
    cards = [
        _make_kc_chunk(chunk_id="kc_g", file_name="terms.txt", card_type="glossary")
    ]
    prompt, _used, label = build_knowledge_card_answer_prompt(
        question="ASC 용어 정의", chunks=cards
    )
    assert label == "glossary"
    assert "용어 정의" in prompt
    assert "실무에서의 의미" in prompt


def test_prompt_falls_back_to_raw_when_no_primary_card():
    raw = [
        _make_raw_chunk(
            chunk_id="r1", file_name="g.txt", role="raw_fallback", chunk_index=0
        ),
    ]
    prompt, used, label = build_knowledge_card_answer_prompt(
        question="가이드 알려줘", chunks=raw
    )
    assert label == "raw_fallback"
    assert "## 주 근거 (KnowledgeCard)" not in prompt
    assert any(c.chunk_id == "r1" for c in used)


def test_prompt_omits_raw_appendix_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "include_raw_evidence_appendix", False)
    cards = [_make_kc_chunk(chunk_id="kc1", file_name="x.txt", card_type="workflow")]
    raw = [
        _make_raw_chunk(
            chunk_id="r1",
            file_name="x.txt",
            role="raw_evidence",
            sanitized_content="raw evidence body",
        )
    ]
    prompt, used, label = build_knowledge_card_answer_prompt(
        question="셋팅 가이드", chunks=cards + raw
    )
    assert label == "default"
    # 섹션 헤더 형태로만 검증 (system instruction 본문에는 인용 문자열이 등장할 수 있음)
    assert "## 보조 근거 (Raw Evidence)" not in prompt
    assert "## Raw Fallback" not in prompt
    # raw evidence 본문도 prompt 에 포함되면 안 됨
    assert "raw evidence body" not in prompt
    # used 에는 raw evidence 가 포함되지 않는다.
    used_ids = {c.chunk_id for c in used}
    assert "r1" not in used_ids


def test_prompt_template_version_label_is_present():
    cards = [_make_kc_chunk(chunk_id="kc1", file_name="x.txt", card_type="workflow")]
    prompt, _used, _label = build_knowledge_card_answer_prompt(
        question="셋팅 가이드", chunks=cards
    )
    assert settings.knowledge_card_answer_template_version in prompt


def test_prompt_preserves_korean_content_in_card_block():
    cards = [
        _make_kc_chunk(
            chunk_id="kc_k",
            file_name="한글.txt",
            card_type="workflow",
            title="메타 캠페인 셋팅 가이드",
            sanitized_content="## 절차\n1) 광고주 확인\n2) 캠페인 생성",
        )
    ]
    prompt, _used, _label = build_knowledge_card_answer_prompt(
        question="메타 셋팅", chunks=cards
    )
    assert "메타 캠페인 셋팅 가이드" in prompt
    assert "광고주 확인" in prompt


def test_prompt_principles_block_is_included():
    cards = [_make_kc_chunk(chunk_id="kc1", file_name="x.txt", card_type="workflow")]
    prompt, _used, _label = build_knowledge_card_answer_prompt(
        question="셋팅 가이드", chunks=cards
    )
    # 핵심 원칙 문장 일부가 prompt 에 포함되는지 검증
    assert "주 근거" in KNOWLEDGE_CARD_PRINCIPLES
    assert "근거 간 차이가 있습니다" in prompt


# ---------------------------------------------------------------------------
# 4. build_qa_prompt 라우팅
# ---------------------------------------------------------------------------
def test_build_qa_prompt_routes_to_knowledge_card_when_primary_exists():
    cards = [_make_kc_chunk(chunk_id="kc1", file_name="x.txt", card_type="workflow")]
    prompt, _used = build_qa_prompt("셋팅 가이드", cards)
    assert "## 주 근거 (KnowledgeCard)" in prompt


def test_build_qa_prompt_uses_legacy_when_no_primary_card():
    raw = [
        _make_raw_chunk(
            chunk_id="r1", file_name="g.txt", role="raw_fallback", chunk_index=0
        )
    ]
    prompt, _used = build_qa_prompt("정산 가이드", raw)
    assert "## 주 근거 (KnowledgeCard)" not in prompt
    # 기존 7섹션 레거시 prompt 는 "## 6. 참고 근거" 가 포함된다.
    assert "## 6. 참고 근거" in prompt


def test_build_qa_prompt_disabled_uses_legacy_even_with_card(monkeypatch):
    monkeypatch.setattr(settings, "answer_with_knowledge_cards", False)
    cards = [_make_kc_chunk(chunk_id="kc1", file_name="x.txt", card_type="workflow")]
    prompt, _used = build_qa_prompt("셋팅 가이드", cards)
    assert "## 주 근거 (KnowledgeCard)" not in prompt


# ---------------------------------------------------------------------------
# 5. QAPipeline answer_mode
# ---------------------------------------------------------------------------
def test_pipeline_answer_mode_knowledge_card_when_primary_card_present(monkeypatch):
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    chunks = [
        _make_kc_chunk(
            chunk_id="kc_wf",
            file_name="card_a.txt",
            card_type="workflow",
            score=0.55,
        ),
        _make_raw_chunk(
            chunk_id="raw_a",
            file_name="card_a.txt",
            role="raw_evidence",
            score=0.50,
        ),
    ]
    pipeline, fake_gen = _make_pipeline(chunks)
    result = pipeline.ask("메타 셋팅 가이드 알려줘", top_k=5, save_log=False)

    assert result["generation_skipped"] is False
    assert fake_gen.calls == 1
    assert result["answer_mode"] == "knowledge_card"
    assert result["primary_card_count"] >= 1
    assert "## 주 근거 (KnowledgeCard)" in fake_gen.last_prompt
    # template version 로그
    assert (
        result["knowledge_card_answer_template_version"]
        == settings.knowledge_card_answer_template_version
    )


def test_pipeline_answer_mode_raw_fallback_when_no_primary_card(monkeypatch):
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    chunks = [
        _make_raw_chunk(
            chunk_id="raw_only",
            file_name="g.txt",
            role="raw_fallback",
            score=0.70,
        ),
    ]
    pipeline, fake_gen = _make_pipeline(chunks)
    result = pipeline.ask("정산 프로세스 알려줘", top_k=5, save_log=False)

    assert result["generation_skipped"] is False
    assert fake_gen.calls == 1
    assert result["answer_mode"] == "raw_fallback"
    assert result["primary_card_count"] == 0
    assert result["raw_fallback_count"] >= 1
    assert "## 주 근거 (KnowledgeCard)" not in fake_gen.last_prompt


def test_pipeline_answer_mode_insufficient_evidence_when_no_chunks_pass():
    chunks = [
        _make_raw_chunk(
            chunk_id="too_low",
            file_name="g.txt",
            role="raw_fallback",
            score=0.05,  # below threshold
        ),
    ]
    pipeline, fake_gen = _make_pipeline(chunks)
    result = pipeline.ask("정산 프로세스 알려줘", top_k=5, save_log=False)

    assert result["generation_skipped"] is True
    assert result["answer_mode"] == "insufficient_evidence"
    assert fake_gen.calls == 0


def test_pipeline_returns_diagnostic_counts(monkeypatch):
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    chunks = [
        _make_kc_chunk(
            chunk_id="kc_a", file_name="card_a.txt", card_type="workflow", score=0.55
        ),
        _make_raw_chunk(
            chunk_id="raw_a", file_name="card_a.txt", role="raw_evidence", score=0.50
        ),
        _make_raw_chunk(
            chunk_id="raw_b", file_name="other.txt", role="raw_fallback", score=0.45
        ),
    ]
    pipeline, _gen = _make_pipeline(chunks)
    result = pipeline.ask("셋팅 가이드", top_k=5, save_log=False, min_retrieved_chunks=1)
    assert result["primary_card_count"] >= 1
    assert result["raw_evidence_count"] >= 1
    assert result["raw_fallback_count"] >= 1


def test_pipeline_disabled_setting_uses_legacy_prompt(monkeypatch):
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)
    monkeypatch.setattr(settings, "answer_with_knowledge_cards", False)

    chunks = [
        _make_kc_chunk(
            chunk_id="kc_a", file_name="card_a.txt", card_type="workflow", score=0.55
        ),
    ]
    pipeline, fake_gen = _make_pipeline(chunks)
    result = pipeline.ask("셋팅 가이드", top_k=5, save_log=False)
    assert result["answer_mode"] == "raw_fallback"
    assert "## 주 근거 (KnowledgeCard)" not in fake_gen.last_prompt


def test_pipeline_empty_question_returns_insufficient_evidence_mode():
    pipeline, _gen = _make_pipeline([])
    result = pipeline.ask("   ", save_log=False)
    assert result["generation_skipped"] is True
    assert result["answer_mode"] == "insufficient_evidence"
    assert result["primary_card_count"] == 0
    assert result["raw_evidence_count"] == 0
    assert result["raw_fallback_count"] == 0


def test_pipeline_result_exposes_primary_cards_for_ui_display(monkeypatch):
    monkeypatch.setattr(settings, "min_similarity_score", 0.0)
    monkeypatch.setattr(settings, "min_final_score", 0.0)

    chunks = [
        _make_kc_chunk(
            chunk_id="kc_a",
            file_name="card_a.txt",
            card_type="workflow",
            primary_topic="meta",
            task_type="setup",
            title="메타 셋팅 가이드",
            score=0.55,
        ),
    ]
    pipeline, _gen = _make_pipeline(chunks)
    result = pipeline.ask("메타 셋팅 가이드", top_k=5, save_log=False)

    cards = result["primary_cards"]
    assert len(cards) == 1
    md = cards[0].metadata
    assert md.get("card_id") == "kc_a"
    assert md.get("card_type") == "workflow"
    assert md.get("primary_topic") == "meta"
    assert md.get("task_type") == "setup"
