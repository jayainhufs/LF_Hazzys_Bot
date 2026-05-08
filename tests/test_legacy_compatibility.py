"""
test_legacy_compatibility.py
============================
"KnowledgeCard" → "NormalizedDocument" 명칭 변경 후에도 기존 import / 저장
포맷 / metadata / env 가 그대로 동작하는지 검증한다.

관점
----
1. schema legacy alias (``KnowledgeCard`` 가 ``NormalizedDocument`` 과 동일)
2. metadata legacy 키 (``card_id`` / ``card_type``) 와 신규 키
   (``normalized_document_id`` / ``normalized_document_type``) 모두 인식
3. content_type legacy (``"knowledge_card"``) 와 신규 (``"normalized_document"``)
   모두 retrieval / qa pipeline 에서 인식
4. env var fallback (``PRIORITIZE_NORMALIZED_DOCUMENTS`` ↔ ``PRIORITIZE_KNOWLEDGE_CARDS`` 등)
5. 신규 명칭 / 함수가 정상적으로 노출되는지 확인
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pytest

from src.config import Settings, settings
from src.rag.reranker import (
    apply_knowledge_card_priority,
    apply_normalized_document_priority,
    card_type_boost_for,
    extract_query_metadata,
    is_knowledge_card_chunk,
    is_normalized_document_chunk,
    normalized_document_type_boost_for,
    rerank_simple,
)
from src.rag.prompt_builder import (
    build_knowledge_card_answer_prompt,
    build_normalized_document_answer_prompt,
    split_chunks_by_retrieval_role,
)
from src.schemas import KnowledgeCard, NormalizedDocument
from src.schemas.normalized_document import (
    VALID_CARD_TYPES,
    VALID_NORMALIZED_DOCUMENT_TYPES,
)


# ---------------------------------------------------------------------------
# 1) Schema alias
# ---------------------------------------------------------------------------
def test_knowledge_card_is_alias_of_normalized_document():
    assert KnowledgeCard is NormalizedDocument


def test_valid_types_aliases_match():
    assert VALID_CARD_TYPES is VALID_NORMALIZED_DOCUMENT_TYPES


def test_normalized_document_property_round_trip():
    doc = NormalizedDocument(
        card_id="kc1",
        card_type="workflow",
        title="t",
        summary="s",
        source_file_name="f.txt",
        source_file_hash="h",
        source_category="guide",
        source_type="guide",
    )
    assert doc.normalized_document_id == "kc1"
    assert doc.normalized_document_type == "workflow"
    doc.normalized_document_id = "kc2"
    doc.normalized_document_type = "faq"
    assert doc.card_id == "kc2"
    assert doc.card_type == "faq"


def test_from_dict_accepts_new_keys():
    """저장된 JSON 이 신규 키 (normalized_document_id) 만 가질 때도 복원되어야 한다."""
    data = {
        "normalized_document_id": "nd_001",
        "normalized_document_type": "checklist",
        "title": "t",
        "summary": "s",
        "source_file_name": "f.txt",
        "source_file_hash": "h",
        "source_category": "guide",
        "source_type": "guide",
    }
    doc = NormalizedDocument.from_dict(data)
    assert doc.card_id == "nd_001"
    assert doc.card_type == "checklist"


def test_from_dict_accepts_legacy_keys():
    """기존 색인 JSON 이 legacy 키 (card_id) 로 저장돼 있어도 복원되어야 한다."""
    data = {
        "card_id": "kc_legacy_001",
        "card_type": "workflow",
        "title": "t",
        "summary": "s",
        "source_file_name": "f.txt",
        "source_file_hash": "h",
        "source_category": "guide",
        "source_type": "guide",
    }
    doc = NormalizedDocument.from_dict(data)
    assert doc.normalized_document_id == "kc_legacy_001"
    assert doc.normalized_document_type == "workflow"


# ---------------------------------------------------------------------------
# 2) Reranker / retriever helper aliases
# ---------------------------------------------------------------------------
def test_is_normalized_document_chunk_alias():
    assert is_knowledge_card_chunk is is_normalized_document_chunk


def test_apply_priority_alias():
    assert apply_knowledge_card_priority is apply_normalized_document_priority


def test_card_type_boost_for_alias():
    assert card_type_boost_for is normalized_document_type_boost_for


# ---------------------------------------------------------------------------
# 3) content_type compatibility
# ---------------------------------------------------------------------------
def _make_chunk(
    *,
    chunk_id: str,
    file_name: str,
    content_type: str,
    metadata: Dict[str, Any],
    score: float = 0.6,
):
    from src.schemas import RetrievedChunk

    md = dict(metadata)
    md.setdefault("file_name", file_name)
    md.setdefault("uploaded_category", "guide")
    md.setdefault("source_type", md.get("source_type", "llm_normalized"))
    md.setdefault("content_type", content_type)
    md.setdefault("source_weight", 1.25)

    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc_{file_name}",
        file_name=file_name,
        source_type=md["source_type"],
        uploaded_category=md["uploaded_category"],
        section_title=None,
        content_type=content_type,
        content=f"body_{chunk_id}",
        score=float(score),
        final_score=float(score),
        metadata=md,
    )


def test_legacy_content_type_knowledge_card_recognized():
    chunk = _make_chunk(
        chunk_id="c1",
        file_name="legacy.txt",
        content_type="knowledge_card",
        metadata={"card_id": "kc_001", "card_type": "workflow"},
    )
    assert is_normalized_document_chunk(chunk) is True


def test_new_content_type_normalized_document_recognized():
    chunk = _make_chunk(
        chunk_id="c2",
        file_name="new.txt",
        content_type="normalized_document",
        metadata={
            "normalized_document_id": "nd_001",
            "normalized_document_type": "checklist",
        },
    )
    assert is_normalized_document_chunk(chunk) is True


def test_legacy_card_type_used_when_new_key_missing():
    """metadata 에 normalized_document_type 이 없어도 legacy card_type 으로 boost 가 계산돼야 한다."""
    cands = [
        _make_chunk(
            chunk_id="legacy",
            file_name="legacy.txt",
            content_type="knowledge_card",
            metadata={
                "card_id": "kc_legacy",
                "card_type": "workflow",
                "primary_topic": "meta",
            },
        )
    ]
    qm = extract_query_metadata("메타 셋팅 가이드 알려줘")
    rerank_simple(cands, query_metadata=qm)
    apply_normalized_document_priority(cands, query_metadata=qm)

    md = cands[0].metadata
    # 신규 키와 legacy 키 모두 채워진다
    assert md["retrieval_role"] == "primary_card"
    assert md["normalized_document_boost"] > 1.0
    assert md["normalized_document_type_boost"] > 1.0
    assert md["knowledge_card_boost"] == md["normalized_document_boost"]
    assert md["card_type_boost"] == md["normalized_document_type_boost"]


def test_split_chunks_recognizes_new_content_type():
    chunk = _make_chunk(
        chunk_id="nd1",
        file_name="x.txt",
        content_type="normalized_document",
        metadata={"normalized_document_id": "nd1", "normalized_document_type": "faq"},
    )
    chunk.metadata.pop("retrieval_role", None)
    groups = split_chunks_by_retrieval_role([chunk])
    assert len(groups["primary_cards"]) == 1


def test_split_chunks_recognizes_legacy_content_type():
    chunk = _make_chunk(
        chunk_id="kc1",
        file_name="x.txt",
        content_type="knowledge_card",
        metadata={"card_id": "kc1", "card_type": "faq"},
    )
    chunk.metadata.pop("retrieval_role", None)
    groups = split_chunks_by_retrieval_role([chunk])
    assert len(groups["primary_cards"]) == 1


# ---------------------------------------------------------------------------
# 4) Prompt builder alias
# ---------------------------------------------------------------------------
def test_build_normalized_document_answer_prompt_alias():
    assert build_knowledge_card_answer_prompt is build_normalized_document_answer_prompt


# ---------------------------------------------------------------------------
# 5) env var fallback
# ---------------------------------------------------------------------------
def _clear_keys(monkeypatch, keys: List[str]) -> None:
    for k in keys:
        monkeypatch.delenv(k, raising=False)


def test_env_fallback_prefers_new_normalized_document_var(monkeypatch):
    """PRIORITIZE_NORMALIZED_DOCUMENTS 가 있으면 신규 값을 우선 사용한다."""
    _clear_keys(
        monkeypatch,
        ["PRIORITIZE_NORMALIZED_DOCUMENTS", "PRIORITIZE_KNOWLEDGE_CARDS"],
    )
    monkeypatch.setenv("PRIORITIZE_NORMALIZED_DOCUMENTS", "false")
    monkeypatch.setenv("PRIORITIZE_KNOWLEDGE_CARDS", "true")
    s = Settings.from_env()
    assert s.prioritize_knowledge_cards is False
    assert s.prioritize_normalized_documents is False


def test_env_fallback_uses_legacy_var_when_new_missing(monkeypatch):
    """PRIORITIZE_NORMALIZED_DOCUMENTS 가 없으면 legacy 값을 사용한다."""
    _clear_keys(
        monkeypatch,
        ["PRIORITIZE_NORMALIZED_DOCUMENTS", "PRIORITIZE_KNOWLEDGE_CARDS"],
    )
    monkeypatch.setenv("PRIORITIZE_KNOWLEDGE_CARDS", "false")
    s = Settings.from_env()
    assert s.prioritize_knowledge_cards is False
    assert s.prioritize_normalized_documents is False


def test_env_fallback_for_content_boost(monkeypatch):
    _clear_keys(
        monkeypatch,
        ["NORMALIZED_DOCUMENT_CONTENT_BOOST", "KNOWLEDGE_CARD_CONTENT_BOOST"],
    )
    monkeypatch.setenv("NORMALIZED_DOCUMENT_CONTENT_BOOST", "1.99")
    s = Settings.from_env()
    assert s.knowledge_card_content_boost == pytest.approx(1.99)
    assert s.normalized_document_content_boost == pytest.approx(1.99)


def test_env_fallback_for_answer_with(monkeypatch):
    _clear_keys(
        monkeypatch,
        ["ANSWER_WITH_NORMALIZED_DOCUMENTS", "ANSWER_WITH_KNOWLEDGE_CARDS"],
    )
    monkeypatch.setenv("ANSWER_WITH_KNOWLEDGE_CARDS", "false")
    s = Settings.from_env()
    assert s.answer_with_knowledge_cards is False
    assert s.answer_with_normalized_documents is False


def test_env_fallback_for_max_primary(monkeypatch):
    _clear_keys(
        monkeypatch,
        ["MAX_PRIMARY_NORMALIZED_DOCUMENTS", "MAX_PRIMARY_CARDS"],
    )
    monkeypatch.setenv("MAX_PRIMARY_NORMALIZED_DOCUMENTS", "9")
    s = Settings.from_env()
    assert s.max_primary_cards == 9
    assert s.max_primary_normalized_documents == 9


def test_env_fallback_for_normalization_model(monkeypatch):
    _clear_keys(
        monkeypatch,
        ["LLM_DOCUMENT_NORMALIZATION_MODEL", "LLM_NORMALIZATION_MODEL"],
    )
    monkeypatch.setenv("LLM_NORMALIZATION_MODEL", "fake-legacy-model")
    s = Settings.from_env()
    assert s.llm_normalization_model == "fake-legacy-model"
    assert s.llm_document_normalization_model == "fake-legacy-model"


# ---------------------------------------------------------------------------
# 6) Settings property aliases on shared instance
# ---------------------------------------------------------------------------
def test_settings_property_aliases_reflect_legacy_values():
    # 신규 property 가 legacy field 를 그대로 노출한다
    assert settings.prioritize_normalized_documents == settings.prioritize_knowledge_cards
    assert settings.normalized_document_content_boost == settings.knowledge_card_content_boost
    assert settings.answer_with_normalized_documents == settings.answer_with_knowledge_cards
    assert settings.max_primary_normalized_documents == settings.max_primary_cards
