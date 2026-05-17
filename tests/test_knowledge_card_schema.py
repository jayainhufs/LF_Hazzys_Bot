"""NormalizedDocument schema 단위 테스트 (legacy 명: KnowledgeCard schema 테스트).

명칭 변경 노트
----------------
- 이 파일은 기존 ``test_knowledge_card_schema.py`` 명칭을 그대로 유지한다.
  (새로 추가된 ``tests/test_normalized_document_schema.py`` 가 동일한 테스트를
  새 명칭 기준으로 다시 실행한다.)
- 새 코드는 ``NormalizedDocument`` 를 우선 사용한다. 기존 ``KnowledgeCard`` 은 그대로
  alias 로 동작해야 하며, 이 파일은 그 호환성을 함께 검증한다.
"""
from __future__ import annotations

import pytest

from src.schemas import KnowledgeCard, NormalizedDocument
from src.schemas.normalized_document import (
    VALID_ANSWER_USE_CASES,
    VALID_NORMALIZED_DOCUMENT_TYPES,
)


LEGACY_DOCUMENT_TYPES = [
    "workflow",
    "checklist",
    "issue",
    "faq",
    "decision",
    "communication_template",
    "glossary",
]

NEW_DOCUMENT_TYPES = [
    "context_note",
    "status_update",
    "action_item",
    "issue_log",
    "decision_log",
    "campaign_summary",
    "communication_history",
    "reference_note",
    "report_insight",
]

ANSWER_USE_CASES = [
    "procedure",
    "summary",
    "troubleshooting",
    "draft_message",
    "compare",
    "history_lookup",
    "checklist",
    "freeform_grounded",
]


def _sample_document(**overrides) -> NormalizedDocument:
    data = {
        "card_id": "kc_001",
        "card_type": "workflow",
        "title": "메타 캠페인 세팅 체크",
        "summary": "메타 캠페인 세팅 시 확인해야 할 핵심 절차를 정리한 정규화 문서.",
        "source_file_name": "[2026년 4월 29일 TODO].txt",
        "source_file_hash": "a" * 64,
        "source_category": "slack",
        "source_type": "slack_manual",
        "document_date": "2026-04-29",
        "display_date": "해당 업무일",
        "primary_topic": "meta",
        "topic_tags": ["meta"],
        "task_type": "campaign_setting",
        "when_to_use": "메타 캠페인 세팅 전 최종 확인이 필요할 때 사용한다.",
        "prerequisites": ["컨첵시트 준비", "랜딩페이지 URL 확정"],
        "steps": ["캠페인 구조를 확인한다.", "광고세트 네이밍과 매핑을 확인한다."],
        "checkpoints": ["토글 ON/OFF 확인", "T&D 변경 여부 확인"],
        "cautions": ["정확한 시간/실명은 출력하지 않는다."],
        "examples": ["세팅 완료 후 검토자에게 크첵 요청"],
        "related_terms": ["ASC", "BAU", "컨첵"],
        "open_questions": ["광고주 공유 시점은 별도 확인 필요"],
        "evidence_spans": [
            {
                "section": "4/29 동제 중간 TODO",
                "chunk_index": 2,
                "summary": "캠페인 및 광고세트 세팅 항목이 언급됨",
            }
        ],
        "parent_raw_chunk_ids": ["raw_001", "raw_002"],
        "answer_use_cases": [],
        "sanitized_markdown": "",
        "metadata": {"prompt_version": "mock-v1"},
    }
    data.update(overrides)
    return NormalizedDocument(**data)


# --- backwards compatibility -------------------------------------------------


def test_knowledge_card_alias_is_normalized_document():
    """legacy ``KnowledgeCard`` 가 ``NormalizedDocument`` 의 alias 인지 확인."""
    assert KnowledgeCard is NormalizedDocument


def test_normalized_document_can_be_constructed_via_legacy_alias():
    legacy = KnowledgeCard(
        card_id="kc_alias",
        card_type="faq",
        title="alias title",
        summary="alias summary",
        source_file_name="x.txt",
        source_file_hash="b" * 64,
        source_category="guide",
        source_type="guide",
    )
    assert isinstance(legacy, NormalizedDocument)
    # 신규 명칭 property 가 동일 값을 반환해야 한다
    assert legacy.normalized_document_id == "kc_alias"
    assert legacy.normalized_document_type == "faq"


# --- core schema -------------------------------------------------------------


def test_normalized_document_can_be_created():
    doc = _sample_document()
    assert doc.normalized_document_id == "kc_001"
    assert doc.normalized_document_type == "workflow"
    assert doc.topic_tags == ["meta"]
    assert doc.answer_use_cases == []


@pytest.mark.parametrize("document_type", LEGACY_DOCUMENT_TYPES)
def test_validate_minimum_accepts_existing_document_types(document_type: str):
    assert document_type in VALID_NORMALIZED_DOCUMENT_TYPES
    assert _sample_document(card_type=document_type).validate_minimum() is True


@pytest.mark.parametrize("document_type", NEW_DOCUMENT_TYPES)
def test_validate_minimum_accepts_v1_5_document_types(document_type: str):
    assert document_type in VALID_NORMALIZED_DOCUMENT_TYPES
    assert _sample_document(card_type=document_type).validate_minimum() is True


def test_answer_use_cases_default_to_empty_list_for_legacy_documents():
    data = _sample_document().to_dict()
    data.pop("answer_use_cases")
    restored = NormalizedDocument.from_dict(data)
    assert restored.answer_use_cases == []
    assert restored.validate_minimum() is True


def test_answer_use_cases_are_preserved_in_roundtrip():
    doc = _sample_document(answer_use_cases=["summary", "draft_message"])
    restored = NormalizedDocument.from_dict(doc.to_dict())
    assert restored.answer_use_cases == ["summary", "draft_message"]
    assert restored.validate_minimum() is True


def test_multiple_answer_use_cases_are_allowed():
    use_cases = ["summary", "history_lookup", "freeform_grounded"]
    doc = _sample_document(answer_use_cases=use_cases)
    assert all(use_case in VALID_ANSWER_USE_CASES for use_case in use_cases)
    assert doc.answer_use_cases == use_cases
    assert doc.validate_minimum() is True


def test_unknown_answer_use_case_fails_minimum_validation():
    doc = _sample_document(answer_use_cases=["summary", "unknown_use_case"])
    assert doc.validate_minimum() is False


def test_to_dict_from_dict_roundtrip():
    doc = _sample_document()
    restored = NormalizedDocument.from_dict(doc.to_dict())
    assert restored == doc


def test_from_dict_handles_missing_optional_lists():
    data = _sample_document().to_dict()
    data["topic_tags"] = None
    data["steps"] = None
    data["answer_use_cases"] = None
    data["metadata"] = None
    restored = NormalizedDocument.from_dict(data)
    assert restored.topic_tags == []
    assert restored.steps == []
    assert restored.answer_use_cases == []
    assert restored.metadata == {}


def test_to_markdown_contains_required_sections():
    md = _sample_document().to_markdown()
    assert "# 메타 캠페인 세팅 체크" in md
    assert "- card_type: workflow" in md
    assert "- primary_topic: meta" in md
    assert "- source_file: [2026년 4월 29일 TODO].txt" in md
    assert "- answer_use_cases: -" in md
    assert "## 요약" in md
    assert "## 언제 사용하는가" in md
    assert "## 선행 조건" in md
    assert "## 업무 절차" in md
    assert "## 체크포인트" in md
    assert "## 주의사항" in md
    assert "## 예시" in md
    assert "## 관련 용어" in md
    assert "## 미확인 사항" in md
    assert "## 근거" in md


def test_to_markdown_renders_evidence_spans():
    md = _sample_document().to_markdown()
    assert "section: 4/29 동제 중간 TODO" in md
    assert "chunk_index: 2" in md
    assert "캠페인 및 광고세트 세팅 항목" in md


def test_validate_minimum_success():
    assert _sample_document().validate_minimum() is True


def test_validate_minimum_allows_sanitized_markdown_as_body():
    doc = _sample_document(
        prerequisites=[],
        steps=[],
        checkpoints=[],
        cautions=[],
        examples=[],
        related_terms=[],
        open_questions=[],
        evidence_spans=[],
        sanitized_markdown="# 별도 정규화 카드\n\n본문",
    )
    assert doc.validate_minimum() is True


def test_validate_minimum_fails_without_required_fields():
    assert _sample_document(card_id="").validate_minimum() is False
    assert _sample_document(card_type="").validate_minimum() is False
    assert _sample_document(card_type="unknown").validate_minimum() is False
    assert _sample_document(title="").validate_minimum() is False
    assert _sample_document(summary="").validate_minimum() is False
    assert _sample_document(source_file_name="").validate_minimum() is False
