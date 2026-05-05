"""KnowledgeCard schema 단위 테스트."""
from __future__ import annotations

from src.schemas import KnowledgeCard


def _sample_card(**overrides) -> KnowledgeCard:
    data = {
        "card_id": "kc_001",
        "card_type": "workflow",
        "title": "메타 캠페인 세팅 체크",
        "summary": "메타 캠페인 세팅 시 확인해야 할 핵심 절차를 정리한 카드.",
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
        "sanitized_markdown": "",
        "metadata": {"prompt_version": "mock-v1"},
    }
    data.update(overrides)
    return KnowledgeCard(**data)


def test_knowledge_card_can_be_created():
    card = _sample_card()
    assert card.card_id == "kc_001"
    assert card.card_type == "workflow"
    assert card.topic_tags == ["meta"]


def test_to_dict_from_dict_roundtrip():
    card = _sample_card()
    restored = KnowledgeCard.from_dict(card.to_dict())
    assert restored == card


def test_from_dict_handles_missing_optional_lists():
    data = _sample_card().to_dict()
    data["topic_tags"] = None
    data["steps"] = None
    data["metadata"] = None
    restored = KnowledgeCard.from_dict(data)
    assert restored.topic_tags == []
    assert restored.steps == []
    assert restored.metadata == {}


def test_to_markdown_contains_required_sections():
    md = _sample_card().to_markdown()
    assert "# 메타 캠페인 세팅 체크" in md
    assert "- card_type: workflow" in md
    assert "- primary_topic: meta" in md
    assert "- source_file: [2026년 4월 29일 TODO].txt" in md
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
    md = _sample_card().to_markdown()
    assert "section: 4/29 동제 중간 TODO" in md
    assert "chunk_index: 2" in md
    assert "캠페인 및 광고세트 세팅 항목" in md


def test_validate_minimum_success():
    assert _sample_card().validate_minimum() is True


def test_validate_minimum_allows_sanitized_markdown_as_body():
    card = _sample_card(
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
    assert card.validate_minimum() is True


def test_validate_minimum_fails_without_required_fields():
    assert _sample_card(card_id="").validate_minimum() is False
    assert _sample_card(card_type="").validate_minimum() is False
    assert _sample_card(card_type="unknown").validate_minimum() is False
    assert _sample_card(title="").validate_minimum() is False
    assert _sample_card(summary="").validate_minimum() is False
    assert _sample_card(source_file_name="").validate_minimum() is False
