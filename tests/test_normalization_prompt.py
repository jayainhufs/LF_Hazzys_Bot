"""Normalizer prompt v1.5 schema guidance tests."""
from __future__ import annotations

from src.normalization.normalization_prompt import (
    build_guide_normalization_prompt,
    build_slack_normalization_prompt,
)


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


def test_guide_prompt_lists_v1_5_document_types_and_answer_use_cases():
    system_instruction, _ = build_guide_normalization_prompt(
        file_name="guide.txt",
        content="가이드 본문",
    )

    for document_type in NEW_DOCUMENT_TYPES:
        assert document_type in system_instruction
    for use_case in ANSWER_USE_CASES:
        assert use_case in system_instruction

    # Existing Guide-friendly types must remain visible.
    for document_type in [
        "workflow",
        "checklist",
        "faq",
        "glossary",
        "communication_template",
        "reference_note",
    ]:
        assert document_type in system_instruction
    assert "answer_use_cases" in system_instruction


def test_slack_prompt_lists_thread_friendly_v1_5_document_types():
    system_instruction, _ = build_slack_normalization_prompt(
        file_name="slack.txt",
        content="Slack 본문",
    )

    for document_type in [
        "status_update",
        "action_item",
        "issue_log",
        "decision_log",
        "communication_history",
    ]:
        assert document_type in system_instruction
    for use_case in ANSWER_USE_CASES:
        assert use_case in system_instruction
    assert "answer_use_cases" in system_instruction
