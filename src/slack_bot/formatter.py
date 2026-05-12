"""
src.slack_bot.formatter
=======================
QA Pipeline 결과 dict (``qa_adapter.answer_slack_question`` 의 반환값) 를
Slack 메시지 텍스트로 변환한다.

Slack 메시지 길이 제한 (~40000 chars) 안에 들어오도록 안전하게 잘라내며,
raw 원문이 그대로 노출되지 않도록 adapter 단계에서 만들어진 짧은 preview
만을 사용한다.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


# Slack 메시지 자체 한계 (~40000) 보다 보수적으로 잡는다.
SLACK_MESSAGE_HARD_LIMIT = 3500
ANSWER_BODY_LIMIT = 2200
SOURCE_PREVIEW_LIMIT = 240
MAX_SOURCES_PER_GROUP = 3
TRUNCATION_NOTICE = "\n…(메시지 길이 제한으로 일부 잘라냄)"


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------
def format_help_message() -> str:
    """질문이 비어 있을 때 사용자에게 보여줄 사용법."""
    return (
        "사용법: 채널에서 봇을 멘션한 뒤 질문을 적어주세요.\n"
        "예) `@봇 세팅 전에 확인해야 할 것 알려줘`"
    )


def format_too_long_message(max_chars: int) -> str:
    """질문이 너무 길어서 잘라냈을 때 보여줄 안내."""
    return (
        f"질문이 너무 깁니다. 첫 {max_chars}자만 사용해 답변을 시도하겠습니다. "
        "질문을 더 짧게 쪼개 주시면 정확도가 올라갑니다."
    )


def format_not_allowed_message() -> str:
    """허용 채널/유저가 아닐 때의 안내 (로그용)."""
    return (
        "이 채널 또는 사용자는 봇 응답이 허용되어 있지 않습니다. "
        "관리자에게 문의해 주세요."
    )


def format_internal_error_message() -> str:
    """내부 오류 시 사용자에게 보여줄 메시지 (traceback 노출 금지)."""
    return (
        ":warning: 답변 처리 중 내부 오류가 발생했습니다. "
        "잠시 후 다시 시도해 주세요. 문제가 계속되면 관리자에게 알려주세요."
    )


def format_qa_result(
    result: Dict[str, Any],
    *,
    question: Optional[str] = None,
) -> str:
    """
    ``qa_adapter.answer_slack_question`` 의 반환 dict 를 Slack 메시지 문자열로
    변환한다.

    포함 섹션:
      - 답변
      - 참고 근거 (Normalized Document / Raw Evidence)
      - 진단 정보 (answer_mode, count 등)
    """
    sections: List[str] = []

    # ----- 답변 -----
    answer = (result.get("answer") or "").strip()
    if not answer:
        answer = "(답변이 비어 있습니다.)"
    answer = _clip(answer, ANSWER_BODY_LIMIT)
    sections.append("*답변*\n" + answer)

    # ----- 참고 근거 -----
    sources = result.get("sources") or {}
    sources_block = _format_sources(sources)
    if sources_block:
        sections.append("*참고 근거*\n" + sources_block)

    # ----- 진단 정보 -----
    diagnostics = result.get("diagnostics") or {}
    diag_block = _format_diagnostics(
        diagnostics,
        primary_count=int(
            result.get("primary_normalized_document_count") or 0
        ),
        raw_evidence_count=int(result.get("raw_evidence_count") or 0),
        raw_fallback_count=int(result.get("raw_fallback_count") or 0),
        answer_mode=str(result.get("answer_mode") or ""),
    )
    if diag_block:
        sections.append("*진단*\n" + diag_block)

    text = "\n\n".join(sections)
    return _enforce_slack_limit(text)


# ---------------------------------------------------------------------------
# 내부 helper
# ---------------------------------------------------------------------------
def _clip(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _enforce_slack_limit(text: str) -> str:
    if len(text) <= SLACK_MESSAGE_HARD_LIMIT:
        return text
    keep = SLACK_MESSAGE_HARD_LIMIT - len(TRUNCATION_NOTICE)
    if keep < 100:
        keep = 100
    return text[:keep].rstrip() + TRUNCATION_NOTICE


def _format_sources(sources: Dict[str, List[Dict[str, Any]]]) -> str:
    if not sources:
        return ""

    primaries = sources.get("primary_normalized_documents") or []
    raw_ev = sources.get("raw_evidence") or []
    raw_fb = sources.get("raw_fallback") or []

    lines: List[str] = []
    if primaries:
        lines.append("• Normalized Document")
        lines.extend(_format_source_lines(primaries))
    if raw_ev:
        lines.append("• Raw Evidence")
        lines.extend(_format_source_lines(raw_ev))
    if raw_fb and not primaries and not raw_ev:
        # primary/evidence 가 모두 없을 때만 fallback 을 보여 준다.
        # (raw_fallback 은 mode 가 "raw_fallback" 인 경우 의미가 큰 정보)
        lines.append("• Raw Fallback")
        lines.extend(_format_source_lines(raw_fb))

    if not lines:
        return ""
    return "\n".join(lines)


def _format_source_lines(items: Iterable[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    items_list = list(items)
    shown = items_list[:MAX_SOURCES_PER_GROUP]
    for it in shown:
        label = (it.get("label") or "").strip() or "(이름 없음)"
        preview = (it.get("preview") or "").strip()
        preview = _clip(preview, SOURCE_PREVIEW_LIMIT)
        if preview:
            out.append(f"   - {label}\n     › {preview}")
        else:
            out.append(f"   - {label}")
    extra = len(items_list) - len(shown)
    if extra > 0:
        out.append(f"   - …외 {extra}건")
    return out


def _format_diagnostics(
    diagnostics: Dict[str, Any],
    *,
    primary_count: int,
    raw_evidence_count: int,
    raw_fallback_count: int,
    answer_mode: str,
) -> str:
    lines: List[str] = []
    mode = diagnostics.get("answer_mode") or answer_mode or "unknown"
    lines.append(f"• answer_mode: `{mode}`")
    lines.append(f"• primary_normalized_document_count: {primary_count}")
    lines.append(f"• raw_evidence_count: {raw_evidence_count}")
    lines.append(f"• raw_fallback_count: {raw_fallback_count}")

    skip_reason = diagnostics.get("skip_reason")
    if diagnostics.get("generation_skipped") and skip_reason:
        lines.append(f"• skip_reason: `{skip_reason}`")

    model_name = diagnostics.get("model_name")
    if model_name:
        lines.append(f"• model: `{model_name}`")
    return "\n".join(lines)
