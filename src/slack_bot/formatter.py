"""
src.slack_bot.formatter
=======================
QA Pipeline 결과 dict (``qa_adapter.answer_slack_question`` 의 반환값) 를
Slack 메시지 텍스트로 변환한다.

설계 원칙
---------
- 기존 ``qa_pipeline`` / ``prompt_builder`` 가 만든 답변 본문 자체는 건드리지
  않는다. Slack 출력 시에만 후처리로 잘라낸다.
- Streamlit 출력에는 영향을 주지 않는다 (Streamlit 은 이 모듈을 사용하지 않음).
- Slack 기본 출력은 답변 본문만 노출한다. 참고 근거 / 진단 정보는
  ``SlackBotSettings.show_sources`` / ``show_diagnostics`` 가 켜져 있거나,
  사용자가 질문에 ``--debug`` 를 붙였을 때만 노출된다.
- 답변 본문의 ``## 6. 참고 근거``, ``## 7. 불확실한 부분`` 등 Slack 에서 잡음이
  되는 섹션은 ``trim_answer_for_slack`` 으로 제거한다.
- Markdown heading (``## 1. 결론``) 은 Slack mrkdwn (``*1. 결론*``) 으로 변환한다.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
# Slack 메시지 자체 한계 (~40000) 보다 보수적으로 잡는다.
# 답변/근거/진단을 모두 합친 최종 메시지 길이의 hard ceiling.
SLACK_MESSAGE_HARD_LIMIT = 3500
# 기본 출력에서 답변 본문 한 덩어리의 권장 상한. 운영에서는
# ``SlackBotSettings.max_response_chars`` 가 1순위로 사용된다.
DEFAULT_MAX_RESPONSE_CHARS = 2500
# debug 모드에서 노출할 source 라벨 개수.
# MVP 2차 Step 1: 검색 진단 가시성을 위해 2 -> 3 으로 확장.
DEBUG_MAX_SOURCES = 3
# debug 모드 source preview 최대 글자수.
# raw content 노출을 늘리는 게 아니라 진단 라벨을 함께 보여주기 위해 약간 축소.
DEBUG_SOURCE_PREVIEW_LIMIT = 120
TRUNCATION_NOTICE = "\n…(메시지 길이 제한으로 일부 잘라냄)"

# Slack 에서 제거할 본문 섹션 헤더들.
# - "## 6. 참고 근거", "### 참고 근거", "## 참고 근거" 등
# - "## 7. 불확실한 부분", "## 불확실한 부분"
# - "## 진단", "### 진단"
# - bare heading "참고 근거" / "진단" (heading 이거나 단독 줄일 때만)
_TRIM_HEADING_KEYWORDS = (
    "참고 근거",
    "참고근거",
    "불확실한 부분",
    "불확실한부분",
    "진단",
    "진단 정보",
    "References",
    "Diagnostics",
)


# ---------------------------------------------------------------------------
# 공개 API — 안내 메시지
# ---------------------------------------------------------------------------
def format_help_message() -> str:
    """질문이 비어 있을 때 사용자에게 보여줄 사용법."""
    return (
        "사용법: 채널에서 봇을 멘션한 뒤 질문을 적어주세요.\n"
        "예) `@봇 세팅 전에 확인해야 할 것 알려줘`\n"
        "참고 근거 / 진단을 같이 보고 싶다면 질문 끝에 `--debug` 를 붙이세요."
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


# ---------------------------------------------------------------------------
# 공개 API — 답변 본문 정리 (단위 테스트가 직접 호출)
# ---------------------------------------------------------------------------
def trim_answer_for_slack(answer: str) -> str:
    """
    Slack 기본 출력용으로 QA answer 를 정리한다.

    아래 섹션이 등장하는 시점부터 뒤를 모두 제거한다.

    - ``## 6. 참고 근거`` / ``## 7. 불확실한 부분``
    - ``## 참고 근거`` / ``### 참고 근거``
    - ``## 불확실한 부분`` / ``### 불확실한 부분``
    - ``## 진단`` / ``### 진단``
    - heading 으로 등장한 ``참고 근거`` / ``진단``
      (예: ``참고 근거`` 한 줄 + 다음 줄부터 항목)

    본문 중간에 "참고 근거" 라는 단어가 일반 문장으로 들어 있는 경우는
    제거하지 않는다. heading / 단독 라인일 때만 trim 한다.
    """
    if not answer:
        return ""
    lines = answer.splitlines()
    cut: Optional[int] = None
    for i, raw_line in enumerate(lines):
        if _is_trim_marker_line(raw_line, lines, i):
            cut = i
            break
    if cut is not None:
        lines = lines[:cut]
    # 트레일 공백 라인 제거.
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def convert_markdown_headings_to_slack(text: str) -> str:
    """
    Markdown heading (``## 1. 결론``) 을 Slack mrkdwn (``*1. 결론*``) 으로
    변환한다.

    - ``#``, ``##``, ``###`` 등 ``#`` 의 개수와 무관하게 모두 ``*...*`` 로 변환.
    - heading 한 줄 안에 이미 ``*`` 가 있으면 그대로 두지 않고 안쪽 ``*`` 만
      벗겨낸 뒤 다시 감싼다 (이중 ``**...**`` 방지).
    - 본문 중간 ``# 해시태그`` 같은 패턴은 건드리지 않는다 — heading 으로
      인식되려면 줄 시작이어야 한다.
    """
    if not text:
        return ""
    out_lines: List[str] = []
    heading_re = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*$")
    for line in text.splitlines():
        m = heading_re.match(line)
        if not m:
            out_lines.append(line)
            continue
        body = m.group(1).strip()
        # 트레일 ``#`` (atx 닫는 헤더) 제거: "## 결론 ##" -> "결론"
        body = re.sub(r"\s+#+\s*$", "", body).strip()
        # 안쪽 ``*`` 는 한 번 벗겨 둔다.
        body = body.replace("*", "")
        if body:
            out_lines.append(f"*{body}*")
        else:
            out_lines.append("")
    return "\n".join(out_lines)


def tidy_slack_text(text: str) -> str:
    """
    Slack mrkdwn 가독성을 위한 가벼운 정돈.

    - U+00A0 / U+2007 / U+202F 등 보이지 않는 공백을 일반 공백으로 변환.
    - 줄 끝의 공백 제거.
    - 연속 빈 줄 3개 이상을 빈 줄 1개로 축소.
    - 전체 앞/뒤 공백 strip.
    """
    if not text:
        return ""
    # 특수 공백 → 공백.
    cleaned = re.sub(r"[\u00a0\u2007\u202f\u2009\u200a\u200b]", " ", text)
    # 라인 단위로 trailing space 제거.
    cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines())
    # 3개 이상 연속 newline → 2개로.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# 공개 API — Slack 답변 메인 포맷터
# ---------------------------------------------------------------------------
def format_qa_result(
    result: Dict[str, Any],
    *,
    question: Optional[str] = None,
    debug: bool = False,
    show_sources: Optional[bool] = None,
    show_diagnostics: Optional[bool] = None,
    max_response_chars: int = DEFAULT_MAX_RESPONSE_CHARS,
) -> str:
    """
    QA adapter 결과를 Slack 메시지 문자열로 변환한다.

    Parameters
    ----------
    result : dict
        ``qa_adapter.answer_slack_question`` 의 반환 dict.
    question : str, optional
        (현재는 사용하지 않지만 향후 확장 위해 유지)
    debug : bool
        ``--debug`` 모드 여부. True 면 ``show_sources`` /
        ``show_diagnostics`` 가 None / False 여도 짧은 근거 요약과 진단
        정보를 노출한다.
    show_sources : bool, optional
        명시적으로 참고 근거 표시 여부를 지정. ``None`` 이면 debug 모드일
        때만 표시.
    show_diagnostics : bool, optional
        명시적으로 진단 표시 여부를 지정. ``None`` 이면 debug 모드일
        때만 표시.
    max_response_chars : int
        답변 본문 (heading 변환 / trim 적용 후) 의 최대 글자수.
    """
    # 기본 정책: 명시 인자가 없으면 debug 가 ON 일 때만 노출.
    effective_show_sources = (
        show_sources if show_sources is not None else debug
    )
    effective_show_diagnostics = (
        show_diagnostics if show_diagnostics is not None else debug
    )

    sections: List[str] = []

    # ----- 답변 본문 -----
    raw_answer = (result.get("answer") or "").strip()
    if not raw_answer:
        answer_body = "(답변이 비어 있습니다.)"
    else:
        # Slack 출력용 후처리:
        # 1) 6/7번 등 잡음 섹션 제거 → 2) heading 변환 → 3) 공백 정돈.
        trimmed = trim_answer_for_slack(raw_answer)
        converted = convert_markdown_headings_to_slack(trimmed)
        answer_body = tidy_slack_text(converted)
        if not answer_body:
            answer_body = "(답변이 비어 있습니다.)"

    # 답변 본문은 최우선 자원이므로 max_response_chars 안에 들어오도록 잘라둔다.
    body_limit = max_response_chars if max_response_chars > 0 else DEFAULT_MAX_RESPONSE_CHARS
    answer_body = _clip(answer_body, body_limit)
    sections.append(answer_body)

    # ----- 참고 근거 (옵션) -----
    if effective_show_sources:
        sources_block = (
            _format_sources_debug(result.get("sources") or {})
            if debug
            else _format_sources_full(result.get("sources") or {})
        )
        if sources_block:
            sections.append(sources_block)

    # ----- 진단 정보 (옵션) -----
    if effective_show_diagnostics:
        diag_block = _format_diagnostics(
            result.get("diagnostics") or {},
            primary_count=int(
                result.get("primary_normalized_document_count") or 0
            ),
            raw_evidence_count=int(result.get("raw_evidence_count") or 0),
            raw_fallback_count=int(result.get("raw_fallback_count") or 0),
            answer_mode=str(result.get("answer_mode") or ""),
        )
        if diag_block:
            sections.append("*진단*\n" + diag_block)

    text = "\n\n".join(sections).strip()
    return _enforce_slack_limit(text)


# ---------------------------------------------------------------------------
# 내부 helper — answer trim
# ---------------------------------------------------------------------------
def _strip_emphasis(s: str) -> str:
    """``*텍스트*`` / ``**텍스트**`` 같은 강조를 벗겨낸다 (heading 비교용)."""
    return re.sub(r"[*_`~]+", "", s).strip()


def _heading_body(line: str) -> Optional[str]:
    """
    한 줄이 markdown heading 이면 heading 본문 (``"6. 참고 근거"`` 등) 을
    반환. heading 이 아니면 None.
    """
    m = re.match(r"^\s{0,3}#{1,6}\s+(.*?)\s*$", line)
    if not m:
        return None
    body = m.group(1).strip()
    body = re.sub(r"\s+#+\s*$", "", body).strip()
    return body or None


def _matches_trim_keyword(body: str) -> bool:
    """heading body 가 trim 대상 키워드를 포함하면 True."""
    if not body:
        return False
    # 번호 / 점 / 콜론 제거 후 키워드 비교 ("6. 참고 근거" / "참고 근거:")
    norm = re.sub(r"^[\d\.\)\s]+", "", body)
    norm = norm.rstrip(" :：").strip()
    norm_compact = norm.replace(" ", "")
    for kw in _TRIM_HEADING_KEYWORDS:
        kw_compact = kw.replace(" ", "")
        if norm == kw or norm_compact == kw_compact:
            return True
    return False


def _is_trim_marker_line(
    line: str, all_lines: List[str], idx: int
) -> bool:
    """
    이 줄이 본문 trim 시작점인지 판정.

    조건:
    1. markdown heading (``#`` 들로 시작) 이고 본문이 trim 키워드 매치, 또는
    2. ``*참고 근거*`` / ``**참고 근거**`` 같은 강조 라인이 trim 키워드 매치, 또는
    3. bare 한 줄 ``참고 근거`` 가 단독 등장하고 다음 줄이 빈 줄 또는 ``-``,
       ``•``, 숫자 시작 등 list 시작이라 heading 으로 보이는 경우.
    """
    body = _heading_body(line)
    if body and _matches_trim_keyword(body):
        return True

    stripped = line.strip()
    if not stripped:
        return False

    # ``*참고 근거*`` 같은 강조-only 한 줄.
    if re.match(r"^[\*_]{1,2}[^*_]+[\*_]{1,2}\s*:?\s*$", stripped):
        emph_body = _strip_emphasis(stripped).rstrip(" :：")
        if _matches_trim_keyword(emph_body):
            return True

    # bare 단독 줄 ("참고 근거" 만 있고 다음 줄이 빈 줄/리스트 시작).
    bare_body = _strip_emphasis(stripped).rstrip(" :：")
    if _matches_trim_keyword(bare_body):
        next_line = all_lines[idx + 1] if idx + 1 < len(all_lines) else ""
        next_stripped = next_line.strip()
        if (
            not next_stripped
            or next_stripped.startswith(("-", "•", "*", "·"))
            or re.match(r"^\d+[\.\)]", next_stripped)
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# 내부 helper — 길이 / 출력
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


# ---------------------------------------------------------------------------
# 내부 helper — 참고 근거 / 진단 (옵션 활성 시에만 사용)
# ---------------------------------------------------------------------------
def _format_sources_full(sources: Dict[str, List[Dict[str, Any]]]) -> str:
    """``SLACK_SHOW_SOURCES=true`` 모드에서 사용하는 상세 근거 블록."""
    if not sources:
        return ""
    primaries = sources.get("primary_normalized_documents") or []
    raw_ev = sources.get("raw_evidence") or []
    raw_fb = sources.get("raw_fallback") or []

    lines: List[str] = []
    if primaries:
        lines.append("• Normalized Document")
        lines.extend(_format_source_lines(primaries, preview_limit=220, max_items=3))
    if raw_ev:
        lines.append("• Raw Evidence")
        lines.extend(_format_source_lines(raw_ev, preview_limit=220, max_items=3))
    if raw_fb and not primaries and not raw_ev:
        lines.append("• Raw Fallback")
        lines.extend(_format_source_lines(raw_fb, preview_limit=220, max_items=3))

    if not lines:
        return ""
    return "*참고 근거*\n" + "\n".join(lines)


def _format_sources_debug(sources: Dict[str, List[Dict[str, Any]]]) -> str:
    """
    ``--debug`` 모드에서 사용하는 짧은 근거 요약.

    MVP 2차 Step 1 (Retrieval Diagnostics 강화):
    label / preview 뿐만 아니라 검색 진단 정보를 함께 노출한다 — 단, raw 원문
    노출을 늘리지 않도록 라벨 위주로 짧게만 보여준다.

    각 source 에는 (가능하면) 다음 진단 라인을 붙인다:
        ``content_type=... · primary_topic=... · role=... · final=...``
    """
    if not sources:
        return ""
    primaries = sources.get("primary_normalized_documents") or []
    raw_ev = sources.get("raw_evidence") or []
    raw_fb = sources.get("raw_fallback") or []

    pool: List[Dict[str, Any]] = []
    pool.extend(primaries)
    pool.extend(raw_ev)
    if not pool:
        pool.extend(raw_fb)
    if not pool:
        return ""
    items = pool[:DEBUG_MAX_SOURCES]
    lines = ["*참고 근거 (debug)*"]
    for it in items:
        label = (it.get("label") or "").strip()
        if not label:
            label = (it.get("file_name") or "(이름 없음)").strip()
        preview = _clip((it.get("preview") or "").strip(), DEBUG_SOURCE_PREVIEW_LIMIT)
        diag_line = _format_source_diag_line(it)
        lines.append(f"• {label}")
        if diag_line:
            lines.append(f"   › {diag_line}")
        if preview:
            lines.append(f"   › {preview}")
    extra = len(pool) - len(items)
    if extra > 0:
        lines.append(f"• …외 {extra}건")
    return "\n".join(lines)


def _format_source_diag_line(it: Dict[str, Any]) -> str:
    """
    Slack ``--debug`` 모드에서 source 한 줄당 사용하는 진단 라인.

    표시 필드 (값이 있을 때만):
    - ``content_type``
    - ``primary_topic``
    - ``retrieval_role``
    - ``final_score``

    raw 원문이나 긴 metadata 는 노출하지 않는다.
    """
    parts: List[str] = []
    ct = (it.get("content_type") or "").strip()
    if ct:
        parts.append(f"content_type=`{ct}`")
    pt = (it.get("primary_topic") or "").strip()
    if pt:
        parts.append(f"primary_topic=`{pt}`")
    role = (it.get("retrieval_role") or "").strip()
    if role:
        parts.append(f"role=`{role}`")
    final_score = it.get("final_score")
    if isinstance(final_score, (int, float)):
        parts.append(f"final=`{float(final_score):.3f}`")
    return " · ".join(parts)


def _format_source_lines(
    items: Iterable[Dict[str, Any]],
    *,
    preview_limit: int,
    max_items: int,
) -> List[str]:
    out: List[str] = []
    items_list = list(items)
    shown = items_list[:max_items]
    for it in shown:
        label = (it.get("label") or "").strip() or "(이름 없음)"
        preview = (it.get("preview") or "").strip()
        preview = _clip(preview, preview_limit)
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
    """
    Slack ``--debug`` 진단 블록.

    MVP 2차 Step 1 (Retrieval Diagnostics 강화):
    검색 단계가 왜 그런 결과를 냈는지 보이도록 ``query_topic`` /
    ``query_intent`` / ``query_date`` / ``retrieved_count`` /
    ``passed_count`` / ``topic_mismatch_count`` / candidate 구성을 함께
    표시한다. 표시 정책만 강화하고, 점수/필터/penalty 는 변경하지 않는다.
    """
    lines: List[str] = []
    mode = diagnostics.get("answer_mode") or answer_mode or "unknown"
    lines.append(f"• answer_mode: `{mode}`")

    # ----- MVP 2차 Step 4: Raw Fallback 오남용 방지 진단 -----
    # evidence_strength 는 가장 먼저 노출해 답변 신뢰도 한눈에 확인할 수 있게 한다.
    evidence_strength = (diagnostics.get("evidence_strength") or "").strip()
    if evidence_strength:
        lines.append(f"• evidence_strength: `{evidence_strength}`")

    if diagnostics.get("weak_evidence_warning"):
        # weak_evidence_warning 이 True 일 때만 노출 (정상 케이스 잡음 최소화).
        lines.append("• weak_evidence_warning: `True`")

    if diagnostics.get("raw_fallback_only"):
        reason = diagnostics.get("raw_fallback_only_reason")
        if reason:
            lines.append(
                f"• raw_fallback_only: `True` · reason=`{reason}`"
            )
        else:
            lines.append("• raw_fallback_only: `True`")

    fb_mismatch = int(diagnostics.get("raw_fallback_topic_mismatch_count") or 0)
    if fb_mismatch:
        fb_ratio = float(
            diagnostics.get("raw_fallback_topic_mismatch_ratio") or 0.0
        )
        lines.append(
            "• raw_fallback_topic_mismatch_count: "
            f"{fb_mismatch} (ratio={fb_ratio:.2f})"
        )

    # ----- query 진단 -----
    query_topic = diagnostics.get("query_topic")
    if query_topic:
        lines.append(f"• query_topic: `{query_topic}`")
    else:
        lines.append("• query_topic: `(none)`")

    intents = diagnostics.get("query_intent") or []
    if isinstance(intents, list) and intents:
        lines.append(f"• query_intent: `{', '.join(str(x) for x in intents)}`")

    q_date = diagnostics.get("query_date")
    if q_date:
        lines.append(f"• query_date: `{q_date}`")

    # ----- retrieval 진단 -----
    retrieved = int(diagnostics.get("retrieved_count") or 0)
    passed = int(diagnostics.get("passed_count") or 0)
    if retrieved or passed:
        lines.append(f"• retrieved_count: {retrieved} · passed_count: {passed}")

    nd_cand = int(diagnostics.get("normalized_document_candidate_count") or 0)
    raw_cand = int(diagnostics.get("raw_candidate_count") or 0)
    if nd_cand or raw_cand:
        lines.append(
            f"• candidate: normalized_document={nd_cand} · raw={raw_cand}"
        )

    mismatch = int(diagnostics.get("topic_mismatch_count") or 0)
    if mismatch:
        lines.append(f"• topic_mismatch_count: {mismatch}")

    # MVP 2차 Step 2: topic-aware 격하 진단 — 발생했을 때만 노출.
    demoted = int(diagnostics.get("topic_mismatch_demoted_count") or 0)
    if demoted:
        lines.append(f"• topic_mismatch_demoted_count: {demoted}")

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
