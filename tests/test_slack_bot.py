"""
test_slack_bot.py
=================
Slack QA Bot MVP 단위 테스트.

외부 Slack API / Bolt App / 실제 Gemini API 는 호출하지 않는다.

검증 항목:
- mention 제거 (``strip_bot_mentions``)
- 빈 질문 → 사용법 안내
- allowed channel/user 체크
- formatter 가 answer/sources/diagnostics 를 Slack 메시지로 변환
- handler 가 qa_pipeline adapter 를 호출하고 thread 에 응답
- token 누락 시 graceful 실패 (``SlackBotSettings.validate``)
- ``run_slack_bot.py`` 가 SLACK_BOT_ENABLED=false 에서 graceful 종료
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, List

import pytest

# tests/conftest.py 가 sys.path 에 ROOT 를 넣어준다.
from src.slack_bot import formatter, handlers
from src.slack_bot.config import SlackBotSettings


# ---------------------------------------------------------------------------
# 공통 helper
# ---------------------------------------------------------------------------
class _FakePost:
    """``say`` 또는 ``client.chat_postMessage`` 호환 fake."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


def _make_settings(**overrides: Any) -> SlackBotSettings:
    base = SlackBotSettings(
        enabled=True,
        mode="socket",
        bot_token="xoxb-test-token",
        app_token="xapp-test-token",
        allowed_channel_ids=set(),
        allowed_user_ids=set(),
        reply_in_thread=True,
        max_question_chars=1000,
        show_sources=False,
        show_diagnostics=False,
        max_response_chars=2500,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


# 실제 운영 답변과 비슷한 형태 (1~7번 섹션) 의 fake 결과.
def _qa_result_with_full_sections() -> Dict[str, Any]:
    answer = (
        "## 1. 결론\n"
        "카카오 메시지 발송 세팅 절차 요약.\n\n"
        "## 2. 업무 처리 순서\n"
        "1. 메시지 대시보드 접속\n"
        "2. 소재 확인\n\n"
        "## 3. 단계별 상세 설명\n"
        "- 세부 설명 본문\n\n"
        "## 4. 실무 주의사항\n"
        "- 발송 전 비즈월렛 잔액 확인\n\n"
        "## 5. 체크리스트\n"
        "- [ ] 소재 확인\n"
        "- [ ] 발송 일자 확인\n\n"
        "## 6. 참고 근거\n"
        "- 카카오 가이드 문서\n"
        "- Slack 스레드 운영 공지\n\n"
        "## 7. 불확실한 부분\n"
        "- 실제 발송 시점은 별도 확인 필요\n"
    )
    return {
        "answer": answer,
        "sources": {
            "primary_normalized_documents": [
                {
                    "label": "kakao_guide.md · 세팅 절차 (workflow)",
                    "preview": "카카오 메시지 발송 세팅 시 ...",
                },
            ],
            "raw_evidence": [],
            "raw_fallback": [],
        },
        "diagnostics": {
            "answer_mode": "knowledge_card",
            "primary_normalized_document_count": 1,
            "raw_evidence_count": 0,
            "raw_fallback_count": 0,
            "generation_skipped": False,
            "skip_reason": None,
            "model_name": "gemini-2.5-flash-lite",
        },
        "answer_mode": "knowledge_card",
        "primary_normalized_document_count": 1,
        "raw_evidence_count": 0,
        "raw_fallback_count": 0,
    }


def _ok_qa_result() -> Dict[str, Any]:
    """qa_adapter.answer_slack_question 가 반환할 표준 형태의 fake 결과."""
    return {
        "answer": "세팅 전에는 가이드 A, 가이드 B 를 확인하세요.",
        "sources": {
            "primary_normalized_documents": [
                {
                    "label": "guide_setup.md · 사전 점검 (workflow)",
                    "preview": "사전 점검 항목 1: ...",
                },
            ],
            "raw_evidence": [
                {
                    "label": "slack_thread_001.txt · 운영 공지",
                    "preview": "운영팀: 세팅 전 ...",
                },
            ],
            "raw_fallback": [],
        },
        "diagnostics": {
            "answer_mode": "knowledge_card",
            "primary_normalized_document_count": 1,
            "raw_evidence_count": 1,
            "raw_fallback_count": 0,
            "generation_skipped": False,
            "skip_reason": None,
            "model_name": "gemini-2.5-flash-lite",
            "embedding_provider": "gemini",
            "embedding_model": "gemini-embedding-001",
            "rewritten_query": None,
            "answer_format_label": "default",
        },
        "answer_mode": "knowledge_card",
        "primary_normalized_document_count": 1,
        "raw_evidence_count": 1,
        "raw_fallback_count": 0,
    }


# ---------------------------------------------------------------------------
# strip_bot_mentions
# ---------------------------------------------------------------------------
class TestStripBotMentions:
    def test_removes_simple_mention(self) -> None:
        assert handlers.strip_bot_mentions("<@U12345> 안녕") == "안녕"

    def test_removes_named_mention(self) -> None:
        assert (
            handlers.strip_bot_mentions("<@U12345|hazzys-bot>: 질문이요")
            == "질문이요"
        )

    def test_removes_multiple_mentions(self) -> None:
        out = handlers.strip_bot_mentions(
            "<@U111> <@U222> 세팅 전에 확인해야 할 것 알려줘"
        )
        assert out == "세팅 전에 확인해야 할 것 알려줘"

    def test_handles_none_and_empty(self) -> None:
        assert handlers.strip_bot_mentions(None) == ""
        assert handlers.strip_bot_mentions("") == ""
        assert handlers.strip_bot_mentions("   ") == ""

    def test_strips_informal_prefix(self) -> None:
        assert handlers.strip_bot_mentions("<@U1> 봇아 이거 알려줘") == "이거 알려줘"

    def test_keeps_inline_mentions_clean_but_does_not_lose_content(self) -> None:
        out = handlers.strip_bot_mentions("질문 <@U1> 본문")
        # mention 은 사라지고 양쪽 텍스트는 유지된다.
        assert "<@U1>" not in out
        assert "질문" in out and "본문" in out

    def test_does_not_strip_debug_flag_dash(self) -> None:
        """``<@bot> --debug 질문`` 처럼 mention 뒤에 ``--debug`` 가 와도
        ``dash`` 가 잘려나가지 않아야 한다."""
        out = handlers.strip_bot_mentions("<@U1> --debug 질문")
        assert "--debug" in out
        assert "질문" in out


# ---------------------------------------------------------------------------
# extract_debug_flag
# ---------------------------------------------------------------------------
class TestExtractDebugFlag:
    def test_trailing_flag(self) -> None:
        q, debug = handlers.extract_debug_flag("질문 본문 --debug")
        assert q == "질문 본문"
        assert debug is True

    def test_leading_flag(self) -> None:
        q, debug = handlers.extract_debug_flag("--debug 질문 본문")
        assert q == "질문 본문"
        assert debug is True

    def test_middle_flag(self) -> None:
        q, debug = handlers.extract_debug_flag("앞 --debug 뒤")
        assert q == "앞 뒤"
        assert debug is True

    def test_no_flag(self) -> None:
        q, debug = handlers.extract_debug_flag("디버그라는 단어가 본문에 있어도")
        assert debug is False
        # 본문은 변하지 않는다.
        assert q == "디버그라는 단어가 본문에 있어도"

    def test_substring_does_not_trigger(self) -> None:
        """``--debugging`` 처럼 단어가 더 붙어 있으면 flag 로 보지 않는다."""
        q, debug = handlers.extract_debug_flag("--debugging 옵션 안내")
        assert debug is False
        assert "--debugging" in q

    def test_handles_empty(self) -> None:
        assert handlers.extract_debug_flag("") == ("", False)


# ---------------------------------------------------------------------------
# 빈 질문 / 너무 긴 질문 / allowed channel / allowed user
# ---------------------------------------------------------------------------
class TestHandlerGuards:
    def test_empty_question_returns_help(self) -> None:
        post = _FakePost()
        cfg = _make_settings()
        out = handlers.handle_app_mention(
            event={"text": "<@U1>", "channel": "C1", "user": "U2", "ts": "1.0"},
            post=post,
            settings=cfg,
            answer_fn=lambda **kw: pytest.fail("answer_fn 이 호출되면 안 된다"),
        )
        assert out["responded"] is True
        assert out["reason"] == "empty_question"
        assert len(post.calls) == 1
        assert "사용법" in post.calls[0]["text"]
        # thread 로 응답해야 한다 (reply_in_thread=True 기본값)
        assert post.calls[0]["thread_ts"] == "1.0"

    def test_disallowed_channel_is_silent(self) -> None:
        post = _FakePost()
        cfg = _make_settings(allowed_channel_ids={"C-allow-only"})
        out = handlers.handle_app_mention(
            event={
                "text": "<@U1> 질문",
                "channel": "C-other",
                "user": "U2",
                "ts": "1.0",
            },
            post=post,
            settings=cfg,
            answer_fn=lambda **kw: pytest.fail("answer_fn 이 호출되면 안 된다"),
        )
        assert out == {"responded": False, "reason": "channel_not_allowed"}
        assert post.calls == []

    def test_allowed_channel_is_passed(self) -> None:
        post = _FakePost()
        cfg = _make_settings(allowed_channel_ids={"C-ok"})
        called = {}

        def _answer(**kw: Any) -> Dict[str, Any]:
            called.update(kw)
            return _ok_qa_result()

        out = handlers.handle_app_mention(
            event={
                "text": "<@U1> 세팅 전에 확인",
                "channel": "C-ok",
                "user": "U2",
                "ts": "1.0",
            },
            post=post,
            settings=cfg,
            answer_fn=_answer,
        )
        assert out["responded"] is True
        assert out["reason"] == "ok"
        assert called["channel_id"] == "C-ok"
        assert "세팅" in called["question"]

    def test_disallowed_user_is_silent(self) -> None:
        post = _FakePost()
        cfg = _make_settings(allowed_user_ids={"U-only"})
        out = handlers.handle_app_mention(
            event={
                "text": "<@U1> 질문",
                "channel": "C1",
                "user": "U-other",
                "ts": "1.0",
            },
            post=post,
            settings=cfg,
            answer_fn=lambda **kw: pytest.fail("answer_fn 이 호출되면 안 된다"),
        )
        assert out == {"responded": False, "reason": "user_not_allowed"}
        assert post.calls == []

    def test_question_truncation_includes_notice(self) -> None:
        post = _FakePost()
        cfg = _make_settings(max_question_chars=10)
        captured: Dict[str, Any] = {}

        def _answer(**kw: Any) -> Dict[str, Any]:
            captured.update(kw)
            return _ok_qa_result()

        long_q = "가" * 50
        handlers.handle_app_mention(
            event={
                "text": f"<@U1> {long_q}",
                "channel": "C1",
                "user": "U2",
                "ts": "1.0",
            },
            post=post,
            settings=cfg,
            answer_fn=_answer,
        )
        assert len(captured["question"]) == 10
        # 안내 문구가 응답 메시지 앞에 붙어야 한다.
        assert any("질문이 너무 깁니다" in c["text"] for c in post.calls)


# ---------------------------------------------------------------------------
# formatter
# ---------------------------------------------------------------------------
class TestFormatter:
    def test_default_output_contains_answer_only(self) -> None:
        """기본 출력은 답변 본문만. 참고 근거 / 진단 블록은 표시되지 않는다."""
        text = formatter.format_qa_result(_ok_qa_result())
        assert "세팅 전에는 가이드" in text
        # 별도 참고 근거 / 진단 블록은 노출되지 않는다.
        assert "*참고 근거*" not in text
        assert "Normalized Document" not in text
        assert "Raw Evidence" not in text
        assert "*진단*" not in text
        assert "answer_mode" not in text
        # legacy 헤딩 ``*답변*`` 도 더 이상 노출되지 않는다.
        assert "*답변*" not in text
        # MVP 2차 Step 1: 기본 출력에는 retrieval diagnostics 라벨이
        # 노출되지 않아야 한다 (--debug 또는 SLACK_SHOW_DIAGNOSTICS 일 때만 노출).
        assert "query_topic" not in text
        assert "retrieved_count" not in text
        assert "topic_mismatch_count" not in text
        assert "참고 근거 (debug)" not in text

    def test_format_qa_result_handles_empty_answer(self) -> None:
        result = _ok_qa_result()
        result["answer"] = ""
        text = formatter.format_qa_result(result)
        assert "(답변이 비어 있습니다.)" in text

    def test_format_qa_result_respects_max_response_chars(self) -> None:
        result = _ok_qa_result()
        result["answer"] = "가" * 100_000
        text = formatter.format_qa_result(result, max_response_chars=300)
        # 본문은 300자 안으로 잘려야 하고, 트레일 "…" 가 붙는다.
        assert len(text) <= 320  # margin 약간
        assert text.endswith("…")

    def test_format_qa_result_truncates_long_text_within_hard_limit(self) -> None:
        result = _ok_qa_result()
        result["answer"] = "가" * 100_000
        text = formatter.format_qa_result(result, max_response_chars=999_999)
        assert len(text) <= formatter.SLACK_MESSAGE_HARD_LIMIT

    def test_help_message_mentions_usage(self) -> None:
        msg = formatter.format_help_message()
        assert "사용법" in msg

    def test_internal_error_message_does_not_leak_traceback(self) -> None:
        msg = formatter.format_internal_error_message()
        assert "Traceback" not in msg
        assert "내부 오류" in msg


# ---------------------------------------------------------------------------
# trim_answer_for_slack
# ---------------------------------------------------------------------------
class TestTrimAnswerForSlack:
    def test_trims_section_6_references(self) -> None:
        answer = (
            "## 1. 결론\n요약\n\n"
            "## 5. 체크리스트\n- 항목\n\n"
            "## 6. 참고 근거\n- 카카오 가이드\n- Slack 스레드\n"
        )
        out = formatter.trim_answer_for_slack(answer)
        assert "## 1. 결론" in out
        assert "## 5. 체크리스트" in out
        assert "참고 근거" not in out
        assert "카카오 가이드" not in out

    def test_trims_section_7_uncertain(self) -> None:
        answer = (
            "## 1. 결론\n요약\n\n"
            "## 7. 불확실한 부분\n- 실제 발송 시점은 ...\n"
        )
        out = formatter.trim_answer_for_slack(answer)
        assert "## 1. 결론" in out
        assert "불확실한 부분" not in out
        assert "실제 발송 시점은" not in out

    def test_trims_plain_references_heading(self) -> None:
        answer = (
            "## 결론\n요약\n\n"
            "## 참고 근거\n- 어떤 가이드 문서\n"
        )
        out = formatter.trim_answer_for_slack(answer)
        assert "## 결론" in out
        assert "참고 근거" not in out

    def test_trims_diagnostic_heading(self) -> None:
        answer = (
            "## 결론\n요약\n\n"
            "### 진단\n- mode: knowledge_card\n"
        )
        out = formatter.trim_answer_for_slack(answer)
        assert "결론" in out
        assert "진단" not in out
        assert "knowledge_card" not in out

    def test_trims_emphasized_references_heading(self) -> None:
        answer = (
            "*결론*\n요약 본문\n\n"
            "*참고 근거*\n- 어떤 가이드\n"
        )
        out = formatter.trim_answer_for_slack(answer)
        assert "*결론*" in out
        assert "*참고 근거*" not in out
        assert "어떤 가이드" not in out

    def test_keeps_inline_reference_words_in_body(self) -> None:
        """본문 문장 안에 "참고 근거" 라는 단어가 들어가는 경우는 잘리면 안 된다."""
        answer = (
            "## 1. 결론\n"
            "위 절차는 가이드의 참고 근거 항목을 따라 정리한 것입니다.\n"
        )
        out = formatter.trim_answer_for_slack(answer)
        assert "참고 근거" in out
        assert "결론" in out

    def test_keeps_text_when_no_trim_marker(self) -> None:
        answer = "## 1. 결론\n요약\n## 2. 순서\n1. 첫 단계\n"
        out = formatter.trim_answer_for_slack(answer)
        assert out.strip() == answer.strip()

    def test_handles_empty(self) -> None:
        assert formatter.trim_answer_for_slack("") == ""
        assert formatter.trim_answer_for_slack(None) == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# convert_markdown_headings_to_slack
# ---------------------------------------------------------------------------
class TestHeadingConversion:
    def test_converts_double_hash(self) -> None:
        assert (
            formatter.convert_markdown_headings_to_slack("## 1. 결론")
            == "*1. 결론*"
        )

    def test_converts_various_levels(self) -> None:
        out = formatter.convert_markdown_headings_to_slack(
            "# 제목\n## 1. 결론\n### 세부"
        )
        assert "*제목*" in out
        assert "*1. 결론*" in out
        assert "*세부*" in out
        assert "#" not in out

    def test_keeps_body_lines_unchanged(self) -> None:
        out = formatter.convert_markdown_headings_to_slack(
            "## 1. 결론\n결론 본문 첫 줄\n- bullet"
        )
        assert "*1. 결론*" in out
        assert "결론 본문 첫 줄" in out
        assert "- bullet" in out

    def test_does_not_touch_mid_line_hash(self) -> None:
        out = formatter.convert_markdown_headings_to_slack(
            "이것은 # 해시 문자가 있는 본문입니다."
        )
        # heading 변환이 일어나지 않아야 한다.
        assert out == "이것은 # 해시 문자가 있는 본문입니다."

    def test_strips_existing_asterisks_in_heading(self) -> None:
        out = formatter.convert_markdown_headings_to_slack("## *1. 결론*")
        assert out == "*1. 결론*"
        assert "**" not in out


# ---------------------------------------------------------------------------
# tidy_slack_text
# ---------------------------------------------------------------------------
class TestTidySlackText:
    def test_collapses_excess_blank_lines(self) -> None:
        text = "a\n\n\n\nb\n\n\n\n\nc"
        out = formatter.tidy_slack_text(text)
        assert out == "a\n\nb\n\nc"

    def test_replaces_special_whitespace(self) -> None:
        text = "결\u00a0론  요약"
        out = formatter.tidy_slack_text(text)
        # non-breaking space 가 일반 공백으로 변환되어야 한다.
        assert "\u00a0" not in out
        assert "결 론" in out


# ---------------------------------------------------------------------------
# format_qa_result : full section trim + heading 변환 + 옵션 토글
# ---------------------------------------------------------------------------
class TestFormatQaResultWithRealisticAnswer:
    def test_default_strips_sections_6_and_7_and_converts_headings(self) -> None:
        text = formatter.format_qa_result(_qa_result_with_full_sections())
        # 1~5번 섹션은 살아 있고 Slack mrkdwn 으로 변환되었다.
        assert "*1. 결론*" in text
        assert "*2. 업무 처리 순서*" in text
        assert "*3. 단계별 상세 설명*" in text
        assert "*4. 실무 주의사항*" in text
        assert "*5. 체크리스트*" in text
        # 6, 7번 섹션과 본문은 모두 제거.
        assert "참고 근거" not in text
        assert "불확실한 부분" not in text
        assert "카카오 가이드 문서" not in text
        assert "실제 발송 시점은" not in text
        # markdown heading 표식은 남아있지 않다.
        assert "## " not in text
        # 기본은 진단 블록도 미표시.
        assert "*진단*" not in text
        assert "answer_mode" not in text

    def test_debug_mode_shows_diagnostics_and_short_sources(self) -> None:
        text = formatter.format_qa_result(
            _qa_result_with_full_sections(),
            debug=True,
        )
        # 본문 trim 은 그대로 적용.
        assert "*1. 결론*" in text
        assert "## 6. 참고 근거" not in text
        # debug 에서는 진단이 표시.
        assert "*진단*" in text
        assert "answer_mode" in text
        assert "primary_normalized_document_count: 1" in text
        # debug source 요약이 노출.
        assert "참고 근거 (debug)" in text
        assert "kakao_guide.md" in text

    def test_debug_mode_shows_step1_retrieval_diagnostics(self) -> None:
        """
        MVP 2차 Step 1: --debug 모드에서 query_topic 등 retrieval diagnostics 가
        노출되어야 한다. diagnostics dict 에 신규 필드가 들어오면 진단 블록에
        ``query_topic`` 라벨이 등장한다.
        """
        result = _qa_result_with_full_sections()
        # diagnostics 에 step1 신규 필드 추가 (qa_adapter 가 채워주는 값과 동일 shape).
        result["diagnostics"].update({
            "query_topic": "kakao",
            "query_intent": ["procedure"],
            "query_date": None,
            "retrieved_count": 7,
            "passed_count": 3,
            "topic_mismatch_count": 2,
            "normalized_document_candidate_count": 1,
            "raw_candidate_count": 6,
        })
        # sources 한 항목에도 진단 필드를 채워 둔다.
        result["sources"]["primary_normalized_documents"][0].update({
            "file_name": "kakao_guide.md",
            "content_type": "knowledge_card",
            "primary_topic": "kakao",
            "retrieval_role": "primary_card",
            "final_score": 0.789,
        })

        text = formatter.format_qa_result(result, debug=True)
        # 진단 블록 신규 라벨
        assert "query_topic" in text
        assert "`kakao`" in text
        assert "retrieved_count: 7" in text
        assert "passed_count: 3" in text
        assert "topic_mismatch_count: 2" in text
        # source 진단 라인 — content_type / primary_topic / role / final
        assert "role=`primary_card`" in text or "primary_card" in text
        assert "final=`0.789`" in text

    def test_show_sources_option_enables_full_block(self) -> None:
        text = formatter.format_qa_result(
            _qa_result_with_full_sections(),
            show_sources=True,
        )
        assert "*참고 근거*" in text
        assert "Normalized Document" in text
        # show_diagnostics 는 켜지 않았으니 진단은 미표시.
        assert "*진단*" not in text

    def test_show_diagnostics_option_enables_only_diagnostics(self) -> None:
        text = formatter.format_qa_result(
            _qa_result_with_full_sections(),
            show_diagnostics=True,
        )
        assert "*진단*" in text
        assert "answer_mode" in text
        # show_sources 는 켜지 않음.
        assert "*참고 근거*" not in text


# ---------------------------------------------------------------------------
# handler 가 adapter 호출 → formatter → post 흐름 통합
# ---------------------------------------------------------------------------
class TestHandlerEndToEndWithMockAdapter:
    def test_handler_calls_adapter_and_posts_to_thread(self) -> None:
        post = _FakePost()
        cfg = _make_settings()
        captured: Dict[str, Any] = {}

        def _fake_answer(**kwargs: Any) -> Dict[str, Any]:
            captured.update(kwargs)
            return _ok_qa_result()

        out = handlers.handle_app_mention(
            event={
                "text": "<@U999> 세팅 전에 확인해야 할 것 알려줘",
                "channel": "C-public",
                "user": "U-asker",
                "ts": "1700000000.000100",
            },
            post=post,
            settings=cfg,
            answer_fn=_fake_answer,
        )
        # adapter 호출 확인
        assert captured["question"] == "세팅 전에 확인해야 할 것 알려줘"
        assert captured["user_id"] == "U-asker"
        assert captured["channel_id"] == "C-public"
        # post 호출 확인
        assert len(post.calls) == 1
        kw = post.calls[0]
        assert kw["channel"] == "C-public"
        assert kw["thread_ts"] == "1700000000.000100"
        # 기본 출력은 답변 본문만. 별도 답변/참고 근거/진단 헤더는 노출되지 않는다.
        assert "세팅 전에는 가이드" in kw["text"]
        assert "*답변*" not in kw["text"]
        assert "*참고 근거*" not in kw["text"]
        assert "*진단*" not in kw["text"]
        # 진단 dict 반환
        assert out["responded"] is True
        assert out.get("debug") is False
        assert out["answer_mode"] == "knowledge_card"
        assert out["primary_normalized_document_count"] == 1

    def test_handler_uses_thread_ts_when_present(self) -> None:
        post = _FakePost()
        cfg = _make_settings()
        handlers.handle_app_mention(
            event={
                "text": "<@U1> 질문",
                "channel": "C1",
                "user": "U1",
                "ts": "111.111",
                "thread_ts": "100.100",
            },
            post=post,
            settings=cfg,
            answer_fn=lambda **kw: _ok_qa_result(),
        )
        assert post.calls[0]["thread_ts"] == "100.100"

    def test_handler_returns_friendly_error_on_adapter_exception(self) -> None:
        post = _FakePost()
        cfg = _make_settings()

        def _boom(**kw: Any) -> Dict[str, Any]:
            raise RuntimeError("internal failure with secret context")

        out = handlers.handle_app_mention(
            event={
                "text": "<@U1> 질문",
                "channel": "C1",
                "user": "U1",
                "ts": "1.0",
            },
            post=post,
            settings=cfg,
            answer_fn=_boom,
        )
        assert out["responded"] is True
        assert out["reason"] == "qa_error"
        assert len(post.calls) == 1
        assert "내부 오류" in post.calls[0]["text"]
        # traceback / 내부 메시지가 새어나가지 않는다.
        assert "secret context" not in post.calls[0]["text"]
        assert "Traceback" not in post.calls[0]["text"]

    def test_handler_strips_debug_flag_and_enables_diagnostics(self) -> None:
        post = _FakePost()
        cfg = _make_settings()
        captured: Dict[str, Any] = {}

        def _answer(**kwargs: Any) -> Dict[str, Any]:
            captured.update(kwargs)
            return _qa_result_with_full_sections()

        out = handlers.handle_app_mention(
            event={
                "text": "<@U999> 카카오 메시지 발송 세팅 절차 알려줘 --debug",
                "channel": "C-public",
                "user": "U-asker",
                "ts": "1.0",
            },
            post=post,
            settings=cfg,
            answer_fn=_answer,
        )
        # qa_pipeline 에는 ``--debug`` 가 빠진 질문이 전달되어야 한다.
        assert "--debug" not in captured["question"]
        assert "카카오 메시지 발송 세팅 절차" in captured["question"]
        # debug=True 로 처리되었다.
        assert out["debug"] is True
        # Slack 메시지에는 진단 정보와 짧은 source 요약이 포함된다.
        text = post.calls[0]["text"]
        assert "*1. 결론*" in text
        assert "*진단*" in text
        assert "answer_mode" in text
        assert "참고 근거 (debug)" in text
        # 본문의 6/7번 섹션은 여전히 잘려나가야 한다.
        assert "## 6. 참고 근거" not in text
        assert "## 7. 불확실한 부분" not in text


# ---------------------------------------------------------------------------
# token 누락 시 graceful failure
# ---------------------------------------------------------------------------
class TestConfigValidation:
    def test_missing_bot_token_reports_problem(self) -> None:
        cfg = SlackBotSettings(
            enabled=True, mode="socket",
            bot_token=None, app_token="xapp-real",
        )
        ok, problems = cfg.validate()
        assert ok is False
        assert any("SLACK_BOT_TOKEN" in p for p in problems)

    def test_placeholder_bot_token_is_treated_as_missing(self) -> None:
        cfg = SlackBotSettings(
            enabled=True, mode="socket",
            bot_token="xoxb-your-bot-token",
            app_token="xapp-your-app-token",
        )
        ok, problems = cfg.validate()
        assert ok is False
        assert any("SLACK_BOT_TOKEN" in p for p in problems)
        assert any("SLACK_APP_TOKEN" in p for p in problems)

    def test_disabled_bot_reports_problem(self) -> None:
        cfg = SlackBotSettings(
            enabled=False, mode="socket",
            bot_token="xoxb-real", app_token="xapp-real",
        )
        ok, problems = cfg.validate()
        assert ok is False
        assert any("SLACK_BOT_ENABLED" in p for p in problems)

    def test_unsupported_mode_reports_problem(self) -> None:
        cfg = SlackBotSettings(
            enabled=True, mode="http",
            bot_token="xoxb-real", app_token="xapp-real",
        )
        ok, problems = cfg.validate()
        assert ok is False
        assert any("SLACK_BOT_MODE" in p for p in problems)

    def test_valid_config_passes(self) -> None:
        cfg = SlackBotSettings(
            enabled=True, mode="socket",
            bot_token="xoxb-real-1234", app_token="xapp-real-1234",
        )
        ok, problems = cfg.validate()
        assert ok is True
        assert problems == []

    def test_load_settings_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.slack_bot import config as cfg_mod

        monkeypatch.setenv("SLACK_BOT_ENABLED", "true")
        monkeypatch.setenv("SLACK_BOT_MODE", "socket")
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-real-token")
        monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-real-token")
        monkeypatch.setenv("SLACK_ALLOWED_CHANNEL_IDS", "C1, C2 ;C3")
        monkeypatch.setenv("SLACK_ALLOWED_USER_IDS", "")
        monkeypatch.setenv("SLACK_REPLY_IN_THREAD", "true")
        monkeypatch.setenv("SLACK_MAX_QUESTION_CHARS", "777")
        monkeypatch.setenv("SLACK_SHOW_SOURCES", "true")
        monkeypatch.setenv("SLACK_SHOW_DIAGNOSTICS", "true")
        monkeypatch.setenv("SLACK_MAX_RESPONSE_CHARS", "1234")

        s = cfg_mod.load_settings()
        assert s.enabled is True
        assert s.bot_token == "xoxb-real-token"
        assert s.app_token == "xapp-real-token"
        assert s.allowed_channel_ids == {"C1", "C2", "C3"}
        assert s.allowed_user_ids == set()
        assert s.max_question_chars == 777
        assert s.show_sources is True
        assert s.show_diagnostics is True
        assert s.max_response_chars == 1234


# ---------------------------------------------------------------------------
# allowed channel/user 체크 메서드 단위
# ---------------------------------------------------------------------------
class TestAllowChecks:
    def test_empty_allow_list_allows_all(self) -> None:
        cfg = _make_settings()
        assert cfg.is_channel_allowed("C-anything") is True
        assert cfg.is_user_allowed("U-anything") is True
        # None 도 허용 (실제 운영에서는 거의 발생 X)
        assert cfg.is_channel_allowed(None) is True
        assert cfg.is_user_allowed(None) is True

    def test_channel_allow_list_filters(self) -> None:
        cfg = _make_settings(allowed_channel_ids={"C1", "C2"})
        assert cfg.is_channel_allowed("C1") is True
        assert cfg.is_channel_allowed("C99") is False
        assert cfg.is_channel_allowed(None) is False

    def test_user_allow_list_filters(self) -> None:
        cfg = _make_settings(allowed_user_ids={"U1"})
        assert cfg.is_user_allowed("U1") is True
        assert cfg.is_user_allowed("U2") is False
        assert cfg.is_user_allowed(None) is False


# ---------------------------------------------------------------------------
# scripts/run_slack_bot.py 가 SLACK_BOT_ENABLED=false 에서 graceful 종료
# ---------------------------------------------------------------------------
class TestRunSlackBotEntrypoint:
    def _load_module(self):
        # scripts/ 는 패키지가 아니므로 importlib.util 로 직접 로드.
        import importlib.util
        root = Path(__file__).resolve().parents[1]
        path = root / "scripts" / "run_slack_bot.py"
        spec = importlib.util.spec_from_file_location("run_slack_bot_mod", path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_disabled_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        monkeypatch.setenv("SLACK_BOT_ENABLED", "false")
        mod = self._load_module()
        rc = mod.main()
        assert rc == 0
        out = capsys.readouterr().out
        assert "SLACK_BOT_ENABLED=false" in out

    def test_missing_tokens_returns_two(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        monkeypatch.setenv("SLACK_BOT_ENABLED", "true")
        monkeypatch.setenv("SLACK_BOT_MODE", "socket")
        monkeypatch.setenv("SLACK_BOT_TOKEN", "")
        monkeypatch.setenv("SLACK_APP_TOKEN", "")
        mod = self._load_module()
        rc = mod.main()
        assert rc == 2
        out = capsys.readouterr().out
        assert "SLACK_BOT_TOKEN" in out
        assert "SLACK_APP_TOKEN" in out
