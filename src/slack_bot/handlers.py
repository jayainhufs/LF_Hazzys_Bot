"""
src.slack_bot.handlers
======================
Slack ``app_mention`` 이벤트 핸들러.

이 모듈은 Slack 클라이언트 / API 자체에는 의존하지 않는다 — 모든 외부
호출은 ``post`` callable (``say`` 또는 ``client.chat_postMessage`` 류) 로
주입받는다. 덕분에 단위 테스트에서 fake post 와 fake QAPipeline 만으로
전체 흐름을 검증할 수 있다.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, Optional, Tuple

from src.logger import get_logger
from src.slack_bot import formatter
from src.slack_bot.config import SlackBotSettings, load_settings
from src.slack_bot.qa_adapter import answer_slack_question

log = get_logger(__name__)

# Slack mention 패턴: <@U12345>, <@W12345|name> 등
_MENTION_PATTERN = re.compile(r"<@([UW][A-Z0-9]+)(\|[^>]*)?>")
# 그 외에 흔한 prefix 한국어 호출 ("봇아", "봇:", "봇 ")
_INFORMAL_PREFIXES = ("봇아", "봇:", "Bot:", "bot:")
# 질문 끝에 ``--debug`` 가 붙으면 디버그 모드로 동작한다. 위치는 어디든 OK.
# (앞/뒤/중간 토큰)
_DEBUG_FLAG_PATTERN = re.compile(r"(?:^|\s)--debug\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# 공개 helper — 단위 테스트가 직접 호출한다.
# ---------------------------------------------------------------------------
def strip_bot_mentions(text: Optional[str]) -> str:
    """
    Slack 메시지 텍스트에서 ``<@U...>`` 형식의 mention 을 모두 제거하고
    앞뒤 공백/구두점을 정리한다.

    Examples
    --------
    >>> strip_bot_mentions("<@U123> 안녕")
    '안녕'
    >>> strip_bot_mentions("<@U123|name>: 질문이요")
    '질문이요'
    >>> strip_bot_mentions(None)
    ''
    """
    if not text:
        return ""
    cleaned = _MENTION_PATTERN.sub(" ", text)
    cleaned = cleaned.replace("\u200b", "").strip()
    # mention 직후의 콜론/세미콜론/꺾쇠 등 정리.
    # NOTE: ``--debug`` 같이 사용자가 의도해 보낸 flag 가 잘려나가지 않도록
    # ``-`` (dash) 는 leading 정리 대상에서 의도적으로 제외한다.
    cleaned = re.sub(r"^[\s,.:;>]+", "", cleaned)
    cleaned = re.sub(r"[\s]+", " ", cleaned).strip()
    for pfx in _INFORMAL_PREFIXES:
        if cleaned.lower().startswith(pfx.lower()):
            cleaned = cleaned[len(pfx):].lstrip(" ,.:;>")
            break
    return cleaned


def clip_question(text: str, max_chars: int) -> str:
    """``max_chars`` 가 양수면 그 글자수까지 잘라낸다."""
    if not text:
        return ""
    if max_chars and max_chars > 0 and len(text) > max_chars:
        return text[:max_chars]
    return text


def extract_debug_flag(text: str) -> Tuple[str, bool]:
    """
    질문 텍스트에서 ``--debug`` 플래그를 떼어낸다.

    Returns
    -------
    (clean_question, debug_on)
        - ``--debug`` 가 들어 있었다면 해당 토큰을 제거한 질문과 ``True``.
        - 없었다면 원본과 ``False``.

    본문 중간에 ``--debug`` 가 있어도 떼어낸다 — 단어 경계만 일치하면 된다.
    예: ``"카카오 메시지 발송 세팅 절차 알려줘 --debug"``
       → ``("카카오 메시지 발송 세팅 절차 알려줘", True)``
    """
    if not text:
        return ("", False)
    if not _DEBUG_FLAG_PATTERN.search(text):
        return (text, False)
    cleaned = _DEBUG_FLAG_PATTERN.sub(" ", text)
    cleaned = re.sub(r"[\s]+", " ", cleaned).strip()
    return (cleaned, True)


# ---------------------------------------------------------------------------
# 메인 핸들러
# ---------------------------------------------------------------------------
def handle_app_mention(
    event: Dict[str, Any],
    *,
    post: Callable[..., Any],
    settings: Optional[SlackBotSettings] = None,
    answer_fn: Callable[..., Dict[str, Any]] = answer_slack_question,
) -> Dict[str, Any]:
    """
    Slack ``app_mention`` 이벤트를 처리한다.

    Parameters
    ----------
    event : dict
        Slack 이벤트 payload. 최소한 ``text``, ``channel``, ``user`` 가
        필요하며 ``ts`` / ``thread_ts`` 가 thread 답변에 사용된다.
    post : callable
        Slack 으로 메시지를 보낼 함수. ``say`` 또는
        ``client.chat_postMessage`` 와 호환되는 시그니처를 가정한다.
        ``post(text=..., channel=..., thread_ts=...)`` 형태로 호출된다
        (불필요한 인자는 callable 쪽에서 무시되도록 ``**kwargs`` 권장).
    settings : SlackBotSettings, optional
        주입하지 않으면 환경변수에서 새로 로드한다 (테스트에서 명시적으로
        주입 가능).
    answer_fn : callable, optional
        ``(question, user_id, channel_id) -> dict`` 형태의 함수.
        기본값은 ``qa_adapter.answer_slack_question``. 테스트에서 mock
        가능하다.

    Returns
    -------
    dict
        진단용 dict. 실제 Slack 응답 여부 (``responded``) 와 사유
        (``reason``) 를 담는다. 테스트가 검증할 때 사용한다.
    """
    cfg = settings if settings is not None else load_settings()

    channel_id: Optional[str] = event.get("channel")
    user_id: Optional[str] = event.get("user")
    raw_text: str = event.get("text") or ""
    # thread_ts 가 있으면 기존 thread 에, 없으면 메시지 ts 를 사용해 새 thread 시작.
    parent_ts: Optional[str] = event.get("thread_ts") or event.get("ts")

    # 1) 허용 채널/유저 체크 (block 시 조용히 종료 — 채널을 어지럽히지 않는다)
    if not cfg.is_channel_allowed(channel_id):
        log.info(
            "Slack handler: channel not allowed (channel_id=%s) → silent skip",
            channel_id,
        )
        return {"responded": False, "reason": "channel_not_allowed"}
    if not cfg.is_user_allowed(user_id):
        log.info(
            "Slack handler: user not allowed (user_id=%s) → silent skip",
            user_id,
        )
        return {"responded": False, "reason": "user_not_allowed"}

    # 2) mention 제거 + 질문 정제
    question = strip_bot_mentions(raw_text)

    # 2-1) ``--debug`` 플래그 분리 (qa_pipeline 으로는 보내지 않음)
    question, debug = extract_debug_flag(question)

    # 3) 질문이 비어 있으면 사용법 안내
    if not question:
        _safe_post(
            post,
            text=formatter.format_help_message(),
            channel=channel_id,
            thread_ts=parent_ts if cfg.reply_in_thread else None,
        )
        return {"responded": True, "reason": "empty_question"}

    # 4) 질문 길이 제한
    truncated = False
    if len(question) > cfg.max_question_chars:
        question = clip_question(question, cfg.max_question_chars)
        truncated = True

    # 5) qa_pipeline adapter 호출
    try:
        result = answer_fn(
            question=question,
            user_id=user_id,
            channel_id=channel_id,
        )
    except Exception:  # noqa: BLE001
        log.exception("Slack handler: QA pipeline raised an exception")
        _safe_post(
            post,
            text=formatter.format_internal_error_message(),
            channel=channel_id,
            thread_ts=parent_ts if cfg.reply_in_thread else None,
        )
        return {"responded": True, "reason": "qa_error"}

    # 6) Slack 메시지로 변환
    # 기본 정책:
    #  - 운영자가 ``SLACK_SHOW_SOURCES=true`` / ``SLACK_SHOW_DIAGNOSTICS=true`` 로
    #    설정해 두었으면 항상 표시.
    #  - 그렇지 않으면 사용자가 질문에 ``--debug`` 를 붙였을 때만 표시.
    show_sources = True if cfg.show_sources else (True if debug else None)
    show_diagnostics = True if cfg.show_diagnostics else (True if debug else None)
    body = formatter.format_qa_result(
        result,
        question=question,
        debug=debug,
        show_sources=show_sources,
        show_diagnostics=show_diagnostics,
        max_response_chars=cfg.max_response_chars,
    )
    if truncated:
        body = (
            formatter.format_too_long_message(cfg.max_question_chars)
            + "\n\n"
            + body
        )

    # 7) thread 에 post
    _safe_post(
        post,
        text=body,
        channel=channel_id,
        thread_ts=parent_ts if cfg.reply_in_thread else None,
    )
    return {
        "responded": True,
        "reason": "ok",
        "debug": debug,
        "answer_mode": result.get("answer_mode"),
        "primary_normalized_document_count": result.get(
            "primary_normalized_document_count"
        ),
    }


# ---------------------------------------------------------------------------
# 내부 helper
# ---------------------------------------------------------------------------
def _safe_post(
    post: Callable[..., Any],
    *,
    text: str,
    channel: Optional[str],
    thread_ts: Optional[str],
) -> None:
    """
    Slack 으로 텍스트 메시지를 안전하게 보낸다.

    - ``say`` 는 ``channel`` 인자가 필수가 아니지만 ``client.chat_postMessage``
      는 필수다. 양쪽 호환을 위해 둘 다 전달한다.
    - thread_ts 가 None 이면 thread 인자 자체를 빼서 호출한다 (Slack API 가
      None 을 거부하는 경우를 피한다).
    - post 호출 실패는 traceback 을 사용자에게 노출하지 않고 로그만 남긴다.
    """
    kwargs: Dict[str, Any] = {"text": text}
    if channel:
        kwargs["channel"] = channel
    if thread_ts:
        kwargs["thread_ts"] = thread_ts
    try:
        post(**kwargs)
    except Exception:  # noqa: BLE001
        log.exception("Slack handler: failed to post message")
