"""
src.slack_bot.config
====================
Slack QA Bot 전용 환경변수 로더.

기존 ``src.config.settings`` 와는 분리해서 관리한다 — Slack 관련 설정은
Slack Bot 프로세스에서만 의미가 있고, Streamlit/CLI 흐름에는 영향을 주지
않는다.

읽는 환경변수
-------------
- ``SLACK_BOT_ENABLED``        : 봇 활성화 여부 (기본 ``false``)
- ``SLACK_BOT_MODE``           : 동작 모드. 현재는 ``"socket"`` 만 지원.
- ``SLACK_BOT_TOKEN``          : Bot User OAuth Token (``xoxb-...``)
- ``SLACK_APP_TOKEN``          : App-Level Token (``xapp-...``,
                                 scope=``connections:write``)
- ``SLACK_ALLOWED_CHANNEL_IDS``: 허용 채널 ID (콤마 구분). 비어 있으면
                                 모든 채널 허용.
- ``SLACK_ALLOWED_USER_IDS``   : 허용 유저 ID (콤마 구분). 비어 있으면
                                 모든 유저 허용.
- ``SLACK_REPLY_IN_THREAD``    : 항상 thread 로 답변할지 여부 (기본 ``true``).
- ``SLACK_MAX_QUESTION_CHARS`` : 한 번에 처리할 질문의 최대 글자수 (기본 1000).
- ``SLACK_SHOW_SOURCES``       : 기본 출력에 참고 근거 블록 포함 여부 (기본 ``false``).
                                 ``--debug`` 모드는 이 값과 무관하게 짧은 source
                                 요약을 표시한다.
- ``SLACK_SHOW_DIAGNOSTICS``   : 기본 출력에 진단 블록 포함 여부 (기본 ``false``).
                                 ``--debug`` 모드는 이 값과 무관하게 진단을 표시한다.
- ``SLACK_MAX_RESPONSE_CHARS`` : Slack 답변 메시지의 최대 글자수 (기본 ``2500``).
                                 초과 시 답변 본문 끝을 잘라낸다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# 내부 helper
# ---------------------------------------------------------------------------
def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    raw = os.getenv(key, default)
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env(key)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _env_int(key: str, default: int) -> int:
    raw = _env(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _split_ids(raw: Optional[str]) -> Set[str]:
    """콤마/공백 구분 ID 문자열을 set 으로 변환."""
    if not raw:
        return set()
    out: Set[str] = set()
    for token in raw.replace(";", ",").split(","):
        token = token.strip()
        if token:
            out.add(token)
    return out


# placeholder 로 쓰이는 값들 (예: .env.example 그대로 둔 경우).
# 이런 값이 들어 있으면 token 이 "설정되지 않은" 것으로 본다.
_BOT_TOKEN_PLACEHOLDERS = {
    "xoxb-your-bot-token",
    "xoxb-",
}
_APP_TOKEN_PLACEHOLDERS = {
    "xapp-your-app-token",
    "xapp-",
}


def _is_placeholder(value: Optional[str], placeholders: Iterable[str]) -> bool:
    if not value:
        return True
    v = value.strip().lower()
    if not v:
        return True
    return v in {p.lower() for p in placeholders}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
@dataclass
class SlackBotSettings:
    """Slack Bot 동작에 필요한 설정 묶음."""

    enabled: bool = False
    mode: str = "socket"
    bot_token: Optional[str] = None
    app_token: Optional[str] = None
    allowed_channel_ids: Set[str] = field(default_factory=set)
    allowed_user_ids: Set[str] = field(default_factory=set)
    reply_in_thread: bool = True
    max_question_chars: int = 1000
    # ----- 출력 가공 옵션 (Slack formatter 가 사용) -----
    show_sources: bool = False
    show_diagnostics: bool = False
    max_response_chars: int = 2500

    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> "SlackBotSettings":
        s = cls()
        s.enabled = _env_bool("SLACK_BOT_ENABLED", s.enabled)
        s.mode = (_env("SLACK_BOT_MODE", s.mode) or s.mode).lower()
        s.bot_token = _env("SLACK_BOT_TOKEN")
        s.app_token = _env("SLACK_APP_TOKEN")
        s.allowed_channel_ids = _split_ids(_env("SLACK_ALLOWED_CHANNEL_IDS"))
        s.allowed_user_ids = _split_ids(_env("SLACK_ALLOWED_USER_IDS"))
        s.reply_in_thread = _env_bool("SLACK_REPLY_IN_THREAD", s.reply_in_thread)
        s.max_question_chars = _env_int(
            "SLACK_MAX_QUESTION_CHARS", s.max_question_chars
        )
        if s.max_question_chars <= 0:
            s.max_question_chars = 1000
        s.show_sources = _env_bool("SLACK_SHOW_SOURCES", s.show_sources)
        s.show_diagnostics = _env_bool("SLACK_SHOW_DIAGNOSTICS", s.show_diagnostics)
        s.max_response_chars = _env_int(
            "SLACK_MAX_RESPONSE_CHARS", s.max_response_chars
        )
        if s.max_response_chars <= 0:
            s.max_response_chars = 2500
        return s

    # ------------------------------------------------------------------
    # 토큰/모드 검증
    # ------------------------------------------------------------------
    def has_bot_token(self) -> bool:
        return not _is_placeholder(self.bot_token, _BOT_TOKEN_PLACEHOLDERS)

    def has_app_token(self) -> bool:
        return not _is_placeholder(self.app_token, _APP_TOKEN_PLACEHOLDERS)

    def is_socket_mode(self) -> bool:
        return self.mode == "socket"

    def validate(self) -> Tuple[bool, List[str]]:
        """
        실행 가능 여부를 검사한다.

        Returns
        -------
        (ok, problems)
            ``ok`` 가 False 면 ``problems`` 에 한국어 안내 메시지가 들어 있다.
        """
        problems: List[str] = []
        if not self.enabled:
            problems.append(
                "SLACK_BOT_ENABLED=false 입니다. .env 에서 SLACK_BOT_ENABLED=true 로 "
                "변경한 뒤 다시 실행하세요."
            )
        if not self.is_socket_mode():
            problems.append(
                f"지원하지 않는 SLACK_BOT_MODE='{self.mode}' 입니다. "
                "현재는 'socket' 모드만 지원합니다."
            )
        if not self.has_bot_token():
            problems.append(
                "SLACK_BOT_TOKEN 이 설정되지 않았습니다. "
                "Slack App → OAuth & Permissions 에서 발급한 'xoxb-...' 토큰을 "
                ".env 의 SLACK_BOT_TOKEN 에 입력하세요."
            )
        if not self.has_app_token():
            problems.append(
                "SLACK_APP_TOKEN 이 설정되지 않았습니다. "
                "Slack App → Basic Information → App-Level Tokens 에서 "
                "scope=connections:write 로 발급한 'xapp-...' 토큰을 "
                ".env 의 SLACK_APP_TOKEN 에 입력하세요."
            )
        return (not problems, problems)

    # ------------------------------------------------------------------
    # allowed channel/user 체크
    # ------------------------------------------------------------------
    def is_channel_allowed(self, channel_id: Optional[str]) -> bool:
        """
        ``allowed_channel_ids`` 가 비어 있으면 모든 채널을 허용한다.
        값이 있으면 정확히 일치하는 채널만 허용한다.
        """
        if not self.allowed_channel_ids:
            return True
        if not channel_id:
            return False
        return channel_id in self.allowed_channel_ids

    def is_user_allowed(self, user_id: Optional[str]) -> bool:
        """
        ``allowed_user_ids`` 가 비어 있으면 모든 사용자를 허용한다.
        값이 있으면 정확히 일치하는 사용자만 허용한다.
        """
        if not self.allowed_user_ids:
            return True
        if not user_id:
            return False
        return user_id in self.allowed_user_ids


def load_settings() -> SlackBotSettings:
    """현재 환경변수 기준으로 ``SlackBotSettings`` 를 새로 만든다."""
    return SlackBotSettings.from_env()
