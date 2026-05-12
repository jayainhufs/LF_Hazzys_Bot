"""
src.slack_bot.app
=================
Slack Bolt App 빌더 + Socket Mode 실행 헬퍼.

이 모듈은 ``slack_bolt`` 가 설치된 경우에만 의미가 있다. 단위 테스트가
``slack_bolt`` 없이도 패키지를 import 할 수 있도록 모든 ``slack_bolt`` import
는 함수 안에서 lazy 하게 수행한다.
"""
from __future__ import annotations

from typing import Any, Optional

from src.logger import get_logger
from src.slack_bot.config import SlackBotSettings, load_settings
from src.slack_bot.handlers import handle_app_mention

log = get_logger(__name__)


class SlackBoltNotInstalledError(RuntimeError):
    """``slack_bolt`` 패키지가 설치되어 있지 않을 때 발생."""


def _import_slack_bolt():
    """slack_bolt 를 lazy import. 미설치 시 친절한 에러로 변환."""
    try:
        from slack_bolt import App  # type: ignore
        from slack_bolt.adapter.socket_mode import SocketModeHandler  # type: ignore
    except ImportError as e:  # pragma: no cover - 실제 환경에 종속
        raise SlackBoltNotInstalledError(
            "slack-bolt 패키지가 설치되어 있지 않습니다. "
            "`pip install -r requirements.txt` 또는 `pip install slack-bolt` 를 실행하세요."
        ) from e
    return App, SocketModeHandler


def build_app(settings: Optional[SlackBotSettings] = None) -> Any:
    """
    Slack Bolt ``App`` 인스턴스를 만들고 ``app_mention`` 핸들러를 등록한다.

    Parameters
    ----------
    settings : SlackBotSettings, optional
        주입하지 않으면 환경변수에서 새로 로드한다.
    """
    cfg = settings if settings is not None else load_settings()
    if not cfg.has_bot_token():
        raise RuntimeError(
            "SLACK_BOT_TOKEN 이 비어 있습니다. .env 의 SLACK_BOT_TOKEN 을 채워 주세요."
        )

    App, _ = _import_slack_bolt()
    app = App(token=cfg.bot_token)

    @app.event("app_mention")
    def _on_app_mention(event, say, logger):  # noqa: ANN001
        # say 는 자동으로 같은 채널/thread_ts 에 응답하지만, 우리는 일관된
        # 동작을 위해 handlers._safe_post 가 channel/thread_ts 를 명시한다.
        try:
            handle_app_mention(event, post=say, settings=cfg)
        except Exception:  # noqa: BLE001
            logger.exception("Slack app_mention 처리 중 예상치 못한 오류")

    return app


def run_socket_mode(settings: Optional[SlackBotSettings] = None) -> None:
    """
    Socket Mode 로 Slack Bot 을 기동한다 (현재 유일하게 지원하는 모드).

    동일 token 으로 여러 PC 에서 동시에 호출하면 안 된다 — Slack 측에서
    중복 연결로 처리되어 메시지가 누락될 수 있다.
    """
    cfg = settings if settings is not None else load_settings()

    ok, problems = cfg.validate()
    if not ok:
        for p in problems:
            log.error("Slack Bot 설정 오류: %s", p)
        raise RuntimeError(
            "Slack Bot 을 시작할 수 없습니다. .env 설정을 확인해 주세요."
        )
    if not cfg.is_socket_mode():
        raise RuntimeError(
            f"현재 SLACK_BOT_MODE='{cfg.mode}' 는 지원되지 않습니다. "
            "이번 MVP 는 'socket' 모드만 지원합니다."
        )

    app = build_app(cfg)
    _, SocketModeHandler = _import_slack_bolt()
    log.info("Slack Bot Socket Mode 시작 — Ctrl+C 로 종료")
    handler = SocketModeHandler(app, cfg.app_token)
    handler.start()
