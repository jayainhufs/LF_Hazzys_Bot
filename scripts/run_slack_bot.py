"""
run_slack_bot.py
================
Slack QA Bot 실행 진입점.

사용법
------
    python scripts/run_slack_bot.py

기본 동작
---------
- ``.env`` 의 ``SLACK_BOT_ENABLED`` 가 false 면 안내 메시지 출력 후 종료.
- ``SLACK_BOT_TOKEN`` 또는 ``SLACK_APP_TOKEN`` 이 누락된 경우 친절한 안내
  메시지를 출력하고 비정상 종료 (exit code 2).
- 모두 OK 면 Socket Mode 로 Bolt App 을 시작한다 (Ctrl+C 로 종료).

이 스크립트는 기존 Streamlit 앱을 대체하지 않는다. Streamlit 은 관리자/
운영 콘솔 (문서 업로드/색인/정규화 문서 관리/검색 테스트/API 상태 확인)
역할을 그대로 유지한다.
"""
from __future__ import annotations

# sys.path 보정 (직접 실행 시에도 src 를 import 가능하게)
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import sys


def _print_banner() -> None:
    print("=" * 60)
    print("Slack QA Bot 실행기")
    print("=" * 60)


def main() -> int:
    _print_banner()

    # 환경변수 로딩은 src.slack_bot.config 가 담당.
    # src.config 도 import 시점에 .env 를 자동으로 load_dotenv 한다.
    from src.config import settings as _global_settings  # noqa: F401  (.env 로드 트리거)
    from src.slack_bot.config import load_settings

    cfg = load_settings()

    if not cfg.enabled:
        print(
            "[INFO] SLACK_BOT_ENABLED=false 로 설정되어 있어 Slack Bot 을 시작하지 않습니다.\n"
            "       .env 에서 SLACK_BOT_ENABLED=true 로 변경한 뒤 다시 실행하세요."
        )
        return 0

    ok, problems = cfg.validate()
    if not ok:
        print("[ERROR] Slack Bot 설정에 문제가 있어 시작할 수 없습니다.")
        for p in problems:
            print(f"   - {p}")
        print(
            "\nSlack App 설정 가이드는 README 의 'Slack QA Bot (선택 기능)' 절을 참고하세요."
        )
        return 2

    print("[OK] 설정 점검 완료. Socket Mode 로 Slack Bot 을 시작합니다.")
    print(f"     - allowed_channel_ids: {sorted(cfg.allowed_channel_ids) or '(전체 허용)'}")
    print(f"     - allowed_user_ids:    {sorted(cfg.allowed_user_ids) or '(전체 허용)'}")
    print(f"     - reply_in_thread:     {cfg.reply_in_thread}")
    print(f"     - max_question_chars:  {cfg.max_question_chars}")

    try:
        from src.slack_bot.app import (
            SlackBoltNotInstalledError,
            run_socket_mode,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] Slack Bot 모듈 import 실패: {type(e).__name__}: {e}")
        return 2

    try:
        run_socket_mode(cfg)
    except SlackBoltNotInstalledError as e:
        print(f"[ERROR] {e}")
        return 2
    except KeyboardInterrupt:
        print("\n[INFO] 사용자 입력으로 종료합니다.")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] Slack Bot 실행 중 오류: {type(e).__name__}: {e}")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
