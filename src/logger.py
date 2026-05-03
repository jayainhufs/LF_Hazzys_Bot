"""
logger.py
=========
프로젝트 공통 로거.

- 모든 모듈은 `from src.logger import get_logger` 후
  `log = get_logger(__name__)` 로 사용한다.
- 한 번만 핸들러를 등록해서 streamlit/CLI 어디서든 로그가 정상 출력되게 한다.
"""
from __future__ import annotations

import logging
import os
import sys

_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

_root_configured = False


def _configure_root() -> None:
    global _root_configured
    if _root_configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FMT))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(_LEVEL)
    _root_configured = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)
