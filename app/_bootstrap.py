"""
_bootstrap.py
=============
Streamlit page 들이 `from src...` 를 import 하기 전에
프로젝트 루트를 sys.path 에 추가해 주는 헬퍼.

각 page 첫 줄에서 `import app._bootstrap  # noqa: F401` 로 import 한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
