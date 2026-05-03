"""
scripts/_bootstrap.py
=====================
scripts/ 안의 CLI 스크립트들이 `from src ...` 를 import 할 수 있도록
sys.path 에 프로젝트 루트를 추가한다.

`python scripts/foo.py` 형태로 직접 실행될 수도, `python -m scripts.foo` 로
실행될 수도 있어 두 경우 모두 동작하도록 inline 으로 sys.path 를 보정한다.

사용법 (각 스크립트 첫 줄):

    from scripts._bootstrap import ensure_path; ensure_path()

또는 더 단순하게 import 만으로 sys.path 를 보정하고 싶다면 본 모듈을 import 하면
부수효과로 sys.path 가 보정된다.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def ensure_path() -> Path:
    return _ROOT
