"""tests 가 src 를 import 할 수 있도록 sys.path 에 프로젝트 루트 추가."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
