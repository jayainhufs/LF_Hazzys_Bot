"""
ingest_folder.py
================
data/raw 하위의 모든 파일을 색인하는 CLI.

사용법:
    python scripts/ingest_folder.py
    python scripts/ingest_folder.py --enable-summary
    python scripts/ingest_folder.py --no-skip
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse

from src.logger import get_logger
from src.pipeline import ingest_folder
from src.utils.cost_utils import tracker

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enable-summary",
        action="store_true",
        help="Excel 한국어 요약을 함께 생성한다 (비용 발생).",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="이미 색인된 파일도 다시 색인한다.",
    )
    args = parser.parse_args()

    summary = ingest_folder(
        enable_excel_summary=args.enable_summary,
        skip_if_indexed=not args.no_skip,
    )

    print("=" * 60)
    print(
        f"total={summary.total} indexed={summary.indexed} "
        f"skipped={summary.skipped} failed={summary.failed} "
        f"chunks_added={summary.chunks_added_total}"
    )
    print("tracker:", tracker.snapshot())
    print("=" * 60)
    if summary.failed > 0:
        print("실패한 파일:")
        for r in summary.results:
            if r.status == "failed":
                print(f"  - {r.file_name} :: {r.error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
