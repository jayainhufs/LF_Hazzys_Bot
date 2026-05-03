"""
reset_vector_db.py
==================
Vector DB / file_registry / processed JSON 을 초기화한다.
원본 파일(data/raw)은 건드리지 않는다.

사용법:
    python scripts/reset_vector_db.py            # 확인 프롬프트
    python scripts/reset_vector_db.py --yes      # 즉시 실행
    python scripts/reset_vector_db.py --keep-registry  # registry 는 유지
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import shutil

from src.config import settings
from src.logger import get_logger
from src.storage.file_registry import FileRegistry
from src.storage.vector_store import VectorStore

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="확인 프롬프트 없이 실행")
    parser.add_argument("--keep-registry", action="store_true", help="indexed_files.json 은 유지")
    parser.add_argument(
        "--keep-processed",
        action="store_true",
        help="data/processed (documents, chunks, summaries) 는 유지",
    )
    args = parser.parse_args()

    print("== reset 대상 ==")
    print(f"- ChromaDB           : {settings.chroma_db_dir}")
    if not args.keep_registry:
        print(f"- File Registry      : {settings.registry_path}")
    if not args.keep_processed:
        print(f"- Processed (chunks) : {settings.chunks_dir}")
        print(f"- Processed (docs)   : {settings.documents_dir}")
        print(f"- Processed (excel)  : {settings.excel_summary_dir}")

    if not args.yes:
        ans = input("정말 초기화하시겠습니까? (y/N): ").strip().lower()
        if ans != "y":
            print("취소.")
            return 0

    # 1) Chroma 컬렉션 reset
    try:
        VectorStore().reset_db()
        print("OK · ChromaDB collection reset")
    except Exception as e:  # noqa: BLE001
        log.error("ChromaDB reset 실패: %s", e)

    # 2) file_registry
    if not args.keep_registry:
        try:
            FileRegistry().reset()
            print("OK · file_registry reset")
        except Exception as e:  # noqa: BLE001
            log.error("registry reset 실패: %s", e)

    # 3) processed
    if not args.keep_processed:
        for d in [settings.documents_dir, settings.chunks_dir, settings.excel_summary_dir]:
            try:
                if d.exists():
                    for child in d.iterdir():
                        if child.is_file() or child.is_symlink():
                            child.unlink()
                        elif child.is_dir():
                            shutil.rmtree(child)
                    print(f"OK · cleared {d}")
            except Exception as e:  # noqa: BLE001
                log.error("디렉터리 정리 실패: %s (%s)", d, e)

    print("완료.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
