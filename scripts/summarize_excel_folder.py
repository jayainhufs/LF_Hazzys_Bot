"""
summarize_excel_folder.py
=========================
data/raw/excel/ 하위의 모든 .xlsx/.xlsm 파일을 시트 단위로 한국어 요약한다.
이미 raw_table_hash 가 같은 요약이 있으면 skip.

사용법:
    python scripts/summarize_excel_folder.py
    python scripts/summarize_excel_folder.py --force
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from src.config import settings
from src.ingestion.excel_parser import parse_excel
from src.logger import get_logger
from src.rag.gemini_client import GeminiError
from src.summarization.excel_summarizer import ExcelSummarizer
from src.utils.hash_utils import file_hash, short_hash
from src.utils.path_utils import iter_files

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="기존 요약 무시하고 강제 재생성")
    args = parser.parse_args()

    if not settings.has_api_key():
        print("ERROR: GOOGLE_API_KEY 가 설정되어 있지 않습니다. .env 를 확인하세요.")
        return 2

    excel_dir: Path = settings.raw_data_dir / settings.category_dirs["excel"]
    files = list(iter_files(excel_dir, exts={".xlsx", ".xlsm"}))
    if not files:
        print(f"{excel_dir} 에 Excel 파일이 없습니다.")
        return 0

    summarizer = ExcelSummarizer()
    ok = fail = skip = 0

    for path in files:
        try:
            fhash = file_hash(path)
            document_id = f"doc_{short_hash(fhash, length=10)}"
            sections = parse_excel(path, document_id)
        except Exception as e:  # noqa: BLE001
            log.error("파싱 실패: %s (%s)", path.name, e)
            fail += 1
            continue

        for sec in sections:
            sheet = sec["metadata"].get("sheet_name") or sec.get("section_title") or "Sheet"
            try:
                summary = summarizer.summarize_section(
                    document_id=document_id,
                    file_name=path.name,
                    sheet_name=sheet,
                    raw_table_text=sec.get("content", ""),
                    table_range=sec["metadata"].get("table_range"),
                    source_raw_path=str(path),
                    force=args.force,
                )
                # cache hit 인 경우 summary.metadata 에 char_count 가 비어있는지로 판별 어려우므로
                # ok 로 통합 카운트
                ok += 1
                print(f"OK   · {path.name} / {sheet}")
            except GeminiError as e:
                fail += 1
                print(f"FAIL · {path.name} / {sheet} :: {e}", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                fail += 1
                print(f"FAIL · {path.name} / {sheet} :: {e}", file=sys.stderr)

    print(f"\n완료: ok={ok}, fail={fail}, skip={skip}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
