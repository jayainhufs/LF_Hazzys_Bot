"""
summary_store.py
================
Excel 한국어 요약본 저장/조회 헬퍼.

저장 위치: settings.excel_summary_dir
- {raw_table_hash[:12]}__{safe_file_stem}__{safe_sheet}.md     # 본문(Markdown)
- {raw_table_hash[:12]}__{safe_file_stem}__{safe_sheet}.json   # ExcelSummary metadata

raw_table_hash 가 같으면 재생성하지 않는다 (캐시).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from src.config import settings
from src.logger import get_logger
from src.schemas import ExcelSummary
from src.utils.path_utils import ensure_dir, safe_filename

log = get_logger(__name__)


def _key(raw_table_hash: str, file_name: str, sheet_name: str) -> str:
    base = f"{raw_table_hash[:12]}__{safe_filename(Path(file_name).stem)}__{safe_filename(sheet_name)}"
    return base[:200]


class SummaryStore:
    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir: Path = ensure_dir(base_dir or settings.excel_summary_dir)

    # ------------------------------------------------------------------
    def has(self, raw_table_hash: str, file_name: str, sheet_name: str) -> bool:
        return self._md_path(raw_table_hash, file_name, sheet_name).exists()

    def _md_path(self, raw_table_hash: str, file_name: str, sheet_name: str) -> Path:
        return self.base_dir / f"{_key(raw_table_hash, file_name, sheet_name)}.md"

    def _json_path(self, raw_table_hash: str, file_name: str, sheet_name: str) -> Path:
        return self.base_dir / f"{_key(raw_table_hash, file_name, sheet_name)}.json"

    def save(self, summary: ExcelSummary) -> Path:
        md_path = self._md_path(summary.raw_table_hash, summary.file_name, summary.sheet_name)
        md_path.write_text(summary.summary_text or "", encoding="utf-8")
        summary.summary_markdown_path = str(md_path)
        json_path = self._json_path(summary.raw_table_hash, summary.file_name, summary.sheet_name)
        json_path.write_text(
            json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return md_path

    def load(
        self, raw_table_hash: str, file_name: str, sheet_name: str
    ) -> Optional[ExcelSummary]:
        json_path = self._json_path(raw_table_hash, file_name, sheet_name)
        if not json_path.exists():
            return None
        try:
            data = json.loads(json_path.read_text(encoding="utf-8") or "{}")
            return ExcelSummary(**data)
        except Exception as e:  # noqa: BLE001
            log.warning("ExcelSummary 로드 실패: %s (%s)", json_path.name, e)
            return None

    def list_all(self) -> List[ExcelSummary]:
        out: List[ExcelSummary] = []
        for p in sorted(self.base_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8") or "{}")
                out.append(ExcelSummary(**data))
            except Exception as e:  # noqa: BLE001
                log.warning("Summary metadata 로드 실패: %s (%s)", p.name, e)
        return out

    def list_for_file(self, file_name: str) -> List[ExcelSummary]:
        return [s for s in self.list_all() if s.file_name == file_name]

    def delete(self, raw_table_hash: str, file_name: str, sheet_name: str) -> None:
        for p in [
            self._md_path(raw_table_hash, file_name, sheet_name),
            self._json_path(raw_table_hash, file_name, sheet_name),
        ]:
            try:
                p.unlink(missing_ok=True)
            except Exception as e:  # noqa: BLE001
                log.warning("summary 파일 삭제 실패: %s (%s)", p.name, e)
