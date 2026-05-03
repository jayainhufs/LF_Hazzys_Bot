"""
file_registry.py
================
indexed_files.json 관리.

저장 항목:
{
  "<file_hash>": {
    "document_id": "...",
    "file_name": "...",
    "file_path": "...",      # project root 기준 상대경로
    "uploaded_category": "...",
    "ingested_at": "...",
    "chunk_count": 12,
    "summary_generated": false,
    "summary_hashes": []
  },
  ...
}

파일이 변경되면 file_hash 가 바뀌므로 자연스럽게 재색인된다.
"""
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from src.config import settings
from src.logger import get_logger
from src.utils.path_utils import ensure_dir, relative_to_project
from src.utils.time_utils import now_iso

log = get_logger(__name__)
_LOCK = Lock()


class FileRegistry:
    def __init__(self, registry_path: Optional[Path] = None) -> None:
        self.path: Path = registry_path or settings.registry_path
        ensure_dir(self.path.parent)
        self._data: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8") or "{}")
        except Exception as e:  # noqa: BLE001
            log.warning("registry 로드 실패, 새로 시작합니다: %s", e)
            return {}

    def _save(self) -> None:
        with _LOCK:
            self.path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    # ------------------------------------------------------------------
    def has_hash(self, file_hash: str) -> bool:
        return file_hash in self._data

    def get(self, file_hash: str) -> Optional[Dict[str, Any]]:
        return self._data.get(file_hash)

    def all(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._data)

    def upsert(
        self,
        *,
        file_hash: str,
        document_id: str,
        file_name: str,
        file_path: Path,
        uploaded_category: str,
        chunk_count: int,
        summary_generated: bool = False,
        summary_hashes: Optional[List[str]] = None,
    ) -> None:
        self._data[file_hash] = {
            "document_id": document_id,
            "file_name": file_name,
            "file_path": relative_to_project(file_path, settings.project_root),
            "uploaded_category": uploaded_category,
            "ingested_at": now_iso(),
            "chunk_count": int(chunk_count),
            "summary_generated": bool(summary_generated),
            "summary_hashes": list(summary_hashes or []),
        }
        self._save()

    def update_summary(
        self, file_hash: str, summary_hashes: List[str], generated: bool = True
    ) -> None:
        if file_hash not in self._data:
            return
        rec = self._data[file_hash]
        rec["summary_generated"] = bool(generated)
        merged = set(rec.get("summary_hashes", [])) | set(summary_hashes)
        rec["summary_hashes"] = sorted(merged)
        self._save()

    def remove(self, file_hash: str) -> None:
        self._data.pop(file_hash, None)
        self._save()

    def reset(self) -> None:
        self._data = {}
        self._save()
