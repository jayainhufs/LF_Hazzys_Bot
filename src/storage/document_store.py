"""
document_store.py
=================
처리된 Document metadata / Chunk 들을 디스크에 보관.

- documents/<document_id>.json   : Document metadata
- chunks/<document_id>.jsonl     : Chunk 1개 = 한 줄 (JSON Lines)

ensure_ascii=False 사용 → 한글 그대로 저장.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

from src.config import settings
from src.logger import get_logger
from src.schemas import Chunk, Document
from src.utils.path_utils import ensure_dir, safe_filename

log = get_logger(__name__)


class DocumentStore:
    def __init__(
        self,
        documents_dir: Optional[Path] = None,
        chunks_dir: Optional[Path] = None,
    ) -> None:
        self.documents_dir: Path = ensure_dir(documents_dir or settings.documents_dir)
        self.chunks_dir: Path = ensure_dir(chunks_dir or settings.chunks_dir)

    # ------------------------- Document -------------------------
    def save_document(self, doc: Document) -> Path:
        path = self.documents_dir / f"{safe_filename(doc.document_id)}.json"
        path.write_text(
            json.dumps(doc.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def load_document(self, document_id: str) -> Optional[Document]:
        path = self.documents_dir / f"{safe_filename(document_id)}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8") or "{}")
            return Document(**data)
        except Exception as e:  # noqa: BLE001
            log.error("Document 로드 실패: %s (%s)", path.name, e)
            return None

    def list_document_ids(self) -> List[str]:
        return sorted(p.stem for p in self.documents_dir.glob("*.json"))

    # ------------------------- Chunks -------------------------
    def save_chunks(self, document_id: str, chunks: Iterable[Chunk]) -> Path:
        path = self.chunks_dir / f"{safe_filename(document_id)}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        return path

    def load_chunks(self, document_id: str) -> List[Chunk]:
        path = self.chunks_dir / f"{safe_filename(document_id)}.jsonl"
        if not path.exists():
            return []
        out: List[Chunk] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    out.append(Chunk(**data))
        except Exception as e:  # noqa: BLE001
            log.error("Chunks JSONL 로드 실패: %s (%s)", path.name, e)
        return out

    def remove(self, document_id: str) -> None:
        for p in [
            self.documents_dir / f"{safe_filename(document_id)}.json",
            self.chunks_dir / f"{safe_filename(document_id)}.jsonl",
        ]:
            try:
                p.unlink(missing_ok=True)
            except Exception as e:  # noqa: BLE001
                log.warning("파일 삭제 실패: %s (%s)", p.name, e)

    def stats(self) -> dict:
        n_docs = sum(1 for _ in self.documents_dir.glob("*.json"))
        n_chunks = 0
        for p in self.chunks_dir.glob("*.jsonl"):
            try:
                with p.open("r", encoding="utf-8") as f:
                    n_chunks += sum(1 for _ in f if _.strip())
            except Exception:  # noqa: BLE001
                continue
        return {"documents": n_docs, "chunks": n_chunks}
