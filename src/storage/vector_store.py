"""
vector_store.py
===============
ChromaDB 기반 로컬 Vector Store wrapper.

- persist 경로: settings.chroma_db_dir
- 컬렉션 이름: settings.chroma_collection (기본 work_knowledge)
- chunk_id 기준 중복 저장 방지 (upsert)
- 저장 시 metadata 와 document text 함께 기록
- 검색 결과는 RetrievedChunk 로 정규화하여 반환
"""
from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence

from src.config import settings
from src.logger import get_logger
from src.rag.bm25 import BM25Document, BM25Scorer
from src.schemas import Chunk, RetrievedChunk

log = get_logger(__name__)

# Chroma metadata 는 primitive 만 지원하므로 None / 복합 타입을 정리해야 한다.
_PRIMITIVES = (str, int, float, bool)


def _coerce_metadata(md: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (md or {}).items():
        if v is None:
            continue
        if isinstance(v, _PRIMITIVES):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _metadata_for_chunk(chunk: Chunk) -> Dict[str, Any]:
    """Return Chroma metadata with Chunk top-level fields preserved."""
    md: Dict[str, Any] = dict(chunk.metadata or {})
    for key in (
        "document_id",
        "source_type",
        "uploaded_category",
        "file_name",
        "content_type",
        "parent_chunk_id",
        "section_title",
        "chunk_index",
    ):
        value = getattr(chunk, key, None)
        if value is not None and key not in md:
            md[key] = value
    return md


class VectorStore:
    """ChromaDB persistent 컬렉션 wrapper."""

    _client_cache: Dict[str, Any] = {}
    _lock = Lock()

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        self.persist_dir: Path = persist_dir or settings.chroma_db_dir
        self.collection_name: str = collection_name or settings.chroma_collection
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = self._get_client(self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @classmethod
    def _get_client(cls, persist_dir: Path):
        key = str(persist_dir.resolve())
        with cls._lock:
            if key in cls._client_cache:
                return cls._client_cache[key]
            try:
                import chromadb
            except ImportError as e:
                raise RuntimeError(
                    "chromadb 가 설치되어 있지 않습니다. requirements.txt 를 다시 설치하세요."
                ) from e
            client = chromadb.PersistentClient(path=str(persist_dir))
            cls._client_cache[key] = client
            return client

    # ------------------------------------------------------------------
    # 저장 / 삭제
    # ------------------------------------------------------------------
    def existing_ids(self, ids: Sequence[str]) -> set[str]:
        if not ids:
            return set()
        try:
            res = self._collection.get(ids=list(ids), include=[])
            return set(res.get("ids", []))
        except Exception as e:  # noqa: BLE001
            log.warning("existing_ids 조회 실패 (계속 진행): %s", e)
            return set()

    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        skip_existing: bool = True,
    ) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError("chunks 와 embeddings 길이가 다릅니다.")

        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [_coerce_metadata(_metadata_for_chunk(c)) for c in chunks]

        if skip_existing:
            already = self.existing_ids(ids)
            if already:
                idx_keep = [i for i, _id in enumerate(ids) if _id not in already]
                ids = [ids[i] for i in idx_keep]
                documents = [documents[i] for i in idx_keep]
                metadatas = [metadatas[i] for i in idx_keep]
                embeddings = [embeddings[i] for i in idx_keep]

        if not ids:
            return 0

        try:
            self._collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        except Exception as e:
            log.error("Chroma upsert 실패: %s", e)
            raise
        return len(ids)

    def delete_by_document(self, document_id: str) -> int:
        try:
            res = self._collection.get(where={"document_id": document_id}, include=[])
            ids = res.get("ids", [])
            if ids:
                self._collection.delete(ids=ids)
            return len(ids)
        except Exception as e:  # noqa: BLE001
            log.warning("문서 단위 삭제 실패: %s", e)
            return 0

    def reset_db(self) -> None:
        try:
            self._client.delete_collection(self.collection_name)
        except Exception as e:  # noqa: BLE001
            log.warning("collection 삭제 시도 중 예외(무시): %s", e)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # 검색
    # ------------------------------------------------------------------
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 8,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        try:
            res = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=int(top_k),
                where=filters or None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            log.error("Chroma query 실패: %s", e)
            raise

        out: List[RetrievedChunk] = []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            score = 1.0 - float(dist) if dist is not None else 0.0
            meta = meta or {}
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    document_id=str(meta.get("document_id", "")),
                    file_name=str(meta.get("file_name", "")),
                    source_type=str(meta.get("source_type", "")),
                    uploaded_category=str(meta.get("uploaded_category", "")),
                    section_title=meta.get("section_title"),
                    content_type=str(meta.get("content_type", "text")),
                    content=doc or "",
                    score=score,
                    final_score=score,
                    parent_chunk_id=meta.get("parent_chunk_id"),
                    metadata=dict(meta),
                )
            )
        return out

    def search_bm25(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """Keyword search over stored Chroma documents using local BM25."""
        if not query or not query.strip():
            return []
        try:
            res = self._collection.get(
                where=filters or None,
                include=["documents", "metadatas"],
            )
        except Exception as e:
            log.error("Chroma BM25 document load 실패: %s", e)
            raise

        ids = res.get("ids", []) or []
        docs = res.get("documents", []) or []
        metas = res.get("metadatas", []) or []
        if not ids:
            return []

        payload_by_id: Dict[str, Dict[str, Any]] = {}
        bm25_docs: List[BM25Document] = []
        for cid, doc, meta in zip(ids, docs, metas):
            meta = meta or {}
            text = doc or ""
            payload_by_id[cid] = {"document": text, "metadata": dict(meta)}
            bm25_docs.append(
                BM25Document(document_id=str(cid), text=text, payload=payload_by_id[cid])
            )

        results = BM25Scorer(bm25_docs).search(query, top_k=top_k)
        out: List[RetrievedChunk] = []
        for result in results:
            payload = result.payload or {}
            doc = payload.get("document") or ""
            meta = dict(payload.get("metadata") or {})
            meta["bm25_score"] = round(float(result.score), 6)
            meta["bm25_rank"] = int(result.rank)
            meta["bm25_normalized_score"] = round(float(result.normalized_score), 6)
            out.append(
                RetrievedChunk(
                    chunk_id=result.document_id,
                    document_id=str(meta.get("document_id", "")),
                    file_name=str(meta.get("file_name", "")),
                    source_type=str(meta.get("source_type", "")),
                    uploaded_category=str(meta.get("uploaded_category", "")),
                    section_title=meta.get("section_title"),
                    content_type=str(meta.get("content_type", "text")),
                    content=doc,
                    score=float(result.normalized_score),
                    final_score=float(result.normalized_score),
                    parent_chunk_id=meta.get("parent_chunk_id"),
                    metadata=meta,
                )
            )
        return out

    # ------------------------------------------------------------------
    # parent-child 보강 검색
    # ------------------------------------------------------------------
    def get_children(
        self, parent_chunk_id: str, sheet_name: Optional[str] = None, limit: int = 3
    ) -> List[RetrievedChunk]:
        where: Dict[str, Any] = {"parent_chunk_id": parent_chunk_id}
        try:
            res = self._collection.get(
                where=where, include=["documents", "metadatas"], limit=int(limit)
            )
        except Exception as e:  # noqa: BLE001
            log.warning("자식 chunk 조회 실패: %s", e)
            return []

        out: List[RetrievedChunk] = []
        ids = res.get("ids", []) or []
        docs = res.get("documents", []) or []
        metas = res.get("metadatas", []) or []
        for cid, doc, meta in zip(ids, docs, metas):
            meta = meta or {}
            if sheet_name and meta.get("sheet_name") and meta["sheet_name"] != sheet_name:
                continue
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    document_id=str(meta.get("document_id", "")),
                    file_name=str(meta.get("file_name", "")),
                    source_type=str(meta.get("source_type", "")),
                    uploaded_category=str(meta.get("uploaded_category", "")),
                    section_title=meta.get("section_title"),
                    content_type=str(meta.get("content_type", "text")),
                    content=doc or "",
                    score=0.0,
                    final_score=0.0,
                    parent_chunk_id=meta.get("parent_chunk_id"),
                    metadata=dict(meta),
                )
            )
        return out

    def stats(self) -> Dict[str, Any]:
        try:
            count = self._collection.count()
        except Exception:  # noqa: BLE001
            count = -1
        return {
            "collection_name": self.collection_name,
            "persist_dir": str(self.persist_dir),
            "count": int(count),
        }
