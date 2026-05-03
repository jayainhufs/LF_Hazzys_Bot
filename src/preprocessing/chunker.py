"""
chunker.py
==========
ParsedSection 들을 RAG에 적합한 Chunk 로 변환한다.

설계 원칙:
- 너무 작게 쪼개지 않는다 (chunk_size 기본 1200자, overlap 200자).
- chunk 앞에는 [파일명][카테고리][출처][시트/섹션][콘텐츠유형] 컨텍스트를 붙인다.
- 표/Excel 데이터는 행 단위가 깨지지 않게 자른다.
- 대화 데이터는 라인 단위 묶음을 유지한다.
- Excel summary chunk 는 잘게 쪼개지 않고 섹션 단위 그대로 유지한다.
- chunk 마다 metadata 를 풍부하게 부여한다 (parent_chunk_id 포함).
- source_weight 는 source_type 기준 기본값을 사용하되, content_type 으로 보정.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from src.config import settings
from src.logger import get_logger
from src.preprocessing.cleaner import clean_text
from src.preprocessing.normalizer import normalize_for_embedding
from src.schemas import Chunk, Document, ParsedSection
from src.utils.hash_utils import short_hash, text_hash
from src.utils.time_utils import now_iso

log = get_logger(__name__)


def get_source_weight(source_type: str, content_type: str, uploaded_category: str) -> float:
    """
    source_weight 결정.
    1) content_type 별 특수 가중치 (excel_summary / excel_raw_table)
    2) source_type 기본값
    3) uploaded_category 기본값
    """
    weights = settings.category_source_weight
    if content_type in {"excel_summary", "excel_raw_table"}:
        return float(weights.get(content_type, 1.0))
    if source_type in weights:
        return float(weights[source_type])
    if uploaded_category in weights:
        return float(weights[uploaded_category])
    return 0.7


def build_context_header(
    *,
    file_name: str,
    uploaded_category: str,
    source_type: str,
    section_title: Optional[str],
    content_type: str,
) -> str:
    """검색 친화적인 메타 prefix."""
    section = section_title or "-"
    return (
        f"[파일명] {file_name}\n"
        f"[카테고리] {uploaded_category}\n"
        f"[출처] {source_type}\n"
        f"[시트/섹션] {section}\n"
        f"[콘텐츠유형] {content_type}\n"
        f"[내용]\n"
    )


# ---------------------------------------------------------------------------
# 분할 알고리즘
# ---------------------------------------------------------------------------
def _split_text_with_overlap(
    text: str, chunk_size: int, overlap: int, line_aware: bool = True
) -> List[str]:
    """
    줄 단위 경계를 우선 존중하며 chunk_size, overlap 으로 분할.
    너무 큰 한 줄은 강제로 잘라낸다.
    """
    text = text or ""
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    if not line_aware:
        return _hard_split(text, chunk_size, overlap)

    lines = text.split("\n")
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush():
        if buf:
            chunks.append("\n".join(buf).strip())

    for line in lines:
        line_len = len(line) + 1  # 개행 포함 가정
        # 한 줄이 chunk_size 보다 크면 강제 분할
        if line_len > chunk_size:
            flush()
            buf = []
            buf_len = 0
            for piece in _hard_split(line, chunk_size, overlap):
                chunks.append(piece)
            continue
        if buf_len + line_len > chunk_size and buf:
            flush()
            # overlap: 최근 buf 의 마지막 일부를 유지
            overlap_lines = []
            cur = 0
            for prev in reversed(buf):
                if cur + len(prev) + 1 > overlap:
                    break
                overlap_lines.insert(0, prev)
                cur += len(prev) + 1
            buf = list(overlap_lines)
            buf_len = sum(len(x) + 1 for x in buf)
        buf.append(line)
        buf_len += line_len
    flush()

    return [c for c in chunks if c.strip()]


def _hard_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    out: List[str] = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(text), step):
        piece = text[i : i + chunk_size]
        if piece.strip():
            out.append(piece)
        if i + chunk_size >= len(text):
            break
    return out


# ---------------------------------------------------------------------------
# 메인 진입점
# ---------------------------------------------------------------------------
def chunk_sections(
    *,
    document: Document,
    sections: Iterable[ParsedSection],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Chunk]:
    """ParsedSection 리스트 → Chunk 리스트."""
    cs = int(chunk_size or settings.chunk_size)
    co = int(chunk_overlap or settings.chunk_overlap)

    file_hash_short = short_hash(document.file_hash, length=8)
    chunks: List[Chunk] = []
    chunk_index = 0

    for section in sections:
        cleaned = clean_text(section.content, section.content_type)
        if not cleaned:
            continue

        # Excel summary 는 섹션 단위 유지 (분할 X)
        if section.content_type == "excel_summary":
            pieces = [cleaned]
        elif section.content_type in {"table", "excel_raw_table", "conversation"}:
            # 표/대화는 라인 경계 우선
            pieces = _split_text_with_overlap(cleaned, cs, co, line_aware=True)
        else:
            pieces = _split_text_with_overlap(cleaned, cs, co, line_aware=True)

        for piece_idx, piece in enumerate(pieces):
            header = build_context_header(
                file_name=document.file_name,
                uploaded_category=document.uploaded_category,
                source_type=document.source_type,
                section_title=section.section_title,
                content_type=section.content_type,
            )
            content_with_header = header + piece
            embedding_text = header + normalize_for_embedding(piece)
            sw = get_source_weight(
                document.source_type, section.content_type, document.uploaded_category
            )
            chunk_id = f"{document.document_id[:10]}_{chunk_index:05d}_{short_hash(piece, length=6)}"

            metadata: Dict[str, Any] = {
                "source_type": document.source_type,
                "uploaded_category": document.uploaded_category,
                "file_name": document.file_name,
                "section_title": section.section_title,
                "content_type": section.content_type,
                "chunk_index": chunk_index,
                "section_chunk_index": piece_idx,
                "parent_chunk_id": None,
                "source_weight": float(sw),
                "created_at": now_iso(),
                "file_hash": document.file_hash,
                "raw_table_hash": section.metadata.get("raw_table_hash"),
                "summary_type": section.metadata.get("summary_type"),
                "sheet_name": section.metadata.get("sheet_name"),
                "chunk_hash": text_hash(piece),
                "file_hash_short": file_hash_short,
            }
            # 추가 metadata 머지
            for k, v in (section.metadata or {}).items():
                metadata.setdefault(k, v)

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    chunk_index=chunk_index,
                    source_type=document.source_type,
                    uploaded_category=document.uploaded_category,
                    file_name=document.file_name,
                    parent_chunk_id=None,
                    section_title=section.section_title,
                    content_type=section.content_type,
                    content=content_with_header,
                    clean_content=piece,
                    embedding_text=embedding_text,
                    metadata=metadata,
                )
            )
            chunk_index += 1

    log.info(
        "Chunking 완료: %s -> %d chunks (chunk_size=%d, overlap=%d)",
        document.file_name,
        len(chunks),
        cs,
        co,
    )
    return chunks


def link_excel_parent_child(chunks: List[Chunk]) -> List[Chunk]:
    """
    Excel parent-child 연결.
    같은 document_id + sheet_name 안에서
    excel_summary chunk 가 있으면 parent 로 두고,
    excel_raw_table chunk 들의 parent_chunk_id 를 설정한다.
    """
    # (doc_id, sheet_name) -> summary chunk_id
    summary_lookup: Dict[tuple, str] = {}
    for c in chunks:
        if c.content_type == "excel_summary":
            key = (c.document_id, c.metadata.get("sheet_name"))
            summary_lookup.setdefault(key, c.chunk_id)

    for c in chunks:
        if c.content_type == "excel_raw_table":
            key = (c.document_id, c.metadata.get("sheet_name"))
            parent_id = summary_lookup.get(key)
            if parent_id:
                c.parent_chunk_id = parent_id
                c.metadata["parent_chunk_id"] = parent_id

    return chunks
