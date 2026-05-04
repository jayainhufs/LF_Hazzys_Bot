"""
chunker.py
==========
ParsedSection 들을 RAG에 적합한 Chunk 로 변환한다.

설계 원칙:
- 너무 작게 쪼개지 않는다 (chunk_size 기본 1200자, overlap 200자).
- chunk 앞에는 [파일명][카테고리][출처][시트/섹션][콘텐츠유형] 컨텍스트를 붙인다.
  Slack TODO 처럼 풍부한 metadata 가 있으면 [문서날짜][TODO단계][업무주제][시간대]
  도 함께 붙여 검색 친화도를 높인다.
- 표/Excel 데이터는 행 단위가 깨지지 않게 자른다.
- 대화 데이터는 라인 단위 묶음을 유지한다.
- Excel summary chunk 는 잘게 쪼개지 않고 섹션 단위 그대로 유지한다.
- chunk 마다 metadata 를 풍부하게 부여한다 (parent_chunk_id 포함).
- source_weight 는 source_type 기준 기본값을 사용하되, content_type 으로 보정.
- Slack parser v2 가 sanitized_content 를 채워주면 embedding_text 와
  metadata.sanitized_content 양쪽에 그 값을 사용해 사람 이름/실시간이 임베딩되지 않게 한다.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from src.config import settings
from src.logger import get_logger
from src.preprocessing.anonymizer import anonymize_text
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
    document_date: Optional[str] = None,
    todo_phase: Optional[str] = None,
    primary_topic: Optional[str] = None,
    topic_tags: Optional[List[str]] = None,
    time_range_display: Optional[str] = None,
) -> str:
    """
    검색 친화적인 메타 prefix.

    기본 5개 라인은 그대로 유지(기존 테스트 호환).
    추가 metadata (document_date / todo_phase / primary_topic / topic_tags / time_range)
    가 있을 때만 라인을 덧붙여 임베딩에서 시간/주제 매칭이 잘 되도록 한다.
    """
    section = section_title or "-"
    parts = [
        f"[파일명] {file_name}",
        f"[카테고리] {uploaded_category}",
        f"[출처] {source_type}",
        f"[시트/섹션] {section}",
        f"[콘텐츠유형] {content_type}",
    ]
    if document_date:
        parts.append(f"[문서날짜] {document_date}")
    if todo_phase and todo_phase != "unknown":
        parts.append(f"[TODO단계] {todo_phase}")
    if primary_topic:
        parts.append(f"[업무주제] {primary_topic}")
    if topic_tags:
        parts.append(f"[topic_tags] {', '.join(topic_tags)}")
    if time_range_display:
        parts.append(f"[시간대] {time_range_display}")
    parts.append("[내용]")
    return "\n".join(parts) + "\n"


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
# Helpers
# ---------------------------------------------------------------------------
def _slice_sanitized(
    cleaned: str,  # noqa: ARG001 - 향후 정렬용
    pieces: List[str],
    section_sanitized: Optional[str],
) -> List[Optional[str]]:
    """
    파서가 미리 만들어둔 섹션 단위 sanitized_content 를 chunk piece 단위로 매핑한다.
    완벽한 1:1 매핑은 어렵기 때문에 다음 단순화 규칙을 쓴다.

    - 섹션이 chunk 1개로 끝나면 그대로 사용
    - 그 외에는 None 을 돌려주고, chunker 가 ``anonymize_text(piece)`` 로
      piece 단위 sanitization 을 다시 수행
    """
    if not pieces:
        return []
    if section_sanitized and len(pieces) == 1:
        return [section_sanitized.strip()]
    return [None] * len(pieces)


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

        # Slack parser v2 가 채워준 sanitized_content (섹션 단위) 가 있으면
        # 같은 길이로 잘라서 chunk 별 sanitized_content 를 만든다.
        section_sanitized = (section.metadata or {}).get("sanitized_content")
        sanitized_pieces: List[Optional[str]] = _slice_sanitized(
            cleaned, pieces, section_sanitized
        )

        for piece_idx, piece in enumerate(pieces):
            section_meta = section.metadata or {}
            doc_date = section_meta.get("document_date")
            todo_phase = section_meta.get("todo_phase")
            primary_topic = section_meta.get("primary_topic")
            topic_tags = section_meta.get("topic_tags") or []
            time_range_display = section_meta.get("time_range_display")

            header = build_context_header(
                file_name=document.file_name,
                uploaded_category=document.uploaded_category,
                source_type=document.source_type,
                section_title=section.section_title,
                content_type=section.content_type,
                document_date=doc_date,
                todo_phase=todo_phase,
                primary_topic=primary_topic,
                topic_tags=topic_tags if isinstance(topic_tags, list) else None,
                time_range_display=time_range_display,
            )
            content_with_header = header + piece

            # 비식별화된 piece (있으면 그것, 없으면 anonymize_text 로 fallback)
            piece_san = sanitized_pieces[piece_idx]
            if piece_san is None:
                piece_san = anonymize_text(piece)
            sanitized_content = header + piece_san

            # 임베딩에는 비식별화된 본문을 사용 → 사람 이름/실시간이 embedding 에
            # 끼어들지 않도록 한다. 다만 header 의 [문서날짜], [업무주제] 는
            # 검색 키워드로 활용되도록 그대로 둔다.
            embedding_text = header + normalize_for_embedding(piece_san)
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
                "raw_table_hash": section_meta.get("raw_table_hash"),
                "summary_type": section_meta.get("summary_type"),
                "sheet_name": section_meta.get("sheet_name"),
                "chunk_hash": text_hash(piece),
                "file_hash_short": file_hash_short,
                # ------ slack v2 / 비식별화 추가 필드 ------
                "document_date": doc_date,
                "date_text": section_meta.get("date_text"),
                "date_source": section_meta.get("date_source"),
                "todo_phase": todo_phase or "unknown",
                "topic_tags": topic_tags if isinstance(topic_tags, list) else [],
                "primary_topic": primary_topic,
                "speaker_roles": section_meta.get("speaker_roles") or [],
                "display_speakers": section_meta.get("display_speakers") or [],
                "time_buckets": section_meta.get("time_buckets") or [],
                "time_range_display": time_range_display,
                "parser_format": section_meta.get("parser_format"),
                "sanitized_content": sanitized_content,
            }
            # 추가 metadata 머지 (위에서 명시적으로 다룬 키는 덮어쓰지 않음)
            for k, v in section_meta.items():
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
