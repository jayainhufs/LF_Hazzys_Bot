"""
pipeline.py
===========
원본 파일 한 개를 끝까지 색인하는 파이프라인.

흐름:
    file_path
      ↓ file_router          → (source_type, sections)
      ↓ Excel summarization  (옵션, ENABLE_EXCEL_SUMMARY 또는 인자)
      ↓ ParsedSection list
      ↓ chunker              → Chunk[]
      ↓ deduplicator         → Chunk[]
      ↓ embedder             → embeddings
      ↓ vector_store.add_chunks
      ↓ document_store.save  (Document JSON, Chunks JSONL)
      ↓ file_registry.upsert

cli (`scripts/ingest_folder.py`) 와 streamlit 색인 페이지가 공통으로 사용한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.config import settings
from src.ingestion.file_router import is_supported, parse_file
from src.logger import get_logger
from src.preprocessing.chunker import chunk_sections, link_excel_parent_child
from src.preprocessing.deduplicator import deduplicate_chunks
from src.rag.embedder import Embedder, get_default_embedder
from src.rag.gemini_client import GeminiError
from src.schemas import Chunk, Document, ParsedSection
from src.storage.document_store import DocumentStore
from src.storage.file_registry import FileRegistry
from src.storage.vector_store import VectorStore
from src.summarization.excel_summarizer import ExcelSummarizer
from src.utils.cost_utils import tracker
from src.utils.hash_utils import file_hash, short_hash, text_hash
from src.utils.path_utils import iter_files, relative_to_project
from src.utils.time_utils import now_iso

log = get_logger(__name__)


@dataclass
class IngestResult:
    file_path: str
    file_name: str
    uploaded_category: str
    status: str  # "indexed" | "skipped" | "failed"
    chunks_added: int = 0
    sections_count: int = 0
    summary_count: int = 0
    document_id: str = ""
    file_hash: str = ""
    error: str = ""
    # ----- LLM normalization (Task 4 부터 사용) -----
    normalized_chunks_added: int = 0
    normalized_card_count: int = 0
    normalized_kind: str = ""
    normalized_skipped_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


@dataclass
class IngestSummary:
    results: List[IngestResult] = field(default_factory=list)
    total: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    chunks_added_total: int = 0
    normalized_chunks_total: int = 0


# ---------------------------------------------------------------------------
# 파일 1개 색인
# ---------------------------------------------------------------------------
def ingest_file(
    path: Path,
    uploaded_category: str,
    *,
    embedder: Optional[Embedder] = None,
    vector_store: Optional[VectorStore] = None,
    document_store: Optional[DocumentStore] = None,
    file_registry: Optional[FileRegistry] = None,
    excel_summarizer: Optional[ExcelSummarizer] = None,
    enable_excel_summary: Optional[bool] = None,
    enable_llm_normalization: Optional[bool] = None,
    gemini_client: Optional[Any] = None,
    normalization_store: Optional[Any] = None,
    skip_if_indexed: bool = True,
) -> IngestResult:
    """파일 1개 색인. 실패해도 raise 하지 않고 IngestResult 로 반환."""
    name = path.name
    res = IngestResult(
        file_path=relative_to_project(path, settings.project_root),
        file_name=name,
        uploaded_category=uploaded_category,
        status="failed",
    )

    try:
        if not path.exists():
            res.error = "파일이 존재하지 않습니다."
            return res
        if not is_supported(path):
            res.status = "skipped"
            res.error = f"지원하지 않는 확장자: {path.suffix}"
            return res

        registry = file_registry or FileRegistry()
        store = document_store or DocumentStore()
        vstore = vector_store or VectorStore()
        emb = embedder or get_default_embedder()

        fhash = file_hash(path)
        res.file_hash = fhash

        if skip_if_indexed and registry.has_hash(fhash):
            res.status = "skipped"
            res.error = "이미 색인된 파일 (hash 동일)"
            existing = registry.get(fhash) or {}
            res.document_id = existing.get("document_id", "")
            res.chunks_added = int(existing.get("chunk_count", 0))
            log.info("[skip] %s (hash 동일)", name)
            return res

        document_id = f"doc_{short_hash(fhash, length=10)}"
        # 1) 파싱
        source_type, raw_sections = parse_file(path, document_id, uploaded_category)

        # ParsedSection 객체로 변환
        parsed_sections: List[ParsedSection] = []
        for i, s in enumerate(raw_sections):
            ps = ParsedSection(
                section_id=f"sec_{document_id}_{i:04d}",
                document_id=document_id,
                section_title=s.get("section_title"),
                content_type=s.get("content_type", "text"),
                content=s.get("content", "") or "",
                metadata=dict(s.get("metadata") or {}),
            )
            # raw_table 인 경우 raw_table_hash 를 미리 계산해 metadata 에 넣어둔다
            if ps.content_type == "excel_raw_table":
                ps.metadata["raw_table_hash"] = text_hash(ps.content)
            parsed_sections.append(ps)

        res.sections_count = len(parsed_sections)

        # 2) Excel summary (옵션)
        use_summary = (
            enable_excel_summary if enable_excel_summary is not None else settings.enable_excel_summary
        )
        summary_hashes: List[str] = []
        if use_summary and source_type == "excel":
            summarizer = excel_summarizer or ExcelSummarizer()
            summary_sections: List[ParsedSection] = []
            for ps in list(parsed_sections):
                if ps.content_type != "excel_raw_table":
                    continue
                sheet_name = ps.metadata.get("sheet_name") or ps.section_title or "Sheet"
                table_range = ps.metadata.get("table_range")
                try:
                    summary = summarizer.summarize_section(
                        document_id=document_id,
                        file_name=name,
                        sheet_name=sheet_name,
                        raw_table_text=ps.content,
                        table_range=table_range,
                        source_raw_path=str(path),
                    )
                except GeminiError as e:
                    log.warning("Excel summary 실패 (%s/%s): %s", name, sheet_name, e)
                    continue
                summary_sections.append(summarizer.to_parsed_section(summary, document_id))
                summary_hashes.append(summary.raw_table_hash)
            # summary 는 raw 보다 앞에 오게 추가
            parsed_sections = summary_sections + parsed_sections
            res.summary_count = len(summary_sections)

        # 3) Document 객체 생성
        document = Document(
            document_id=document_id,
            source_type=source_type,
            uploaded_category=uploaded_category,
            file_name=name,
            file_path=relative_to_project(path, settings.project_root),
            file_hash=fhash,
            title=path.stem,
            created_at=now_iso(),
            ingested_at=now_iso(),
            metadata={
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "ext": path.suffix.lower(),
            },
        )

        # 4) 청크 + parent-child 연결 + 중복 제거
        chunks: List[Chunk] = chunk_sections(document=document, sections=parsed_sections)
        chunks = link_excel_parent_child(chunks)
        chunks = deduplicate_chunks(chunks)

        if not chunks:
            res.status = "skipped"
            res.error = "유효한 chunk 가 없습니다 (빈 파일?)."
            return res

        # 5) 임베딩
        embed_inputs = [c.embedding_text or c.clean_content or c.content for c in chunks]
        try:
            embeddings = emb.embed_documents(embed_inputs)
        except Exception as e:
            res.error = f"임베딩 실패: {e}"
            log.error("임베딩 실패: %s (%s)", name, e)
            return res

        if len(embeddings) != len(chunks):
            res.error = (
                f"임베딩 결과 길이 불일치: chunks={len(chunks)} embeddings={len(embeddings)}"
            )
            return res
        if any(not v for v in embeddings):
            res.error = "임베딩 결과에 빈 벡터가 포함되었습니다."
            return res

        # 6) 벡터 저장
        try:
            n_added = vstore.add_chunks(chunks, embeddings, skip_existing=True)
        except Exception as e:
            res.error = f"ChromaDB 저장 실패: {e}"
            log.error("ChromaDB 저장 실패: %s (%s)", name, e)
            return res

        # 7) document / chunks 디스크 저장
        store.save_document(document)
        store.save_chunks(document_id, chunks)

        # 7.5) (옵션) LLM-based Document Normalization branch
        #      - ENABLE_LLM_NORMALIZATION=false 일 때는 절대 실행되지 않는다.
        #      - normalization 실패는 raw ingest 흐름에 영향을 주지 않는다.
        use_normalization = (
            enable_llm_normalization
            if enable_llm_normalization is not None
            else settings.enable_llm_normalization
        )
        if use_normalization:
            try:
                from src.normalization.pipeline_integration import run_normalization_branch

                norm_result = run_normalization_branch(
                    document=document,
                    parsed_sections=parsed_sections,
                    raw_chunks=chunks,
                    embedder=emb,
                    vector_store=vstore,
                    document_store=store,
                    gemini_client=gemini_client,
                    normalization_store=normalization_store,
                    settings_obj=settings,
                )
                res.normalized_kind = str(norm_result.get("kind") or "")
                res.normalized_card_count = int(norm_result.get("card_count") or 0)
                res.normalized_chunks_added = int(norm_result.get("chunks_added") or 0)
                res.normalized_skipped_reason = str(
                    norm_result.get("skipped_reason") or ""
                )
            except Exception as e:  # noqa: BLE001
                # run_normalization_branch 자체가 raise 하지 않도록 설계되었지만
                # 모듈 import 실패 등 예외 상황에서도 raw ingest 가 죽지 않게 한다.
                log.warning(
                    "LLM normalization branch 진입 실패 (raw ingest 는 계속 진행): %s", e
                )
                res.normalized_skipped_reason = f"branch entry 실패: {e}"

        # 8) registry
        registry.upsert(
            file_hash=fhash,
            document_id=document_id,
            file_name=name,
            file_path=path,
            uploaded_category=uploaded_category,
            chunk_count=len(chunks),
            summary_generated=bool(summary_hashes),
            summary_hashes=summary_hashes,
        )

        res.status = "indexed"
        res.document_id = document_id
        res.chunks_added = int(n_added)
        log.info(
            "[indexed] %s | chunks=%d, summaries=%d, normalized=%d",
            name,
            n_added,
            res.summary_count,
            res.normalized_chunks_added,
        )
        return res

    except Exception as e:  # noqa: BLE001
        log.exception("색인 중 예외: %s", e)
        res.status = "failed"
        res.error = f"{type(e).__name__}: {e}"
        return res


# ---------------------------------------------------------------------------
# 폴더 스캔 → 색인
# ---------------------------------------------------------------------------
SUPPORTED_EXTS = {".xlsx", ".xlsm", ".docx", ".txt", ".md"}


def discover_files(raw_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    data/raw/<카테고리>/ 하위의 지원 파일들을 (path, uploaded_category) 로 나열.
    """
    raw_root = raw_root or settings.raw_data_dir
    cat_to_subdir = settings.category_dirs  # {"slack": "slack_manual", ...}
    subdir_to_cat = {v: k for k, v in cat_to_subdir.items()}
    out: List[Dict[str, Any]] = []
    for sub, cat in subdir_to_cat.items():
        folder = raw_root / sub
        if not folder.exists():
            continue
        for p in iter_files(folder, exts=SUPPORTED_EXTS):
            out.append({"path": p, "uploaded_category": cat})
    # 카테고리 폴더가 아닌 raw 직속 파일도 misc 로 처리 (안전망)
    for p in iter_files(raw_root, exts=SUPPORTED_EXTS):
        # 이미 위에서 잡힌 파일은 제외
        if any(item["path"] == p for item in out):
            continue
        # 카테고리 서브폴더 직속이 아닌 경우만 misc
        rel = p.relative_to(raw_root)
        first = rel.parts[0] if rel.parts else ""
        if first in cat_to_subdir.values():
            continue
        out.append({"path": p, "uploaded_category": "misc"})
    return out


def ingest_folder(
    raw_root: Optional[Path] = None,
    *,
    enable_excel_summary: Optional[bool] = None,
    enable_llm_normalization: Optional[bool] = None,
    gemini_client: Optional[Any] = None,
    normalization_store: Optional[Any] = None,
    on_progress: Optional[Callable[[IngestResult, int, int], None]] = None,
    skip_if_indexed: bool = True,
) -> IngestSummary:
    """폴더 전체 색인."""
    raw_root = raw_root or settings.raw_data_dir
    files = discover_files(raw_root)
    summary = IngestSummary(total=len(files))
    if not files:
        return summary

    # 객체 1번씩만 만들고 재사용
    embedder = get_default_embedder()
    vstore = VectorStore()
    dstore = DocumentStore()
    registry = FileRegistry()
    summarizer: Optional[ExcelSummarizer] = (
        ExcelSummarizer() if enable_excel_summary or settings.enable_excel_summary else None
    )

    for i, item in enumerate(files):
        res = ingest_file(
            item["path"],
            item["uploaded_category"],
            embedder=embedder,
            vector_store=vstore,
            document_store=dstore,
            file_registry=registry,
            excel_summarizer=summarizer,
            enable_excel_summary=enable_excel_summary,
            enable_llm_normalization=enable_llm_normalization,
            gemini_client=gemini_client,
            normalization_store=normalization_store,
            skip_if_indexed=skip_if_indexed,
        )
        summary.results.append(res)
        if res.status == "indexed":
            summary.indexed += 1
            summary.chunks_added_total += int(res.chunks_added)
            summary.normalized_chunks_total += int(res.normalized_chunks_added)
        elif res.status == "skipped":
            summary.skipped += 1
        else:
            summary.failed += 1
        if on_progress:
            try:
                on_progress(res, i + 1, len(files))
            except Exception:  # noqa: BLE001
                pass

    log.info(
        "ingest_folder 완료: total=%d, indexed=%d, skipped=%d, failed=%d, "
        "chunks_added=%d, normalized=%d, tracker=%s",
        summary.total,
        summary.indexed,
        summary.skipped,
        summary.failed,
        summary.chunks_added_total,
        summary.normalized_chunks_total,
        tracker.snapshot(),
    )
    return summary
