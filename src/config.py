"""
config.py
=========
프로젝트 전역 설정 로더.

- `.env` 파일을 읽어 환경변수에서 모든 설정값을 가져온다.
- 모든 경로는 `pathlib.Path` 로 다뤄 Mac/Windows 양쪽에서 동작한다.
- 모델명은 하드코딩하지 않고, .env 에서 자유롭게 교체 가능하다.

다른 모듈에서는 다음과 같이 사용한다::

    from src.config import settings
    print(settings.generation_model)
    print(settings.chroma_db_dir)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 프로젝트 루트 결정
#   - 이 파일 위치: <PROJECT_ROOT>/src/config.py
#   - 따라서 parents[1] 가 PROJECT_ROOT 가 된다.
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# .env 가 있으면 로드 (없어도 에러내지 않음)
_DOTENV_PATH = PROJECT_ROOT / ".env"
if _DOTENV_PATH.exists():
    load_dotenv(_DOTENV_PATH)


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    """환경변수에서 값을 읽되, 빈 문자열이면 None 처리."""
    value = os.getenv(key, default)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env(key)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _env_int(key: str, default: int) -> int:
    raw = _env(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _resolve_path(raw_path: str) -> Path:
    """문자열 경로를 받아 절대 경로 Path 로 정규화."""
    p = Path(raw_path)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p


@dataclass
class Settings:
    """전역 설정 객체."""

    # ----- 기본 정보 -----
    app_name: str = "Work RAG Assistant"
    env: str = "local"

    # ----- 경로 -----
    project_root: Path = PROJECT_ROOT
    raw_data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    processed_data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
    chroma_db_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "storage" / "chroma_db")
    qa_log_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "storage" / "qa_logs")
    registry_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "storage" / "registry" / "indexed_files.json"
    )
    excel_summary_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "summaries" / "excel"
    )
    documents_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "documents"
    )
    chunks_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "chunks"
    )

    # ----- Google Gemini -----
    google_api_key: Optional[str] = None
    generation_model: str = "gemini-2.5-flash-lite"
    fallback_generation_model: str = "gemini-2.5-flash"
    excel_summary_model: str = "gemini-2.5-flash-lite"

    # ----- Embedding -----
    embedding_provider: str = "gemini"  # gemini | local
    gemini_embedding_model: str = "gemini-embedding-001"
    local_embedding_model: str = "BAAI/bge-m3"

    # ----- RAG -----
    top_k: int = 8
    chunk_size: int = 1200
    chunk_overlap: int = 200
    enable_query_rewrite: bool = False
    enable_excel_summary: bool = False
    max_context_chars: int = 12000
    max_chunks_per_file: int = 3
    request_timeout_seconds: int = 60
    chroma_collection: str = "work_knowledge"

    # ----- 카테고리 → raw 폴더 -----
    category_dirs: dict = field(default_factory=lambda: {
        "slack": "slack_manual",
        "guide": "guide",
        "kakao": "kakao",
        "excel": "excel",
        "misc": "misc",
    })

    # ----- 카테고리 → source_weight 기본값 -----
    category_source_weight: dict = field(default_factory=lambda: {
        "excel_summary": 1.1,
        "excel_raw_table": 1.0,
        "excel": 1.0,
        "guide": 0.9,
        "slack_manual": 0.85,
        "slack": 0.85,
        "word": 0.75,
        "markdown": 0.7,
        "txt": 0.6,
        "kakao": 0.5,
        "misc": 0.5,
    })

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        s.app_name = _env("APP_NAME", s.app_name) or s.app_name
        s.env = _env("ENV", s.env) or s.env

        if (raw := _env("RAW_DATA_DIR")):
            s.raw_data_dir = _resolve_path(raw)
        if (raw := _env("PROCESSED_DATA_DIR")):
            s.processed_data_dir = _resolve_path(raw)
            s.documents_dir = s.processed_data_dir / "documents"
            s.chunks_dir = s.processed_data_dir / "chunks"
        if (raw := _env("CHROMA_DB_DIR")):
            s.chroma_db_dir = _resolve_path(raw)
        if (raw := _env("QA_LOG_DIR")):
            s.qa_log_dir = _resolve_path(raw)
        if (raw := _env("REGISTRY_PATH")):
            s.registry_path = _resolve_path(raw)
        if (raw := _env("EXCEL_SUMMARY_DIR")):
            s.excel_summary_dir = _resolve_path(raw)

        s.google_api_key = _env("GOOGLE_API_KEY")
        s.generation_model = _env("GENERATION_MODEL", s.generation_model) or s.generation_model
        s.fallback_generation_model = (
            _env("FALLBACK_GENERATION_MODEL", s.fallback_generation_model)
            or s.fallback_generation_model
        )
        s.excel_summary_model = (
            _env("EXCEL_SUMMARY_MODEL", s.excel_summary_model) or s.excel_summary_model
        )

        provider = (_env("EMBEDDING_PROVIDER", s.embedding_provider) or s.embedding_provider).lower()
        if provider not in {"gemini", "local"}:
            provider = "gemini"
        s.embedding_provider = provider
        s.gemini_embedding_model = (
            _env("GEMINI_EMBEDDING_MODEL", s.gemini_embedding_model) or s.gemini_embedding_model
        )
        s.local_embedding_model = (
            _env("LOCAL_EMBEDDING_MODEL", s.local_embedding_model) or s.local_embedding_model
        )

        s.top_k = _env_int("TOP_K", s.top_k)
        s.chunk_size = _env_int("CHUNK_SIZE", s.chunk_size)
        s.chunk_overlap = _env_int("CHUNK_OVERLAP", s.chunk_overlap)
        s.enable_query_rewrite = _env_bool("ENABLE_QUERY_REWRITE", s.enable_query_rewrite)
        s.enable_excel_summary = _env_bool("ENABLE_EXCEL_SUMMARY", s.enable_excel_summary)
        s.max_context_chars = _env_int("MAX_CONTEXT_CHARS", s.max_context_chars)
        s.max_chunks_per_file = _env_int("MAX_CHUNKS_PER_FILE", s.max_chunks_per_file)
        s.request_timeout_seconds = _env_int("REQUEST_TIMEOUT_SECONDS", s.request_timeout_seconds)
        s.chroma_collection = _env("CHROMA_COLLECTION", s.chroma_collection) or s.chroma_collection

        return s

    # ------------------------------------------------------------------
    # 헬퍼
    # ------------------------------------------------------------------
    def ensure_dirs(self) -> None:
        """필수 폴더가 없으면 생성한다."""
        for p in [
            self.raw_data_dir,
            self.processed_data_dir,
            self.documents_dir,
            self.chunks_dir,
            self.excel_summary_dir,
            self.chroma_db_dir,
            self.qa_log_dir,
            self.registry_path.parent,
        ]:
            p.mkdir(parents=True, exist_ok=True)
        # 카테고리 폴더
        for sub in self.category_dirs.values():
            (self.raw_data_dir / sub).mkdir(parents=True, exist_ok=True)

    def category_to_dir(self, uploaded_category: str) -> Path:
        sub = self.category_dirs.get(uploaded_category, "misc")
        path = self.raw_data_dir / sub
        path.mkdir(parents=True, exist_ok=True)
        return path

    def has_api_key(self) -> bool:
        return bool(self.google_api_key)


# 모듈 로드 시점에 한 번 생성. 다른 모듈에서는 `from src.config import settings` 로 사용.
settings: Settings = Settings.from_env()
settings.ensure_dirs()
