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


def _env_float(key: str, default: float) -> float:
    raw = _env(key)
    if raw is None:
        return default
    try:
        return float(raw)
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
    top_k: int = 5
    chunk_size: int = 1200
    chunk_overlap: int = 200
    enable_query_rewrite: bool = False
    enable_excel_summary: bool = False
    max_context_chars: int = 6000
    max_chunks_per_file: int = 2
    request_timeout_seconds: int = 60
    chroma_collection: str = "work_knowledge"

    # ----- 검색 정밀도 (Retrieval precision) -----
    min_similarity_score: float = 0.35
    min_final_score: float = 0.30
    min_retrieved_chunks: int = 1
    use_mmr: bool = True
    mmr_lambda: float = 0.7

    # ----- Date / Topic-aware retrieval -----
    date_exact_match_boost: float = 1.25
    date_mismatch_penalty: float = 0.55
    enable_date_filter: bool = False
    topic_match_boost: float = 1.20
    topic_mismatch_penalty: float = 0.80

    # ----- 출력 비식별화 (Anonymization) -----
    # 원본 raw 문서는 절대 변경하지 않는다. UI/QA 답변/prompt context 출력에만 적용.
    anonymize_output: bool = True
    show_raw_content: bool = False
    show_speaker_names: bool = False
    show_exact_timestamps: bool = False
    show_exact_dates: bool = False
    mask_mentions: bool = True
    mask_links: bool = True
    mask_file_names: bool = False
    anonymized_date_label: str = "업무일"
    anonymized_time_label: str = "시간대"

    # ----- LLM 기반 지식카드 정규화 (Task 1: 설정만 준비) -----
    enable_llm_normalization: bool = False
    llm_normalization_model: str = "gemini-2.5-flash-lite"
    normalization_output_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "normalized"
    )
    normalization_cache_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "normalized" / "cache"
    )
    normalization_max_chars_per_call: int = 18000
    normalization_max_cards_per_file: int = 30
    normalization_temperature: float = 0.1
    normalization_use_anonymized_input: bool = True
    normalization_save_json: bool = True
    normalization_save_markdown: bool = True
    normalization_card_source_weight: float = 1.25
    normalization_parent_raw_top_k: int = 3

    # ----- KnowledgeCard 우선 retrieval (Task 6) -----
    # PRIORITIZE_KNOWLEDGE_CARDS=true 면 reranker/retriever 가 knowledge_card chunk 를
    # raw chunk 보다 우선 반환한다. raw chunk 는 fallback / parent evidence 용도로 유지.
    prioritize_knowledge_cards: bool = True
    knowledge_card_content_boost: float = 1.35
    workflow_card_boost: float = 1.30
    checklist_card_boost: float = 1.25
    faq_card_boost: float = 1.20
    decision_card_boost: float = 1.20
    communication_template_boost: float = 1.20
    glossary_card_boost: float = 1.10
    raw_evidence_boost: float = 0.85
    enable_parent_raw_evidence: bool = True
    parent_raw_evidence_top_k: int = 2

    # ----- KnowledgeCard 중심 답변 (Task 7) -----
    # ANSWER_WITH_KNOWLEDGE_CARDS=true 면 QA prompt 가 primary_card 를 1차 근거로 사용한다.
    # raw evidence/fallback 은 보조 또는 fallback 용도로만 사용한다.
    answer_with_knowledge_cards: bool = True
    max_primary_cards: int = 5
    max_raw_evidence_chunks: int = 3
    include_raw_evidence_appendix: bool = True
    knowledge_card_answer_template_version: str = "knowledge_card_v1"

    # ----- 카테고리 → raw 폴더 -----
    category_dirs: dict = field(default_factory=lambda: {
        "slack": "slack_manual",
        "guide": "guide",
        "kakao": "kakao",
        "excel": "excel",
        "misc": "misc",
    })

    # ----- 카테고리 → source_weight 기본값 -----
    # 검색 결과 precision 강화를 위해 source_type 별 가중치를 분리.
    #   guide                 : 절차/공식 문서 → 1.0
    #   excel_summary         : 한국어 업무 요약 (검색 1차 대상) → 1.1
    #   excel_raw_table       : 숫자 근거용 → 0.95
    #   slack_manual          : 실무 맥락/히스토리 보강 → 0.8 (guide 보다 낮음)
    #   word/markdown/txt     : 일반 문서 → 0.75/0.7/0.65
    #   kakao                 : 잡음이 많음 → 0.45
    #   misc                  : 기본 → 0.5
    category_source_weight: dict = field(default_factory=lambda: {
        "excel_summary": 1.1,
        "excel_raw_table": 0.95,
        "excel": 1.0,
        "guide": 1.0,
        "slack_manual": 0.8,
        "slack": 0.8,
        "word": 0.75,
        "markdown": 0.7,
        "txt": 0.65,
        "kakao": 0.45,
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

        # 검색 정밀도
        s.min_similarity_score = _env_float("MIN_SIMILARITY_SCORE", s.min_similarity_score)
        s.min_final_score = _env_float("MIN_FINAL_SCORE", s.min_final_score)
        s.min_retrieved_chunks = _env_int("MIN_RETRIEVED_CHUNKS", s.min_retrieved_chunks)
        s.use_mmr = _env_bool("USE_MMR", s.use_mmr)
        s.mmr_lambda = _env_float("MMR_LAMBDA", s.mmr_lambda)

        # Date / Topic-aware retrieval
        s.date_exact_match_boost = _env_float("DATE_EXACT_MATCH_BOOST", s.date_exact_match_boost)
        s.date_mismatch_penalty = _env_float("DATE_MISMATCH_PENALTY", s.date_mismatch_penalty)
        s.enable_date_filter = _env_bool("ENABLE_DATE_FILTER", s.enable_date_filter)
        s.topic_match_boost = _env_float("TOPIC_MATCH_BOOST", s.topic_match_boost)
        s.topic_mismatch_penalty = _env_float("TOPIC_MISMATCH_PENALTY", s.topic_mismatch_penalty)

        # Anonymization
        s.anonymize_output = _env_bool("ANONYMIZE_OUTPUT", s.anonymize_output)
        s.show_raw_content = _env_bool("SHOW_RAW_CONTENT", s.show_raw_content)
        s.show_speaker_names = _env_bool("SHOW_SPEAKER_NAMES", s.show_speaker_names)
        s.show_exact_timestamps = _env_bool("SHOW_EXACT_TIMESTAMPS", s.show_exact_timestamps)
        s.show_exact_dates = _env_bool("SHOW_EXACT_DATES", s.show_exact_dates)
        s.mask_mentions = _env_bool("MASK_MENTIONS", s.mask_mentions)
        s.mask_links = _env_bool("MASK_LINKS", s.mask_links)
        s.mask_file_names = _env_bool("MASK_FILE_NAMES", s.mask_file_names)
        s.anonymized_date_label = (
            _env("ANONYMIZED_DATE_LABEL", s.anonymized_date_label) or s.anonymized_date_label
        )
        s.anonymized_time_label = (
            _env("ANONYMIZED_TIME_LABEL", s.anonymized_time_label) or s.anonymized_time_label
        )

        # LLM 기반 지식카드 정규화 (Task 1: 설정만 준비)
        s.enable_llm_normalization = _env_bool(
            "ENABLE_LLM_NORMALIZATION", s.enable_llm_normalization
        )
        s.llm_normalization_model = (
            _env("LLM_NORMALIZATION_MODEL", s.llm_normalization_model)
            or s.llm_normalization_model
        )
        if (raw := _env("NORMALIZATION_OUTPUT_DIR")):
            s.normalization_output_dir = _resolve_path(raw)
        if (raw := _env("NORMALIZATION_CACHE_DIR")):
            s.normalization_cache_dir = _resolve_path(raw)
        s.normalization_max_chars_per_call = _env_int(
            "NORMALIZATION_MAX_CHARS_PER_CALL", s.normalization_max_chars_per_call
        )
        s.normalization_max_cards_per_file = _env_int(
            "NORMALIZATION_MAX_CARDS_PER_FILE", s.normalization_max_cards_per_file
        )
        s.normalization_temperature = _env_float(
            "NORMALIZATION_TEMPERATURE", s.normalization_temperature
        )
        s.normalization_use_anonymized_input = _env_bool(
            "NORMALIZATION_USE_ANONYMIZED_INPUT", s.normalization_use_anonymized_input
        )
        s.normalization_save_json = _env_bool(
            "NORMALIZATION_SAVE_JSON", s.normalization_save_json
        )
        s.normalization_save_markdown = _env_bool(
            "NORMALIZATION_SAVE_MARKDOWN", s.normalization_save_markdown
        )
        s.normalization_card_source_weight = _env_float(
            "NORMALIZATION_CARD_SOURCE_WEIGHT", s.normalization_card_source_weight
        )
        s.normalization_parent_raw_top_k = _env_int(
            "NORMALIZATION_PARENT_RAW_TOP_K", s.normalization_parent_raw_top_k
        )

        # KnowledgeCard 우선 retrieval (Task 6)
        s.prioritize_knowledge_cards = _env_bool(
            "PRIORITIZE_KNOWLEDGE_CARDS", s.prioritize_knowledge_cards
        )
        s.knowledge_card_content_boost = _env_float(
            "KNOWLEDGE_CARD_CONTENT_BOOST", s.knowledge_card_content_boost
        )
        s.workflow_card_boost = _env_float("WORKFLOW_CARD_BOOST", s.workflow_card_boost)
        s.checklist_card_boost = _env_float("CHECKLIST_CARD_BOOST", s.checklist_card_boost)
        s.faq_card_boost = _env_float("FAQ_CARD_BOOST", s.faq_card_boost)
        s.decision_card_boost = _env_float("DECISION_CARD_BOOST", s.decision_card_boost)
        s.communication_template_boost = _env_float(
            "COMMUNICATION_TEMPLATE_BOOST", s.communication_template_boost
        )
        s.glossary_card_boost = _env_float("GLOSSARY_CARD_BOOST", s.glossary_card_boost)
        s.raw_evidence_boost = _env_float("RAW_EVIDENCE_BOOST", s.raw_evidence_boost)
        s.enable_parent_raw_evidence = _env_bool(
            "ENABLE_PARENT_RAW_EVIDENCE", s.enable_parent_raw_evidence
        )
        s.parent_raw_evidence_top_k = _env_int(
            "PARENT_RAW_EVIDENCE_TOP_K", s.parent_raw_evidence_top_k
        )

        # KnowledgeCard 중심 답변 (Task 7)
        s.answer_with_knowledge_cards = _env_bool(
            "ANSWER_WITH_KNOWLEDGE_CARDS", s.answer_with_knowledge_cards
        )
        s.max_primary_cards = _env_int("MAX_PRIMARY_CARDS", s.max_primary_cards)
        s.max_raw_evidence_chunks = _env_int(
            "MAX_RAW_EVIDENCE_CHUNKS", s.max_raw_evidence_chunks
        )
        s.include_raw_evidence_appendix = _env_bool(
            "INCLUDE_RAW_EVIDENCE_APPENDIX", s.include_raw_evidence_appendix
        )
        s.knowledge_card_answer_template_version = (
            _env(
                "KNOWLEDGE_CARD_ANSWER_TEMPLATE_VERSION",
                s.knowledge_card_answer_template_version,
            )
            or s.knowledge_card_answer_template_version
        )

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
