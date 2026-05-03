"""
excel_summarizer.py
===================
Excel raw_table_text → Gemini 한국어 업무 설명 (Markdown) 요약 생성기.

특징:
- raw_table_hash 가 같은 입력은 재요약하지 않는다 (cache).
- summary 결과는 SummaryStore 가 디스크에 저장한다.
- 호출 비용을 줄이기 위해 너무 긴 raw_table_text 는 cap 한다.
- summary 결과는 ParsedSection (content_type="excel_summary") 형태로 변환할 수
  있는 헬퍼를 함께 제공한다 (chunker 가 그대로 받아서 처리할 수 있게).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import settings
from src.logger import get_logger
from src.rag.gemini_client import GeminiClient, GeminiError, get_default_client
from src.schemas import ExcelSummary, ParsedSection
from src.summarization.summary_prompt import build_excel_summary_prompt
from src.summarization.summary_store import SummaryStore
from src.utils.cost_utils import tracker
from src.utils.hash_utils import short_hash, text_hash
from src.utils.time_utils import now_iso

log = get_logger(__name__)

# 요약 호출 시 단일 입력 글자 cap (원문이 너무 길면 잘라서 보낸다)
MAX_INPUT_CHARS = 8000


def _cap_text(text: str, cap: int = MAX_INPUT_CHARS) -> str:
    if not text:
        return ""
    if len(text) <= cap:
        return text
    head = text[: cap - 200]
    tail = text[-200:]
    return f"{head}\n\n... (중략, 원본은 길어서 일부만 전달) ...\n\n{tail}"


class ExcelSummarizer:
    def __init__(
        self,
        client: Optional[GeminiClient] = None,
        store: Optional[SummaryStore] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.client = client or get_default_client()
        self.store = store or SummaryStore()
        self.model_name: str = model_name or settings.excel_summary_model

    # ------------------------------------------------------------------
    def summarize_section(
        self,
        *,
        document_id: str,
        file_name: str,
        sheet_name: str,
        raw_table_text: str,
        table_range: Optional[str],
        source_raw_path: str,
        force: bool = False,
    ) -> ExcelSummary:
        raw_table_hash = text_hash(raw_table_text or "")
        cached = None if force else self.store.load(raw_table_hash, file_name, sheet_name)
        if cached:
            tracker.add_cache_hit()
            log.info("[summary cache hit] %s/%s", file_name, sheet_name)
            return cached

        tracker.add_cache_miss()

        prompt = build_excel_summary_prompt(
            file_name=file_name,
            sheet_name=sheet_name,
            table_range=table_range,
            raw_table_text=_cap_text(raw_table_text),
        )

        try:
            md = self.client.generate_text(
                prompt,
                model=self.model_name,
                temperature=0.2,
                max_output_tokens=2000,
            )
        except GeminiError as e:
            log.error("Excel summary 생성 실패: %s/%s (%s)", file_name, sheet_name, e)
            raise

        if not md or not md.strip():
            raise GeminiError(
                f"Excel summary 응답이 비었습니다: {file_name}/{sheet_name}. 모델/quota 확인 필요."
            )

        tracker.add_excel_summary()

        summary_id = f"sum_{short_hash(raw_table_hash, length=10)}_{short_hash(sheet_name, length=4)}"
        summary = ExcelSummary(
            summary_id=summary_id,
            document_id=document_id,
            file_name=file_name,
            sheet_name=sheet_name,
            table_range=table_range,
            raw_table_hash=raw_table_hash,
            summary_text=md.strip(),
            summary_markdown_path="",  # store.save 에서 채움
            source_raw_path=source_raw_path,
            created_at=now_iso(),
            model_name=self.model_name,
            metadata={
                "char_count": len(md),
                "raw_table_chars": len(raw_table_text or ""),
            },
        )
        self.store.save(summary)
        log.info("[summary saved] %s/%s", file_name, sheet_name)
        return summary

    # ------------------------------------------------------------------
    def to_parsed_section(
        self,
        summary: ExcelSummary,
        document_id: str,
    ) -> ParsedSection:
        """ExcelSummary → ParsedSection (content_type='excel_summary')."""
        section_id = f"sec_{summary.summary_id}"
        return ParsedSection(
            section_id=section_id,
            document_id=document_id,
            section_title=summary.sheet_name,
            content_type="excel_summary",
            content=summary.summary_text,
            metadata={
                "sheet_name": summary.sheet_name,
                "raw_table_hash": summary.raw_table_hash,
                "summary_type": "korean_business_summary",
                "summary_id": summary.summary_id,
                "summary_markdown_path": summary.summary_markdown_path,
                "table_range": summary.table_range,
                "model_name": summary.model_name,
                "source_raw_path": summary.source_raw_path,
            },
        )
