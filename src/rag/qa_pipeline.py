"""
qa_pipeline.py
==============
업무 QA 전체 파이프라인.

Steps:
1) 질문 정제
2) (옵션) query rewrite
3) retriever 로 chunk 검색 + threshold + MMR + parent-child 보강 (retriever 내부)
4) 근거 부족(threshold 통과 chunk < MIN_RETRIEVED_CHUNKS) 이면 Gemini generation **호출하지 않고** 안내 메시지 반환
5) 충분한 근거가 있을 때만 prompt_builder 로 프롬프트 구성 → Gemini generation
6) qa_logs 에 JSON 저장 (generation_skipped 여부 포함)
7) (answer, retrieved_chunks, meta) 반환
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import settings
from src.logger import get_logger
from src.rag.embedder import Embedder, get_default_embedder
from src.rag.generator import Generator
from src.rag.prompt_builder import build_qa_prompt
from src.rag.query_rewriter import rewrite_query_if_enabled
from src.rag.retriever import Retriever
from src.schemas import QALog, RetrievedChunk
from src.utils.path_utils import ensure_dir
from src.utils.time_utils import now_compact, now_iso

log = get_logger(__name__)


INSUFFICIENT_EVIDENCE_MESSAGE = (
    "현재 적재된 자료만으로는 질문에 직접 답할 수 있는 근거가 부족합니다.\n"
    "검색된 후보 문서는 있었지만 관련도 기준을 통과하지 못했습니다.\n"
    "검색어를 더 구체화하거나, 관련 가이드/Slack 스레드/Excel 파일을 추가로 업로드한 뒤 다시 시도해 주세요."
)


class QAPipeline:
    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        generator: Optional[Generator] = None,
        embedder: Optional[Embedder] = None,
    ) -> None:
        self.embedder = embedder or get_default_embedder()
        self.retriever = retriever or Retriever(embedder=self.embedder)
        self.generator = generator or Generator()

    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
        uploaded_category: Optional[str] = None,
        enable_query_rewrite: Optional[bool] = None,
        save_log: bool = True,
        min_retrieved_chunks: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not question or not question.strip():
            return {
                "answer": "질문이 비어 있습니다.",
                "retrieved_chunks": [],
                "used_chunks": [],
                "model_name": "",
                "embedding_provider": self.embedder.provider,
                "embedding_model": self.embedder.model_name,
                "rewritten_query": None,
                "generation_skipped": True,
                "skip_reason": "empty_question",
                "retrieval_summary": {},
            }

        # 1) 정제
        q = question.strip()

        # 2) query rewrite
        rewritten = rewrite_query_if_enabled(q, enable=enable_query_rewrite)
        search_query = rewritten or q

        # 3) 검색 (진단 정보 포함)
        from src.rag.retriever import RetrievalDetails
        try:
            details = self.retriever.retrieve_with_details(
                search_query,
                top_k=top_k,
                uploaded_category=uploaded_category,
            )
        except Exception as e:  # noqa: BLE001
            log.error("검색 실패: %s", e)
            details = RetrievalDetails()

        passed: List[RetrievedChunk] = list(details.passed)
        candidates: List[RetrievedChunk] = list(details.candidates)
        summary = dict(details.summary)

        # 4) 근거 부족 판정
        threshold_n = int(
            min_retrieved_chunks
            if min_retrieved_chunks is not None
            else settings.min_retrieved_chunks
        )
        generation_skipped = False
        skip_reason: Optional[str] = None
        used_chunks: List[RetrievedChunk] = []
        used_model = ""
        prompt_chars = 0

        if len(passed) < threshold_n:
            generation_skipped = True
            skip_reason = (
                "no_candidates"
                if not candidates
                else "below_threshold"
            )
            answer = INSUFFICIENT_EVIDENCE_MESSAGE
            log.info(
                "QA: generation skipped (passed=%d / candidates=%d / threshold=%d)",
                len(passed), len(candidates), threshold_n,
            )
        else:
            # 5) prompt + generation
            prompt, used_chunks = build_qa_prompt(q, passed, rewritten_query=rewritten)
            prompt_chars = len(prompt or "")
            answer, used_model = self.generator.generate(prompt)

        # 6) log
        log_obj = QALog(
            question=q,
            rewritten_query=rewritten,
            answer=answer,
            retrieved_chunks=[
                {
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "file_name": c.file_name,
                    "section_title": c.section_title,
                    "content_type": c.content_type,
                    "uploaded_category": c.uploaded_category,
                    "source_type": c.source_type,
                    "score": c.score,
                    "final_score": c.final_score,
                    "passed_threshold": c.passed_threshold,
                    "filter_reason": c.filter_reason,
                    "preview": (c.content or "")[:600],
                    "metadata": c.metadata,
                }
                for c in (used_chunks or passed or candidates)
            ],
            model_name=used_model or "",
            embedding_provider=self.embedder.provider,
            created_at=now_iso(),
            metadata={
                "embedding_model": self.embedder.model_name,
                "uploaded_category_filter": uploaded_category or "all",
                "top_k": int(top_k or settings.top_k),
                "n_retrieved": len(candidates),
                "n_passed": len(passed),
                "n_dropped": max(0, len(candidates) - len(passed)),
                "min_similarity": summary.get("min_similarity"),
                "min_final": summary.get("min_final"),
                "max_per_file": summary.get("max_per_file"),
                "use_mmr": summary.get("use_mmr"),
                "min_retrieved_chunks": threshold_n,
                "generation_skipped": generation_skipped,
                "skip_reason": skip_reason,
                "prompt_chars": prompt_chars,
                "used_chunks": len(used_chunks),
                "query_class": summary.get("query_class"),
            },
        )
        if save_log:
            try:
                self._save_log(log_obj)
            except Exception as e:  # noqa: BLE001
                log.warning("QA 로그 저장 실패: %s", e)

        return {
            "answer": answer,
            "retrieved_chunks": passed if passed else candidates,
            "all_candidates": candidates,
            "used_chunks": used_chunks,
            "model_name": used_model,
            "embedding_provider": self.embedder.provider,
            "embedding_model": self.embedder.model_name,
            "rewritten_query": rewritten,
            "generation_skipped": generation_skipped,
            "skip_reason": skip_reason,
            "retrieval_summary": summary,
            "prompt_chars": prompt_chars,
        }

    # ------------------------------------------------------------------
    def _save_log(self, log_obj: QALog) -> Path:
        ensure_dir(settings.qa_log_dir)
        path = settings.qa_log_dir / f"qa_{now_compact()}.json"
        path.write_text(
            json.dumps(log_obj.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path
