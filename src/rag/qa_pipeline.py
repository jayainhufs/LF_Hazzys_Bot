"""
qa_pipeline.py
==============
업무 QA 전체 파이프라인.

Steps:
1) 질문 정제
2) (옵션) query rewrite
3) retriever 로 chunk 검색
4) reranker 로 재정렬 (retriever 안에서 이미 적용됨)
5) Excel summary -> raw_table 자식 보강 (retriever 안에서 처리)
6) prompt_builder 로 프롬프트 구성
7) generator 로 답변 생성
8) qa_logs 에 JSON 저장
9) (answer, retrieved_chunks, meta) 반환
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import settings
from src.logger import get_logger
from src.rag.embedder import Embedder, get_default_embedder
from src.rag.generator import Generator
from src.rag.prompt_builder import build_no_context_answer_prompt, build_qa_prompt
from src.rag.query_rewriter import rewrite_query_if_enabled
from src.rag.retriever import Retriever
from src.schemas import QALog, RetrievedChunk
from src.utils.path_utils import ensure_dir
from src.utils.time_utils import now_compact, now_iso

log = get_logger(__name__)


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
    ) -> Dict[str, Any]:
        if not question or not question.strip():
            return {
                "answer": "질문이 비어 있습니다.",
                "retrieved_chunks": [],
                "model_name": "",
                "embedding_provider": self.embedder.provider,
                "rewritten_query": None,
            }

        # 1) 정제
        q = question.strip()

        # 2) query rewrite
        rewritten = rewrite_query_if_enabled(q, enable=enable_query_rewrite)
        search_query = rewritten or q

        # 3+4+5) 검색 + rerank + parent-child 보강
        try:
            chunks: List[RetrievedChunk] = self.retriever.retrieve(
                search_query,
                top_k=top_k,
                uploaded_category=uploaded_category,
            )
        except Exception as e:
            log.error("검색 실패: %s", e)
            chunks = []

        # 6) prompt
        if chunks:
            prompt, used_chunks = build_qa_prompt(q, chunks, rewritten_query=rewritten)
        else:
            prompt = build_no_context_answer_prompt(q)
            used_chunks = []

        # 7) generation
        answer, used_model = self.generator.generate(prompt)

        # 8) log
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
                    "preview": (c.content or "")[:600],
                    "metadata": c.metadata,
                }
                for c in (used_chunks or chunks)
            ],
            model_name=used_model or "",
            embedding_provider=self.embedder.provider,
            created_at=now_iso(),
            metadata={
                "embedding_model": self.embedder.model_name,
                "uploaded_category_filter": uploaded_category or "all",
                "top_k": int(top_k or settings.top_k),
                "n_retrieved": len(chunks),
            },
        )
        if save_log:
            try:
                self._save_log(log_obj)
            except Exception as e:  # noqa: BLE001
                log.warning("QA 로그 저장 실패: %s", e)

        return {
            "answer": answer,
            "retrieved_chunks": chunks,
            "used_chunks": used_chunks,
            "model_name": used_model,
            "embedding_provider": self.embedder.provider,
            "embedding_model": self.embedder.model_name,
            "rewritten_query": rewritten,
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
