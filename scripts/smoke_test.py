"""
smoke_test.py
=============
외부 API 호출 없이 프로젝트의 핵심 모듈/파이프라인이 정상인지 빠르게 검증한다.

확인 항목:
1) config 로딩
2) src 패키지 전부 import
3) txt_parser, markdown_parser 동작
4) (openpyxl 사용 가능 시) excel_parser 동작
5) chunker + parent-child link 동작
6) cleaner / normalizer 동작
7) GOOGLE_API_KEY 없을 때 GeminiClient 가 graceful 한 한국어 에러를 발생시키는지
8) (chromadb 사용 가능 시) VectorStore 임시 디렉터리에서 collection 생성 가능한지

사용법:
    python scripts/smoke_test.py

이 스크립트는 외부 네트워크/API Key 가 없어도 통과해야 한다.
종료 코드: 모두 통과하면 0, 하나라도 실패하면 2.
"""
from __future__ import annotations

# sys.path 보정 (직접 실행 시에도 src 를 import 가능하게)
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import importlib
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import List, Tuple


def _section(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def _check(label: str, fn) -> Tuple[str, bool, str]:
    try:
        fn()
        print(f"  [PASS] {label}")
        return (label, True, "")
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {label}")
        print(f"         {type(e).__name__}: {e}")
        traceback.print_exc()
        return (label, False, f"{type(e).__name__}: {e}")


def main() -> int:
    results: List[Tuple[str, bool, str]] = []

    # ---------------------------------------------------------------- 1
    _section("1) config 로딩")
    def _t1():
        from src.config import settings
        assert settings.app_name
        assert isinstance(settings.project_root, Path)
        # 필수 디렉터리는 import 시 자동 생성되어야 한다
        for p in [
            settings.raw_data_dir,
            settings.processed_data_dir,
            settings.documents_dir,
            settings.chunks_dir,
            settings.excel_summary_dir,
            settings.chroma_db_dir,
            settings.qa_log_dir,
            settings.registry_path.parent,
        ]:
            assert p.exists(), f"누락된 디렉터리: {p}"
    results.append(_check("config 로딩 + 필수 디렉터리 존재", _t1))

    # ---------------------------------------------------------------- 2
    _section("2) src 패키지 import")
    modules = [
        "src.logger", "src.utils.path_utils", "src.utils.hash_utils",
        "src.utils.time_utils", "src.utils.encoding_utils",
        "src.utils.cost_utils", "src.utils.token_utils",
        "src.schemas",
        "src.preprocessing.cleaner", "src.preprocessing.normalizer",
        "src.preprocessing.chunker", "src.preprocessing.deduplicator",
        "src.summarization.summary_prompt", "src.summarization.summary_store",
        "src.summarization.excel_summarizer",
        "src.storage.document_store", "src.storage.file_registry",
        "src.ingestion.txt_parser", "src.ingestion.markdown_parser",
        "src.ingestion.kakao_parser", "src.ingestion.slack_manual_parser",
        "src.ingestion.metadata_extractor", "src.ingestion.file_router",
        "src.rag.gemini_client", "src.rag.embedder", "src.rag.local_embedder",
        "src.rag.reranker", "src.rag.query_rewriter",
        "src.rag.prompt_builder", "src.rag.generator",
        "src.evaluation.retrieval_eval", "src.evaluation.embedding_eval",
        "src.evaluation.qa_eval",
        "src.pipeline",
    ]
    def _t2():
        for m in modules:
            importlib.import_module(m)
    results.append(_check(f"src 패키지 {len(modules)}개 import", _t2))

    # ---------------------------------------------------------------- 3
    _section("3) txt_parser / markdown_parser 동작")
    def _t3():
        from src.ingestion.txt_parser import parse_txt
        from src.ingestion.markdown_parser import parse_markdown

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            t = tdp / "memo.txt"
            t.write_text("ROAS 는 광고 수익률 지표이다.\n캠페인 세팅 점검.", encoding="utf-8")
            sec = parse_txt(t, document_id="d", uploaded_category="misc")
            assert len(sec) == 1
            assert "ROAS" in sec[0]["content"]

            # CP949 fallback
            t2 = tdp / "memo_cp949.txt"
            t2.write_bytes("한글 인코딩 테스트".encode("cp949"))
            sec2 = parse_txt(t2, document_id="d", uploaded_category="misc")
            assert len(sec2) == 1
            assert "한글 인코딩" in sec2[0]["content"]

            md = tdp / "guide.md"
            md.write_text("# 가이드\n## ROAS 기준\n2.0 이상 권장.", encoding="utf-8")
            sec3 = parse_markdown(md, document_id="d", uploaded_category="guide")
            assert sec3
            titles = {s["section_title"] for s in sec3}
            assert "ROAS 기준" in titles
    results.append(_check("txt + markdown parser 기본 동작 + CP949 fallback", _t3))

    # ---------------------------------------------------------------- 4
    _section("4) excel_parser 동작 (openpyxl 사용 가능 시)")
    def _t4():
        try:
            from openpyxl import Workbook
        except ImportError:
            print("  (openpyxl 미설치 -> skip)")
            return
        from src.ingestion.excel_parser import parse_excel
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "guide.xlsx"
            wb = Workbook()
            ws = wb.active
            ws.title = "캠페인세팅"
            ws.append(["캠페인", "예산", "ROAS"])
            ws.append(["A", 1000000, 3.2])
            ws.append(["B", 500000, 2.5])
            wb.save(p)
            wb.close()
            sec = parse_excel(p, document_id="d")
            assert sec, "Excel 시트 1개 이상 반환되어야 함"
            assert sec[0]["content_type"] == "excel_raw_table"
            assert "캠페인" in sec[0]["content"]
    results.append(_check("Excel parser (시트 -> raw_table_text)", _t4))

    # ---------------------------------------------------------------- 5
    _section("5) chunker + Excel parent-child link")
    def _t5():
        from src.preprocessing.chunker import chunk_sections, link_excel_parent_child, get_source_weight
        from src.schemas import Document, ParsedSection

        doc = Document(
            document_id="doc_t5",
            source_type="excel",
            uploaded_category="excel",
            file_name="t.xlsx",
            file_path="data/raw/excel/t.xlsx",
            file_hash="0" * 64,
            title="t",
            created_at="",
            ingested_at="",
            metadata={},
        )
        secs = [
            ParsedSection(
                section_id="s1", document_id=doc.document_id,
                section_title="캠페인세팅", content_type="excel_summary",
                content="# 1. 시트/표 개요\n캠페인 세팅 가이드 ...",
                metadata={"sheet_name": "캠페인세팅"},
            ),
            ParsedSection(
                section_id="s2", document_id=doc.document_id,
                section_title="캠페인세팅", content_type="excel_raw_table",
                content="컬럼\t값\nA\t1\nB\t2",
                metadata={"sheet_name": "캠페인세팅"},
            ),
        ]
        chunks = chunk_sections(document=doc, sections=secs)
        chunks = link_excel_parent_child(chunks)
        assert chunks
        # parent-child
        sums = [c for c in chunks if c.content_type == "excel_summary"]
        raws = [c for c in chunks if c.content_type == "excel_raw_table"]
        assert sums and raws
        for r in raws:
            assert r.parent_chunk_id == sums[0].chunk_id, "parent_chunk_id 가 연결되어야 함"
        # source_weight
        assert get_source_weight("excel", "excel_summary", "excel") > get_source_weight("excel", "excel_raw_table", "excel")
        # context header 포함
        for c in chunks:
            assert "[파일명]" in c.content
            assert "[카테고리]" in c.content
    results.append(_check("chunker + parent-child + source_weight + context header", _t5))

    # ---------------------------------------------------------------- 6
    _section("6) cleaner / normalizer")
    def _t6():
        from src.preprocessing.cleaner import clean_text
        from src.preprocessing.normalizer import normalize_for_embedding

        msg = "[멘션]\u200b 안녕\n\n\n오늘 https://example.com/foo 점검."
        cleaned = clean_text(msg, "text")
        assert "\u200b" not in cleaned
        normalized = normalize_for_embedding(cleaned)
        assert "[link:example.com]" in normalized
    results.append(_check("cleaner / normalizer 정제", _t6))

    # ---------------------------------------------------------------- 7
    _section("7) GOOGLE_API_KEY 없을 때 GeminiClient graceful 실패")
    def _t7():
        from src.rag.gemini_client import GeminiClient, GeminiConfig, GeminiError
        cfg = GeminiConfig(api_key=None, timeout_seconds=5, max_retries=0)
        c = GeminiClient(cfg)
        try:
            c.generate_text("hello")
        except GeminiError as e:
            msg = str(e)
            assert ("GOOGLE_API_KEY" in msg) or ("키" in msg), f"메시지가 친절하지 않음: {msg}"
            return
        raise AssertionError("API key 없음에도 GeminiError 가 발생하지 않음")
    results.append(_check("API Key 없음 -> GeminiError 친절 메시지", _t7))

    # ---------------------------------------------------------------- 8
    _section("8) ChromaDB collection 생성 (chromadb 설치 시)")
    def _t8():
        try:
            import chromadb  # noqa: F401
        except ImportError:
            print("  (chromadb 미설치 -> skip)")
            return
        from src.storage.vector_store import VectorStore
        with tempfile.TemporaryDirectory() as td:
            vs = VectorStore(persist_dir=Path(td), collection_name="smoke_test")
            stats = vs.stats()
            assert stats["count"] >= 0
    results.append(_check("VectorStore 임시 디렉터리에서 collection 생성", _t8))

    # ---------------------------------------------------------------- summary
    print()
    print("=" * 60)
    print("SMOKE TEST 요약")
    print("=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - passed
    for name, ok, err in results:
        flag = "PASS" if ok else "FAIL"
        suffix = "" if ok else f" :: {err}"
        print(f"  [{flag}] {name}{suffix}")
    print()
    print(f"총 {len(results)}건 중 PASS={passed}, FAIL={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
