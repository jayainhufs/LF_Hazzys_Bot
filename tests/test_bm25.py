from __future__ import annotations

from src.rag.bm25 import BM25Document, BM25Scorer, tokenize_bm25


def test_tokenizer_preserves_acronyms_and_compact_aliases():
    tokens = tokenize_bm25("ASC ROAS CPA T&D Advantage+ 피드광고 정산시트")

    assert "asc" in tokens
    assert "roas" in tokens
    assert "cpa" in tokens
    assert "t&d" in tokens
    assert "td" in tokens
    assert "advantage+" in tokens
    assert "advantage" in tokens
    assert "피드광고" in tokens
    assert "정산시트" in tokens


def test_bm25_ranks_exact_business_keyword_match_first():
    scorer = BM25Scorer([
        BM25Document("doc_meta", "ASC Advantage+ ROAS 캠페인보드 피드광고 셋팅"),
        BM25Document("doc_settlement", "정산시트 입금 거래명세서 발행"),
        BM25Document("doc_general", "일반 업무 가이드 체크리스트"),
    ])

    results = scorer.search("ASC ROAS 캠페인보드", top_k=3)

    assert results
    assert results[0].document_id == "doc_meta"
    assert results[0].rank == 1
    assert results[0].normalized_score == 1.0


def test_bm25_handles_korean_keyword_query():
    scorer = BM25Scorer([
        BM25Document("doc_feed", "메타 피드광고 카탈로그 캠페인 셋팅 절차"),
        BM25Document("doc_calc", "월말 정산시트 검수 절차"),
    ])

    results = scorer.search("피드광고 카탈로그", top_k=2)

    assert [r.document_id for r in results] == ["doc_feed"]


def test_bm25_empty_query_returns_no_results():
    scorer = BM25Scorer([BM25Document("doc", "ASC ROAS")])

    assert scorer.search("") == []
    assert scorer.search("   ") == []


def test_bm25_top_k_zero_returns_no_results():
    scorer = BM25Scorer([BM25Document("doc", "ASC ROAS")])

    assert scorer.search("ASC", top_k=0) == []


def test_bm25_negative_top_k_returns_no_results():
    scorer = BM25Scorer([BM25Document("doc", "ASC ROAS")])

    assert scorer.search("ASC", top_k=-1) == []


def test_bm25_query_is_case_insensitive():
    docs = [
        BM25Document("doc_meta", "ASC ROAS campaign board"),
        BM25Document("doc_calc", "CPA settlement sheet"),
    ]
    scorer = BM25Scorer(docs)

    upper = scorer.search("ASC ROAS", top_k=2)
    lower = scorer.search("asc roas", top_k=2)
    mixed = scorer.search("Asc RoAs", top_k=2)

    assert [r.document_id for r in upper] == [r.document_id for r in lower]
    assert [r.document_id for r in upper] == [r.document_id for r in mixed]


def test_bm25_advantage_plus_alias_matches_document():
    scorer = BM25Scorer([
        BM25Document("doc_advantage", "Meta Advantage+ ASC ROAS setup"),
        BM25Document("doc_general", "General campaign checklist"),
    ])

    with_plus = scorer.search("Advantage+", top_k=2)
    without_plus = scorer.search("Advantage", top_k=2)

    assert with_plus
    assert without_plus
    assert with_plus[0].document_id == "doc_advantage"
    assert without_plus[0].document_id == "doc_advantage"


def test_duplicate_query_terms_do_not_change_ranking():
    docs = [
        BM25Document("doc_a", "ASC ROAS 캠페인"),
        BM25Document("doc_b", "ASC 정산시트"),
    ]

    single = BM25Scorer(docs).search("ASC ROAS", top_k=2)
    duplicate = BM25Scorer(docs).search("ASC ASC ROAS", top_k=2)

    assert [r.document_id for r in single] == [r.document_id for r in duplicate]
