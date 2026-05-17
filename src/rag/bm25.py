"""
Small BM25 keyword scorer used by optional hybrid retrieval.

This module is deliberately independent from ChromaDB and the retriever so it
can be tested without embeddings, external APIs, or local storage.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence


_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+(?:[&+./_-][0-9A-Za-z가-힣]+)*\+?")
_SEPARATOR_RE = re.compile(r"[&+./_-]+")


def tokenize_bm25(text: str) -> List[str]:
    """
    Tokenize Korean/English business text for keyword retrieval.

    The tokenizer keeps abbreviation-like tokens such as ``t&d`` and also adds a
    compact alias such as ``td`` so users can search either form.
    """
    if not text:
        return []

    tokens: List[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        token = match.group(0).strip("_")
        token = token.strip()
        if not token:
            continue
        tokens.append(token)

        compact = _SEPARATOR_RE.sub("", token).strip()
        if compact and compact != token:
            tokens.append(compact)

        if _SEPARATOR_RE.search(token):
            for part in _SEPARATOR_RE.split(token):
                part = part.strip()
                if part:
                    tokens.append(part)

    return tokens


@dataclass(frozen=True)
class BM25Document:
    document_id: str
    text: str
    payload: Any = None


@dataclass(frozen=True)
class BM25Result:
    document_id: str
    score: float
    rank: int
    normalized_score: float
    payload: Any = None


@dataclass
class _IndexedDocument:
    document_id: str
    tokens: List[str]
    term_freq: Dict[str, int]
    payload: Any = None


@dataclass
class BM25Scorer:
    documents: Sequence[BM25Document]
    k1: float = 1.5
    b: float = 0.75
    _indexed: List[_IndexedDocument] = field(init=False, repr=False)
    _doc_freq: Dict[str, int] = field(init=False, repr=False)
    _avgdl: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        indexed: List[_IndexedDocument] = []
        doc_freq: Dict[str, int] = {}

        for doc in self.documents:
            tokens = tokenize_bm25(doc.text)
            term_freq: Dict[str, int] = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
            for token in term_freq:
                doc_freq[token] = doc_freq.get(token, 0) + 1
            indexed.append(
                _IndexedDocument(
                    document_id=doc.document_id,
                    tokens=tokens,
                    term_freq=term_freq,
                    payload=doc.payload,
                )
            )

        self._indexed = indexed
        self._doc_freq = doc_freq
        self._avgdl = (
            sum(len(doc.tokens) for doc in indexed) / float(len(indexed))
            if indexed
            else 0.0
        )

    def score(self, query: str, document_id: str) -> float:
        """Return the raw BM25 score for one indexed document."""
        query_terms = _unique_terms(tokenize_bm25(query))
        if not query_terms or not self._indexed or self._avgdl <= 0.0:
            return 0.0

        by_id = {doc.document_id: doc for doc in self._indexed}
        doc = by_id.get(document_id)
        if doc is None:
            return 0.0
        return self._score_terms(query_terms, doc)

    def search(self, query: str, top_k: int = 10) -> List[BM25Result]:
        """Return ranked BM25 results with normalized scores in ``[0, 1]``."""
        if top_k <= 0:
            return []

        query_terms = _unique_terms(tokenize_bm25(query))
        if not query_terms or not self._indexed or self._avgdl <= 0.0:
            return []

        scored: List[tuple[str, float, Any]] = []
        for doc in self._indexed:
            score = self._score_terms(query_terms, doc)
            if score > 0.0:
                scored.append((doc.document_id, score, doc.payload))

        scored.sort(key=lambda item: item[1], reverse=True)
        scored = scored[: int(top_k)]
        max_score = scored[0][1] if scored else 0.0

        results: List[BM25Result] = []
        for rank, (doc_id, score, payload) in enumerate(scored, start=1):
            normalized = score / max_score if max_score > 0.0 else 0.0
            results.append(
                BM25Result(
                    document_id=doc_id,
                    score=score,
                    rank=rank,
                    normalized_score=normalized,
                    payload=payload,
                )
            )
        return results

    def _score_terms(self, query_terms: Sequence[str], doc: _IndexedDocument) -> float:
        n_docs = len(self._indexed)
        doc_len = len(doc.tokens)
        if doc_len == 0:
            return 0.0

        score = 0.0
        for term in query_terms:
            tf = doc.term_freq.get(term, 0)
            if tf <= 0:
                continue
            df = self._doc_freq.get(term, 0)
            # Robertson/Sparck Jones IDF with +1 smoothing keeps rare business
            # keywords useful while avoiding negative scores for common terms.
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denom = tf + self.k1 * (1.0 - self.b + self.b * doc_len / self._avgdl)
            score += idf * (tf * (self.k1 + 1.0)) / denom
        return float(score)


def _unique_terms(tokens: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out
