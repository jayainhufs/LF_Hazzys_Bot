"""
reranker.py
===========
1차 MVP rule-based reranker.

final_score = similarity_score * source_weight * category_boost * content_type_boost * recency_score

- BM25 / LLM reranker / cross-encoder 는 아직 사용하지 않는다.
- query keyword heuristic 으로 카테고리 boost 를 살짝 보정한다.
- diversity 보정용 MMR 비슷한 휴리스틱(`apply_diversity_penalty`) 도 함께 제공한다.

추후 Gemini 기반 LLM reranker / cross-encoder reranker 를 붙일 수 있도록
인터페이스만 미리 정의해 둔다.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from src.config import settings
from src.logger import get_logger
from src.schemas import RetrievedChunk

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# 가중치 테이블
# ---------------------------------------------------------------------------
# uploaded_category 기본 boost (질문 분류 없이 기본 적용)
DEFAULT_CATEGORY_BOOST: Dict[str, float] = {
    "guide": 1.05,
    "excel": 1.05,
    "slack": 1.0,
    "kakao": 0.9,
    "misc": 0.9,
}

# content_type 별 boost
CONTENT_TYPE_BOOST: Dict[str, float] = {
    "guide": 1.15,
    "excel_summary": 1.15,
    "excel_raw_table": 1.0,
    "table": 1.0,
    "text": 1.0,
    "conversation": 0.85,
    "note": 0.8,
}


# ---------------------------------------------------------------------------
# Query 분류 (keyword heuristic)
# ---------------------------------------------------------------------------
_SETTLEMENT_KW = (
    "정산", "세금계산서", "인보이스", "sf", "모비사인", "입금",
    "광고주 공유용", "공유용 정산", "재무팀", "발행 요청",
)
_TODO_KW = (
    "오늘", "어제", "todo", "보미님", "스레드", "말씀", "피드백",
    "오늘자", "당일", "어제자",
)
_SETTING_KW = (
    "asc", "bau", "메타", "캠페인 세팅", "캠페인세팅", "광고세트",
    "컨첵시트", "컨첵 시트", "토글", "크첵",
)
_KAKAO_KW = ("카카오톡", "카카오 메시지", "카카오메시지", "카카오 발송", "발송",)


def classify_query(query: str) -> Dict[str, bool]:
    """질문 자동 분류는 복잡하게 하지 않고, keyword-based heuristic 으로만 구현한다."""
    text = (query or "").lower()
    return {
        "is_settlement": any(k in text for k in _SETTLEMENT_KW),
        "is_todo": any(k in text for k in _TODO_KW),
        "is_setting": any(k in text for k in _SETTING_KW),
        "is_kakao": any(k in text for k in _KAKAO_KW),
    }


def category_boost_for(
    uploaded_category: str,
    query_class: Dict[str, bool],
    base_boost: Optional[Dict[str, float]] = None,
) -> float:
    """
    질문 키워드별로 카테고리 부스트를 살짝 보정한다.

    - 절차/가이드형 질문(정산 등): guide 우선, slack/kakao 보조
    - 대화/히스토리형 질문(TODO 등): slack 보조 강화, guide 살짝 낮춤
    - 메타 세팅/당일 진행상황: slack + guide 둘 다 고려
    - 카카오 발송 관련: slack + guide + kakao 보조
    """
    base = base_boost or DEFAULT_CATEGORY_BOOST
    cat = (uploaded_category or "misc").lower()
    boost = float(base.get(cat, 1.0))

    if query_class.get("is_settlement"):
        if cat == "guide":
            boost *= 1.30
        elif cat == "excel":
            boost *= 1.05
        elif cat == "slack":
            boost *= 0.90
        elif cat == "kakao":
            boost *= 0.80
        elif cat == "misc":
            boost *= 0.85

    if query_class.get("is_todo"):
        # TODO/당일 업무 흐름은 slack thread 가 1차 근거.
        # conversation content_type 의 0.85 페널티를 상쇄할 만큼 강하게 부스트한다.
        if cat == "slack":
            boost *= 1.50
        elif cat == "guide":
            boost *= 0.85
        elif cat == "kakao":
            boost *= 0.95

    if query_class.get("is_setting"):
        if cat in {"slack", "guide"}:
            boost *= 1.10

    if query_class.get("is_kakao"):
        if cat == "slack":
            boost *= 1.15
        elif cat == "guide":
            boost *= 1.10
        elif cat == "kakao":
            boost *= 1.05

    return boost


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _recency_score(created_at_iso: Optional[str]) -> float:
    """오래된 chunk 의 가중치를 살짝 깎는다 (0.85~1.0)."""
    if not created_at_iso:
        return 1.0
    try:
        dt = datetime.fromisoformat(created_at_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days = max(0.0, (now - dt).total_seconds() / 86400.0)
    except Exception:  # noqa: BLE001
        return 1.0
    if days <= 7:
        return 1.0
    if days <= 90:
        return 0.97
    if days <= 365:
        return 0.93
    return 0.85


def _content_type_boost(content_type: str, source_type: str) -> float:
    """
    content_type → boost. user 스펙을 그대로 따른다.

    - guide / excel_summary : 1.15
    - excel_raw_table / table / text : 1.0
    - conversation : 0.85 (slack 대화 등)
    - note : 0.8

    TODO/당일 업무 흐름 질문에서 conversation 페널티를 보상하는 처리는
    `category_boost_for` 의 is_todo 분기에서 담당한다.
    """
    ct = (content_type or "text").lower()
    if ct in CONTENT_TYPE_BOOST:
        return CONTENT_TYPE_BOOST[ct]
    return 1.0


# ---------------------------------------------------------------------------
# 메인 reranker
# ---------------------------------------------------------------------------
def rerank_simple(
    candidates: List[RetrievedChunk],
    category_boost: Optional[Dict[str, float]] = None,
    query: Optional[str] = None,
) -> List[RetrievedChunk]:
    """
    final_score 를 채우고 내림차순 정렬한다.

    Parameters
    ----------
    candidates : VectorStore.search 결과
    category_boost : uploaded_category 별 기본 boost (None 이면 DEFAULT_CATEGORY_BOOST)
    query : 주어지면 keyword-heuristic 으로 카테고리 부스트를 보정한다.

    또한 각 chunk 의 metadata 에 다음 진단 필드를 채워둔다 (UI 노출용).
    - similarity_score
    - source_weight
    - category_boost
    - content_type_boost
    - recency_score
    """
    base_boost = category_boost or DEFAULT_CATEGORY_BOOST
    query_class = classify_query(query) if query else {
        "is_settlement": False, "is_todo": False,
        "is_setting": False, "is_kakao": False,
    }

    for c in candidates:
        sim = max(0.0, float(c.score or 0.0))
        sw = float(c.metadata.get("source_weight") or 0.5)
        cat = (c.uploaded_category or c.metadata.get("uploaded_category") or "misc").lower()
        cb = category_boost_for(cat, query_class, base_boost=base_boost)
        ct_b = _content_type_boost(c.content_type, c.source_type)  # noqa: ARG001
        rs = _recency_score(c.metadata.get("created_at"))

        c.final_score = sim * sw * cb * ct_b * rs
        # UI 진단용 (read-only 의도)
        c.metadata["similarity_score"] = round(sim, 6)
        c.metadata["source_weight"] = round(sw, 6)
        c.metadata["category_boost"] = round(cb, 6)
        c.metadata["content_type_boost"] = round(ct_b, 6)
        c.metadata["recency_score"] = round(rs, 6)
        c.metadata["query_class"] = query_class

    candidates.sort(key=lambda x: x.final_score, reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Diversity / MMR-ish penalty
# ---------------------------------------------------------------------------
def apply_diversity_penalty(
    candidates: List[RetrievedChunk],
    mmr_lambda: float = 0.7,
    same_file_penalty: float = 0.70,
    same_section_penalty: float = 0.90,
) -> List[RetrievedChunk]:
    """
    BM25/임베딩 cosine 직접 비교 없이도 동작하는 가벼운 다양성 휴리스틱.

    - 동일 file_name 이 반복될수록 감점
    - 동일 section_title + content_type 이 반복될수록 감점
    - 최종 점수는 final_score * lambda + diversity_bonus * (1 - lambda)

    `diversity_bonus` 는 "처음 등장한 파일/섹션이면 1.0, 반복되면 페널티" 형태로 단순화한다.
    """
    if not candidates:
        return candidates

    file_seen: Dict[str, int] = {}
    sec_seen: Dict[Tuple[str, str], int] = {}
    for c in candidates:
        f = (c.file_name or "").strip()
        sec_title = (c.section_title or "").strip()
        sec_key: Optional[Tuple[str, str]] = (
            (sec_title, (c.content_type or "").strip()) if sec_title else None
        )
        f_count = file_seen.get(f, 0) if f else 0
        s_count = sec_seen.get(sec_key, 0) if sec_key is not None else 0

        diversity_bonus = 1.0
        if f and f_count >= 1:
            diversity_bonus *= same_file_penalty ** f_count
        if sec_key is not None and s_count >= 1:
            diversity_bonus *= same_section_penalty ** s_count

        # MMR 비슷한 보간
        c.final_score = (
            float(mmr_lambda) * float(c.final_score)
            + float(1.0 - mmr_lambda) * float(c.final_score) * diversity_bonus
        )
        c.metadata["mmr_diversity_bonus"] = round(diversity_bonus, 6)

        if f:
            file_seen[f] = f_count + 1
        if sec_key is not None:
            sec_seen[sec_key] = s_count + 1

    candidates.sort(key=lambda x: x.final_score, reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# 미래 확장용 인터페이스 (현 MVP 미사용)
# ---------------------------------------------------------------------------
def rerank_with_llm(
    query: str, candidates: List[RetrievedChunk], top_k: int = 8
) -> List[RetrievedChunk]:
    """
    TODO: Gemini 또는 cross-encoder 를 활용한 reranker.
    현재 MVP 에서는 호출하지 않는다.
    """
    log.info("rerank_with_llm 은 아직 구현되지 않았습니다. rerank_simple 결과를 반환합니다.")
    candidates.sort(key=lambda x: x.final_score, reverse=True)
    return candidates[:top_k]
