"""
reranker.py
===========
1차 MVP rule-based reranker.

final_score = similarity_score
              * source_weight
              * category_boost
              * content_type_boost
              * recency_score
              * date_match_boost
              * topic_match_boost

- BM25 / LLM reranker / cross-encoder 는 아직 사용하지 않는다.
- query keyword heuristic 으로 카테고리 boost 를 살짝 보정한다.
- query 에 날짜/주제(topic) 가 들어 있으면 chunk metadata 와 매칭해
  date / topic boost / penalty 를 추가로 적용한다.
- diversity 보정용 MMR 비슷한 휴리스틱(`apply_diversity_penalty`) 도 함께 제공한다.

명칭 변경 노트:
- 이 모듈의 "knowledge_card" 관련 helper 들은 "Normalized Document" 명칭으로
  바뀌었다. 기존 import 호환을 위해 ``apply_knowledge_card_priority`` /
  ``is_knowledge_card_chunk`` / ``card_type_boost_for`` 는 새 함수의
  alias 로 그대로 유지한다.
- chunk metadata 는 신규 표준 ``content_type="normalized_document"`` 와
  legacy ``content_type="knowledge_card"`` 양쪽을 모두 인식한다.
- chunk metadata 의 ``card_type`` 은 ``normalized_document_type`` 으로
  대체될 수 있으며, 둘 다 동일하게 인식된다.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.config import settings
from src.ingestion.date_extractor import extract_document_date
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


# Topic 키워드 (slack_manual_parser._TOPIC_KEYWORDS 와 정합)
_QUERY_TOPIC_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "meta": (
        "메타", "asc", "bau", "캠페인", "광고세트", "카탈로그", "컨첵", "크첵",
        "t&d", "td", "랜딩", "네이밍", "매핑", "토글",
    ),
    "kakao": ("카카오", "카카오톡", "카카오메시지", "메시지 발송", "발송", "잔액", "충전"),
    "settlement": (
        "정산", "세금계산서", "인보이스", "모비사인", "sf", "입금", "거래명세서",
    ),
    "outdoor": ("옥외", "파르나스", "편성표", "구좌", "선입금"),
    "report": ("dr", "rd", "리포트", "월간보고", "월간 보고"),
    "nbt": ("nbt", "토스"),
    "greenp": ("그린피", "greenp"),
    "youtube": ("유튜브", "youtube", "구글 유튜브"),
}

_INTENT_PROCEDURE = (
    "순서", "프로세스", "어떻게 진행", "어떻게 해야", "방법", "단계",
    "가이드", "체크리스트", "셋팅", "세팅", "어떻게",
)
_INTENT_TODO_LOOKUP = (
    "그날", "놓치면 안", "놓치면 안되는", "놓치면 안 되는", "뭐였어",
    "뭐 였어", "todo", "할 일",
)
_INTENT_EXPLANATION = ("설명", "차이", "이유", "원인", "왜")
_INTENT_ISSUE_LOOKUP = (
    "이슈", "문제", "어려웠", "발생한", "발송 관련", "어땠어",
    "안 됨", "안됨", "실패", "오류",
)
# Task 6: Normalized Document 우선순위를 위한 추가 intent
_INTENT_COMMUNICATION = ("문안", "메일", "공유", "전달", "회신")
_INTENT_GLOSSARY = ("용어", "무슨 뜻", "정의")


def classify_query(query: str) -> Dict[str, bool]:
    """질문 자동 분류는 복잡하게 하지 않고, keyword-based heuristic 으로만 구현한다."""
    text = (query or "").lower()
    return {
        "is_settlement": any(k in text for k in _SETTLEMENT_KW),
        "is_todo": any(k in text for k in _TODO_KW),
        "is_setting": any(k in text for k in _SETTING_KW),
        "is_kakao": any(k in text for k in _KAKAO_KW),
    }


# ---------------------------------------------------------------------------
# Query metadata (date / topics / intent)
# ---------------------------------------------------------------------------
def _detect_intent(text: str) -> List[str]:
    out: List[str] = []
    if any(k in text for k in _INTENT_PROCEDURE):
        out.append("procedure")
    if any(k in text for k in _INTENT_TODO_LOOKUP):
        out.append("todo_lookup")
    if any(k in text for k in _INTENT_EXPLANATION):
        out.append("explanation")
    if any(k in text for k in _INTENT_ISSUE_LOOKUP):
        out.append("issue_lookup")
    if any(k in text for k in _INTENT_COMMUNICATION):
        out.append("communication")
    if any(k in text for k in _INTENT_GLOSSARY):
        out.append("glossary")
    return out


def extract_query_metadata(query: str) -> Dict[str, Any]:
    """
    질문에서 다음을 추출한다.
    - query_date : "YYYY-MM-DD" | None
    - query_topics : list[str]
    - query_intent : list[str]
    - query_class : classify_query 결과 (호환)
    """
    if not query:
        return {
            "query_date": None,
            "query_topics": [],
            "query_intent": [],
            "query_class": classify_query(""),
            "query_date_text": None,
        }

    text = query.strip()
    low = text.lower()

    # 1) 날짜 추출 (date_extractor 재사용 — 본문 검색이라 default_year 적용)
    info = extract_document_date(file_name=None, content=text)
    query_date = info.get("document_date")
    query_date_text = info.get("date_text")

    # 2) topic 키워드 매칭
    topics: List[str] = []
    for tag, kws in _QUERY_TOPIC_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in low:
                topics.append(tag)
                break

    # 3) intent
    intent = _detect_intent(low)

    return {
        "query_date": query_date,
        "query_date_text": query_date_text,
        "query_topics": topics,
        "query_intent": intent,
        "query_class": classify_query(text),
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


def _date_match_factor(
    chunk_date: Optional[str],
    query_date: Optional[str],
    *,
    boost: float,
    penalty: float,
) -> Tuple[float, str]:
    """
    chunk.document_date 와 query_date 비교 → (multiplier, label).

    label:
      - "exact"    : 일치
      - "mismatch" : 둘 다 있는데 다름
      - "none"     : 둘 중 하나라도 비어있어 비교 불가
    """
    if not query_date:
        return 1.0, "none"
    if not chunk_date:
        # query 에는 날짜가 있는데 chunk 에는 없음 → 약하게 페널티 적용
        # (guide 처럼 날짜 없는 문서를 너무 누르지 않도록 sqrt 로 완화)
        weak = penalty + (1.0 - penalty) * 0.5  # ex) 0.55 → 0.775
        return weak, "none"
    if chunk_date == query_date:
        return float(boost), "exact"
    return float(penalty), "mismatch"


def _topic_match_factor(
    chunk_topics: List[str],
    chunk_source_type: str,
    query_topics: List[str],
    query_intent: List[str],
    *,
    boost: float,
    penalty: float,
) -> Tuple[float, str]:
    """
    chunk.topic_tags 와 query_topics 비교 → (multiplier, label).

    label:
      - "match"    : 1개 이상 겹침
      - "mismatch" : 둘 다 있는데 안 겹침
      - "none"     : query_topics 가 비어 있거나 chunk_topics 가 비어 있어 비교 불가
    """
    if not query_topics:
        return 1.0, "none"
    if not chunk_topics:
        # query 에 topic 이 있는데 chunk 에는 topic_tags 가 없음.
        # procedure 의도이고 chunk_source_type=guide 라면 약하게 (페널티 거의 안 줌).
        if "procedure" in query_intent and (chunk_source_type or "").lower() in {"guide"}:
            return 0.95, "none"
        weak = penalty + (1.0 - penalty) * 0.5
        return weak, "none"
    overlap = set(t.lower() for t in chunk_topics) & set(t.lower() for t in query_topics)
    if overlap:
        return float(boost), "match"
    # 둘 다 있는데 겹치지 않음 → 강한 페널티
    return float(penalty), "mismatch"


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
    query_metadata: Optional[Dict[str, Any]] = None,
    *,
    date_exact_match_boost: Optional[float] = None,
    date_mismatch_penalty: Optional[float] = None,
    topic_match_boost: Optional[float] = None,
    topic_mismatch_penalty: Optional[float] = None,
) -> List[RetrievedChunk]:
    """
    final_score 를 채우고 내림차순 정렬한다.

    Parameters
    ----------
    candidates : VectorStore.search 결과
    category_boost : uploaded_category 별 기본 boost (None 이면 DEFAULT_CATEGORY_BOOST)
    query : 주어지면 keyword-heuristic 으로 카테고리 부스트를 보정한다.
    query_metadata : extract_query_metadata 결과. 없으면 query 로부터 자동 추출.

    또한 각 chunk 의 metadata 에 다음 진단 필드를 채워둔다 (UI 노출용).
    - similarity_score
    - source_weight
    - category_boost
    - content_type_boost
    - recency_score
    - date_boost / date_match
    - topic_boost / topic_match
    """
    base_boost = category_boost or DEFAULT_CATEGORY_BOOST

    if query_metadata is None:
        query_metadata = extract_query_metadata(query or "")
    query_class = query_metadata.get("query_class") or classify_query(query or "")
    query_date: Optional[str] = query_metadata.get("query_date")
    query_topics: List[str] = list(query_metadata.get("query_topics") or [])
    query_intent: List[str] = list(query_metadata.get("query_intent") or [])

    d_boost = float(
        date_exact_match_boost
        if date_exact_match_boost is not None
        else settings.date_exact_match_boost
    )
    d_pen = float(
        date_mismatch_penalty
        if date_mismatch_penalty is not None
        else settings.date_mismatch_penalty
    )
    t_boost = float(
        topic_match_boost if topic_match_boost is not None else settings.topic_match_boost
    )
    t_pen = float(
        topic_mismatch_penalty
        if topic_mismatch_penalty is not None
        else settings.topic_mismatch_penalty
    )

    for c in candidates:
        sim = max(0.0, float(c.score or 0.0))
        sw = float(c.metadata.get("source_weight") or 0.5)
        cat = (c.uploaded_category or c.metadata.get("uploaded_category") or "misc").lower()
        cb = category_boost_for(cat, query_class, base_boost=base_boost)
        ct_b = _content_type_boost(c.content_type, c.source_type)  # noqa: ARG001
        rs = _recency_score(c.metadata.get("created_at"))

        # date / topic 매칭
        chunk_date = c.metadata.get("document_date")
        date_factor, date_label = _date_match_factor(
            chunk_date, query_date, boost=d_boost, penalty=d_pen
        )

        chunk_topics = c.metadata.get("topic_tags") or []
        if not isinstance(chunk_topics, list):
            chunk_topics = []
        topic_factor, topic_label = _topic_match_factor(
            chunk_topics,
            (c.source_type or "").lower(),
            query_topics,
            query_intent,
            boost=t_boost,
            penalty=t_pen,
        )

        c.final_score = sim * sw * cb * ct_b * rs * date_factor * topic_factor
        # UI 진단용 (read-only 의도)
        c.metadata["similarity_score"] = round(sim, 6)
        c.metadata["source_weight"] = round(sw, 6)
        c.metadata["category_boost"] = round(cb, 6)
        c.metadata["content_type_boost"] = round(ct_b, 6)
        c.metadata["recency_score"] = round(rs, 6)
        c.metadata["date_boost"] = round(date_factor, 6)
        c.metadata["date_match"] = date_label
        c.metadata["topic_boost"] = round(topic_factor, 6)
        c.metadata["topic_match"] = topic_label
        c.metadata["query_class"] = query_class
        c.metadata["query_date"] = query_date
        c.metadata["query_topics"] = query_topics
        c.metadata["query_intent"] = query_intent

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
# Task 6: Normalized Document 우선 retrieval helpers
# ---------------------------------------------------------------------------
# normalized_document_type → 매칭하는 query intent label
_NORMALIZED_DOCUMENT_TYPE_INTENT_MAP: Dict[str, str] = {
    "workflow": "procedure",
    "checklist": "procedure",
    "faq": "explanation",
    "decision": "explanation",
    "issue": "issue_lookup",
    "communication_template": "communication",
    "glossary": "glossary",
}
# legacy alias (기존 import 호환)
_CARD_TYPE_INTENT_MAP = _NORMALIZED_DOCUMENT_TYPE_INTENT_MAP

# normalized_document_type 매칭 시 추가 부스트 배수.
# (intent 부합 문서는 spec boost 위에 1.10x 더 곱해서 강조한다.)
_NORMALIZED_DOCUMENT_TYPE_INTENT_BONUS = 1.10
# legacy alias
_CARD_TYPE_INTENT_BONUS = _NORMALIZED_DOCUMENT_TYPE_INTENT_BONUS


# 신규 표준 content_type / source_type
_NORMALIZED_DOCUMENT_CONTENT_TYPES = {"normalized_document", "knowledge_card"}
_NORMALIZED_DOCUMENT_SOURCE_TYPES = {"llm_normalized"}


def is_normalized_document_chunk(
    chunk_or_meta: Any,
) -> bool:
    """
    chunk 가 LLM-based Document Normalization 결과 (Normalized Document) 인지 판정.

    아래 조건 중 하나라도 만족하면 Normalized Document 로 본다:
    - content_type == "normalized_document" (신규 표준)
    - content_type == "knowledge_card"      (legacy compatibility)
    - source_type  == "llm_normalized"

    `chunk_or_meta` 는 RetrievedChunk 또는 dict-like metadata 모두 허용.
    """
    if chunk_or_meta is None:
        return False
    md: Dict[str, Any]
    content_type: str = ""
    source_type: str = ""
    if isinstance(chunk_or_meta, dict):
        md = chunk_or_meta
        content_type = str(md.get("content_type") or "").lower()
        source_type = str(md.get("source_type") or "").lower()
    else:
        md = getattr(chunk_or_meta, "metadata", {}) or {}
        content_type = (
            getattr(chunk_or_meta, "content_type", None)
            or md.get("content_type")
            or ""
        )
        source_type = (
            getattr(chunk_or_meta, "source_type", None)
            or md.get("source_type")
            or ""
        )
        content_type = str(content_type or "").lower()
        source_type = str(source_type or "").lower()
    if content_type in _NORMALIZED_DOCUMENT_CONTENT_TYPES:
        return True
    if source_type in _NORMALIZED_DOCUMENT_SOURCE_TYPES:
        return True
    return False


# legacy alias — 기존 import 유지
is_knowledge_card_chunk = is_normalized_document_chunk


def _resolve_normalized_document_type(meta: Dict[str, Any]) -> str:
    """metadata 에서 normalized_document_type 을 신규/legacy 키 양쪽에서 읽어 온다."""
    if not isinstance(meta, dict):
        return ""
    val = (
        meta.get("normalized_document_type")
        or meta.get("card_type")
        or ""
    )
    return str(val or "").strip().lower()


def normalized_document_type_boost_for(
    document_type: Optional[str],
    query_intent: Optional[List[str]] = None,
    *,
    settings_obj: Optional[Any] = None,
) -> Tuple[float, bool]:
    """
    Normalized Document 의 document_type 별 boost 와 query intent 부합 여부 반환.

    동작:
    - document_type 별 spec boost (settings 의 *_card_boost) 를 기본 적용한다.
      (env 이름은 legacy 호환 유지를 위해 그대로 둔다.)
    - query_intent 와 document_type 이 매칭되면 추가로 1.10x 를 곱해 강조한다.
    - document_type 이 비어 있거나 매핑에 없으면 (1.0, False) 반환.

    Returns
    -------
    (boost, intent_match)
    """
    s = settings_obj or settings
    ct = (document_type or "").strip().lower()
    if not ct:
        return 1.0, False

    boost_map: Dict[str, float] = {
        "workflow": float(getattr(s, "workflow_card_boost", 1.0) or 1.0),
        "checklist": float(getattr(s, "checklist_card_boost", 1.0) or 1.0),
        "faq": float(getattr(s, "faq_card_boost", 1.0) or 1.0),
        "decision": float(getattr(s, "decision_card_boost", 1.0) or 1.0),
        "communication_template": float(
            getattr(s, "communication_template_boost", 1.0) or 1.0
        ),
        "glossary": float(getattr(s, "glossary_card_boost", 1.0) or 1.0),
    }
    boost = boost_map.get(ct, 1.0)

    expected_intent = _NORMALIZED_DOCUMENT_TYPE_INTENT_MAP.get(ct)
    intents = list(query_intent or [])
    intent_match = bool(expected_intent and expected_intent in intents)
    if intent_match:
        boost *= _NORMALIZED_DOCUMENT_TYPE_INTENT_BONUS
    return boost, intent_match


# legacy alias — 기존 import 유지
card_type_boost_for = normalized_document_type_boost_for


def _normalized_file_keys(candidates: List[RetrievedChunk]) -> set:
    """
    Normalized Document chunk 가 포함된 source_file_hash / file_name 집합을 반환.

    raw chunk 가 같은 파일 출신인지 판정할 때 사용한다.
    """
    keys: set = set()
    for c in candidates or []:
        if not is_normalized_document_chunk(c):
            continue
        md = getattr(c, "metadata", {}) or {}
        fh = md.get("source_file_hash") or md.get("file_hash")
        fn = getattr(c, "file_name", None) or md.get("file_name")
        if fh:
            keys.add(f"hash:{fh}")
        if fn:
            keys.add(f"name:{fn}")
    return keys


def _chunk_belongs_to_normalized_file(
    chunk: RetrievedChunk, normalized_keys: set
) -> bool:
    if not normalized_keys:
        return False
    md = getattr(chunk, "metadata", {}) or {}
    fh = md.get("source_file_hash") or md.get("file_hash")
    fn = getattr(chunk, "file_name", None) or md.get("file_name")
    if fh and f"hash:{fh}" in normalized_keys:
        return True
    if fn and f"name:{fn}" in normalized_keys:
        return True
    return False


def apply_normalized_document_priority(
    candidates: List[RetrievedChunk],
    *,
    query_metadata: Optional[Dict[str, Any]] = None,
    settings_obj: Optional[Any] = None,
) -> List[RetrievedChunk]:
    """
    rerank_simple 결과를 받은 뒤 Normalized Document 를 raw chunk 보다 우선시키는 후처리.

    동작:
    - PRIORITIZE_NORMALIZED_DOCUMENTS (legacy: PRIORITIZE_KNOWLEDGE_CARDS) = false 면
      metadata 진단 필드만 채우고 점수는 유지.
    - true 인 경우:
        - Normalized Document chunk:
            final_score *= NORMALIZED_DOCUMENT_CONTENT_BOOST
                          * normalized_document_type_boost
            retrieval_role = "primary_card"  (legacy 라벨 유지)
        - 같은 source_file 의 raw chunk:
            final_score *= RAW_EVIDENCE_BOOST
            retrieval_role = "raw_evidence"
        - 그 외 raw chunk:
            retrieval_role = "raw_fallback"

    metadata 호환:
    - normalized_document_type / card_type 둘 다 인식 (신규 우선, legacy fallback).
    - 진단 필드는 신규 / legacy 키를 모두 채워, 기존 UI 와의 호환을 유지한다.

    candidates 는 final_score 내림차순으로 재정렬되어 반환된다.
    """
    if not candidates:
        return candidates

    s = settings_obj or settings
    enabled = bool(getattr(s, "prioritize_knowledge_cards", True))
    kc_boost = float(getattr(s, "knowledge_card_content_boost", 1.0) or 1.0)
    raw_evidence_boost = float(getattr(s, "raw_evidence_boost", 1.0) or 1.0)

    qm = query_metadata or {}
    query_intent = list(qm.get("query_intent") or [])

    normalized_keys = _normalized_file_keys(candidates) if enabled else set()

    for c in candidates:
        md = getattr(c, "metadata", None)
        if md is None:
            md = {}
            c.metadata = md
        is_kc = is_normalized_document_chunk(c)

        if not enabled:
            md.setdefault("normalized_document_boost", 1.0)
            md.setdefault("normalized_document_type_boost", 1.0)
            md.setdefault("normalized_document_type_match", False)
            # legacy 진단 필드 (기존 UI 호환)
            md.setdefault("knowledge_card_boost", 1.0)
            md.setdefault("card_type_boost", 1.0)
            md.setdefault("card_type_match", False)
            if is_kc:
                md.setdefault("retrieval_role", "primary_card")
            else:
                md.setdefault("retrieval_role", "raw_fallback")
            continue

        if is_kc:
            doc_type = _resolve_normalized_document_type(md)
            ct_boost, intent_match = normalized_document_type_boost_for(
                doc_type, query_intent, settings_obj=s
            )
            c.final_score = float(c.final_score or 0.0) * kc_boost * ct_boost
            md["normalized_document_boost"] = round(kc_boost, 6)
            md["normalized_document_type_boost"] = round(ct_boost, 6)
            md["normalized_document_type_match"] = bool(intent_match)
            # legacy 진단 필드 (기존 UI 호환)
            md["knowledge_card_boost"] = round(kc_boost, 6)
            md["card_type_boost"] = round(ct_boost, 6)
            md["card_type_match"] = bool(intent_match)
            md["retrieval_role"] = "primary_card"
        else:
            md["normalized_document_boost"] = 1.0
            md["normalized_document_type_boost"] = 1.0
            md["normalized_document_type_match"] = False
            md["knowledge_card_boost"] = 1.0
            md["card_type_boost"] = 1.0
            md["card_type_match"] = False
            if _chunk_belongs_to_normalized_file(c, normalized_keys):
                c.final_score = float(c.final_score or 0.0) * raw_evidence_boost
                md["retrieval_role"] = "raw_evidence"
                md["raw_evidence_boost"] = round(raw_evidence_boost, 6)
            else:
                md["retrieval_role"] = "raw_fallback"

    candidates.sort(key=lambda x: x.final_score, reverse=True)
    return candidates


# legacy alias — 기존 import 유지
apply_knowledge_card_priority = apply_normalized_document_priority


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
