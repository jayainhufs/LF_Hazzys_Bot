"""LLM-based Document Normalization 패키지.

raw 업무 문서를 LLM 으로 정규화해 ``NormalizedDocument`` 단위 (절차, 체크리스트,
이슈, FAQ 등) 로 변환하고, 이후 검색/QA 의 1차 근거로 사용한다.

명칭 변경:
- ``GuideKnowledgeNormalizer`` → ``GuideDocumentNormalizer``
- ``SlackThreadKnowledgeNormalizer`` → ``SlackThreadDocumentNormalizer``
- ``knowledge_cards_to_chunks`` → ``normalized_documents_to_chunks``

기존 import 호환을 위해 모든 옛 이름을 alias 로 함께 export 한다.
"""

from src.normalization.card_viewer import (
    card_to_display_dict,
    filter_cards,
    filter_normalized_documents,
    list_normalized_json_files,
    list_normalized_markdown_files,
    load_all_cards_from_store,
    load_all_normalized_documents_from_store,
    markdown_for_card,
    markdown_for_normalized_document,
    normalized_document_to_display_dict,
    summarize_cards,
    summarize_normalized_documents,
)
from src.normalization.guide_normalizer import (
    GuideDocumentNormalizer,
    GuideKnowledgeNormalizer,
)
from src.normalization.normalization_prompt import (
    GUIDE_NORMALIZER_PROMPT_VERSION,
    SLACK_NORMALIZER_PROMPT_VERSION,
    build_guide_normalization_prompt,
    build_slack_normalization_prompt,
)
from src.normalization.normalization_store import NormalizationStore
from src.normalization.pipeline_integration import (
    attach_parent_raw_chunk_ids,
    extract_normalization_inputs,
    knowledge_cards_to_chunks,
    normalize_document_for_pipeline,
    normalized_documents_to_chunks,
    run_normalization_branch,
    should_normalize_file,
)
from src.normalization.slack_normalizer import (
    SlackThreadDocumentNormalizer,
    SlackThreadKnowledgeNormalizer,
)

__all__ = [
    "NormalizationStore",
    # 신규 명칭
    "GuideDocumentNormalizer",
    "SlackThreadDocumentNormalizer",
    "normalized_documents_to_chunks",
    "load_all_normalized_documents_from_store",
    "summarize_normalized_documents",
    "filter_normalized_documents",
    "normalized_document_to_display_dict",
    "markdown_for_normalized_document",
    # legacy alias (backward compatibility)
    "GuideKnowledgeNormalizer",
    "SlackThreadKnowledgeNormalizer",
    "knowledge_cards_to_chunks",
    "load_all_cards_from_store",
    "summarize_cards",
    "filter_cards",
    "card_to_display_dict",
    "markdown_for_card",
    # 그 외 공통
    "list_normalized_json_files",
    "list_normalized_markdown_files",
    "GUIDE_NORMALIZER_PROMPT_VERSION",
    "SLACK_NORMALIZER_PROMPT_VERSION",
    "build_guide_normalization_prompt",
    "build_slack_normalization_prompt",
    "should_normalize_file",
    "extract_normalization_inputs",
    "attach_parent_raw_chunk_ids",
    "normalize_document_for_pipeline",
    "run_normalization_branch",
]
