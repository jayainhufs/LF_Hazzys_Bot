"""LLM 기반 업무 지식카드 정규화 패키지."""

from src.normalization.guide_normalizer import GuideKnowledgeNormalizer
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
    run_normalization_branch,
    should_normalize_file,
)
from src.normalization.slack_normalizer import SlackThreadKnowledgeNormalizer

__all__ = [
    "NormalizationStore",
    "GuideKnowledgeNormalizer",
    "SlackThreadKnowledgeNormalizer",
    "GUIDE_NORMALIZER_PROMPT_VERSION",
    "SLACK_NORMALIZER_PROMPT_VERSION",
    "build_guide_normalization_prompt",
    "build_slack_normalization_prompt",
    "should_normalize_file",
    "extract_normalization_inputs",
    "attach_parent_raw_chunk_ids",
    "knowledge_cards_to_chunks",
    "normalize_document_for_pipeline",
    "run_normalization_branch",
]
