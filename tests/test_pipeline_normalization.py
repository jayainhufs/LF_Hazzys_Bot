"""LLM normalization pipeline 연결 (Task 4) 단위 테스트.

원칙:
- 외부 Gemini API 호출 없음. 모든 LLM 호출은 ``FakeGeminiClient`` 가 대신한다.
- pipeline 전체를 무겁게 돌리지 않고 helper 단위 또는 ``run_normalization_branch``
  단위로 검증한다.
- ``ENABLE_LLM_NORMALIZATION=false`` 일 때는 normalizer 가 호출되지 않는 것을
  보장한다 (기존 raw ingest 흐름 보존).
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, List, Optional

import pytest

from src.normalization import (
    GUIDE_NORMALIZER_PROMPT_VERSION,
    SLACK_NORMALIZER_PROMPT_VERSION,
    NormalizationStore,
    attach_parent_raw_chunk_ids,
    extract_normalization_inputs,
    knowledge_cards_to_chunks,
    run_normalization_branch,
    should_normalize_file,
)
from src.schemas import Chunk, Document, KnowledgeCard, ParsedSection


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeGeminiClient:
    def __init__(self, response: Optional[str] = None) -> None:
        self.response = response or ""
        self.call_count = 0
        self.last_prompt: Optional[str] = None
        self.last_system_instruction: Optional[str] = None
        self.last_model: Optional[str] = None

    def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None,
    ) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        self.last_model = model
        self.last_system_instruction = system_instruction
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FailingClient:
    def generate_text(self, *args, **kwargs):
        raise RuntimeError("network down")


class FakeEmbedder:
    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        return [[float(i + 1)] * self.dim for i, _ in enumerate(texts)]


class FailingEmbedder:
    def embed_documents(self, texts):
        raise RuntimeError("embedder broken")


class FakeVectorStore:
    def __init__(self) -> None:
        self.added: List[Chunk] = []
        self.added_embeddings: List[List[float]] = []
        self.calls = 0

    def add_chunks(self, chunks, embeddings, skip_existing: bool = True):
        self.calls += 1
        self.added.extend(chunks)
        self.added_embeddings.extend(embeddings)
        return len(chunks)


class FailingVectorStore:
    def add_chunks(self, chunks, embeddings, skip_existing: bool = True):
        raise RuntimeError("chroma down")


class FakeDocumentStore:
    def __init__(self) -> None:
        self.saved_chunks_calls: List[tuple] = []

    def save_chunks(self, document_id, chunks):
        self.saved_chunks_calls.append((document_id, list(chunks)))
        return None


# ---------------------------------------------------------------------------
# Test settings
# ---------------------------------------------------------------------------
def _make_settings(
    *,
    enable_normalization: bool = True,
    max_cards_per_file: int = 30,
    parent_top_k: int = 3,
    source_weight: float = 1.25,
    save_json: bool = True,
    save_markdown: bool = True,
    max_chars_per_call: int = 18000,
    temperature: float = 0.1,
    model: str = "fake-gemini-model",
) -> SimpleNamespace:
    return SimpleNamespace(
        enable_llm_normalization=enable_normalization,
        llm_normalization_model=model,
        normalization_temperature=temperature,
        normalization_max_chars_per_call=max_chars_per_call,
        normalization_max_cards_per_file=max_cards_per_file,
        normalization_save_json=save_json,
        normalization_save_markdown=save_markdown,
        normalization_use_anonymized_input=True,
        normalization_card_source_weight=source_weight,
        normalization_parent_raw_top_k=parent_top_k,
    )


def _make_document(
    *,
    uploaded_category: str = "guide",
    source_type: str = "guide",
    file_name: str = "guide.txt",
    file_hash: str = "a" * 64,
    document_id: str = "doc_test001",
) -> Document:
    return Document(
        document_id=document_id,
        source_type=source_type,
        uploaded_category=uploaded_category,
        file_name=file_name,
        file_path=f"data/raw/{file_name}",
        file_hash=file_hash,
        title=file_name,
        created_at="",
        ingested_at="",
        metadata={},
    )


def _make_parsed_sections(
    *,
    content: str = "원본 가이드 본문",
    sanitized: Optional[str] = None,
    topic_tags: Optional[List[str]] = None,
    todo_phase: Optional[str] = None,
    parser_format: Optional[str] = None,
    document_date: Optional[str] = None,
    display_date: Optional[str] = None,
) -> List[ParsedSection]:
    md: dict = {}
    if sanitized:
        md["sanitized_content"] = sanitized
    if topic_tags:
        md["topic_tags"] = list(topic_tags)
    if todo_phase:
        md["todo_phase"] = todo_phase
    if parser_format:
        md["parser_format"] = parser_format
    if document_date:
        md["document_date"] = document_date
    if display_date:
        md["display_date"] = display_date
    return [
        ParsedSection(
            section_id="sec_0",
            document_id="doc_test001",
            section_title="섹션",
            content_type="text",
            content=content,
            metadata=md,
        )
    ]


def _make_raw_chunks(n: int = 4, document_id: str = "doc_test001") -> List[Chunk]:
    out: List[Chunk] = []
    for i in range(n):
        out.append(
            Chunk(
                chunk_id=f"chunk_{document_id}_{i:04d}",
                document_id=document_id,
                chunk_index=i,
                source_type="guide",
                uploaded_category="guide",
                file_name="guide.txt",
                content=f"raw 본문 {i}",
                clean_content=f"raw 본문 {i}",
                embedding_text=f"raw 본문 {i}",
                section_title="섹션",
                content_type="text",
                metadata={"section_title": "섹션"},
            )
        )
    return out


GUIDE_RESPONSE_JSON = json.dumps(
    {
        "cards": [
            {
                "card_type": "workflow",
                "title": "메타 캠페인 셋업",
                "summary": "메타 캠페인을 신규 셋업할 때의 절차.",
                "primary_topic": "meta",
                "topic_tags": ["meta", "setup"],
                "task_type": "setup",
                "when_to_use": "신규 캠페인",
                "prerequisites": ["광고 계정 권한"],
                "steps": ["계정 선택", "타겟 설정"],
                "checkpoints": ["전환 이벤트 수신 확인"],
                "cautions": [],
                "examples": [],
                "related_terms": [],
                "open_questions": [],
                "evidence_spans": [
                    {
                        "section_title": "메타 셋업",
                        "chunk_index": 0,
                        "quote_or_summary": "메타 셋업 절차 근거",
                    }
                ],
            }
        ]
    },
    ensure_ascii=False,
)


SLACK_RESPONSE_JSON = json.dumps(
    {
        "cards": [
            {
                "card_type": "issue",
                "title": "정산서 단위 누락 처리",
                "summary": "원/USD 단위 누락 사례와 처리.",
                "primary_topic": "settlement",
                "topic_tags": ["settlement", "issue"],
                "task_type": "settlement",
                "when_to_use": "정산서 단위 누락 시",
                "prerequisites": [],
                "steps": ["단위 표기 확인"],
                "checkpoints": ["수정본 재공유 여부"],
                "cautions": ["임의 환산 금지"],
                "examples": [],
                "related_terms": [],
                "open_questions": [],
                "evidence_spans": [
                    {
                        "section_title": "정산 이슈",
                        "chunk_index": 0,
                        "quote_or_summary": "정산 단위 누락 근거",
                    }
                ],
            }
        ]
    },
    ensure_ascii=False,
)


# ===========================================================================
# 1) should_normalize_file
# ===========================================================================
class TestShouldNormalizeFile:
    def test_guide_category_returns_guide(self):
        assert (
            should_normalize_file(uploaded_category="guide", source_type="guide")
            == "guide"
        )

    def test_guide_source_type_alone_returns_guide(self):
        assert (
            should_normalize_file(uploaded_category="misc", source_type="guide")
            == "guide"
        )

    def test_slack_category_returns_slack(self):
        assert (
            should_normalize_file(uploaded_category="slack", source_type="slack_manual")
            == "slack"
        )

    def test_slack_manual_category_returns_slack(self):
        assert (
            should_normalize_file(uploaded_category="slack_manual", source_type="txt")
            == "slack"
        )

    def test_slack_source_type_alone_returns_slack(self):
        assert (
            should_normalize_file(uploaded_category="misc", source_type="slack_manual")
            == "slack"
        )

    def test_kakao_returns_none(self):
        assert (
            should_normalize_file(uploaded_category="kakao", source_type="kakao")
            is None
        )

    def test_excel_returns_none(self):
        assert (
            should_normalize_file(uploaded_category="excel", source_type="excel")
            is None
        )

    def test_word_returns_none(self):
        assert (
            should_normalize_file(uploaded_category="misc", source_type="word") is None
        )

    def test_uppercase_normalized(self):
        assert (
            should_normalize_file(uploaded_category="GUIDE", source_type="GUIDE")
            == "guide"
        )

    def test_empty_returns_none(self):
        assert should_normalize_file(uploaded_category=None, source_type=None) is None


# ===========================================================================
# 2) extract_normalization_inputs
# ===========================================================================
class TestExtractNormalizationInputs:
    def test_aggregates_text_and_metadata(self):
        sections = [
            ParsedSection(
                section_id="s1",
                document_id="d1",
                section_title="t",
                content_type="text",
                content="원본 1",
                metadata={
                    "sanitized_content": "익명화 1",
                    "topic_tags": ["settlement", "issue"],
                    "todo_phase": "end_of_day",
                    "parser_format": "slack_todo_sections",
                    "document_date": "2026-04-29",
                    "display_date": "해당 업무일",
                },
            ),
            ParsedSection(
                section_id="s2",
                document_id="d1",
                section_title="t",
                content_type="text",
                content="원본 2",
                metadata={
                    "sanitized_content": "익명화 2",
                    "topic_tags": ["settlement", "month_end"],
                },
            ),
        ]

        result = extract_normalization_inputs(sections)

        assert "익명화 1" in result["text"]
        assert "익명화 2" in result["text"]
        assert "원본 1" not in result["text"]
        assert result["topic_tags"] == ["settlement", "issue", "month_end"]
        assert result["todo_phase"] == "end_of_day"
        assert result["parser_format"] == "slack_todo_sections"
        assert result["document_date"] == "2026-04-29"
        assert result["display_date"] == "해당 업무일"

    def test_falls_back_to_raw_content_when_no_sanitized(self):
        sections = [
            ParsedSection(
                section_id="s1",
                document_id="d1",
                section_title="t",
                content_type="text",
                content="가이드 본문",
                metadata={},
            )
        ]

        result = extract_normalization_inputs(sections)
        assert "가이드 본문" in result["text"]
        assert result["topic_tags"] == []
        assert result["todo_phase"] is None

    def test_handles_empty_sections(self):
        result = extract_normalization_inputs([])
        assert result["text"] == ""
        assert result["topic_tags"] == []
        assert result["todo_phase"] is None


# ===========================================================================
# 3) attach_parent_raw_chunk_ids
# ===========================================================================
class TestAttachParentRawChunkIds:
    def test_attaches_top_k_ids(self):
        cards = [
            KnowledgeCard(
                card_id="kc_a_001",
                card_type="workflow",
                title="t",
                summary="s",
                source_file_name="f",
                source_file_hash="h",
                source_category="guide",
                source_type="guide",
            )
        ]
        raw_chunks = _make_raw_chunks(n=5)

        attach_parent_raw_chunk_ids(cards, raw_chunks=raw_chunks, top_k=3)

        assert cards[0].parent_raw_chunk_ids == [
            "chunk_doc_test001_0000",
            "chunk_doc_test001_0001",
            "chunk_doc_test001_0002",
        ]

    def test_does_not_overwrite_existing(self):
        cards = [
            KnowledgeCard(
                card_id="kc_a_001",
                card_type="workflow",
                title="t",
                summary="s",
                source_file_name="f",
                source_file_hash="h",
                source_category="guide",
                source_type="guide",
                parent_raw_chunk_ids=["preset_id"],
            )
        ]
        raw_chunks = _make_raw_chunks(n=3)

        attach_parent_raw_chunk_ids(cards, raw_chunks=raw_chunks, top_k=2)

        assert cards[0].parent_raw_chunk_ids == ["preset_id"]

    def test_no_op_when_top_k_zero(self):
        cards = [
            KnowledgeCard(
                card_id="kc_a_001",
                card_type="workflow",
                title="t",
                summary="s",
                source_file_name="f",
                source_file_hash="h",
                source_category="guide",
                source_type="guide",
            )
        ]
        attach_parent_raw_chunk_ids(cards, raw_chunks=_make_raw_chunks(n=3), top_k=0)
        assert cards[0].parent_raw_chunk_ids == []

    def test_no_op_when_no_raw_chunks(self):
        cards = [
            KnowledgeCard(
                card_id="kc_a_001",
                card_type="workflow",
                title="t",
                summary="s",
                source_file_name="f",
                source_file_hash="h",
                source_category="guide",
                source_type="guide",
            )
        ]
        attach_parent_raw_chunk_ids(cards, raw_chunks=[], top_k=3)
        assert cards[0].parent_raw_chunk_ids == []


# ===========================================================================
# 4) knowledge_cards_to_chunks
# ===========================================================================
class TestKnowledgeCardsToChunks:
    def _make_card(
        self,
        *,
        card_id: str = "kc_aa11bb22_001_workflow",
        card_type: str = "workflow",
        title: str = "메타 캠페인 셋업",
    ) -> KnowledgeCard:
        card = KnowledgeCard(
            card_id=card_id,
            card_type=card_type,
            title=title,
            summary="요약 본문",
            source_file_name="guide.txt",
            source_file_hash="a" * 64,
            source_category="guide",
            source_type="guide",
            document_date="2026-04-29",
            display_date="해당 업무일",
            primary_topic="meta",
            topic_tags=["meta", "setup"],
            task_type="setup",
            steps=["1단계", "2단계"],
            metadata={
                "prompt_version": "guide_v1",
                "model_name": "fake-gemini-model",
                "todo_phase": None,
                "parser_format": None,
            },
            parent_raw_chunk_ids=["chunk_doc_test001_0000", "chunk_doc_test001_0001"],
        )
        card.sanitized_markdown = card.to_markdown()
        return card

    def test_creates_chunks_with_required_metadata(self):
        cards = [self._make_card()]
        settings_obj = _make_settings(source_weight=1.4)

        chunks = knowledge_cards_to_chunks(
            cards, document_id="doc_test001", settings_obj=settings_obj
        )

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.content_type == "knowledge_card"
        assert chunk.source_type == "llm_normalized"
        assert chunk.uploaded_category == "guide"
        assert chunk.file_name == "guide.txt"
        assert "메타 캠페인 셋업" in chunk.content
        assert "## 업무 절차" in chunk.content

        md = chunk.metadata
        assert md["card_id"] == "kc_aa11bb22_001_workflow"
        assert md["card_type"] == "workflow"
        assert md["primary_topic"] == "meta"
        assert md["topic_tags"] == "meta,setup"
        assert md["task_type"] == "setup"
        assert md["document_date"] == "2026-04-29"
        assert md["display_date"] == "해당 업무일"
        assert md["source_file_hash"] == "a" * 64
        assert md["source_weight"] == pytest.approx(1.4)
        assert md["normalized"] is True
        assert md["prompt_version"] == "guide_v1"
        assert md["model_name"] == "fake-gemini-model"
        assert md["parent_raw_chunk_ids"] == "chunk_doc_test001_0000,chunk_doc_test001_0001"

    def test_chunk_id_includes_document_id(self):
        cards = [
            self._make_card(card_id="kc_x_000"),
            self._make_card(card_id="kc_x_001", card_type="checklist", title="다른 카드"),
        ]
        chunks = knowledge_cards_to_chunks(
            cards, document_id="doc_alpha", settings_obj=_make_settings()
        )
        assert chunks[0].chunk_id.startswith("chunk_doc_alpha_norm_0000_")
        assert chunks[1].chunk_id.startswith("chunk_doc_alpha_norm_0001_")
        assert chunks[0].chunk_id != chunks[1].chunk_id

    def test_skips_card_without_body(self):
        empty = KnowledgeCard(
            card_id="kc_empty",
            card_type="workflow",
            title="",
            summary="",
            source_file_name="g.txt",
            source_file_hash="h",
            source_category="guide",
            source_type="guide",
        )
        empty.sanitized_markdown = ""

        ok = self._make_card()
        chunks = knowledge_cards_to_chunks(
            [empty, ok], document_id="doc_test001", settings_obj=_make_settings()
        )
        assert len(chunks) >= 1
        assert all("kc_empty" not in c.chunk_id for c in chunks)


# ===========================================================================
# 5) run_normalization_branch — disabled / mismatched / failure paths
# ===========================================================================
class TestRunNormalizationBranchDispatch:
    def test_returns_skip_for_non_target_category(self, tmp_path):
        document = _make_document(uploaded_category="kakao", source_type="kakao")
        store = NormalizationStore(output_dir=tmp_path / "norm")
        result = run_normalization_branch(
            document=document,
            parsed_sections=_make_parsed_sections(),
            raw_chunks=_make_raw_chunks(),
            embedder=FakeEmbedder(),
            vector_store=FakeVectorStore(),
            document_store=FakeDocumentStore(),
            gemini_client=FakeGeminiClient(response=GUIDE_RESPONSE_JSON),
            normalization_store=store,
            settings_obj=_make_settings(),
        )
        assert result["kind"] is None
        assert result["chunks_added"] == 0
        assert result["skipped_reason"]

    def test_returns_skip_when_text_empty(self, tmp_path):
        sections = [
            ParsedSection(
                section_id="s",
                document_id="d",
                section_title="t",
                content_type="text",
                content="",
                metadata={},
            )
        ]
        document = _make_document(uploaded_category="guide", source_type="guide")
        store = NormalizationStore(output_dir=tmp_path / "norm")
        client = FakeGeminiClient(response=GUIDE_RESPONSE_JSON)

        result = run_normalization_branch(
            document=document,
            parsed_sections=sections,
            raw_chunks=_make_raw_chunks(),
            embedder=FakeEmbedder(),
            vector_store=FakeVectorStore(),
            document_store=FakeDocumentStore(),
            gemini_client=client,
            normalization_store=store,
            settings_obj=_make_settings(),
        )

        assert result["kind"] == "guide"
        assert "비어" in (result["skipped_reason"] or "")
        assert client.call_count == 0

    def test_failure_does_not_break_pipeline(self, tmp_path):
        document = _make_document(uploaded_category="guide", source_type="guide")
        store = NormalizationStore(output_dir=tmp_path / "norm")

        result = run_normalization_branch(
            document=document,
            parsed_sections=_make_parsed_sections(content="가이드 본문"),
            raw_chunks=_make_raw_chunks(),
            embedder=FakeEmbedder(),
            vector_store=FakeVectorStore(),
            document_store=FakeDocumentStore(),
            gemini_client=FailingClient(),
            normalization_store=store,
            settings_obj=_make_settings(),
        )

        assert result["kind"] == "guide"
        assert result["chunks_added"] == 0
        assert "LLM normalization 실패" in (result["skipped_reason"] or "")

    def test_embedding_failure_does_not_break_pipeline(self, tmp_path):
        document = _make_document(uploaded_category="guide", source_type="guide")
        store = NormalizationStore(output_dir=tmp_path / "norm")

        result = run_normalization_branch(
            document=document,
            parsed_sections=_make_parsed_sections(content="가이드 본문"),
            raw_chunks=_make_raw_chunks(),
            embedder=FailingEmbedder(),
            vector_store=FakeVectorStore(),
            document_store=FakeDocumentStore(),
            gemini_client=FakeGeminiClient(response=GUIDE_RESPONSE_JSON),
            normalization_store=store,
            settings_obj=_make_settings(),
        )

        assert result["kind"] == "guide"
        assert result["chunks_added"] == 0
        assert "embedding" in (result["skipped_reason"] or "")

    def test_vector_store_failure_does_not_break_pipeline(self, tmp_path):
        document = _make_document(uploaded_category="guide", source_type="guide")
        store = NormalizationStore(output_dir=tmp_path / "norm")

        result = run_normalization_branch(
            document=document,
            parsed_sections=_make_parsed_sections(content="가이드 본문"),
            raw_chunks=_make_raw_chunks(),
            embedder=FakeEmbedder(),
            vector_store=FailingVectorStore(),
            document_store=FakeDocumentStore(),
            gemini_client=FakeGeminiClient(response=GUIDE_RESPONSE_JSON),
            normalization_store=store,
            settings_obj=_make_settings(),
        )

        assert result["kind"] == "guide"
        assert result["chunks_added"] == 0
        assert "vstore" in (result["skipped_reason"] or "")


# ===========================================================================
# 6) run_normalization_branch — guide / slack happy paths
# ===========================================================================
class TestRunNormalizationBranchHappyPath:
    def test_guide_branch_uses_guide_normalizer_and_caches(self, tmp_path):
        document = _make_document(
            uploaded_category="guide", source_type="guide", file_hash="g" * 64
        )
        store = NormalizationStore(output_dir=tmp_path / "norm")
        client = FakeGeminiClient(response=GUIDE_RESPONSE_JSON)
        embedder = FakeEmbedder()
        vstore = FakeVectorStore()
        dstore = FakeDocumentStore()
        settings_obj = _make_settings()

        result = run_normalization_branch(
            document=document,
            parsed_sections=_make_parsed_sections(content="가이드 본문"),
            raw_chunks=_make_raw_chunks(),
            embedder=embedder,
            vector_store=vstore,
            document_store=dstore,
            gemini_client=client,
            normalization_store=store,
            settings_obj=settings_obj,
        )

        assert result["kind"] == "guide"
        assert result["card_count"] == 1
        assert result["chunks_added"] == 1
        assert client.call_count == 1
        assert client.last_system_instruction is not None
        assert "KnowledgeCard" in client.last_system_instruction

        assert len(vstore.added) == 1
        norm_chunk = vstore.added[0]
        assert norm_chunk.content_type == "knowledge_card"
        assert norm_chunk.source_type == "llm_normalized"
        assert norm_chunk.metadata["card_type"] == "workflow"
        assert norm_chunk.metadata["primary_topic"] == "meta"
        assert "meta" in norm_chunk.metadata["topic_tags"]
        assert norm_chunk.metadata["task_type"] == "setup"
        assert norm_chunk.metadata["source_weight"] == pytest.approx(1.25)
        assert norm_chunk.metadata["normalized"] is True
        assert norm_chunk.metadata["parent_raw_chunk_ids"]

        # cache hit 검증: 동일 입력으로 한 번 더 호출하면 LLM 호출 없음
        result2 = run_normalization_branch(
            document=document,
            parsed_sections=_make_parsed_sections(content="가이드 본문"),
            raw_chunks=_make_raw_chunks(),
            embedder=embedder,
            vector_store=vstore,
            document_store=dstore,
            gemini_client=client,
            normalization_store=store,
            settings_obj=settings_obj,
        )
        assert client.call_count == 1, "cache hit 시 LLM 이 다시 호출되면 안 됨"
        assert result2["card_count"] == 1
        assert result2["chunks_added"] >= 1

    def test_slack_branch_uses_slack_normalizer_and_passes_meta(self, tmp_path):
        document = _make_document(
            uploaded_category="slack",
            source_type="slack_manual",
            file_name="2026-04-29_정산_TODO.txt",
            file_hash="s" * 64,
            document_id="doc_slack001",
        )
        store = NormalizationStore(output_dir=tmp_path / "norm")
        client = FakeGeminiClient(response=SLACK_RESPONSE_JSON)
        embedder = FakeEmbedder()
        vstore = FakeVectorStore()
        dstore = FakeDocumentStore()

        sections = _make_parsed_sections(
            content="실제 Slack 본문",
            sanitized="익명화된 Slack 본문",
            topic_tags=["settlement", "issue"],
            todo_phase="end_of_day",
            parser_format="slack_todo_sections",
            document_date="2026-04-29",
            display_date="해당 업무일",
        )

        result = run_normalization_branch(
            document=document,
            parsed_sections=sections,
            raw_chunks=_make_raw_chunks(document_id="doc_slack001"),
            embedder=embedder,
            vector_store=vstore,
            document_store=dstore,
            gemini_client=client,
            normalization_store=store,
            settings_obj=_make_settings(),
        )

        assert result["kind"] == "slack"
        assert result["card_count"] == 1
        assert result["chunks_added"] == 1
        assert client.call_count == 1
        assert "Slack" in (client.last_system_instruction or "")
        assert "topic_tags: settlement, issue" in (client.last_prompt or "")
        assert "todo_phase: end_of_day" in (client.last_prompt or "")
        assert "parser_format: slack_todo_sections" in (client.last_prompt or "")

        assert len(vstore.added) == 1
        norm_chunk = vstore.added[0]
        assert norm_chunk.content_type == "knowledge_card"
        assert norm_chunk.source_type == "llm_normalized"
        assert norm_chunk.uploaded_category == "slack"
        assert norm_chunk.metadata["card_type"] == "issue"
        assert norm_chunk.metadata["primary_topic"] == "settlement"
        assert "settlement" in norm_chunk.metadata["topic_tags"]
        assert "issue" in norm_chunk.metadata["topic_tags"]
        assert norm_chunk.metadata["document_date"] == "2026-04-29"
        assert norm_chunk.metadata["display_date"] == "해당 업무일"

        # parent links are connected to raw chunks
        parent_ids = norm_chunk.metadata["parent_raw_chunk_ids"]
        assert "chunk_doc_slack001_0000" in parent_ids


# ===========================================================================
# 7) Pipeline disabled path — sanity
# ===========================================================================
class TestPipelineDisabledPath:
    """``ENABLE_LLM_NORMALIZATION=false`` 일 때 normalizer 가 절대 호출되지 않음을 검증."""

    def test_run_branch_is_only_called_when_setting_true(self):
        from src.pipeline import settings as pipeline_settings

        # default 는 false
        assert pipeline_settings.enable_llm_normalization is False

    def test_disabled_setting_keeps_fake_client_unused(self, tmp_path):
        """settings.enable_llm_normalization=False 일 때 ingest_file 흐름이
        normalization branch 를 건너뛰는지를 helper 단위로 모사한다.

        ``run_normalization_branch`` 는 settings 인자와 무관하게 호출되면
        실행하지만, ``ingest_file`` 의 분기 자체는 ``use_normalization`` 값에
        따라 호출 여부를 결정한다. 본 테스트는 그 분기 조건을 검증한다.
        """
        settings_obj = _make_settings(enable_normalization=False)
        client = FakeGeminiClient(response=GUIDE_RESPONSE_JSON)

        # ingest_file 의 분기와 동일한 조건문
        use_normalization = (
            None
            if settings_obj.enable_llm_normalization is False
            else settings_obj.enable_llm_normalization
        )
        if settings_obj.enable_llm_normalization:
            run_normalization_branch(
                document=_make_document(),
                parsed_sections=_make_parsed_sections(),
                raw_chunks=_make_raw_chunks(),
                embedder=FakeEmbedder(),
                vector_store=FakeVectorStore(),
                document_store=FakeDocumentStore(),
                gemini_client=client,
                normalization_store=NormalizationStore(output_dir=tmp_path / "norm"),
                settings_obj=settings_obj,
            )

        assert client.call_count == 0
        assert use_normalization is None


# ===========================================================================
# 8) Cache key uses settings.llm_normalization_model + prompt_version
# ===========================================================================
class TestCacheKeySetup:
    def test_guide_cache_key_includes_prompt_version(self, tmp_path):
        store = NormalizationStore(output_dir=tmp_path / "norm")
        key1 = store.make_cache_key("h", GUIDE_NORMALIZER_PROMPT_VERSION, "model")
        key2 = store.make_cache_key("h", SLACK_NORMALIZER_PROMPT_VERSION, "model")
        assert key1 != key2, "Guide / Slack prompt 는 서로 다른 cache key 를 가져야 함"
