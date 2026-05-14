# LF_Hazzys_Bot MVP Development Navigation

## 1. 이 문서의 목적

- 이 문서는 LF_Hazzys_Bot 프로젝트의 개발 방향을 유지하기 위한 네비게이션 문서다.
- Cursor 또는 다른 개발 보조 도구가 작업할 때, 현재 프로젝트가 어디까지 왔고 다음에 무엇을 해야 하는지 파악하기 위해 참조한다.
- `README.md` 는 공개용 프로젝트 설명 문서이고, 이 문서는 개발 진행 방향과 MVP 단계별 판단 근거를 정리하는 문서다.
- 기능 개발 전에는 이 문서를 먼저 확인하고, 현재 Step 의 목적과 제약을 벗어나지 않도록 한다.
- 이 문서가 곧 “개발 우선순위의 단일 출처(single source of truth)” 다. 새로운 Step 을 시작하거나 끝낼 때 이 문서의 "현재 위치" 와 "각 Step 의 현재 상태" 를 함께 업데이트한다.

## 2. 프로젝트의 초기 목표

- 신규 담당자 온보딩이나 기존 운영 건 인수인계 시 반복적으로 발생하는 설명 리소스를 줄이는 것이 목적이다.
- 업무 자료가 가이드, Slack Thread, 메신저 대화, Excel, Mail/PDF 등에 분산되어 있기 때문에, 이를 하나의 로컬 지식베이스로 관리하고자 한다.
- 단순 문서 검색이 아니라, 업무 절차, 체크리스트, 이슈, FAQ, 공유 문안, 운영 판단 단위로 업무 맥락을 검색하고 답변하는 것을 목표로 한다.
- 특정 브랜드에 고정된 구조가 아니라, 파일 구성만 바꾸면 다른 브랜드 운영 건에도 적용 가능한 구조를 지향한다.
- 외부 Vector DB 나 SaaS 를 사용하지 않고, 로컬 ChromaDB 와 로컬 파일 저장소를 중심으로 동작한다.
- 외부 LLM 호출은 Gemini API 중심으로 제한하며, 민감정보와 원본 데이터는 GitHub 에 커밋하지 않는다.

## 3. MVP 1차에서 구현한 것

### 3.1 Streamlit 기반 관리자 / 운영 콘솔

- 문서 업로드
- 문서 색인
- 검색 테스트
- 업무 QA
- 정규화 문서 관리
- API 상태 확인

### 3.2 로컬 RAG 구조

- Parser / Cleaning / Anonymization
- Raw Chunk 생성
- Embedding (Gemini / Local 모드 전환 가능)
- Local ChromaDB 색인
- Retriever / Reranker
- Prompt Builder
- Gemini 기반 답변 생성

### 3.3 LLM-based Document Normalization

- raw 업무 문서를 다음과 같은 단위로 정규화한다.
  - workflow
  - checklist
  - issue
  - FAQ
  - decision
  - glossary
  - communication_template
- Normalized Document 를 생성한다.
- Raw Chunk 와 Normalized Document 를 병렬로 색인한다.
- Normalized Document 를 QA 의 1차 근거로 사용한다.
- Raw Chunk 는 raw evidence / raw fallback 으로 유지한다.

### 3.4 Guide / Slack Thread 기반 normalizer

- Guide 문서 정규화
- Slack-style Thread 정규화
- NormalizationStore 와 cache 적용

### 3.5 Streamlit 정규화 문서 관리 UI

- 생성된 Normalized Document 를 조회 / 재정규화 / 관리할 수 있는 UI 를 제공한다.

### 3.6 Normalized Document 우선 retrieval / QA prompt

- retriever 가 Normalized Document chunk 를 raw chunk 보다 우선시킨다.
- QA prompt 는 Normalized Document 를 "주 근거" 로, raw chunk 를 "보조 근거" 로 분리해 구성한다.

### 3.7 공개 README / 내부 참조 문서 분리

- `README.md` 는 공개용 프로젝트 설명이다.
- `docs_internal/` 은 내부 맥락 문서용이며 GitHub 에 커밋하지 않는다.
- 민감정보, 실명, 실제 업무 히스토리는 GitHub 에 커밋하지 않는다.

## 4. MVP 1차에서 확인된 이슈

- 데이터가 늘어나면서 Vector Search 만으로는 질문 topic 과 다른 근거가 섞일 수 있다.
- 예를 들어 meta 관련 질문에 kakao 관련 주의사항이 섞이는 현상이 확인되었다.
- 이는 LLM 이 단순히 “멍청해진 것” 이라기보다, 검색 후보 선별 / topic filtering / reranking / raw fallback 제어가 부족한 **RAG 품질 문제** 에 가깝다.
- Normalized Document 가 존재하더라도, 실제 QA 에서 primary 근거로 잡히지 않고 raw fallback 으로 떨어질 수 있다.
- Slack Bot 기본 출력은 초기에는 너무 장황했고, 참고 근거 / 진단 정보가 과도하게 노출되는 문제가 있었다.
- 이에 따라 Slack 전용 formatter 를 개선해 기본 출력은 간결하게 만들고, debug 모드에서만 진단을 노출하도록 개선했다.

## 5. Slack Bot 연동에서 구현한 것

- Slack QA Bot 은 Streamlit 을 대체하지 않는다.
- Streamlit 은 관리자 / 운영 콘솔이다.
- Slack Bot 은 사용자가 Slack 채널에서 질문하는 인터페이스다.
- Slack Bot 은 자체 RAG 로직을 갖지 않고 **기존 `qa_pipeline` 을 호출하는 얇은 adapter** 다.
- Socket Mode 기반으로 구현했다.
- `app_mention` 이벤트만 처리한다.
- 아래 기능은 의도적으로 구현하지 않았다.
  - Slack Thread 자동 수집
  - 채널 history 조회
  - 파일 / 이미지 다운로드
  - 자동 색인
- Slack 기본 출력은 답변 본문(1~5번 섹션) 만 보여준다.
- 참고 근거, raw fallback, 진단 정보는 기본적으로 숨김 처리한다.
- 질문 끝에 `--debug` 가 있을 때만 진단 정보와 짧은 source 요약을 보여준다.

## 6. MVP 2차의 큰 목표

**MVP 2차의 목표는 기능 확장보다 RAG 품질 고도화다.**

### 핵심 목표

- 질문 topic 과 다른 근거가 섞이는 문제를 줄인다.
- Normalized Document 가 있으면 raw chunk 보다 우선 사용되도록 한다.
- raw fallback 이 과하게 사용되지 않도록 한다.
- Slack debug 모드와 Streamlit 검색 테스트에서 검색 실패 원인을 쉽게 확인할 수 있게 한다.
- 이후 Kakao, Excel, Mail/PDF, Hybrid Retrieval 확장을 위한 기반을 정리한다.

## 7. MVP 2차 아젠다

각 Step 은 위에서 아래 순서대로 진행한다. 검색 품질 고도화 (Step 1~5) 가 끝난 뒤에 Hybrid Retrieval / Contextual Chunking (Step 6~7) 으로 넘어간다. 데이터 소스 확장 (Step 8) 은 그 이후다.

### Step 1. Retrieval Diagnostics 강화

**목표**

- 검색 품질을 바로 바꾸기 전에, 왜 특정 chunk 가 검색되었는지 확인할 수 있게 만든다.

**구현 내용**

- 진단에 다음 필드를 추가한다.
  - `query_topic`
  - `query_intent`
  - `query_date`
  - `retrieved_count`
  - `passed_count`
  - `topic_mismatch_count`
  - `normalized_document_candidate_count`
  - `raw_candidate_count`
- chunk 별 진단 필드를 추가한다.
  - `content_type`
  - `primary_topic`
  - `retrieval_role`
  - `final_score`
  - `topic_match` / `date_match`
- Slack `--debug` 와 Streamlit 에서 확인 가능하게 한다.

**주의**

- Step 1 에서는 penalty / filtering / reranking 정책을 변경하지 않는다.

**현재 상태**

- ✅ 완료됨.

---

### Step 2. Topic-aware Retrieval / Penalty 강화

**목표**

- 질문 topic 과 다른 topic 의 chunk 가 답변에 섞이지 않도록 한다.

**구현 내용**

- `_GENERIC_TOPICS` 상수 도입 (`common`, `general`, `shared`, `etc`, `unknown` 등).
- `_topic_match_factor` 라벨 확장.
  - `match` / `mismatch` / `none` 외에 `neutral` 추가.
  - chunk 가 generic topic 만 가지면 `neutral` 로 분류 → mismatch penalty 미적용.
- `is_clear_topic_mismatch` helper 추가.
  - query topic 이 명확하고, chunk 가 명확한 다른 topic 만 가질 때만 True.
  - generic 또는 빈 topic 은 mismatch 로 보지 않음.
- `apply_normalized_document_priority` 강화.
  - 명확한 topic mismatch normalized document chunk 는
    `retrieval_role="primary_card"` 로 승격하지 않고 `raw_fallback` 으로 격하.
  - 격하 시 `knowledge_card_boost` / `card_type_boost` 미적용 (`1.0`).
  - 같은 source_file 의 raw chunk 라도 mismatch 면 `raw_evidence` 승격 X.
  - 모든 격하 chunk 는 metadata 에 `topic_mismatch_demoted=True` 가 기록됨.
- `split_chunks_by_retrieval_role` 정렬 우선순위 정정.
  - `retrieval_role` 라벨이 있으면 `content_type` 보다 우선.
  - 격하된 normalized document chunk 가 `primary_cards` 그룹으로 다시 끌려가지 않도록 보장.
- 진단 카운트 추가.
  - `topic_mismatch_demoted_count` 를 retriever summary / qa_pipeline 결과 /
    Slack diagnostics 에 노출 (`--debug` 모드에서 0 이 아닐 때만 표시).
- 기존 `topic_mismatch_penalty=0.80` 환경변수는 그대로 사용 (새 env 추가 없음).

**주의**

- penalty 를 무작정 세게 걸지 말고, Step 1 에서 추가한 diagnostics 를 보고 테스트 기반으로 조정한다.
- topic_mismatch_penalty 값 변경 시 기존 테스트 (`tests/test_date_topic_retrieval.py`, `tests/test_knowledge_card_retrieval.py`) 의 boost / penalty 관련 단위 테스트가 깨지지 않도록 한다.
- raw_fallback 자체의 사용 한도 / `insufficient_evidence` 판단은 Step 4 로 미룬다.

**현재 상태**

- ✅ 완료됨.

---

### Step 3. Normalized Document 우선순위 점검

**목표**

- Normalized Document 가 있는데도 raw fallback 으로 떨어지는 문제를 줄인다.

**구현 내용**

- 이번 Step 은 정책을 **새로** 추가하지 않고, Step 1 + Step 2 결과로 동작이
  의도대로 작동하는지 **점검 + 회귀 방어 테스트** 를 보강한다.
- public helper alias 추가.
  - `src/rag/reranker.py` 에 `get_normalized_document_type(chunk_or_meta)` 공개 함수 추가.
  - 내부 `_resolve_normalized_document_type` 을 위임하며, 신규 `normalized_document_type`
    우선, legacy `card_type` fallback 으로 인식한다.
- 점검 / 회귀 테스트 파일 신규 추가: `tests/test_normalized_document_priority.py`.
  - Normalized Document 인식 (`content_type=normalized_document` / 신규 표준,
    legacy `knowledge_card`, `source_type=llm_normalized`).
  - document_type 해석 (신규 / legacy 키).
  - match Normalized Document 의 `primary_card` 승격 및 raw chunk 보다 우선.
  - mismatch Normalized Document 는 Step 2 정책에 따라 `raw_fallback` 으로 demote 유지.
  - `primary_normalized_document_count` / `primary_normalized_documents` 집계 정확성.
  - `answer_mode="knowledge_card"` legacy 라벨 유지.
  - `normalized_document_boost` / `knowledge_card_boost` (legacy alias) 모두 채워짐 검증.
  - 실제 문제 케이스: meta workflow + meta raw + kakao normalized 시나리오.
  - Slack `--debug` 출력에 `primary_normalized_document_count` 노출, 기본 출력 진단 숨김.
- 기존 인식 로직 / boost 정책 / 환경변수는 변경하지 않음.

**주의**

- 신규 표준 명칭 (`normalized_document`) 과 legacy 명칭 (`knowledge_card`) 을 모두 인식해야 한다.
- 색인 데이터를 다시 만들지 않아도 동작해야 한다.
- Step 2 의 `topic_mismatch_demoted` 정책이 그대로 유지되어야 한다 (mismatch chunk 가 다시 primary 로 끌려가지 않음).

**현재 상태**

- ✅ 완료됨.

---

### Step 4. Raw Fallback 오남용 방지

**목표**

- raw fallback 은 마지막 수단으로만 사용되도록 한다.
- raw_fallback 기반 답변의 신뢰도를 진단으로 명확히 표시한다.
- raw_fallback 자체는 제거하지 않는다 (정규화되지 않은 초기 데이터 호환 유지).

**구현 내용**

- ``src/rag/qa_pipeline.py`` 에 ``_summarize_raw_fallback_policy`` helper 추가.
  - qa_pipeline 결과 dict / log metadata 의 top-level 에 아래 진단 필드 노출.
    - ``raw_fallback_only``
    - ``raw_fallback_only_reason``
      - ``"no_primary_normalized_document"`` : 후보 normalized document 는 있었으나 primary 로 승격되지 못함 (topic mismatch demote 등)
      - ``"no_normalized_document_candidate"`` : normalized document 자체가 검색되지 않음
    - ``raw_fallback_topic_mismatch_count``
    - ``raw_fallback_topic_mismatch_ratio``
      (분모는 topic 정보가 명확한 non-generic raw_fallback 수. generic / common /
      unknown 만 가진 chunk 는 분모에 포함하지 않아 mismatch 강도가 희석되지 않게 한다.)
    - ``primary_evidence_available``
      (primary normalized document 가 있거나 raw_evidence 가 있을 때 True)
    - ``normalized_document_available``
      (retrieval candidate 단계에서 normalized document 가 하나라도 있었는지)
    - ``weak_evidence_warning``
    - ``evidence_strength`` ∈ {``strong``, ``medium``, ``weak``, ``insufficient``}
- ``raw_fallback_only`` 판정 정책.
  - ``primary_normalized_document_count == 0`` AND ``raw_fallback_count > 0`` 이면 True.
  - raw_evidence 가 있어도 raw_fallback chunk 가 함께 있으면 raw_fallback_only 는 True 일 수 있다 — 이 경우 ``primary_evidence_available`` 로 raw_evidence 존재 여부를 별도 진단한다.
- ``evidence_strength`` 분류 정책.
  - ``generation_skipped`` → ``insufficient``
  - primary Normalized Document 존재 → ``strong``
  - raw_evidence 존재 → ``medium``
  - raw_fallback 만 존재
    - 모든 raw_fallback 이 topic match / generic 이고 mismatch_count == 0 → ``medium``
    - mismatch ratio >= 0.7 → ``insufficient``
    - 그 외 → ``weak``
- ``weak_evidence_warning`` 정책 (보수적).
  - ``raw_fallback_only == True``
  - ``query_topic`` 명확함
  - ``raw_fallback_count >= 2``
  - ``raw_fallback_topic_mismatch_ratio >= 0.7``
  - 위 4 조건을 모두 만족할 때만 True.
- raw_fallback 의 ``topic_match`` 판정은 Step 2 helper ``is_clear_topic_mismatch`` 를 그대로 재사용.
  - chunk topic 이 generic (common / general / unknown / etc) 만 있으면 mismatch 로 잡지 않는다.
  - query topic 이 None 이면 mismatch warning 을 강하게 켜지 않는다.
- ``answer_mode`` 라벨은 그대로 유지 (``knowledge_card`` / ``raw_fallback`` /
  ``insufficient_evidence``). 신규 answer_mode 는 추가하지 않는다.
- Slack ``--debug`` 출력 (``src/slack_bot/formatter.py``) 에 새 진단 라인 추가.
  - ``evidence_strength`` (항상)
  - ``weak_evidence_warning`` (True 일 때만)
  - ``raw_fallback_only`` (True 일 때만, reason 포함)
  - ``raw_fallback_topic_mismatch_count`` (>0 일 때만 ratio 와 함께)
  - 기본 출력은 변경하지 않는다.
- Slack adapter (``src/slack_bot/qa_adapter.py``) 가 신규 진단을 diagnostics
  dict 에 그대로 전달한다.
- Streamlit ``app/pages/3_업무_QA.py`` caption 에 ``evidence_strength`` /
  ``raw_fallback_only`` / mismatch count / ratio / weak_evidence_warning 표시.
  - weak_evidence_warning=True 일 때 ``st.warning`` 으로 약한 근거 안내 노출.
- ``tests/test_raw_fallback_policy.py`` 신규 추가.
  - evidence_strength 분류 (strong / medium / weak / insufficient).
  - raw_fallback_only 판정 (primary 있을 때 / 없을 때 / common 만 있을 때).
  - weak_evidence_warning 정책 (query_topic 없을 때 켜지지 않음, generic chunk 가
    분모에 들어가도 정상 동작, mismatch ratio >= 0.7 일 때만 True).
  - 시나리오 A: 메타 질문 + kakao×2 + common×1 raw_fallback → weak_evidence_warning=True.
  - 시나리오 B: 메타 normalized + kakao raw_fallback → strong / warning=False.
  - 시나리오 C: query_topic 불명확 + raw_fallback only → warning 과도 X.
  - Slack adapter / formatter --debug 표시 통합 테스트.
  - Step 1/2/3 진단 / answer_mode 호환성 회귀.

**주의**

- 답변 자체를 막지 않는다 — "근거가 약함" 을 답변과 진단에 명확히 표시하는 방향을 우선했다.
- 외부 ``answer_mode`` 라벨 (``knowledge_card`` / ``raw_fallback`` / ``insufficient_evidence``) 은 그대로 유지해 Streamlit / Slack / 기존 테스트와의 호환성을 깨지 않는다.
- BM25 / Hybrid Retrieval / Contextual Chunking 은 이 Step 범위가 아니다 — Step 6 이후로 미룬다.

**현재 상태**

- ✅ 완료됨.

---

### Step 5. Slack Debug 진단 강화

**목표**

- Slack `--debug` 에서 검색 실패 원인을 더 쉽게 파악한다.
- Step 1~4 에서 늘어난 diagnostics 를 Slack thread 에서 운영자가 빠르게 읽을 수 있도록 구조화한다.
- 기본 Slack 출력은 계속 간결하게 유지하고, debug 모드에서만 진단을 노출한다.

**구현 내용**

- `src/slack_bot/formatter.py` 의 debug 출력 구조를 아래 섹션으로 정리.
  - `*진단 요약*`
    - `evidence_strength`
    - `answer_mode`
    - `query_topic`
    - `weak_evidence_warning`
    - `raw_fallback_only` / `raw_fallback_only_reason` (해당 시)
  - `*검색 후보*`
    - `retrieved` / `passed`
    - `normalized_document_candidate` / `raw_candidate`
    - `primary_normalized_document`
    - `raw_evidence` / `raw_fallback`
  - `*Topic 진단*`
    - `topic_mismatch_count`
    - `topic_mismatch_demoted_count`
    - `raw_fallback_topic_mismatch: count / raw_fallback_count (ratio=...)`
    - `query_intent` / `query_date` (있을 때)
  - `*Top Sources*`
    - 최대 3개 source 만 표시.
    - `file_name / section_title` 형태의 제목.
    - `content_type`, `primary_topic`, `role`, `final_score`, `topic_match`,
      `topic_mismatch_demoted` 를 source 별로 표시.
    - `final_score` 는 소수점 3자리로 표시.
    - preview 는 120자 이하로 제한.
- debug helper 분리.
  - `_format_debug_summary`
  - `_format_debug_candidate_counts`
  - `_format_debug_topic_diagnostics`
  - `_format_sources_debug`
  - `_format_source_debug_fields`
- Step 4 의 weak evidence 상태를 debug 요약에서 명확히 표시.
  - `evidence_strength` 는 항상 표시.
  - `weak_evidence_warning` 은 True/False 를 항상 표시.
  - `raw_fallback_only=True` 인 경우 reason 과 함께 표시.
- Slack 기본 출력은 변경하지 않음.
  - 질문에 `--debug` 가 없으면 답변 본문만 노출.
  - 참고 근거 / 진단 / raw preview 숨김 유지.
- `tests/test_slack_bot.py` 에 Step 5 구조화 테스트 추가.
  - `진단 요약`, `검색 후보`, `Topic 진단`, `Top Sources` 섹션 표시.
  - weak evidence 필드 표시.
  - Top Sources 최대 3개 제한.
  - score 소수점 3자리 표시.
  - 긴 preview 제한.
  - 기본 출력 debug 섹션 미노출.
- 기존 Step 1~4 회귀 테스트 기대값을 새 debug 구조에 맞게 갱신.

**현재 상태**

- ✅ 완료됨.

---

### Step 6. Hybrid Retrieval / BM25 도입

**목표**

- 약어, 캠페인명, 시트명, 매체명 등 exact keyword 검색 성능을 높인다.

**예상 작업**

- BM25 index 추가
- Vector Search 와 BM25 결과 병합
- RRF 또는 weighted fusion 적용
- 최종 reranking 유지

**주의**

- 기존 retriever / reranker 인터페이스 (`Retriever.retrieve_with_details`) 는 그대로 유지해 호출자 코드를 깨지 않는다.

**현재 상태**

- 미시작.

---

### Step 7. Contextual Chunking 강화

**목표**

- chunk 자체에 업무 맥락을 더 잘 붙여 검색 정확도를 높인다.

**예상 작업**

- `embedding_text` 에 `source_category`, `primary_topic`, `task_type`, `section_title`, `when_to_use` 등 context header 강화
- Normalized Document chunk 와 Raw Chunk 의 embedding text 구조 점검

**주의**

- embedding text 구조를 바꾸면 재색인이 필요할 수 있다. 변경 범위를 작게 유지하고, 마이그레이션 절차를 함께 정리한다.

**현재 상태**

- 미시작.

---

### Step 8. 데이터 소스 확장

검색 품질 고도화 (Step 1~7) 이후 진행한다.

**예상 순서**

1. Kakao / Messenger 대화 정규화
2. Excel 핵심 시트 정규화
3. Mail / PDF 히스토리 검색
4. OCR / 이미지 분석

**현재 상태**

- 미시작.

## 8. 현재 개발 원칙

- 대규모 재작성 금지.
- 한 번에 하나의 Step 만 진행한다.
- 기존 Streamlit / Slack Bot / `qa_pipeline` 구조를 유지한다.
- Slack Bot 은 계속 `qa_pipeline` adapter 역할만 수행한다.
- RAG core 로직을 Slack Bot 내부에 복사하지 않는다.
- 기능 변경 전 diagnostics 와 테스트를 먼저 확인한다.
- 기존 테스트가 깨지면 안 된다.
- 외부 API 를 실제 호출하는 테스트를 만들지 않는다.
- `data/raw`, `data/processed`, `storage`, `.env`, `docs_internal` 은 커밋하지 않는다.
- 공개 문서에는 민감정보, 실명, 실제 업무 히스토리를 기록하지 않는다.

## 9. Cursor 작업 시 지켜야 할 규칙

- 작업 전 이 문서를 먼저 읽고 현재 Step 을 확인한다.
- 현재 Step 의 목표 외 작업을 하지 않는다.
- 다음 Step 작업을 미리 구현하지 않는다.
- 변경 파일을 최소화한다.
- 작업 완료 후 반드시 다음을 요약한다.
  1. 이번 Step 이름
  2. 해결하려는 문제
  3. 수정한 파일
  4. 변경한 동작
  5. 변경하지 않은 것
  6. 테스트 결과
  7. 다음 Step 에서 할 일

## 10. 현재 위치

- ✅ MVP 1차 완료
- ✅ Slack QA Bot 연결 완료
- ✅ Slack 기본 출력 간결화 완료
- ✅ MVP 2차 Step 1 — Retrieval Diagnostics 강화 완료
- ✅ MVP 2차 Step 2 — Topic-aware Retrieval / Penalty 강화 완료
- ✅ MVP 2차 Step 3 — Normalized Document 우선순위 점검 완료
- ✅ MVP 2차 Step 4 — Raw Fallback 오남용 방지 완료
- ✅ MVP 2차 Step 5 — Slack Debug 진단 강화 완료
- ⏭️ 다음 예정 Step 은 **Step 6 — Hybrid Retrieval / BM25 도입**
