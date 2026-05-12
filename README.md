# Work RAG Assistant

> 사내 업무 문서를 로컬에서 색인하고 한국어로 질의응답하는 **로컬 RAG (Retrieval Augmented
> Generation) 챗봇** 프로토타입. 외부 Vector DB 나 SaaS 를 사용하지 않고, 모든 원본 자료와
> 색인 결과는 로컬 디스크에 저장된다. 기본 RAG 기능의 외부 호출은 답변/요약/임베딩 시점의
> **Google Gemini API** 중심으로 제한된다. 단, 선택 기능인 Slack QA Bot 을 활성화하면 Slack
> 메시지 수신/답변을 위해 **Slack API (Socket Mode)** 를 추가로 사용한다.

이 저장소는 **공개용 코드/설정/예시** 만 포함한다. 회사 / 광고주 / 매체사 / 사람 이름 같은
실제 업무 데이터는 어떤 형태로도 저장소에 포함되지 않는다 (자세한 내용은 [Security & Privacy
Notes](#security--privacy-notes) 참고).

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [지원 파일 유형](#3-지원-파일-유형)
4. [LLM-based Document Normalization 아키텍처](#4-llm-based-document-normalization-아키텍처)
5. [Task 1~7 기능 요약](#5-task-17-기능-요약)
6. [설치 및 실행 (Mac)](#6-설치-및-실행-mac)
7. [설치 및 실행 (Windows)](#7-설치-및-실행-windows)
8. [환경변수 설정 (.env.example)](#8-환경변수-설정-envexample)
9. [Streamlit 페이지 설명](#9-streamlit-페이지-설명)
10. [임베딩 / 검색 / 답변 옵션](#10-임베딩--검색--답변-옵션)
11. [테스트 명령어](#11-테스트-명령어)
12. [Security & Privacy Notes](#security--privacy-notes)
13. [현재 미지원 기능](#12-현재-미지원-기능)
14. [향후 확장 계획](#13-향후-확장-계획)
15. [Slack QA Bot (선택 기능)](#14-slack-qa-bot-선택-기능)

---

## 1. 프로젝트 개요

- 사용자가 자신의 업무 문서(가이드, 메신저 스레드 텍스트, 메일/카톡 내보내기, Excel 등)를
  로컬에 적재하면, 자연어 질문에 대해 단계별 절차 / 체크리스트 / 참고 근거를 한국어로
  답변하는 RAG 챗봇이다.
- **핵심 차별점 — LLM-based Document Normalization**
  - raw 업무 문서를 그대로 embedding 하는 단순 RAG 구조가 아니라, ingestion 단계에서 LLM 으로
    한 번 더 구조화하여 검색과 QA 에 적합한 **Normalized Document** (정규화 문서) 를 생성한 뒤
    이를 1차 근거로 활용한다.
  - **Normalized Document** 는 업무 절차 (workflow), 체크리스트 (checklist), 이슈 (issue),
    FAQ, 공유 문안 (communication_template), 용어 (glossary), 운영 판단 (decision) 등으로
    구조화된 문서 단위다.
  - **Raw Chunk** 는 원문 기반 보조 근거 또는 fallback 으로 함께 유지되어, 정규화 결과만으로
    답변이 어렵거나 LLM 정규화가 OFF 인 경우에도 RAG 가 동작한다.
- 설계 원칙
  - **외부 Vector DB 미사용**: Pinecone / Weaviate / Qdrant Cloud / Supabase 등 모두 사용하지
    않는다. 색인 결과는 로컬 [ChromaDB](https://github.com/chroma-core/chroma) (`./storage/
    chroma_db`) 에 저장한다.
  - **자체 LLM 추론 미사용**: Ollama / llama.cpp 등을 사용하지 않는다. 답변과 (옵션) 임베딩만
    Google Gemini API 로 호출한다.
  - **문서 수집/색인 단계의 외부 메신저 API 미사용**: Slack / 카카오톡 대화 자료는 사용자가
    직접 텍스트 파일로 내보내 업로드하는 방식을 기본으로 한다. 단, Slack QA Bot 선택 기능을
    활성화하면 Slack 에서 봇 멘션을 수신하고 thread 에 답변을 반환하기 위해 Slack API
    (Socket Mode) 를 사용한다. 현재 MVP 에서는 Slack Thread 자동 수집, 채널 history 조회,
    파일/이미지 다운로드, 자동 색인은 수행하지 않는다.
  - **모든 원본은 로컬 디스크**에만 보관된다. `data/raw`, `data/processed`, `storage/` 는
    저장소에 커밋하지 않는다.
- 임베딩 / 답변 / 검색은 모두 한국어 업무 맥락을 가정한다. UI 와 로그도 한국어 우선이다.

---

## 2. 시스템 아키텍처

raw 문서를 그대로 embedding 하지 않고, **LLM-based Document Normalization** 을 통해
**Normalized Document** 를 생성한 뒤 이를 검색과 답변의 1차 근거로 사용한다.
raw chunk 는 원문 보조 근거 또는 fallback 으로 함께 유지된다.

```
[사용자 업로드 파일]
  ├─ 텍스트 가이드 (txt / md / docx)
  ├─ 메신저 스레드 텍스트 (txt / md)
  ├─ 메신저 내보내기 텍스트 (txt)
  └─ 표 형식 자료 (xlsx / xlsm)
        ↓
[로컬 파일 저장소  ./data/raw/<카테고리>]
        ↓
[Parser / Cleaning / Anonymization]
        ↓
[(옵션) Excel 한국어 업무 요약 — Gemini API]
        ↓
        ┌──────────────────────────────────────────┐
        │                                          │
        ▼                                          ▼
[Raw Chunk 생성]                  [(옵션) LLM-based Document Normalization]
        │                                          ↓
        │                          [Normalized Document 생성]
        │                                          ↓
        │                          [Normalized Document Chunk 생성]
        │                                          │
        └──────────────────┬───────────────────────┘
                           ▼
        [Embedding (Gemini API | sentence-transformers 로컬)]
                           ↓
                [Local ChromaDB  ./storage/chroma_db]
                           ↓
        [Normalized Document 우선 Retrieval + Reranking + Parent-Child 보강]
                           ↓
                [Raw Evidence / Raw Fallback 보강]
                           ↓
        [Prompt Builder — Normalized Document 중심 답변 (또는 raw fallback)]
                           ↓
                   [Gemini Generation API]
                           ↓
            [근거 기반 한국어 답변 + 진단 메타데이터]
```

핵심 모듈

```
src/
  config.py             # .env 로드 + 경로/상수
  ingestion/            # 파일 → ParsedSection
  preprocessing/        # 정제, 청크, 비식별화
  summarization/        # Excel 한국어 요약
  normalization/        # LLM-based Document Normalization (Task 1~4)
  rag/                  # embedder, retriever, reranker, generator, qa_pipeline, prompt_builder
  storage/              # ChromaDB / file registry / processed JSONL
  schemas/              # dataclass (Document, Chunk, NormalizedDocument, RetrievedChunk, ...)
  utils/                # path / hash / time / encoding / token / cost
app/
  main.py               # Streamlit 홈
  pages/                # 1_문서_업로드 ~ 7_정규화_문서_관리
  components/           # UI helper
```

---

## 3. 지원 파일 유형

| 카테고리 | 용도 | 권장 확장자 |
| --- | --- | --- |
| **Guide** | 일반 업무 가이드 / 매뉴얼 / 운영 정책 문서 | docx / md / txt |
| **Slack-style 메신저 스레드 텍스트** | 사용자가 직접 복사해 저장한 메신저 대화 텍스트 | txt / md |
| **Kakao-style 메신저 내보내기 텍스트** | 메신저 대화를 텍스트 파일로 내보낸 자료 | txt |
| **Excel 표** | 표 형식 가이드 / 정산 시트 / 정의서 | xlsx / xlsm |
| **기타** | 분류 외 자료 (실험적) | * |

> 문서 수집/색인 목적의 메신저 자료는 사용자가 직접 텍스트로 내보내 업로드하는 방식을
> 기본으로 한다. Slack QA Bot 은 별도 선택 기능이며, Slack 에서 질문을 받아 기존 QA Pipeline
> 을 호출하는 사용자 질문 인터페이스 역할만 수행한다.

PDF / 이미지 OCR / Hybrid Retrieval / 자동 폴더 watcher 등은 현재 MVP 에서 지원하지 않는다.
([미지원 기능](#12-현재-미지원-기능) 참고)

---

## 4. LLM-based Document Normalization 아키텍처

**LLM-based Document Normalization** 은 raw 업무 문서를 LLM 으로 절차 / 체크리스트 / 이슈 /
FAQ / 공유 문안 등 검색·QA 친화적 구조로 정규화하는 **ingestion-time preprocessing 단계**다.
raw chunk 색인 흐름과는 **별도의 병렬 branch** 로 동작한다 — raw chunk 를 대체하는 것이
아니라, 같은 입력으로부터 별도의 Normalized Document 를 생성해 **함께** 색인한다.

```
[Parser / Cleaning / Anonymization 결과]
        │
        ├──────────────────── (병렬) ────────────────────┐
        ▼                                                ▼
[Raw Chunk 생성]                  [(옵션) LLM-based Document Normalization]
        │                                  │
        │                                  ├─ Guide       → workflow / checklist / faq /
        │                                  │                glossary / communication_template /
        │                                  │                decision
        │                                  └─ Slack-style → issue / decision / checklist /
        │                                                   communication_template / faq /
        │                                                   workflow
        │                                  ↓
        │                  [NormalizationStore  ./data/processed/normalized/{json,markdown}]
        │                  file_hash + prompt_version + model_name cache → 중복 호출 방지
        │                                  ↓
        │                  [Normalized Document Chunk 생성
        │                   (content_type=normalized_document, source_type=llm_normalized)]
        │                                  │
        └──────────────────────┬───────────┘
                               ▼
                       [Embedding + ChromaDB 색인]
                               ▼
   [Retriever / Reranker — Normalized Document 우선 boost → retrieval_role 라벨링]
                               ▼
   [QA Prompt Builder — Normalized Document 중심 답변 (raw evidence/fallback 보조)]
```

핵심 원칙

- **raw chunk 는 그대로 유지된다.** Normalized Document 생성 여부와 무관하게 원문 chunk 색인
  흐름은 항상 동작한다 (`ENABLE_LLM_NORMALIZATION=false` 일 때는 raw 흐름만 동작).
- **Normalized Document 는 1차 근거.** 검색 / 답변 단계에서 Normalized Document 가 raw chunk
  보다 우선 사용된다.
- **Raw Chunk 는 보조 근거 / fallback.** Normalized Document 가 없거나 부족할 때 raw evidence
  로 보강하거나, 정규화가 OFF 일 때 fallback 으로 사용된다.

`NormalizedDocument` 의 type 은 다음 중 하나다.

- `workflow` — 반복 가능한 업무 절차
- `checklist` — 확인 항목 중심
- `faq` — 자주 묻는 질문
- `decision` — 운영 판단 / 방향
- `glossary` — 용어 정의
- `communication_template` — 광고주 / 매체사 / 내부 공유 문안
- `issue` — 이슈 상황 / 원인 / 대응

각 Normalized Document 의 metadata 에는 `normalized_document_id`,
`normalized_document_type`, `primary_topic`, `topic_tags`, `task_type`, `document_date`,
`display_date`, `parent_raw_chunk_ids` 등이 함께 저장되어 검색 우선순위와 답변 근거 분리에
사용된다.

> **Backward compatibility**: 과거에 색인된 데이터는 `content_type="knowledge_card"` /
> `card_id` / `card_type` 으로 저장되어 있을 수 있다. retriever / reranker / qa pipeline
> 은 신규 (`normalized_document` / `normalized_document_id` / `normalized_document_type`)
> 와 legacy 키를 모두 인식하므로 기존 저장 데이터를 다시 색인하지 않아도 그대로 동작한다.
> 새로 색인되는 chunk 는 신규 표준을 사용한다.

---

## 5. Task 1~7 기능 요약

LLM-based Document Normalization MVP 는 7 단계로 구현되어 있다 (모두 완료).

| Task | 기능 | 핵심 산출물 |
| --- | --- | --- |
| 1 | NormalizedDocument schema + NormalizationStore | `src/schemas/normalized_document.py`, `src/normalization/normalization_store.py` |
| 2 | Guide document normalizer (LLM 호출 + cache) | `src/normalization/guide_normalizer.py` (`GuideDocumentNormalizer`), `normalization_prompt.py` |
| 3 | Slack-style 스레드 document normalizer (LLM 호출 + cache) | `src/normalization/slack_normalizer.py` (`SlackThreadDocumentNormalizer`), `normalization_prompt.py` |
| 4 | `ENABLE_LLM_NORMALIZATION` 옵션을 ingest pipeline 에 연결 | `src/normalization/pipeline_integration.py`, `src/pipeline.py` |
| 5 | Streamlit 정규화 문서 관리 UI (read-only) | `app/pages/7_지식카드_관리.py` (UI 표시명: "정규화 문서 관리"), `src/normalization/card_viewer.py` |
| 6 | 검색 단계 Normalized Document 우선 retrieval / reranker | `src/rag/reranker.py` (`apply_normalized_document_priority`), `src/rag/retriever.py` |
| 7 | QA prompt 가 Normalized Document 를 1차 근거로 사용 | `src/rag/prompt_builder.py` (`build_normalized_document_answer_prompt`), `src/rag/qa_pipeline.py` |

검색 / 답변 단계의 핵심 진단 필드

- `retrieval_role` ∈ {`primary_card`, `raw_evidence`, `raw_fallback`}
  → `primary_card` 는 Normalized Document 1차 근거를 가리키는 코드 레벨 라벨이다.
  외부 시스템 / 로깅 / 테스트 호환을 위해 라벨 문자열 자체는 변경하지 않는다.
- `answer_mode` ∈ {`knowledge_card`, `raw_fallback`, `insufficient_evidence`}
  → `knowledge_card` 는 "Normalized Document 중심 답변 모드" 를 의미하는 코드 레벨 라벨이며,
  외부 호환을 위해 문자열 자체는 변경하지 않는다.
- `normalized_document_id` / `normalized_document_type` (legacy: `card_id` / `card_type`
  도 같은 값으로 함께 채워진다)
- `primary_topic` / `task_type`
- `normalized_document_boost`, `normalized_document_type_boost`,
  `normalized_document_type_match`
- `parent_raw_chunk_ids`, `final_score`

이 필드들은 Streamlit 의 검색 테스트 / QA 페이지에서 그대로 확인할 수 있다. 기존
`knowledge_card_boost` / `card_type_boost` / `card_type_match` 등 legacy 진단 키도 동일한
값으로 함께 채워져, 기존 UI / 로깅 / 외부 분석 코드가 깨지지 않는다.

---

## 6. 설치 및 실행 (Mac)

```bash
# 1) 가상환경 (Python 3.11 권장)
python3.11 -m venv .venv
source .venv/bin/activate

# 2) 의존성
pip install --upgrade pip
pip install -r requirements.txt

# 3) 환경변수
cp .env.example .env
# .env 를 열어 GOOGLE_API_KEY 입력

# 4) (선택) smoke test : 외부 API 호출 없이 핵심 동작 확인
python scripts/smoke_test.py

# 5) 실행
streamlit run app/main.py
# 또는
bash scripts/run_app.sh
```

> 모든 명령은 반드시 가상환경(`.venv`) 안에서 실행한다. 시스템 파이썬에서 직접 실행하면
> numpy / chromadb 충돌이 발생할 수 있다.

---

## 7. 설치 및 실행 (Windows)

```bat
:: 1) 가상환경 (Python 3.11 권장)
python -m venv .venv
.venv\Scripts\activate

:: 2) 의존성
python -m pip install --upgrade pip
pip install -r requirements.txt

:: 3) 환경변수
copy .env.example .env
:: 메모장으로 .env 열고 GOOGLE_API_KEY 입력

:: 4) (선택) smoke test
python scripts\smoke_test.py

:: 5) 실행
streamlit run app/main.py
:: 또는
scripts\run_app.bat
```

Windows 추가 주의사항

- `app/pages/` 에는 한글 파일명이 포함되어 있다. Python 3.11 + Streamlit 1.10+ 에서 한글
  파일명을 정상 지원하지만, 콘솔이 깨지면 PowerShell 사용 또는 `chcp 65001` 로 UTF-8 콘솔
  전환 권장.
- `python` 이 인식되지 않는 환경에서는 `py -3.11` 로 대체할 수 있다.
- 회사 PC 의 보안 정책상 PyPI 접근이 차단된 경우, 사내 미러 또는 오프라인 wheel 패키지 사용을
  검토.
- sentence-transformers / chromadb 는 첫 설치 시 PyTorch 다운로드로 5~10분 소요될 수 있다.
  사양이 낮은 환경에서는 `EMBEDDING_PROVIDER=gemini` 권장.

---

## 8. 환경변수 설정 (.env.example)

`.env.example` 을 복사해 `.env` 를 만들고 필요한 값만 채운다. `.env` 는 절대 커밋하지
않는다 (`.gitignore` 에 등록되어 있음).

| 환경변수 | 용도 | 기본값 |
| --- | --- | --- |
| `GOOGLE_API_KEY` | Gemini API Key | (비어 있음) |
| `GENERATION_MODEL` | Gemini 답변 생성 모델 | `.env.example` 참고 |
| `EMBEDDING_PROVIDER` | `gemini` 또는 `local` | `gemini` |
| `LOCAL_EMBEDDING_MODEL` | local provider 사용 시 sentence-transformers 모델 | `.env.example` 참고 |
| `TOP_K` / `MIN_SIMILARITY_SCORE` / `MIN_FINAL_SCORE` / `MIN_RETRIEVED_CHUNKS` | 검색 / 임계값 | `.env.example` 참고 |
| `MAX_CHUNKS_PER_FILE` / `USE_MMR` / `MMR_LAMBDA` | 다양성 / 파일별 cap | `.env.example` 참고 |
| `ENABLE_DATE_FILTER` / `DATE_EXACT_MATCH_BOOST` / `DATE_MISMATCH_PENALTY` | 날짜 인식 검색 | `.env.example` 참고 |
| `TOPIC_MATCH_BOOST` / `TOPIC_MISMATCH_PENALTY` | 토픽 인식 검색 | `.env.example` 참고 |
| `ANONYMIZE_OUTPUT` 등 비식별화 토글 | UI / 답변 비식별화 | `.env.example` 참고 |
| `ENABLE_LLM_NORMALIZATION` | LLM-based Document Normalization ON/OFF (기본 OFF) | `false` |
| `LLM_DOCUMENT_NORMALIZATION_MODEL` (legacy: `LLM_NORMALIZATION_MODEL`) | 정규화에 사용할 Gemini 모델 | `.env.example` 참고 |
| `NORMALIZATION_MAX_DOCUMENTS_PER_FILE` (legacy: `NORMALIZATION_MAX_CARDS_PER_FILE`) / `NORMALIZED_DOCUMENT_SOURCE_WEIGHT` (legacy: `NORMALIZATION_CARD_SOURCE_WEIGHT`) 등 | 정규화 동작 | `.env.example` 참고 |
| `PRIORITIZE_NORMALIZED_DOCUMENTS` (legacy: `PRIORITIZE_KNOWLEDGE_CARDS`) / `NORMALIZED_DOCUMENT_CONTENT_BOOST` (legacy: `KNOWLEDGE_CARD_CONTENT_BOOST`) 등 | 검색 단계 Normalized Document 우선순위 | `.env.example` 참고 |
| `ANSWER_WITH_NORMALIZED_DOCUMENTS` (legacy: `ANSWER_WITH_KNOWLEDGE_CARDS`) / `MAX_PRIMARY_NORMALIZED_DOCUMENTS` (legacy: `MAX_PRIMARY_CARDS`) / `MAX_RAW_EVIDENCE_CHUNKS` 등 | 답변 단계 Normalized Document 중심 prompt | `.env.example` 참고 |

> 위 신규 환경변수가 없으면 자동으로 legacy 이름의 환경변수를 fallback 으로 읽는다.
> 기존 `.env` 가 있어도 그대로 동작한다.

전체 항목은 `.env.example` 에 주석과 함께 정리되어 있다.

---

## 9. Streamlit 페이지 설명

| 페이지 (UI 표시명) | 역할 |
| --- | --- |
| `1_문서_업로드` (문서 업로드) | 카테고리 선택 후 파일 업로드 → `data/raw/<카테고리>/` 에 저장 |
| `2_문서_색인` (문서 색인) | 새 파일 색인 / Excel 한국어 요약 옵션 / LLM 기반 문서 정규화 결과 카운트 표시 |
| `3_업무_QA` (업무 QA) | 질문 → 검색 → 답변. `answer_mode`, primary Normalized Document 목록, retrieval_role 진단 표시 |
| `4_검색_테스트` (검색 테스트) | 답변 생성 없이 retrieval 결과만 확인. content_type / normalized_document_type 필터 제공 |
| `5_API_상태확인` (API 상태 확인) | Gemini API Key / 모델 사용 가능 여부 점검 |
| `6_Excel_요약관리` (Excel 요약 관리) | Excel 시트별 한국어 업무 요약 재생성 / 캐시 확인 |
| `7_지식카드_관리` (정규화 문서 관리) | 생성된 Normalized Document JSON / Markdown 을 read-only 로 확인. 파일명은 한글 경로 호환을 위해 유지하되, UI 표시명은 "정규화 문서 관리" 로 변경되어 있다 |

CLI 도 함께 제공한다.

```bash
python scripts/ingest_folder.py             # 폴더 일괄 색인
python scripts/ingest_folder.py --enable-summary
python scripts/ingest_folder.py --no-skip
python scripts/summarize_excel_folder.py    # Excel 요약 일괄 생성
python scripts/reset_vector_db.py           # ChromaDB / processed 초기화 (data/raw 는 보존)
```

---

## 10. 임베딩 / 검색 / 답변 옵션

- 임베딩
  - `EMBEDDING_PROVIDER=gemini` (기본) — 빠르고 사양 부담이 적다. 임베딩 텍스트가 Gemini API
    로 전송된다.
  - `EMBEDDING_PROVIDER=local` — sentence-transformers 로 로컬 임베딩. 외부 전송 없음.
    `LOCAL_EMBEDDING_MODEL` 한 줄로 모델 교체 가능.
- 검색
  - 점수 = `(1 - cosine_distance)` 기반 raw similarity 에 source / 카테고리 / 콘텐츠 유형 /
    날짜 / 토픽 / Normalized Document / normalized_document_type 부스트가 곱해져
    `final_score` 가 계산된다.
  - `MIN_SIMILARITY_SCORE`, `MIN_FINAL_SCORE`, `MAX_CHUNKS_PER_FILE`, `USE_MMR` 로
    품질/다양성을 통제할 수 있다.
  - `MIN_RETRIEVED_CHUNKS` 미만이면 Gemini Generation 을 호출하지 않고 "근거 부족" 안내 메시지
    를 반환한다 (비용 절감).
- 답변
  - `ANSWER_WITH_NORMALIZED_DOCUMENTS=true` 이고 통과 chunk 안에 primary Normalized
    Document 가 있으면 **Normalized Document 중심 prompt** 로 답변한다 (코드 레벨 라벨은
    `answer_mode=knowledge_card` — 외부 호환 목적으로 문자열 자체는 변경하지 않는다).
  - primary Normalized Document 가 없거나 정규화가 OFF 면 **raw chunk prompt** 로 fallback
    한다 (`answer_mode=raw_fallback`).
  - 통과 chunk 자체가 부족하면 Gemini 호출 없이 "근거 부족" 메시지를 반환한다
    (`answer_mode=insufficient_evidence`).
  - Normalized Document type 에 따라 답변 형식이 default / communication_template /
    glossary 중 하나로 분기된다.
  - 답변과 참고 근거 모두에서 사람 실명 / @멘션 / 정확한 시간 / 원본 날짜를 노출하지 않도록
    prompt 와 비식별화 가드가 작동한다.
  - legacy 환경변수 `ANSWER_WITH_KNOWLEDGE_CARDS` / `MAX_PRIMARY_CARDS` 도 fallback 으로
    인식되므로 기존 `.env` 가 그대로 동작한다.

---

## 11. 테스트 명령어

```bash
# 1) 외부 API 호출 없는 smoke test
python scripts/smoke_test.py

# 2) 단위 테스트 / 시나리오 테스트 (외부 API 미호출)
python -m pytest tests -v

# 3) 일부 기능별 테스트만
python -m pytest tests/test_normalized_document_schema.py    -v
python -m pytest tests/test_normalization_store.py           -v
python -m pytest tests/test_guide_normalizer.py              -v
python -m pytest tests/test_slack_normalizer.py              -v
python -m pytest tests/test_pipeline_normalization.py        -v
python -m pytest tests/test_normalized_document_viewer.py    -v
python -m pytest tests/test_normalized_document_retrieval.py -v
python -m pytest tests/test_normalized_document_qa.py        -v
python -m pytest tests/test_legacy_compatibility.py          -v
python -m pytest tests/test_retrieval_precision.py           -v
python -m pytest tests/test_slack_bot.py                     -v
```

> 모든 테스트는 외부 Gemini API 호출 없이 동작하도록 fake client / fake generator 로 작성되어
> 있다. 네트워크 / API Key 없이도 통과해야 한다.

---

## Security & Privacy Notes

이 저장소는 공개용 코드/설정 만 포함한다. 실제 업무 자료, 회사 / 광고주 / 매체사 / 사람 이름 /
운영 사례 / 정산 내역 등은 어떤 형태로도 저장소에 커밋해서는 안 된다.

### 커밋 금지 대상 (.gitignore 로 차단됨)

| 경로 / 패턴 | 사유 |
| --- | --- |
| `.env`, `.env.local` | API Key 등 민감 환경변수 |
| `data/raw/**/*` | 업로드된 원본 업무 자료 |
| `data/processed/**/*` | 파싱/정규화/요약 결과 (원문 일부 포함) |
| `data/sample/**/*` (`README.md` / `.gitkeep` 제외) | 사용자가 둔 샘플 자료 |
| `storage/chroma_db/**/*` | 로컬 Vector DB |
| `storage/qa_logs/**/*` | 질문/답변 JSON 로그 |
| `storage/registry/**/*` | 색인 registry |
| `docs_internal/`, `*.private.md`, `INTERNAL_*.md`, `PRIVATE_*.md` | 내부 참조용 문서 (자세한 내용은 아래) |

### Gemini API 로 전송되는 데이터 범위

| 시점 | 전송되는 텍스트 | 비고 |
| --- | --- | --- |
| 색인 시 (`EMBEDDING_PROVIDER=gemini`) | 각 chunk 의 `embedding_text` | 임베딩 생성 |
| 색인 시 (`EMBEDDING_PROVIDER=local`) | 전송 없음 | sentence-transformers 로 로컬 임베딩 |
| Excel 한국어 요약 ON | 시트 단위 raw 표 텍스트 (cap 적용) | 요약 생성 |
| LLM-based Document Normalization ON | 정제된 문서 본문 + 메타데이터 | Normalized Document 생성 |
| 업무 QA 답변 시 | 사용자 질문 + 검색된 chunk 컨텍스트 | 답변 생성 |
| Query rewrite ON (기본 OFF) | 사용자 질문 | 검색 친화적 재작성 |
| `4_검색_테스트` 페이지 | 질문 임베딩 1회 (`gemini` provider 한정) | 답변 생성 호출 없음 |
| `5_API_상태확인` 페이지 | 사용자가 버튼을 눌렀을 때만 ping | 페이지 조회만으로는 무전송 |

> 외부 전송이 가장 적은 운영 모드 = `EMBEDDING_PROVIDER=local` + `ENABLE_LLM_NORMALIZATION=
> false` + `ENABLE_QUERY_REWRITE=false` + Excel 요약 비활성화. 이 경우 외부 호출은 사용자가
> 직접 질문할 때 1 회씩만 발생한다.

### 자료를 업로드하기 전에 확인해야 할 것

- 사내 보안 / 개인정보 / 광고주 NDA 정책상 외부 API (Gemini) 로 보내도 되는 범위인지 확인.
- 안 되는 자료는 업로드 전에 직접 마스킹 / 제거.
- 업로드 후에도 비식별화 옵션(`ANONYMIZE_OUTPUT=true`, `MASK_MENTIONS=true`,
  `MASK_LINKS=true`, `SHOW_RAW_CONTENT=false` 등)을 그대로 두는 것을 권장.

### 공개 README 작성 원칙

- 실제 회사명 / 브랜드명 / 광고주명 / 매체사명 / 사람 이름 / 부서명을 적지 않는다.
- 예시는 `브랜드 A`, `광고주`, `매체사`, `작성자`, `검토자`, `담당자` 같은 일반화된 표현만
  사용한다.
- 실제 운영 사례 / 캠페인명 / 정산 히스토리 / 메신저 대화 본문 / 업무 일정은 적지 않는다.
- 회사/팀 내부에서만 이해 가능한 표현 (사내 약어, 내부 시트명, 내부 결재 절차, 내부 도구명
  등) 도 적지 않는다.

### 내부 참조용 문서

회사 / 브랜드 / 운영 맥락 등 내부 정보는 별도의 **`docs_internal/`** 폴더에서만 다룬다.
이 폴더는 `.gitignore` 에 의해 git 추적에서 완전히 제외된다.

- `docs_internal/PROJECT_CONTEXT.md` — 본인 / 동료 / Cursor 가 로컬에서만 참조하는 내부 맥락
  문서 (커밋 금지).
- `docs_internal_template/PROJECT_CONTEXT.template.md` — 위 문서를 만들 때 사용할 **공개용
  템플릿**. 민감 정보가 들어가지 않은 항목 헤더만 포함한다.

내부 문서 작성 시에도 가능한 한 비식별화된 표현 (`작성자`, `검토자`, `담당자`, `브랜드 A`
등) 을 사용한다.

### README 와 RAG 지식베이스의 관계

- `README.md` 는 **프로젝트 설명 문서**이며, 앱 코드 어디에서도 자동으로 ingest 되지 않는다.
- 실제 RAG 지식베이스는 사용자가 `data/raw/<카테고리>/` 에 둔 업로드 파일을 색인할 때만
  생성된다.
- 따라서 `README.md` 의 문구는 답변 근거로 사용되지 않는다 (필요하면 사용자가 명시적으로
  `data/raw/` 에 복사한 경우에만).

---

## 12. 현재 미지원 기능

- PDF 파싱
- 이미지 OCR
- BM25 / Hybrid Retrieval
- LLM 기반 reranker (cross-encoder, Gemini reranker)
- 자동 폴더 watcher
- 사용자 피드백 기반 평가 자동화
- 사내 SSO / 인증

---

## 13. 향후 확장 계획

- PDF 파서 (`pypdf`, `pdfplumber`)
- OCR (`pytesseract`)
- BM25 + Vector hybrid retrieval
- Gemini / cross-encoder reranker
- 임베딩 모델 A/B 테스트 자동화
- retrieval hit rate / answer groundedness 지표화
- 사용자 피드백 기반 evaluation
- LLM-based Document Normalization 확장 (현재 Guide / Slack 스레드 → 향후 카카오 / 메일 /
  Excel 영역으로 확대)

---

## 14. Slack QA Bot (선택 기능)

Streamlit UI 외에 **Slack 채널에서 봇을 멘션해 질문을 던질 수 있는 사용자 인터페이스**
를 선택적으로 제공한다. Slack Bot 은 Streamlit 을 **대체하지 않으며**, 두 인터페이스의
역할은 명확히 분리된다.

| 인터페이스 | 역할 |
| --- | --- |
| **Streamlit UI** | 관리자 / 운영 콘솔 — 문서 업로드, 색인, 정규화 문서 관리, 검색 테스트, API 상태 확인 |
| **Slack Bot**    | 사용자 질문 인터페이스 — 채널에서 봇 멘션 → 기존 QA Pipeline 호출 → thread 답변 |

```
        Streamlit UI ─┐
                      ├→ 공통 src/rag/qa_pipeline.py → Retriever/Reranker → Local ChromaDB → Gemini → Answer
        Slack Bot   ──┘
```

Slack Bot 은 자체 RAG 검색/답변 로직을 가지지 않고, 항상 기존 ``QAPipeline`` 을 그대로
호출하는 얇은 adapter (``src/slack_bot/qa_adapter.py``) 로 동작한다. 따라서
ingestion / normalization / retrieval / reranking / qa 흐름은 Streamlit 과 100%
동일하다.

### 14.1 동작 흐름

1. Slack ``app_mention`` 이벤트 수신
2. 봇 mention (``<@U...>``) 제거
3. 질문 텍스트 정제 (양 끝 공백 / 콜론 정리, 길이 제한)
4. ``SLACK_ALLOWED_CHANNEL_IDS`` / ``SLACK_ALLOWED_USER_IDS`` 가 설정되어 있으면 검사
5. ``qa_adapter.answer_slack_question`` → 기존 ``QAPipeline.ask`` 호출
6. 답변 / 참고 근거 (Normalized Document, Raw Evidence) / 진단 정보를 Slack 메시지로 포맷
7. 원본 메시지 thread 에 답변 post (``SLACK_REPLY_IN_THREAD=true`` 기본값)

### 14.2 필요한 Slack App 설정

1. <https://api.slack.com/apps> 에서 새 앱 생성 (From scratch).
2. **Socket Mode** 활성화 (Settings → Socket Mode → On).
   - 별도의 public HTTP endpoint 가 필요 없으며, 사내 PC / 로컬 환경에서 그대로 테스트
     가능하다.
3. **App-Level Token** 생성 (Settings → Basic Information → App-Level Tokens).
   - Scope: ``connections:write``
   - 발급된 ``xapp-...`` 을 ``.env`` 의 ``SLACK_APP_TOKEN`` 에 입력.
4. **Bot Token Scopes** 부여 (Features → OAuth & Permissions → Scopes → Bot Token Scopes).
   - ``app_mentions:read``
   - ``chat:write``
5. **Event Subscriptions** 활성화 후 Bot Events 에 ``app_mention`` 추가.
6. Workspace 에 앱 설치 → 발급된 ``xoxb-...`` Bot User OAuth Token 을 ``.env`` 의
   ``SLACK_BOT_TOKEN`` 에 입력.
7. 봇을 사용할 채널에 봇을 초대 (``/invite @<봇이름>``).

### 14.3 환경변수 (.env.example 참고)

| 환경변수 | 용도 | 기본값 |
| --- | --- | --- |
| ``SLACK_BOT_ENABLED`` | 봇 활성화 여부 | ``false`` |
| ``SLACK_BOT_MODE`` | 동작 모드 (현재 ``socket`` 만 지원) | ``socket`` |
| ``SLACK_BOT_TOKEN`` | Bot User OAuth Token (``xoxb-...``) | (비어 있음) |
| ``SLACK_APP_TOKEN`` | App-Level Token (``xapp-...``, ``connections:write``) | (비어 있음) |
| ``SLACK_ALLOWED_CHANNEL_IDS`` | 응답 허용 채널 ID 콤마 구분 (비우면 전체 허용) | (비어 있음) |
| ``SLACK_ALLOWED_USER_IDS`` | 응답 허용 사용자 ID 콤마 구분 (비우면 전체 허용) | (비어 있음) |
| ``SLACK_REPLY_IN_THREAD`` | 항상 thread 로 답변할지 여부 | ``true`` |
| ``SLACK_MAX_QUESTION_CHARS`` | 한 번에 처리할 질문의 최대 글자수 (초과 시 잘라냄) | ``1000`` |

> **주의** — Slack token 은 절대 코드에 하드코딩하지 말고 ``.env`` (커밋 금지) 에서만
> 관리한다. ``.env`` 는 ``.gitignore`` 로 차단되어 있다.

### 14.4 실행 방법

```bash
# (.venv 활성화 상태에서)
pip install -r requirements.txt          # slack-bolt 가 함께 설치됨

# .env 에 SLACK_BOT_ENABLED=true 와 두 토큰을 채운 뒤
python scripts/run_slack_bot.py
```

- ``SLACK_BOT_ENABLED=false`` 면 안내 메시지를 출력하고 즉시 종료한다.
- 토큰이 누락된 경우 한국어 안내와 함께 exit code ``2`` 로 종료한다.
- Socket Mode 로 Slack 에 접속하므로 별도의 reverse proxy / ngrok / 외부 endpoint 가
  필요 없다.
- 종료는 Ctrl+C.

### 14.5 사용 예

```
사용자: @LF_HAZZYS_BOT 세팅 전에 확인해야 할 것 알려줘

봇 (thread):
*답변*
…

*참고 근거*
• Normalized Document
   - guide_setup.md · 사전 점검 (workflow)
     › 사전 점검 항목 1: …
• Raw Evidence
   - slack_thread_001.txt · 운영 공지
     › 운영팀: 세팅 전 …

*진단*
• answer_mode: `knowledge_card`
• primary_normalized_document_count: 1
• raw_evidence_count: 1
• raw_fallback_count: 0
• model: `gemini-2.5-flash-lite`
```

raw 원문은 그대로 노출되지 않는다. 모든 chunk preview 는 anonymizer / sanitizer 정책
(``ANONYMIZE_OUTPUT`` 등) 을 통과한 결과만 사용한다.

### 14.6 운영 주의사항

- **동일 token 으로 여러 PC 에서 동시에 봇 프로세스를 실행하지 않는다.** Slack 측에서
  중복 Socket Mode 연결로 처리되어 메시지 누락이 발생할 수 있다.
- **허용 채널 / 유저 화이트리스트** 를 설정해 운영 채널과 테스트 채널을 분리하는 것을
  권장한다.
- 봇은 항상 thread 로 답변하므로 채널 본문이 어지러워지지 않는다 (``SLACK_REPLY_IN_THREAD=
  true`` 기본).
- 내부 오류 시 traceback 을 Slack 에 노출하지 않고 한국어 안내만 보낸다 — 실제 traceback
  은 봇 프로세스의 로그에서만 확인 가능.
- ``data/raw``, ``data/processed``, ``storage`` 는 Slack Bot 이 직접 수정하지 않는다 —
  색인 / 정규화 / 캐시는 모두 Streamlit / CLI 흐름을 통해서만 갱신한다.

### 14.7 이번 MVP 에서 제외된 범위

다음은 의도적으로 제외했다 (필요 시 별도 작업으로 확장).

- Slack Thread 자동 수집 (스레드 전체를 봇이 직접 읽어 색인하지 않음)
- Slack 채널 history 조회
- Slack 파일 / 이미지 다운로드
- 자동 색인 (질문 들어올 때 봇이 새 문서를 ingest 하지 않음)
- DM 처리 (현재는 ``app_mention`` 이벤트만 처리)
- HTTP / Events API 모드 (현재는 Socket Mode 만 지원)

---

## 라이선스 / 책임 범위

- 본 저장소의 코드는 **연구 / 학습 / 사내 프로토타이핑** 용도로 작성되었다.
- 실제 운영 데이터에 적용할 때는 자체 보안 / 개인정보 / NDA 정책에 따라 비식별화 옵션과
  외부 호출 범위를 검토 후 사용해야 한다.
- 외부 API (Gemini) 사용에 따른 비용 / 한도 / 데이터 처리 정책은 사용자 책임 하에 관리한다.
