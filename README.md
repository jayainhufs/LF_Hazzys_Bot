# Work RAG Assistant

> 사내 업무 문서를 로컬에서 색인하고 한국어로 질의응답하는 **로컬 RAG (Retrieval Augmented
> Generation) 챗봇** 프로토타입. 외부 Vector DB 나 SaaS 를 사용하지 않고, 모든 원본 자료와
> 색인 결과는 로컬 디스크에 저장된다. 외부 호출은 답변/요약/임베딩 시점의 **Google Gemini
> API** 한 곳에서만 일어난다.

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

---

## 1. 프로젝트 개요

- 사용자가 자신의 업무 문서(가이드, 메신저 스레드 텍스트, 메일/카톡 내보내기, Excel 등)를
  로컬에 적재하면, 자연어 질문에 대해 단계별 절차 / 체크리스트 / 참고 근거를 한국어로
  답변하는 RAG 챗봇이다.
- 설계 원칙
  - **외부 Vector DB 미사용**: Pinecone / Weaviate / Qdrant Cloud / Supabase 등 모두 사용하지
    않는다. 색인 결과는 로컬 [ChromaDB](https://github.com/chroma-core/chroma) (`./storage/
    chroma_db`) 에 저장한다.
  - **자체 LLM 추론 미사용**: Ollama / llama.cpp 등을 사용하지 않는다. 답변과 (옵션) 임베딩만
    Google Gemini API 로 호출한다.
  - **외부 메신저 API 미사용**: Slack Bot, Slack API, 카카오톡 API 등에 직접 접근하지 않는다.
    필요한 대화는 사용자가 직접 텍스트 파일로 내보내 업로드한다.
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

> 외부 메신저 API 를 호출하지 않는다. 메신저 자료는 사용자가 직접 텍스트로 내보내 업로드해야
> 한다.

PDF / 이미지 OCR / Hybrid Retrieval / 자동 폴더 watcher 등은 현재 MVP 에서 지원하지 않는다.
([미지원 기능](#12-현재-미지원-기능) 참고)

---

## 4. LLM-based Document Normalization 아키텍처

raw 텍스트를 그대로 chunking / 임베딩하는 대신, LLM 으로 한 번 더 구조화해
**`NormalizedDocument`** 형태의 업무 지식 단위로 정리한다. 검색과 답변은 Normalized
Document 단위를 1차 근거로 사용하고, raw chunk 는 보조 근거 또는 fallback 으로만 활용한다.

```
[raw 텍스트 chunk]                    ←  기존 RAG 색인 흐름은 그대로 유지
        ↓
[(옵션) LLM-based Document Normalization]
  ├─ Guide       → workflow / checklist / faq / glossary / communication_template / decision
  └─ Slack-style → issue / decision / checklist / communication_template / faq / workflow
        ↓
[NormalizationStore  ./data/processed/normalized/{json,markdown}]
  └─ file_hash + prompt_version + model_name 기반 cache 로 중복 호출 방지
        ↓
[ChromaDB 에 추가 색인 — content_type=normalized_document / source_type=llm_normalized]
        ↓
[Retriever / Reranker 가 Normalized Document 우선 boost → retrieval_role 라벨링]
        ↓
[QA Prompt Builder — Normalized Document 중심 답변 형식 (default / communication_template / glossary)]
```

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

> **Backward compatibility**: 기존에 색인된 데이터는 `content_type="knowledge_card"` /
> `card_id` / `card_type` 으로 저장되어 있을 수 있다. retriever / reranker / qa pipeline
> 은 신규 (`normalized_document` / `normalized_document_id` / `normalized_document_type`)
> 와 legacy 키를 모두 인식한다. 새로 색인되는 chunk 는 신규 표준을 사용한다.

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
  (라벨 문자열 자체는 호환을 위해 유지된다.)
- `answer_mode` ∈ {`knowledge_card`, `raw_fallback`, `insufficient_evidence`}
  (라벨 문자열 자체는 호환을 위해 유지된다.)
- `normalized_document_id` / `normalized_document_type` (legacy: `card_id` / `card_type`)
- `primary_topic` / `task_type`
- `normalized_document_boost`, `normalized_document_type_boost`,
  `normalized_document_type_match`
- `parent_raw_chunk_ids`, `final_score`

이 필드들은 Streamlit 의 검색 테스트 / QA 페이지에서 그대로 확인할 수 있다. 기존
`knowledge_card_boost` / `card_type_boost` / `card_type_match` 등 legacy 키도 동일한 값으로
함께 채워진다.

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

| 페이지 | 역할 |
| --- | --- |
| `1_문서_업로드` | 카테고리 선택 후 파일 업로드 → `data/raw/<카테고리>/` 에 저장 |
| `2_문서_색인` | 새 파일 색인 / Excel 한국어 요약 옵션 / LLM-based Document Normalization 결과 카운트 표시 |
| `3_업무_QA` | 질문 → 검색 → 답변. `answer_mode`, primary Normalized Document 목록, retrieval_role 진단 표시 |
| `4_검색_테스트` | 답변 생성 없이 retrieval 결과만 확인. content_type / normalized_document_type 필터 제공 |
| `5_API_상태확인` | Gemini API Key / 모델 사용 가능 여부 점검 |
| `6_Excel_요약관리` | Excel 시트별 한국어 업무 요약 재생성 / 캐시 확인 |
| `7_지식카드_관리` | (UI 표시명: "정규화 문서 관리") 정규화된 Normalized Document JSON / Markdown 을 read-only 로 확인 |

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
  - `ANSWER_WITH_NORMALIZED_DOCUMENTS=true` (legacy: `ANSWER_WITH_KNOWLEDGE_CARDS=true`)
    이고 통과 chunk 안에 primary Normalized Document 가 있으면 Normalized Document 중심
    prompt 로 답변한다 (`answer_mode=knowledge_card` 라벨은 호환을 위해 유지).
  - 그 외에는 기존 raw chunk prompt 로 동작한다 (`answer_mode=raw_fallback`).
  - Normalized Document type 에 따라 답변 형식이 default / communication_template /
    glossary 중 하나로 분기된다.
  - 답변과 참고 근거 모두에서 사람 실명 / @멘션 / 정확한 시간 / 원본 날짜를 노출하지 않도록
    prompt 와 비식별화 가드가 작동한다.

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
- 카카오 / 메일 / Excel 영역으로 LLM-based Document Normalization 확장

---

## 라이선스 / 책임 범위

- 본 저장소의 코드는 **연구 / 학습 / 사내 프로토타이핑** 용도로 작성되었다.
- 실제 운영 데이터에 적용할 때는 자체 보안 / 개인정보 / NDA 정책에 따라 비식별화 옵션과
  외부 호출 범위를 검토 후 사용해야 한다.
- 외부 API (Gemini) 사용에 따른 비용 / 한도 / 데이터 처리 정책은 사용자 책임 하에 관리한다.
