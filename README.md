# Work RAG Assistant

> 광고대행사 퍼포먼스마케팅 인턴 업무 지원용 **로컬 RAG 챗봇** 프로토타입.
> Slack 복붙 자료 / 가이드 문서 / 카카오톡 대화 / Excel 가이드를 적재하고,
> Google Gemini API로 한국어 답변을 생성한다.

---

## 1. 프로젝트 소개

업무를 익히는 단계의 인턴이 **회사 내부 자료**를 바탕으로 자연어로 질문하고,
실제 업무 처리 순서, 단계별 설명, 주의사항, 체크리스트, 근거까지 함께 답변받기 위한
**Modular RAG** 시스템이다.

핵심 특징

- 외부 Vector DB / 외부 SaaS / 로컬 LLM 사용하지 않음
- 모든 원본 파일과 색인 결과는 **로컬 디스크**에 저장
- 외부로 나가는 호출은 **Google Gemini API** 뿐
- 임베딩은 **Gemini Embedding** / **로컬 sentence-transformers** 두 가지 중 선택 가능
- Excel은 단순 셀 임베딩이 아니라 **Gemini 한국어 업무 요약** + raw 표 텍스트의 이중 구조

---

## 2. 시스템 아키텍처

```
[사용자 파일]
  ├─ Slack 복붙 TXT/MD/DOCX
  ├─ 업무 가이드 TXT/MD/DOCX
  ├─ 카카오톡 TXT
  └─ Excel XLSX/XLSM
        ↓
[로컬 파일 저장소: ./data/raw]
        ↓
[Parser Layer]
  ├─ Slack Manual Parser
  ├─ Guide Parser (word / md / txt)
  ├─ Kakao Parser
  └─ Excel Parser
        ↓
[Cleaning & Normalization]
        ↓
[Excel Semantic Summary (선택 실행)]
  ├─ Gemini 한국어 상세 요약
  └─ summary cache (file_hash 기준)
        ↓
[Chunking + Metadata]
        ↓
[Embedding (Gemini | Local)]
        ↓
[Local ChromaDB ./storage/chroma_db]
        ↓
[Retrieval + Reranking + Parent-Child]
        ↓
[Prompt Builder]
        ↓
[Gemini Generation API]
        ↓
[근거 기반 한국어 답변]
```

---

## 3. 보안 구조

### 3.1 어디에 무엇이 저장되는가

| 항목 | 위치 |
| --- | --- |
| 원본 파일 | `./data/raw/<카테고리>/` (로컬) |
| 처리된 chunk / document JSON | `./data/processed/` (로컬) |
| Excel 한국어 요약 | `./data/processed/summaries/excel/` (로컬) |
| Vector DB | `./storage/chroma_db/` (로컬, ChromaDB) |
| 질문/답변 로그 | `./storage/qa_logs/` (로컬) |
| 외부 Vector DB / SaaS | **사용 안함** (Pinecone, Weaviate, Supabase, Qdrant Cloud 등) |
| 자체 LLM 추론 | **사용 안함** (Ollama, llama.cpp 등 미사용) |
| Slack API / Slack Bot | **사용 안함** (대화는 직접 복사한 텍스트 파일로 적재) |

### 3.2 Google API 로 전송되는 데이터 범위

| 시점 | 전송되는 텍스트 | 비고 |
| --- | --- | --- |
| 색인 시 (`EMBEDDING_PROVIDER=gemini`) | 각 chunk 의 `embedding_text` (보통 1500자 이내) | 임베딩 생성 목적 |
| 색인 시 (`EMBEDDING_PROVIDER=local`) | 전송 없음 | sentence-transformers 로 로컬 임베딩 |
| Excel 상세 요약 ON 시 | 시트별 raw_table_text (cap 8000자) | 한국어 업무 요약 생성 |
| 업무 QA 답변 시 | 사용자 질문 + 검색된 chunk 컨텍스트 | 답변 생성 |
| Query rewrite ON 시 (기본 OFF) | 사용자 질문만 | 검색 친화적 재작성 |
| 4_검색_테스트 페이지 | 질문 임베딩 1회만 (Gemini provider 한정) | Generation 호출 없음 |
| 5_API_상태확인 페이지 | 사용자가 버튼을 누른 경우에만 ping | 페이지 조회만으로는 무전송 |
| 6_Excel_요약관리 페이지 | 사용자가 (재)생성 버튼을 눌렀을 때만 | 캐시되어 있으면 재호출 안 함 |

> 회사 보안 정책상 외부로 보내면 안되는 텍스트는 **업로드 전에** 본인이 직접 마스킹/제거할 것.
> 보안이 가장 엄격한 환경에서는 `EMBEDDING_PROVIDER=local` 로 두고 Generation API 만 한정적으로 사용하는 것을 권장한다.

---

## 4. Mac 개발 환경 세팅

```bash
# 1) 가상환경 (Python 3.11 권장)
python3.11 -m venv .venv
source .venv/bin/activate

# 2) 의존성
pip install --upgrade pip
pip install -r requirements.txt

# 3) 환경변수
cp .env.example .env
# .env 를 열어서 GOOGLE_API_KEY 입력

# 4) (선택) smoke test : 외부 API 호출 없이 핵심 동작 확인
python scripts/smoke_test.py

# 5) 실행
streamlit run app/main.py
# 또는
bash scripts/run_app.sh
```

> 모든 명령은 **반드시 가상환경(`.venv`) 안**에서 실행하세요.
> 시스템 파이썬에서 직접 실행하면 numpy / chromadb 충돌 등 환경 문제가 발생할 수 있습니다.

---

## 5. Windows 회사 PC 실행 방법

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

:: 4) (선택) smoke test : 외부 API 호출 없이 핵심 동작 확인
python scripts\smoke_test.py

:: 5) 실행
streamlit run app/main.py
:: 또는
scripts\run_app.bat
```

### Windows 추가 주의사항

- 멀티페이지 파일명에 한글이 포함되어 있습니다 (`app/pages/1_문서_업로드.py` 등).
  Streamlit 1.10+ 와 Python 3.11 은 한글 파일명을 정상 지원하지만,
  **콘솔 출력이 깨져 보인다면** PowerShell 사용 또는 `chcp 65001` 로 UTF-8 콘솔로 전환하세요.
  ```bat
  chcp 65001
  .venv\Scripts\activate
  streamlit run app/main.py
  ```
- `python` 명령이 동작하지 않으면 `py -3.11` 로 대체할 수 있습니다.
- 회사 PC 의 보안 정책으로 PyPI 접근이 차단된 경우, 사내 PyPI 미러 또는 오프라인 wheel 패키지 사용을 검토하세요.
- sentence-transformers / chromadb 첫 설치 시 PyTorch 다운로드로 5~10분 소요될 수 있습니다.
  회사 노트북 사양이 낮다면 `EMBEDDING_PROVIDER=gemini` 권장.

---

## 6. GitHub 업로드 후 Windows에서 git clone하여 실행하기

Mac에서:

```bash
git init
git add .
git commit -m "init: work-rag-assistant MVP"
git remote add origin <GitHub repo URL>
git push -u origin main
```

회사 Windows PC에서:

```bat
git clone <GitHub repo URL>
cd work-rag-assistant
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
:: .env 수정
streamlit run app/main.py
```

> `.gitignore` 에 의해 `data/`, `storage/`, `.env` 는 push 되지 않는다.
> 따라서 회사 PC에서는 데이터를 새로 적재하면 된다.

---

## 7. Google API Key 설정

1. https://aistudio.google.com/app/apikey 에서 API Key 발급
2. `.env` 파일의 `GOOGLE_API_KEY=` 뒤에 붙여넣기
3. Streamlit 앱의 **5_API_상태확인** 페이지에서 연결 테스트

> Key가 노출되면 즉시 Google AI Studio에서 회수(Revoke)한다.

---

## 8. 임베딩 전략

문서가 한국어 90% / 영어 10% 구성이라 임베딩 모델 선택이 검색 품질에 큰 영향을 준다.

| Provider | 모델 | 장점 | 단점 |
| --- | --- | --- | --- |
| `gemini` (기본) | `gemini-embedding-001` | 노트북 사양 부담 거의 없음, 빠름 | 텍스트가 Google API로 전송됨 |
| `local` | `BAAI/bge-m3` 등 | 외부 전송 없음, 한국어 검색 품질 우수 후보 다수 | 회사 노트북에서 느릴 수 있음 |

`.env`의 `EMBEDDING_PROVIDER` 값만 바꾸면 동일 인터페이스로 교체된다.

향후 교체 후보 (모두 `LOCAL_EMBEDDING_MODEL` 한 줄로 교체 가능):

- `BAAI/bge-m3`
- `intfloat/multilingual-e5-large`
- `nlpai-lab/KURE-v1`
- `jhgan/ko-sroberta-multitask`
- `dragonkue/bge-m3-ko`

`pages/4_검색_테스트.py` 에서 동일 질문에 대해 모델별 retrieval 결과를 비교할 수 있다.

---

## 9. Excel 처리 전략

Excel은 절대 단순 셀 텍스트만 임베딩하지 않는다. 다음 두 산출물을 만든다.

1. **raw_table_text**
   - 시트 단위로 셀 값을 보존한 텍스트 (숫자 / 컬럼명 / 행 정보 유지)
   - 근거 확인용
2. **semantic_korean_summary**
   - Gemini로 생성한 **한국어 업무 설명문** (Markdown)
   - 컬럼 의미, 업무 맥락, 사용 시점, 주의사항, 검색 키워드 포함
   - **검색의 1차 대상**

추가로 `parent-child retrieval`을 적용한다.

- parent: Excel 시트 한국어 요약 chunk
- child: 같은 sheet의 raw_table_text chunk

질문 시 parent가 검색되면 답변 단계에서 child 일부를 함께 컨텍스트에 넣어 숫자 근거를 확보한다.

---

## 10. 토큰/비용 최적화 전략 (월 ~$20 운영 가정)

- `file_hash` / `chunk_hash` / `excel_summary_hash` 기반 **중복 방지**
- Excel 상세 요약은 **선택 실행** (`ENABLE_EXCEL_SUMMARY=true` 또는 색인 페이지 옵션)
- 검색 테스트 페이지는 **generation 호출 안함**
- `TOP_K`, `MAX_CONTEXT_CHARS`, `MAX_CHUNKS_PER_FILE` 로 컨텍스트 길이 통제
- query rewrite 기본 비활성화
- LLM reranker 사용 안함 (rule-based reranking)
- API 호출 횟수 / chunk 수 / cache hit 여부를 로깅

---

## 11. 데이터 폴더 구조

```
data/
  raw/                # 원본 업로드 파일
    slack_manual/
    guide/
    kakao/
    excel/
    misc/
  processed/
    documents/        # document metadata JSON
    chunks/           # chunk JSONL
    summaries/
      excel/          # Excel 한국어 요약 .md / .json
  sample/             # 형식 예시 안내

storage/
  chroma_db/          # ChromaDB persist 디렉터리
  registry/
    indexed_files.json
  qa_logs/            # 질문/답변 JSON 로그
```

---

## 12. 4가지 업로드 카테고리

| 카테고리 | 용도 | 권장 확장자 | source_weight |
| --- | --- | --- | --- |
| **Slack 대화** | Slack 채널/스레드를 직접 복사한 자료 | txt / md / docx | 0.85 |
| **가이드** | 회사 업무 가이드 문서 | docx / md / txt | 0.9 |
| **카톡 대화** | 카카오톡 대화 내보내기 TXT | txt | 0.5 |
| **Excel** | Excel 가이드 (raw + semantic summary) | xlsx / xlsm | 1.0 (raw) / 1.1 (summary) |
| 기타 | 분류 안되는 자료 | * | 0.5 |

> Slack API / Slack Bot 은 회사 보안 정책상 사용하지 않는다.
> 필요한 대화를 **본인이 직접 복사**해 텍스트 파일로 저장한 뒤 업로드한다.

---

## 13. 파일 업로드 방법

1. `streamlit run app/main.py`
2. **1_문서_업로드** 페이지 진입
3. 카테고리 선택 후 파일 드래그앤드롭
4. 자동으로 `data/raw/<카테고리>/` 에 저장됨

---

## 14. 색인 실행 방법

- Streamlit: **2_문서_색인** 페이지에서 "새 파일만 색인" 버튼
- CLI: `python scripts/ingest_folder.py` (선택: `--enable-summary`, `--no-skip`)

이미 색인된 파일은 `file_hash` 로 자동 skip 됩니다.

### 14.1 재색인 / 초기화 흐름

| 상황 | 명령 |
| --- | --- |
| 같은 파일을 다시 색인 (hash 동일) | 자동 skip. 강제 재색인하려면 `python scripts/ingest_folder.py --no-skip` |
| 파일 내용을 수정한 경우 | `file_hash` 가 바뀌므로 자연스럽게 재색인 |
| Vector DB 만 비우기 (원본/요약 유지) | `python scripts/reset_vector_db.py --keep-processed --keep-registry` |
| 전부 초기화 (원본은 유지, chunk/요약/registry/Vector DB 모두 비움) | `python scripts/reset_vector_db.py` |
| 원본까지 지우기 | 직접 `data/raw/<카테고리>/` 안의 파일 삭제 후 위 명령 |

> `reset_vector_db.py` 는 `data/raw` 는 절대 건드리지 않습니다.
> 색인 결과(processed/chunks, summaries, ChromaDB, registry) 만 정리합니다.

---

## 15. Excel summary 생성 방법

- Streamlit: **2_문서_색인** 페이지에서 “Excel 상세 요약 생성” 체크박스 ON
- 또는 **6_Excel_요약관리** 페이지에서 파일 단위로 재생성
- CLI: `python scripts/summarize_excel_folder.py`

> 비용 발생. `raw_table_hash`가 동일하면 자동 재사용.

---

## 16. Streamlit 실행 방법

```bash
# Mac
bash scripts/run_app.sh
# Windows
scripts\run_app.bat
```

기본 포트는 `http://localhost:8501`.

---

## 17. 검색 테스트 방법

**4_검색_테스트** 페이지에서 질문을 입력하면 LLM 호출 없이 retrieval 결과만 확인할 수 있다.
임베딩 모델을 바꿔가며 동일 질문의 검색 품질을 비교하기 좋다.

페이지에서 다음 진단 정보를 확인할 수 있다.

- candidate / passed / dropped 카운트
- 각 chunk 의 `score`, `final_score`, `source_weight`, `category_boost`, `content_type_boost`
- `passed_threshold` 통과 여부 및 `filter_reason` (similarity 미달 / final_score 미달 / 파일 cap 초과)
- score 해석: `score = 1 - cosine_distance` (값이 높을수록 유사)
- 슬라이더로 즉석에서 `MIN_SIMILARITY_SCORE`, `MIN_FINAL_SCORE`, `MAX_CHUNKS_PER_FILE`, `USE_MMR` 조정

---

## 17.1 검색 결과가 너무 넓게 나올 때 조정하는 방법

검색 테스트에서 관련 단어 하나만 일치해도 chunk 가 줄줄이 딸려 오는 경우, 아래 환경변수로 조정할 수 있다.
모든 값은 `.env` 또는 검색 테스트 페이지의 슬라이더에서 즉시 변경 가능하다.

| 환경변수 | 효과 |
| --- | --- |
| `TOP_K` | 가져올 최종 근거 수. 낮추면 답변에 들어가는 근거가 줄어든다. |
| `MIN_SIMILARITY_SCORE` | raw similarity (`1 - cosine_distance`) 임계값. 높이면 약한 관련 chunk 가 제거된다. |
| `MIN_FINAL_SCORE` | source_weight / category_boost / content_type_boost 까지 반영한 최종 점수 임계값. 높이면 근거 품질이 올라가지만 결과가 0이 될 수 있다. |
| `MIN_RETRIEVED_CHUNKS` | 통과 chunk 가 이 값 미만이면 Gemini Generation 을 호출하지 않고 "근거 부족" 안내를 반환한다. |
| `MAX_CHUNKS_PER_FILE` | 같은 파일에서 너무 많은 chunk 가 들어오는 것을 막는다. 1~2 로 두면 한 문서가 결과를 독점하지 않는다. |
| `USE_MMR` | `true` 면 비슷한 chunk 반복을 줄이는 다양성 보정을 적용한다. |
| `MMR_LAMBDA` | 관련성(λ)과 다양성(1-λ) 사이의 비율. 0.7 이면 관련성 70% / 다양성 30%. |

추가 정책:

- **Slack 대화** 는 히스토리/실무 맥락 참고용으로 가중치를 낮게(`source_weight=0.8`) 둔다.
- **Guide 문서** 는 공식 절차 근거로 가중치를 높게(`source_weight=1.0`) 둔다.
- "정산", "세금계산서", "인보이스", "SF", "모비사인", "입금", "광고주 공유용" 같은 키워드가 질문에 들어 있으면
  Guide 자료가 우선 검색되도록 카테고리 부스트가 자동 보정된다.
- "오늘", "어제", "TODO", "보미님", "스레드", "말씀", "피드백" 같은 키워드가 질문에 들어 있으면
  Slack thread 자료의 부스트가 자동으로 강화된다.
- "ASC", "BAU", "메타", "캠페인 세팅", "컨첵시트", "토글" 같은 키워드는 Slack + Guide 둘 다 우대한다.
- "카카오톡 / 카카오메시지 / 발송" 키워드는 Slack 을 강하게, Guide / Kakao 자료도 보조로 우대한다.
- 카카오톡 자료는 잡음이 많아 기본 가중치가 낮다 (`source_weight=0.45`).
- **근거가 부족하면 Gemini Generation 을 호출하지 않으므로 비용을 절감**할 수 있다.
  (3_업무_QA 페이지에서 "근거 부족" 안내 메시지가 답변 자리에 표시된다.)

---

## 17.2 Slack TODO 검색 품질과 비식별화 출력

Slack 복붙 자료(`uploaded_category=slack`)는 그대로 임베딩하면 다음과 같은 문제가 발생한다.

- 하루치 대화가 통째로 한 chunk(`fallback_block`) 가 되어 BAU/ASC, 카카오 발송, DR 확인,
  메타 세팅, TODO 가 한 덩어리로 섞이고, 다른 날짜 질문에도 잘못 끌려 나온다.
- 사람 이름, @멘션, "오전 10:02" 같은 정확한 시간이 검색 결과/QA 답변에 그대로 노출된다.

이를 줄이기 위해 다음 기능을 제공한다.

### Slack parser v2 (구조화)

다음 패턴을 인식해 발화자 단위 message 와 "TODO 섹션" 단위로 잘라서 적재한다.

- `이동제[마케팅4팀]  [오전 10:02]` 형태의 헤더 + 다음 줄 본문
- `[오전 10:57] @김보미[마케팅4팀] ...` 같은 시간+멘션 한 줄 메시지
- `[2026년 4월 29일 TODO]` 제목, `4/29 동제 오전/중간/퇴근! TODO`, `동제 명일 TODO`,
  `내일 동제쓰 TODO` 같은 섹션 헤더
- `셋팅 내용`, `크첵 해주실 내용`, `1. 4월 ASC 캠페인 ... 원인 분석` 같은 sub-section

각 섹션의 metadata 에는 다음이 채워진다.

- `document_date` (예: `2026-04-29`), `date_text`, `date_source`
- `todo_phase` (`initial` / `morning` / `mid` / `end_of_day` / `tomorrow` / `unknown`)
- `topic_tags` (예: `["meta", "kakao"]`), `primary_topic`
- `original_speakers`, `speaker_roles`, `display_speakers` (이름 → "작성자/검토자/담당자 A")
- `time_buckets`, `time_range_display` (예: `"오전 · 오후"`)
- `parser_format` (`structured_slack_messages` / `slack_todo_sections` / `fallback_block`)
- `sanitized_content` — UI / QA 답변에 표시할 비식별화 본문

### 날짜 / 주제 인식 검색 (date / topic-aware retrieval)

질문에서 다음을 자동 추출한다.

- `query_date` — `4월 29일`, `4/29`, `2026-04-29` 형태 모두 인식
- `query_topics` — `메타`, `카카오`, `정산`, `옥외`, `리포트`, `NBT`, `그린피`, `유튜브` 등
- `query_intent` — `procedure` / `todo_lookup` / `explanation` / `issue_lookup`

이를 기반으로 final_score 에 다음이 곱해진다.

| 환경변수 | 효과 |
| --- | --- |
| `DATE_EXACT_MATCH_BOOST` (1.25) | 질문 날짜 == chunk.document_date 일 때 강한 가중치 |
| `DATE_MISMATCH_PENALTY` (0.55) | 질문 날짜와 다른 날짜 chunk 에 강한 페널티 |
| `ENABLE_DATE_FILTER` (false) | true 면 다른 날짜 chunk 자체를 검색 결과에서 제거 |
| `TOPIC_MATCH_BOOST` (1.20) | query_topics 와 chunk.topic_tags 가 1개라도 겹치면 가중치 |
| `TOPIC_MISMATCH_PENALTY` (0.80) | 둘 다 있는데 안 겹치면 페널티 (guide+procedure 조합은 약하게만) |

### 비식별화 출력 (anonymization)

UI 검색 결과 / QA 답변 / prompt context 에서는 기본적으로 `sanitized_content` 를 사용한다.
원본 raw 문서는 `data/raw/...` 에 그대로 남아있고 변경되지 않는다.

| 환경변수 | 기본값 | 설명 |
| --- | --- | --- |
| `ANONYMIZE_OUTPUT` | `true` | UI/QA 답변/prompt 에 sanitized 본문을 우선 사용 |
| `SHOW_RAW_CONTENT` | `false` | true 면 원문도 expander 로 보여줌(디버깅용) |
| `SHOW_SPEAKER_NAMES` | `false` | true 면 "이동제[마케팅4팀]" 같은 실명 그대로 노출 |
| `SHOW_EXACT_TIMESTAMPS` | `false` | true 면 "오전 10:02" 같은 정확한 시간 그대로 노출 |
| `SHOW_EXACT_DATES` | `false` | true 면 "2026년 4월 29일" 같은 원본 날짜 그대로 노출 |
| `MASK_MENTIONS` | `true` | "@김보미[마케팅4팀]" → "@담당자" 치환 |
| `MASK_LINKS` | `true` | URL/드라이브 링크 → "[링크]" / "[이미지]" / "[파일]" |
| `MASK_FILE_NAMES` | `false` | (확장 여지) 파일명에 실명이 들어 있을 때 활성화 |
| `ANONYMIZED_DATE_LABEL` | `업무일` | "해당 업무일" 식 라벨 |
| `ANONYMIZED_TIME_LABEL` | `시간대` | "[시간대]" 식 라벨 |

### 권장 운영 셋팅

- 파일명에 날짜를 넣어 적재하면 검색 품질이 올라간다.  
  예: `[2026년 4월 29일 TODO].txt`, `[2026년 4월 30일 TODO].txt`
- TODO 섹션 제목(`4/29 오전 TODO`, `4/29 중간 TODO`, `4/29 퇴근 TODO`, `명일 TODO`)을 유지하면
  parser v2 가 phase 단위로 chunk 를 나눠 노이즈가 줄어든다.
- 결과가 다른 날짜 문서까지 끌고 오면 검색 테스트 페이지에서 `ENABLE_DATE_FILTER`
  체크박스로 즉시 검증할 수 있다.
- 팀 내 공유용으로 사용할 때는 다음 셋팅을 권장한다.

```env
ANONYMIZE_OUTPUT=true
SHOW_RAW_CONTENT=false
SHOW_SPEAKER_NAMES=false
SHOW_EXACT_TIMESTAMPS=false
SHOW_EXACT_DATES=false
MASK_MENTIONS=true
MASK_LINKS=true
```

---

## 17.3 LLM 기반 업무 지식카드 정규화 로드맵

LLM 기반 업무 지식카드 정규화는 raw Slack/Guide/Excel/메일/카카오 자료를 바로 chunking 하기 전에,
업무 절차/이슈/체크리스트/FAQ 같은 구조화된 `KnowledgeCard` 로 정리해 검색과 QA 품질을 높이기 위한 단계적 기능이다.

### Task 1 (완료) — 기반 구조

- `KnowledgeCard` schema 를 추가해 workflow / issue / checklist / faq / decision / glossary /
  communication_template 형태의 정규화 결과를 담을 수 있게 했다.
- `NormalizationStore` 를 추가해 정규화 결과 JSON/Markdown 을
  `data/processed/normalized` 아래에 저장할 수 있게 했다.
- 같은 raw 파일은 `file_hash + prompt_version + model_name` 기반 cache key 로 중복 호출을 방지한다.
- 원본 raw 파일은 수정하지 않는다.

### Task 2 (완료) — Guide normalizer + LLM 호출 + cache

- Guide txt 문서를 `KnowledgeCard` 로 변환하는 `GuideKnowledgeNormalizer` 를 추가했다.
  prompt 는 `src/normalization/normalization_prompt.py` 의
  `build_guide_normalization_prompt()` 로 구성하며, prompt 버전은
  `GUIDE_NORMALIZER_PROMPT_VERSION = "guide_v1"` 로 고정했다.
- LLM 호출은 기존 `GeminiClient.generate_text(system_instruction=..., temperature=...)`
  를 그대로 재사용한다. 응답이 ```json 코드블록으로 감싸진 경우도 안전하게 파싱한다.
- 결과는 `NormalizationStore` 를 통해
  `data/processed/normalized/json/<file_hash_short>.json` 과
  `data/processed/normalized/markdown/<file_hash_short>.md` 로 저장된다.
- `file_hash + prompt_version + model_name` cache key 로 중복 호출을 방지한다.
  cache hit 이면 LLM 호출 없이 저장된 JSON 에서 KnowledgeCard 를 복원한다.
- 카드 수는 `NORMALIZATION_MAX_CARDS_PER_FILE` 로 상한이 적용된다.
- 외부 API 없이도 검증 가능하도록 `tests/test_guide_normalizer.py` 에 mock 기반
  단위 테스트(파싱, cache hit/miss, 한글 보존, 카드 수 제한 등)를 추가했다.
- 실제 pipeline 연결은 Task 4 에서, Streamlit 지식카드 관리 UI 는 Task 5 에서 진행한다.

### Task 3 (완료) — Slack Thread normalizer + LLM 호출 + cache

- Slack Thread txt 문서를 `KnowledgeCard` 로 변환하는 `SlackThreadKnowledgeNormalizer` 를 추가했다.
  prompt 는 `src/normalization/normalization_prompt.py` 의
  `build_slack_normalization_prompt()` 로 구성하며, prompt 버전은
  `SLACK_NORMALIZER_PROMPT_VERSION = "slack_thread_v1"` 로 고정했다.
- Slack Thread 는 공식 가이드가 아니라 실무 진행 로그 / 이슈 / 결정 / 다음 액션의
  근거로 다룬다. 따라서 카드 타입은 `issue` / `decision` / `checklist` /
  `communication_template` / `faq` 가 우선이고, `workflow` 는 반복 가능한 절차가
  명확히 도출됐을 때만 사용한다.
- Slack parser v2 가 추출한 `topic_tags` / `todo_phase` / `parser_format` 을
  user prompt 에 명시해 카드 분리·토픽 추론을 돕고, 같은 메타는 결과
  `KnowledgeCard.metadata` 에도 보존한다 (Task 6~7 의 retrieval / QA 단계에서 사용).
- 익명화 가드: 사람 실명 / @멘션 / 정확한 시간을 출력하지 않고, 날짜는
  "해당 업무일 / 다음 업무일 / 전일 / 월초 / 월말" 같은 업무 맥락 표현으로 바꾼다.
  링크·이미지·파일명은 `[링크]` / `[이미지]` / `[파일]` 로 치환한다.
- LLM 호출 / JSON 파싱(```json ``` fence 제거 포함) / cache 흐름은
  Guide normalizer 의 helper 를 그대로 재사용해 코드 중복을 피했다.
- 결과 저장 경로와 cache key 정책은 Guide 와 동일 (`file_hash + prompt_version
  + model_name` cache 로 중복 호출을 방지). cache value 에는 `source_kind`
  필드로 `slack_thread` 를 함께 기록한다.
- 외부 API 없이도 검증 가능하도록 `tests/test_slack_normalizer.py` 에 mock 기반
  단위 테스트(파싱, cache hit/miss, 한글 보존, 카드 수 제한, prompt 익명화 가드,
  topic_tags 병합 등)를 추가했다.
- 실제 pipeline 연결은 Task 4 에서, Streamlit 지식카드 관리 UI 는 Task 5 에서 진행한다.

### Task 4 (완료) — pipeline 에 `ENABLE_LLM_NORMALIZATION` 옵션 연결

- `src/normalization/pipeline_integration.py` 를 추가해 ingest pipeline 분기를
  helper 함수로 분리했다 (`should_normalize_file`, `extract_normalization_inputs`,
  `attach_parent_raw_chunk_ids`, `knowledge_cards_to_chunks`,
  `normalize_document_for_pipeline`, `run_normalization_branch`).
- `src/pipeline.py` 의 `ingest_file` / `ingest_folder` 가 `enable_llm_normalization`,
  `gemini_client`, `normalization_store` 옵션을 받게 했다.
  - `ENABLE_LLM_NORMALIZATION=false`(기본) 일 때는 어떤 normalizer 도 호출되지 않으며
    기존 raw ingest 흐름과 완전히 동일하게 동작한다.
  - `true` 일 때는 raw chunk 저장 직후 LLM normalization branch 가 실행되어
    `KnowledgeCard` 들을 만들고 ChromaDB 에 별도 chunk(`content_type=
    "knowledge_card"`, `source_type="llm_normalized"`) 로 추가 저장한다.
- 분기 정책 (Task 4 범위):
  - `uploaded_category=="guide"` 또는 `source_type=="guide"` →
    `GuideKnowledgeNormalizer`
  - `uploaded_category in {"slack","slack_manual"}` 또는
    `source_type=="slack_manual"` → `SlackThreadKnowledgeNormalizer`
  - 카카오 / 메일 / Excel / Word 등은 normalization skip (이후 Task 에서 확장)
- 안전성 가드:
  - `run_normalization_branch` 는 raise 하지 않는다. LLM 호출 / embedding /
    vector store / file 저장 어떤 단계가 실패해도 `log.warning` 만 남기고
    raw indexing 흐름은 계속 진행된다.
  - `IngestResult` 에 `normalized_kind` / `normalized_card_count` /
    `normalized_chunks_added` / `normalized_skipped_reason` 필드를 추가해
    호출 측이 결과를 그대로 활용할 수 있게 했다.
- KnowledgeCard → Chunk 변환 (`knowledge_cards_to_chunks`):
  - content 는 `card.sanitized_markdown` 우선, 없으면 `card.to_markdown()`.
  - metadata 에 `card_id`, `card_type`, `primary_topic`, `topic_tags`,
    `task_type`, `document_date`, `display_date`, `source_file_hash`,
    `parent_raw_chunk_ids`, `source_weight` (`NORMALIZATION_CARD_SOURCE_WEIGHT`),
    `normalized=True`, `prompt_version`, `model_name`, `todo_phase`,
    `parser_format` 을 함께 기록한다 (Task 6 의 retrieval 우선순위 적용 시 사용).
- 같은 raw 파일은 `file_hash + prompt_version + model_name` cache 로 LLM 재호출이
  방지되며, JSON / Markdown 결과는 `NormalizationStore` 가 관리한다.
- 외부 Gemini API 호출 없이 검증 가능하도록 `tests/test_pipeline_normalization.py`
  에 mock 기반 단위 테스트 (분기, 입력 추출, parent link, chunk 변환, branch
  happy path, 실패 fallback 등) 30 건을 추가했다.
- Streamlit 지식카드 관리 UI 노출은 Task 5, 검색 우선순위 적용은 Task 6,
  QA 답변 반영은 Task 7 에서 진행한다.

### Task 5 (완료) — Streamlit 지식카드 관리 UI

- `app/pages/7_지식카드_관리.py` 를 추가해 생성된 `KnowledgeCard` JSON/Markdown 을
  Streamlit 에서 read-only 로 확인할 수 있게 했다.
- `src/normalization/card_viewer.py` 를 추가해 UI helper 를 분리했다.
  - `list_normalized_json_files` / `list_normalized_markdown_files`
  - `load_all_cards_from_store`
  - `summarize_cards`
  - `filter_cards`
  - `card_to_display_dict`
  - `markdown_for_card`
- 페이지 상단에서 JSON 파일 수, Markdown 파일 수, 전체 카드 수, cache index 상태,
  normalization output directory / json_dir / markdown_dir / cache_index 위치를 확인할 수 있다.
- card_type / primary_topic / source_file_name / text query 로 카드 목록을 필터링할 수 있다.
  text query 는 title, summary, when_to_use, steps, checkpoints, cautions, related_terms 를 대상으로 한다.
- 카드 목록은 title, card_type, primary_topic, task_type, source_file_name, display_date,
  checkpoints_count, steps_count, open_questions_count 컬럼으로 표시한다.
- 카드 상세에서는 title, summary, card_type, primary_topic, topic_tags, task_type,
  source_file_name, display_date, when_to_use, prerequisites, steps, checkpoints, cautions,
  examples, related_terms, open_questions, evidence_spans, sanitized markdown preview 를 확인할 수 있다.
- raw 원문은 표시하지 않고 `sanitized_markdown` / `card.to_markdown()` 중심으로 보여준다.
  metadata 는 expander 안에서만 확인한다.
- 안전한 관리 기능만 제공한다: 새로고침, 카드 JSON 다시 로드, 선택 카드 Markdown/JSON 다운로드.
  삭제 / 강제 재정규화 버튼은 아직 제공하지 않는다.
- `app/pages/2_문서_색인.py` 에 `ENABLE_LLM_NORMALIZATION` 현재 상태 표시와
  Guide/Slack 정규화 안내, ingest 결과의 `normalized_card_count` /
  `normalized_chunks_added` / `normalized_skipped_reason` 표시를 최소로 추가했다.
- 외부 API 호출 없이 검증 가능하도록 `tests/test_knowledge_card_viewer.py` 에 helper 단위 테스트
  (저장소 로드, 요약 분포, 필터, 검색, table dict, 한글 보존, 빈 저장소 처리 등) 를 추가했다.
- 검색 우선순위 적용은 Task 6, QA 답변 반영은 Task 7 에서 진행한다.

### Task 6 (완료) — 검색에서 `knowledge_card` 우선 retrieval 적용

- 검색 단계에서 `content_type=="knowledge_card"` / `source_type=="llm_normalized"`
  chunk 를 raw chunk 보다 우선 사용한다. raw chunk 는 보조 근거 / fallback 용도로
  계속 유지한다 (삭제하지 않음).
- 새 설정 (`.env.example` / `src/config.py`):
  - `PRIORITIZE_KNOWLEDGE_CARDS=true` (기본 ON, false 면 기존 점수 흐름과 동일)
  - `KNOWLEDGE_CARD_CONTENT_BOOST=1.35`
  - card_type 별 boost: `WORKFLOW_CARD_BOOST=1.30`, `CHECKLIST_CARD_BOOST=1.25`,
    `FAQ_CARD_BOOST=1.20`, `DECISION_CARD_BOOST=1.20`,
    `COMMUNICATION_TEMPLATE_BOOST=1.20`, `GLOSSARY_CARD_BOOST=1.10`
  - `RAW_EVIDENCE_BOOST=0.85` (같은 source_file 의 raw chunk 점수 보정)
  - `ENABLE_PARENT_RAW_EVIDENCE=true`, `PARENT_RAW_EVIDENCE_TOP_K=2`
- `src/rag/reranker.py` 에 helper 추가:
  - `is_knowledge_card_chunk(chunk_or_meta)` — content_type/source_type 둘 다 인식
  - `card_type_boost_for(card_type, query_intent)` — card_type 별 spec boost 와
    query intent 부합 시 추가 1.10x 부스트 반환
  - `apply_knowledge_card_priority(candidates, query_metadata)` — rerank_simple
    뒤에 호출되어 knowledge_card 점수 재가중 + 같은 파일 raw chunk 를
    `raw_evidence` 로, 그 외 raw chunk 를 `raw_fallback` 으로 라벨링
- 질문 의도(intent) 매핑은 `_INTENT_PROCEDURE` 에 "가이드", "체크리스트",
  "셋팅", "세팅", "어떻게" 를 추가하고, `_INTENT_COMMUNICATION` ("문안",
  "메일", "공유", "전달", "회신") 과 `_INTENT_GLOSSARY` ("용어", "무슨 뜻",
  "정의") 를 신설했다. card_type → intent 매핑은 다음과 같다:
  - workflow / checklist → procedure
  - faq / decision → explanation
  - communication_template → communication
  - glossary → glossary
  - issue → issue_lookup
- `src/rag/retriever.py` 의 `retrieve_with_details` 가 rerank 직후
  `apply_knowledge_card_priority` 를 호출하도록 연결했다. summary 에
  `prioritize_knowledge_cards`, `knowledge_card_content_boost`,
  `raw_evidence_boost`, `enable_parent_raw_evidence`,
  `parent_raw_evidence_top_k` 와 함께 결과 단의 `knowledge_card_count` /
  `raw_evidence_count` / `raw_fallback_count` 를 채운다.
  `ENABLE_PARENT_RAW_EVIDENCE=false` 면 노출 단계에서
  `parent_raw_chunk_ids` 가 빈 리스트로 마스킹된다.
- 각 chunk metadata 에 진단 필드 `retrieval_role`
  (`primary_card` / `raw_evidence` / `raw_fallback`),
  `knowledge_card_boost`, `card_type_boost`, `card_type_match` 가 기록된다.
- `app/pages/4_검색_테스트.py` UI 보강:
  - 상단에 PRIORITIZE_KNOWLEDGE_CARDS / KC_CONTENT_BOOST /
    RAW_EVIDENCE_BOOST / ENABLE_PARENT_RAW_EVIDENCE 표시
  - 메트릭 행에 `knowledge_card`, `raw_evidence`, `raw_fallback` 카운트 추가
  - 결과 테이블에 `retrieval_role`, `content_type`, `source_type`,
    `card_id`, `card_type`, `card_type_match`, `kc_boost`, `card_type_boost`,
    `parent_raw_chunk_ids`, `primary_topic`, `task_type`, `final_score` 컬럼 추가
  - `content_type` 필터 (전체/knowledge_card/conversation/text/raw_table) 와
    `card_type` 필터 (전체/workflow/checklist/issue/faq/decision/glossary/
    communication_template) 추가
  - 진단용 페이지이므로 Generation API 는 호출하지 않는다.
- 기존 raw chunk 흐름은 삭제하지 않는다. KC 후보가 threshold 를 통과하지 못하면
  raw chunk 가 fallback 으로 남는다 (`tests/test_retrieval_precision.py` 회귀 케이스 추가).
- 외부 API 없이도 검증 가능하도록 `tests/test_knowledge_card_retrieval.py` 에
  21 건 (helper 단위, intent 매칭, role 마킹, parent_raw_chunk_ids 보존,
  card_type 별 우선순위 시나리오, retriever summary 카운트, prioritize 비활성화,
  date/topic threshold 회귀) 을 추가했다.
- QA 답변을 knowledge_card 중심으로 재구성하는 작업은 Task 7 에서 진행한다.

### Task 7 (예정)

- Task 7: QA prompt 에서 `knowledge_card` 중심 답변

---

## 18. 자주 발생하는 오류와 해결 방법

| 증상 | 원인 | 해결 |
| --- | --- | --- |
| `GOOGLE_API_KEY 가 비어 있습니다` | `.env` 미설정 | `.env`에 키 입력 후 Streamlit 재시작 |
| `모델을 찾을 수 없습니다` | `.env`의 모델명이 사용 불가 | **5_API_상태확인** 페이지에서 사용 가능한 모델 확인 후 `.env` 갱신 |
| `quota 가 초과되었거나 rate limit` | Gemini 일일 한도 초과 | 잠시 후 재시도 또는 모델/한도 변경 |
| `timeout` / `connection error` | 네트워크/방화벽 | VPN, 프록시, 사내 방화벽 확인 |
| Excel 한글 깨짐 | Windows CP949 등 | 파일을 UTF-8 로 다시 저장하거나 `txt_parser` 인코딩 fallback 사용 |
| ChromaDB 잠금/락 | 다른 프로세스가 점유 | Streamlit 종료 후 재시작 (`storage/chroma_db` 잠금 파일 정리) |
| 첫 임베딩이 매우 느림 | sentence-transformers 모델 다운로드 | 한 번만 발생, 다음부터 캐시 사용 |
| `ModuleNotFoundError: src` | venv 미활성화 또는 cwd 잘못 | 프로젝트 루트(`work-rag-assistant/`)에서 venv 활성화 후 실행 |
| Windows 콘솔 한글 깨짐 | cmd 기본 인코딩이 cp949 | `chcp 65001` 로 UTF-8 전환하거나 PowerShell 사용 |
| numpy / chromadb segfault | 시스템 파이썬에 broken numpy | 반드시 `.venv` 안에서 `pip install -r requirements.txt` |

---

## 19. API 비용 절감 팁

- Excel 요약은 정말 필요한 파일만 ON
- 검색만 빠르게 확인할 때는 **4_검색_테스트** 페이지 사용
- `ENABLE_QUERY_REWRITE=false` 유지
- `TOP_K`를 6~8 정도로 유지
- 동일 파일을 반복 업로드하지 않기 (hash 기반 중복 방지가 동작하지만 업로드 자체에서 줄이는 것이 안전)

---

## 20. 현재 MVP에서 지원하지 않는 기능

- PDF 파싱
- 이미지 OCR
- BM25 / Hybrid Retrieval
- LLM 기반 reranker (cross-encoder, Gemini reranker)
- 자동 폴더 watcher
- 사용자 피드백 기반 평가 자동화
- 사내 SSO / 인증

---

## 21. 향후 개선 계획

- PDF 파서 추가 (`pypdf`, `pdfplumber`)
- OCR (`pytesseract`)
- BM25 + Vector hybrid retrieval
- Gemini / cross-encoder reranker
- FastAPI + React 분리
- 임베딩 모델 A/B 테스트 자동화
- retrieval hit rate / answer groundedness 지표화
- 사용자 피드백 기반 evaluation

---

## 22. Smoke Test

외부 API 호출 없이 핵심 모듈/파이프라인이 정상인지 빠르게 검증합니다.

```bash
# venv 활성화 후
python scripts/smoke_test.py
```

확인 항목:

1. `config` 로딩 + 필수 디렉터리 자동 생성
2. `src` 패키지 전체 import (34개 모듈)
3. `txt_parser` / `markdown_parser` 동작 (UTF-8 + CP949 fallback)
4. `excel_parser` 동작 (openpyxl 사용 가능 시)
5. `chunker` + Excel `parent-child` 연결 + `source_weight` + 컨텍스트 헤더
6. `cleaner` / `normalizer` 정제
7. `GOOGLE_API_KEY` 미설정 시 `GeminiClient` 가 친절한 한국어 에러를 반환
8. `VectorStore` 임시 디렉터리에서 collection 생성 (chromadb 사용 가능 시)

> 모든 항목이 PASS 면 종료 코드 0, 하나라도 실패하면 종료 코드 2.
> 외부 네트워크 / API Key 없어도 통과하도록 설계되어 있습니다.

---

## 23. 자주 쓰는 명령어 모음

```bash
# venv 활성화
source .venv/bin/activate          # Mac
.venv\Scripts\activate             # Windows

# 핵심 동작 점검 (외부 호출 없음)
python scripts/smoke_test.py

# Gemini API 연결/모델 점검
python scripts/check_gemini_api.py

# 폴더 일괄 색인
python scripts/ingest_folder.py
python scripts/ingest_folder.py --enable-summary
python scripts/ingest_folder.py --no-skip

# Excel 한국어 요약 일괄 생성 (비용 발생)
python scripts/summarize_excel_folder.py
python scripts/summarize_excel_folder.py --force

# Vector DB / 색인 결과 초기화
python scripts/reset_vector_db.py                          # 전체 초기화 (확인 프롬프트)
python scripts/reset_vector_db.py --yes                    # 확인 없이 즉시
python scripts/reset_vector_db.py --keep-processed --keep-registry --yes   # ChromaDB만

# Streamlit 앱
streamlit run app/main.py
bash scripts/run_app.sh             # Mac
scripts\run_app.bat                 # Windows
```
