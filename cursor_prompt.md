# Cursor용 시스템 프롬프트 (개발자 가이드)

이 파일은 본인 / 동료가 Cursor에서 이 프로젝트를 다룰 때 참고하는 컨벤션 모음이다.

## 0. 절대 원칙

1. **로컬 LLM / 외부 Vector DB 사용 금지**
   - Ollama, llama.cpp, Pinecone, Weaviate Cloud, Qdrant Cloud, Supabase 모두 사용하지 않는다.
2. LLM 추론은 **Google Gemini API** 만 사용한다 (`google-genai` SDK).
3. 모든 경로는 `pathlib.Path`로 다룬다. `"/"` 또는 `"\\"` 하드코딩 금지.
4. 모델명은 하드코딩하지 말고 **`.env`** 에서 읽는다. (`GENERATION_MODEL`, `EMBEDDING_PROVIDER` 등)
5. 한글 파일명 / UTF-8 / CP949 모두 고려한다.
6. JSON 저장 시 `ensure_ascii=False`.
7. 외부로 보내는 텍스트는 **사용자 명시 호출** (검색 / 요약 / 답변) 시에만.

## 1. 모듈 책임

```
src/
  config.py         # .env 로드 + 경로/상수 노출
  logger.py         # 공통 로거
  ingestion/        # 파일 -> ParsedSection list
  preprocessing/    # 정제 + 청크 + 중복 제거
  summarization/    # Excel 한국어 요약 (Gemini)
  storage/          # 디스크 / Chroma / registry
  rag/              # gemini_client, embedder, retriever, reranker, generator, qa_pipeline
  evaluation/       # 추후 평가용 placeholder
  utils/            # path/hash/time/encoding/cost/token
  schemas/          # dataclass 정의 (Document/Chunk/...)
app/
  main.py           # Streamlit 홈
  pages/            # 1_문서_업로드 ~ 6_Excel_요약관리
  components/       # UI 헬퍼
```

## 2. RAG 흐름 한 줄 요약

```
파일 -> 파싱 -> 정제 -> (Excel 요약) -> 청크 -> 임베딩 -> ChromaDB
질문 -> (rewrite) -> vector search -> rerank -> parent-child 보강 -> prompt -> Gemini -> 답변+근거+로그
```

## 3. 새 기능을 추가할 때 체크리스트

- [ ] `src/config.py` 에 환경변수 노출
- [ ] `src/schemas/` 에 dataclass 추가
- [ ] `src/utils/` 에 유틸 추가
- [ ] 단위 동작 확인용 `notebooks/` 추가 가능
- [ ] `tests/` 에 최소 케이스 추가
- [ ] `README.md` 업데이트
