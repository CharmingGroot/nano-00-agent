# nano-00-agent 테스트 보고서 + 코드 점검

**일시**: 2026-03-26
**버전**: 10 commits, Phase 1-5 완료
**Python**: 3.12.13
**테스트 결과**: 168/168 PASSED (26.70s)

---

## 1. 테스트 결과 요약

| 테스트 파일 | 테스트 수 | 결과 | 검증 대상 |
|------------|----------|------|----------|
| test_structured_outputs.py | 32 | ✅ ALL PASS | Goal/Reflection/Compression/State/Context/HITL 스키마 |
| test_knowledge_pipeline.py | 15 | ✅ ALL PASS | Parser/Chunker/Retriever 출력 스키마 |
| test_knowledge_bulk.py | 14 | ✅ ALL PASS | CSV×100, XLSX×100, 청킹×100, 포인터×100 |
| test_phase3.py | 33 | ✅ ALL PASS | ToolRegistry, SkillDB, IntentClassifier, DAG Executor |
| test_phase4_pipeline.py | 23 | ✅ ALL PASS | 파이프라인 초기상태, 토큰, 압축, 포인터, 멀티턴 |
| test_phase5_tools.py | 17 | ✅ ALL PASS | web_search×100, PDF×50, Notion×50 |
| test_multi_turn_e2e.py | 15 | ✅ ALL PASS | 50턴 상태무결성, 압축트리거, Reflection×50 |
| test_goal_generator.py | 5 | ✅ ALL PASS | Goal 파싱, fallback |
| test_reflector.py | 2 | ✅ ALL PASS | 정상/에러 Reflection |
| test_token_counter.py | 6 | ✅ ALL PASS | 토큰 카운팅, 임계값 |
| test_tool_result_compressor.py | 6 | ✅ ALL PASS | 압축, 트렁케이션, 포인터 형식 |
| **합계** | **168** | **168 PASSED** | |

### E2E 검증 (실제 Ollama qwen3.5:9b)

| 시나리오 | 결과 |
|---------|------|
| `"안녕하세요"` → Goal 생성 + 응답 | ✅ Goal: "Provide polite response", 2070 토큰 |
| CSV 업로드 (`test_tariff.csv`) | ✅ 1 chunk, document_id 반환 |
| 지식 검색 `"전기차 관세율은?"` | ✅ score 0.6152, 관세 데이터 반환 |
| 구조화 상태 JSON 8개 섹션 | ✅ goal, user_intent, intent_chain, task_graph, accumulated_data, knowledge_context, token_budget, hitl_state |

---

## 2. 기획 대비 구현 준수 검증

### 원래 기획 핵심 요구사항 대비 구현 현황

| # | 기획 요구사항 | 구현 여부 | 상세 |
|---|-------------|----------|------|
| 1 | **모든 LLM 호출은 LLMGateway를 통과** | ✅ 구현됨 | `llm_gateway.py` — chat(), chat_with_tool_loop(), embed() 전부 경유 |
| 2 | **Goal 생성 (질의 수신 직후)** | ✅ 구현됨 | `goal_generator.py` — 매 요청 시 구조화 Goal JSON 생성 |
| 3 | **Reflection (매 tool call 후)** | ✅ 구현됨 | `reflector.py` — goal_progress, deviation_detected, intent_chain_entry |
| 4 | **ToolResultCompressor (Goal 기반 압축)** | ✅ 구현됨 | Stage 1 LLM 압축 + Stage 2 강제 트렁케이션, JSON 출력 보장 3회 시도 |
| 5 | **Tool result 원본 DB 저장** | ⚠️ 부분 구현 | `tool_results` 테이블 존재, pipeline에서 `session.add()` 하지만 **commit 누락** |
| 6 | **포인터 = ID + 설명(desc)** | ✅ 구현됨 | `ptr:tool_result:uuid` + `desc` 동반, 100회 반복 검증 통과 |
| 7 | **구조화 상태 JSON (8개 섹션)** | ✅ 구현됨 | goal, user_intent, intent_chain, task_graph, accumulated_data, knowledge_context, token_budget, hitl_state |
| 8 | **intent_chain (자연어 의도 체인)** | ✅ 구현됨 | 매 스텝 자동 추가, Reflection 태그 포함 |
| 9 | **150K 토큰 안전망** | ✅ 구현됨 | `token_counter.py` — should_compress() 150K 체크 |
| 10 | **StateCompressor (구조화 JSON 압축)** | ✅ 구현됨 | `state_compressor.py` — LLM 호출로 상태 압축, 파이프라인 통합 |
| 11 | **스킬 DB 저장 (YAML이 아닌 PostgreSQL)** | ✅ 구현됨 | `skills` 테이블, `repository.py` CRUD, `/admin/skills` API |
| 12 | **도구 YAML 정의 (정적)** | ✅ 구현됨 | `registries/tools/*.yaml` — 5개 도구 정의 |
| 13 | **태스크 경계 HITL** | ✅ 구현됨 | `hitl_manager.py` — requires_hitl 체크, 일시정지/재개 |
| 14 | **Tool-Call 루프 (멀티턴 도구 호출)** | ✅ 구현됨 | `llm_gateway.py` chat_with_tool_loop() — max 10회 반복, 토큰 재계산 |
| 15 | **Docker 기반 (pg + pgvector + Redis)** | ✅ 구현됨 | `docker-compose.yml` — pgvector/pgvector:pg17, redis:7-alpine |
| 16 | **Ollama 로컬 모델 전용 (11434)** | ✅ 구현됨 | qwen3.5:9b + nomic-embed-text 연동 확인 |
| 17 | **지식 임베딩 (PDF, CSV, XLSX)** | ✅ 구현됨 | `ingestion.py` + `chunker.py` + `embedder.py` — 512tok/50overlap |
| 18 | **pgvector 코사인 유사도 검색** | ✅ 구현됨 | `retriever.py` — `<=>` operator, top-K 반환 |
| 19 | **IntentClassifier (LLM 기반)** | ✅ 구현됨 | 1회 LLM 호출로 intent/skill/complexity 분류 |
| 20 | **TaskDecomposer (스킬 DAG 또는 LLM 분해)** | ✅ 구현됨 | 스킬 매칭 → DB에서 DAG 로드, 없으면 LLM fallback |
| 21 | **TaskGraphExecutor (위상 정렬 실행)** | ✅ 구현됨 | Kahn's algorithm, 의존성 스킵, Reflector 통합 |
| 22 | **3경로 처리 (simple/tool-enabled/complex)** | ✅ 구현됨 | complexity ≤1: simple, ≤3+tools: tool-enabled, >3: full decomposition |

### 미구현 항목

| # | 기획 요구사항 | 상태 | 사유 |
|---|-------------|------|------|
| A | **Celery 워커 (병렬 fan-out)** | ❌ 미구현 | celery_app.py 존재하지만 pipeline에서 사용 안 함. 현재 모든 실행이 동기적 |
| B | **Dataset 커넥터 (Google Sheets, Airflow)** | ❌ 미구현 | YAML 정의만 존재, 실제 커넥터 코드 없음 |
| C | **deepseek-r1:14b fallback** | ❌ 미구현 | 설정에 존재하지만 모델 전환 로직 없음 |
| D | **세션 간 기억 (DB 기반 상태 영속화)** | ⚠️ 부분 구현 | ConversationState 모델 존재, 실제 DB 저장/로드는 인메모리 dict 사용 중 |
| E | **SkillRouter (lazy loading)** | ⚠️ 부분 구현 | 클래스 존재하지만 pipeline에서 직접 ToolRegistry 사용 |

---

## 3. 코드 견고성 점검 결과

### CRITICAL (즉시 수정 필요) — 4건

| # | 파일 | 문제 | 영향 |
|---|------|------|------|
| C1 | `llm_gateway.py` | httpx.AsyncClient 리소스 누수 — 싱글톤이 close() 호출 안 됨 | 서버 장시간 운영 시 연결 고갈 |
| C2 | `chat.py:31` | `_conversation_states` dict 동시 접근 시 race condition | 같은 conversation_id로 동시 요청 시 상태 손상 |
| C3 | `search_knowledge.py:24` | `kwargs["query"]` 직접 접근 — KeyError 미처리 | 필수 파라미터 없으면 500 에러 |
| C4 | `pipeline.py:280-332` | `on_tool_result` 콜백 내부 exception 미처리 | 압축/Reflection 실패 시 전체 tool loop 중단 |

### HIGH (빠른 수정 필요) — 6건

| # | 파일 | 문제 |
|---|------|------|
| H1 | `llm_gateway.py:82` | `resp.json()` 파싱 실패 시 JSONDecodeError 미처리 |
| H2 | `embedder.py:57` | 빈 임베딩 배열 접근 시 IndexError 가능 |
| H3 | `retriever.py:41,63` | 벡터 리터럴 SQL 문자열 직접 삽입 (안전하지만 확장 위험) |
| H4 | `service.py:55,69` | SQLAlchemy async `session.add()` → `commit()` 트랜잭션 관리 불확실 |
| H5 | `web_search.py:34-46` | 외부 API 호출에 retry/backoff 없음 |
| H6 | `ingestion.py:37-50` | PDF `fitz.open()` 예외 시 `doc.close()` 스킵 (리소스 누수) |

### MEDIUM — 5건

| # | 문제 요약 |
|---|----------|
| M1 | TaskGraph 에러 시 traceback 미보존 (`str(exc)`만 저장) |
| M2 | `chat.py:25-28` bare `except Exception: pass` — 모든 에러 삼킴 |
| M3 | pipeline에서 nested dict 접근 시 KeyError 가능성 |
| M4 | embedder에서 빈 chunks 리스트 처리 누락 |
| M5 | task_graph.py에서 `import re`가 함수 뒤에 위치 |

### LOW — 3건

| # | 문제 요약 |
|---|----------|
| L1 | 복합 태스크 경로에서 토큰 누적 안 됨 |
| L2 | 언어 코드 하드코딩 ("사용자 요청:") |
| L3 | tool result의 JSON 직렬화 시 비표준 타입 실패 가능 |

---

## 4. 권장 조치

### 즉시 (Phase 6에서)

1. **LLMGateway에 FastAPI lifespan 이벤트 연결** → 앱 종료 시 `gateway.close()` 호출
2. **conversation_states를 DB 기반으로 전환** 또는 `asyncio.Lock` per conversation_id 추가
3. **tool handler에 파라미터 검증** 추가 (`kwargs.get()` + ValueError)
4. **on_tool_result 콜백에 try/except** — 압축 실패 시 원본 반환 fallback
5. **PDF fitz.open()에 try/finally** 추가

### 단기 (1-2주)

6. Celery 워커 연결 — fan-out 스텝 병렬 실행
7. DB 기반 ConversationState 영속화
8. 외부 API retry/backoff (tenacity)
9. deepseek-r1:14b fallback 로직

### 중기 (1개월)

10. Dataset 커넥터 실제 구현 (Google Sheets)
11. 구조화 로깅 (correlation ID per conversation)
12. Rate limiting
13. 모니터링 + 메트릭 수집
