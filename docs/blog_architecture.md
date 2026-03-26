# 공공 sLLM의 한계를 미들웨어로 극복하기 — 로컬 AI 에이전트 아키텍처 설계

> 로컬 LLM(9B)으로도 복잡한 멀티스텝 태스크를 수행할 수 있는 에이전트 시스템을 어떻게 설계했는가

## 목차
1. [왜 직접 만들었는가](#왜-직접-만들었는가)
2. [문제 정의](#문제-정의)
3. [핵심 설계 철학](#핵심-설계-철학)
4. [전체 아키텍처](#전체-아키텍처)
5. [Goal → Tool-Call Loop → Reflection 파이프라인](#goal--tool-call-loop--reflection-파이프라인)
6. [포인터 기반 컨텍스트 관리](#포인터-기반-컨텍스트-관리)
7. [도구 결과 압축 (20K→4K 토큰)](#도구-결과-압축)
8. [구조화 상태 JSON](#구조화-상태-json)
9. [기술 스택과 구현](#기술-스택과-구현)
10. [실제 동작 예시](#실제-동작-예시)
11. [남은 과제](#남은-과제)

---

## 왜 직접 만들었는가

처음엔 기존 프레임워크를 쓰려고 했습니다.

LangChain의 `create_react_agent`, `create_tool_calling_agent`부터 시작해서, 최근 나온 LangGraph의 `StateGraph` 기반 에이전트까지 전부 시도해봤습니다. 단순한 데모 — "검색해서 요약해줘" 수준 — 에서는 잘 동작합니다.

**문제는 공공 환경의 실제 요구사항에서 터졌습니다.**

공공 프로젝트에서는 이런 걸 요구합니다:
- **로컬 모델 전용** (외부 API 호출 불가, 망분리 환경)
- **50턴 넘는 멀티턴 대화**에서 이전 맥락을 정확히 참조
- **20,000~40,000 토큰짜리 도구 결과**를 컨텍스트 오버플로우 없이 처리
- **태스크 중간에 사람이 확인**(HITL)하고 승인해야 다음 단계로 진행
- **9B급 sLLM**으로 이 모든 걸 해야 함 (GPU 예산 한정)

LangChain/LangGraph로는 이게 안 됩니다. 구체적으로:

**1. 컨텍스트 관리를 모델에 떠넘깁니다.**
LangChain의 에이전트는 대화 히스토리를 통째로 LLM에 넣습니다. 10턴만 지나도 sLLM의 컨텍스트가 가득 차고, 맥락이 조용히 손실됩니다. "요약해서 넣어라"(`ConversationSummaryMemory`)는 해법이 있지만, sLLM이 생성한 자유 텍스트 요약은 정보 손실이 심하고 구조화되어 있지 않아서 정확한 참조가 불가능합니다.

**2. 도구 호출 결과의 크기를 통제할 수 없습니다.**
RAG로 문서 20개 청크를 가져오거나, 웹 검색으로 10페이지 본문을 크롤링하면 수만 토큰입니다. LangChain은 이 결과를 그대로 메시지에 넣습니다. 토큰 예산을 추적하거나, Goal 기준으로 불필요한 결과를 압축하는 메커니즘이 없습니다.

**3. 미들웨어 레이어가 없습니다.**
"모든 LLM 호출 전에 토큰을 세고, 임계값을 넘으면 상태를 압축하고, 도구 결과는 Goal 기준으로 필터링하고, 매 스텝 후에 진행률을 평가한다" — 이런 **LLM 호출을 감싸는 미들웨어 파이프라인**을 LangChain 위에 얹으려면, 결국 LangChain의 내부 구조를 전부 우회하게 됩니다. 프레임워크를 쓰는 의미가 없어집니다.

**4. 커스텀 상태 관리의 한계.**
LangGraph의 `StateGraph`는 상태를 정의할 수 있지만, "포인터 기반 컨텍스트 — DB에 원본 저장, LLM에는 ID+설명만 전달"이나 "구조화된 상태 JSON 8개 섹션을 매 턴마다 관리"같은 수준의 세밀한 제어는 직접 구현해야 합니다.

결국 **프레임워크의 추상화가 도움이 되는 게 아니라 방해가 되는 지점**에 도달했습니다.

그렇다고 완전히 백지에서 시작한 건 아닙니다. 오픈소스 에이전트 프로젝트들에서 많은 영감을 받았습니다.

- **OpenClaw**의 워크플로우 오케스트레이션 설계 — 태스크를 선언적으로 정의하고 DAG로 실행하는 패턴
- **Hermes Agent**의 도구 호출 루프 구조 — LLM이 도구를 반복 호출하며 결과를 누적하는 방식
- 이 외에도 AutoGen, CrewAI 등 멀티에이전트 프레임워크들의 상태 관리 패턴을 참고했습니다

이 프로젝트들이 보여준 **"에이전트는 LLM 한 번 호출이 아니라, 루프와 상태 관리의 문제다"**라는 통찰이 설계의 출발점이 됐습니다. 다만 이들도 sLLM + 공공 환경의 제약 (망분리, 로컬 전용, 극한의 토큰 절약)까지는 커버하지 못해서, 핵심 미들웨어는 직접 설계했습니다.

결국 공공 환경에서의 방향은 명확합니다. **n8n 같은 노드 기반 워크플로우 오케스트레이터를 좀 더 Agentic하게 사용하는 것.** 이미 공공 SI 현장에서는 n8n, Apache Airflow 같은 워크플로우 도구로 업무를 자동화하고 있습니다. 여기에 LLM을 "판단 노드"로 끼워 넣으면 — 워크플로우의 분기를 사람이 아니라 LLM이 결정하고, 도구 호출 순서를 동적으로 조정하고, 결과를 보고 다음 액션을 선택하는 — Agentic Workflow가 됩니다.

이 프로젝트는 그 방향의 구현체입니다. 스킬을 DAG(방향 비순환 그래프)로 정의하고, 미들웨어가 노드 간 데이터 흐름과 상태를 관리하고, LLM은 각 노드에서 판단만 합니다. n8n의 노드 기반 실행 모델에서 영감을 받되, LLM 특유의 문제(맥락 손실, 토큰 폭발, 할루시네이션)를 미들웨어 레벨에서 잡는 구조입니다.

모델은 판단만 하고, 나머지는 전부 미들웨어가 담당하는 구조로.

---

## 문제 정의

공공 환경에서 사용할 수 있는 sLLM(Small Language Model, 9B~14B급)에는 세 가지 근본적인 한계가 있습니다.

**1. 맥락 손실을 스스로 감지하지 못합니다.**
sLLM은 컨텍스트 윈도우가 가득 차도 "정보가 부족하다"고 인지하지 못합니다. 조용히 엉뚱한 답변을 생성합니다.

**2. 세션 간 기억이 없습니다.**
매 요청이 독립적이라 "아까 분석한 관세 데이터를 다시 보여줘"라는 요청에 응답할 수 없습니다.

**3. 단일 요청의 토큰이 매우 큽니다.**
웹 검색 결과(20,000~40,000 토큰), 문서 RAG 결과(10,000~15,000 토큰) 등 도구 호출 결과가 컨텍스트를 빠르게 소진합니다.

이 세 가지 문제는 모델 자체로는 해결할 수 없습니다. **LangChain이나 LangGraph 같은 프레임워크로도 해결할 수 없습니다.** 모델 바깥에서, 프레임워크 바깥에서, 미들웨어 레벨에서 보완해야 합니다.

---

## 핵심 설계 철학

> **"sLLM이 못하는 걸 모델한테 시키지 말고, 미들웨어가 하자."**

| 원칙 | 설명 |
|------|------|
| 모델은 판단만, 미들웨어가 인프라 전담 | 토큰 카운팅, 상태 압축, 포인터 관리는 코드가 담당 |
| Goal 기반 파이프라인 | 매 요청마다 구조화된 목표를 생성하여 전 과정의 판단 기준으로 사용 |
| 포인터 기반 컨텍스트 | LLM에 raw 데이터 대신 포인터(ID+설명)만 전달, 실제 데이터는 DB |
| 구조화 상태 JSON | 자유 텍스트 요약이 아닌 타입된 JSON으로 상태 관리 |
| 선언적 도구/스킬 | 하드코딩 없이 YAML(도구) + DB(스킬)로 동적 관리 |

---

## 전체 아키텍처

```
Client (Web/CLI)
       │
       ▼
┌─ API Layer (FastAPI) ────────────────────────────────┐
│  /chat  /knowledge/upload  /knowledge/search  /admin │
└──────────────────┬───────────────────────────────────┘
                   ▼
┌─ Orchestrator ───────────────────────────────────────┐
│  IntentClassifier → TaskDecomposer → TaskGraphExec   │
└──────────────────┬───────────────────────────────────┘
                   ▼
┌─ Middleware Engine ──────────────────────────────────┐
│  GoalGenerator → SkillRouter → ContextManager        │
│  → TokenCounter → StateCompressor → HITLManager      │
│  → LLMGateway (Tool-Call Loop + Reflection)          │
│  → ToolResultCompressor → PointerResolver            │
│                                                      │
│  ※ 모든 Ollama 호출은 반드시 LLMGateway를 통과       │
└──────────────────┬───────────────────────────────────┘
         ┌─────────┼──────────┐
         ▼         ▼          ▼
   Ollama:11434  PostgreSQL  Redis
   qwen3.5:9b   +pgvector   (queue/cache)
   nomic-embed-text
```

**모든 LLM 호출이 `LLMGateway`를 통과**하는 것이 핵심입니다. 어떤 컴포넌트도 직접 Ollama를 호출하지 않습니다. Gateway가 토큰 추적, 압축 트리거, 도구 루프를 관장합니다.

---

## Goal → Tool-Call Loop → Reflection 파이프라인

사용자 질문이 들어오면 세 단계를 거칩니다.

### Phase A: Goal 생성

질의 수신 직후, LLM 1회 호출로 **구조화된 Goal 객체**를 만듭니다:

```json
{
  "final_objective": "수입자동차 관세 현황에서 전기차와 SUV 비교 분석",
  "success_criteria": ["전기차 관세율 확인", "SUV 관세율 확인", "비교 분석 완료"],
  "required_outputs": ["비교 분석 결과"],
  "estimated_steps": 3
}
```

이 Goal이 **이후 모든 과정의 판단 기준**이 됩니다. 도구 결과 압축 시 "이 데이터가 Goal에 필요한가?", Reflection 시 "Goal 대비 진행률은?", 포인터 선별 시 "이번 질문에 어떤 포인터가 필요한가?" — 전부 Goal을 참조합니다.

### Phase B: Tool-Call Loop

```
[프롬프트 조립] → [Ollama 호출] → tool_calls 있는가?
                       ▲               │ Yes        │ No
                       │               ▼            ▼
                  [토큰 재계산] ← [도구 실행]    [최종 응답]
                       ▲               │
                       │               ▼
                  [Reflection] ← [결과 압축 (Goal 기준)]
```

1회 사용자 질문이 내부적으로 **N회 LLM↔도구 왕복**을 발생시킵니다.
각 왕복마다: Goal 확인 → 도구 실행 → 결과 압축 → Reflection → 토큰 재계산.
`max_tool_iterations=10` 안전장치로 무한 루프를 방지합니다.

### Phase C: Reflection

매 도구 호출 완료 후, 미들웨어가 Goal 대비 진행을 평가합니다:

```json
{
  "step_completed": "search_knowledge",
  "goal_progress": { "progress_pct": 33, "criteria_met": ["전기차 관세율 확인"] },
  "deviation_detected": false,
  "intent_chain_entry": "search_knowledge 완료 → 33% 진행 [Reflection: Goal 정상 진행]"
}
```

- `deviation_detected=true` → Goal에서 벗어남 감지 → LLM에 재조정 요청
- `should_abort=true` → 복구 불가 → 사용자에게 중간 결과 반환
- 결과는 `intent_chain`에 자동 추가 → 다음 턴에서 "왜 이 작업을 하고 있는지" 맥락 유지

---

## 포인터 기반 컨텍스트 관리

### 문제

50턴 대화에서 매 턴마다 도구 결과를 대화 히스토리에 넣으면 컨텍스트가 즉시 초과합니다.

### 해법

LLM 컨텍스트에는 **포인터(ID + 설명)**만 보유하고, 실제 데이터는 DB에 저장합니다.

```json
{
  "accumulated_data": {
    "search_knowledge_abc123": {
      "ptr": "ptr:tool_result:550e8400-e29b-41d4-a716-446655440000",
      "desc": "지식 검색 결과 (전기차 관세) — 5개 청크, 유사도 0.85",
      "token_count_raw": 32000,
      "token_count_compressed": 3500
    }
  }
}
```

### 포인터 해석 (PointerResolver)

Turn 50에서 "Turn 3에서 했던 관세 분석 결과 다시 보여줘"라고 하면:

1. **수집**: `state["accumulated_data"]`에서 모든 포인터 목록 추출
2. **선별**: LLM에 포인터 카탈로그(ptr + desc)를 보여주고 "이번 질문에 필요한 포인터를 골라라" 요청
3. **조회**: 선택된 포인터로 `tool_results` 또는 `chunks` 테이블에서 실제 데이터 fetch
4. **주입**: fetch한 데이터를 시스템 프롬프트의 `relevant_data` 섹션에 삽입

이 방식으로 50턴 넘게 대화해도 **이전 턴의 결과를 정확히 참조**할 수 있습니다.

---

## 도구 결과 압축

### 현실적 규모

| 소스 | Raw 토큰 | 압축 후 |
|------|---------|--------|
| 웹 검색 (10페이지 크롤링) | 32,000 | 3,500 |
| 지식 RAG (상위 20 청크) | 18,000 | 3,200 |
| 데이터셋 조회 (수백 행) | 25,000 | 4,000 |

### 2-Stage 압축

**Stage 1: Goal 기반 LLM 압축**
- Goal의 `final_objective` + `success_criteria`와 raw 결과를 LLM에 전달
- LLM이 Goal 기준으로 관련 항목만 남기고, 팩트/수치만 추출
- **JSON 출력 보장**: 1차 시도 → 실패 시 리트라이 → 3차 강제 래핑

**Stage 2: 강제 트렁케이션 (안전망)**
- Stage 1 후에도 `hard_limit(8000)` 초과 시 상위 N개만 남기고 잘라냄
- 잘린 부분의 포인터 유지 → 필요 시 재접근 가능

**원본은 반드시 DB에 보존**: `tool_results` 테이블에 `raw_output` + `compressed_output` 분리 저장.

---

## 구조화 상태 JSON

자유 텍스트 요약 대신, **8개 섹션으로 구성된 타입된 JSON**:

```json
{
  "goal": {
    "final_objective": "...",
    "success_criteria": ["...", "..."],
    "progress_pct": 50,
    "criteria_status": {"기준1": "done", "기준2": "pending"}
  },
  "user_intent": { "intent": "tool_use", "skill": null },
  "intent_chain": [
    "사용자가 관세 비교 요청",
    "search_knowledge 완료 → 33% [Reflection: Goal 정상 진행]",
    "비교 분석 완료 → 100% [Reflection: Goal 달성]"
  ],
  "task_graph": { "status": "running", "completed": [...], "pending": [...] },
  "accumulated_data": { "search_abc": { "ptr": "ptr:tool_result:uuid", "desc": "..." } },
  "knowledge_context": { "active_chunk_ids": ["chunk:abc"] },
  "token_budget": { "model": "qwen3.5:9b", "limit": 256000, "threshold": 150000, "used": 48000 },
  "hitl_state": { "awaiting": false }
}
```

`intent_chain`이 핵심입니다. sLLM도 이해할 수 있는 **자연어 의도 체인**으로, "왜 이 태스크를 하고 있는지"를 추적합니다. 매 스텝 완료 시 미들웨어가 자동으로 한 줄씩 추가합니다.

토큰이 150K에 도달하면 `StateCompressor`가 LLM을 호출하여 상태를 압축합니다. intent_chain은 최근 10개만 유지하고, accumulated_data는 포인터만 남깁니다.

---

## 기술 스택과 구현

| 구성요소 | 선택 | 역할 |
|---------|------|------|
| Python 3.12 | 언어 | async/await 기반 |
| FastAPI | API | /chat, /knowledge, /admin |
| PostgreSQL 17 + pgvector 0.8.2 | 벡터 DB | 지식 임베딩 + 유사도 검색 |
| Redis 7 | 큐/캐시 | Celery broker |
| Ollama | LLM 런타임 | qwen3.5:9b (256K ctx) |
| nomic-embed-text | 임베딩 | 768차원, 8K 컨텍스트 |
| Docker Compose | 인프라 | pgvector + Redis + API + Worker |

### 지식 파이프라인

```
PDF/CSV/XLSX → DocumentParser → TextChunker(512tok, 50overlap)
→ Embedder(nomic-embed-text) → pgvector(Vector(768))
→ KnowledgeRetriever(cosine similarity, top-K)
```

### 도구/스킬 시스템

- **도구(Tool)**: 정적 YAML 정의 + Python 핸들러. 코드와 함께 배포.
- **스킬(Skill)**: PostgreSQL `skills` 테이블에 JSONB로 저장. API로 CRUD 가능.
- 스킬은 도구의 DAG(방향 비순환 그래프)로, 위상 정렬로 실행됩니다.
- `{{steps.search.output}}` 같은 템플릿으로 스텝 간 데이터 전달.

---

## 실제 동작 예시

### "AI 연구논문을 검색하고, 트렌드별로 순위를 매겨서 요약한 뒤 PDF로 출력해줘"

```
IntentClassifier (LLM 1회)
  → intent: skill_match, skill: research_and_report

Skill DAG 실행:
  1. web_search (API) → 10건 수집                    [0 LLM]
  2. classify_trends → 3개 트렌드 분류               [LLM 1회]
  3. rank_trends → 순위 매김                          [LLM 1회]
  4. summarize_each → 각 트렌드 상세 요약             [LLM 3회, fan-out]
  5. generate_pdf → PDF 파일 생성                     [0 LLM, 코드]

총: 6회 원자적 LLM 호출 (하나의 거대한 프롬프트 대신)
```

각 LLM 호출은 Goal + intent_chain + 이전 스텝 결과(압축본)만 포함한 **최소 프롬프트**를 받습니다.

---

## 남은 과제

1. **Celery 병렬 fan-out**: 현재 순차 실행 → Celery 워커로 병렬화
2. **DB 기반 상태 영속화**: 현재 인메모리 dict → ConversationState 테이블 활용
3. **모델 fallback**: qwen3.5:9b 실패 시 deepseek-r1:14b 자동 전환
4. **Dataset 커넥터**: Google Sheets, Airflow 등 외부 데이터 소스 연동
5. **구조화 로깅**: correlation ID 기반 요청 추적
6. **실제 50턴 멀티턴 검증**: 포인터 기반 맥락 복원의 실전 테스트

---

## 마무리

sLLM의 한계를 모델 학습으로 해결하려는 시도는 한계가 있습니다. 대신 **미들웨어 레이어에서 맥락 관리, 토큰 추적, 상태 압축을 담당**하면, 9B 모델로도 복잡한 멀티스텝 에이전트를 구현할 수 있습니다.

핵심은:
- **Goal을 먼저 생성**해서 전 과정의 판단 기준으로 삼고
- **포인터로 컨텍스트를 관리**해서 50턴 넘어도 이전 맥락을 참조할 수 있게 하고
- **도구 결과를 Goal 기준으로 압축**해서 컨텍스트 오버플로우를 방지하는 것

이 세 가지입니다.

> GitHub: [CharmingGroot/nano-00-agent](https://github.com/CharmingGroot/nano-00-agent)
> 테스트: 187개 통과 (Python 3.12)

---

*이 글은 nano-00-agent 프로젝트의 아키텍처 설계 과정을 정리한 것입니다.*
*공공 sLLM 에이전트 시스템에 관심 있는 분들에게 도움이 되길 바랍니다.*

**태그**: `sLLM`, `AI에이전트`, `Ollama`, `RAG`, `pgvector`, `FastAPI`, `미들웨어`, `컨텍스트관리`, `로컬LLM`, `에이전트아키텍처`, `LangChain한계`, `n8n`, `AgenticWorkflow`, `공공AI`, `오픈소스LLM`, `Docker`, `PostgreSQL`, `벡터DB`, `HITL`, `태스크분해`
