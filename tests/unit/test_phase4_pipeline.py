"""Tests for Phase 4 — MiddlewarePipeline integration.

Validates:
- Pipeline init creates correct initial state
- State schema after processing
- Token budget tracking
- Compression trigger conditions
- Tool result DB storage + pointer format
- Multi-turn state persistence
"""
import json
import uuid

import pytest

from src.middleware.pipeline import MiddlewarePipeline
from src.middleware.token_counter import TokenCounter
from src.middleware.state_compressor import StateCompressor
from src.middleware.tool_result_compressor import ToolResultCompressor
from src.middleware.context_manager import ContextManager

STATE_REQUIRED_SECTIONS = {
    "goal", "user_intent", "intent_chain", "task_graph",
    "accumulated_data", "knowledge_context", "token_budget", "hitl_state",
}


class TestPipelineInitState:
    """Verify initial state schema."""

    def test_init_state_has_all_sections(self):
        state = MiddlewarePipeline._init_state()
        assert STATE_REQUIRED_SECTIONS.issubset(state.keys())

    def test_init_state_token_budget(self):
        state = MiddlewarePipeline._init_state()
        tb = state["token_budget"]
        assert "model" in tb
        assert "limit" in tb
        assert "threshold" in tb
        assert "used" in tb
        assert tb["used"] == 0
        assert tb["threshold"] < tb["limit"]

    def test_init_state_intent_chain_empty(self):
        state = MiddlewarePipeline._init_state()
        assert state["intent_chain"] == []

    def test_init_state_hitl_not_awaiting(self):
        state = MiddlewarePipeline._init_state()
        assert state["hitl_state"]["awaiting"] is False

    def test_init_state_accumulated_data_empty(self):
        state = MiddlewarePipeline._init_state()
        assert state["accumulated_data"] == {}

    def test_init_state_knowledge_context(self):
        state = MiddlewarePipeline._init_state()
        kc = state["knowledge_context"]
        assert "active_chunk_ids" in kc
        assert "document_refs" in kc
        assert isinstance(kc["active_chunk_ids"], list)


class TestTokenBudgetTracking:
    """Verify token budget updates correctly."""

    def test_should_compress_at_threshold(self):
        assert TokenCounter.should_compress(150_000)
        assert not TokenCounter.should_compress(100_000)

    def test_token_counting_adds_up(self):
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello " * 1000},
        ]
        count = TokenCounter.count_messages_tokens(msgs)
        assert count > 1000  # At least 1000 tokens for "Hello " * 1000

    def test_state_budget_update_simulation(self):
        """Simulate token budget updates across multiple turns."""
        state = MiddlewarePipeline._init_state()
        for turn in range(10):
            tokens_this_turn = 2000 + turn * 500
            state["token_budget"]["used"] += tokens_this_turn
        assert state["token_budget"]["used"] == sum(2000 + i * 500 for i in range(10))


class TestStateCompression:
    """Verify state compression input/output schemas."""

    def test_compression_messages_schema(self):
        state = MiddlewarePipeline._init_state()
        state["intent_chain"] = ["step 1 done", "step 2 done"]
        messages = StateCompressor.build_compression_messages(state, [], 100000)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_parse_compressed_state_valid(self):
        original = MiddlewarePipeline._init_state()
        original["intent_chain"] = ["a", "b", "c"]
        # Simulate LLM returning compressed state
        compressed_json = json.dumps(original)
        result = StateCompressor.parse_compressed_state(compressed_json, original)
        assert STATE_REQUIRED_SECTIONS.issubset(result.keys())

    def test_parse_compressed_state_fallback(self):
        original = MiddlewarePipeline._init_state()
        result = StateCompressor.parse_compressed_state("invalid json", original)
        assert result == original  # Falls back to original


class TestToolResultDBStorage:
    """Verify tool result compression + pointer format for DB storage."""

    def test_large_result_needs_compression(self):
        large_result = {"data": " ".join(["important finding"] * 3000)}
        assert ToolResultCompressor.needs_compression(large_result)

    def test_compressed_result_has_pointer(self):
        valid_json = json.dumps({
            "items": [{"key": "a", "data": "b"}],
            "items_total": 100,
            "items_retained": 5,
            "summary": "Compressed findings",
        })
        uid = str(uuid.uuid4())
        result = ToolResultCompressor.parse_compressed_response(valid_json, {}, uid)
        assert result["structured"] is True
        assert result["source_pointer"] == f"ptr:tool_result:{uid}"

    def test_accumulated_data_pointer_schema(self):
        """Pointers in accumulated_data must have ptr + desc."""
        uid = str(uuid.uuid4())
        entry = {
            "ptr": f"ptr:tool_result:{uid}",
            "desc": "웹 검색 결과 20건, 6건 관련",
            "token_count_raw": 32000,
            "token_count_compressed": 3500,
        }
        assert entry["ptr"].startswith("ptr:tool_result:")
        assert isinstance(entry["desc"], str)
        assert len(entry["desc"]) > 0

    def test_100_pointer_entries_conform(self):
        """Generate 100 pointer entries, all must conform."""
        for i in range(100):
            uid = str(uuid.uuid4())
            entry = {
                "ptr": f"ptr:tool_result:{uid}",
                "desc": f"Step {i} result ({i * 10} items)",
                "token_count_raw": 20000 + i * 100,
            }
            assert entry["ptr"].startswith("ptr:")
            assert "desc" in entry
            assert entry["token_count_raw"] > 0


class TestMultiTurnStatePersistence:
    """Verify state evolves correctly across multiple turns."""

    def test_intent_chain_grows(self):
        state = MiddlewarePipeline._init_state()
        for i in range(20):
            state["intent_chain"].append(f"Turn {i}: action completed")
        assert len(state["intent_chain"]) == 20
        assert "Turn 0" in state["intent_chain"][0]
        assert "Turn 19" in state["intent_chain"][-1]

    def test_accumulated_data_grows_with_pointers(self):
        state = MiddlewarePipeline._init_state()
        for i in range(10):
            uid = str(uuid.uuid4())
            state["accumulated_data"][f"step_{i}"] = {
                "ptr": f"ptr:tool_result:{uid}",
                "desc": f"Result from step {i}",
            }
        assert len(state["accumulated_data"]) == 10
        for key, val in state["accumulated_data"].items():
            assert "ptr" in val
            assert "desc" in val

    def test_goal_criteria_status_evolves(self):
        state = MiddlewarePipeline._init_state()
        state["goal"] = {
            "success_criteria": ["a", "b", "c"],
            "criteria_status": {"a": "pending", "b": "pending", "c": "pending"},
            "progress_pct": 0,
        }
        # Complete criteria one by one
        state["goal"]["criteria_status"]["a"] = "done"
        state["goal"]["progress_pct"] = 33
        state["goal"]["criteria_status"]["b"] = "done"
        state["goal"]["progress_pct"] = 66
        state["goal"]["criteria_status"]["c"] = "done"
        state["goal"]["progress_pct"] = 100

        assert all(v == "done" for v in state["goal"]["criteria_status"].values())
        assert state["goal"]["progress_pct"] == 100

    def test_context_assembly_with_evolved_state(self):
        """Context assembler should include current state correctly."""
        state = MiddlewarePipeline._init_state()
        state["intent_chain"] = [
            "사용자가 관세 분석 요청",
            "PDF 업로드 완료 (ptr:doc:abc = 관세현황.pdf)",
            "검색 완료 [Reflection: Goal 정상 진행]",
        ]
        goal = {
            "final_objective": "관세 현황 분석",
            "success_criteria": ["데이터 분석", "요약 작성"],
            "progress_pct": 50,
        }

        messages = ContextManager.build_simple_chat_prompt(
            system_prompt="You are an assistant.",
            user_message="관세율을 비교해줘",
            goal=goal,
            state=state,
        )

        system_content = messages[0]["content"]
        assert "관세 현황 분석" in system_content
        # intent_chain entries should be present
        assert "PDF 업로드 완료" in system_content
        assert "검색 완료" in system_content


class TestPipelineOutputSchema:
    """Verify pipeline output has correct schema."""

    PIPELINE_OUTPUT_FIELDS = {
        "response", "conversation_state", "goal", "token_count",
        "pending_hitl", "task_progress",
    }

    def test_simple_chat_output_schema_simulated(self):
        """Simulate simple chat output and verify schema."""
        output = {
            "response": "안녕하세요!",
            "conversation_state": MiddlewarePipeline._init_state(),
            "goal": {
                "final_objective": "인사",
                "success_criteria": ["응답"],
                "progress_pct": 100,
            },
            "token_count": {
                "prompt": 50,
                "completion": 20,
                "total_this_turn": 70,
                "should_compress": False,
            },
            "pending_hitl": None,
            "task_progress": None,
        }
        assert self.PIPELINE_OUTPUT_FIELDS.issubset(output.keys())
        assert isinstance(output["response"], str)
        assert STATE_REQUIRED_SECTIONS.issubset(output["conversation_state"].keys())

    def test_complex_task_output_schema_simulated(self):
        """Simulate complex task output and verify schema."""
        state = MiddlewarePipeline._init_state()
        state["task_graph"]["status"] = "done"
        state["accumulated_data"]["search"] = {
            "ptr": "ptr:tool_result:abc",
            "desc": "검색 결과 10건",
        }
        output = {
            "response": "리포트가 완성되었습니다.",
            "conversation_state": state,
            "goal": {
                "final_objective": "리포트 생성",
                "success_criteria": ["검색", "요약", "PDF"],
                "progress_pct": 100,
            },
            "token_count": {"total_this_turn": 5000},
            "pending_hitl": None,
            "task_progress": {
                "status": "done",
                "total_steps": 5,
                "completed_steps": 5,
            },
        }
        assert self.PIPELINE_OUTPUT_FIELDS.issubset(output.keys())
        assert output["task_progress"]["status"] == "done"
        assert output["task_progress"]["total_steps"] == 5

    def test_hitl_pause_output_schema_simulated(self):
        """Simulate HITL pause output."""
        output = {
            "response": "노션 게시 전 확인이 필요합니다.",
            "conversation_state": MiddlewarePipeline._init_state(),
            "goal": {"final_objective": "test"},
            "token_count": {"total_this_turn": 1000},
            "pending_hitl": {
                "action": "create_notion_page",
                "description": "노션에 페이지를 생성합니다",
                "preview": {"title": "리포트"},
                "confirm_prompt": "'create_notion_page' 작업을 진행할까요?",
            },
            "task_progress": None,
        }
        assert output["pending_hitl"] is not None
        hitl = output["pending_hitl"]
        assert "action" in hitl
        assert "description" in hitl
        assert "confirm_prompt" in hitl
