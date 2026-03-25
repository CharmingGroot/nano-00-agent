"""Tests for structured output schemas across all middleware components.

Verifies that every component produces outputs conforming to the
expected JSON schemas — Goal, Reflection, Compression, State, Pointers, etc.
"""
import json
import uuid

import pytest

from src.middleware.goal_generator import GoalGenerator
from src.middleware.reflector import Reflector
from src.middleware.tool_result_compressor import ToolResultCompressor
from src.middleware.context_manager import ContextManager
from src.middleware.hitl_manager import HITLManager


# ============================================================
# Goal Schema Validation
# ============================================================

GOAL_REQUIRED_FIELDS = {"final_objective", "success_criteria", "required_outputs", "estimated_steps", "language"}


class TestGoalSchema:
    """Goal object must always have required fields, correct types."""

    def test_goal_has_all_required_fields(self):
        content = json.dumps({
            "final_objective": "AI 논문 트렌드 분석 리포트 생성",
            "success_criteria": ["트렌드 3개 이상 분류", "각 트렌드 요약"],
            "required_outputs": ["pdf_file_path", "notion_page_url"],
            "estimated_steps": 6,
            "language": "ko",
        })
        goal = GoalGenerator.parse_goal_response(content, "test")
        assert GOAL_REQUIRED_FIELDS.issubset(goal.keys())

    def test_goal_success_criteria_is_list(self):
        content = json.dumps({
            "final_objective": "test",
            "success_criteria": ["a", "b"],
            "required_outputs": [],
            "estimated_steps": 1,
            "language": "ko",
        })
        goal = GoalGenerator.parse_goal_response(content, "test")
        assert isinstance(goal["success_criteria"], list)
        assert all(isinstance(c, str) for c in goal["success_criteria"])

    def test_goal_required_outputs_is_list(self):
        content = json.dumps({
            "final_objective": "test",
            "success_criteria": ["a"],
            "required_outputs": ["pdf_path"],
            "estimated_steps": 1,
            "language": "ko",
        })
        goal = GoalGenerator.parse_goal_response(content, "test")
        assert isinstance(goal["required_outputs"], list)

    def test_goal_estimated_steps_is_int(self):
        content = json.dumps({
            "final_objective": "test",
            "success_criteria": ["a"],
            "required_outputs": [],
            "estimated_steps": 4,
            "language": "ko",
        })
        goal = GoalGenerator.parse_goal_response(content, "test")
        assert isinstance(goal["estimated_steps"], int)
        assert goal["estimated_steps"] > 0

    def test_goal_fallback_has_required_fields(self):
        """Even fallback goal must conform to schema."""
        goal = GoalGenerator.parse_goal_response("not json", "원래 질의")
        assert GOAL_REQUIRED_FIELDS.issubset(goal.keys())
        assert isinstance(goal["success_criteria"], list)
        assert isinstance(goal["required_outputs"], list)
        assert isinstance(goal["estimated_steps"], int)

    def test_goal_criteria_status_init(self):
        """criteria_status must have one entry per success_criteria, all 'pending'."""
        goal = {
            "success_criteria": ["분류 완료", "요약 작성", "PDF 생성"],
        }
        status = GoalGenerator.init_criteria_status(goal)
        assert len(status) == 3
        assert all(v == "pending" for v in status.values())
        assert set(status.keys()) == set(goal["success_criteria"])


# ============================================================
# Reflection Schema Validation
# ============================================================

REFLECTION_REQUIRED_FIELDS = {
    "step_completed", "goal_progress", "next_action",
    "deviation_detected", "should_abort", "intent_chain_entry",
}
GOAL_PROGRESS_FIELDS = {"criteria_met", "criteria_remaining", "progress_pct"}


class TestReflectionSchema:
    """Reflection output must have correct structure."""

    def _make_reflection(self, error=False):
        goal = {
            "success_criteria": ["a", "b", "c"],
            "criteria_status": {"a": "done", "b": "pending", "c": "pending"},
        }
        task_graph = {"completed": ["step1"], "pending": ["step2", "step3"]}
        output = {"error": "fail"} if error else {"data": "ok"}
        return Reflector.reflect(
            step_completed="step1",
            step_output=output,
            goal=goal,
            task_graph_status=task_graph,
        )

    def test_reflection_has_all_required_fields(self):
        result = self._make_reflection()
        assert REFLECTION_REQUIRED_FIELDS.issubset(result.keys())

    def test_reflection_goal_progress_structure(self):
        result = self._make_reflection()
        gp = result["goal_progress"]
        assert GOAL_PROGRESS_FIELDS.issubset(gp.keys())
        assert isinstance(gp["criteria_met"], list)
        assert isinstance(gp["criteria_remaining"], list)
        assert isinstance(gp["progress_pct"], int)
        assert 0 <= gp["progress_pct"] <= 100

    def test_reflection_deviation_is_bool(self):
        result = self._make_reflection()
        assert isinstance(result["deviation_detected"], bool)
        assert isinstance(result["should_abort"], bool)

    def test_reflection_intent_chain_entry_is_string(self):
        result = self._make_reflection()
        assert isinstance(result["intent_chain_entry"], str)
        assert "Reflection" in result["intent_chain_entry"]

    def test_reflection_error_sets_deviation(self):
        result = self._make_reflection(error=True)
        assert result["deviation_detected"] is True

    def test_reflection_next_action_type(self):
        result = self._make_reflection()
        assert isinstance(result["next_action"], str)


# ============================================================
# ToolResultCompressor Output Schema
# ============================================================

COMPRESSED_RESULT_REQUIRED_FIELDS = {"structured", "source_pointer"}


class TestToolResultCompressorSchema:
    """Compressed tool results must conform to schema."""

    def test_valid_compression_has_structured_true(self):
        content = json.dumps({
            "items": [{"key": "x", "data": "y", "relevance": "z"}],
            "items_total": 10,
            "items_retained": 1,
            "summary": "test",
        })
        result = ToolResultCompressor.parse_compressed_response(content, {}, "test-id")
        assert result["structured"] is True
        assert "ptr:tool_result:test-id" in result["source_pointer"]

    def test_invalid_compression_has_structured_false(self):
        result = ToolResultCompressor.parse_compressed_response("not json", {}, "test-id")
        assert result["structured"] is False
        assert "source_pointer" in result
        assert "ptr:tool_result:test-id" in result["source_pointer"]

    def test_force_truncate_schema(self):
        result = ToolResultCompressor.force_truncate({"data": "x" * 10000}, "test-id")
        assert COMPRESSED_RESULT_REQUIRED_FIELDS.issubset(result.keys())
        assert result["structured"] is False
        assert "truncated_content" in result
        assert "note" in result

    def test_pointer_format(self):
        """All pointers must follow ptr:{type}:{uuid} format."""
        result = ToolResultCompressor.parse_compressed_response(
            '{"items": [], "summary": "x", "items_total": 0, "items_retained": 0}',
            {}, "550e8400-e29b-41d4-a716-446655440000"
        )
        ptr = result["source_pointer"]
        assert ptr.startswith("ptr:tool_result:")
        # UUID should be present
        assert "550e8400" in ptr


# ============================================================
# ContextManager Output Schema
# ============================================================

class TestContextManagerSchema:
    """Assembled prompts must have correct message structure."""

    def test_assemble_prompt_returns_list_of_dicts(self):
        messages = ContextManager.assemble_prompt(
            system_prompt="You are an assistant.",
            goal={"final_objective": "test", "success_criteria": ["a"], "progress_pct": 0},
            state={"intent_chain": ["step 1 done"]},
            current_step="step2",
            step_instruction="Do the thing",
        )
        assert isinstance(messages, list)
        assert len(messages) >= 1
        for msg in messages:
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ("system", "user", "assistant", "tool")

    def test_assemble_prompt_includes_goal(self):
        messages = ContextManager.assemble_prompt(
            system_prompt="base",
            goal={"final_objective": "find trends", "success_criteria": ["a"], "progress_pct": 50},
            state=None,
            current_step="step1",
            step_instruction="search",
        )
        system_content = messages[0]["content"]
        assert "find trends" in system_content
        assert "50%" in system_content

    def test_assemble_prompt_includes_intent_chain(self):
        messages = ContextManager.assemble_prompt(
            system_prompt="base",
            goal=None,
            state={"intent_chain": ["검색 완료", "분류 완료"]},
            current_step="step3",
            step_instruction="summarize",
        )
        system_content = messages[0]["content"]
        assert "검색 완료" in system_content
        assert "분류 완료" in system_content

    def test_assemble_prompt_includes_relevant_data_with_pointers(self):
        messages = ContextManager.assemble_prompt(
            system_prompt="base",
            goal=None,
            state=None,
            current_step="step1",
            step_instruction="analyze",
            relevant_data=[
                {"ptr": "ptr:chunk:abc123", "desc": "관세 데이터 청크", "content": "관세율 15%"},
            ],
        )
        system_content = messages[0]["content"]
        assert "ptr:chunk:abc123" in system_content
        assert "관세 데이터 청크" in system_content
        assert "관세율 15%" in system_content

    def test_simple_chat_prompt_structure(self):
        messages = ContextManager.build_simple_chat_prompt(
            system_prompt="You are nano-agent.",
            user_message="안녕하세요",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "안녕하세요"


# ============================================================
# HITL Manager Output Schema
# ============================================================

HITL_RESPONSE_FIELDS = {"action", "description", "preview", "confirm_prompt"}


class TestHITLManagerSchema:
    """HITL responses must have correct structure."""

    def test_hitl_response_has_all_fields(self):
        resp = HITLManager.build_hitl_response(
            action="create_notion_page",
            description="노션에 리포트 페이지를 생성합니다.",
            preview={"title": "AI 트렌드 리포트", "sections": 3},
        )
        assert HITL_RESPONSE_FIELDS.issubset(resp.keys())

    def test_hitl_response_types(self):
        resp = HITLManager.build_hitl_response(
            action="publish",
            description="게시합니다",
        )
        assert isinstance(resp["action"], str)
        assert isinstance(resp["description"], str)
        assert isinstance(resp["preview"], dict)
        assert isinstance(resp["confirm_prompt"], str)

    def test_hitl_confirmation_check(self):
        assert HITLManager.is_confirmed(
            {"confirmed": True, "action": "publish"}, "publish"
        )
        assert not HITLManager.is_confirmed(
            {"confirmed": True, "action": "publish"}, "delete"
        )
        assert not HITLManager.is_confirmed(None, "publish")
        assert not HITLManager.is_confirmed(
            {"confirmed": False, "action": "publish"}, "publish"
        )


# ============================================================
# Structured State JSON Schema (integration-level)
# ============================================================

STATE_REQUIRED_SECTIONS = {"goal", "user_intent", "intent_chain", "task_graph",
                           "accumulated_data", "knowledge_context", "token_budget", "hitl_state"}


class TestStructuredStateSchema:
    """Full structured state JSON must have all sections."""

    @staticmethod
    def _build_full_state():
        """Build a complete state JSON as the system would produce it."""
        return {
            "goal": {
                "goal_id": str(uuid.uuid4()),
                "final_objective": "AI 연구논문 트렌드 리포트 PDF 생성",
                "success_criteria": ["트렌드 3개 분류", "요약 작성", "PDF 생성"],
                "required_outputs": ["pdf_file_path"],
                "progress_pct": 33,
                "criteria_status": {
                    "트렌드 3개 분류": "done",
                    "요약 작성": "in_progress",
                    "PDF 생성": "pending",
                },
            },
            "user_intent": {
                "original_request": "AI 연구논문을 검색하고...",
                "intent": "research_and_generate_report",
                "skill": "research_and_report",
                "language": "ko",
            },
            "intent_chain": [
                "사용자가 AI 연구논문 트렌드 리포트를 요청함",
                "웹 검색 완료 (ptr:tool_result:abc = Tavily 검색 10건)",
                "트렌드 분류 완료 [Reflection: Goal 정상 진행]",
            ],
            "task_graph": {
                "status": "running",
                "current_step": "summarize_each",
                "completed": ["search", "classify"],
                "pending": ["summarize_each", "build_pdf"],
            },
            "accumulated_data": {
                "search_results": {
                    "ptr": "ptr:tool_result:550e8400-e29b-41d4-a716-446655440000",
                    "desc": "Tavily 웹검색 결과 10건",
                    "token_count_compressed": 3500,
                    "token_count_raw": 32000,
                    "items_total": 10,
                    "items_retained": 6,
                },
            },
            "knowledge_context": {
                "active_chunk_ids": ["chunk:abc", "chunk:def"],
                "document_refs": [
                    {"doc_id": str(uuid.uuid4()), "filename": "test.pdf", "relevance": 0.92},
                ],
            },
            "token_budget": {
                "model": "qwen3.5:9b",
                "limit": 256000,
                "threshold": 150000,
                "used": 48000,
            },
            "hitl_state": {
                "awaiting": False,
                "pending_action": None,
            },
        }

    def test_state_has_all_required_sections(self):
        state = self._build_full_state()
        assert STATE_REQUIRED_SECTIONS.issubset(state.keys())

    def test_state_goal_section(self):
        state = self._build_full_state()
        goal = state["goal"]
        assert "goal_id" in goal
        assert "final_objective" in goal
        assert "success_criteria" in goal
        assert "progress_pct" in goal
        assert "criteria_status" in goal
        assert isinstance(goal["criteria_status"], dict)

    def test_state_intent_chain_is_list_of_strings(self):
        state = self._build_full_state()
        chain = state["intent_chain"]
        assert isinstance(chain, list)
        assert all(isinstance(e, str) for e in chain)

    def test_state_task_graph_section(self):
        state = self._build_full_state()
        tg = state["task_graph"]
        assert "status" in tg
        assert "current_step" in tg
        assert "completed" in tg
        assert "pending" in tg
        assert isinstance(tg["completed"], list)
        assert isinstance(tg["pending"], list)

    def test_state_accumulated_data_pointers_have_desc(self):
        """Every pointer in accumulated_data must have a 'desc' field."""
        state = self._build_full_state()
        for key, val in state["accumulated_data"].items():
            if isinstance(val, dict) and "ptr" in val:
                assert "desc" in val, f"Pointer {key} missing 'desc' field"
                assert val["ptr"].startswith("ptr:"), f"Pointer format wrong: {val['ptr']}"

    def test_state_token_budget_section(self):
        state = self._build_full_state()
        tb = state["token_budget"]
        assert "model" in tb
        assert "limit" in tb
        assert "threshold" in tb
        assert "used" in tb
        assert isinstance(tb["limit"], int)
        assert tb["threshold"] < tb["limit"]

    def test_state_knowledge_context_chunk_ids(self):
        state = self._build_full_state()
        kc = state["knowledge_context"]
        assert "active_chunk_ids" in kc
        assert isinstance(kc["active_chunk_ids"], list)

    def test_state_hitl_section(self):
        state = self._build_full_state()
        hitl = state["hitl_state"]
        assert "awaiting" in hitl
        assert isinstance(hitl["awaiting"], bool)
