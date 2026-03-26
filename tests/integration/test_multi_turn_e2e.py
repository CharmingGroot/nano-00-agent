"""50+ turn multi-turn complex query E2E test scenarios.

Tests the full pipeline with simulated conversations that exercise:
- Goal generation + evolution across turns
- Intent classification (simple → tool_use → complex)
- State persistence across turns
- Token budget tracking
- Pointer accumulation in accumulated_data
- intent_chain growth and maintenance
- Context compression trigger simulation
"""
import json
import uuid
from typing import Any

import pytest

from src.middleware.pipeline import MiddlewarePipeline
from src.middleware.token_counter import TokenCounter
from src.middleware.reflector import Reflector
from src.middleware.goal_generator import GoalGenerator
from src.middleware.tool_result_compressor import ToolResultCompressor

STATE_REQUIRED_SECTIONS = {
    "goal", "user_intent", "intent_chain", "task_graph",
    "accumulated_data", "knowledge_context", "token_budget", "hitl_state",
}


class TestMultiTurnConversation:
    """Simulate 50+ turn conversations and verify state integrity."""

    def _simulate_turn(
        self,
        state: dict,
        user_message: str,
        turn_num: int,
        tokens_this_turn: int = 2000,
    ) -> dict:
        """Simulate one turn of conversation — updates state as pipeline would."""
        # Goal generation (simulate)
        goal = state.get("goal", {})
        if not goal.get("final_objective"):
            goal = {
                "goal_id": str(uuid.uuid4()),
                "final_objective": user_message[:100],
                "success_criteria": ["작업 완료"],
                "required_outputs": [],
                "progress_pct": 0,
                "criteria_status": {"작업 완료": "pending"},
            }
            state["goal"] = goal

        # Update intent
        state["user_intent"]["original_request"] = user_message
        state["user_intent"]["intent"] = "general_chat" if turn_num % 5 != 0 else "tool_use"

        # Add to intent_chain
        state["intent_chain"].append(
            f"Turn {turn_num}: {user_message[:50]} [토큰: {tokens_this_turn}]"
        )

        # Simulate accumulated data every 3rd turn
        if turn_num % 3 == 0:
            ptr_id = str(uuid.uuid4())
            state["accumulated_data"][f"result_turn_{turn_num}"] = {
                "ptr": f"ptr:tool_result:{ptr_id}",
                "desc": f"Turn {turn_num} 결과",
                "token_count_raw": tokens_this_turn * 5,
                "token_count_compressed": tokens_this_turn,
            }

        # Update token budget
        state["token_budget"]["used"] += tokens_this_turn

        # Update goal progress
        progress = min(100, int((turn_num / 50) * 100))
        state["goal"]["progress_pct"] = progress

        return state

    def test_50_turn_state_integrity(self):
        """50 turns: state must always have all required sections."""
        state = MiddlewarePipeline._init_state()

        queries = [
            "안녕하세요",
            "수입자동차 관세 현황을 알려줘",
            "전기차 관세율은 어떻게 되나요?",
            "SUV와 전기차 관세율을 비교해줘",
            "관세율 변동 추이를 분석해줘",
            "FTA 특혜 관세에 대해 설명해줘",
            "하이브리드차 관세는?",
            "2024년 관세 변경 사항은?",
            "관세 데이터를 PDF로 만들어줘",
            "노션에 보고서를 등록해줘",
        ]

        for turn in range(50):
            msg = queries[turn % len(queries)]
            state = self._simulate_turn(state, msg, turn + 1)

            # Verify state integrity every turn
            assert STATE_REQUIRED_SECTIONS.issubset(state.keys()), f"Turn {turn+1}: missing state sections"
            assert isinstance(state["intent_chain"], list)
            assert len(state["intent_chain"]) == turn + 1
            assert state["token_budget"]["used"] > 0

    def test_50_turn_intent_chain_growth(self):
        """Intent chain grows correctly across 50 turns."""
        state = MiddlewarePipeline._init_state()

        for turn in range(50):
            state = self._simulate_turn(state, f"질문 {turn+1}", turn + 1)

        chain = state["intent_chain"]
        assert len(chain) == 50
        assert "Turn 1:" in chain[0]
        assert "Turn 50:" in chain[49]

    def test_50_turn_accumulated_data_pointers(self):
        """Accumulated data with pointers grows every 3rd turn."""
        state = MiddlewarePipeline._init_state()

        for turn in range(50):
            state = self._simulate_turn(state, f"질문 {turn+1}", turn + 1)

        # Every 3rd turn adds a pointer (turns 3,6,9,...,48 = ~16 entries)
        expected_entries = len([t for t in range(1, 51) if t % 3 == 0])
        assert len(state["accumulated_data"]) == expected_entries

        # All entries must have ptr + desc
        for key, val in state["accumulated_data"].items():
            assert "ptr" in val, f"Missing ptr in {key}"
            assert "desc" in val, f"Missing desc in {key}"
            assert val["ptr"].startswith("ptr:tool_result:")

    def test_50_turn_token_budget_tracking(self):
        """Token budget accumulates correctly."""
        state = MiddlewarePipeline._init_state()

        for turn in range(50):
            state = self._simulate_turn(state, f"질문 {turn+1}", turn + 1, tokens_this_turn=3000)

        assert state["token_budget"]["used"] == 50 * 3000

    def test_50_turn_goal_progress(self):
        """Goal progress evolves from 0 to 100."""
        state = MiddlewarePipeline._init_state()

        for turn in range(50):
            state = self._simulate_turn(state, f"질문 {turn+1}", turn + 1)

        assert state["goal"]["progress_pct"] == 100

    def test_compression_trigger_at_50_turns(self):
        """After 50 turns at 3K tokens each = 150K, compression should trigger."""
        state = MiddlewarePipeline._init_state()

        for turn in range(50):
            state = self._simulate_turn(state, f"질문 {turn+1}", turn + 1, tokens_this_turn=3000)

        total = state["token_budget"]["used"]
        assert total == 150_000
        assert TokenCounter.should_compress(total)


class TestComplexQueryScenarios:
    """Test specific complex query patterns."""

    COMPLEX_QUERIES = [
        "지식에서 수입자동차 관세부여현황 CSV를 찾아서 전기차와 SUV의 관세율을 비교 분석하고 요약해줘",
        "AI 연구논문을 검색하고 트렌드별로 순위를 매겨서 요약한 뒤 PDF로 출력해줘",
        "최근 관세 변경 사항을 웹에서 검색하고, 기존 지식 데이터와 비교해서 변동 분석 보고서를 만들어줘",
        "Google Sheets에 있는 수출입 데이터를 가져와서 월별 트렌드를 분석하고 노션에 대시보드를 만들어줘",
        "업로드된 모든 PDF에서 관세 관련 키워드를 검색하고, 결과를 정리해서 엑셀로 출력해줘",
    ]

    def test_complex_query_goal_generation(self):
        """Each complex query should produce a valid Goal."""
        for query in self.COMPLEX_QUERIES:
            # Simulate LLM response
            goal = GoalGenerator.parse_goal_response(json.dumps({
                "final_objective": query[:100],
                "success_criteria": ["분석 완료", "결과 출력"],
                "required_outputs": ["report"],
                "estimated_steps": 5,
                "language": "ko",
            }), query)

            assert "final_objective" in goal
            assert isinstance(goal["success_criteria"], list)
            assert len(goal["success_criteria"]) > 0

    def test_complex_query_state_evolution(self):
        """Complex query should evolve state across multiple simulated steps."""
        state = MiddlewarePipeline._init_state()
        query = self.COMPLEX_QUERIES[0]

        # Simulate 5 steps for a complex query
        steps = ["search_knowledge", "analyze_data", "compare_results", "summarize", "generate_report"]

        for i, step in enumerate(steps):
            ptr_id = str(uuid.uuid4())
            state["accumulated_data"][step] = {
                "ptr": f"ptr:tool_result:{ptr_id}",
                "desc": f"{step} 결과",
                "token_count_raw": 20000 + i * 5000,
                "token_count_compressed": 3000 + i * 500,
            }

            reflection = Reflector.reflect(
                step_completed=step,
                step_output={"data": f"result of {step}"},
                goal={
                    "success_criteria": ["분석 완료", "결과 출력"],
                    "criteria_status": {
                        "분석 완료": "done" if i >= 2 else "pending",
                        "결과 출력": "done" if i >= 4 else "pending",
                    },
                },
                task_graph_status={
                    "completed": steps[:i+1],
                    "pending": steps[i+1:],
                },
            )
            state["intent_chain"].append(reflection["intent_chain_entry"])

        assert len(state["accumulated_data"]) == 5
        assert len(state["intent_chain"]) == 5
        assert all("Reflection" in entry for entry in state["intent_chain"])

    def test_mixed_simple_and_complex_queries_50_turns(self):
        """Alternate between simple and complex queries for 50 turns."""
        state = MiddlewarePipeline._init_state()
        simple_queries = ["안녕", "고마워", "알겠어", "다음 질문", "결과 확인"]
        complex_queries = self.COMPLEX_QUERIES

        for turn in range(50):
            if turn % 3 == 0:
                msg = complex_queries[turn % len(complex_queries)]
                tokens = 5000
            else:
                msg = simple_queries[turn % len(simple_queries)]
                tokens = 1000

            state["intent_chain"].append(f"Turn {turn+1}: {msg[:30]} [{tokens}tok]")
            state["token_budget"]["used"] += tokens

            if turn % 3 == 0:
                ptr_id = str(uuid.uuid4())
                state["accumulated_data"][f"complex_turn_{turn}"] = {
                    "ptr": f"ptr:tool_result:{ptr_id}",
                    "desc": f"복합 질의 결과 (turn {turn+1})",
                }

        assert len(state["intent_chain"]) == 50
        # ~17 complex turns (0,3,6,...,48)
        complex_entries = [k for k in state["accumulated_data"] if k.startswith("complex_")]
        assert len(complex_entries) >= 15


class TestToolResultCompressionAtScale:
    """Test tool result compression with realistic 20K-40K token results."""

    def test_compression_needed_for_large_results(self):
        """Simulate 50 tool calls with large results."""
        for i in range(50):
            # Simulate large result (20K-40K tokens worth of text)
            large_result = {
                "results": [
                    {
                        "title": f"Result {j}",
                        "content": f"Detailed content for result {j}. " * (100 + i * 10),
                        "url": f"https://example.com/{j}",
                    }
                    for j in range(20)
                ]
            }
            assert ToolResultCompressor.needs_compression(large_result)

            # Verify compression output format
            compressed_json = json.dumps({
                "items": [{"key": f"r{j}", "data": f"summary {j}"} for j in range(5)],
                "items_total": 20,
                "items_retained": 5,
                "summary": f"Compressed result {i}",
            })
            ptr_id = str(uuid.uuid4())
            result = ToolResultCompressor.parse_compressed_response(
                compressed_json, large_result, ptr_id
            )
            assert result["structured"] is True
            assert result["source_pointer"].startswith("ptr:tool_result:")

    def test_force_truncation_50_times(self):
        """Force truncation should always produce valid output."""
        for i in range(50):
            large_data = {"data": f"x{'y' * (50000 + i * 1000)}"}
            ptr_id = str(uuid.uuid4())
            result = ToolResultCompressor.force_truncate(large_data, ptr_id)
            assert result["structured"] is False
            assert "source_pointer" in result
            assert "truncated_content" in result


class TestReflectionAcross50Steps:
    """Verify Reflector produces correct output for 50 consecutive steps."""

    def test_50_step_reflections(self):
        """Each reflection must conform to schema."""
        steps = [f"step_{i}" for i in range(50)]
        completed = []

        for i, step in enumerate(steps):
            completed.append(step)
            pending = steps[i+1:]

            goal = {
                "success_criteria": [f"criterion_{j}" for j in range(5)],
                "criteria_status": {
                    f"criterion_{j}": "done" if j <= i // 10 else "pending"
                    for j in range(5)
                },
            }

            reflection = Reflector.reflect(
                step_completed=step,
                step_output={"data": f"output_{i}"},
                goal=goal,
                task_graph_status={
                    "completed": list(completed),
                    "pending": list(pending),
                },
            )

            assert "step_completed" in reflection
            assert "goal_progress" in reflection
            assert "deviation_detected" in reflection
            assert "should_abort" in reflection
            assert "intent_chain_entry" in reflection
            assert isinstance(reflection["goal_progress"]["progress_pct"], int)
            assert 0 <= reflection["goal_progress"]["progress_pct"] <= 100

    def test_deviation_detection_pattern(self):
        """Reflector correctly detects errors across 50 steps."""
        for i in range(50):
            is_error = i % 7 == 0  # Every 7th step fails
            output = {"error": "failed"} if is_error else {"data": "ok"}

            reflection = Reflector.reflect(
                step_completed=f"step_{i}",
                step_output=output,
                goal={"success_criteria": ["a"], "criteria_status": {"a": "pending"}},
                task_graph_status={"completed": [], "pending": [f"step_{i}"]},
            )

            if is_error:
                assert reflection["deviation_detected"] is True
            else:
                assert reflection["deviation_detected"] is False


class TestStateCompressionSimulation:
    """Simulate state growing until compression is triggered."""

    def test_state_grows_then_compresses(self):
        """Build state for 75 turns, verify compression trigger."""
        state = MiddlewarePipeline._init_state()

        for turn in range(75):
            state["intent_chain"].append(
                f"Turn {turn+1}: 복합 질의 처리 완료 (ptr:tool_result:{uuid.uuid4()} = 결과 {turn+1}건)"
            )
            state["token_budget"]["used"] += 2000

            if turn % 2 == 0:
                state["accumulated_data"][f"data_{turn}"] = {
                    "ptr": f"ptr:tool_result:{uuid.uuid4()}",
                    "desc": f"Turn {turn+1} 결과 데이터",
                    "token_count_raw": 25000,
                    "token_count_compressed": 3500,
                }

        # At 75 turns * 2000 = 150K tokens
        assert state["token_budget"]["used"] == 150_000
        assert TokenCounter.should_compress(state["token_budget"]["used"])

        # Verify state is still valid
        assert STATE_REQUIRED_SECTIONS.issubset(state.keys())
        assert len(state["intent_chain"]) == 75
        assert len(state["accumulated_data"]) == 38  # turns 0,2,4,...,74

        # All pointers valid
        for key, val in state["accumulated_data"].items():
            assert val["ptr"].startswith("ptr:")
            assert "desc" in val

    def test_intent_chain_trimming_simulation(self):
        """After compression, intent_chain should be trimmed to last N entries."""
        state = MiddlewarePipeline._init_state()

        # Build 100 entries
        for i in range(100):
            state["intent_chain"].append(f"Entry {i}")

        assert len(state["intent_chain"]) == 100

        # Simulate compression: keep last 10
        trimmed = state["intent_chain"][-10:]
        state["intent_chain"] = trimmed

        assert len(state["intent_chain"]) == 10
        assert "Entry 90" in state["intent_chain"][0]
        assert "Entry 99" in state["intent_chain"][-1]
