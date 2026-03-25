"""Tests for GoalGenerator."""
import json
from src.middleware.goal_generator import GoalGenerator


def test_build_goal_messages():
    """Goal messages should contain system prompt and user query."""
    messages = GoalGenerator.build_goal_messages("AI 논문을 검색해줘")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "AI 논문" in messages[1]["content"]


def test_parse_goal_response_valid_json():
    """Parse a valid JSON goal response."""
    content = json.dumps({
        "final_objective": "AI 논문 트렌드 분석",
        "success_criteria": ["3개 트렌드 분류", "각 트렌드 요약"],
        "required_outputs": ["summary_text"],
        "estimated_steps": 4,
        "language": "ko",
    })
    goal = GoalGenerator.parse_goal_response(content, "AI 논문 검색")
    assert goal["final_objective"] == "AI 논문 트렌드 분석"
    assert len(goal["success_criteria"]) == 2


def test_parse_goal_response_json_in_code_block():
    """Parse JSON wrapped in markdown code block."""
    content = '```json\n{"final_objective": "test", "success_criteria": ["a"]}\n```'
    goal = GoalGenerator.parse_goal_response(content, "test")
    assert goal["final_objective"] == "test"


def test_parse_goal_response_invalid_json():
    """Fallback on invalid JSON."""
    goal = GoalGenerator.parse_goal_response("not valid json", "original query")
    assert goal["final_objective"] == "original query"
    assert goal["success_criteria"] == ["사용자 요청 완료"]


def test_init_criteria_status():
    """Initialize criteria status as all pending."""
    goal = {"success_criteria": ["a", "b", "c"]}
    status = GoalGenerator.init_criteria_status(goal)
    assert all(v == "pending" for v in status.values())
    assert len(status) == 3
