"""Tests for Reflector."""
from src.middleware.reflector import Reflector


def test_reflect_normal_progress():
    """Normal progress reflection."""
    goal = {
        "success_criteria": ["트렌드 분류", "요약 작성", "PDF 생성"],
        "criteria_status": {
            "트렌드 분류": "done",
            "요약 작성": "pending",
            "PDF 생성": "pending",
        },
    }
    task_graph = {
        "completed": ["search", "classify"],
        "pending": ["summarize", "build_pdf"],
    }
    result = Reflector.reflect(
        step_completed="classify",
        step_output={"trends": ["a", "b"]},
        goal=goal,
        task_graph_status=task_graph,
    )
    assert result["deviation_detected"] is False
    assert result["should_abort"] is False
    assert result["goal_progress"]["progress_pct"] == 50
    assert "Reflection" in result["intent_chain_entry"]


def test_reflect_with_error():
    """Deviation detected on error output."""
    goal = {
        "success_criteria": ["완료"],
        "criteria_status": {"완료": "pending"},
    }
    task_graph = {
        "completed": [],
        "pending": ["search"],
    }
    result = Reflector.reflect(
        step_completed="search",
        step_output={"error": "connection timeout"},
        goal=goal,
        task_graph_status=task_graph,
    )
    assert result["deviation_detected"] is True
