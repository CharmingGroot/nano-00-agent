"""Reflector — evaluates progress against Goal after each tool call.

This is middleware logic (not an LLM call) that checks task_graph progress
against Goal.success_criteria and detects deviations.
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)


class Reflector:
    """Evaluates Goal progress after each tool call completion.

    Not an LLM call — uses structured data comparison.
    Updates intent_chain and criteria_status.
    """

    @staticmethod
    def reflect(
        step_completed: str,
        step_output: Any,
        goal: dict,
        task_graph_status: dict,
    ) -> dict[str, Any]:
        """Produce a reflection after a step completes.

        Returns:
            {
                "step_completed": str,
                "goal_progress": {
                    "criteria_met": [...],
                    "criteria_remaining": [...],
                    "progress_pct": int
                },
                "next_action": str,
                "deviation_detected": bool,
                "should_abort": bool,
                "intent_chain_entry": str
            }
        """
        completed_steps = task_graph_status.get("completed", [])
        pending_steps = task_graph_status.get("pending", [])
        total_steps = len(completed_steps) + len(pending_steps)
        progress_pct = int((len(completed_steps) / max(total_steps, 1)) * 100)

        # Check criteria against what we know
        criteria = goal.get("success_criteria", [])
        criteria_status = goal.get("criteria_status", {})
        criteria_met = [c for c, s in criteria_status.items() if s == "done"]
        criteria_remaining = [c for c, s in criteria_status.items() if s != "done"]

        # Detect deviation: if step failed or output is empty/error
        deviation = False
        should_abort = False
        if isinstance(step_output, dict):
            if step_output.get("error"):
                deviation = True
                # Abort if critical step failed
                if step_completed in (task_graph_status.get("pending", [])[:1]):
                    should_abort = True

        # Determine next action
        next_action = pending_steps[0] if pending_steps else "complete"

        # Build intent chain entry
        status_tag = "Goal 정상 진행" if not deviation else "Goal 이탈 감지"
        intent_entry = (
            f"{step_completed} 완료 → {progress_pct}% 진행 "
            f"[Reflection: {status_tag}]"
        )

        return {
            "step_completed": step_completed,
            "goal_progress": {
                "criteria_met": criteria_met,
                "criteria_remaining": criteria_remaining,
                "progress_pct": progress_pct,
            },
            "next_action": next_action,
            "deviation_detected": deviation,
            "should_abort": should_abort,
            "intent_chain_entry": intent_entry,
        }
