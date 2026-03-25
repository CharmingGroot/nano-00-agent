"""HITLManager — Human-in-the-loop pause/resume at task boundaries."""
import logging
from typing import Any

logger = logging.getLogger(__name__)


class HITLManager:
    """Manages human-in-the-loop confirmations at task boundaries.

    When a task step has hitl_required=True, execution pauses and
    returns a pending_hitl response to the user. Execution resumes
    only when the user confirms via hitl_confirmation in the next request.
    """

    @staticmethod
    def should_pause(task_node: dict) -> bool:
        """Check if a task node requires HITL confirmation."""
        return task_node.get("requires_hitl", False) or task_node.get("hitl_required", False)

    @staticmethod
    def build_hitl_response(
        action: str,
        description: str,
        preview: dict | None = None,
    ) -> dict[str, Any]:
        """Build a HITL pause response to send back to the user."""
        return {
            "action": action,
            "description": description,
            "preview": preview or {},
            "confirm_prompt": f"'{action}' 작업을 진행할까요?",
        }

    @staticmethod
    def is_confirmed(hitl_confirmation: dict | None, expected_action: str) -> bool:
        """Check if the user has confirmed the HITL action."""
        if not hitl_confirmation:
            return False
        return (
            hitl_confirmation.get("confirmed", False)
            and hitl_confirmation.get("action") == expected_action
        )
