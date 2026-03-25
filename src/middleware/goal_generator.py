"""GoalGenerator — creates structured Goal objects from user queries."""
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# System prompt for Goal generation
GOAL_SYSTEM_PROMPT = """You are a goal analyzer. Given a user's request, extract a structured goal.

Return ONLY a JSON object with this exact schema:
{
  "final_objective": "one-line description of what the user ultimately wants",
  "success_criteria": ["criterion 1", "criterion 2", ...],
  "required_outputs": ["output_1", "output_2", ...],
  "estimated_steps": <integer>,
  "language": "ko or en"
}

Rules:
- success_criteria: measurable conditions that must be true when the task is complete
- required_outputs: concrete deliverables (files, pages, data)
- estimated_steps: rough count of discrete actions needed
- Keep it concise. Do NOT explain, just return JSON."""


class GoalGenerator:
    """Generates a structured Goal object from a user query.

    Called once at the start of each user request.
    The Goal object is then used throughout the pipeline for:
    - ToolResultCompressor: filter irrelevant results
    - Reflector: evaluate progress
    - ContextManager: include goal in every LLM prompt
    """

    @staticmethod
    def build_goal_messages(user_query: str) -> list[dict[str, str]]:
        """Build messages for Goal generation LLM call."""
        return [
            {"role": "system", "content": GOAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]

    @staticmethod
    def parse_goal_response(content: str, user_query: str) -> dict[str, Any]:
        """Parse LLM response into a Goal dict. Falls back to minimal goal on failure."""
        try:
            # Try to extract JSON from response
            # Handle cases where LLM wraps in ```json ... ```
            cleaned = content.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1])
            goal = json.loads(cleaned)
            # Validate required fields
            assert "final_objective" in goal
            assert "success_criteria" in goal
            return goal
        except (json.JSONDecodeError, AssertionError, KeyError) as e:
            logger.warning("Failed to parse Goal from LLM response: %s", e)
            # Fallback: minimal goal
            return {
                "final_objective": user_query[:200],
                "success_criteria": ["사용자 요청 완료"],
                "required_outputs": [],
                "estimated_steps": 1,
                "language": "ko",
            }

    @staticmethod
    def init_criteria_status(goal: dict) -> dict[str, str]:
        """Initialize criteria_status from success_criteria."""
        return {c: "pending" for c in goal.get("success_criteria", [])}
