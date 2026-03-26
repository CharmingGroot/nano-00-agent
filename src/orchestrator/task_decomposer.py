"""TaskDecomposer — generates a task graph from intent classification.

If a matching skill exists, loads its step DAG directly.
Otherwise, uses the LLM to generate an ad-hoc task plan.
"""
import json
import logging
from typing import Any

from src.middleware.llm_gateway import LLMGateway, LLMRequest
from src.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

DECOMPOSE_SYSTEM_PROMPT = """\
You are a task planner. Given a user request and the list of available tools, \
produce a step-by-step execution plan as a JSON array.

Available tools: {tools}

Respond with ONLY a JSON object (no markdown, no explanation):
{{
  "steps": [
    {{
      "id": "step_1",
      "tool": "<tool_name>",
      "args": {{<arguments, may reference prior steps as {{{{steps.step_N.output}}}}>}},
      "depends_on": []
    }},
    ...
  ]
}}

Rules:
- Each step must use one of the available tools.
- Use {{{{steps.<id>.output.<key>}}}} to reference previous step outputs.
- depends_on lists step IDs that must complete first.
- Keep the plan minimal — only the steps truly required.
"""


class TaskDecomposer:
    """Decompose a user request into a TaskGraph-compatible structure."""

    def __init__(
        self,
        gateway: LLMGateway,
        skill_registry: SkillRegistry,
    ) -> None:
        self._gateway = gateway
        self._skill_registry = skill_registry

    async def decompose(
        self,
        intent: dict[str, Any],
        user_message: str,
        available_tools: list[str],
    ) -> dict[str, Any]:
        """Return a task graph structure.

        If intent indicates a skill match, loads from DB cache.
        Otherwise, asks the LLM to plan ad-hoc steps.

        Returns:
            {
                "source": "skill" | "llm",
                "skill_name": str | None,
                "steps": [{"id", "tool", "args", "depends_on", "loop_over"?}, ...],
                "parameters": dict,
            }
        """
        skill_name = intent.get("skill")

        # 1. Try to use an existing skill
        if skill_name:
            skill = self._skill_registry.get_skill(skill_name)
            if skill:
                logger.info("TaskDecomposer: using skill '%s'", skill_name)
                return {
                    "source": "skill",
                    "skill_name": skill_name,
                    "steps": skill["steps"],
                    "parameters": intent.get("parameters", {}),
                }

        # 2. No matching skill — ask LLM
        logger.info("TaskDecomposer: generating ad-hoc plan via LLM")
        return await self._plan_with_llm(user_message, available_tools, intent.get("parameters", {}))

    async def _plan_with_llm(
        self,
        user_message: str,
        available_tools: list[str],
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Ask the LLM to decompose the request into steps."""
        tools_str = ", ".join(available_tools) if available_tools else "(none)"
        system_prompt = DECOMPOSE_SYSTEM_PROMPT.format(tools=tools_str)

        request = LLMRequest(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )

        response = await self._gateway.chat(request)
        steps = self._parse_plan(response.content)

        return {
            "source": "llm",
            "skill_name": None,
            "steps": steps,
            "parameters": parameters,
        }

    @staticmethod
    def _parse_plan(raw: str) -> list[dict[str, Any]]:
        """Parse the LLM plan output into a steps list."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            first_nl = cleaned.index("\n")
            cleaned = cleaned[first_nl + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("TaskDecomposer: failed to parse LLM plan: %s", raw[:300])
            return []

        steps = parsed.get("steps", parsed) if isinstance(parsed, dict) else parsed
        if not isinstance(steps, list):
            return []

        # Validate each step has at least id and tool
        valid_steps = []
        for s in steps:
            if isinstance(s, dict) and "id" in s and "tool" in s:
                step = {
                    "id": s["id"],
                    "tool": s["tool"],
                    "args": s.get("args", {}),
                    "depends_on": s.get("depends_on", []),
                }
                if "loop_over" in s:
                    step["loop_over"] = s["loop_over"]
                valid_steps.append(step)

        return valid_steps
