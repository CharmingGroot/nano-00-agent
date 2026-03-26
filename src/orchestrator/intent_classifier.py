"""IntentClassifier — LLM-based intent classification.

Takes user message + available skills/tools and returns structured intent info.
"""
import json
import logging
from typing import Any

from src.middleware.llm_gateway import LLMGateway, LLMRequest

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """\
You are an intent classifier. Given the user's message, classify the intent.

This agent has an INTERNAL KNOWLEDGE BASE with uploaded documents.
When users ask about specific data, facts, or documents, prefer internal tools over web search.

Available skills (pre-built workflows):
{skills_section}

Available tools (atomic actions):
{tools_section}

Respond with ONLY a JSON object (no markdown, no explanation) in this exact format:
{{
  "intent": "<one of: skill_match | tool_use | chitchat | clarification_needed>",
  "skill": "<matched skill name or null>",
  "required_tools": ["<tool_name>", ...],
  "complexity": "<one of: simple | moderate | complex>",
  "parameters": {{<extracted parameters from user message>}}
}}

Rules:
- If the message clearly maps to an existing skill, set intent="skill_match" and specify the skill name.
- If no skill matches but tools are needed, set intent="tool_use" and list the tools.
- Prefer knowledge base search tools over web search when the question is about specific data.
- If it's a casual/general question with no data need, set intent="chitchat".
- If the message is ambiguous, set intent="clarification_needed".
- Always extract any parameters you can from the message.
"""


class IntentClassifier:
    """Classify user intent using a single LLM call."""

    def __init__(self, gateway: LLMGateway) -> None:
        self._gateway = gateway

    async def classify(
        self,
        user_message: str,
        available_skills: list[dict[str, Any]],
        available_tools: list[dict[str, Any]] | list[str],
    ) -> dict[str, Any]:
        """Classify the user message and return structured intent.

        Args:
            user_message: The raw user message.
            available_skills: List of skill dicts with at least 'name' and 'description'.
            available_tools: List of tool dicts {name, description} or plain name strings.

        Returns:
            {
                "intent": str,        # skill_match | tool_use | chitchat | clarification_needed
                "skill": str | None,
                "required_tools": list[str],
                "complexity": str,     # simple | moderate | complex
                "parameters": dict
            }
        """
        skills_section = self._format_skills(available_skills)
        tools_section = self._format_tools(available_tools)

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            skills_section=skills_section,
            tools_section=tools_section,
        )

        request = LLMRequest(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )

        response = await self._gateway.chat(request)
        return self._parse_response(response.content)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_tools(tools: list[dict[str, Any]] | list[str]) -> str:
        """Format tool list with descriptions for the LLM."""
        if not tools:
            return "(none)"
        lines = []
        for t in tools:
            if isinstance(t, dict):
                name = t.get("name", "unknown")
                desc = t.get("description", "")
                lines.append(f"- {name}: {desc[:150]}")
            else:
                lines.append(f"- {t}")
        return "\n".join(lines)

    @staticmethod
    def _format_skills(skills: list[dict[str, Any]]) -> str:
        if not skills:
            return "(none)"
        lines = []
        for s in skills:
            name = s.get("name", "unknown")
            desc = s.get("description", "")
            triggers = s.get("triggers", {})
            lines.append(f"- {name}: {desc}")
            if triggers:
                lines.append(f"  triggers: {json.dumps(triggers)}")
        return "\n".join(lines)

    @staticmethod
    def _parse_response(raw: str) -> dict[str, Any]:
        """Parse the LLM's JSON response, with fallback for malformed output."""
        # Strip markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Remove opening fence
            first_newline = cleaned.index("\n")
            cleaned = cleaned[first_newline + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        default = {
            "intent": "clarification_needed",
            "skill": None,
            "required_tools": [],
            "complexity": "simple",
            "parameters": {},
        }

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("IntentClassifier: failed to parse LLM response as JSON: %s", raw[:200])
            return default

        # Validate required keys
        valid_intents = {"skill_match", "tool_use", "chitchat", "clarification_needed"}
        valid_complexity = {"simple", "moderate", "complex"}

        result = {
            "intent": parsed.get("intent") if parsed.get("intent") in valid_intents else "clarification_needed",
            "skill": parsed.get("skill"),
            "required_tools": parsed.get("required_tools", []),
            "complexity": parsed.get("complexity") if parsed.get("complexity") in valid_complexity else "simple",
            "parameters": parsed.get("parameters", {}),
        }

        # Ensure required_tools is a list
        if not isinstance(result["required_tools"], list):
            result["required_tools"] = []

        # Ensure parameters is a dict
        if not isinstance(result["parameters"], dict):
            result["parameters"] = {}

        return result
