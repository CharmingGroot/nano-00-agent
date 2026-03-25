"""ContextManager — assembles prompts with Goal + state + pointers."""
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ContextManager:
    """Assembles the prompt payload for each LLM call.

    Key principle: LLM receives a minimal, focused prompt with:
    1. Goal (always present)
    2. intent_chain (why we're doing this)
    3. Current step instructions
    4. Only the data needed for THIS step (via pointers → resolved chunks)

    NOT the full conversation history.
    """

    @staticmethod
    def assemble_prompt(
        system_prompt: str,
        goal: dict | None,
        state: dict | None,
        current_step: str | None,
        step_instruction: str | None,
        relevant_data: list[dict] | None = None,
        tool_schemas: list[dict] | None = None,
    ) -> list[dict[str, str]]:
        """Assemble a focused prompt for an atomic LLM call.

        Args:
            system_prompt: Base system prompt (role, capabilities)
            goal: Structured Goal object
            state: Current conversation state JSON
            current_step: Name of the current task step
            step_instruction: What the LLM should do in this step
            relevant_data: Resolved chunks/data for this step (already fetched from DB)
            tool_schemas: Tool definitions to include

        Returns:
            List of message dicts ready for LLM
        """
        messages = []

        # 1. System prompt with Goal + context
        system_parts = [system_prompt]

        if goal:
            system_parts.append(
                f"\n## Current Goal\n"
                f"Objective: {goal.get('final_objective', 'N/A')}\n"
                f"Criteria: {json.dumps(goal.get('success_criteria', []), ensure_ascii=False)}\n"
                f"Progress: {goal.get('progress_pct', 0)}%"
            )

        if state:
            intent_chain = state.get("intent_chain", [])
            if intent_chain:
                system_parts.append(
                    f"\n## What has been done so far\n"
                    + "\n".join(f"- {entry}" for entry in intent_chain[-5:])  # Last 5 entries
                )

        if relevant_data:
            system_parts.append("\n## Relevant Data for This Step")
            for item in relevant_data:
                ptr = item.get("ptr", "")
                desc = item.get("desc", "")
                content = item.get("content", "")
                system_parts.append(f"\n[{ptr}] {desc}\n{content}")

        if tool_schemas:
            system_parts.append(
                f"\n## Available Tools\n"
                f"You have access to {len(tool_schemas)} tools. Use them as needed."
            )

        messages.append({"role": "system", "content": "\n".join(system_parts)})

        # 2. User message with step instruction
        if step_instruction:
            messages.append({"role": "user", "content": step_instruction})
        elif current_step:
            messages.append({"role": "user", "content": f"Execute step: {current_step}"})

        return messages

    @staticmethod
    def build_simple_chat_prompt(
        system_prompt: str,
        user_message: str,
        goal: dict | None = None,
        state: dict | None = None,
    ) -> list[dict[str, str]]:
        """Build a simple chat prompt (non-skill, direct conversation)."""
        messages = []

        system_parts = [system_prompt]
        if goal:
            system_parts.append(
                f"\n## Goal: {goal.get('final_objective', 'N/A')}"
            )
        if state and state.get("intent_chain"):
            chain = state["intent_chain"][-3:]
            system_parts.append(
                "\n## Context:\n" + "\n".join(f"- {e}" for e in chain)
            )

        messages.append({"role": "system", "content": "\n".join(system_parts)})
        messages.append({"role": "user", "content": user_message})
        return messages
