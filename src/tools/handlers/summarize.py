"""Summarize tool handler — wraps LLMGateway.chat()."""
from typing import Any

from src.tools.base import BaseTool
from src.middleware.llm_gateway import LLMGateway, LLMRequest


class SummarizeHandler(BaseTool):
    """Summarize content using the LLM."""

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute a summarization.

        Keyword Args:
            content: The text to summarize (required).
            instruction: Custom instruction (default: "Summarize the key points concisely.").
            output_format: "text" or "json" (default: "text").

        Returns:
            {"summary": "<summary text>"}
        """
        content: str = kwargs["content"]
        instruction: str = kwargs.get("instruction", "Summarize the key points concisely.")
        output_format: str = kwargs.get("output_format", "text")

        system_msg = instruction
        if output_format == "json":
            system_msg += "\nReturn your answer as a valid JSON object."

        request = LLMRequest(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": content},
            ],
        )

        gateway = LLMGateway()
        try:
            response = await gateway.chat(request)
        finally:
            await gateway.close()

        return {"summary": response.content}
