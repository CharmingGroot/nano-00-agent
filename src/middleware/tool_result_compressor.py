"""ToolResultCompressor — Goal-based compression of tool outputs.

Handles 20K-40K token tool results by compressing to <4K tokens.
3-stage strategy:
  Stage 1: Goal-based LLM compression (with JSON output guarantee)
  Stage 2: Force truncation (safety net)
Raw results always saved to DB via pointer.
"""
import json
import logging
from typing import Any

from config.settings import settings
from src.middleware.token_counter import TokenCounter

logger = logging.getLogger(__name__)

# System prompt for compression
COMPRESS_SYSTEM_PROMPT = """You are a data compressor. Given a Goal and raw tool output, extract ONLY the information relevant to the Goal.

Return ONLY a valid JSON object with this schema:
{
  "items": [
    {"key": "short identifier", "data": "extracted fact/value", "relevance": "why this matters for the goal"}
  ],
  "items_total": <number of items in raw input>,
  "items_retained": <number of items in compressed output>,
  "summary": "one-line summary of what was found"
}

Rules:
- ONLY include items directly relevant to the Goal
- Extract facts, numbers, dates — not background text
- Keep each item's data under 100 words
- Return valid JSON only, no explanation"""


class ToolResultCompressor:
    """Compresses tool results based on Goal relevance.

    Stage 1: LLM compression (with retry + forced wrapping on failure)
    Stage 2: Force truncation if still over hard_limit
    """

    @staticmethod
    def needs_compression(result: Any) -> bool:
        """Check if a tool result exceeds the soft limit."""
        text = json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result)
        token_count = TokenCounter.count_tokens(text)
        return token_count >= settings.tool_result_soft_limit

    @staticmethod
    def get_token_count(result: Any) -> int:
        """Count tokens in a tool result."""
        text = json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result)
        return TokenCounter.count_tokens(text)

    @staticmethod
    def build_compression_messages(goal: dict, raw_result: Any) -> list[dict[str, str]]:
        """Build messages for the compression LLM call."""
        result_text = json.dumps(raw_result, ensure_ascii=False) if isinstance(raw_result, dict) else str(raw_result)
        return [
            {"role": "system", "content": COMPRESS_SYSTEM_PROMPT},
            {"role": "user", "content": f"## Goal\n{json.dumps(goal, ensure_ascii=False)}\n\n## Raw Tool Output\n{result_text}"},
        ]

    @staticmethod
    def parse_compressed_response(content: str, raw_result: Any, pointer_id: str) -> dict[str, Any]:
        """Parse LLM compression response. 3-attempt strategy:
        1. Parse as JSON
        2. (caller retries with LLM)
        3. Force wrap as unstructured
        """
        try:
            cleaned = content.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1])
            compressed = json.loads(cleaned)
            compressed["structured"] = True
            compressed["source_pointer"] = f"ptr:tool_result:{pointer_id}"
            return compressed
        except (json.JSONDecodeError, KeyError):
            # Force wrap — data preserved but structure flagged as broken
            return {
                "compressed_text": content[:2000],  # Truncate if needed
                "structured": False,
                "source_pointer": f"ptr:tool_result:{pointer_id}",
                "note": "LLM compression output was not valid JSON, raw text preserved",
            }

    @staticmethod
    def force_truncate(result: Any, pointer_id: str) -> dict[str, Any]:
        """Stage 2: Force truncation when over hard_limit."""
        text = json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result)
        # Take first hard_limit worth of content
        truncated = text[:settings.tool_result_hard_limit * 4]  # rough char-to-token ratio
        token_count = TokenCounter.count_tokens(truncated)

        return {
            "truncated_content": truncated,
            "structured": False,
            "source_pointer": f"ptr:tool_result:{pointer_id}",
            "note": f"Force truncated to ~{token_count} tokens. Full result at pointer.",
            "items_total": "unknown",
            "items_retained": "partial",
        }
