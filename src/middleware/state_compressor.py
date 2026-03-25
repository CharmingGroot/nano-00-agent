"""StateCompressor — compresses conversation state to structured JSON."""
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

COMPRESS_STATE_PROMPT = """You are a state compressor. Given the current conversation state and recent messages,
produce an updated structured state JSON.

Rules:
- Keep the EXACT same JSON schema as input
- Update intent_chain: add entries for what happened, remove old entries if > 10
- Update accumulated_data: keep only what's needed for remaining steps
- Update token_budget with new values
- Keep all pointers (ptr:...) — never remove pointer references
- Do NOT include raw data — only pointers and summaries
- Return ONLY valid JSON"""


class StateCompressor:
    """Compresses conversation state when approaching token limits.

    Called by LLMGateway when TokenCounter detects threshold breach.
    Uses an LLM call (flagged as is_compression=True to prevent recursion).
    """

    @staticmethod
    def build_compression_messages(
        current_state: dict,
        recent_messages: list[dict],
        token_budget_used: int,
    ) -> list[dict[str, str]]:
        """Build messages for the state compression LLM call."""
        return [
            {"role": "system", "content": COMPRESS_STATE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"## Current State\n{json.dumps(current_state, ensure_ascii=False)}\n\n"
                    f"## Recent Messages (last 5)\n{json.dumps(recent_messages[-5:], ensure_ascii=False)}\n\n"
                    f"## Token Budget Used: {token_budget_used}\n\n"
                    f"Produce the updated state JSON."
                ),
            },
        ]

    @staticmethod
    def parse_compressed_state(content: str, fallback_state: dict) -> dict:
        """Parse LLM compression response into state dict."""
        try:
            cleaned = content.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1])
            return json.loads(cleaned)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to parse compressed state: %s", e)
            return fallback_state
