"""Token counter — tracks token usage per conversation."""
import logging

import tiktoken

from config.settings import settings

logger = logging.getLogger(__name__)

# Use cl100k_base as a reasonable approximation for token counting
# Actual token counts come from Ollama response, this is for pre-flight estimation
_encoder = tiktoken.get_encoding("cl100k_base")


class TokenCounter:
    """Tracks and estimates token usage."""

    @staticmethod
    def count_tokens(text: str) -> int:
        """Estimate token count for a string."""
        return len(_encoder.encode(text))

    @staticmethod
    def count_messages_tokens(messages: list[dict]) -> int:
        """Estimate total tokens for a list of chat messages."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += TokenCounter.count_tokens(content)
            # Add overhead per message (role tokens, formatting)
            total += 4
        return total

    @staticmethod
    def should_compress(total_tokens: int, model: str = "qwen3.5:9b") -> bool:
        """Check if we've hit the safety threshold and need compression."""
        threshold = settings.token_safety_threshold
        over = total_tokens >= threshold
        if over:
            logger.warning(
                "Token safety threshold reached: %d >= %d (model=%s)",
                total_tokens, threshold, model,
            )
        return over

    @staticmethod
    def get_context_limit(model: str) -> int:
        """Get context window limit for a model."""
        limits = {
            "qwen3.5:9b": settings.context_limit_qwen35_9b,
            "deepseek-r1:14b": settings.context_limit_deepseek_r1_14b,
        }
        return limits.get(model, 128_000)
