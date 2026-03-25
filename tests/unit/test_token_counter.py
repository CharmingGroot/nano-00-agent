"""Tests for TokenCounter."""
from src.middleware.token_counter import TokenCounter


def test_count_tokens_basic():
    """Basic token counting."""
    count = TokenCounter.count_tokens("Hello, world!")
    assert count > 0
    assert count < 10


def test_count_tokens_korean():
    """Korean text token counting."""
    count = TokenCounter.count_tokens("안녕하세요, 세계!")
    assert count > 0


def test_count_messages_tokens():
    """Message list token counting."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    count = TokenCounter.count_messages_tokens(messages)
    assert count > 0
    # Should be more than just the text tokens (includes overhead)
    assert count > TokenCounter.count_tokens("You are a helpful assistant.Hello!")


def test_should_compress_below_threshold():
    """Should not compress when below threshold."""
    assert not TokenCounter.should_compress(10_000)


def test_should_compress_above_threshold():
    """Should compress when at or above threshold."""
    assert TokenCounter.should_compress(150_000)
    assert TokenCounter.should_compress(200_000)


def test_get_context_limit():
    """Context limits per model."""
    assert TokenCounter.get_context_limit("qwen3.5:9b") == 256_000
    assert TokenCounter.get_context_limit("deepseek-r1:14b") == 128_000
    assert TokenCounter.get_context_limit("unknown-model") == 128_000
