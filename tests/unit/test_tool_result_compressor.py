"""Tests for ToolResultCompressor."""
from src.middleware.tool_result_compressor import ToolResultCompressor


def test_needs_compression_small():
    """Small results don't need compression."""
    result = {"data": "hello world"}
    assert not ToolResultCompressor.needs_compression(result)


def test_needs_compression_large():
    """Large results need compression."""
    # 4000 tokens ≈ ~16000 chars for English, need more for repeated chars
    result = {"data": " ".join(["important data point number"] * 3000)}  # ~15K+ tokens
    assert ToolResultCompressor.needs_compression(result)


def test_get_token_count():
    """Token count estimation."""
    result = {"key": "value"}
    count = ToolResultCompressor.get_token_count(result)
    assert count > 0


def test_parse_compressed_response_valid_json():
    """Parse valid compression response."""
    content = '{"items": [{"key": "a", "data": "b"}], "items_total": 10, "items_retained": 1, "summary": "test"}'
    parsed = ToolResultCompressor.parse_compressed_response(content, {}, "test-uuid")
    assert parsed["structured"] is True
    assert "ptr:tool_result:test-uuid" in parsed["source_pointer"]


def test_parse_compressed_response_invalid_json():
    """Force wrap on invalid JSON."""
    parsed = ToolResultCompressor.parse_compressed_response(
        "not json at all", {}, "test-uuid"
    )
    assert parsed["structured"] is False
    assert "source_pointer" in parsed


def test_force_truncate():
    """Force truncation produces expected structure."""
    result = {"data": "x" * 50000}
    truncated = ToolResultCompressor.force_truncate(result, "test-uuid")
    assert truncated["structured"] is False
    assert "ptr:tool_result:test-uuid" in truncated["source_pointer"]
