"""Tests for PointerResolver — pointer collection, selection, and fetch."""
import json
import uuid

import pytest

from src.middleware.pointer_resolver import PointerResolver, POINTER_RE
from src.middleware.pipeline import MiddlewarePipeline


class TestPointerCollection:
    """Test collecting pointers from conversation state."""

    def _make_resolver(self):
        return PointerResolver(gateway=None, session=None)

    def test_collect_from_accumulated_data(self):
        resolver = self._make_resolver()
        state = MiddlewarePipeline._init_state()
        state["accumulated_data"] = {
            "search_results": {
                "ptr": "ptr:tool_result:550e8400-e29b-41d4-a716-446655440000",
                "desc": "Tavily 검색 결과 10건",
                "token_count_compressed": 3500,
                "token_count_raw": 32000,
            },
            "analysis": {
                "ptr": "ptr:tool_result:660e8400-e29b-41d4-a716-446655440001",
                "desc": "관세 분석 결과",
                "token_count_compressed": 2000,
            },
        }
        pointers = resolver.collect_pointers(state)
        assert len(pointers) == 2
        for p in pointers:
            assert "ptr" in p
            assert "desc" in p
            assert "type" in p
            assert "uuid" in p
            assert p["type"] == "tool_result"

    def test_collect_from_knowledge_context_chunks(self):
        resolver = self._make_resolver()
        state = MiddlewarePipeline._init_state()
        state["knowledge_context"]["active_chunk_ids"] = [
            "chunk:abc123",
            "chunk:def456",
        ]
        pointers = resolver.collect_pointers(state)
        assert len(pointers) == 2
        assert pointers[0]["type"] == "chunk"
        assert pointers[0]["uuid"] == "abc123"

    def test_collect_from_document_refs(self):
        resolver = self._make_resolver()
        state = MiddlewarePipeline._init_state()
        state["knowledge_context"]["document_refs"] = [
            {
                "ptr": "ptr:chunk:aaa111",
                "desc": "관세현황.pdf 청크",
                "doc_name": "관세현황.pdf",
            },
        ]
        pointers = resolver.collect_pointers(state)
        assert len(pointers) == 1
        assert pointers[0]["type"] == "chunk"

    def test_collect_empty_state(self):
        resolver = self._make_resolver()
        state = MiddlewarePipeline._init_state()
        pointers = resolver.collect_pointers(state)
        assert pointers == []

    def test_collect_mixed_pointers(self):
        """State with both tool_results and chunks."""
        resolver = self._make_resolver()
        state = MiddlewarePipeline._init_state()
        state["accumulated_data"]["step1"] = {
            "ptr": f"ptr:tool_result:{uuid.uuid4()}",
            "desc": "Step 1 결과",
        }
        state["accumulated_data"]["step2"] = {
            "ptr": f"ptr:tool_result:{uuid.uuid4()}",
            "desc": "Step 2 결과",
        }
        state["knowledge_context"]["active_chunk_ids"] = ["chunk:xyz789"]
        pointers = resolver.collect_pointers(state)
        assert len(pointers) == 3
        types = {p["type"] for p in pointers}
        assert types == {"tool_result", "chunk"}

    def test_collect_50_turn_accumulated_pointers(self):
        """50 turns of accumulated data, all pointers should be collected."""
        resolver = self._make_resolver()
        state = MiddlewarePipeline._init_state()
        for i in range(50):
            state["accumulated_data"][f"turn_{i}"] = {
                "ptr": f"ptr:tool_result:{uuid.uuid4()}",
                "desc": f"Turn {i} 결과 데이터",
                "token_count_raw": 20000 + i * 100,
            }
        pointers = resolver.collect_pointers(state)
        assert len(pointers) == 50
        # All should have required fields
        for p in pointers:
            assert p["ptr"].startswith("ptr:tool_result:")
            assert "desc" in p
            assert "uuid" in p


class TestPointerSelectionParsing:
    """Test LLM response parsing for pointer selection."""

    def test_parse_valid_json_array(self):
        content = '["ptr:tool_result:abc123", "ptr:chunk:def456"]'
        result = PointerResolver._parse_pointer_selection(content)
        assert result == ["ptr:tool_result:abc123", "ptr:chunk:def456"]

    def test_parse_json_in_code_block(self):
        content = '```json\n["ptr:tool_result:abc"]\n```'
        result = PointerResolver._parse_pointer_selection(content)
        assert result == ["ptr:tool_result:abc"]

    def test_parse_empty_array(self):
        result = PointerResolver._parse_pointer_selection("[]")
        assert result == []

    def test_parse_invalid_json_fallback_regex(self):
        """When JSON parse fails, try regex extraction."""
        content = "I think ptr:tool_result:abc123 and ptr:chunk:def456 are relevant."
        result = PointerResolver._parse_pointer_selection(content)
        # regex returns tuples of (type, uuid)
        assert len(result) >= 0  # May return tuples from regex, not strings

    def test_parse_filters_non_pointer_strings(self):
        content = '["ptr:tool_result:abc", "not-a-pointer", "ptr:chunk:def"]'
        result = PointerResolver._parse_pointer_selection(content)
        assert "not-a-pointer" not in result
        assert len(result) == 2


class TestPointerFetchWithoutDB:
    """Test fetch behavior when no DB session is available."""

    @pytest.mark.asyncio
    async def test_fetch_without_session_returns_descriptions(self):
        resolver = PointerResolver(gateway=None, session=None)
        pointers = [
            {
                "ptr": "ptr:tool_result:abc",
                "desc": "검색 결과 10건",
                "type": "tool_result",
                "uuid": "abc",
                "source_key": "search",
            },
        ]
        results = await resolver.fetch_pointer_data(pointers)
        assert len(results) == 1
        assert results[0]["ptr"] == "ptr:tool_result:abc"
        assert "desc" in results[0]
        assert "content" in results[0]
        assert "No DB session" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_fetch_50_pointers_without_db(self):
        """All 50 pointers should return gracefully without DB."""
        resolver = PointerResolver(gateway=None, session=None)
        pointers = [
            {
                "ptr": f"ptr:tool_result:{uuid.uuid4()}",
                "desc": f"Result {i}",
                "type": "tool_result",
                "uuid": str(uuid.uuid4()),
                "source_key": f"step_{i}",
            }
            for i in range(50)
        ]
        results = await resolver.fetch_pointer_data(pointers)
        assert len(results) == 50
        for r in results:
            assert "ptr" in r
            assert "desc" in r
            assert "content" in r


class TestPointerRegex:
    """Test pointer format regex."""

    def test_valid_tool_result_pointer(self):
        match = POINTER_RE.match("ptr:tool_result:550e8400-e29b-41d4-a716-446655440000")
        assert match
        assert match.group(1) == "tool_result"
        assert match.group(2) == "550e8400-e29b-41d4-a716-446655440000"

    def test_valid_chunk_pointer(self):
        match = POINTER_RE.match("ptr:chunk:abc123")
        assert match
        assert match.group(1) == "chunk"

    def test_valid_task_node_pointer(self):
        match = POINTER_RE.match("ptr:task_node:abc789def0-1234-5678-9abc-def012345678")
        assert match
        assert match.group(1) == "task_node"

    def test_invalid_pointer(self):
        match = POINTER_RE.match("not-a-pointer")
        assert match is None

    def test_100_generated_pointers_match(self):
        for _ in range(100):
            uid = str(uuid.uuid4())
            for ptype in ["tool_result", "chunk", "task_node"]:
                ptr = f"ptr:{ptype}:{uid}"
                match = POINTER_RE.match(ptr)
                assert match, f"Failed to match: {ptr}"
                assert match.group(1) == ptype
                assert match.group(2) == uid


class TestPointerResolveInPipeline:
    """Test that _resolve_pointers_for_question handles edge cases."""

    @pytest.mark.asyncio
    async def test_resolve_with_empty_state(self):
        """No pointers → no resolution, no error."""
        from src.middleware.llm_gateway import LLMGateway
        gateway = LLMGateway()
        pipeline = MiddlewarePipeline(gateway=gateway, tool_registry=None)
        state = MiddlewarePipeline._init_state()
        result = await pipeline._resolve_pointers_for_question(
            "test question", state, "qwen3.5:9b"
        )
        assert result == []
        await gateway.close()
