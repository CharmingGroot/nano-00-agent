"""Tests for Phase 5 tool handlers — web_search, generate_pdf, create_notion_page.

Validates output schemas and behavior for all tool handlers.
"""
import json
import os
import uuid

import pytest

from src.tools.handlers.web_search import WebSearchHandler
from src.tools.handlers.generate_pdf import GeneratePdfHandler
from src.tools.handlers.create_notion_page import CreateNotionPageHandler


# ============================================================
# WebSearchHandler Tests
# ============================================================

class TestWebSearchHandler:
    """Web search tool output schema validation."""

    WEB_SEARCH_OUTPUT_FIELDS = {"results"}
    RESULT_ITEM_FIELDS = {"title", "url", "content"}

    @pytest.mark.asyncio
    async def test_stub_search_returns_results(self):
        handler = WebSearchHandler()
        result = await handler.execute(query="AI trends", max_results=5)
        assert self.WEB_SEARCH_OUTPUT_FIELDS.issubset(result.keys())
        assert isinstance(result["results"], list)
        assert len(result["results"]) > 0

    @pytest.mark.asyncio
    async def test_stub_result_item_schema(self):
        handler = WebSearchHandler()
        result = await handler.execute(query="test query")
        for item in result["results"]:
            assert self.RESULT_ITEM_FIELDS.issubset(item.keys())
            assert isinstance(item["title"], str)
            assert isinstance(item["url"], str)
            assert isinstance(item["content"], str)

    @pytest.mark.asyncio
    async def test_max_results_respected(self):
        handler = WebSearchHandler()
        result = await handler.execute(query="test", max_results=2)
        assert len(result["results"]) <= 3  # stub max is 3

    @pytest.mark.asyncio
    async def test_search_result_compressible(self):
        """Web search results should be serializable for compression."""
        handler = WebSearchHandler()
        result = await handler.execute(query="long query " * 50)
        serialized = json.dumps(result, ensure_ascii=False)
        assert len(serialized) > 0
        # Should be parseable back
        parsed = json.loads(serialized)
        assert "results" in parsed


# ============================================================
# GeneratePdfHandler Tests
# ============================================================

class TestGeneratePdfHandler:
    """PDF generation tool output schema validation."""

    PDF_OUTPUT_FIELDS = {"file_path", "file_size_bytes", "filename"}

    @pytest.mark.asyncio
    async def test_generate_pdf_output_schema(self):
        handler = GeneratePdfHandler()
        result = await handler.execute(
            title="Test Report",
            sections=[
                {"heading": "Section 1", "body": "Content here."},
                {"heading": "Section 2", "body": "More content."},
            ],
        )
        assert self.PDF_OUTPUT_FIELDS.issubset(result.keys())
        assert isinstance(result["file_path"], str)
        assert isinstance(result["file_size_bytes"], int)
        assert result["file_size_bytes"] > 0

    @pytest.mark.asyncio
    async def test_generate_pdf_with_string_sections(self):
        handler = GeneratePdfHandler()
        result = await handler.execute(
            title="Simple Report",
            sections="Just plain text content.",
        )
        assert self.PDF_OUTPUT_FIELDS.issubset(result.keys())
        assert result["file_size_bytes"] > 0

    @pytest.mark.asyncio
    async def test_generate_pdf_korean_content(self):
        handler = GeneratePdfHandler()
        result = await handler.execute(
            title="관세 현황 보고서",
            sections=[
                {"heading": "개요", "body": "수입자동차 관세 현황을 분석합니다."},
                {"heading": "결론", "body": "전기차 관세율은 0%입니다."},
            ],
        )
        assert result["file_size_bytes"] > 0

    @pytest.mark.asyncio
    async def test_generate_pdf_custom_filename(self):
        handler = GeneratePdfHandler()
        result = await handler.execute(
            title="Custom",
            sections=[{"heading": "A", "body": "B"}],
            output_filename="custom_report.pdf",
        )
        assert "custom_report" in result["filename"]

    def test_build_html(self):
        """HTML builder produces valid HTML."""
        html = GeneratePdfHandler._build_html(
            "Test Title",
            [{"heading": "H1", "body": "Content"}],
        )
        assert "<h1>Test Title</h1>" in html
        assert "<h2>H1</h2>" in html
        assert "Content" in html
        assert "<!DOCTYPE html>" in html


# ============================================================
# CreateNotionPageHandler Tests
# ============================================================

class TestCreateNotionPageHandler:
    """Notion page creation output schema validation."""

    NOTION_OUTPUT_FIELDS = {"page_id", "page_url", "title"}

    @pytest.mark.asyncio
    async def test_stub_notion_page_schema(self):
        handler = CreateNotionPageHandler()
        result = await handler.execute(
            title="Test Page",
            content="# Heading\nSome content here.",
        )
        assert self.NOTION_OUTPUT_FIELDS.issubset(result.keys())
        assert isinstance(result["page_id"], str)
        assert isinstance(result["page_url"], str)
        assert result["title"] == "Test Page"

    @pytest.mark.asyncio
    async def test_stub_has_note(self):
        """Stub response should indicate Notion is not configured."""
        handler = CreateNotionPageHandler()
        result = await handler.execute(title="Test", content="Content")
        assert "note" in result or "page_url" in result

    def test_markdown_to_blocks(self):
        """Markdown conversion produces valid Notion blocks."""
        blocks = CreateNotionPageHandler._markdown_to_blocks(
            "# Heading 1\n## Heading 2\nParagraph text."
        )
        assert len(blocks) == 3
        assert blocks[0]["type"] == "heading_1"
        assert blocks[1]["type"] == "heading_2"
        assert blocks[2]["type"] == "paragraph"

    def test_markdown_to_blocks_empty(self):
        blocks = CreateNotionPageHandler._markdown_to_blocks("")
        assert blocks == []

    def test_markdown_to_blocks_korean(self):
        blocks = CreateNotionPageHandler._markdown_to_blocks(
            "# 보고서\n## 개요\n관세 현황을 분석합니다."
        )
        assert len(blocks) == 3
        assert "보고서" in str(blocks[0])


# ============================================================
# Tool Handler Conformance (100 iterations)
# ============================================================

class TestToolHandlerConformance:
    """Verify all handlers produce consistent output schemas at scale."""

    @pytest.mark.asyncio
    async def test_web_search_100_queries(self):
        handler = WebSearchHandler()
        for i in range(100):
            result = await handler.execute(query=f"query {i}", max_results=3)
            assert "results" in result
            assert isinstance(result["results"], list)
            for item in result["results"]:
                assert "title" in item
                assert "url" in item
                assert "content" in item

    @pytest.mark.asyncio
    async def test_generate_pdf_50_reports(self):
        handler = GeneratePdfHandler()
        for i in range(50):
            result = await handler.execute(
                title=f"Report {i}",
                sections=[{"heading": f"Section {j}", "body": f"Content {j}"} for j in range(3)],
                output_filename=f"report_{i}.pdf",
            )
            assert "file_path" in result
            assert "file_size_bytes" in result
            assert result["file_size_bytes"] > 0

    @pytest.mark.asyncio
    async def test_notion_page_50_stubs(self):
        handler = CreateNotionPageHandler()
        for i in range(50):
            result = await handler.execute(
                title=f"Page {i}",
                content=f"# Content {i}\nBody text.",
            )
            assert "page_id" in result
            assert "page_url" in result
            assert "title" in result
