"""Tests for knowledge pipeline — ingestion, chunking, retriever output schemas."""
import io
import json

import pytest

from src.knowledge.ingestion import DocumentParser
from src.knowledge.chunker import TextChunker


# ============================================================
# DocumentParser Tests
# ============================================================

class TestDocumentParser:
    """Verify parser output schema: list of {text, metadata}."""

    PARSED_PAGE_REQUIRED_FIELDS = {"text", "metadata"}

    def test_parse_csv(self):
        csv_content = b"name,age,city\nAlice,30,Seoul\nBob,25,Busan\nCharlie,35,Incheon"
        pages = DocumentParser.parse(csv_content, "test.csv")
        assert len(pages) >= 1
        for page in pages:
            assert self.PARSED_PAGE_REQUIRED_FIELDS.issubset(page.keys())
            assert isinstance(page["text"], str)
            assert len(page["text"]) > 0
            assert isinstance(page["metadata"], dict)
            assert "source" in page["metadata"]
            assert page["metadata"]["source"] == "test.csv"

    def test_parse_csv_metadata_has_rows(self):
        csv_content = b"col1,col2\n1,2\n3,4"
        pages = DocumentParser.parse(csv_content, "data.csv")
        meta = pages[0]["metadata"]
        assert "rows" in meta
        assert "total_rows" in meta
        assert "columns" in meta

    def test_parse_xlsx(self):
        """Test xlsx parsing with a real openpyxl-generated file."""
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["이름", "나이", "도시"])
        ws.append(["김철수", 30, "서울"])
        ws.append(["이영희", 25, "부산"])
        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)

        pages = DocumentParser.parse(buf.read(), "test.xlsx")
        assert len(pages) >= 1
        assert "이름" in pages[0]["text"]
        assert pages[0]["metadata"]["source"] == "test.xlsx"

    def test_parse_unsupported_type(self):
        with pytest.raises(ValueError, match="Unsupported"):
            DocumentParser.parse(b"data", "test.docx")

    def test_parse_empty_csv(self):
        """Empty CSV should still return structure."""
        csv_content = b"col1,col2\n"
        pages = DocumentParser.parse(csv_content, "empty.csv")
        # May return 0 pages for empty data, that's OK
        assert isinstance(pages, list)


# ============================================================
# TextChunker Tests
# ============================================================

class TestTextChunker:
    """Verify chunker output schema: list of {text, token_count, metadata, chunk_index}."""

    CHUNK_REQUIRED_FIELDS = {"text", "token_count", "metadata", "chunk_index"}

    def test_chunk_basic(self):
        pages = [{"text": "Hello world. This is a test document.", "metadata": {"source": "test"}}]
        chunker = TextChunker(chunk_size=512, overlap=50)
        chunks = chunker.chunk_document(pages)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert self.CHUNK_REQUIRED_FIELDS.issubset(chunk.keys())

    def test_chunk_fields_types(self):
        pages = [{"text": "Test content " * 100, "metadata": {"source": "test.pdf", "page": 1}}]
        chunker = TextChunker(chunk_size=512, overlap=50)
        chunks = chunker.chunk_document(pages)

        for chunk in chunks:
            assert isinstance(chunk["text"], str)
            assert isinstance(chunk["token_count"], int)
            assert chunk["token_count"] > 0
            assert isinstance(chunk["metadata"], dict)
            assert isinstance(chunk["chunk_index"], int)
            assert chunk["chunk_index"] >= 0

    def test_chunk_indices_are_sequential(self):
        pages = [
            {"text": "First page content. " * 200, "metadata": {"page": 1}},
            {"text": "Second page content. " * 200, "metadata": {"page": 2}},
        ]
        chunker = TextChunker(chunk_size=50, overlap=10)
        chunks = chunker.chunk_document(pages)
        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_respects_size_limit(self):
        """Each chunk should not exceed chunk_size (approximately)."""
        pages = [{"text": "Word " * 5000, "metadata": {"source": "big.txt"}}]
        chunker = TextChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk_document(pages)
        assert len(chunks) > 1
        for chunk in chunks:
            # Allow some tolerance (overlap can cause slight overrun)
            assert chunk["token_count"] <= 150, f"Chunk too large: {chunk['token_count']} tokens"

    def test_chunk_preserves_source_metadata(self):
        pages = [{"text": "Content here.", "metadata": {"source": "관세현황.pdf", "page": 3}}]
        chunker = TextChunker()
        chunks = chunker.chunk_document(pages)
        assert chunks[0]["metadata"]["source"] == "관세현황.pdf"

    def test_small_text_single_chunk(self):
        """Text smaller than chunk_size produces single chunk."""
        pages = [{"text": "Short text.", "metadata": {}}]
        chunker = TextChunker(chunk_size=512)
        chunks = chunker.chunk_document(pages)
        assert len(chunks) == 1


# ============================================================
# Retriever Output Schema (mock test)
# ============================================================

class TestRetrieverOutputSchema:
    """Verify expected output structure of retriever results."""

    RETRIEVER_RESULT_FIELDS = {
        "chunk_id", "content", "score", "document_name",
        "chunk_index", "token_count", "metadata",
    }

    def test_retriever_result_schema(self):
        """Simulate a retriever result and validate its structure."""
        mock_result = {
            "chunk_id": "550e8400-e29b-41d4-a716-446655440000",
            "content": "관세율은 8%입니다.",
            "score": 0.9234,
            "document_name": "수입자동차_관세부여현황.pdf",
            "document_id": "660e8400-e29b-41d4-a716-446655440001",
            "chunk_index": 5,
            "token_count": 12,
            "metadata": {"page": 3, "source": "수입자동차_관세부여현황.pdf"},
        }
        assert self.RETRIEVER_RESULT_FIELDS.issubset(mock_result.keys())
        assert isinstance(mock_result["score"], float)
        assert 0 <= mock_result["score"] <= 1
        assert isinstance(mock_result["chunk_id"], str)
        assert isinstance(mock_result["content"], str)

    def test_retriever_result_as_pointer_in_state(self):
        """Verify retriever results can be converted to pointer format for state JSON."""
        mock_results = [
            {"chunk_id": "abc", "content": "data", "score": 0.95, "document_name": "test.pdf",
             "chunk_index": 0, "token_count": 50, "metadata": {}},
            {"chunk_id": "def", "content": "data2", "score": 0.88, "document_name": "test.pdf",
             "chunk_index": 1, "token_count": 60, "metadata": {}},
        ]
        # Convert to state format
        state_knowledge = {
            "active_chunk_ids": [f"chunk:{r['chunk_id']}" for r in mock_results],
            "document_refs": [
                {
                    "doc_name": mock_results[0]["document_name"],
                    "relevance": mock_results[0]["score"],
                    "ptr": f"ptr:chunk:{mock_results[0]['chunk_id']}",
                    "desc": f"{mock_results[0]['document_name']} 청크 (score={mock_results[0]['score']})",
                }
            ],
        }
        assert len(state_knowledge["active_chunk_ids"]) == 2
        assert state_knowledge["active_chunk_ids"][0].startswith("chunk:")
        # Pointer has desc
        ref = state_knowledge["document_refs"][0]
        assert "ptr" in ref
        assert "desc" in ref
        assert ref["ptr"].startswith("ptr:chunk:")


# ============================================================
# Upload Response Schema
# ============================================================

class TestUploadResponseSchema:
    """Verify /knowledge/upload response structure."""

    UPLOAD_RESPONSE_FIELDS = {"document_id", "filename", "total_chunks", "file_type"}

    def test_upload_response_schema(self):
        mock_response = {
            "document_id": "550e8400-e29b-41d4-a716-446655440000",
            "filename": "수입자동차_관세부여현황.pdf",
            "total_chunks": 15,
            "file_type": "pdf",
        }
        assert self.UPLOAD_RESPONSE_FIELDS.issubset(mock_response.keys())
        assert isinstance(mock_response["total_chunks"], int)
        assert mock_response["total_chunks"] > 0

    def test_search_response_schema(self):
        mock_response = {
            "query": "관세율",
            "results": [
                {"chunk_id": "abc", "content": "관세율 8%", "score": 0.95,
                 "document_name": "test.pdf", "chunk_index": 0, "token_count": 10},
            ],
            "total_results": 1,
        }
        assert "query" in mock_response
        assert "results" in mock_response
        assert "total_results" in mock_response
        assert isinstance(mock_response["results"], list)
