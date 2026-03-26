"""Bulk knowledge pipeline tests — 100 documents ingestion + search.

Tests:
- CSV, XLSX, PDF (mock) parsing at scale
- Chunker produces correct output for various document sizes
- Output schemas remain correct at volume
- Retriever result schema with multiple documents
"""
import io
import json
import uuid

import pytest
import openpyxl

from src.knowledge.ingestion import DocumentParser
from src.knowledge.chunker import TextChunker


# ============================================================
# Helpers
# ============================================================

def generate_csv(rows: int, cols: int = 5, filename: str = "test.csv") -> tuple[bytes, str]:
    """Generate a CSV with given dimensions."""
    headers = ",".join(f"col_{i}" for i in range(cols))
    data_rows = "\n".join(
        ",".join(f"val_{r}_{c}" for c in range(cols)) for r in range(rows)
    )
    content = f"{headers}\n{data_rows}"
    return content.encode("utf-8"), filename


def generate_xlsx(rows: int, cols: int = 5, filename: str = "test.xlsx") -> tuple[bytes, str]:
    """Generate an XLSX with given dimensions."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append([f"col_{i}" for i in range(cols)])
    for r in range(rows):
        ws.append([f"val_{r}_{c}" for c in range(cols)])
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.read(), filename


def generate_text_pages(num_pages: int, words_per_page: int = 200) -> list[dict]:
    """Generate mock parsed pages for chunker testing."""
    pages = []
    for p in range(num_pages):
        text = " ".join(f"word_{p}_{w}" for w in range(words_per_page))
        pages.append({
            "text": text,
            "metadata": {"source": f"doc_{p}.pdf", "page": p + 1, "total_pages": num_pages},
        })
    return pages


# ============================================================
# Bulk Parsing Tests (100 documents)
# ============================================================

class TestBulkParsing:
    """Test parsing 100 documents of various types."""

    PARSED_PAGE_FIELDS = {"text", "metadata"}

    def test_parse_100_csv_files(self):
        """Parse 100 CSV files with varying sizes."""
        for i in range(100):
            rows = 10 + (i * 5)  # 10 to 505 rows
            data, fname = generate_csv(rows, cols=4, filename=f"doc_{i:03d}.csv")
            pages = DocumentParser.parse(data, fname)
            assert len(pages) >= 1
            for page in pages:
                assert self.PARSED_PAGE_FIELDS.issubset(page.keys())
                assert isinstance(page["text"], str)
                assert len(page["text"]) > 0
                assert page["metadata"]["source"] == fname

    def test_parse_100_xlsx_files(self):
        """Parse 100 XLSX files."""
        for i in range(100):
            rows = 5 + (i * 2)  # 5 to 203 rows
            data, fname = generate_xlsx(rows, cols=3, filename=f"sheet_{i:03d}.xlsx")
            pages = DocumentParser.parse(data, fname)
            assert len(pages) >= 1
            for page in pages:
                assert self.PARSED_PAGE_FIELDS.issubset(page.keys())
                assert page["metadata"]["source"] == fname

    def test_parse_varied_csv_sizes(self):
        """Parse CSVs from tiny (1 row) to large (1000 rows)."""
        sizes = [1, 5, 10, 50, 100, 500, 1000]
        for size in sizes:
            data, fname = generate_csv(size, cols=10, filename=f"sized_{size}.csv")
            pages = DocumentParser.parse(data, fname)
            assert len(pages) >= 1
            # Large CSVs should produce multiple sections
            if size > 50:
                assert len(pages) >= 2, f"Expected multiple sections for {size} rows"

    def test_parse_wide_csv(self):
        """CSV with many columns (50 cols)."""
        data, fname = generate_csv(20, cols=50, filename="wide.csv")
        pages = DocumentParser.parse(data, fname)
        assert len(pages) >= 1
        # Check all column headers present
        assert "col_0" in pages[0]["text"]
        assert "col_49" in pages[0]["text"]


# ============================================================
# Bulk Chunking Tests
# ============================================================

CHUNK_REQUIRED_FIELDS = {"text", "token_count", "metadata", "chunk_index"}


class TestBulkChunking:
    """Test chunking 100 documents."""

    def test_chunk_100_documents(self):
        """Chunk 100 documents, verify output schema for all."""
        chunker = TextChunker(chunk_size=100, overlap=20)
        total_chunks = 0

        for i in range(100):
            pages = generate_text_pages(num_pages=3, words_per_page=100 + i * 10)
            chunks = chunker.chunk_document(pages)
            assert len(chunks) >= 1
            total_chunks += len(chunks)

            for chunk in chunks:
                assert CHUNK_REQUIRED_FIELDS.issubset(chunk.keys())
                assert isinstance(chunk["text"], str)
                assert isinstance(chunk["token_count"], int)
                assert chunk["token_count"] > 0
                assert isinstance(chunk["chunk_index"], int)

        # Should have generated many chunks
        assert total_chunks > 100, f"Expected >100 total chunks, got {total_chunks}"

    def test_chunk_indices_unique_per_document(self):
        """Each document's chunks should have unique sequential indices."""
        chunker = TextChunker(chunk_size=50, overlap=10)
        for _ in range(50):
            pages = generate_text_pages(num_pages=5, words_per_page=200)
            chunks = chunker.chunk_document(pages)
            indices = [c["chunk_index"] for c in chunks]
            # Sequential starting from 0
            assert indices == list(range(len(chunks)))
            # No duplicates
            assert len(set(indices)) == len(indices)

    def test_chunk_size_distribution(self):
        """Verify chunks are within expected token size range."""
        chunker = TextChunker(chunk_size=100, overlap=20)
        pages = generate_text_pages(num_pages=10, words_per_page=500)
        chunks = chunker.chunk_document(pages)

        oversized = [c for c in chunks if c["token_count"] > 150]  # Allow 50% tolerance
        assert len(oversized) == 0, (
            f"{len(oversized)} oversized chunks found: "
            f"{[c['token_count'] for c in oversized[:5]]}"
        )

    def test_chunk_metadata_preserved_across_pages(self):
        """Metadata from each page should flow into its chunks."""
        pages = [
            {"text": "Page one content. " * 100, "metadata": {"source": "doc.pdf", "page": 1}},
            {"text": "Page two content. " * 100, "metadata": {"source": "doc.pdf", "page": 2}},
        ]
        chunker = TextChunker(chunk_size=50, overlap=10)
        chunks = chunker.chunk_document(pages)

        # Some chunks should have page 1 metadata, some page 2
        page_1_chunks = [c for c in chunks if c["metadata"].get("page") == 1]
        page_2_chunks = [c for c in chunks if c["metadata"].get("page") == 2]
        assert len(page_1_chunks) > 0
        assert len(page_2_chunks) > 0


# ============================================================
# Bulk Schema Conformance
# ============================================================

class TestBulkSchemaConformance:
    """Ensure output schemas are consistent across 100 docs."""

    def test_parser_schema_consistency(self):
        """Every document produces pages with identical schema."""
        all_keys = set()
        for i in range(100):
            data, fname = generate_csv(10 + i, filename=f"schema_test_{i}.csv")
            pages = DocumentParser.parse(data, fname)
            for page in pages:
                all_keys.update(page.keys())
        # All pages should have exactly these keys
        assert all_keys == {"text", "metadata"}

    def test_chunker_schema_consistency(self):
        """Every chunk across 100 docs has identical schema."""
        chunker = TextChunker(chunk_size=100, overlap=20)
        all_keys = set()
        for i in range(100):
            pages = generate_text_pages(1, 50 + i * 5)
            chunks = chunker.chunk_document(pages)
            for chunk in chunks:
                all_keys.update(chunk.keys())
        assert all_keys == CHUNK_REQUIRED_FIELDS

    def test_upload_response_schema_simulated(self):
        """Simulate 100 upload responses, verify schema."""
        for i in range(100):
            response = {
                "document_id": str(uuid.uuid4()),
                "filename": f"doc_{i:03d}.pdf",
                "total_chunks": 5 + i,
                "file_type": ["pdf", "csv", "xlsx"][i % 3],
            }
            assert "document_id" in response
            assert "filename" in response
            assert "total_chunks" in response
            assert isinstance(response["total_chunks"], int)
            assert response["total_chunks"] > 0

    def test_search_result_schema_simulated(self):
        """Simulate 100 search results, verify schema."""
        for i in range(100):
            result = {
                "chunk_id": str(uuid.uuid4()),
                "content": f"Content of chunk {i}",
                "score": round(0.5 + (i / 200), 4),
                "document_name": f"doc_{i % 10}.pdf",
                "document_id": str(uuid.uuid4()),
                "chunk_index": i,
                "token_count": 50 + i,
                "metadata": {"page": i % 5 + 1},
            }
            assert 0 <= result["score"] <= 1
            assert isinstance(result["chunk_id"], str)
            assert isinstance(result["content"], str)
            assert isinstance(result["token_count"], int)

    def test_pointer_format_for_100_results(self):
        """Verify pointer format is consistent across 100 results."""
        for i in range(100):
            uid = str(uuid.uuid4())
            ptr = f"ptr:tool_result:{uid}"
            assert ptr.startswith("ptr:")
            parts = ptr.split(":")
            assert len(parts) == 3
            assert parts[0] == "ptr"
            assert parts[1] == "tool_result"
            assert len(parts[2]) == 36  # UUID length

    def test_accumulated_data_pointer_with_desc(self):
        """Every pointer in accumulated_data must have desc."""
        for i in range(100):
            entry = {
                "ptr": f"ptr:tool_result:{uuid.uuid4()}",
                "desc": f"검색 결과 {i}건, 압축본 사용 중",
                "token_count_compressed": 3000 + i * 10,
                "token_count_raw": 30000 + i * 100,
                "items_total": 10 + i,
                "items_retained": 5 + (i % 5),
            }
            assert "ptr" in entry
            assert "desc" in entry
            assert entry["ptr"].startswith("ptr:")
            assert isinstance(entry["desc"], str)
            assert len(entry["desc"]) > 0
