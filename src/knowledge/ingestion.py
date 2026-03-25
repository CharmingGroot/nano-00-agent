"""Document ingestion pipeline — PDF, CSV, XLSX parsing."""
import io
import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pandas as pd

logger = logging.getLogger(__name__)


class DocumentParser:
    """Parses various document types into plain text with metadata."""

    @staticmethod
    def parse(file_bytes: bytes, filename: str) -> list[dict[str, Any]]:
        """Parse a document into a list of text pages/sections with metadata.

        Returns:
            List of {"text": str, "metadata": {"page": int, "source": str, ...}}
        """
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            return DocumentParser._parse_pdf(file_bytes, filename)
        elif suffix == ".csv":
            return DocumentParser._parse_csv(file_bytes, filename)
        elif suffix in (".xlsx", ".xls"):
            return DocumentParser._parse_xlsx(file_bytes, filename)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    @staticmethod
    def _parse_pdf(file_bytes: bytes, filename: str) -> list[dict[str, Any]]:
        """Extract text from PDF page by page."""
        pages = []
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if text:
                pages.append({
                    "text": text,
                    "metadata": {
                        "source": filename,
                        "page": page_num + 1,
                        "total_pages": len(doc),
                    },
                })
        doc.close()
        logger.info("Parsed PDF '%s': %d pages with text", filename, len(pages))
        return pages

    @staticmethod
    def _parse_csv(file_bytes: bytes, filename: str) -> list[dict[str, Any]]:
        """Parse CSV into text rows grouped into sections."""
        df = pd.read_csv(io.BytesIO(file_bytes))
        return DocumentParser._dataframe_to_sections(df, filename)

    @staticmethod
    def _parse_xlsx(file_bytes: bytes, filename: str) -> list[dict[str, Any]]:
        """Parse Excel into text rows grouped into sections."""
        df = pd.read_excel(io.BytesIO(file_bytes))
        return DocumentParser._dataframe_to_sections(df, filename)

    @staticmethod
    def _dataframe_to_sections(
        df: pd.DataFrame, filename: str, rows_per_section: int = 50
    ) -> list[dict[str, Any]]:
        """Convert DataFrame to text sections (grouped rows)."""
        sections = []
        headers = " | ".join(str(c) for c in df.columns)
        total_rows = len(df)

        for start in range(0, total_rows, rows_per_section):
            end = min(start + rows_per_section, total_rows)
            chunk_df = df.iloc[start:end]
            rows_text = "\n".join(
                " | ".join(str(v) for v in row) for _, row in chunk_df.iterrows()
            )
            text = f"{headers}\n{rows_text}"
            sections.append({
                "text": text,
                "metadata": {
                    "source": filename,
                    "rows": f"{start + 1}-{end}",
                    "total_rows": total_rows,
                    "columns": list(df.columns),
                },
            })

        logger.info("Parsed '%s': %d sections from %d rows", filename, len(sections), total_rows)
        return sections
