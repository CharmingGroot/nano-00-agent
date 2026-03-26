"""PDF generation tool handler — creates PDFs from structured content."""
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from src.tools.base import BaseTool

logger = logging.getLogger(__name__)

FILESTORE_DIR = Path(os.environ.get("FILESTORE_DIR", "/tmp/nano-agent-files"))


class GeneratePdfHandler(BaseTool):
    """Generate a PDF document from structured content using WeasyPrint."""

    async def execute(
        self,
        title: str,
        sections: list[dict[str, str]] | str,
        output_filename: str = "report.pdf",
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Ensure output directory exists
        FILESTORE_DIR.mkdir(parents=True, exist_ok=True)

        # Build HTML content
        html = self._build_html(title, sections)

        # Generate PDF
        file_id = str(uuid.uuid4())[:8]
        filename = f"{file_id}_{output_filename}"
        filepath = FILESTORE_DIR / filename

        try:
            from weasyprint import HTML
            HTML(string=html).write_pdf(str(filepath))
            file_size = filepath.stat().st_size
        except (ImportError, OSError):
            # Fallback: save HTML if WeasyPrint not available or system libs missing
            filepath = filepath.with_suffix(".html")
            filepath.write_text(html, encoding="utf-8")
            file_size = filepath.stat().st_size
            logger.warning("WeasyPrint not available — saved as HTML")

        logger.info("Generated PDF: %s (%d bytes)", filepath, file_size)
        return {
            "file_path": str(filepath),
            "file_size_bytes": file_size,
            "filename": filename,
        }

    @staticmethod
    def _build_html(title: str, sections: list[dict[str, str]] | str) -> str:
        """Build HTML from title + sections."""
        css = """
        <style>
            body { font-family: 'Noto Sans KR', sans-serif; margin: 40px; line-height: 1.6; }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            h2 { color: #34495e; margin-top: 30px; }
            p { color: #555; }
            .meta { color: #999; font-size: 0.9em; }
        </style>
        """

        body = f"<h1>{title}</h1>\n"

        if isinstance(sections, str):
            body += f"<p>{sections}</p>"
        elif isinstance(sections, list):
            for section in sections:
                if isinstance(section, dict):
                    heading = section.get("heading", section.get("key", ""))
                    content = section.get("body", section.get("data", section.get("summary", "")))
                    if heading:
                        body += f"<h2>{heading}</h2>\n"
                    body += f"<p>{content}</p>\n"
                else:
                    body += f"<p>{section}</p>\n"

        return f"<!DOCTYPE html><html><head><meta charset='utf-8'>{css}</head><body>{body}</body></html>"
