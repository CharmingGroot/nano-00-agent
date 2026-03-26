"""Notion page creation tool handler."""
import logging
import os
from typing import Any

import httpx

from src.tools.base import BaseTool

logger = logging.getLogger(__name__)

NOTION_API_URL = "https://api.notion.com/v1/pages"
NOTION_API_VERSION = "2022-06-28"


class CreateNotionPageHandler(BaseTool):
    """Create a new page in Notion.

    Requires NOTION_API_TOKEN and NOTION_PARENT_PAGE_ID environment variables.
    If not configured, returns a stub response for development.
    """

    async def execute(
        self,
        title: str,
        content: str,
        pdf_attachment: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        api_token = os.environ.get("NOTION_API_TOKEN")
        parent_id = os.environ.get("NOTION_PARENT_PAGE_ID")

        if not api_token or not parent_id:
            logger.warning("Notion not configured — returning stub response")
            return self._stub_response(title)

        # Build Notion page payload
        payload = {
            "parent": {"page_id": parent_id},
            "properties": {
                "title": {
                    "title": [{"text": {"content": title}}]
                }
            },
            "children": self._markdown_to_blocks(content),
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                NOTION_API_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_token}",
                    "Notion-Version": NOTION_API_VERSION,
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        page_id = data.get("id", "")
        page_url = data.get("url", "")
        logger.info("Created Notion page: %s (%s)", title, page_url)

        return {
            "page_id": page_id,
            "page_url": page_url,
            "title": title,
        }

    @staticmethod
    def _markdown_to_blocks(content: str) -> list[dict]:
        """Convert simple markdown text to Notion blocks."""
        blocks = []
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("# "):
                blocks.append({
                    "object": "block",
                    "type": "heading_1",
                    "heading_1": {"rich_text": [{"text": {"content": line[2:]}}]},
                })
            elif line.startswith("## "):
                blocks.append({
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"text": {"content": line[3:]}}]},
                })
            else:
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"text": {"content": line}}]},
                })
        return blocks

    @staticmethod
    def _stub_response(title: str) -> dict[str, Any]:
        """Stub response when Notion is not configured."""
        return {
            "page_id": "stub-page-id",
            "page_url": "https://notion.so/stub-page",
            "title": title,
            "note": "Notion not configured — set NOTION_API_TOKEN and NOTION_PARENT_PAGE_ID",
        }
