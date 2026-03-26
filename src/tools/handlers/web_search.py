"""Web search tool handler — searches via Tavily or fallback."""
import logging
import os
from typing import Any

import httpx

from src.tools.base import BaseTool

logger = logging.getLogger(__name__)

TAVILY_API_URL = "https://api.tavily.com/search"


class WebSearchHandler(BaseTool):
    """Search the web using Tavily API.

    If TAVILY_API_KEY is not set, returns a stub response for development.
    """

    async def execute(
        self,
        query: str,
        max_results: int = 10,
        language: str = "ko",
        **kwargs: Any,
    ) -> dict[str, Any]:
        api_key = os.environ.get("TAVILY_API_KEY")

        if not api_key:
            logger.warning("TAVILY_API_KEY not set — returning stub results")
            return self._stub_results(query, max_results)

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                TAVILY_API_URL,
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "advanced",
                    "include_answer": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in data.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
            })

        logger.info("Web search '%s': %d results", query, len(results))
        return {"results": results}

    @staticmethod
    def _stub_results(query: str, max_results: int) -> dict[str, Any]:
        """Generate stub results for development without API key."""
        return {
            "results": [
                {
                    "title": f"[Stub] Result {i+1} for: {query}",
                    "url": f"https://example.com/result/{i+1}",
                    "content": f"This is a stub search result for '{query}'. "
                    f"Configure TAVILY_API_KEY for real results.",
                }
                for i in range(min(max_results, 3))
            ]
        }
