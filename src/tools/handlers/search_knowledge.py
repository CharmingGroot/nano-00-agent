"""SearchKnowledge tool handler — wraps KnowledgeService.search()."""
from typing import Any

from src.tools.base import BaseTool
from src.knowledge.service import KnowledgeService
from src.middleware.llm_gateway import LLMGateway
from src.db.session import async_session_factory


class SearchKnowledgeHandler(BaseTool):
    """Search the embedded knowledge base and return matching chunks."""

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute a knowledge search.

        Keyword Args:
            query: Search query string (required).
            top_k: Number of results (default 20).
            file_filter: Optional filename filter.

        Returns:
            {"chunks": [{"chunk_id": ..., "content": ..., "score": ..., "document_name": ...}, ...]}
        """
        query: str = kwargs["query"]
        top_k: int = kwargs.get("top_k", 20)
        file_filter: str | None = kwargs.get("file_filter")

        async with async_session_factory() as session:
            gateway = LLMGateway()
            try:
                service = KnowledgeService(session=session, gateway=gateway)
                raw_results = await service.search(
                    query=query,
                    top_k=top_k,
                    file_filter=file_filter,
                )
            finally:
                await gateway.close()

        # Normalise to output_schema
        chunks = []
        for r in raw_results:
            chunks.append({
                "chunk_id": str(r.get("chunk_id", "")),
                "content": r.get("content", ""),
                "score": r.get("score", 0.0),
                "document_name": r.get("document_name", r.get("filename", "")),
            })

        return {"chunks": chunks}
