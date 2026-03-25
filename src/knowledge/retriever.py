"""Retriever — pgvector cosine similarity search for knowledge chunks."""
import logging
import uuid
from typing import Any

from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.knowledge import Chunk, Document
from config.settings import settings

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    """Searches the knowledge base using pgvector cosine similarity.

    Returns top-K chunks most similar to the query embedding.
    Optionally filters by document filename.
    """

    def __init__(self, session: AsyncSession):
        self._session = session

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = settings.retriever_top_k,
        file_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar chunks using cosine distance.

        Args:
            query_embedding: Query vector (768 dimensions)
            top_k: Number of results to return
            file_filter: Optional partial filename match

        Returns:
            List of {chunk_id, content, score, document_name, metadata}
        """
        embedding_str = "[" + ",".join(str(f) for f in query_embedding) + "]"

        # Build query with cosine distance operator <=>
        # Note: asyncpg can't bind vector params via :param, so we use
        # a safe string interpolation for the vector literal only.
        # The embedding comes from our own Ollama call, not user input.
        where_clause = ""
        params: dict[str, Any] = {"top_k": top_k}

        if file_filter:
            where_clause = "WHERE d.filename ILIKE :file_filter"
            params["file_filter"] = f"%{file_filter}%"

        query = f"""
            SELECT
                c.id AS chunk_id,
                c.content,
                c.metadata AS chunk_metadata,
                c.chunk_index,
                c.token_count,
                d.filename AS document_name,
                d.id AS document_id,
                (c.embedding <=> '{embedding_str}'::vector) AS distance
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            {where_clause}
            ORDER BY distance ASC
            LIMIT :top_k
        """

        result = await self._session.execute(text(query), params)
        rows = result.fetchall()

        results = []
        for row in rows:
            # Cosine distance → similarity score (1 - distance)
            similarity = 1.0 - float(row.distance)
            results.append({
                "chunk_id": str(row.chunk_id),
                "content": row.content,
                "score": round(similarity, 4),
                "document_name": row.document_name,
                "document_id": str(row.document_id),
                "chunk_index": row.chunk_index,
                "token_count": row.token_count,
                "metadata": row.chunk_metadata or {},
            })

        logger.info(
            "Retrieved %d chunks (top_k=%d, file_filter=%s, best_score=%.4f)",
            len(results), top_k, file_filter,
            results[0]["score"] if results else 0.0,
        )
        return results
