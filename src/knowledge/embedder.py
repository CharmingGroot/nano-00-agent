"""Embedder — generates vector embeddings via Ollama nomic-embed-text."""
import logging
from typing import Any

from src.middleware.llm_gateway import LLMGateway
from config.settings import settings

logger = logging.getLogger(__name__)


class Embedder:
    """Generates embeddings for text chunks using Ollama's embedding API.

    Uses nomic-embed-text (768 dimensions) by default.
    Batches requests for efficiency.
    """

    def __init__(self, gateway: LLMGateway | None = None):
        self._gateway = gateway or LLMGateway()

    async def embed_chunks(
        self,
        chunks: list[dict[str, Any]],
        batch_size: int = 32,
    ) -> list[dict[str, Any]]:
        """Embed all chunks, adding 'embedding' field to each.

        Args:
            chunks: Output from TextChunker.chunk_document()
            batch_size: Number of texts per Ollama /api/embed call

        Returns:
            Same chunks with added 'embedding' field (list[float])
        """
        texts = [c["text"] for c in chunks]
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(
                "Embedding batch %d-%d of %d",
                i, min(i + batch_size, len(texts)), len(texts),
            )
            embeddings = await self._gateway.embed(batch)
            all_embeddings.extend(embeddings)

        if len(all_embeddings) != len(chunks):
            logger.error(
                "Embedding count mismatch: got %d, expected %d",
                len(all_embeddings), len(chunks),
            )
            raise ValueError("Embedding count mismatch")

        for chunk, embedding in zip(chunks, all_embeddings):
            chunk["embedding"] = embedding

        logger.info("Embedded %d chunks (dim=%d)", len(chunks), len(all_embeddings[0]) if all_embeddings else 0)
        return chunks

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string for similarity search."""
        embeddings = await self._gateway.embed([query])
        return embeddings[0]
