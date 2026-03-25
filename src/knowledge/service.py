"""Knowledge service — orchestrates ingestion, chunking, embedding, and storage."""
import logging
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.knowledge.ingestion import DocumentParser
from src.knowledge.chunker import TextChunker
from src.knowledge.embedder import Embedder
from src.knowledge.retriever import KnowledgeRetriever
from src.models.knowledge import Document, Chunk
from src.middleware.llm_gateway import LLMGateway

logger = logging.getLogger(__name__)


class KnowledgeService:
    """Full knowledge pipeline: upload → parse → chunk → embed → store → search."""

    def __init__(self, session: AsyncSession, gateway: LLMGateway | None = None):
        self._session = session
        self._gateway = gateway or LLMGateway()
        self._chunker = TextChunker()
        self._embedder = Embedder(self._gateway)
        self._retriever = KnowledgeRetriever(session)

    async def ingest_document(self, file_bytes: bytes, filename: str) -> dict[str, Any]:
        """Full ingestion pipeline: parse → chunk → embed → store in DB.

        Returns:
            {"document_id": str, "filename": str, "total_chunks": int, "file_type": str}
        """
        # 1. Parse
        file_type = filename.rsplit(".", 1)[-1].lower()
        pages = DocumentParser.parse(file_bytes, filename)

        if not pages:
            raise ValueError(f"No content extracted from '{filename}'")

        # 2. Chunk
        chunks = self._chunker.chunk_document(pages)

        # 3. Embed
        chunks = await self._embedder.embed_chunks(chunks)

        # 4. Store document
        doc = Document(
            filename=filename,
            file_type=file_type,
            metadata_={"total_pages": len(pages)},
            total_chunks=len(chunks),
        )
        self._session.add(doc)
        await self._session.flush()  # Get doc.id

        # 5. Store chunks with embeddings
        for chunk_data in chunks:
            chunk = Chunk(
                document_id=doc.id,
                chunk_index=chunk_data["chunk_index"],
                content=chunk_data["text"],
                token_count=chunk_data["token_count"],
                embedding=chunk_data["embedding"],
                metadata_=chunk_data["metadata"],
            )
            self._session.add(chunk)

        await self._session.commit()

        result = {
            "document_id": str(doc.id),
            "filename": filename,
            "total_chunks": len(chunks),
            "file_type": file_type,
        }
        logger.info("Ingested document: %s", result)
        return result

    async def search(
        self,
        query: str,
        top_k: int = 20,
        file_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search knowledge base by query text.

        Returns list of relevant chunks with similarity scores.
        """
        query_embedding = await self._embedder.embed_query(query)
        return await self._retriever.search(
            query_embedding=query_embedding,
            top_k=top_k,
            file_filter=file_filter,
        )
