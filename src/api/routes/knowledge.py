"""Knowledge upload and search endpoints."""
import logging

from fastapi import APIRouter, Depends, UploadFile, File, Query, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_session
from src.knowledge.service import KnowledgeService
from src.middleware.llm_gateway import LLMGateway

logger = logging.getLogger(__name__)
router = APIRouter()

_gateway = LLMGateway()


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    total_chunks: int
    file_type: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 20
    file_filter: str | None = None


class ChunkResult(BaseModel):
    chunk_id: str
    content: str
    score: float
    document_name: str
    chunk_index: int
    token_count: int


class SearchResponse(BaseModel):
    query: str
    results: list[ChunkResult]
    total_results: int


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    """Upload a document (PDF, CSV, XLSX) for knowledge ingestion.

    Pipeline: parse → chunk (512 tok, 50 overlap) → embed (nomic-embed-text) → store in pgvector
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    allowed_types = {".pdf", ".csv", ".xlsx", ".xls"}
    suffix = "." + file.filename.rsplit(".", 1)[-1].lower()
    if suffix not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {allowed_types}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    service = KnowledgeService(session, _gateway)
    result = await service.ingest_document(file_bytes, file.filename)
    return UploadResponse(**result)


@router.post("/search", response_model=SearchResponse)
async def search_knowledge(
    request: SearchRequest,
    session: AsyncSession = Depends(get_session),
):
    """Search the knowledge base by semantic similarity.

    Uses nomic-embed-text to embed the query, then pgvector cosine similarity.
    """
    service = KnowledgeService(session, _gateway)
    results = await service.search(
        query=request.query,
        top_k=request.top_k,
        file_filter=request.file_filter,
    )

    return SearchResponse(
        query=request.query,
        results=[ChunkResult(**{k: v for k, v in r.items() if k in ChunkResult.model_fields}) for r in results],
        total_results=len(results),
    )
