"""Chat endpoint — main entry point for user interactions.

Routes all requests through the MiddlewarePipeline which handles:
Goal generation, intent classification, task decomposition,
context management, token tracking, state compression, and HITL.
"""
import logging
import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_session
from src.middleware.llm_gateway import LLMGateway
from src.middleware.pipeline import MiddlewarePipeline
from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)
router = APIRouter()

# Singletons (will be properly DI'd later)
_gateway = LLMGateway()
_tool_registry = ToolRegistry()
try:
    _tool_registry.load_all()
except Exception:
    pass  # OK if registries dir doesn't exist (e.g., in tests)

# In-memory conversation states (will be DB-backed in production)
_conversation_states: dict[str, dict] = {}


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str
    model: str = "qwen3.5:9b"
    hitl_confirmation: dict | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    token_count: dict
    goal: dict | None = None
    pending_hitl: dict | None = None
    task_progress: dict | None = None
    conversation_state: dict | None = None


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    session: AsyncSession = Depends(get_session),
):
    """Main chat endpoint. All processing goes through MiddlewarePipeline."""
    conversation_id = request.conversation_id or str(uuid.uuid4())

    # Load existing state or start fresh
    existing_state = _conversation_states.get(conversation_id)

    pipeline = MiddlewarePipeline(
        gateway=_gateway,
        tool_registry=_tool_registry,
        session=session,
    )

    result = await pipeline.process(
        user_message=request.message,
        conversation_state=existing_state,
        model=request.model,
        hitl_confirmation=request.hitl_confirmation,
    )

    # Save updated state
    _conversation_states[conversation_id] = result["conversation_state"]

    return ChatResponse(
        conversation_id=conversation_id,
        response=result["response"],
        token_count=result["token_count"],
        goal=result.get("goal"),
        pending_hitl=result.get("pending_hitl"),
        task_progress=result.get("task_progress"),
        conversation_state=result.get("conversation_state"),
    )
