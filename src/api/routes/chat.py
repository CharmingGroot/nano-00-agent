"""Chat endpoint — main entry point for user interactions."""
import logging
import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_session
from src.middleware.llm_gateway import LLMGateway, LLMRequest
from src.middleware.token_counter import TokenCounter
from src.middleware.goal_generator import GoalGenerator
from src.middleware.context_manager import ContextManager

logger = logging.getLogger(__name__)
router = APIRouter()

# Singleton gateway (will be properly DI'd later)
_gateway = LLMGateway()

SYSTEM_PROMPT = """You are nano-00-agent, an AI assistant powered by local LLMs.
You help users with document analysis, research, and various tasks.
Always respond in the user's language."""


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


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    session: AsyncSession = Depends(get_session),
):
    """Main chat endpoint. All LLM calls go through LLMGateway."""
    conversation_id = request.conversation_id or str(uuid.uuid4())

    # Step 0: Generate Goal from user query
    goal_messages = GoalGenerator.build_goal_messages(request.message)
    goal_req = LLMRequest(
        model=request.model,
        messages=goal_messages,
        conversation_id=conversation_id,
    )
    goal_response = await _gateway.chat(goal_req)
    goal = GoalGenerator.parse_goal_response(goal_response.content, request.message)
    goal["goal_id"] = str(uuid.uuid4())
    goal["progress_pct"] = 0
    goal["criteria_status"] = GoalGenerator.init_criteria_status(goal)

    logger.info("Goal generated: %s", goal.get("final_objective", "N/A"))

    # Step 1: Assemble prompt with Goal
    messages = ContextManager.build_simple_chat_prompt(
        system_prompt=SYSTEM_PROMPT,
        user_message=request.message,
        goal=goal,
    )

    # Step 2: Send through LLMGateway
    llm_request = LLMRequest(
        model=request.model,
        messages=messages,
        conversation_id=conversation_id,
        goal=goal,
    )
    llm_response = await _gateway.chat(llm_request)

    # Step 3: Track tokens
    total_tokens = (
        goal_response.token_count_prompt + goal_response.token_count_completion
        + llm_response.token_count_prompt + llm_response.token_count_completion
    )

    return ChatResponse(
        conversation_id=conversation_id,
        response=llm_response.content,
        token_count={
            "prompt": llm_response.token_count_prompt,
            "completion": llm_response.token_count_completion,
            "total_this_turn": total_tokens,
            "should_compress": TokenCounter.should_compress(total_tokens, request.model),
        },
        goal=goal,
    )
