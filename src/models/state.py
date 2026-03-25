"""ConversationState model — structured state JSON with Goal."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Integer, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column
from src.db.base import Base


class ConversationState(Base):
    """Stores the current structured state JSON for a conversation.

    This is the compressed representation that includes:
    - goal: structured Goal object with criteria and progress
    - user_intent: classified intent
    - intent_chain: natural language chain of what's been done and why
    - task_graph: current execution state
    - accumulated_data: pointers to results with descriptions
    - knowledge_context: active chunk references
    - token_budget: current token usage
    - hitl_state: human-in-the-loop state
    """
    __tablename__ = "conversation_states"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), unique=True)
    state_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    version: Mapped[int] = mapped_column(Integer, default=1)  # incremented on each compression
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
