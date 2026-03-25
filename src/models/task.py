"""TaskGraph and TaskNode models for task decomposition tracking."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Integer, Boolean, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.db.base import Base


class TaskGraph(Base):
    __tablename__ = "task_graphs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|running|paused_hitl|done|failed
    graph_definition: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    nodes: Mapped[list["TaskNode"]] = relationship(back_populates="graph")


class TaskNode(Base):
    __tablename__ = "task_nodes"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    graph_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("task_graphs.id", ondelete="CASCADE"))
    tool_name: Mapped[str] = mapped_column(String(128))
    step_index: Mapped[int] = mapped_column(Integer)
    depends_on: Mapped[list] = mapped_column(JSONB, default=list)
    input_mapping: Mapped[dict] = mapped_column(JSONB, default=dict)
    output: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|running|done|failed|skipped
    hitl_required: Mapped[bool] = mapped_column(Boolean, default=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0)

    graph: Mapped["TaskGraph"] = relationship(back_populates="nodes")
    tool_results: Mapped[list["ToolResult"]] = relationship(back_populates="task_node")


# Import here to avoid circular
from src.models.tool_result import ToolResult  # noqa: E402, F401
