"""ToolResult model — stores raw + compressed tool outputs."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Integer, Float, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.db.base import Base


class ToolResult(Base):
    __tablename__ = "tool_results"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_node_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("task_nodes.id", ondelete="CASCADE"))
    tool_name: Mapped[str] = mapped_column(String(128))
    raw_output: Mapped[dict] = mapped_column(JSONB, nullable=False)
    compressed_output: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    compression_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    token_count_raw: Mapped[int] = mapped_column(Integer, default=0)
    token_count_compressed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    task_node = relationship("TaskNode", back_populates="tool_results")
