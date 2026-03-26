"""Skill DB CRUD — async functions for the Skill model."""
import uuid
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.skill import Skill

logger = logging.getLogger(__name__)


async def create_skill(
    session: AsyncSession,
    *,
    name: str,
    steps: list[dict[str, Any]],
    description: str | None = None,
    version: str = "1.0",
    triggers: dict[str, Any] | None = None,
    is_active: bool = True,
) -> Skill:
    """Insert a new Skill row and return it."""
    skill = Skill(
        id=uuid.uuid4(),
        name=name,
        version=version,
        description=description,
        triggers=triggers,
        steps=steps,
        is_active=is_active,
    )
    session.add(skill)
    await session.flush()
    logger.info("Created skill: %s (id=%s)", name, skill.id)
    return skill


async def get_skill(session: AsyncSession, *, name: str) -> Skill | None:
    """Fetch a single Skill by name."""
    stmt = select(Skill).where(Skill.name == name)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_skill_by_id(session: AsyncSession, *, skill_id: uuid.UUID) -> Skill | None:
    """Fetch a single Skill by primary key."""
    stmt = select(Skill).where(Skill.id == skill_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_skills(
    session: AsyncSession,
    *,
    active_only: bool = True,
) -> list[Skill]:
    """Return all skills, optionally filtered to active ones."""
    stmt = select(Skill).order_by(Skill.name)
    if active_only:
        stmt = stmt.where(Skill.is_active.is_(True))
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def update_skill(
    session: AsyncSession,
    *,
    name: str,
    **fields: Any,
) -> Skill | None:
    """Update a Skill's columns by name. Returns the updated Skill or None."""
    # Only allow safe columns
    allowed = {"description", "version", "triggers", "steps", "is_active"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        return await get_skill(session, name=name)

    updates["updated_at"] = datetime.now(timezone.utc)

    stmt = (
        update(Skill)
        .where(Skill.name == name)
        .values(**updates)
        .returning(Skill)
    )
    result = await session.execute(stmt)
    row = result.scalar_one_or_none()
    if row:
        logger.info("Updated skill: %s fields=%s", name, list(updates.keys()))
    return row


async def delete_skill(session: AsyncSession, *, name: str) -> bool:
    """Hard-delete a Skill. Returns True if a row was deleted."""
    stmt = delete(Skill).where(Skill.name == name)
    result = await session.execute(stmt)
    deleted = result.rowcount > 0
    if deleted:
        logger.info("Deleted skill: %s", name)
    return deleted


async def deactivate_skill(session: AsyncSession, *, name: str) -> Skill | None:
    """Soft-delete by setting is_active=False."""
    return await update_skill(session, name=name, is_active=False)
