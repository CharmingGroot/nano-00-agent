"""SkillRegistry — loads active skills from DB into an in-memory cache."""
import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.models.skill import Skill
from src.skills.repository import list_skills as db_list_skills, get_skill as db_get_skill

logger = logging.getLogger(__name__)


class SkillRegistry:
    """In-memory cache of active skills sourced from the database.

    Call ``await registry.load(session)`` at startup (or ``reload``) to populate.
    """

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Loading / reloading
    # ------------------------------------------------------------------

    async def load(self, session: AsyncSession) -> None:
        """Load all active skills from the DB into memory."""
        skills = await db_list_skills(session, active_only=True)
        self._cache.clear()
        for skill in skills:
            self._cache[skill.name] = self._skill_to_dict(skill)
        logger.info("SkillRegistry loaded %d active skills", len(self._cache))

    async def reload(self, session: AsyncSession) -> None:
        """Alias for load — re-read DB and refresh cache."""
        await self.load(session)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_skill(self, name: str) -> dict[str, Any] | None:
        """Get a cached skill dict by name."""
        return self._cache.get(name)

    def list_skills(self) -> list[dict[str, Any]]:
        """Return all cached skills as dicts."""
        return list(self._cache.values())

    def list_skill_names(self) -> list[str]:
        """Return names of all cached skills."""
        return list(self._cache.keys())

    def has_skill(self, name: str) -> bool:
        return name in self._cache

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _skill_to_dict(skill: Skill) -> dict[str, Any]:
        return {
            "id": str(skill.id),
            "name": skill.name,
            "version": skill.version,
            "description": skill.description,
            "triggers": skill.triggers,
            "steps": skill.steps,
            "is_active": skill.is_active,
        }
