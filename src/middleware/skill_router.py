"""SkillRouter — lazy loads tools and skills on demand."""
import logging
from typing import Any

logger = logging.getLogger(__name__)


class SkillRouter:
    """Resolves which tools/skills are needed for a given request.

    Loads tool definitions from YAML registry and skill definitions from DB.
    Only loads what's needed (lazy loading).
    """

    def __init__(self):
        self._loaded_tools: dict[str, Any] = {}
        self._loaded_skills: dict[str, Any] = {}

    def get_tool_schemas(self, tool_names: list[str]) -> list[dict]:
        """Get Ollama-compatible tool schemas for the given tool names."""
        schemas = []
        for name in tool_names:
            if name in self._loaded_tools:
                schemas.append(self._loaded_tools[name].get("schema", {}))
        return schemas

    def register_tool(self, name: str, definition: dict):
        """Register a tool definition (from YAML)."""
        self._loaded_tools[name] = definition
        logger.debug("Registered tool: %s", name)

    def register_skill(self, name: str, definition: dict):
        """Register a skill definition (from DB)."""
        self._loaded_skills[name] = definition
        logger.debug("Registered skill: %s", name)

    def get_skill(self, name: str) -> dict | None:
        """Get a skill definition by name."""
        return self._loaded_skills.get(name)

    def list_tool_names(self) -> list[str]:
        """List all registered tool names."""
        return list(self._loaded_tools.keys())

    def list_skill_names(self) -> list[str]:
        """List all registered skill names."""
        return list(self._loaded_skills.keys())
