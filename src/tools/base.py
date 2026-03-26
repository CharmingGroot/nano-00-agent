"""BaseTool abstract class for all tool handlers."""
from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Abstract base class for tool handlers.

    Each tool handler implements execute() which performs one atomic action.
    Tool definitions (name, schemas) come from YAML in registries/tools/.
    """

    @abstractmethod
    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the tool with given arguments.

        Returns a dict with the tool's output. Must conform to the tool's output_schema.
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
