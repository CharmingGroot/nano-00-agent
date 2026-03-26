"""ToolRegistry — loads YAML tool definitions, resolves handler classes, executes tools."""
import importlib
import logging
from pathlib import Path
from typing import Any

import yaml

from src.tools.base import BaseTool

logger = logging.getLogger(__name__)

TOOLS_DIR = Path(__file__).resolve().parent.parent.parent / "registries" / "tools"


class ToolRegistry:
    """Central registry for all tools.

    Loads YAML definitions from registries/tools/, dynamically imports handler classes,
    and provides execute() + get_ollama_tool_schemas() methods.
    """

    def __init__(self) -> None:
        self._definitions: dict[str, dict[str, Any]] = {}
        self._handlers: dict[str, BaseTool] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_all(self, tools_dir: Path | None = None) -> None:
        """Scan *tools_dir* for YAML files and register each tool."""
        directory = tools_dir or TOOLS_DIR
        if not directory.is_dir():
            logger.warning("Tools directory does not exist: %s", directory)
            return

        for yaml_file in sorted(directory.glob("*.yaml")):
            try:
                self.load_yaml(yaml_file)
            except Exception:
                logger.exception("Failed to load tool YAML: %s", yaml_file)

    def load_yaml(self, yaml_path: Path) -> None:
        """Load a single YAML tool definition and resolve its handler class."""
        with open(yaml_path, "r") as f:
            defn = yaml.safe_load(f)

        name = defn["name"]
        handler_path = defn["handler"]  # e.g. src.tools.handlers.search_knowledge.SearchKnowledgeHandler

        # Dynamically import handler class
        module_path, class_name = handler_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        handler_cls = getattr(module, class_name)

        if not (isinstance(handler_cls, type) and issubclass(handler_cls, BaseTool)):
            raise TypeError(
                f"Handler {handler_path} must be a subclass of BaseTool, got {handler_cls}"
            )

        self._definitions[name] = defn
        self._handlers[name] = handler_cls()
        logger.info("Registered tool: %s -> %s", name, handler_path)

    def register(self, name: str, definition: dict[str, Any], handler: BaseTool) -> None:
        """Register a tool programmatically (useful for testing)."""
        self._definitions[name] = definition
        self._handlers[name] = handler

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(self, tool_name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Look up a tool by name and execute it with the given arguments.

        Returns the tool handler's output dict.
        Raises KeyError if the tool is not registered.
        """
        if tool_name not in self._handlers:
            raise KeyError(f"Tool not registered: {tool_name}")

        handler = self._handlers[tool_name]
        safe_args = args or {}
        logger.info("Executing tool: %s with args: %s", tool_name, list(safe_args.keys()))

        try:
            result = await handler.execute(**safe_args)
        except Exception as exc:
            logger.exception("Tool %s execution failed", tool_name)
            result = {"error": str(exc)}

        return result

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    def get_ollama_tool_schemas(self, tool_names: list[str] | None = None) -> list[dict[str, Any]]:
        """Return Ollama-compatible tool definitions for the given tool names.

        Format per tool:
        {
            "type": "function",
            "function": {
                "name": "<tool_name>",
                "description": "<description>",
                "parameters": { <input_schema> }
            }
        }

        If *tool_names* is ``None``, returns schemas for all registered tools.
        """
        names = tool_names if tool_names is not None else list(self._definitions.keys())
        schemas: list[dict[str, Any]] = []

        for name in names:
            defn = self._definitions.get(name)
            if defn is None:
                logger.warning("Tool definition not found for schema: %s", name)
                continue

            schemas.append({
                "type": "function",
                "function": {
                    "name": defn["name"],
                    "description": defn.get("description", ""),
                    "parameters": defn.get("input_schema", {}),
                },
            })

        return schemas

    def get_definition(self, tool_name: str) -> dict[str, Any] | None:
        """Get the raw YAML definition dict for a tool."""
        return self._definitions.get(tool_name)

    def list_tool_names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._definitions.keys())

    def has_tool(self, tool_name: str) -> bool:
        """Check whether a tool is registered."""
        return tool_name in self._handlers
