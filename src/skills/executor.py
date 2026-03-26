"""SkillExecutor — walks a skill's step DAG in topological order.

Each step in a skill's ``steps`` JSONB list looks like::

    {
        "id": "step_1",
        "tool": "search_knowledge",
        "args": {"query": "{{parameters.query}}", "top_k": 10},
        "depends_on": [],
        "loop_over": null          # or "{{steps.step_1.output.chunks}}"
    }

Template variables:
    {{parameters.<key>}}           — resolved from the initial parameters dict
    {{steps.<step_id>.output}}     — full output of a previous step
    {{steps.<step_id>.output.key}} — specific key in a previous step's output
"""
import logging
import re
from typing import Any

from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_TEMPLATE_RE = re.compile(r"\{\{(.+?)\}\}")


class SkillExecutor:
    """Execute a skill's steps in dependency order, resolving templates between steps."""

    def __init__(self, tool_registry: ToolRegistry) -> None:
        self._tool_registry = tool_registry

    async def run(
        self,
        steps: list[dict[str, Any]],
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute all steps in topological order.

        Args:
            steps: The skill's step definitions (from Skill.steps JSONB).
            parameters: Initial input parameters to the skill invocation.

        Returns:
            {"step_outputs": {<step_id>: <output_dict>, ...}, "final_output": <last step output>}
        """
        parameters = parameters or {}
        ordered = self._topological_sort(steps)

        step_outputs: dict[str, Any] = {}
        context = {"parameters": parameters, "steps": {}}

        for step in ordered:
            step_id = step["id"]
            tool_name = step["tool"]
            raw_args = step.get("args", {})
            loop_over_expr = step.get("loop_over")

            logger.info("SkillExecutor: running step %s (tool=%s)", step_id, tool_name)

            if loop_over_expr:
                # Fan-out: resolve the iterable, run the tool once per item
                items = self._resolve_value(loop_over_expr, context)
                if not isinstance(items, list):
                    items = [items]

                results = []
                for idx, item in enumerate(items):
                    iter_ctx = {**context, "item": item, "item_index": idx}
                    resolved_args = self._resolve_args(raw_args, iter_ctx)
                    result = await self._tool_registry.execute(tool_name, resolved_args)
                    results.append(result)

                step_output = {"items": results}
            else:
                resolved_args = self._resolve_args(raw_args, context)
                step_output = await self._tool_registry.execute(tool_name, resolved_args)

            step_outputs[step_id] = step_output
            context["steps"][step_id] = {"output": step_output}

        # The final output is the last step's result
        final_output = step_outputs[ordered[-1]["id"]] if ordered else {}

        return {
            "step_outputs": step_outputs,
            "final_output": final_output,
        }

    # ------------------------------------------------------------------
    # Template resolution
    # ------------------------------------------------------------------

    def _resolve_args(self, args: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Recursively resolve template strings in an args dict."""
        resolved: dict[str, Any] = {}
        for key, value in args.items():
            resolved[key] = self._resolve_value(value, context)
        return resolved

    def _resolve_value(self, value: Any, context: dict[str, Any]) -> Any:
        """Resolve a single value — may be a string with {{...}} or a nested structure."""
        if isinstance(value, str):
            # Check if the entire string is a single template (return native type)
            full_match = _TEMPLATE_RE.fullmatch(value)
            if full_match:
                return self._lookup(full_match.group(1).strip(), context)

            # Otherwise do string interpolation for partial templates
            def _replacer(m: re.Match) -> str:
                resolved = self._lookup(m.group(1).strip(), context)
                return str(resolved)

            return _TEMPLATE_RE.sub(_replacer, value)

        if isinstance(value, dict):
            return {k: self._resolve_value(v, context) for k, v in value.items()}

        if isinstance(value, list):
            return [self._resolve_value(v, context) for v in value]

        return value

    @staticmethod
    def _lookup(path: str, context: dict[str, Any]) -> Any:
        """Resolve a dotted path like ``steps.step_1.output.chunks`` against *context*."""
        parts = path.split(".")
        current: Any = context
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list):
                try:
                    current = current[int(part)]
                except (ValueError, IndexError):
                    return None
            else:
                return None
            if current is None:
                return None
        return current

    # ------------------------------------------------------------------
    # DAG helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _topological_sort(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Kahn's algorithm for topological sort of steps based on depends_on."""
        step_map = {s["id"]: s for s in steps}
        in_degree: dict[str, int] = {s["id"]: 0 for s in steps}
        adjacency: dict[str, list[str]] = {s["id"]: [] for s in steps}

        for s in steps:
            for dep_id in s.get("depends_on", []):
                if dep_id in adjacency:
                    adjacency[dep_id].append(s["id"])
                    in_degree[s["id"]] += 1

        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        ordered: list[dict[str, Any]] = []

        while queue:
            # Sort for deterministic output
            queue.sort()
            node = queue.pop(0)
            ordered.append(step_map[node])
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(ordered) != len(steps):
            executed_ids = {s["id"] for s in ordered}
            remaining = [s["id"] for s in steps if s["id"] not in executed_ids]
            raise ValueError(f"Cycle detected in skill steps — remaining: {remaining}")

        return ordered
