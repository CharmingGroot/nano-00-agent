"""TaskGraph execution engine — topological dispatch with status tracking.

Takes a decomposed plan (list of steps) and executes them in order,
dispatching to SkillExecutor for skill-sourced plans or ToolRegistry
for individual tool steps. Integrates with Reflector after each step.
"""
import logging
from typing import Any

from src.tools.registry import ToolRegistry
from src.skills.executor import SkillExecutor
from src.middleware.reflector import Reflector

logger = logging.getLogger(__name__)

# Valid node statuses
PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
SKIPPED = "skipped"


class TaskGraphExecutor:
    """Execute a task graph end-to-end with status tracking and reflection."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        skill_executor: SkillExecutor | None = None,
        goal: dict[str, Any] | None = None,
    ) -> None:
        self._tool_registry = tool_registry
        self._skill_executor = skill_executor or SkillExecutor(tool_registry)
        self._goal = goal or {}
        self._node_status: dict[str, str] = {}
        self._node_outputs: dict[str, Any] = {}
        self._reflections: list[dict[str, Any]] = []

    async def execute(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Execute the full plan.

        Args:
            plan: Output from TaskDecomposer.decompose() —
                  has ``source``, ``steps``, ``parameters``, ``skill_name``.

        Returns:
            {
                "status": "done" | "failed",
                "node_statuses": {step_id: status, ...},
                "node_outputs": {step_id: output, ...},
                "reflections": [...],
                "final_output": <last successful step output>,
            }
        """
        source = plan.get("source", "llm")
        steps = plan.get("steps", [])
        parameters = plan.get("parameters", {})

        if not steps:
            return {
                "status": DONE,
                "node_statuses": {},
                "node_outputs": {},
                "reflections": [],
                "final_output": None,
            }

        # If this plan came from a skill, delegate to SkillExecutor
        if source == "skill":
            return await self._execute_via_skill(steps, parameters)

        # Otherwise, execute steps in topological order ourselves
        return await self._execute_steps(steps, parameters)

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    async def _execute_steps(
        self,
        steps: list[dict[str, Any]],
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Topological walk of steps, one at a time."""
        ordered = _topological_sort(steps)
        context: dict[str, Any] = {"parameters": parameters, "steps": {}}

        # Initialise all as pending
        for s in ordered:
            self._node_status[s["id"]] = PENDING

        overall_status = DONE
        final_output: Any = None

        for step in ordered:
            step_id = step["id"]
            tool_name = step["tool"]
            raw_args = step.get("args", {})

            # Check dependencies
            deps = step.get("depends_on", [])
            dep_failed = any(self._node_status.get(d) == FAILED for d in deps)
            if dep_failed:
                self._node_status[step_id] = SKIPPED
                logger.info("Skipping step %s — dependency failed", step_id)
                continue

            self._node_status[step_id] = RUNNING
            logger.info("TaskGraph: executing step %s (tool=%s)", step_id, tool_name)

            try:
                resolved_args = _resolve_args(raw_args, context)
                result = await self._tool_registry.execute(tool_name, resolved_args)
                # ToolRegistry catches exceptions and returns {"error": ...}
                if isinstance(result, dict) and "error" in result:
                    self._node_status[step_id] = FAILED
                    self._node_outputs[step_id] = result
                    context["steps"][step_id] = {"output": result}
                    overall_status = FAILED
                else:
                    self._node_status[step_id] = DONE
                    self._node_outputs[step_id] = result
                    context["steps"][step_id] = {"output": result}
                    final_output = result
            except Exception as exc:
                logger.exception("TaskGraph: step %s failed", step_id)
                result = {"error": str(exc)}
                self._node_status[step_id] = FAILED
                self._node_outputs[step_id] = result
                context["steps"][step_id] = {"output": result}
                overall_status = FAILED

            # Reflect
            reflection = Reflector.reflect(
                step_completed=step_id,
                step_output=result,
                goal=self._goal,
                task_graph_status={
                    "completed": [s for s, st in self._node_status.items() if st == DONE],
                    "pending": [s for s, st in self._node_status.items() if st == PENDING],
                },
            )
            self._reflections.append(reflection)

            if reflection.get("should_abort"):
                logger.warning("TaskGraph: aborting — reflector says should_abort")
                overall_status = FAILED
                break

        return {
            "status": overall_status,
            "node_statuses": dict(self._node_status),
            "node_outputs": dict(self._node_outputs),
            "reflections": self._reflections,
            "final_output": final_output,
        }

    async def _execute_via_skill(
        self,
        steps: list[dict[str, Any]],
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Delegate to SkillExecutor and wrap result."""
        try:
            result = await self._skill_executor.run(steps=steps, parameters=parameters)
            # Mark all steps done
            for s in steps:
                self._node_status[s["id"]] = DONE
                self._node_outputs[s["id"]] = result["step_outputs"].get(s["id"])

            return {
                "status": DONE,
                "node_statuses": dict(self._node_status),
                "node_outputs": dict(self._node_outputs),
                "reflections": [],
                "final_output": result.get("final_output"),
            }
        except Exception as exc:
            logger.exception("TaskGraph: skill execution failed")
            return {
                "status": FAILED,
                "node_statuses": {s["id"]: FAILED for s in steps},
                "node_outputs": {},
                "reflections": [],
                "final_output": {"error": str(exc)},
            }


# ------------------------------------------------------------------
# Standalone helpers (also used by SkillExecutor)
# ------------------------------------------------------------------

def _topological_sort(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Kahn's algorithm."""
    step_map = {s["id"]: s for s in steps}
    in_degree: dict[str, int] = {s["id"]: 0 for s in steps}
    adjacency: dict[str, list[str]] = {s["id"]: [] for s in steps}

    for s in steps:
        for dep_id in s.get("depends_on", []):
            if dep_id in adjacency:
                adjacency[dep_id].append(s["id"])
                in_degree[s["id"]] += 1

    queue = sorted([sid for sid, deg in in_degree.items() if deg == 0])
    ordered: list[dict[str, Any]] = []

    while queue:
        node = queue.pop(0)
        ordered.append(step_map[node])
        for neighbor in sorted(adjacency[node]):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(ordered) != len(steps):
        raise ValueError("Cycle detected in task graph")

    return ordered


import re

_TEMPLATE_RE = re.compile(r"\{\{(.+?)\}\}")


def _resolve_args(args: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Resolve {{...}} templates in args dict."""
    resolved: dict[str, Any] = {}
    for key, value in args.items():
        resolved[key] = _resolve_value(value, context)
    return resolved


def _resolve_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        full_match = _TEMPLATE_RE.fullmatch(value)
        if full_match:
            return _lookup(full_match.group(1).strip(), context)

        def _replacer(m: re.Match) -> str:
            return str(_lookup(m.group(1).strip(), context))

        return _TEMPLATE_RE.sub(_replacer, value)
    if isinstance(value, dict):
        return {k: _resolve_value(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_value(v, context) for v in value]
    return value


def _lookup(path: str, context: dict[str, Any]) -> Any:
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
