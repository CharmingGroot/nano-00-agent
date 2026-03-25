"""Generic atomic task runner for Celery workers."""
import logging

logger = logging.getLogger(__name__)


async def execute_task_node(node_id: str, tool_name: str, input_data: dict) -> dict:
    """Execute a single atomic task node.

    This is the worker-side executor. Each task node runs one tool
    with its resolved inputs and returns the output.

    TODO: Wire to ToolRegistry for actual execution in Phase 3.
    """
    logger.info("Executing task node %s: tool=%s", node_id, tool_name)
    return {"status": "not_implemented", "node_id": node_id, "tool_name": tool_name}
