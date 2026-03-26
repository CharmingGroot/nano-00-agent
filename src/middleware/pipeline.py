"""Middleware Pipeline — integrates all middleware into a single flow.

This is the REAL integration point that wires together:
GoalGenerator → ContextManager → TokenCounter → StateCompressor
→ SkillRouter → LLMGateway (Tool-Call Loop) → ToolResultCompressor → Reflector

Every user request flows through this pipeline.
"""
import json
import logging
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from src.middleware.llm_gateway import LLMGateway, LLMRequest
from src.middleware.goal_generator import GoalGenerator
from src.middleware.context_manager import ContextManager
from src.middleware.token_counter import TokenCounter
from src.middleware.state_compressor import StateCompressor
from src.middleware.tool_result_compressor import ToolResultCompressor
from src.middleware.reflector import Reflector
from src.middleware.hitl_manager import HITLManager
from src.tools.registry import ToolRegistry
from src.orchestrator.intent_classifier import IntentClassifier
from src.orchestrator.task_decomposer import TaskDecomposer
from src.orchestrator.task_graph import TaskGraphExecutor
from src.skills.executor import SkillExecutor
from src.models.tool_result import ToolResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are nano-00-agent, an AI assistant powered by local LLMs.
You help users with document analysis, research, and various tasks.
Always respond in the user's language.
When you need information, use the available tools."""


class MiddlewarePipeline:
    """End-to-end middleware pipeline for processing user requests.

    Lifecycle per request:
    1. Goal generation
    2. Intent classification
    3. Task decomposition (if complex)
    4. For each step: context assembly → token check → compress if needed → LLM call → tool loop
    5. Reflection after each step
    6. Final response assembly
    """

    def __init__(
        self,
        gateway: LLMGateway,
        tool_registry: ToolRegistry,
        session: AsyncSession | None = None,
    ):
        self._gateway = gateway
        self._tool_registry = tool_registry
        self._session = session
        self._classifier = IntentClassifier()
        self._decomposer = TaskDecomposer()

    async def process(
        self,
        user_message: str,
        conversation_state: dict | None = None,
        model: str = settings.default_chat_model,
        hitl_confirmation: dict | None = None,
    ) -> dict[str, Any]:
        """Process a user request through the full middleware pipeline.

        Returns:
            {
                "response": str,
                "conversation_state": dict,  # updated state JSON
                "goal": dict,
                "token_count": dict,
                "pending_hitl": dict | None,
                "task_progress": dict | None,
            }
        """
        state = conversation_state or self._init_state()
        total_tokens = 0

        # ── Phase A: Goal Generation ──────────────────────────────────
        goal_messages = GoalGenerator.build_goal_messages(user_message)
        goal_req = LLMRequest(model=model, messages=goal_messages)
        goal_resp = await self._gateway.chat(goal_req)
        total_tokens += goal_resp.token_count_prompt + goal_resp.token_count_completion

        goal = GoalGenerator.parse_goal_response(goal_resp.content, user_message)
        goal["goal_id"] = str(uuid.uuid4())
        goal["progress_pct"] = 0
        goal["criteria_status"] = GoalGenerator.init_criteria_status(goal)

        state["goal"] = goal
        state["user_intent"]["original_request"] = user_message
        state["intent_chain"].append(f"사용자 요청: {user_message[:100]}")

        logger.info("Goal: %s", goal.get("final_objective"))

        # ── Phase B: Intent Classification ────────────────────────────
        available_tools = self._tool_registry.list_tool_names()
        classify_result = IntentClassifier.parse_response(
            (await self._gateway.chat(LLMRequest(
                model=model,
                messages=IntentClassifier.build_messages(
                    user_message, available_tools, []
                ),
            ))).content,
            user_message,
        )
        total_tokens += 200  # approximate

        state["user_intent"]["intent"] = classify_result.get("intent", "general_chat")
        complexity = classify_result.get("complexity", 1)

        # ── Phase C: Simple Chat vs Complex Task ──────────────────────
        if complexity <= 1:
            # Simple chat — direct LLM call with Goal context
            return await self._simple_chat(user_message, state, goal, model, total_tokens)

        # ── Phase D: Task Decomposition + Execution ───────────────────
        plan = TaskDecomposer.decompose(
            classify_result=classify_result,
            skills_db={},  # Will load from DB in Phase 5
            user_message=user_message,
        )

        if plan.get("source") == "none":
            return await self._simple_chat(user_message, state, goal, model, total_tokens)

        # Execute task graph
        executor = TaskGraphExecutor(
            tool_registry=self._tool_registry,
            goal=goal,
        )

        # Check HITL for each step before execution
        steps = plan.get("steps", [])
        for step in steps:
            if HITLManager.should_pause(step):
                if not HITLManager.is_confirmed(hitl_confirmation, step.get("tool", "")):
                    state["hitl_state"]["awaiting"] = True
                    state["hitl_state"]["pending_action"] = step.get("tool")
                    return {
                        "response": f"'{step.get('tool')}' 작업을 진행하기 전에 확인이 필요합니다.",
                        "conversation_state": state,
                        "goal": goal,
                        "token_count": {"total_this_turn": total_tokens},
                        "pending_hitl": HITLManager.build_hitl_response(
                            action=step.get("tool", ""),
                            description=f"Step '{step.get('id')}' requires confirmation",
                        ),
                        "task_progress": None,
                    }

        result = await executor.execute(plan)

        # Update state with results
        state["task_graph"] = {
            "status": result["status"],
            "completed": [s for s, st in result["node_statuses"].items() if st == "done"],
            "pending": [],
            "current_step": "complete",
        }

        for reflection in result.get("reflections", []):
            state["intent_chain"].append(reflection.get("intent_chain_entry", ""))

        goal["progress_pct"] = 100 if result["status"] == "done" else 50

        # Compress tool results and store in accumulated_data
        for step_id, output in result.get("node_outputs", {}).items():
            if output and ToolResultCompressor.needs_compression(output):
                # Store raw to DB
                if self._session:
                    tr = ToolResult(
                        task_node_id=None,  # No task_node in this flow yet
                        tool_name=step_id,
                        raw_output=output if isinstance(output, dict) else {"data": str(output)},
                        token_count_raw=ToolResultCompressor.get_token_count(output),
                    )
                    self._session.add(tr)

                # Add pointer to state
                ptr_id = str(uuid.uuid4())
                state["accumulated_data"][step_id] = {
                    "ptr": f"ptr:tool_result:{ptr_id}",
                    "desc": f"{step_id} 결과 (압축본 사용 중)",
                    "token_count_raw": ToolResultCompressor.get_token_count(output),
                }

        # ── Token Budget Check ────────────────────────────────────────
        state["token_budget"]["used"] = total_tokens
        if TokenCounter.should_compress(total_tokens, model):
            # Trigger state compression
            compress_msgs = StateCompressor.build_compression_messages(
                state, [], total_tokens
            )
            compress_resp = await self._gateway.chat(LLMRequest(
                model=model, messages=compress_msgs
            ))
            state = StateCompressor.parse_compressed_state(
                compress_resp.content, state
            )
            logger.info("State compressed at %d tokens", total_tokens)

        # Build final response
        final_output = result.get("final_output")
        if isinstance(final_output, dict):
            response_text = final_output.get("summary", json.dumps(final_output, ensure_ascii=False))
        else:
            response_text = str(final_output) if final_output else "작업이 완료되었습니다."

        return {
            "response": response_text,
            "conversation_state": state,
            "goal": goal,
            "token_count": {"total_this_turn": total_tokens},
            "pending_hitl": None,
            "task_progress": {
                "status": result["status"],
                "total_steps": len(steps),
                "completed_steps": len(result.get("node_statuses", {})),
            },
        }

    async def _simple_chat(
        self,
        user_message: str,
        state: dict,
        goal: dict,
        model: str,
        tokens_so_far: int,
    ) -> dict[str, Any]:
        """Handle simple (non-skill) chat with Goal + context."""
        messages = ContextManager.build_simple_chat_prompt(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            goal=goal,
            state=state,
        )

        # Token check before sending
        msg_tokens = TokenCounter.count_messages_tokens(messages)
        if TokenCounter.should_compress(msg_tokens + tokens_so_far, model):
            compress_msgs = StateCompressor.build_compression_messages(
                state, messages, msg_tokens + tokens_so_far
            )
            compress_resp = await self._gateway.chat(LLMRequest(
                model=model, messages=compress_msgs
            ))
            state = StateCompressor.parse_compressed_state(
                compress_resp.content, state
            )
            # Rebuild messages with compressed state
            messages = ContextManager.build_simple_chat_prompt(
                system_prompt=SYSTEM_PROMPT,
                user_message=user_message,
                goal=goal,
                state=state,
            )

        resp = await self._gateway.chat(LLMRequest(model=model, messages=messages))
        total = tokens_so_far + resp.token_count_prompt + resp.token_count_completion

        state["token_budget"]["used"] = total
        state["intent_chain"].append(f"응답 완료 [토큰: {total}]")

        return {
            "response": resp.content,
            "conversation_state": state,
            "goal": goal,
            "token_count": {
                "prompt": resp.token_count_prompt,
                "completion": resp.token_count_completion,
                "total_this_turn": total,
                "should_compress": TokenCounter.should_compress(total, model),
            },
            "pending_hitl": None,
            "task_progress": None,
        }

    @staticmethod
    def _init_state() -> dict[str, Any]:
        """Initialize a fresh conversation state."""
        return {
            "goal": {},
            "user_intent": {
                "original_request": "",
                "intent": "",
                "skill": "",
                "language": "ko",
            },
            "intent_chain": [],
            "task_graph": {
                "status": "idle",
                "current_step": "",
                "completed": [],
                "pending": [],
            },
            "accumulated_data": {},
            "knowledge_context": {
                "active_chunk_ids": [],
                "document_refs": [],
            },
            "token_budget": {
                "model": settings.default_chat_model,
                "limit": settings.context_limit_qwen35_9b,
                "threshold": settings.token_safety_threshold,
                "used": 0,
            },
            "hitl_state": {
                "awaiting": False,
                "pending_action": None,
            },
        }
