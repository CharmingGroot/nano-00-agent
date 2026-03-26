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
from src.middleware.pointer_resolver import PointerResolver

logger = logging.getLogger(__name__)

BASE_SYSTEM_PROMPT = """You are nano-00-agent, an AI assistant powered by local LLMs.
You have access to an internal knowledge base with uploaded documents (PDF, CSV, XLSX).

IMPORTANT: When users ask about data that might be in the knowledge base,
always search the knowledge base first before answering from general knowledge.

Always respond in the user's language."""


def build_system_prompt(tool_registry: ToolRegistry) -> str:
    """Build system prompt dynamically from registered tools. No hardcoding."""
    tools = tool_registry.list_tool_names()
    if not tools:
        return BASE_SYSTEM_PROMPT

    tool_descriptions = []
    for name in tools:
        defn = tool_registry.get_definition(name)
        if defn:
            desc = defn.get("description", name)
            tool_descriptions.append(f"- {name}: {desc.strip()[:100]}")

    tools_section = "\n".join(tool_descriptions)
    return f"{BASE_SYSTEM_PROMPT}\n\nAvailable tools:\n{tools_section}"


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
        self._system_prompt = build_system_prompt(tool_registry) if tool_registry else BASE_SYSTEM_PROMPT
        self._classifier = IntentClassifier(gateway)
        self._pointer_resolver = PointerResolver(gateway, session)
        from src.skills.registry import SkillRegistry as SkillReg
        self._skill_registry = SkillReg()
        self._decomposer = TaskDecomposer(gateway, self._skill_registry)

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
        # Pass tool names WITH descriptions so LLM can make informed choices
        available_tool_infos = []
        for name in self._tool_registry.list_tool_names():
            defn = self._tool_registry.get_definition(name)
            available_tool_infos.append({
                "name": name,
                "description": defn.get("description", name) if defn else name,
            })

        classify_result = await self._classifier.classify(
            user_message=user_message,
            available_skills=[],
            available_tools=available_tool_infos,
        )
        total_tokens += 200  # approximate

        intent = classify_result.get("intent", "general_chat")
        state["user_intent"]["intent"] = intent
        try:
            complexity = int(classify_result.get("complexity", 1))
        except (ValueError, TypeError):
            complexity = 1

        # If intent suggests tool use or skill match, ensure complexity >= 2
        required_tools = classify_result.get("required_tools", [])
        skill_match = classify_result.get("skill")
        if required_tools or skill_match or intent in ("tool_use", "knowledge_qa", "research"):
            complexity = max(complexity, 2)

        logger.info("Intent=%s complexity=%d tools=%s", intent, complexity, required_tools)

        # ── Phase C: Route based on required_tools from IntentClassifier ──
        # No hardcoded keywords — routing is entirely LLM-driven.
        # Any tool in required_tools that is a "source" (search_knowledge, web_search,
        # query_dataset, etc.) triggers source-augmented chat.
        source_tools = [t for t in required_tools if self._tool_registry.has_tool(t)]

        if complexity <= 1 and not source_tools:
            return await self._simple_chat(user_message, state, goal, model, total_tokens)

        if source_tools:
            # Source-augmented chat: execute source tools first, then answer with context
            return await self._source_augmented_chat(
                user_message, state, goal, model, total_tokens, source_tools
            )

        if complexity <= 3 and required_tools:
            return await self._tool_enabled_chat(
                user_message, state, goal, model, total_tokens, required_tools
            )

        # ── Phase D: Task Decomposition + Execution (complexity > 3) ──
        plan = await self._decomposer.decompose(
            classify_result=classify_result,
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

    async def _source_augmented_chat(
        self,
        user_message: str,
        state: dict,
        goal: dict,
        model: str,
        tokens_so_far: int,
        source_tools: list[str],
    ) -> dict[str, Any]:
        """Execute source tools first, then answer with retrieved context.

        "Source" = any registered tool that retrieves data. The IntentClassifier
        query_dataset, etc.). The middleware calls them directly — more reliable
        than waiting for sLLM to generate tool_calls.

        No hardcoding: any tool registered in ToolRegistry can be a source.
        The IntentClassifier decides which source tools to use.
        """
        all_source_data = []

        for tool_name in source_tools:
            logger.info("Source-augmented: executing %s for '%s'", tool_name, user_message[:50])

            try:
                # Build generic args — all source tools accept "query"
                tool_args = {"query": user_message}
                result = await self._tool_registry.execute(tool_name, tool_args)
            except Exception as e:
                logger.warning("Source tool %s failed: %s", tool_name, e)
                continue

            # Store raw result as pointer
            ptr_id = str(uuid.uuid4())
            result_text = json.dumps(result, ensure_ascii=False)
            raw_tokens = TokenCounter.count_tokens(result_text)

            state["accumulated_data"][f"{tool_name}_{ptr_id[:8]}"] = {
                "ptr": f"ptr:tool_result:{ptr_id}",
                "desc": f"{tool_name} 결과 ({user_message[:30]})",
                "token_count_raw": raw_tokens,
            }

            # Compress if needed
            if ToolResultCompressor.needs_compression(result):
                compress_msgs = ToolResultCompressor.build_compression_messages(goal, result)
                try:
                    compress_resp = await self._gateway.chat(LLMRequest(model=model, messages=compress_msgs))
                    compressed = ToolResultCompressor.parse_compressed_response(
                        compress_resp.content, result, ptr_id
                    )
                    all_source_data.append({
                        "tool": tool_name,
                        "data": compressed,
                        "raw_tokens": raw_tokens,
                    })
                    state["accumulated_data"][f"{tool_name}_{ptr_id[:8]}"]["token_count_compressed"] = (
                        ToolResultCompressor.get_token_count(compressed)
                    )
                except Exception as e:
                    logger.warning("Compression failed for %s: %s", tool_name, e)
                    all_source_data.append({"tool": tool_name, "data": result, "raw_tokens": raw_tokens})
            else:
                all_source_data.append({"tool": tool_name, "data": result, "raw_tokens": raw_tokens})

            # Reflection
            reflection = Reflector.reflect(
                step_completed=tool_name,
                step_output=result,
                goal=goal,
                task_graph_status={
                    "completed": list(state["accumulated_data"].keys()),
                    "pending": [],
                },
            )
            state["intent_chain"].append(reflection["intent_chain_entry"])

        if not all_source_data:
            state["intent_chain"].append("소스 검색 결과 없음 — 일반 대화로 전환")
            return await self._simple_chat(user_message, state, goal, model, tokens_so_far)

        # Format all source results as context
        source_context_parts = []
        for sd in all_source_data:
            data = sd["data"]
            if isinstance(data, dict):
                # Unified: extract items from any source tool response
                items = data.get("chunks", data.get("results", data.get("items", [])))
                if isinstance(items, list):
                    for item in items[:5]:
                        if isinstance(item, dict):
                            label = item.get("document_name", item.get("title", sd["tool"]))
                            score = item.get("score", "")
                            content = item.get("content", item.get("data", json.dumps(item, ensure_ascii=False)))
                            score_str = f" | 유사도: {score:.2f}" if isinstance(score, (int, float)) else ""
                            source_context_parts.append(
                                f"[소스: {sd['tool']} | {label}{score_str}]\n{content}"
                            )
                else:
                    source_context_parts.append(
                        f"[소스: {sd['tool']}]\n{json.dumps(data, ensure_ascii=False)[:2000]}"
                    )
            else:
                source_context_parts.append(str(data)[:2000])

        source_context = "\n\n".join(source_context_parts)

        # Resolve previous turn pointers
        previous_data = await self._resolve_pointers_for_question(user_message, state, model)
        previous_context = ""
        if previous_data:
            previous_context = "\n\n## 이전 대화에서 참조한 데이터\n" + "\n".join(
                f"[{d['ptr']}] {d['desc']}\n{d['content']}" for d in previous_data
            )

        # Build prompt
        messages = [
            {
                "role": "system",
                "content": (
                    f"{self._system_prompt}\n\n"
                    f"## Current Goal\n{goal.get('final_objective', 'N/A')}\n\n"
                    f"## 검색된 소스 데이터 (반드시 이 데이터를 기반으로 답변하세요)\n{source_context}"
                    f"{previous_context}"
                ),
            },
            {"role": "user", "content": user_message},
        ]

        resp = await self._gateway.chat(LLMRequest(model=model, messages=messages))
        total = tokens_so_far + resp.token_count_prompt + resp.token_count_completion

        state["token_budget"]["used"] = total
        state["intent_chain"].append(f"소스 기반 응답 완료 [토큰: {total}]")

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

    async def _tool_enabled_chat(
        self,
        user_message: str,
        state: dict,
        goal: dict,
        model: str,
        tokens_so_far: int,
        required_tools: list[str],
    ) -> dict[str, Any]:
        """Chat with tools enabled — LLM can call tools via tool-call loop.

        Used for medium-complexity requests (2-3) where the LLM should
        decide which tools to call and in what order.
        """
        # Get tool schemas for Ollama
        tool_schemas = self._tool_registry.get_ollama_tool_schemas(required_tools)
        if not tool_schemas:
            tool_schemas = self._tool_registry.get_ollama_tool_schemas(
                self._tool_registry.list_tool_names()
            )

        # ── Resolve pointers from previous turns ──────────────────────
        relevant_data = await self._resolve_pointers_for_question(
            user_message, state, model
        )

        messages = ContextManager.assemble_prompt(
            system_prompt=self._system_prompt,
            goal=goal,
            state=state,
            current_step=None,
            step_instruction=user_message,
            relevant_data=relevant_data,
            tool_schemas=tool_schemas,
        )

        async def tool_executor(tool_name: str, tool_args: dict) -> dict:
            return await self._tool_registry.execute(tool_name, tool_args)

        async def on_tool_result(tool_name: str, raw_result: Any, goal: dict | None) -> Any:
            """Compress tool results if needed, store raw to state."""
            raw_tokens = ToolResultCompressor.get_token_count(raw_result)

            # Store pointer in accumulated_data
            ptr_id = str(uuid.uuid4())
            state["accumulated_data"][tool_name] = {
                "ptr": f"ptr:tool_result:{ptr_id}",
                "desc": f"{tool_name} 결과 ({raw_tokens} 토큰)",
                "token_count_raw": raw_tokens,
            }

            if ToolResultCompressor.needs_compression(raw_result):
                # Compress via LLM
                compress_msgs = ToolResultCompressor.build_compression_messages(
                    goal or {}, raw_result
                )
                compress_resp = await self._gateway.chat(LLMRequest(
                    model=model, messages=compress_msgs
                ))
                compressed = ToolResultCompressor.parse_compressed_response(
                    compress_resp.content, raw_result, ptr_id
                )
                state["accumulated_data"][tool_name]["token_count_compressed"] = (
                    ToolResultCompressor.get_token_count(compressed)
                )

                # Reflection
                reflection = Reflector.reflect(
                    step_completed=tool_name,
                    step_output=compressed,
                    goal=goal or {},
                    task_graph_status={
                        "completed": list(state["accumulated_data"].keys()),
                        "pending": [],
                    },
                )
                state["intent_chain"].append(reflection["intent_chain_entry"])

                return compressed

            # No compression needed
            reflection = Reflector.reflect(
                step_completed=tool_name,
                step_output=raw_result,
                goal=goal or {},
                task_graph_status={
                    "completed": list(state["accumulated_data"].keys()),
                    "pending": [],
                },
            )
            state["intent_chain"].append(reflection["intent_chain_entry"])
            return raw_result

        # Run through tool-call loop
        request = LLMRequest(
            model=model,
            messages=messages,
            tools=tool_schemas,
            goal=goal,
        )
        response = await self._gateway.chat_with_tool_loop(
            request=request,
            tool_executor=tool_executor,
            on_tool_result=on_tool_result,
        )

        total = tokens_so_far + response.token_count_prompt + response.token_count_completion
        state["token_budget"]["used"] = total
        state["intent_chain"].append(f"응답 완료 [토큰: {total}]")

        return {
            "response": response.content,
            "conversation_state": state,
            "goal": goal,
            "token_count": {
                "prompt": response.token_count_prompt,
                "completion": response.token_count_completion,
                "total_this_turn": total,
                "should_compress": TokenCounter.should_compress(total, model),
            },
            "pending_hitl": None,
            "task_progress": None,
        }

    async def _resolve_pointers_for_question(
        self,
        user_message: str,
        state: dict,
        model: str,
    ) -> list[dict[str, Any]]:
        """Resolve relevant pointers from previous turns for the current question.

        1. Collect all pointers from state
        2. If pointers exist, ask LLM to select relevant ones
        3. Fetch selected pointer data from DB
        """
        pointers = self._pointer_resolver.collect_pointers(state)
        if not pointers:
            return []

        try:
            selected = await self._pointer_resolver.select_relevant_pointers(
                user_message, pointers, model
            )
            if not selected:
                return []

            resolved = await self._pointer_resolver.fetch_pointer_data(selected)
            logger.info(
                "Resolved %d pointers for question: %s",
                len(resolved), user_message[:50],
            )
            return resolved
        except Exception as e:
            logger.warning("Pointer resolution failed: %s — continuing without context", e)
            return []

    async def _simple_chat(
        self,
        user_message: str,
        state: dict,
        goal: dict,
        model: str,
        tokens_so_far: int,
    ) -> dict[str, Any]:
        """Handle simple (non-skill) chat with Goal + context."""
        # Resolve pointers from previous turns
        relevant_data = await self._resolve_pointers_for_question(
            user_message, state, model
        )

        messages = ContextManager.build_simple_chat_prompt(
            system_prompt=self._system_prompt,
            user_message=user_message,
            goal=goal,
            state=state,
        )

        # Inject resolved pointer data into system prompt if available
        if relevant_data:
            data_section = "\n\n## Retrieved Context from Previous Turns\n"
            for item in relevant_data:
                data_section += f"\n[{item['ptr']}] {item['desc']}\n{item['content']}\n"
            messages[0]["content"] += data_section

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
                system_prompt=self._system_prompt,
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
