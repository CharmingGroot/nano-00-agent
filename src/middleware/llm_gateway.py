"""LLMGateway — single entry point for all Ollama LLM calls.

Every component in the system must call Ollama through this gateway.
Implements: Goal-aware prompt assembly, tool-call loop, result compression, reflection.
"""
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class LLMRequest:
    """Structured request to the LLM."""
    model: str = settings.default_chat_model
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] | None = None
    stream: bool = False
    # Internal flags
    is_compression: bool = False  # Prevent recursive compression
    is_result_compression: bool = False  # Tool result compression call
    conversation_id: str | None = None
    goal: dict | None = None  # Current Goal object


@dataclass
class LLMResponse:
    """Structured response from the LLM."""
    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    token_count_prompt: int = 0
    token_count_completion: int = 0
    model: str = ""
    raw: dict = field(default_factory=dict)


class LLMGateway:
    """Central gateway for all LLM interactions.

    Flow:
    1. Receive LLMRequest
    2. (ContextManager assembles prompt with Goal + state + chunks)
    3. (TokenCounter checks budget)
    4. (StateCompressor if over threshold)
    5. Send to Ollama
    6. If tool_calls in response → execute tools → compress results → loop back to 5
    7. Reflection after each tool call
    8. Return final LLMResponse
    """

    def __init__(self):
        self._client = httpx.AsyncClient(
            base_url=settings.ollama_base_url,
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

    async def chat(self, request: LLMRequest) -> LLMResponse:
        """Send a chat request to Ollama. Core method — all calls go through here."""
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
            "stream": request.stream,
        }
        if request.tools:
            payload["tools"] = request.tools

        logger.info(
            "LLMGateway.chat: model=%s messages=%d tools=%s",
            request.model,
            len(request.messages),
            len(request.tools) if request.tools else 0,
        )

        resp = await self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()

        return LLMResponse(
            content=data.get("message", {}).get("content", ""),
            tool_calls=data.get("message", {}).get("tool_calls"),
            token_count_prompt=data.get("prompt_eval_count", 0),
            token_count_completion=data.get("eval_count", 0),
            model=data.get("model", request.model),
            raw=data,
        )

    async def chat_with_tool_loop(
        self,
        request: LLMRequest,
        tool_executor: Any = None,  # Will be ToolRegistry.execute
        on_tool_result: Any = None,  # Callback for compression + reflection
    ) -> LLMResponse:
        """Chat with automatic tool-call loop.

        Loops until:
        - No more tool_calls in response, OR
        - max_tool_iterations reached
        """
        iteration = 0
        messages = list(request.messages)
        cumulative_prompt_tokens = 0
        cumulative_completion_tokens = 0

        while iteration < settings.max_tool_iterations:
            req = LLMRequest(
                model=request.model,
                messages=messages,
                tools=request.tools,
                stream=False,
                conversation_id=request.conversation_id,
                goal=request.goal,
            )

            response = await self.chat(req)
            cumulative_prompt_tokens += response.token_count_prompt
            cumulative_completion_tokens += response.token_count_completion

            if not response.tool_calls:
                # No more tool calls — return final response
                response.token_count_prompt = cumulative_prompt_tokens
                response.token_count_completion = cumulative_completion_tokens
                return response

            # Process each tool call
            # Add assistant message with tool_calls
            messages.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": response.tool_calls,
            })

            for tool_call in response.tool_calls:
                tool_name = tool_call.get("function", {}).get("name", "unknown")
                tool_args = tool_call.get("function", {}).get("arguments", {})

                logger.info(
                    "Tool call [%d/%d]: %s(%s)",
                    iteration + 1, settings.max_tool_iterations,
                    tool_name, json.dumps(tool_args, ensure_ascii=False)[:200],
                )

                # Execute tool
                if tool_executor:
                    raw_result = await tool_executor(tool_name, tool_args)
                else:
                    raw_result = {"error": f"No executor for tool: {tool_name}"}

                # Compress result if callback provided (Goal-based compression)
                result_content = raw_result
                if on_tool_result:
                    result_content = await on_tool_result(
                        tool_name=tool_name,
                        raw_result=raw_result,
                        goal=request.goal,
                    )

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "content": json.dumps(result_content, ensure_ascii=False)
                    if isinstance(result_content, dict) else str(result_content),
                })

            iteration += 1

        # Max iterations reached — return what we have
        logger.warning(
            "Max tool iterations (%d) reached for conversation %s",
            settings.max_tool_iterations,
            request.conversation_id,
        )
        return LLMResponse(
            content=f"[System: max tool iterations ({settings.max_tool_iterations}) reached. Partial results returned.]",
            token_count_prompt=cumulative_prompt_tokens,
            token_count_completion=cumulative_completion_tokens,
            model=request.model,
        )

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Generate embeddings via Ollama /api/embed endpoint."""
        payload = {
            "model": model or settings.embedding_model,
            "input": texts,
        }
        resp = await self._client.post("/api/embed", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("embeddings", [])

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
