"""PointerResolver — resolves ptr:xxx:uuid references to actual data from DB.

Flow:
1. Collect all pointers from state["accumulated_data"] + state["knowledge_context"]
2. Ask LLM which pointers are relevant for the current question
3. Fetch selected pointers from DB (tool_results or chunks table)
4. Return as relevant_data for ContextManager prompt assembly
"""
import json
import logging
import re
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.middleware.llm_gateway import LLMGateway, LLMRequest
from config.settings import settings

logger = logging.getLogger(__name__)

POINTER_RE = re.compile(r"ptr:(tool_result|chunk|task_node):([a-f0-9\-]+)")

SELECT_POINTERS_PROMPT = """You are a context selector. Given a user's question and a list of available data pointers, select which pointers contain information needed to answer the question.

Return ONLY a JSON array of pointer IDs that are relevant. Example: ["ptr:tool_result:abc123", "ptr:chunk:def456"]

If none are relevant, return an empty array: []

Rules:
- Only select pointers whose description is clearly related to the question
- Prefer compressed results over raw when available
- Select at most 5 pointers to keep context manageable
- Return valid JSON array only, no explanation"""


class PointerResolver:
    """Resolves pointers from state to actual data via DB lookup.

    Two-step process:
    1. LLM selects relevant pointers from the full list
    2. DB fetches the actual data for selected pointers
    """

    def __init__(self, gateway: LLMGateway, session: AsyncSession | None = None):
        self._gateway = gateway
        self._session = session

    def collect_pointers(self, state: dict) -> list[dict[str, Any]]:
        """Collect all pointers from state with their metadata.

        Returns list of:
        {
            "ptr": "ptr:tool_result:uuid",
            "desc": "human-readable description",
            "type": "tool_result" | "chunk",
            "uuid": "actual-uuid",
            "token_count_compressed": int (if available),
            "source_key": "key in accumulated_data or knowledge_context"
        }
        """
        pointers = []

        # From accumulated_data
        for key, val in state.get("accumulated_data", {}).items():
            if isinstance(val, dict) and "ptr" in val:
                match = POINTER_RE.match(val["ptr"])
                if match:
                    pointers.append({
                        "ptr": val["ptr"],
                        "desc": val.get("desc", f"Data from {key}"),
                        "type": match.group(1),
                        "uuid": match.group(2),
                        "token_count_compressed": val.get("token_count_compressed"),
                        "token_count_raw": val.get("token_count_raw"),
                        "source_key": key,
                    })

        # From knowledge_context
        for chunk_id in state.get("knowledge_context", {}).get("active_chunk_ids", []):
            # chunk_id format: "chunk:uuid"
            if chunk_id.startswith("chunk:"):
                uuid_part = chunk_id.split(":", 1)[1]
                pointers.append({
                    "ptr": f"ptr:chunk:{uuid_part}",
                    "desc": f"Knowledge chunk {uuid_part[:8]}",
                    "type": "chunk",
                    "uuid": uuid_part,
                    "source_key": f"chunk_{uuid_part[:8]}",
                })

        for doc_ref in state.get("knowledge_context", {}).get("document_refs", []):
            if isinstance(doc_ref, dict) and "ptr" in doc_ref:
                match = POINTER_RE.match(doc_ref["ptr"])
                if match:
                    pointers.append({
                        "ptr": doc_ref["ptr"],
                        "desc": doc_ref.get("desc", doc_ref.get("doc_name", "document")),
                        "type": match.group(1),
                        "uuid": match.group(2),
                        "source_key": f"doc_{match.group(2)[:8]}",
                    })

        logger.info("Collected %d pointers from state", len(pointers))
        return pointers

    async def select_relevant_pointers(
        self,
        user_question: str,
        pointers: list[dict[str, Any]],
        model: str = settings.default_chat_model,
    ) -> list[dict[str, Any]]:
        """Ask LLM to select which pointers are relevant for the current question.

        Args:
            user_question: Current user message
            pointers: All available pointers with metadata

        Returns:
            Subset of pointers that LLM selected as relevant
        """
        if not pointers:
            return []

        # Build pointer catalog for LLM
        catalog = []
        for p in pointers:
            entry = f"- {p['ptr']} | {p['desc']}"
            if p.get("token_count_compressed"):
                entry += f" | ~{p['token_count_compressed']} tokens"
            catalog.append(entry)

        catalog_text = "\n".join(catalog)

        messages = [
            {"role": "system", "content": SELECT_POINTERS_PROMPT},
            {"role": "user", "content": (
                f"## Current Question\n{user_question}\n\n"
                f"## Available Pointers ({len(pointers)} total)\n{catalog_text}"
            )},
        ]

        resp = await self._gateway.chat(LLMRequest(model=model, messages=messages))

        # Parse LLM response
        selected_ptrs = self._parse_pointer_selection(resp.content)

        # Match back to full pointer objects
        ptr_map = {p["ptr"]: p for p in pointers}
        selected = [ptr_map[ptr] for ptr in selected_ptrs if ptr in ptr_map]

        logger.info(
            "LLM selected %d/%d pointers for question: %s",
            len(selected), len(pointers), user_question[:50],
        )
        return selected

    async def fetch_pointer_data(
        self,
        pointers: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Fetch actual data from DB for the selected pointers.

        Returns list of:
        {
            "ptr": "ptr:tool_result:uuid",
            "desc": "description",
            "content": "actual data content (string or JSON)",
        }
        """
        if not self._session:
            logger.warning("No DB session — returning pointer descriptions only")
            return [
                {"ptr": p["ptr"], "desc": p["desc"], "content": f"[No DB session — {p['desc']}]"}
                for p in pointers
            ]

        results = []
        for p in pointers:
            data = await self._fetch_single_pointer(p)
            results.append(data)

        return results

    async def _fetch_single_pointer(self, pointer: dict[str, Any]) -> dict[str, Any]:
        """Fetch a single pointer's data from the appropriate table."""
        ptr_type = pointer["type"]
        ptr_uuid = pointer["uuid"]

        try:
            if ptr_type == "tool_result":
                return await self._fetch_tool_result(ptr_uuid, pointer)
            elif ptr_type == "chunk":
                return await self._fetch_chunk(ptr_uuid, pointer)
            else:
                return {
                    "ptr": pointer["ptr"],
                    "desc": pointer["desc"],
                    "content": f"[Unknown pointer type: {ptr_type}]",
                }
        except Exception as e:
            logger.error("Failed to fetch pointer %s: %s", pointer["ptr"], e)
            return {
                "ptr": pointer["ptr"],
                "desc": pointer["desc"],
                "content": f"[Fetch failed: {e}]",
            }

    async def _fetch_tool_result(
        self, uuid_str: str, pointer: dict[str, Any]
    ) -> dict[str, Any]:
        """Fetch from tool_results table — prefer compressed_output over raw."""
        query = text("""
            SELECT compressed_output, raw_output, tool_name
            FROM tool_results
            WHERE id = :id
            LIMIT 1
        """)
        result = await self._session.execute(query, {"id": uuid_str})
        row = result.fetchone()

        if not row:
            return {
                "ptr": pointer["ptr"],
                "desc": pointer["desc"],
                "content": f"[Tool result {uuid_str[:8]} not found in DB]",
            }

        # Prefer compressed over raw
        data = row.compressed_output or row.raw_output
        content = json.dumps(data, ensure_ascii=False) if isinstance(data, dict) else str(data)

        return {
            "ptr": pointer["ptr"],
            "desc": f"{pointer['desc']} (tool: {row.tool_name})",
            "content": content,
        }

    async def _fetch_chunk(
        self, uuid_str: str, pointer: dict[str, Any]
    ) -> dict[str, Any]:
        """Fetch from chunks table."""
        query = text("""
            SELECT c.content, c.metadata, d.filename
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.id = :id
            LIMIT 1
        """)
        result = await self._session.execute(query, {"id": uuid_str})
        row = result.fetchone()

        if not row:
            return {
                "ptr": pointer["ptr"],
                "desc": pointer["desc"],
                "content": f"[Chunk {uuid_str[:8]} not found in DB]",
            }

        return {
            "ptr": pointer["ptr"],
            "desc": f"{row.filename} (chunk)",
            "content": row.content,
        }

    @staticmethod
    def _parse_pointer_selection(content: str) -> list[str]:
        """Parse LLM response into list of pointer strings."""
        try:
            cleaned = content.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1])
            selected = json.loads(cleaned)
            if isinstance(selected, list):
                return [s for s in selected if isinstance(s, str) and s.startswith("ptr:")]
            return []
        except (json.JSONDecodeError, KeyError):
            # Try to extract ptr: patterns from free text
            return POINTER_RE.findall(content)
