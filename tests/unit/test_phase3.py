"""Phase 3 unit tests.

Tests:
- ToolRegistry: load YAML, resolve handler, get_ollama_tool_schemas format
- SkillRepository: CRUD schema validation
- IntentClassifier: output schema
- TaskDecomposer: valid DAG output
- SkillExecutor: template variable resolution {{steps.X.output}}
- Admin API response schemas
"""
import json
import os
import sys
import uuid
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ======================================================================
# 1. ToolRegistry tests
# ======================================================================

class TestToolRegistry:
    """Test ToolRegistry loading, resolution, and schema generation."""

    def _make_registry_with_dummy(self):
        """Create a ToolRegistry with a dummy tool registered programmatically."""
        from src.tools.registry import ToolRegistry
        from src.tools.base import BaseTool

        class DummyHandler(BaseTool):
            async def execute(self, **kwargs: Any) -> dict[str, Any]:
                return {"echo": kwargs}

        registry = ToolRegistry()
        definition = {
            "name": "dummy_tool",
            "description": "A test tool",
            "handler": "test.DummyHandler",
            "input_schema": {
                "type": "object",
                "required": ["msg"],
                "properties": {
                    "msg": {"type": "string", "description": "Message"},
                },
            },
            "output_schema": {
                "type": "object",
                "properties": {"echo": {"type": "object"}},
            },
        }
        registry.register("dummy_tool", definition, DummyHandler())
        return registry

    def test_register_and_list(self):
        registry = self._make_registry_with_dummy()
        assert "dummy_tool" in registry.list_tool_names()
        assert registry.has_tool("dummy_tool")
        assert not registry.has_tool("nonexistent")

    def test_get_definition(self):
        registry = self._make_registry_with_dummy()
        defn = registry.get_definition("dummy_tool")
        assert defn is not None
        assert defn["name"] == "dummy_tool"

    @pytest.mark.asyncio
    async def test_execute(self):
        registry = self._make_registry_with_dummy()
        result = await registry.execute("dummy_tool", {"msg": "hello"})
        assert result == {"echo": {"msg": "hello"}}

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        registry = self._make_registry_with_dummy()
        with pytest.raises(KeyError, match="Tool not registered"):
            await registry.execute("nonexistent_tool", {})

    def test_get_ollama_tool_schemas_format(self):
        """Verify Ollama-compatible schema format."""
        registry = self._make_registry_with_dummy()
        schemas = registry.get_ollama_tool_schemas(["dummy_tool"])

        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["type"] == "function"
        assert "function" in schema
        func = schema["function"]
        assert func["name"] == "dummy_tool"
        assert "description" in func
        assert "parameters" in func
        assert func["parameters"]["type"] == "object"
        assert "msg" in func["parameters"]["properties"]

    def test_get_ollama_tool_schemas_all(self):
        """When tool_names is None, return all registered tools."""
        registry = self._make_registry_with_dummy()
        schemas = registry.get_ollama_tool_schemas(None)
        assert len(schemas) == 1

    def test_get_ollama_tool_schemas_missing(self):
        """Missing tool names are skipped with a warning."""
        registry = self._make_registry_with_dummy()
        schemas = registry.get_ollama_tool_schemas(["dummy_tool", "no_such_tool"])
        assert len(schemas) == 1

    def test_load_yaml_from_directory(self):
        """Test loading YAML files from the real registries/tools directory."""
        from src.tools.registry import ToolRegistry

        registry = ToolRegistry()
        tools_dir = Path(__file__).resolve().parent.parent.parent / "registries" / "tools"

        if not tools_dir.is_dir():
            pytest.skip("registries/tools/ directory not found")

        # This will import real handler classes — they must exist
        registry.load_all(tools_dir)
        assert len(registry.list_tool_names()) >= 2  # at least search_knowledge + summarize

    def test_load_yaml_single(self):
        """Test loading a single YAML tool definition with a mock handler."""
        from src.tools.registry import ToolRegistry
        from src.tools.base import BaseTool

        # Create a temp YAML
        yaml_content = """\
name: test_yaml_tool
version: "1.0"
description: "Test tool loaded from YAML"
handler: src.tools.handlers.summarize.SummarizeHandler

input_schema:
  type: object
  required: [content]
  properties:
    content:
      type: string
      description: Content

output_schema:
  type: object
  properties:
    summary: { type: string }

token_estimate: 100
requires_hitl: false
timeout_seconds: 10
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            registry = ToolRegistry()
            registry.load_yaml(yaml_path)
            assert registry.has_tool("test_yaml_tool")
            schemas = registry.get_ollama_tool_schemas(["test_yaml_tool"])
            assert len(schemas) == 1
            assert schemas[0]["function"]["name"] == "test_yaml_tool"
        finally:
            yaml_path.unlink()


# ======================================================================
# 2. SkillRepository tests
# ======================================================================

class TestSkillRepository:
    """Test skill CRUD schema validation (mocked DB session)."""

    def _make_mock_session(self):
        session = AsyncMock()
        session.flush = AsyncMock()
        session.commit = AsyncMock()
        return session

    @pytest.mark.asyncio
    async def test_create_skill_returns_skill(self):
        from src.skills.repository import create_skill

        session = self._make_mock_session()
        skill = await create_skill(
            session,
            name="test_skill",
            steps=[
                {"id": "s1", "tool": "search_knowledge", "args": {"query": "test"}, "depends_on": []},
            ],
            description="A test skill",
        )
        assert skill.name == "test_skill"
        assert skill.version == "1.0"
        assert skill.is_active is True
        assert len(skill.steps) == 1
        assert skill.steps[0]["id"] == "s1"
        session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_skill_step_schema(self):
        """Verify step structure includes required fields."""
        from src.skills.repository import create_skill

        session = self._make_mock_session()
        steps = [
            {"id": "a", "tool": "summarize", "args": {"content": "hi"}, "depends_on": []},
            {"id": "b", "tool": "search_knowledge", "args": {"query": "x"}, "depends_on": ["a"]},
        ]
        skill = await create_skill(session, name="multi_step", steps=steps)
        assert len(skill.steps) == 2
        for step in skill.steps:
            assert "id" in step
            assert "tool" in step
            assert "args" in step
            assert "depends_on" in step


# ======================================================================
# 3. IntentClassifier tests
# ======================================================================

class TestIntentClassifier:
    """Test IntentClassifier output schema."""

    @pytest.mark.asyncio
    async def test_classify_output_schema(self):
        """Verify the classify method returns the expected keys."""
        from src.orchestrator.intent_classifier import IntentClassifier

        mock_gateway = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "intent": "tool_use",
            "skill": None,
            "required_tools": ["search_knowledge"],
            "complexity": "simple",
            "parameters": {"query": "test"},
        })
        mock_gateway.chat = AsyncMock(return_value=mock_response)

        classifier = IntentClassifier(gateway=mock_gateway)
        result = await classifier.classify(
            user_message="Search for test documents",
            available_skills=[],
            available_tools=["search_knowledge", "summarize"],
        )

        # Schema validation
        assert "intent" in result
        assert result["intent"] in {"skill_match", "tool_use", "chitchat", "clarification_needed"}
        assert "skill" in result
        assert "required_tools" in result
        assert isinstance(result["required_tools"], list)
        assert "complexity" in result
        assert result["complexity"] in {"simple", "moderate", "complex"}
        assert "parameters" in result
        assert isinstance(result["parameters"], dict)

    @pytest.mark.asyncio
    async def test_classify_skill_match(self):
        from src.orchestrator.intent_classifier import IntentClassifier

        mock_gateway = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "intent": "skill_match",
            "skill": "research_and_summarize",
            "required_tools": ["search_knowledge", "summarize"],
            "complexity": "moderate",
            "parameters": {"query": "machine learning"},
        })
        mock_gateway.chat = AsyncMock(return_value=mock_response)

        classifier = IntentClassifier(gateway=mock_gateway)
        result = await classifier.classify(
            user_message="Research and summarize machine learning",
            available_skills=[{"name": "research_and_summarize", "description": "Search and summarize"}],
            available_tools=["search_knowledge", "summarize"],
        )
        assert result["intent"] == "skill_match"
        assert result["skill"] == "research_and_summarize"

    @pytest.mark.asyncio
    async def test_classify_malformed_response(self):
        """Malformed LLM output should return the default/fallback."""
        from src.orchestrator.intent_classifier import IntentClassifier

        mock_gateway = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Not JSON at all"
        mock_gateway.chat = AsyncMock(return_value=mock_response)

        classifier = IntentClassifier(gateway=mock_gateway)
        result = await classifier.classify("hello", [], [])
        assert result["intent"] == "clarification_needed"
        assert result["skill"] is None

    @pytest.mark.asyncio
    async def test_classify_markdown_fenced_response(self):
        """LLM may wrap JSON in markdown fences."""
        from src.orchestrator.intent_classifier import IntentClassifier

        mock_gateway = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '```json\n{"intent":"chitchat","skill":null,"required_tools":[],"complexity":"simple","parameters":{}}\n```'
        mock_gateway.chat = AsyncMock(return_value=mock_response)

        classifier = IntentClassifier(gateway=mock_gateway)
        result = await classifier.classify("hi", [], [])
        assert result["intent"] == "chitchat"


# ======================================================================
# 4. TaskDecomposer tests
# ======================================================================

class TestTaskDecomposer:
    """Test TaskDecomposer output structure."""

    @pytest.mark.asyncio
    async def test_decompose_with_skill_match(self):
        """When a skill exists, decompose returns its steps directly."""
        from src.orchestrator.task_decomposer import TaskDecomposer
        from src.skills.registry import SkillRegistry

        mock_gateway = AsyncMock()
        skill_registry = SkillRegistry()
        skill_registry._cache["my_skill"] = {
            "name": "my_skill",
            "description": "test",
            "steps": [
                {"id": "s1", "tool": "search_knowledge", "args": {"query": "{{parameters.q}}"}, "depends_on": []},
            ],
        }

        decomposer = TaskDecomposer(gateway=mock_gateway, skill_registry=skill_registry)
        intent = {"intent": "skill_match", "skill": "my_skill", "parameters": {"q": "hello"}}
        result = await decomposer.decompose(intent, "hello", ["search_knowledge"])

        assert result["source"] == "skill"
        assert result["skill_name"] == "my_skill"
        assert len(result["steps"]) == 1
        assert result["parameters"] == {"q": "hello"}

    @pytest.mark.asyncio
    async def test_decompose_llm_fallback(self):
        """When no skill matches, use LLM to generate plan."""
        from src.orchestrator.task_decomposer import TaskDecomposer
        from src.skills.registry import SkillRegistry

        llm_plan = json.dumps({
            "steps": [
                {"id": "s1", "tool": "search_knowledge", "args": {"query": "AI"}, "depends_on": []},
                {"id": "s2", "tool": "summarize", "args": {"content": "{{steps.s1.output.chunks}}"}, "depends_on": ["s1"]},
            ]
        })
        mock_gateway = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = llm_plan
        mock_gateway.chat = AsyncMock(return_value=mock_response)

        decomposer = TaskDecomposer(gateway=mock_gateway, skill_registry=SkillRegistry())
        intent = {"intent": "tool_use", "skill": None, "parameters": {}}
        result = await decomposer.decompose(intent, "Summarize AI docs", ["search_knowledge", "summarize"])

        assert result["source"] == "llm"
        assert result["skill_name"] is None
        steps = result["steps"]
        assert len(steps) == 2

        # Validate DAG structure
        for step in steps:
            assert "id" in step
            assert "tool" in step
            assert "args" in step
            assert "depends_on" in step
            assert isinstance(step["depends_on"], list)

        # Check dependency ordering is valid
        step_ids = [s["id"] for s in steps]
        for step in steps:
            for dep in step["depends_on"]:
                assert dep in step_ids, f"Dependency {dep} not in step list"

    @pytest.mark.asyncio
    async def test_decompose_malformed_llm(self):
        """Malformed LLM output should return empty steps."""
        from src.orchestrator.task_decomposer import TaskDecomposer
        from src.skills.registry import SkillRegistry

        mock_gateway = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "I can't produce JSON right now"
        mock_gateway.chat = AsyncMock(return_value=mock_response)

        decomposer = TaskDecomposer(gateway=mock_gateway, skill_registry=SkillRegistry())
        result = await decomposer.decompose({"intent": "tool_use", "skill": None, "parameters": {}}, "test", [])
        assert result["steps"] == []


# ======================================================================
# 5. SkillExecutor tests
# ======================================================================

class TestSkillExecutor:
    """Test SkillExecutor template resolution and DAG ordering."""

    @pytest.mark.asyncio
    async def test_template_resolution_parameters(self):
        """{{parameters.X}} should resolve from input parameters."""
        from src.skills.executor import SkillExecutor
        from src.tools.registry import ToolRegistry
        from src.tools.base import BaseTool

        class EchoTool(BaseTool):
            async def execute(self, **kwargs: Any) -> dict[str, Any]:
                return {"echoed": kwargs}

        registry = ToolRegistry()
        registry.register("echo", {"name": "echo"}, EchoTool())

        executor = SkillExecutor(registry)
        steps = [
            {"id": "s1", "tool": "echo", "args": {"msg": "{{parameters.greeting}}"}, "depends_on": []},
        ]
        result = await executor.run(steps, parameters={"greeting": "hello world"})
        assert result["step_outputs"]["s1"]["echoed"]["msg"] == "hello world"

    @pytest.mark.asyncio
    async def test_template_resolution_step_output(self):
        """{{steps.X.output.key}} should resolve from a previous step's output."""
        from src.skills.executor import SkillExecutor
        from src.tools.registry import ToolRegistry
        from src.tools.base import BaseTool

        class SearchTool(BaseTool):
            async def execute(self, **kwargs: Any) -> dict[str, Any]:
                return {"chunks": ["chunk1", "chunk2"]}

        class SummaryTool(BaseTool):
            async def execute(self, **kwargs: Any) -> dict[str, Any]:
                return {"summary": f"Summarized {kwargs.get('content')}"}

        registry = ToolRegistry()
        registry.register("search", {"name": "search"}, SearchTool())
        registry.register("summary", {"name": "summary"}, SummaryTool())

        executor = SkillExecutor(registry)
        steps = [
            {"id": "s1", "tool": "search", "args": {"query": "test"}, "depends_on": []},
            {"id": "s2", "tool": "summary", "args": {"content": "{{steps.s1.output.chunks}}"}, "depends_on": ["s1"]},
        ]
        result = await executor.run(steps, parameters={})

        # s2 should have received the chunks from s1
        assert result["step_outputs"]["s1"]["chunks"] == ["chunk1", "chunk2"]
        summary = result["step_outputs"]["s2"]["summary"]
        assert "chunk1" in summary
        assert "chunk2" in summary

    @pytest.mark.asyncio
    async def test_topological_order(self):
        """Steps must execute in dependency order."""
        from src.skills.executor import SkillExecutor
        from src.tools.registry import ToolRegistry
        from src.tools.base import BaseTool

        execution_order = []

        class TrackTool(BaseTool):
            async def execute(self, **kwargs: Any) -> dict[str, Any]:
                execution_order.append(kwargs.get("step_id"))
                return {"done": True}

        registry = ToolRegistry()
        registry.register("track", {"name": "track"}, TrackTool())

        executor = SkillExecutor(registry)
        steps = [
            {"id": "c", "tool": "track", "args": {"step_id": "c"}, "depends_on": ["a", "b"]},
            {"id": "a", "tool": "track", "args": {"step_id": "a"}, "depends_on": []},
            {"id": "b", "tool": "track", "args": {"step_id": "b"}, "depends_on": ["a"]},
        ]
        await executor.run(steps, parameters={})

        # a must be before b, b must be before c
        assert execution_order.index("a") < execution_order.index("b")
        assert execution_order.index("b") < execution_order.index("c")

    @pytest.mark.asyncio
    async def test_cycle_detection(self):
        """Cyclic dependencies should raise ValueError."""
        from src.skills.executor import SkillExecutor
        from src.tools.registry import ToolRegistry

        executor = SkillExecutor(ToolRegistry())
        steps = [
            {"id": "a", "tool": "x", "args": {}, "depends_on": ["b"]},
            {"id": "b", "tool": "x", "args": {}, "depends_on": ["a"]},
        ]
        with pytest.raises(ValueError, match="Cycle detected"):
            await executor.run(steps)

    @pytest.mark.asyncio
    async def test_loop_over(self):
        """loop_over should fan-out execution over a list."""
        from src.skills.executor import SkillExecutor
        from src.tools.registry import ToolRegistry
        from src.tools.base import BaseTool

        class ProcessTool(BaseTool):
            async def execute(self, **kwargs: Any) -> dict[str, Any]:
                return {"processed": kwargs.get("item_text", "")}

        registry = ToolRegistry()
        registry.register("process", {"name": "process"}, ProcessTool())

        executor = SkillExecutor(registry)
        steps = [
            {"id": "s1", "tool": "process", "args": {"item_text": "init"}, "depends_on": []},
            {
                "id": "s2",
                "tool": "process",
                "args": {"item_text": "{{item}}"},
                "depends_on": ["s1"],
                "loop_over": "{{parameters.items}}",
            },
        ]
        result = await executor.run(steps, parameters={"items": ["a", "b", "c"]})
        assert len(result["step_outputs"]["s2"]["items"]) == 3

    @pytest.mark.asyncio
    async def test_final_output_is_last_step(self):
        from src.skills.executor import SkillExecutor
        from src.tools.registry import ToolRegistry
        from src.tools.base import BaseTool

        class ReturnTool(BaseTool):
            async def execute(self, **kwargs: Any) -> dict[str, Any]:
                return {"value": kwargs.get("v", 0)}

        registry = ToolRegistry()
        registry.register("ret", {"name": "ret"}, ReturnTool())

        executor = SkillExecutor(registry)
        steps = [
            {"id": "s1", "tool": "ret", "args": {"v": 1}, "depends_on": []},
            {"id": "s2", "tool": "ret", "args": {"v": 2}, "depends_on": ["s1"]},
        ]
        result = await executor.run(steps)
        assert result["final_output"] == {"value": 2}


# ======================================================================
# 6. Admin API tests
# ======================================================================

class TestAdminAPI:
    """Test Admin API response schemas using TestClient."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from src.api.app import create_app
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def mock_session(self):
        """Patch get_session to return a mock async session."""
        session = AsyncMock()
        session.flush = AsyncMock()
        session.commit = AsyncMock()
        return session

    def test_create_skill_schema(self, client, mock_session):
        """POST /admin/skills should accept the correct schema."""
        from src.models.skill import Skill

        # Mock the repository functions
        with patch("src.api.routes.admin.skill_repo") as mock_repo, \
             patch("src.api.routes.admin.get_session") as mock_get_session:

            async def _session():
                yield mock_session

            mock_get_session.return_value = _session()

            mock_repo.get_skill = AsyncMock(return_value=None)
            skill = Skill(
                id=uuid.uuid4(),
                name="test_skill",
                version="1.0",
                description="Test",
                triggers=None,
                steps=[{"id": "s1", "tool": "search_knowledge", "args": {}, "depends_on": []}],
                is_active=True,
            )
            mock_repo.create_skill = AsyncMock(return_value=skill)

            resp = client.post("/admin/skills", json={
                "name": "test_skill",
                "description": "Test",
                "steps": [
                    {"id": "s1", "tool": "search_knowledge", "args": {}, "depends_on": []},
                ],
            })

            assert resp.status_code == 201
            data = resp.json()
            assert data["name"] == "test_skill"
            assert "id" in data
            assert "steps" in data
            assert "is_active" in data

    def test_list_skills_schema(self, client, mock_session):
        """GET /admin/skills should return a list of SkillResponse."""
        from src.models.skill import Skill

        with patch("src.api.routes.admin.skill_repo") as mock_repo, \
             patch("src.api.routes.admin.get_session") as mock_get_session:

            async def _session():
                yield mock_session

            mock_get_session.return_value = _session()

            skill = Skill(
                id=uuid.uuid4(),
                name="skill_a",
                version="1.0",
                description="A",
                triggers=None,
                steps=[{"id": "s1", "tool": "summarize", "args": {}, "depends_on": []}],
                is_active=True,
            )
            mock_repo.list_skills = AsyncMock(return_value=[skill])

            resp = client.get("/admin/skills")
            assert resp.status_code == 200
            data = resp.json()
            assert isinstance(data, list)
            if data:
                item = data[0]
                assert "id" in item
                assert "name" in item
                assert "steps" in item
                assert "is_active" in item

    def test_get_skill_not_found(self, client, mock_session):
        """GET /admin/skills/{name} with unknown name returns 404."""
        with patch("src.api.routes.admin.skill_repo") as mock_repo, \
             patch("src.api.routes.admin.get_session") as mock_get_session:

            async def _session():
                yield mock_session

            mock_get_session.return_value = _session()
            mock_repo.get_skill = AsyncMock(return_value=None)

            resp = client.get("/admin/skills/nonexistent")
            assert resp.status_code == 404

    def test_delete_skill_response(self, client, mock_session):
        """DELETE /admin/skills/{name} should return confirmation."""
        from src.models.skill import Skill

        with patch("src.api.routes.admin.skill_repo") as mock_repo, \
             patch("src.api.routes.admin.get_session") as mock_get_session:

            async def _session():
                yield mock_session

            mock_get_session.return_value = _session()

            skill = Skill(
                id=uuid.uuid4(),
                name="to_delete",
                version="1.0",
                description=None,
                triggers=None,
                steps=[],
                is_active=False,
            )
            mock_repo.deactivate_skill = AsyncMock(return_value=skill)

            resp = client.delete("/admin/skills/to_delete")
            assert resp.status_code == 200
            data = resp.json()
            assert "detail" in data
            assert data["name"] == "to_delete"


# ======================================================================
# 7. TaskGraphExecutor tests
# ======================================================================

class TestTaskGraphExecutor:
    """Test TaskGraphExecutor status tracking and reflection integration."""

    @pytest.mark.asyncio
    async def test_execute_simple_plan(self):
        from src.orchestrator.task_graph import TaskGraphExecutor
        from src.tools.registry import ToolRegistry
        from src.tools.base import BaseTool

        class OkTool(BaseTool):
            async def execute(self, **kwargs: Any) -> dict[str, Any]:
                return {"ok": True}

        registry = ToolRegistry()
        registry.register("ok", {"name": "ok"}, OkTool())

        executor = TaskGraphExecutor(tool_registry=registry, goal={"success_criteria": []})
        plan = {
            "source": "llm",
            "skill_name": None,
            "steps": [
                {"id": "s1", "tool": "ok", "args": {}, "depends_on": []},
            ],
            "parameters": {},
        }
        result = await executor.execute(plan)
        assert result["status"] == "done"
        assert result["node_statuses"]["s1"] == "done"
        assert result["final_output"] == {"ok": True}

    @pytest.mark.asyncio
    async def test_execute_failed_step(self):
        from src.orchestrator.task_graph import TaskGraphExecutor
        from src.tools.registry import ToolRegistry
        from src.tools.base import BaseTool

        class FailTool(BaseTool):
            async def execute(self, **kwargs: Any) -> dict[str, Any]:
                raise RuntimeError("boom")

        registry = ToolRegistry()
        registry.register("fail", {"name": "fail"}, FailTool())

        executor = TaskGraphExecutor(tool_registry=registry, goal={"success_criteria": []})
        plan = {
            "source": "llm",
            "skill_name": None,
            "steps": [
                {"id": "s1", "tool": "fail", "args": {}, "depends_on": []},
            ],
            "parameters": {},
        }
        result = await executor.execute(plan)
        assert result["status"] == "failed"
        assert result["node_statuses"]["s1"] == "failed"

    @pytest.mark.asyncio
    async def test_execute_empty_plan(self):
        from src.orchestrator.task_graph import TaskGraphExecutor
        from src.tools.registry import ToolRegistry

        executor = TaskGraphExecutor(tool_registry=ToolRegistry())
        result = await executor.execute({"source": "llm", "steps": [], "parameters": {}})
        assert result["status"] == "done"
        assert result["final_output"] is None


# ======================================================================
# 8. BaseTool ABC tests
# ======================================================================

class TestBaseTool:

    def test_cannot_instantiate(self):
        from src.tools.base import BaseTool
        with pytest.raises(TypeError):
            BaseTool()

    def test_concrete_subclass(self):
        from src.tools.base import BaseTool

        class Concrete(BaseTool):
            async def execute(self, **kwargs: Any) -> dict[str, Any]:
                return {}

        tool = Concrete()
        assert tool.name == "Concrete"
