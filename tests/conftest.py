"""Shared test fixtures."""
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all models so SQLAlchemy can resolve relationships
import src.models.conversation  # noqa: F401
import src.models.knowledge  # noqa: F401
import src.models.skill  # noqa: F401
import src.models.task  # noqa: F401
import src.models.tool_result  # noqa: F401
import src.models.state  # noqa: F401
