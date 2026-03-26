"""FastAPI application factory."""
import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.routes import admin, chat, health, knowledge

# Import all models so SQLAlchemy can resolve relationships at startup
import src.models.conversation  # noqa: F401
import src.models.knowledge  # noqa: F401
import src.models.skill  # noqa: F401
import src.models.task  # noqa: F401
import src.models.tool_result  # noqa: F401
import src.models.state  # noqa: F401

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    app = FastAPI(
        title="nano-00-agent",
        description="Public sLLM agent with middleware-driven context management",
        version="0.1.0",
    )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Return detailed error info instead of generic 500."""
        tb = traceback.format_exc()
        logger.error("Unhandled exception: %s\n%s", exc, tb)
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "type": type(exc).__name__, "detail": tb[-500:]},
        )

    # Include routers
    app.include_router(health.router, tags=["health"])
    app.include_router(chat.router, prefix="/chat", tags=["chat"])
    app.include_router(knowledge.router, prefix="/knowledge", tags=["knowledge"])
    app.include_router(admin.router, prefix="/admin", tags=["admin"])

    return app
