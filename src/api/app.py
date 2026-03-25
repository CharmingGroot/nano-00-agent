"""FastAPI application factory."""
import logging

from fastapi import FastAPI

from src.api.routes import chat, health


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    app = FastAPI(
        title="nano-00-agent",
        description="Public sLLM agent with middleware-driven context management",
        version="0.1.0",
    )

    # Include routers
    app.include_router(health.router, tags=["health"])
    app.include_router(chat.router, prefix="/chat", tags=["chat"])

    return app
