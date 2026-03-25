"""Application settings driven by environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All configuration is env-driven. No hardcoded values except defaults."""

    # --- Database ---
    database_url: str = "postgresql+asyncpg://nano:nano@localhost:5432/nano_agent"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434"
    default_chat_model: str = "qwen3.5:9b"
    fallback_chat_model: str = "deepseek-r1:14b"
    embedding_model: str = "nomic-embed-text"
    embedding_dimensions: int = 768

    # --- Token Management ---
    # Context limits per model (tokens)
    context_limit_qwen35_9b: int = 256_000
    context_limit_deepseek_r1_14b: int = 128_000
    # Safety threshold — trigger compression before hitting limit
    token_safety_threshold: int = 150_000
    # Tool-call loop
    max_tool_iterations: int = 10

    # --- Tool Result Compression ---
    tool_result_soft_limit: int = 4_000   # tokens — above this, compress
    tool_result_hard_limit: int = 8_000   # tokens — above this, force truncate
    compression_max_retries: int = 2       # retries for JSON output guarantee

    # --- Knowledge ---
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 50
    retriever_top_k: int = 20

    # --- Logging ---
    log_level: str = "INFO"

    model_config = {"env_prefix": "", "case_sensitive": False}


settings = Settings()
