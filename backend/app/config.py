from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://rag:rag_secret@localhost:5432/ragdb"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:14b"
    embedding_model: str = "intfloat/multilingual-e5-large"
    llm_temperature: float = 0.9
    llm_num_predict: int = 768
    arabic_polish_enabled: bool = True
    max_context_chunks: int = 5
    context_chunk_max_chars: int = 2500
    vector_store_path: Path = Path("./data/vector_store")
    upload_dir: Path = Path("./data/uploads")
    chunk_size: int = 512
    chunk_overlap: int = 64
    turbovec_bit_width: int = 4
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    hybrid_search_enabled: bool = True
    # Optional RAG-Fusion LLM expansion (adds latency; off in production Docker).
    query_expansion_enabled: bool = False
    query_expansion_timeout_seconds: float = 10.0
    # Short LLM query variants for Arabic portfolio chat (cross-lingual retrieval).
    portfolio_query_expansion_enabled: bool = True
    portfolio_query_expansion_timeout_seconds: float = 8.0
    multi_query_count: int = 3
    top_k: int = 5
    retrieval_min_k: int = 10
    retrieval_fetch_multiplier: int = 4
    rrf_k: int = 60
    reranker_min_score: float = -0.5
    context_score_margin: float = 1.5
    portfolio_llm_num_predict: int = 2048
    portfolio_llm_num_predict_medium: int = 1024
    portfolio_llm_num_predict_short: int = 512
    portfolio_max_context_chunks: int = 10
    portfolio_context_chunk_max_chars: int = 2000
    portfolio_context_max_chars: int = 5500
    portfolio_top_k: int = 10
    portfolio_retrieval_min_k: int = 8
    portfolio_retrieval_fetch_multiplier: int = 2
    portfolio_reranker_enabled: bool = True
    torch_device: str = "cuda"

    @field_validator("database_url", mode="before")
    @classmethod
    def _normalize_database_url(cls, v: str) -> str:
        """
        Render commonly injects DATABASE_URL as postgres://... (sync driver).
        SQLAlchemy async engine expects postgresql+asyncpg://...
        """
        if not isinstance(v, str):
            return v

        raw = v.strip()
        if raw.startswith("postgresql+asyncpg://"):
            return raw
        if raw.startswith("postgres://"):
            return "postgresql+asyncpg://" + raw.removeprefix("postgres://")
        if raw.startswith("postgresql://"):
            return "postgresql+asyncpg://" + raw.removeprefix("postgresql://")
        return raw


@lru_cache
def get_settings() -> Settings:
    return Settings()
