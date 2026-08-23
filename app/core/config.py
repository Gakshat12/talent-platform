"""Configuration settings module for the AI Talent Intelligence Platform using Pydantic Settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or default values."""

    # LLM Settings
    openrouter_api_key: str = Field(default="", description="API key for OpenRouter service")
    llm_model: str = Field(default="openrouter/free", description="Model name for LLM operations")
    llm_max_tokens: int = Field(default=4096, description="Maximum tokens for LLM generation")
    llm_temperature: float = Field(default=0.0, description="Temperature for LLM generation")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", description="Base URL for OpenRouter API"
    )

    # Embedding & Retrieval Settings
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", description="SentenceTransformer embedding model name"
    )
    embedding_dimension: int = Field(default=384, description="Vector dimension of embedding model")
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", description="CrossEncoder reranker model name"
    )
    bm25_top_k: int = Field(default=500, description="Top K results for BM25 sparse retrieval")
    dense_top_k: int = Field(default=500, description="Top K results for FAISS dense retrieval")
    rrf_top_k: int = Field(default=2000, description="Top K candidate pool limit after RRF fusion")
    final_top_k: int = Field(default=100, description="Final top K candidates returned after reranking")

    # Scoring Weights
    weight_evidence_alignment: float = Field(
        default=0.30, description="Weight for skill evidence alignment score component"
    )
    weight_experience_fit: float = Field(
        default=0.25, description="Weight for years of experience fit score component"
    )
    weight_credibility: float = Field(
        default=0.20, description="Weight for career credibility score component"
    )
    weight_hireability: float = Field(
        default=0.15, description="Weight for career progression & hireability score component"
    )

    # Direct name references for scoring weights
    evidence_alignment: float = Field(
        default=0.30, description="Weight for skill evidence alignment score component"
    )
    experience_fit: float = Field(
        default=0.25, description="Weight for years of experience fit score component"
    )
    credibility: float = Field(
        default=0.20, description="Weight for career credibility score component"
    )
    hireability: float = Field(
        default=0.15, description="Weight for career progression & hireability score component"
    )

    # File Paths
    faiss_index_path: str = Field(
        default="indexes/candidates.index", description="Path to saved FAISS index file"
    )
    bm25_index_path: str = Field(
        default="indexes/candidates.bm25.json",
        description="Path to persisted BM25 corpus/index metadata",
    )
    candidates_data_path: str = Field(
        default="data/candidates.jsonl", description="Path to candidates dataset JSONL file"
    )

    # API Server Settings
    api_host: str = Field(default="0.0.0.0", description="Host address for FastAPI server")
    api_port: int = Field(default=8000, description="Port number for FastAPI server")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


# Global module singleton instance
settings = Settings()
