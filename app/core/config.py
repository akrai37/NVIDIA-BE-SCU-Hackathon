from functools import lru_cache
from typing import Optional
import os

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Application-level configuration loaded from environment variables."""

    nvidia_api_key: Optional[str] = Field(default=None, alias="NVIDIA_API_KEY")
    nvidia_base_url: str = Field(
        default=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    )
    nvidia_embeddings_model: str = Field(
        default=os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/embed-qa-4")
    )
    nvidia_llm_model: str = Field(
        default=os.getenv("NVIDIA_LLM_MODEL", "nvidia/nemotron-mini-4b-instruct")
    )
    chunk_size: int = Field(default=900, ge=200, le=2000)
    chunk_overlap: int = Field(default=150, ge=0, lt=500)
    top_k: int = Field(default=6, ge=1, le=12)
    guidance_temperature: float = Field(default=0.2, ge=0, le=1)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""

    data = {"NVIDIA_API_KEY": os.getenv("NVIDIA_API_KEY")}
    return Settings.model_validate(data)
