from functools import lru_cache
from typing import Optional
import os
from pathlib import Path

from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Settings(BaseModel):
    """Application-level configuration loaded from environment variables."""

    nvidia_api_key: Optional[str] = Field(default=None, alias="NVIDIA_API_KEY")
    nvidia_base_url: str = Field(
        default=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    )
    nvidia_embeddings_model: str = Field(
        default=os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/nv-embed-qa")
    )
    nvidia_llm_model: str = Field(
        default=os.getenv("NVIDIA_LLM_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")
    )
    chunk_size: int = Field(default=900, ge=200, le=2000)
    chunk_overlap: int = Field(default=150, ge=0, lt=500)
    top_k: int = Field(default=6, ge=1, le=12)
    guidance_temperature: float = Field(default=0.2, ge=0, le=1)
    enable_ocr: bool = Field(default=False, alias="ENABLE_OCR")
    nvidia_ocr_model: str = Field(
        default="nvidia/ocr-nvble-12b-vision", alias="NVIDIA_OCR_MODEL"
    )
    ocr_render_scale: float = Field(
        default=2.0, ge=1.0, le=4.0, alias="OCR_RENDER_SCALE"
    )
    session_timeout_minutes: int = Field(
        default=120,
        ge=5,
        le=1440,
        alias="SESSION_TIMEOUT_MINUTES",
        description="Session timeout in minutes (5-1440, default: 120)",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""

    data = {
        "NVIDIA_API_KEY": os.getenv("NVIDIA_API_KEY"),
        "ENABLE_OCR": os.getenv("ENABLE_OCR", "false"),
        "NVIDIA_OCR_MODEL": os.getenv(
            "NVIDIA_OCR_MODEL", "nvidia/ocr-nvble-12b-vision"
        ),
        "OCR_RENDER_SCALE": os.getenv("OCR_RENDER_SCALE", "2.0"),
        "SESSION_TIMEOUT_MINUTES": os.getenv("SESSION_TIMEOUT_MINUTES", "120"),
    }
    return Settings.model_validate(data)
