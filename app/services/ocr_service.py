from __future__ import annotations

import base64
import io
from typing import Any, Protocol

import httpx
from PIL import Image

from app.core.config import get_settings


class OcrUnavailableError(RuntimeError):
    """Raised when OCR cannot be performed due to configuration or runtime issues."""


class OcrServiceProtocol(Protocol):
    def extract_text(self, image: Image.Image) -> str:
        """Return the OCR text for the given image."""
        ...


class NvidiaOcrService(OcrServiceProtocol):
    """Thin wrapper around NVIDIA OCR NIM endpoints."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.nvidia_api_key:
            raise OcrUnavailableError("NVIDIA_API_KEY is not configured for OCR usage.")

        self._client = httpx.Client(
            base_url=settings.nvidia_base_url,
            headers={
                "Authorization": f"Bearer {settings.nvidia_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        self._model = settings.nvidia_ocr_model

    def close(self) -> None:
        self._client.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # pragma: no cover - best effort cleanup
            pass

    def extract_text(self, image: Image.Image) -> str:
        """Return OCR text for the given PIL image."""
        image_bytes = self._encode_image(image)
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_base64": image_bytes,
                        }
                    ],
                }
            ],
        }
        response = self._client.post("/vision/ocr", json=payload)
        response.raise_for_status()
        data = response.json()
        return self._coerce_text(data)

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def _coerce_text(response: Any) -> str:
        if not response:
            return ""
        try:
            choices = response.get("choices") or []
            for choice in choices:
                message = choice.get("message") or {}
                content = message.get("content") or []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return str(part.get("text") or "")
            return response.get("text", "")
        except AttributeError:
            return ""