#!/usr/bin/env python3
"""Local dev runner that exercises the pipeline without calling NVIDIA endpoints.

Usage:
  python scripts/dev_e2e.py samples/grant_brief.pdf

This script:
- parses a PDF into chunks
- creates a fake embedding service that returns deterministic random vectors
- creates an in-memory vector store and runs a similarity search
- synthesizes a fake guidance payload
- converts the payload into the final response object the API would return

This is useful to validate parsing, chunking, retrieval wiring and response assembly when you don't have NVIDIA credentials.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path so `app` imports work when running the script directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.document_processor import DocumentProcessor
from typing import Iterable, List, Protocol, cast

from app.services.vector_store import InMemoryVectorStore
from app.services.embedding_service import EmbeddingService

from app.services.vector_store import InMemoryVectorStore
from app.services.analysis_service import DocumentAnalyzer


class SupportsEmbedding(Protocol):
    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        ...


class FakeEmbeddingService(SupportsEmbedding):
    """Minimal shim that behaves like EmbeddingService for dev runs."""

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            seed = abs(hash(text)) % (2 ** 32)
            rng = np.random.default_rng(seed)
            vectors.append(rng.random(self.dim).astype(float).tolist())
        return vectors

    def embed_query(self, text: str) -> List[float]:
        seed = abs(hash(text)) % (2 ** 32)
        rng = np.random.default_rng(seed)
        return rng.random(self.dim).astype(float).tolist()


class FakeGuidanceService:
    def generate(self, *, context: str, title: str, pages: int):
        # Minimal, well-formed payload matching the LLM contract used by analysis_service
        return {
            "summary": f"(DEV) Synthetic summary for {title}",
            "key_highlights": ["(DEV) Highlight 1", "(DEV) Highlight 2"],
            "categorized_insights": {
                "critical": [
                    {"label": "Deadline", "description": "There is a submission deadline", "source_chunk_id": None}
                ],
                "important": [],
                "informational": [],
            },
            "extracted_data": [
                {"name": "Total budget", "value": "$10000", "source_chunk_id": None}
            ],
            "recommended_next_steps": [
                {"action": "Assign grant lead", "priority": "important", "rationale": "Prepare application", "due_date": None, "owner": None, "source_chunk_id": None}
            ],
            "references": [],
        }


def run(pdf_path: Path):
    print("[dev_e2e] parsing", pdf_path)
    with open(pdf_path, "rb") as fh:
        data = fh.read()

    processor = DocumentProcessor()
    bundle = processor.process(file_bytes=data, filename=pdf_path.name)
    print(f"[dev_e2e] parsed document: title={bundle.title} pages={bundle.page_count} chunks={len(bundle.chunks)}")

    emb = FakeEmbeddingService(dim=128)
    store = InMemoryVectorStore(cast(EmbeddingService, emb))
    store.add_chunks(bundle.chunks)

    queries = {
        "summary": "Summarise the overall intent, funding purpose, and strategic fit.",
        "deadlines": "List every deadline, reporting cadence, and submission window.",
    }

    scored_contexts = {k: store.similarity_search(q, top_k=3) for k, q in queries.items()}

    # build merged context similar to the real analyzer
    lines = []
    for label, scored in scored_contexts.items():
        lines.append(f"## {label.upper()}")
        for item in scored:
            preview = item.chunk.content.replace("\n", " ")
            lines.append(f"[{item.chunk.chunk_id}] page={item.chunk.page_number} score={item.score:.3f}: {preview[:200]}")
        lines.append("")
    merged_context = "\n".join(lines)

    guidance = FakeGuidanceService().generate(context=merged_context, title=bundle.title, pages=bundle.page_count)

    analyzer = DocumentAnalyzer()
    response = analyzer._to_response(bundle=bundle, scored_contexts=scored_contexts, payload=guidance)

    # print a JSON-friendly dump
    print(json.dumps(response.model_dump(), indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/dev_e2e.py samples/grant_brief.pdf")
        sys.exit(2)
    path = Path(sys.argv[1])
    if not path.exists():
        print("File not found:", path)
        sys.exit(2)
    if path.stat().st_size == 0:
        print(f"File {path} is empty — please provide a non-empty PDF for testing.")
        sys.exit(2)
    run(path)
