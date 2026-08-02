"""Sentence Transformers embedding implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.embeddings.base_embedder import BaseEmbedder
from src.preprocessing.chunk import Chunk


class SentenceTransformerEmbedder(BaseEmbedder):
    """Generate dense embeddings using Sentence Transformers."""

    DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        query_prefix: str = (
            "Represent this sentence for searching relevant passages: "
        ),
    ) -> None:
        """Initialize the embedding model."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.query_prefix = query_prefix

        self.model = SentenceTransformer(
            model_name_or_path=model_name,
            device=device,
        )

        model_dimension = self._get_model_dimension()

        if model_dimension is None:
            raise RuntimeError(
                "The embedding model did not report its dimension."
            )

        self._dimension = int(model_dimension)

    @property
    def dimension(self) -> int:
        """Return embedding dimensionality."""
        return self._dimension

    def encode_chunks(
        self,
        chunks: list[Chunk],
    ) -> np.ndarray:
        """Encode document chunks."""
        if not chunks:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        texts = [chunk.text for chunk in chunks]

        return self._encode_texts(texts)

    def encode_query(
        self,
        query: str,
    ) -> np.ndarray:
        """Encode a retrieval query."""
        query = query.strip()

        if not query:
            raise ValueError("query cannot be empty.")

        query = f"{self.query_prefix}{query}"

        embeddings = self._encode_texts([query])

        return embeddings[0]

    def _get_model_dimension(self) -> int | None:
        """Return embedding dimension across library versions."""

        method = getattr(
            self.model,
            "get_embedding_dimension",
            None,
        )

        if callable(method):
            return method()

        method = getattr(
            self.model,
            "get_sentence_embedding_dimension",
            None,
        )

        if callable(method):
            return method()

        return None

    def _encode_texts(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """Encode text into float32 embeddings."""

        embeddings: Any = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.ndim != 2:
            raise RuntimeError(
                "Expected a 2-D embedding matrix."
            )

        if embeddings.shape[1] != self.dimension:
            raise RuntimeError(
                "Embedding dimension mismatch."
            )

        if not np.isfinite(embeddings).all():
            raise RuntimeError(
                "Embeddings contain NaN or Inf values."
            )

        return embeddings
