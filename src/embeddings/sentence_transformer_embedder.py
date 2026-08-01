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
        """Initialize the embedding model.

        Args:
            model_name: Hugging Face model identifier.
            device: Torch device such as ``cpu`` or ``cuda``.
            batch_size: Number of texts encoded in each batch.
            normalize_embeddings: Whether to L2-normalize output vectors.
            query_prefix: Prefix added before retrieval queries.

        Raises:
            ValueError: If ``batch_size`` is not positive.
            RuntimeError: If the model does not report its embedding dimension.
        """
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
        """Return the dimensionality of generated embeddings."""
        return self._dimension

    def encode_chunks(
        self,
        chunks: list[Chunk],
    ) -> np.ndarray:
        """Encode document chunks into dense vectors.

        Args:
            chunks: Ordered chunks to encode.

        Returns:
            Float32 array with shape
            ``(number_of_chunks, embedding_dimension)``.
        """
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
        """Encode a retrieval query into one dense vector.

        Args:
            query: User query text.

        Returns:
            Float32 vector with shape ``(embedding_dimension,)``.

        Raises:
            ValueError: If the query is empty or only whitespace.
        """
        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError("query cannot be empty.")

        query_text = f"{self.query_prefix}{cleaned_query}"
        embeddings = self._encode_texts([query_text])

        return embeddings[0]

    def _get_model_dimension(self) -> int | None:
        """Return embedding dimension across library API versions.

        Newer Sentence Transformers versions expose
        ``get_embedding_dimension`` while older versions and some test doubles
        expose ``get_sentence_embedding_dimension``.
        """
        new_api = getattr(
            self.model,
            "get_embedding_dimension",
            None,
        )

        if callable(new_api):
            return new_api()

        legacy_api = getattr(
            self.model,
            "get_sentence_embedding_dimension",
            None,
        )

        if callable(legacy_api):
            return legacy_api()

        return None

    def _encode_texts(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """Encode texts and return a stable two-dimensional NumPy array."""
        embeddings: Any = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )

        array = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if array.ndim == 1:
            array = array.reshape(1, -1)

        if array.ndim != 2:
            raise RuntimeError(
                "The embedding model returned an invalid array shape."
            )

        if array.shape[1] != self.dimension:
            raise RuntimeError(
                "The returned embedding dimension does not match "
                "the model dimension."
            )

        return array
