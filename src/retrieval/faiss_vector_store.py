"""FAISS vector store implementation."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from src.preprocessing.chunk import Chunk
from src.retrieval.base_vector_store import BaseVectorStore
from src.retrieval.result import RetrievalResult


class FAISSVectorStore(BaseVectorStore):
    """Vector store backed by a normalized FAISS inner-product index."""

    def __init__(self, dimension: int) -> None:
        """Initialize an empty FAISS vector store.

        Args:
            dimension: Embedding-vector dimension.

        Raises:
            ValueError: If dimension is not positive.
        """
        if dimension <= 0:
            raise ValueError("dimension must be positive.")

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        """Return the number of stored chunks."""
        return len(self._chunks)

    def clear(self) -> None:
        """Remove all indexed vectors and chunks."""
        self.index.reset()
        self._chunks.clear()

    def add(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
    ) -> None:
        """Add chunks and corresponding embeddings to the store.

        Embeddings are converted to contiguous float32 arrays and
        L2-normalized before being inserted into IndexFlatIP. Inner-product
        search over normalized vectors is equivalent to cosine similarity.

        Args:
            chunks: Chunks corresponding to embedding rows.
            embeddings: Matrix shaped ``(len(chunks), dimension)``.

        Raises:
            ValueError: If inputs are empty inconsistently, malformed,
                non-finite, zero-length, or dimensionally incompatible.
            RuntimeError: If the FAISS index and chunk mapping diverge.
        """
        array = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if not chunks:
            if array.size == 0:
                return

            raise ValueError(
                "embeddings must be empty when chunks are empty."
            )

        if array.ndim != 2:
            raise ValueError(
                "embeddings must be a two-dimensional array."
            )

        if array.shape[0] != len(chunks):
            raise ValueError(
                "The number of chunks must match the number "
                "of embedding rows."
            )

        if array.shape[1] != self.dimension:
            raise ValueError(
                "Embedding dimension does not match the store dimension."
            )

        if not np.isfinite(array).all():
            raise ValueError(
                "Embeddings contain NaN or infinite values."
            )

        norms = np.linalg.norm(
            array,
            axis=1,
        )

        if np.any(norms == 0):
            raise ValueError(
                "Zero vectors cannot be added to the vector store."
            )

        normalized = np.ascontiguousarray(
            array.copy(),
            dtype=np.float32,
        )

        faiss.normalize_L2(normalized)

        previous_total = self.index.ntotal

        try:
            self.index.add(normalized)
            self._chunks.extend(chunks)
        except Exception:
            # IndexFlatIP cannot remove only the latest batch reliably here,
            # so fail before mutating whenever possible.
            raise

        expected_total = previous_total + len(chunks)

        if self.index.ntotal != expected_total:
            raise RuntimeError(
                "FAISS did not index the expected number of vectors."
            )

        if self.index.ntotal != len(self._chunks):
            raise RuntimeError(
                "FAISS index and chunk mapping became inconsistent."
            )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Return the most similar stored chunks."""
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError("top_k must be a positive integer.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if self.index.ntotal != len(self._chunks):
            raise RuntimeError(
                "FAISS index and chunk mapping are inconsistent."
            )

        if not self._chunks:
            return []

        query = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query.ndim == 1:
            query = query.reshape(1, -1)
        elif query.ndim != 2 or query.shape[0] != 1:
            raise ValueError(
                "query_embedding must have shape "
                "(dimension,) or (1, dimension)."
            )

        if query.shape[1] != self.dimension:
            raise ValueError(
                "Query embedding dimension does not match "
                "the store dimension."
            )

        if not np.isfinite(query).all():
            raise ValueError(
                "Query embedding contains NaN or infinite values."
            )

        if np.linalg.norm(query) == 0:
            raise ValueError(
                "Query embedding cannot be a zero vector."
            )

        normalized_query = np.ascontiguousarray(
            query.copy(),
            dtype=np.float32,
        )

        faiss.normalize_L2(normalized_query)

        result_count = min(
            top_k,
            len(self._chunks),
        )

        scores, indices = self.index.search(
            normalized_query,
            result_count,
        )

        results: list[RetrievalResult] = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            row_index = int(index)

            if row_index < 0:
                continue

            if row_index >= len(self._chunks):
                raise RuntimeError(
                    "FAISS returned an index outside "
                    "the chunk mapping."
                )

            results.append(
                RetrievalResult(
                    chunk=self._chunks[row_index],
                    score=float(score),
                )
            )

        return results

    def save(self, path: str | Path) -> None:
        """Persist the index and chunk metadata."""
        raise NotImplementedError

    def load(self, path: str | Path) -> None:
        """Load a previously persisted store."""
        raise NotImplementedError
