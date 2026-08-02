"""FAISS vector store implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.preprocessing.chunk import Chunk
from src.retrieval.base_vector_store import BaseVectorStore
from src.retrieval.result import RetrievalResult


class FAISSVectorStore(BaseVectorStore):
    """Store and retrieve chunks using a normalized FAISS inner-product index."""

    INDEX_FILENAME = "index.faiss"
    METADATA_FILENAME = "chunks.json"
    FORMAT_VERSION = 1

    def __init__(self, dimension: int) -> None:
        """Initialize an empty vector store."""
        if not isinstance(dimension, int) or isinstance(dimension, bool):
            raise ValueError("dimension must be a positive integer.")
        if dimension <= 0:
            raise ValueError("dimension must be positive.")

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        """Return the number of stored chunks."""
        return len(self._chunks)

    @property
    def ntotal(self) -> int:
        """Return the number of vectors stored by FAISS."""
        return int(self.index.ntotal)

    def clear(self) -> None:
        """Remove all vectors and chunks."""
        self.index.reset()
        self._chunks.clear()

    def add(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
    ) -> None:
        """Add chunks and their corresponding embeddings."""
        array = np.asarray(embeddings, dtype=np.float32)

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

        norms = np.linalg.norm(array, axis=1)
        if np.any(norms == 0):
            raise ValueError(
                "Zero vectors cannot be added to the vector store."
            )

        normalized = np.ascontiguousarray(
            array.copy(),
            dtype=np.float32,
        )
        faiss.normalize_L2(normalized)

        self.index.add(normalized)
        self._chunks.extend(chunks)
        self._validate_internal_state()

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

        self._validate_internal_state()

        if not self._chunks:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)

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

        result_count = min(top_k, len(self._chunks))
        scores, indices = self.index.search(
            normalized_query,
            result_count,
        )

        results: list[RetrievalResult] = []

        for score, index in zip(scores[0], indices[0]):
            row_index = int(index)

            if row_index < 0:
                continue
            if row_index >= len(self._chunks):
                raise RuntimeError(
                    "FAISS returned an index outside the chunk mapping."
                )

            results.append(
                RetrievalResult(
                    chunk=self._chunks[row_index],
                    score=float(score),
                )
            )

        return results

    def save(self, path: str | Path) -> None:
        """Persist the FAISS index and chunk data."""
        self._validate_internal_state()

        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)

        index_path = directory / self.INDEX_FILENAME
        metadata_path = directory / self.METADATA_FILENAME

        payload = {
            "format_version": self.FORMAT_VERSION,
            "dimension": self.dimension,
            "chunks": [
                self._chunk_to_dict(chunk)
                for chunk in self._chunks
            ],
        }

        try:
            serialized = json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Chunk metadata must be JSON-serializable."
            ) from exc

        faiss.write_index(self.index, str(index_path))
        metadata_path.write_text(
            serialized + "\n",
            encoding="utf-8",
        )

    def load(self, path: str | Path) -> None:
        """Load a persisted FAISS index and chunk mapping."""
        directory = Path(path)
        index_path = directory / self.INDEX_FILENAME
        metadata_path = directory / self.METADATA_FILENAME

        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)

        try:
            payload = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Stored chunk metadata is not valid JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError(
                "Stored vector-store metadata must be a JSON object."
            )

        if payload.get("format_version") != self.FORMAT_VERSION:
            raise ValueError(
                "Unsupported vector-store format version."
            )

        if payload.get("dimension") != self.dimension:
            raise ValueError(
                "Stored dimension does not match this vector store."
            )

        chunk_records = payload.get("chunks")
        if not isinstance(chunk_records, list):
            raise ValueError(
                "Stored chunk mapping must be a list."
            )

        loaded_index = faiss.read_index(str(index_path))
        if loaded_index.d != self.dimension:
            raise ValueError(
                "Loaded FAISS index dimension does not match "
                "the store dimension."
            )

        loaded_chunks = [
            self._chunk_from_dict(record)
            for record in chunk_records
        ]

        if loaded_index.ntotal != len(loaded_chunks):
            raise RuntimeError(
                "Loaded index and chunk mapping contain "
                "different numbers of items."
            )

        self.index = loaded_index
        self._chunks = loaded_chunks
        self._validate_internal_state()

    def _validate_internal_state(self) -> None:
        """Ensure the index and chunk mapping remain synchronized."""
        if self.index.d != self.dimension:
            raise RuntimeError(
                "FAISS index dimension is inconsistent with the store."
            )

        if self.index.ntotal != len(self._chunks):
            raise RuntimeError(
                "FAISS index and chunk mapping are inconsistent."
            )

    @staticmethod
    def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
        """Convert a Chunk into JSON-compatible data."""
        return {
            "text": chunk.text,
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "metadata": chunk.metadata,
        }

    @staticmethod
    def _chunk_from_dict(record: Any) -> Chunk:
        """Reconstruct a Chunk from persisted data."""
        if not isinstance(record, dict):
            raise ValueError(
                "Each stored chunk must be a JSON object."
            )

        required_fields = {
            "text",
            "chunk_id",
            "source",
            "metadata",
        }

        missing = required_fields.difference(record)
        if missing:
            missing_names = ", ".join(sorted(missing))
            raise ValueError(
                f"Stored chunk is missing fields: {missing_names}."
            )

        text = record["text"]
        chunk_id = record["chunk_id"]
        source = record["source"]
        metadata = record["metadata"]

        if not isinstance(text, str):
            raise ValueError("Stored chunk text must be a string.")
        if not isinstance(chunk_id, int) or isinstance(chunk_id, bool):
            raise ValueError(
                "Stored chunk_id must be an integer."
            )
        if not isinstance(source, str):
            raise ValueError(
                "Stored chunk source must be a string."
            )
        if not isinstance(metadata, dict):
            raise ValueError(
                "Stored chunk metadata must be a dictionary."
            )

        return Chunk(
            text=text,
            chunk_id=chunk_id,
            source=source,
            metadata=metadata,
        )
