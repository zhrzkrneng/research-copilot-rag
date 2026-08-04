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
    """Vector store backed by a normalized FAISS inner-product index."""

    INDEX_FILENAME = "index.faiss"
    CHUNKS_FILENAME = "chunks.json"
    METADATA_FILENAME = "store.json"

    def __init__(self, dimension: int) -> None:
        if not isinstance(dimension, int) or isinstance(dimension, bool):
            raise TypeError("dimension must be an integer.")
        if dimension <= 0:
            raise ValueError("dimension must be positive.")

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self.index.reset()
        self._chunks.clear()

    def add(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
    ) -> None:
        if not isinstance(chunks, list):
            raise TypeError("chunks must be a list.")
        if not all(isinstance(chunk, Chunk) for chunk in chunks):
            raise TypeError(
                "all items in chunks must be Chunk instances."
            )

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

        previous_total = self.index.ntotal
        self.index.add(normalized)
        self._chunks.extend(chunks)

        if self.index.ntotal != previous_total + len(chunks):
            raise RuntimeError(
                "FAISS did not index the expected number of vectors."
            )

        self._validate_consistency()

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError("top_k must be a positive integer.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        self._validate_consistency()

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

        count = min(top_k, len(self._chunks))
        scores, indices = self.index.search(
            normalized_query,
            count,
        )

        results: list[RetrievalResult] = []

        for score, index in zip(scores[0], indices[0]):
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
        """Persist the FAISS index and chunks to a directory."""
        directory = self._normalize_directory(path)
        self._validate_consistency()
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(directory / self.INDEX_FILENAME),
        )

        chunks_payload = [
            {
                "text": chunk.text,
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "metadata": chunk.metadata,
            }
            for chunk in self._chunks
        ]

        self._write_json(
            directory / self.CHUNKS_FILENAME,
            chunks_payload,
        )
        self._write_json(
            directory / self.METADATA_FILENAME,
            {
                "dimension": self.dimension,
                "count": len(self._chunks),
                "index_type": "IndexFlatIP",
            },
        )

    def load(self, path: str | Path) -> None:
        """Load a previously persisted store."""
        directory = self._normalize_directory(path)

        if not directory.exists():
            raise FileNotFoundError(directory)
        if not directory.is_dir():
            raise ValueError("path must point to a directory.")

        index_path = directory / self.INDEX_FILENAME
        chunks_path = directory / self.CHUNKS_FILENAME
        metadata_path = directory / self.METADATA_FILENAME

        for required_path in (
            index_path,
            chunks_path,
            metadata_path,
        ):
            if not required_path.is_file():
                raise FileNotFoundError(required_path)

        metadata = self._read_json(metadata_path)

        if not isinstance(metadata, dict):
            raise ValueError(
                "store metadata must be a JSON object."
            )

        stored_dimension = metadata.get("dimension")
        stored_count = metadata.get("count")

        if stored_dimension != self.dimension:
            raise ValueError(
                "Stored index dimension does not match "
                "the current store dimension."
            )
        if (
            not isinstance(stored_count, int)
            or isinstance(stored_count, bool)
            or stored_count < 0
        ):
            raise ValueError("stored count is invalid.")

        loaded_index = faiss.read_index(str(index_path))

        if loaded_index.d != self.dimension:
            raise ValueError(
                "Loaded FAISS index dimension is invalid."
            )

        chunks_payload = self._read_json(chunks_path)

        if not isinstance(chunks_payload, list):
            raise ValueError(
                "stored chunks must be a JSON array."
            )

        loaded_chunks = [
            self._chunk_from_mapping(item)
            for item in chunks_payload
        ]

        if len(loaded_chunks) != stored_count:
            raise ValueError(
                "Stored chunk count does not match metadata."
            )
        if loaded_index.ntotal != stored_count:
            raise ValueError(
                "Stored FAISS vector count does not match metadata."
            )

        self.index = loaded_index
        self._chunks = loaded_chunks
        self._validate_consistency()

    def _validate_consistency(self) -> None:
        if self.index.ntotal != len(self._chunks):
            raise RuntimeError(
                "FAISS index and chunk mapping are inconsistent."
            )

    @staticmethod
    def _normalize_directory(path: str | Path) -> Path:
        if not isinstance(path, (str, Path)):
            raise TypeError(
                "path must be a string or pathlib.Path."
            )
        return Path(path).expanduser()

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        try:
            serialized = json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Chunk metadata must be JSON serializable."
            ) from exc

        path.write_text(serialized, encoding="utf-8")

    @staticmethod
    def _read_json(path: Path) -> Any:
        try:
            return json.loads(
                path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON file: {path.name}."
            ) from exc

    @staticmethod
    def _chunk_from_mapping(data: Any) -> Chunk:
        if not isinstance(data, dict):
            raise ValueError(
                "Each stored chunk must be a JSON object."
            )

        required = {
            "text",
            "chunk_id",
            "source",
            "metadata",
        }

        if set(data) != required:
            raise ValueError(
                "Stored chunk fields are invalid."
            )

        text = data["text"]
        chunk_id = data["chunk_id"]
        source = data["source"]
        metadata = data["metadata"]

        if not isinstance(text, str):
            raise ValueError(
                "Stored chunk text must be a string."
            )
        if (
            not isinstance(chunk_id, int)
            or isinstance(chunk_id, bool)
        ):
            raise ValueError(
                "Stored chunk_id must be an integer."
            )
        if not isinstance(source, str):
            raise ValueError(
                "Stored chunk source must be a string."
            )
        if not isinstance(metadata, dict):
            raise ValueError(
                "Stored chunk metadata must be an object."
            )

        return Chunk(
            text=text,
            chunk_id=chunk_id,
            source=source,
            metadata=metadata,
        )
