"""Tests for DenseRetriever index construction."""

from __future__ import annotations

import numpy as np
import pytest

from src.embeddings.base_embedder import BaseEmbedder
from src.preprocessing.chunk import Chunk
from src.retrieval.base_vector_store import BaseVectorStore
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.result import RetrievalResult


class DummyEmbedder(BaseEmbedder):
    """Deterministic embedder used by unit tests."""

    def __init__(
        self,
        embeddings: np.ndarray | None = None,
        dimension: int = 3,
    ) -> None:
        self._dimension = dimension
        self.embeddings = embeddings
        self.encode_chunks_calls = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_chunks(
        self,
        chunks: list[Chunk],
    ) -> np.ndarray:
        self.encode_chunks_calls += 1

        if self.embeddings is not None:
            return self.embeddings

        return np.array(
            [
                [float(index + 1)] + [0.0] * (self.dimension - 1)
                for index, _ in enumerate(chunks)
            ],
            dtype=np.float32,
        )

    def encode_query(
        self,
        query: str,
    ) -> np.ndarray:
        raise NotImplementedError


class DummyVectorStore(BaseVectorStore):
    """In-memory spy vector store used by unit tests."""

    def __init__(self) -> None:
        self.clear_calls = 0
        self.add_calls = 0
        self.added_chunks: list[Chunk] | None = None
        self.added_embeddings: np.ndarray | None = None

    def __len__(self) -> int:
        return 0 if self.added_chunks is None else len(self.added_chunks)

    def clear(self) -> None:
        self.clear_calls += 1
        self.added_chunks = None
        self.added_embeddings = None

    def add(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
    ) -> None:
        self.add_calls += 1
        self.added_chunks = chunks
        self.added_embeddings = embeddings

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError


def make_chunk(text: str, chunk_id: int) -> Chunk:
    return Chunk(
        text=text,
        chunk_id=chunk_id,
        source="paper.pdf",
    )


def test_constructor_preserves_dependencies():
    embedder = DummyEmbedder()
    store = DummyVectorStore()

    retriever = DenseRetriever(embedder, store)

    assert retriever.embedder is embedder
    assert retriever.vector_store is store


def test_build_index_encodes_and_adds_chunks():
    chunks = [
        make_chunk("alpha", 0),
        make_chunk("beta", 1),
    ]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    embedder = DummyEmbedder(
        embeddings=embeddings,
        dimension=3,
    )
    store = DummyVectorStore()
    retriever = DenseRetriever(embedder, store)

    retriever.build_index(chunks)

    assert embedder.encode_chunks_calls == 1
    assert store.clear_calls == 1
    assert store.add_calls == 1
    assert store.added_chunks is chunks
    np.testing.assert_array_equal(
        store.added_embeddings,
        embeddings,
    )


def test_build_index_rebuilds_existing_store():
    chunks = [make_chunk("new", 0)]

    embedder = DummyEmbedder(dimension=3)
    store = DummyVectorStore()
    store.added_chunks = [make_chunk("old", 99)]

    retriever = DenseRetriever(embedder, store)
    retriever.build_index(chunks)

    assert store.clear_calls == 1
    assert store.add_calls == 1
    assert store.added_chunks == chunks


def test_build_index_empty_list_clears_without_embedding():
    embedder = DummyEmbedder()
    store = DummyVectorStore()
    store.added_chunks = [make_chunk("old", 99)]

    retriever = DenseRetriever(embedder, store)
    retriever.build_index([])

    assert store.clear_calls == 1
    assert store.add_calls == 0
    assert embedder.encode_chunks_calls == 0
    assert store.added_chunks is None


def test_build_index_rejects_non_list_input():
    retriever = DenseRetriever(
        DummyEmbedder(),
        DummyVectorStore(),
    )

    with pytest.raises(TypeError):
        retriever.build_index(
            (make_chunk("alpha", 0),)  # type: ignore[arg-type]
        )


def test_build_index_rejects_non_chunk_items():
    retriever = DenseRetriever(
        DummyEmbedder(),
        DummyVectorStore(),
    )

    with pytest.raises(TypeError):
        retriever.build_index(
            [make_chunk("alpha", 0), "invalid"]  # type: ignore[list-item]
        )


def test_build_index_rejects_invalid_embedding_rank():
    embedder = DummyEmbedder(
        embeddings=np.array([1.0, 0.0, 0.0]),
        dimension=3,
    )
    retriever = DenseRetriever(
        embedder,
        DummyVectorStore(),
    )

    with pytest.raises(RuntimeError):
        retriever.build_index(
            [make_chunk("alpha", 0)]
        )


def test_build_index_rejects_embedding_count_mismatch():
    embedder = DummyEmbedder(
        embeddings=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        dimension=3,
    )
    retriever = DenseRetriever(
        embedder,
        DummyVectorStore(),
    )

    with pytest.raises(RuntimeError):
        retriever.build_index(
            [make_chunk("alpha", 0)]
        )


def test_build_index_rejects_embedding_dimension_mismatch():
    embedder = DummyEmbedder(
        embeddings=np.array(
            [[1.0, 0.0]],
            dtype=np.float32,
        ),
        dimension=3,
    )
    retriever = DenseRetriever(
        embedder,
        DummyVectorStore(),
    )

    with pytest.raises(RuntimeError):
        retriever.build_index(
            [make_chunk("alpha", 0)]
        )


@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, -np.inf],
)
def test_build_index_rejects_non_finite_embeddings(invalid_value):
    embedder = DummyEmbedder(
        embeddings=np.array(
            [[1.0, invalid_value, 0.0]],
            dtype=np.float32,
        ),
        dimension=3,
    )
    retriever = DenseRetriever(
        embedder,
        DummyVectorStore(),
    )

    with pytest.raises(RuntimeError):
        retriever.build_index(
            [make_chunk("alpha", 0)]
        )
