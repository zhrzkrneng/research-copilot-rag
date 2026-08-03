"""Tests for DenseRetriever."""

from __future__ import annotations

import numpy as np
import pytest

from src.embeddings.base_embedder import BaseEmbedder
from src.preprocessing.chunk import Chunk
from src.retrieval.base_vector_store import BaseVectorStore
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.result import RetrievalResult


class DummyEmbedder(BaseEmbedder):
    """Deterministic embedder used in unit tests."""

    def __init__(
        self,
        *,
        dimension: int = 3,
        chunk_embeddings: np.ndarray | None = None,
        query_embedding: np.ndarray | None = None,
    ) -> None:
        self._dimension = dimension
        self.chunk_embeddings = chunk_embeddings
        self.query_embedding = query_embedding
        self.encode_chunks_calls = 0
        self.encode_query_calls = 0
        self.last_query: str | None = None

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        self.encode_chunks_calls += 1
        if self.chunk_embeddings is not None:
            return self.chunk_embeddings
        return np.array(
            [
                [float(index + 1)] + [0.0] * (self.dimension - 1)
                for index, _ in enumerate(chunks)
            ],
            dtype=np.float32,
        )

    def encode_query(self, query: str) -> np.ndarray:
        self.encode_query_calls += 1
        self.last_query = query
        if self.query_embedding is not None:
            return self.query_embedding
        return np.array(
            [1.0] + [0.0] * (self.dimension - 1),
            dtype=np.float32,
        )


class DummyVectorStore(BaseVectorStore):
    """Spy vector store used in unit tests."""

    def __init__(
        self,
        search_results: list[RetrievalResult] | None = None,
    ) -> None:
        self.clear_calls = 0
        self.add_calls = 0
        self.search_calls = 0
        self.added_chunks: list[Chunk] | None = None
        self.added_embeddings: np.ndarray | None = None
        self.last_query_embedding: np.ndarray | None = None
        self.last_top_k: int | None = None
        self.search_results = search_results or []

    def __len__(self) -> int:
        return 0 if self.added_chunks is None else len(self.added_chunks)

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        self.add_calls += 1
        self.added_chunks = chunks
        self.added_embeddings = embeddings

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        self.search_calls += 1
        self.last_query_embedding = query_embedding
        self.last_top_k = top_k
        return self.search_results

    def clear(self) -> None:
        self.clear_calls += 1
        self.added_chunks = None
        self.added_embeddings = None

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError


def make_chunk(text: str, chunk_id: int) -> Chunk:
    return Chunk(text=text, chunk_id=chunk_id, source="paper.pdf")


def test_constructor_preserves_dependencies():
    embedder = DummyEmbedder()
    store = DummyVectorStore()
    retriever = DenseRetriever(embedder, store)
    assert retriever.embedder is embedder
    assert retriever.vector_store is store


def test_build_index_encodes_and_adds_chunks():
    chunks = [make_chunk("alpha", 0), make_chunk("beta", 1)]
    embeddings = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    embedder = DummyEmbedder(chunk_embeddings=embeddings)
    store = DummyVectorStore()
    retriever = DenseRetriever(embedder, store)

    retriever.build_index(chunks)

    assert embedder.encode_chunks_calls == 1
    assert store.clear_calls == 1
    assert store.add_calls == 1
    assert store.added_chunks is chunks
    np.testing.assert_array_equal(store.added_embeddings, embeddings)


def test_build_index_empty_list_only_clears_store():
    embedder = DummyEmbedder()
    store = DummyVectorStore()
    retriever = DenseRetriever(embedder, store)

    retriever.build_index([])

    assert store.clear_calls == 1
    assert store.add_calls == 0
    assert embedder.encode_chunks_calls == 0


def test_retrieve_encodes_query_once_and_searches_once():
    expected = [
        RetrievalResult(
            chunk=make_chunk("result", 0),
            score=0.9,
        )
    ]
    embedder = DummyEmbedder()
    store = DummyVectorStore(search_results=expected)
    retriever = DenseRetriever(embedder, store)

    results = retriever.retrieve("  pneumonia treatment  ", top_k=3)

    assert results is expected
    assert embedder.encode_query_calls == 1
    assert embedder.last_query == "pneumonia treatment"
    assert store.search_calls == 1
    assert store.last_top_k == 3
    np.testing.assert_array_equal(
        store.last_query_embedding,
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_retrieve_rejects_empty_query(query):
    retriever = DenseRetriever(DummyEmbedder(), DummyVectorStore())
    with pytest.raises(ValueError):
        retriever.retrieve(query)


def test_retrieve_rejects_non_string_query():
    retriever = DenseRetriever(DummyEmbedder(), DummyVectorStore())
    with pytest.raises(TypeError):
        retriever.retrieve(123)  # type: ignore[arg-type]


@pytest.mark.parametrize("top_k", [0, -1])
def test_retrieve_rejects_non_positive_top_k(top_k):
    retriever = DenseRetriever(DummyEmbedder(), DummyVectorStore())
    with pytest.raises(ValueError):
        retriever.retrieve("query", top_k=top_k)


@pytest.mark.parametrize("top_k", [1.5, "5", True])
def test_retrieve_rejects_non_integer_top_k(top_k):
    retriever = DenseRetriever(DummyEmbedder(), DummyVectorStore())
    with pytest.raises(TypeError):
        retriever.retrieve("query", top_k=top_k)  # type: ignore[arg-type]


def test_retrieve_rejects_invalid_query_embedding_rank():
    embedder = DummyEmbedder(
        query_embedding=np.array([[1.0, 0.0, 0.0]])
    )
    retriever = DenseRetriever(embedder, DummyVectorStore())
    with pytest.raises(RuntimeError):
        retriever.retrieve("query")


def test_retrieve_rejects_query_dimension_mismatch():
    embedder = DummyEmbedder(
        dimension=3,
        query_embedding=np.array([1.0, 0.0]),
    )
    retriever = DenseRetriever(embedder, DummyVectorStore())
    with pytest.raises(RuntimeError):
        retriever.retrieve("query")


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_retrieve_rejects_non_finite_query_embedding(invalid_value):
    embedder = DummyEmbedder(
        query_embedding=np.array(
            [1.0, invalid_value, 0.0],
            dtype=np.float32,
        )
    )
    retriever = DenseRetriever(embedder, DummyVectorStore())
    with pytest.raises(RuntimeError):
        retriever.retrieve("query")
