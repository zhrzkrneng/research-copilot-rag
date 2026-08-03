"""Dense retriever implementation."""

from __future__ import annotations

import numpy as np

from src.embeddings.base_embedder import BaseEmbedder
from src.preprocessing.chunk import Chunk
from src.retrieval.base_retriever import BaseRetriever
from src.retrieval.base_vector_store import BaseVectorStore
from src.retrieval.result import RetrievalResult


class DenseRetriever(BaseRetriever):
    """Dense retriever built from an embedder and a vector store."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
    ) -> None:
        """Initialize the retriever."""
        self.embedder = embedder
        self.vector_store = vector_store

    def build_index(
        self,
        chunks: list[Chunk],
    ) -> None:
        """Build a fresh vector index from document chunks."""
        if not isinstance(chunks, list):
            raise TypeError("chunks must be provided as a list.")

        if not all(isinstance(chunk, Chunk) for chunk in chunks):
            raise TypeError(
                "all items in chunks must be Chunk instances."
            )

        self.vector_store.clear()

        if not chunks:
            return

        embeddings = np.asarray(
            self.embedder.encode_chunks(chunks),
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise RuntimeError(
                "The embedder must return a two-dimensional array."
            )

        if embeddings.shape[0] != len(chunks):
            raise RuntimeError(
                "The embedder returned a different number of vectors "
                "than input chunks."
            )

        if embeddings.shape[1] != self.embedder.dimension:
            raise RuntimeError(
                "The embedding dimension does not match "
                "the embedder dimension."
            )

        if not np.isfinite(embeddings).all():
            raise RuntimeError(
                "The embedder returned NaN or infinite values."
            )

        self.vector_store.add(
            chunks,
            embeddings,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve the most relevant chunks for a query."""
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError("query cannot be empty.")

        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError("top_k must be an integer.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_embedding = np.asarray(
            self.embedder.encode_query(cleaned_query),
            dtype=np.float32,
        )

        if query_embedding.ndim != 1:
            raise RuntimeError(
                "The embedder must return a one-dimensional "
                "query vector."
            )

        if query_embedding.shape[0] != self.embedder.dimension:
            raise RuntimeError(
                "The query embedding dimension does not match "
                "the embedder dimension."
            )

        if not np.isfinite(query_embedding).all():
            raise RuntimeError(
                "The embedder returned NaN or infinite values."
            )

        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )
