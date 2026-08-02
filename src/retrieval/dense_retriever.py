"""Dense retriever implementation."""

from __future__ import annotations

from src.embeddings.base_embedder import BaseEmbedder
from src.preprocessing.chunk import Chunk
from src.retrieval.base_retriever import BaseRetriever
from src.retrieval.base_vector_store import BaseVectorStore
from src.retrieval.result import RetrievalResult


class DenseRetriever(BaseRetriever):
    """Dense retrieval using embeddings and a vector store."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
    ) -> None:

        self.embedder = embedder
        self.vector_store = vector_store

    def build_index(
        self,
        chunks: list[Chunk],
    ) -> None:
        raise NotImplementedError

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        raise NotImplementedError
