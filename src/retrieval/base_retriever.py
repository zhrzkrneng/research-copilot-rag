"""Retriever interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.preprocessing.chunk import Chunk
from src.retrieval.result import RetrievalResult


class BaseRetriever(ABC):
    """Abstract interface for retrieval implementations."""

    @abstractmethod
    def build_index(
        self,
        chunks: list[Chunk],
    ) -> None:
        """Build or rebuild the retrieval index from chunks."""
        raise NotImplementedError

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve the most relevant chunks for a query."""
        raise NotImplementedError
