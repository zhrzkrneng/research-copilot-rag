"""Retriever interface."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from src.preprocessing.chunk import Chunk
from src.retrieval.result import RetrievalResult


class BaseRetriever(ABC):
    """Abstract retriever interface."""

    @abstractmethod
    def build_index(
        self,
        chunks: list[Chunk],
    ) -> None:
        """Build an index from chunks."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve relevant chunks."""
