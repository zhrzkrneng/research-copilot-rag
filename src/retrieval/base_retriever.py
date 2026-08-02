"""Base retriever interface."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from src.preprocessing.chunk import Chunk


class BaseRetriever(ABC):
    """Abstract interface for dense and sparse retrievers."""

    @abstractmethod
    def build_index(
        self,
        chunks: list[Chunk],
    ) -> None:
        """Create an index from chunks."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ):
        """Return the most relevant chunks."""
