"""Abstract interface for vector stores."""

from abc import ABC, abstractmethod

import numpy as np

from src.preprocessing.chunk import Chunk
from src.retrieval.result import RetrievalResult


class BaseVectorStore(ABC):

    @abstractmethod
    def add(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(
        self,
        path: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(
        self,
        path: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError
