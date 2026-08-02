"""Abstract interface for embedding models."""

from abc import ABC, abstractmethod

import numpy as np

from src.preprocessing.chunk import Chunk


class BaseEmbedder(ABC):
    """Base interface implemented by all embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of generated embeddings."""
        raise NotImplementedError

    @abstractmethod
    def encode_chunks(
        self,
        chunks: list[Chunk],
    ) -> np.ndarray:
        """Encode document chunks as a two-dimensional array.

        Args:
            chunks: Ordered chunks to encode.

        Returns:
            Array with shape ``(number_of_chunks, embedding_dimension)``.
        """
        raise NotImplementedError

    @abstractmethod
    def encode_query(
        self,
        query: str,
    ) -> np.ndarray:
        """Encode a single retrieval query.

        Args:
            query: Search query.

        Returns:
            One-dimensional embedding vector.
        """
        raise NotImplementedError
