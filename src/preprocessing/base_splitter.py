"""Abstract interface for document splitters."""

from abc import ABC, abstractmethod

from src.core.document import Document
from src.preprocessing.chunk import Chunk


class BaseSplitter(ABC):
    """Base interface for all document splitters."""

    @abstractmethod
    def split(self, document: Document) -> list[Chunk]:
        """Split a document into ordered chunks.

        Args:
            document: Input document to split.

        Returns:
            Ordered list of generated chunks.
        """
        raise NotImplementedError
