"""
Abstract base class for all document loaders.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from src.core.document import Document


class BaseLoader(ABC):
    """
    Base interface for all document loaders.
    """

    @abstractmethod
    def load(self, path: str | Path) -> Document:
        """
        Load a document from disk.

        Args:
            path: Path to the input file.

        Returns:
            Document object.
        """
        raise NotImplementedError
