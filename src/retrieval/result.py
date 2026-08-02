"""Retrieval result model."""

from dataclasses import dataclass

from src.preprocessing.chunk import Chunk


@dataclass(slots=True)
class RetrievalResult:
    """One retrieved chunk together with its similarity score."""

    chunk: Chunk
    score: float
