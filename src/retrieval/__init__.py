"""Retrieval interfaces and implementations."""

from src.retrieval.base_retriever import BaseRetriever
from src.retrieval.base_vector_store import BaseVectorStore
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.faiss_vector_store import FAISSVectorStore
from src.retrieval.result import RetrievalResult

__all__ = [
    "BaseRetriever",
    "BaseVectorStore",
    "DenseRetriever",
    "FAISSVectorStore",
    "RetrievalResult",
]
