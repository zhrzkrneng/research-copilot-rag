"""Integration tests for the retrieval pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.embeddings.base_embedder import BaseEmbedder
from src.ingestion.pdf_loader import PDFLoader
from src.preprocessing.chunk import Chunk
from src.preprocessing.recursive_splitter import RecursiveSplitter
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.faiss_vector_store import FAISSVectorStore


class KeywordEmbedder(BaseEmbedder):
    """Deterministic keyword-based embedder for integration tests."""

    KEYWORDS = (
        "pneumonia",
        "retrieval",
        "transformer",
        "diagnosis",
    )

    @property
    def dimension(self) -> int:
        return len(self.KEYWORDS)

    def encode_chunks(
        self,
        chunks: list[Chunk],
    ) -> np.ndarray:
        return np.vstack(
            [
                self._encode_text(chunk.text)
                for chunk in chunks
            ]
        ).astype(np.float32)

    def encode_query(
        self,
        query: str,
    ) -> np.ndarray:
        return self._encode_text(query).astype(np.float32)

    def _encode_text(
        self,
        text: str,
    ) -> np.ndarray:
        lowered = text.lower()

        vector = np.array(
            [
                float(lowered.count(keyword))
                for keyword in self.KEYWORDS
            ],
            dtype=np.float32,
        )

        if np.linalg.norm(vector) == 0:
            vector[0] = 1e-6

        return vector


def test_pdf_to_retrieval_pipeline():
    loader = PDFLoader()

    document = loader.load(
        Path("tests/data/sample.pdf")
    )

    splitter = RecursiveSplitter(
        chunk_size=120,
        chunk_overlap=20,
    )

    chunks = splitter.split(document)

    embedder = KeywordEmbedder()

    store = FAISSVectorStore(
        dimension=embedder.dimension,
    )

    retriever = DenseRetriever(
        embedder=embedder,
        vector_store=store,
    )

    retriever.build_index(chunks)

    results = retriever.retrieve(
        "pneumonia diagnosis",
        top_k=3,
    )

    assert chunks
    assert len(store) == len(chunks)
    assert 1 <= len(results) <= 3

    for result in results:
        assert isinstance(result.chunk, Chunk)
        assert isinstance(result.score, float)
        assert result.chunk.text
        assert result.chunk.source


def test_pipeline_preserves_chunk_metadata():
    loader = PDFLoader()

    document = loader.load(
        Path("tests/data/sample.pdf")
    )

    splitter = RecursiveSplitter(
        chunk_size=100,
        chunk_overlap=10,
    )

    chunks = splitter.split(document)

    embedder = KeywordEmbedder()
    store = FAISSVectorStore(
        dimension=embedder.dimension,
    )

    retriever = DenseRetriever(
        embedder,
        store,
    )

    retriever.build_index(chunks)

    result = retriever.retrieve(
        "retrieval",
        top_k=1,
    )[0]

    assert result.chunk.chunk_id is not None
    assert isinstance(result.chunk.metadata, dict)
    assert result.chunk.source


def test_top_k_larger_than_number_of_chunks():
    loader = PDFLoader()

    document = loader.load(
        Path("tests/data/sample.pdf")
    )

    splitter = RecursiveSplitter(
        chunk_size=500,
        chunk_overlap=0,
    )

    chunks = splitter.split(document)

    embedder = KeywordEmbedder()
    store = FAISSVectorStore(
        dimension=embedder.dimension,
    )

    retriever = DenseRetriever(
        embedder,
        store,
    )

    retriever.build_index(chunks)

    results = retriever.retrieve(
        "transformer retrieval",
        top_k=100,
    )

    assert len(results) == len(chunks)
