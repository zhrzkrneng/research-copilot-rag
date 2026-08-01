import numpy as np
import pytest

from src.embeddings.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from src.preprocessing.chunk import Chunk


class FakeSentenceTransformer:
    def __init__(
        self,
        model_name_or_path: str,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name_or_path
        self.device = device

    def get_sentence_embedding_dimension(self) -> int:
        return 4

    def encode(
        self,
        sentences: list[str],
        **kwargs,
    ) -> np.ndarray:
        vectors = []

        for sentence in sentences:
            value = float(len(sentence))
            vector = np.array(
                [value, value + 1, value + 2, value + 3],
                dtype=np.float32,
            )

            if kwargs.get("normalize_embeddings"):
                norm = np.linalg.norm(vector)
                vector = vector / norm

            vectors.append(vector)

        return np.vstack(vectors)


@pytest.fixture
def patched_embedder(monkeypatch):
    monkeypatch.setattr(
        "src.embeddings.sentence_transformer_embedder."
        "SentenceTransformer",
        FakeSentenceTransformer,
    )

    return SentenceTransformerEmbedder(
        model_name="fake-model",
        batch_size=2,
    )


def make_chunk(text: str, chunk_id: int) -> Chunk:
    return Chunk(
        text=text,
        chunk_id=chunk_id,
        source="paper.pdf",
    )


def test_embedder_reports_dimension(patched_embedder):
    assert patched_embedder.dimension == 4


def test_encode_chunks_returns_expected_shape(patched_embedder):
    chunks = [
        make_chunk("first chunk", 0),
        make_chunk("second chunk", 1),
    ]

    embeddings = patched_embedder.encode_chunks(chunks)

    assert embeddings.shape == (2, 4)
    assert embeddings.dtype == np.float32


def test_encode_chunks_returns_empty_matrix(patched_embedder):
    embeddings = patched_embedder.encode_chunks([])

    assert embeddings.shape == (0, 4)
    assert embeddings.dtype == np.float32


def test_encode_query_returns_one_vector(patched_embedder):
    embedding = patched_embedder.encode_query(
        "What is retrieval augmented generation?"
    )

    assert embedding.shape == (4,)
    assert embedding.dtype == np.float32


def test_encode_query_rejects_empty_text(patched_embedder):
    with pytest.raises(ValueError):
        patched_embedder.encode_query("   ")


def test_batch_size_must_be_positive(monkeypatch):
    monkeypatch.setattr(
        "src.embeddings.sentence_transformer_embedder."
        "SentenceTransformer",
        FakeSentenceTransformer,
    )

    with pytest.raises(ValueError):
        SentenceTransformerEmbedder(
            model_name="fake-model",
            batch_size=0,
        )


def test_normalized_vectors_have_unit_norm(patched_embedder):
    chunks = [make_chunk("scientific text", 0)]

    embeddings = patched_embedder.encode_chunks(chunks)

    assert np.isclose(
        np.linalg.norm(embeddings[0]),
        1.0,
        atol=1e-5,
    )
