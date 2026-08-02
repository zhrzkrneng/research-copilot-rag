import pytest

from src.retrieval.faiss_vector_store import FAISSVectorStore


def test_store_creation():
    store = FAISSVectorStore(dimension=384)

    assert len(store) == 0
    assert store.dimension == 384
    assert store.index.d == 384


def test_dimension_must_be_positive():
    with pytest.raises(ValueError):
        FAISSVectorStore(dimension=0)

    with pytest.raises(ValueError):
        FAISSVectorStore(dimension=-1)


def test_clear_empty_store():
    store = FAISSVectorStore(dimension=4)

    store.clear()

    assert len(store) == 0
    assert store.index.ntotal == 0

import numpy as np

from src.preprocessing.chunk import Chunk


def make_chunk(text: str, chunk_id: int) -> Chunk:
    return Chunk(
        text=text,
        chunk_id=chunk_id,
        source="paper.pdf",
    )


def test_add_one_chunk():
    store = FAISSVectorStore(dimension=3)

    chunks = [make_chunk("alpha", 0)]
    embeddings = np.array(
        [[1.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    store.add(chunks, embeddings)

    assert len(store) == 1
    assert store.index.ntotal == 1


def test_add_multiple_chunks():
    store = FAISSVectorStore(dimension=3)

    chunks = [
        make_chunk("alpha", 0),
        make_chunk("beta", 1),
    ]

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    store.add(chunks, embeddings)

    assert len(store) == 2
    assert store.index.ntotal == 2


def test_add_multiple_batches():
    store = FAISSVectorStore(dimension=2)

    store.add(
        [make_chunk("first", 0)],
        np.array([[1.0, 0.0]]),
    )

    store.add(
        [make_chunk("second", 1)],
        np.array([[0.0, 1.0]]),
    )

    assert len(store) == 2
    assert store.index.ntotal == 2


def test_add_converts_embeddings_to_float32():
    store = FAISSVectorStore(dimension=2)

    store.add(
        [make_chunk("alpha", 0)],
        np.array(
            [[1.0, 2.0]],
            dtype=np.float64,
        ),
    )

    assert len(store) == 1


def test_add_rejects_mismatched_counts():
    store = FAISSVectorStore(dimension=2)

    with pytest.raises(ValueError):
        store.add(
            [make_chunk("alpha", 0)],
            np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ),
        )


def test_add_rejects_invalid_rank():
    store = FAISSVectorStore(dimension=2)

    with pytest.raises(ValueError):
        store.add(
            [make_chunk("alpha", 0)],
            np.array([1.0, 0.0]),
        )


def test_add_rejects_dimension_mismatch():
    store = FAISSVectorStore(dimension=3)

    with pytest.raises(ValueError):
        store.add(
            [make_chunk("alpha", 0)],
            np.array([[1.0, 0.0]]),
        )


@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, -np.inf],
)
def test_add_rejects_non_finite_values(invalid_value):
    store = FAISSVectorStore(dimension=2)

    with pytest.raises(ValueError):
        store.add(
            [make_chunk("alpha", 0)],
            np.array([[1.0, invalid_value]]),
        )


def test_add_rejects_zero_vectors():
    store = FAISSVectorStore(dimension=2)

    with pytest.raises(ValueError):
        store.add(
            [make_chunk("alpha", 0)],
            np.array([[0.0, 0.0]]),
        )


def test_add_empty_inputs_is_noop():
    store = FAISSVectorStore(dimension=2)

    store.add(
        [],
        np.empty((0, 2), dtype=np.float32),
    )

    assert len(store) == 0
    assert store.index.ntotal == 0

