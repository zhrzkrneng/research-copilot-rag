"""Tests for FAISS vector-store persistence."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.preprocessing.chunk import Chunk
from src.retrieval.faiss_vector_store import FAISSVectorStore


def make_chunks() -> list[Chunk]:
    return [
        Chunk(
            text="alpha document",
            chunk_id=0,
            source="sample.txt",
            metadata={"page": 1},
        ),
        Chunk(
            text="beta document",
            chunk_id=1,
            source="sample.txt",
            metadata={"page": 2},
        ),
    ]


def make_embeddings() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )


def test_save_creates_expected_files(tmp_path):
    store = FAISSVectorStore(dimension=3)
    store.add(make_chunks(), make_embeddings())

    directory = tmp_path / "index"
    store.save(directory)

    assert (directory / "index.faiss").is_file()
    assert (directory / "chunks.json").is_file()
    assert (directory / "store.json").is_file()


def test_round_trip_preserves_chunks_and_search(tmp_path):
    original = FAISSVectorStore(dimension=3)
    original.add(make_chunks(), make_embeddings())
    original.save(tmp_path / "index")

    restored = FAISSVectorStore(dimension=3)
    restored.load(tmp_path / "index")

    assert len(restored) == 2

    results = restored.search(
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        top_k=2,
    )

    assert results[0].chunk.text == "alpha document"
    assert results[0].chunk.metadata == {"page": 1}
    assert results[0].score == pytest.approx(1.0)


def test_round_trip_supports_empty_store(tmp_path):
    original = FAISSVectorStore(dimension=3)
    original.save(tmp_path / "empty")

    restored = FAISSVectorStore(dimension=3)
    restored.load(tmp_path / "empty")

    assert len(restored) == 0
    assert restored.search(
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    ) == []


def test_load_replaces_existing_state(tmp_path):
    source = FAISSVectorStore(dimension=3)
    source.add(make_chunks(), make_embeddings())
    source.save(tmp_path / "index")

    target = FAISSVectorStore(dimension=3)
    target.add(
        [
            Chunk(
                text="old",
                chunk_id=99,
                source="old.txt",
                metadata={},
            )
        ],
        np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
    )

    target.load(tmp_path / "index")

    assert len(target) == 2
    assert target.search(
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        top_k=1,
    )[0].chunk.text == "alpha document"


def test_load_rejects_dimension_mismatch(tmp_path):
    store = FAISSVectorStore(dimension=3)
    store.add(make_chunks(), make_embeddings())
    store.save(tmp_path / "index")

    incompatible = FAISSVectorStore(dimension=4)

    with pytest.raises(ValueError):
        incompatible.load(tmp_path / "index")


def test_load_rejects_missing_directory(tmp_path):
    store = FAISSVectorStore(dimension=3)

    with pytest.raises(FileNotFoundError):
        store.load(tmp_path / "missing")


def test_load_rejects_missing_required_file(tmp_path):
    directory = tmp_path / "broken"
    directory.mkdir()
    (directory / "store.json").write_text(
        json.dumps({"dimension": 3, "count": 0}),
        encoding="utf-8",
    )

    store = FAISSVectorStore(dimension=3)

    with pytest.raises(FileNotFoundError):
        store.load(directory)


def test_load_rejects_invalid_chunk_count(tmp_path):
    store = FAISSVectorStore(dimension=3)
    store.add(make_chunks(), make_embeddings())
    directory = tmp_path / "index"
    store.save(directory)

    metadata_path = directory / "store.json"
    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )
    metadata["count"] = 999
    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    restored = FAISSVectorStore(dimension=3)

    with pytest.raises(ValueError):
        restored.load(directory)


def test_save_rejects_non_serializable_metadata(tmp_path):
    store = FAISSVectorStore(dimension=3)
    chunk = Chunk(
        text="text",
        chunk_id=0,
        source="sample.txt",
        metadata={"bad": {1, 2, 3}},
    )
    store.add(
        [chunk],
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
    )

    with pytest.raises(ValueError):
        store.save(tmp_path / "index")
