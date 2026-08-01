import pytest

from src.core.document import Document
from src.preprocessing.chunk import Chunk
from src.preprocessing.recursive_splitter import RecursiveSplitter


def make_document(text: str) -> Document:
    return Document(
        text=text,
        source="paper.txt",
        file_type="txt",
        metadata={"title": "Test Paper"},
    )


def test_short_document_produces_one_chunk():
    splitter = RecursiveSplitter(
        chunk_size=100,
        chunk_overlap=10,
    )

    chunks = splitter.split(
        make_document("A short scientific paragraph.")
    )

    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].text == "A short scientific paragraph."
    assert chunks[0].chunk_id == 0
    assert chunks[0].source == "paper.txt"


def test_long_document_produces_multiple_chunks():
    text = " ".join(
        f"sentence-{index}"
        for index in range(100)
    )

    splitter = RecursiveSplitter(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = splitter.split(make_document(text))

    assert len(chunks) > 1
    assert all(
        len(chunk.text) <= 100
        for chunk in chunks
    )
    assert [
        chunk.chunk_id
        for chunk in chunks
    ] == list(range(len(chunks)))


def test_splitter_preserves_metadata():
    splitter = RecursiveSplitter(
        chunk_size=50,
        chunk_overlap=10,
    )

    chunks = splitter.split(
        make_document("word " * 80)
    )

    assert chunks
    assert all(
        chunk.metadata["title"] == "Test Paper"
        for chunk in chunks
    )
    assert all(
        "start" in chunk.metadata
        for chunk in chunks
    )
    assert all(
        "end" in chunk.metadata
        for chunk in chunks
    )
    assert all(
        "length" in chunk.metadata
        for chunk in chunks
    )


def test_empty_document_returns_empty_list():
    splitter = RecursiveSplitter()

    assert splitter.split(
        make_document("")
    ) == []

    assert splitter.split(
        make_document("   \n\n")
    ) == []


def test_invalid_chunk_size_raises_error():
    with pytest.raises(ValueError):
        RecursiveSplitter(
            chunk_size=0,
        )


def test_negative_overlap_raises_error():
    with pytest.raises(ValueError):
        RecursiveSplitter(
            chunk_size=100,
            chunk_overlap=-1,
        )


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        RecursiveSplitter(
            chunk_size=100,
            chunk_overlap=100,
        )


def test_custom_separators_are_supported():
    splitter = RecursiveSplitter(
        chunk_size=20,
        chunk_overlap=5,
        separators=["|", ""],
    )

    chunks = splitter.split(
        make_document(
            "alpha|beta|gamma|delta|epsilon"
        )
    )

    assert len(chunks) >= 2
    assert all(
        len(chunk.text) <= 20
        for chunk in chunks
    )
