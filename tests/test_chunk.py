from src.preprocessing.chunk import Chunk


def test_chunk_creation():
    chunk = Chunk(
        text="Hello World",
        chunk_id=0,
        source="paper.pdf",
    )

    assert chunk.text == "Hello World"
    assert chunk.chunk_id == 0
    assert chunk.source == "paper.pdf"
    assert chunk.metadata == {}
