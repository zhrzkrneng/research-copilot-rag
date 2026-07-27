
from pathlib import Path

from src.ingestion.text_loader import TextLoader
from src.core.document import Document


def test_text_loader_returns_document():
    loader = TextLoader()

    document = loader.load(Path("tests/data/sample.txt"))

    assert isinstance(document, Document)
    assert document.text == "Hello ResearchCopilot!"
    assert document.source == "sample.txt"
    assert document.file_type == "txt"
    assert document.metadata == {}
