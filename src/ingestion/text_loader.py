"""
Text document loader.
"""

from pathlib import Path

from src.core.document import Document
from src.ingestion.base_loader import BaseLoader


class TextLoader(BaseLoader):
    """
    Loader for plain text (.txt) files.
    """

    def load(self, path: Path) -> Document:
        text = path.read_text(encoding="utf-8")

        return Document(
            text=text,
            source=path.name,
            file_type=path.suffix.lstrip("."),
            metadata={},
        )
