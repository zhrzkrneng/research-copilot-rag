"""
PDF document loader.
"""

from pathlib import Path

import fitz

from src.core.document import Document
from src.ingestion.base_loader import BaseLoader


class PDFLoader(BaseLoader):
    """
    Load text from PDF files.
    """

    def load(self, path: Path) -> Document:
        if not path.exists():
            raise FileNotFoundError(path)

        pdf = fitz.open(path)

        pages = []

        for page in pdf:
            pages.append(page.get_text())

        pdf.close()

        return Document(
            text="\n".join(pages),
            source=path.name,
            file_type="pdf",
            metadata={},
        )
