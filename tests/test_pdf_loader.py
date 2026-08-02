
from pathlib import Path

import fitz
import pytest

from src.core.document import Document
from src.ingestion.pdf_loader import PDFLoader


def test_pdf_loader_returns_document(tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"

    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Hello ResearchCopilot!")
    pdf.save(pdf_path)
    pdf.close()

    loader = PDFLoader()
    document = loader.load(pdf_path)

    assert isinstance(document, Document)
    assert "Hello ResearchCopilot!" in document.text
    assert document.source == "sample.pdf"
    assert document.file_type == "pdf"
    assert document.metadata == {}


def test_pdf_loader_reads_multiple_pages(tmp_path: Path):
    pdf_path = tmp_path / "multi_page.pdf"

    pdf = fitz.open()

    first_page = pdf.new_page()
    first_page.insert_text((72, 72), "First page")

    second_page = pdf.new_page()
    second_page.insert_text((72, 72), "Second page")

    pdf.save(pdf_path)
    pdf.close()

    loader = PDFLoader()
    document = loader.load(pdf_path)

    assert "First page" in document.text
    assert "Second page" in document.text
    assert document.text.index("First page") < document.text.index(
        "Second page"
    )


def test_pdf_loader_handles_empty_pdf(tmp_path: Path):
    pdf_path = tmp_path / "empty.pdf"

    pdf = fitz.open()
    pdf.new_page()
    pdf.save(pdf_path)
    pdf.close()

    loader = PDFLoader()
    document = loader.load(pdf_path)

    assert document.text.strip() == ""
    assert document.source == "empty.pdf"
    assert document.file_type == "pdf"


def test_pdf_loader_raises_for_missing_file(tmp_path: Path):
    missing_path = tmp_path / "missing.pdf"

    loader = PDFLoader()

    with pytest.raises(FileNotFoundError):
        loader.load(missing_path)
