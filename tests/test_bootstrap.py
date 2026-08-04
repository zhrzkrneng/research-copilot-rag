"""Tests for application bootstrap utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.app import bootstrap


class DummyDocument:
    def __init__(self, text: str = "example text") -> None:
        self.text = text
        self.source = "sample.txt"
        self.metadata = {}


class DummyChunk:
    pass


class DummyLoader:
    def __init__(self, document: DummyDocument | None = None) -> None:
        self.document = document or DummyDocument()
        self.loaded_path: Path | None = None

    def load(self, path: Path):
        self.loaded_path = path
        return self.document


class DummySplitter:
    def __init__(self, chunks=None) -> None:
        self.chunks = list(chunks or [DummyChunk()])
        self.received = None

    def split(self, document):
        self.received = document
        return self.chunks


class DummyEmbedder:
    dimension = 4


class DummyGenerator:
    pass


class DummyRetriever:
    def __init__(self) -> None:
        self.indexed_chunks = None

    def build_index(self, chunks):
        self.indexed_chunks = chunks


class DummyPipeline:
    def __init__(self, retriever=None) -> None:
        self.retriever = retriever or DummyRetriever()


def test_pipeline_config_defaults():
    config = bootstrap.PipelineConfig()

    assert config.chunk_size == 512
    assert config.chunk_overlap == 64
    assert config.embedding_batch_size == 32
    assert config.normalize_embeddings is True
    assert config.device == "auto"


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"embedding_model_name": ""}, ValueError),
        ({"generation_model_name": ""}, ValueError),
        ({"device": ""}, ValueError),
        ({"chunk_size": 0}, ValueError),
        ({"chunk_overlap": -1}, ValueError),
        (
            {"chunk_size": 10, "chunk_overlap": 10},
            ValueError,
        ),
        ({"embedding_batch_size": 0}, ValueError),
        ({"normalize_embeddings": "yes"}, TypeError),
        ({"include_scores": 1}, TypeError),
        ({"trust_remote_code": 1}, TypeError),
    ],
)
def test_pipeline_config_rejects_invalid_values(
    kwargs,
    exception,
):
    with pytest.raises(exception):
        bootstrap.PipelineConfig(**kwargs)


def test_build_loader_selects_pdf(tmp_path):
    path = tmp_path / "sample.pdf"
    path.write_bytes(b"%PDF")

    loader = bootstrap.build_loader(path)

    assert loader.__class__.__name__ == "PDFLoader"


def test_build_loader_selects_text(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("hello", encoding="utf-8")

    loader = bootstrap.build_loader(path)

    assert loader.__class__.__name__ == "TextLoader"


def test_build_loader_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "sample.md"
    path.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError):
        bootstrap.build_loader(path)


def test_build_loader_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        bootstrap.build_loader(tmp_path / "missing.txt")


def test_load_document_uses_selected_loader(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "sample.txt"
    path.write_text("hello", encoding="utf-8")
    loader = DummyLoader()

    monkeypatch.setattr(
        bootstrap,
        "build_loader",
        lambda document_path: loader,
    )

    document = bootstrap.load_document(path)

    assert document is loader.document
    assert loader.loaded_path == path


def test_build_splitter_uses_config():
    config = bootstrap.PipelineConfig(
        chunk_size=200,
        chunk_overlap=25,
    )

    splitter = bootstrap.build_splitter(config)

    assert splitter.chunk_size == 200
    assert splitter.chunk_overlap == 25


def test_build_embedder_forwards_configuration(monkeypatch):
    captured = {}

    class FakeEmbedder:
        DEFAULT_MODEL_NAME = "fake"

        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.dimension = 4

    monkeypatch.setattr(
        bootstrap,
        "SentenceTransformerEmbedder",
        FakeEmbedder,
    )

    config = bootstrap.PipelineConfig(
        embedding_model_name="embed-model",
        device="cpu",
        embedding_batch_size=8,
        normalize_embeddings=False,
        query_prefix="query: ",
    )

    result = bootstrap.build_embedder(config)

    assert isinstance(result, FakeEmbedder)
    assert captured == {
        "model_name": "embed-model",
        "device": "cpu",
        "batch_size": 8,
        "normalize_embeddings": False,
        "query_prefix": "query: ",
    }


def test_build_generator_forwards_configuration(monkeypatch):
    captured = {}

    class FakeGenerator:
        DEFAULT_MODEL_NAME = "fake"

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        bootstrap,
        "HFGenerator",
        FakeGenerator,
    )

    config = bootstrap.PipelineConfig(
        generation_model_name="generation-model",
        device="cpu",
        trust_remote_code=True,
    )

    result = bootstrap.build_generator(config)

    assert isinstance(result, FakeGenerator)
    assert captured == {
        "model_name": "generation-model",
        "device": "cpu",
        "trust_remote_code": True,
    }


def test_build_indexed_pipeline_indexes_chunks(monkeypatch, tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("hello", encoding="utf-8")

    document = DummyDocument()
    chunks = [DummyChunk(), DummyChunk()]
    splitter = DummySplitter(chunks)
    retriever = DummyRetriever()
    pipeline = DummyPipeline(retriever)

    monkeypatch.setattr(
        bootstrap,
        "load_document",
        lambda document_path: document,
    )
    monkeypatch.setattr(
        bootstrap,
        "build_splitter",
        lambda config: splitter,
    )
    monkeypatch.setattr(
        bootstrap,
        "build_pipeline",
        lambda **kwargs: pipeline,
    )

    result = bootstrap.build_indexed_pipeline(
        path,
        config=bootstrap.PipelineConfig(),
        embedder=DummyEmbedder(),
        generator=DummyGenerator(),
    )

    assert result is pipeline
    assert splitter.received is document
    assert retriever.indexed_chunks == chunks


def test_build_indexed_pipeline_rejects_empty_chunks(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "sample.txt"
    path.write_text("hello", encoding="utf-8")

    monkeypatch.setattr(
        bootstrap,
        "load_document",
        lambda document_path: DummyDocument(""),
    )
    class EmptySplitter:
        def split(self, document):
            return []

    monkeypatch.setattr(
        bootstrap,
        "build_splitter",
        lambda config: EmptySplitter(),
    )

    with pytest.raises(ValueError):
        bootstrap.build_indexed_pipeline(
            path,
            config=bootstrap.PipelineConfig(),
            embedder=DummyEmbedder(),
            generator=DummyGenerator(),
        )
