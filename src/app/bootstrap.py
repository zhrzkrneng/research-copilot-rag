"""Application bootstrap utilities for Research Copilot RAG."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.embeddings.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from src.generation.hf_generator import HFGenerator
from src.generation.prompt_builder import PromptBuilder
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.text_loader import TextLoader
from src.preprocessing.recursive_splitter import RecursiveSplitter
from src.rag.rag_pipeline import RAGPipeline
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.faiss_vector_store import FAISSVectorStore


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Configuration used to construct and index a RAG pipeline."""

    embedding_model_name: str = (
        SentenceTransformerEmbedder.DEFAULT_MODEL_NAME
    )
    generation_model_name: str = HFGenerator.DEFAULT_MODEL_NAME
    device: str = "auto"
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedding_batch_size: int = 32
    normalize_embeddings: bool = True
    query_prefix: str = (
        "Represent this sentence for searching relevant passages: "
    )
    include_scores: bool = False
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.embedding_model_name, str):
            raise TypeError(
                "embedding_model_name must be a string."
            )

        if not self.embedding_model_name.strip():
            raise ValueError(
                "embedding_model_name cannot be empty."
            )

        if not isinstance(self.generation_model_name, str):
            raise TypeError(
                "generation_model_name must be a string."
            )

        if not self.generation_model_name.strip():
            raise ValueError(
                "generation_model_name cannot be empty."
            )

        if not isinstance(self.device, str):
            raise TypeError("device must be a string.")

        if not self.device.strip():
            raise ValueError("device cannot be empty.")

        if (
            not isinstance(self.chunk_size, int)
            or isinstance(self.chunk_size, bool)
        ):
            raise TypeError("chunk_size must be an integer.")

        if self.chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero."
            )

        if (
            not isinstance(self.chunk_overlap, int)
            or isinstance(self.chunk_overlap, bool)
        ):
            raise TypeError(
                "chunk_overlap must be an integer."
            )

        if self.chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative."
            )

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        if (
            not isinstance(self.embedding_batch_size, int)
            or isinstance(self.embedding_batch_size, bool)
        ):
            raise TypeError(
                "embedding_batch_size must be an integer."
            )

        if self.embedding_batch_size <= 0:
            raise ValueError(
                "embedding_batch_size must be greater than zero."
            )

        if not isinstance(self.normalize_embeddings, bool):
            raise TypeError(
                "normalize_embeddings must be a boolean."
            )

        if not isinstance(self.query_prefix, str):
            raise TypeError("query_prefix must be a string.")

        if not isinstance(self.include_scores, bool):
            raise TypeError("include_scores must be a boolean.")

        if not isinstance(self.trust_remote_code, bool):
            raise TypeError(
                "trust_remote_code must be a boolean."
            )


def build_loader(document_path: str | Path):
    """Return the correct loader for a supported document path."""
    path = _validate_document_path(document_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return PDFLoader()

    if suffix == ".txt":
        return TextLoader()

    raise ValueError(
        "Unsupported document type. Expected a .pdf or .txt file."
    )


def load_document(document_path: str | Path):
    """Load one supported document into the project Document type."""
    path = _validate_document_path(document_path)
    loader = build_loader(path)
    return loader.load(path)


def build_splitter(
    config: PipelineConfig,
) -> RecursiveSplitter:
    """Build the configured document splitter."""
    _validate_config(config)

    return RecursiveSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )


def build_embedder(
    config: PipelineConfig,
) -> SentenceTransformerEmbedder:
    """Build the configured sentence-transformer embedder."""
    _validate_config(config)

    embedding_device = (
        None
        if config.device.strip().lower() == "auto"
        else config.device.strip()
    )

    return SentenceTransformerEmbedder(
        model_name=config.embedding_model_name.strip(),
        device=embedding_device,
        batch_size=config.embedding_batch_size,
        normalize_embeddings=config.normalize_embeddings,
        query_prefix=config.query_prefix,
    )


def build_retriever(
    embedder: SentenceTransformerEmbedder,
) -> DenseRetriever:
    """Build a dense retriever with a matching FAISS store."""
    if not isinstance(embedder, SentenceTransformerEmbedder):
        raise TypeError(
            "embedder must be a SentenceTransformerEmbedder."
        )

    vector_store = FAISSVectorStore(
        dimension=embedder.dimension,
    )

    return DenseRetriever(
        embedder=embedder,
        vector_store=vector_store,
    )


def build_prompt_builder(
    config: PipelineConfig,
) -> PromptBuilder:
    """Build the configured prompt builder."""
    _validate_config(config)

    return PromptBuilder(
        include_scores=config.include_scores,
    )


def build_generator(
    config: PipelineConfig,
) -> HFGenerator:
    """Build the configured Hugging Face generator."""
    _validate_config(config)

    return HFGenerator(
        model_name=config.generation_model_name.strip(),
        device=config.device.strip(),
        trust_remote_code=config.trust_remote_code,
    )


def build_pipeline(
    *,
    embedder: SentenceTransformerEmbedder,
    generator: HFGenerator,
    config: PipelineConfig,
) -> RAGPipeline:
    """Build an unindexed RAG pipeline from ready components."""
    _validate_config(config)

    if not isinstance(embedder, SentenceTransformerEmbedder):
        raise TypeError(
            "embedder must be a SentenceTransformerEmbedder."
        )

    if not isinstance(generator, HFGenerator):
        raise TypeError(
            "generator must be an HFGenerator."
        )

    retriever = build_retriever(embedder)
    prompt_builder = build_prompt_builder(config)

    return RAGPipeline(
        retriever=retriever,
        prompt_builder=prompt_builder,
        generator=generator,
    )


def build_indexed_pipeline(
    document_path: str | Path,
    *,
    config: PipelineConfig | None = None,
    embedder: SentenceTransformerEmbedder | None = None,
    generator: HFGenerator | None = None,
) -> RAGPipeline:
    """Build and index a complete pipeline for one document."""
    resolved_config = config or PipelineConfig()
    _validate_config(resolved_config)

    document = load_document(document_path)
    splitter = build_splitter(resolved_config)
    chunks = splitter.split(document)

    if not chunks:
        raise ValueError(
            "The document produced no non-empty chunks."
        )

    resolved_embedder = embedder or build_embedder(
        resolved_config
    )
    resolved_generator = generator or build_generator(
        resolved_config
    )

    pipeline = build_pipeline(
        embedder=resolved_embedder,
        generator=resolved_generator,
        config=resolved_config,
    )

    pipeline.retriever.build_index(chunks)

    return pipeline


def _validate_document_path(
    document_path: str | Path,
) -> Path:
    """Validate and normalize a document path."""
    if not isinstance(document_path, (str, Path)):
        raise TypeError(
            "document_path must be a string or pathlib.Path."
        )

    path = Path(document_path).expanduser()

    if not path.exists():
        raise FileNotFoundError(path)

    if not path.is_file():
        raise ValueError(
            "document_path must point to a file."
        )

    return path


def _validate_config(config: PipelineConfig) -> None:
    """Validate that a PipelineConfig instance was supplied."""
    if not isinstance(config, PipelineConfig):
        raise TypeError(
            "config must be a PipelineConfig instance."
        )


__all__ = [
    "PipelineConfig",
    "build_embedder",
    "build_generator",
    "build_indexed_pipeline",
    "build_loader",
    "build_pipeline",
    "build_prompt_builder",
    "build_retriever",
    "build_splitter",
    "load_document",
]
