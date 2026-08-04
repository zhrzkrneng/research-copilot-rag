"""Typed project configuration models."""

from __future__ import annotations

from dataclasses import dataclass, field, field, field
from typing import Any

from src.app.bootstrap import PipelineConfig


def _require_non_empty_string(
    value: object,
    name: str,
) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")

    if not value.strip():
        raise ValueError(f"{name} cannot be empty.")


def _require_positive_int(
    value: object,
    name: str,
) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")

    if value <= 0:
        raise ValueError(
            f"{name} must be greater than zero."
        )


def _reject_unknown_keys(
    data: dict[str, Any],
    allowed: set[str],
    section: str,
) -> None:
    unknown = set(data) - allowed

    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(
            f"Unknown {section} configuration keys: {names}."
        )


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Embedding-model configuration."""

    model_name: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 32
    normalize: bool = True
    query_prefix: str = (
        "Represent this sentence for searching relevant passages: "
    )

    def __post_init__(self) -> None:
        _require_non_empty_string(self.model_name, "model_name")
        _require_positive_int(self.batch_size, "batch_size")

        if not isinstance(self.normalize, bool):
            raise TypeError("normalize must be a boolean.")

        if not isinstance(self.query_prefix, str):
            raise TypeError("query_prefix must be a string.")


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Text-generation configuration."""

    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str = "auto"
    max_new_tokens: int = 512
    temperature: float = 0.0
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        _require_non_empty_string(self.model_name, "model_name")
        _require_non_empty_string(self.device, "device")
        _require_positive_int(
            self.max_new_tokens,
            "max_new_tokens",
        )

        if (
            not isinstance(self.temperature, (int, float))
            or isinstance(self.temperature, bool)
        ):
            raise TypeError("temperature must be a number.")

        if float(self.temperature) < 0.0:
            raise ValueError("temperature cannot be negative.")

        if not isinstance(self.trust_remote_code, bool):
            raise TypeError(
                "trust_remote_code must be a boolean."
            )


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Retrieval and chunking configuration."""

    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    max_contexts: int | None = None
    include_scores: bool = False

    def __post_init__(self) -> None:
        _require_positive_int(self.chunk_size, "chunk_size")

        if (
            not isinstance(self.chunk_overlap, int)
            or isinstance(self.chunk_overlap, bool)
        ):
            raise TypeError("chunk_overlap must be an integer.")

        if self.chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative."
            )

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        _require_positive_int(self.top_k, "top_k")

        if self.max_contexts is not None:
            _require_positive_int(
                self.max_contexts,
                "max_contexts",
            )

        if not isinstance(self.include_scores, bool):
            raise TypeError(
                "include_scores must be a boolean."
            )


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Complete Research Copilot configuration."""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    def to_pipeline_config(self) -> PipelineConfig:
        """Convert to the bootstrap layer configuration."""
        return PipelineConfig(
            embedding_model_name=self.embedding.model_name,
            generation_model_name=self.generation.model_name,
            device=self.generation.device,
            chunk_size=self.retrieval.chunk_size,
            chunk_overlap=self.retrieval.chunk_overlap,
            embedding_batch_size=self.embedding.batch_size,
            normalize_embeddings=self.embedding.normalize,
            query_prefix=self.embedding.query_prefix,
            include_scores=self.retrieval.include_scores,
            trust_remote_code=self.generation.trust_remote_code,
        )

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any],
    ) -> "AppConfig":
        """Build configuration from a nested mapping."""
        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary.")

        _reject_unknown_keys(
            data,
            {"embedding", "generation", "retrieval"},
            "root",
        )

        embedding_data = data.get("embedding", {})
        generation_data = data.get("generation", {})
        retrieval_data = data.get("retrieval", {})

        for name, section in (
            ("embedding", embedding_data),
            ("generation", generation_data),
            ("retrieval", retrieval_data),
        ):
            if not isinstance(section, dict):
                raise TypeError(
                    f"{name} must be a dictionary."
                )

        _reject_unknown_keys(
            embedding_data,
            {
                "model_name",
                "batch_size",
                "normalize",
                "query_prefix",
            },
            "embedding",
        )
        _reject_unknown_keys(
            generation_data,
            {
                "model_name",
                "device",
                "max_new_tokens",
                "temperature",
                "trust_remote_code",
            },
            "generation",
        )
        _reject_unknown_keys(
            retrieval_data,
            {
                "chunk_size",
                "chunk_overlap",
                "top_k",
                "max_contexts",
                "include_scores",
            },
            "retrieval",
        )

        return cls(
            embedding=EmbeddingConfig(**embedding_data),
            generation=GenerationConfig(**generation_data),
            retrieval=RetrievalConfig(**retrieval_data),
        )
