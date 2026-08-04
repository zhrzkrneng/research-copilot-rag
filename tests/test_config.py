"""Tests for typed project configuration."""

from __future__ import annotations

import json

import pytest

from src.config import (
    AppConfig,
    EmbeddingConfig,
    GenerationConfig,
    RetrievalConfig,
    load_config,
)


def test_default_config_values():
    config = AppConfig()

    assert config.embedding.model_name == (
        "BAAI/bge-small-en-v1.5"
    )
    assert config.generation.model_name == (
        "Qwen/Qwen2.5-1.5B-Instruct"
    )
    assert config.retrieval.top_k == 5


def test_from_mapping_builds_nested_config():
    config = AppConfig.from_mapping(
        {
            "embedding": {
                "model_name": "embed",
                "batch_size": 8,
            },
            "generation": {
                "model_name": "generate",
                "device": "cpu",
            },
            "retrieval": {
                "chunk_size": 200,
                "chunk_overlap": 20,
                "top_k": 3,
            },
        }
    )

    assert config.embedding.model_name == "embed"
    assert config.embedding.batch_size == 8
    assert config.generation.device == "cpu"
    assert config.retrieval.chunk_size == 200
    assert config.retrieval.top_k == 3


def test_to_pipeline_config_maps_values():
    config = AppConfig(
        embedding=EmbeddingConfig(
            model_name="embed",
            batch_size=16,
            normalize=False,
            query_prefix="query: ",
        ),
        generation=GenerationConfig(
            model_name="generate",
            device="cpu",
            trust_remote_code=True,
        ),
        retrieval=RetrievalConfig(
            chunk_size=300,
            chunk_overlap=30,
            include_scores=True,
        ),
    )

    pipeline = config.to_pipeline_config()

    assert pipeline.embedding_model_name == "embed"
    assert pipeline.embedding_batch_size == 16
    assert pipeline.normalize_embeddings is False
    assert pipeline.query_prefix == "query: "
    assert pipeline.generation_model_name == "generate"
    assert pipeline.device == "cpu"
    assert pipeline.chunk_size == 300
    assert pipeline.chunk_overlap == 30
    assert pipeline.include_scores is True
    assert pipeline.trust_remote_code is True


@pytest.mark.parametrize(
    ("factory", "kwargs", "exception"),
    [
        (
            EmbeddingConfig,
            {"model_name": ""},
            ValueError,
        ),
        (
            EmbeddingConfig,
            {"batch_size": 0},
            ValueError,
        ),
        (
            GenerationConfig,
            {"temperature": -0.1},
            ValueError,
        ),
        (
            GenerationConfig,
            {"max_new_tokens": 0},
            ValueError,
        ),
        (
            RetrievalConfig,
            {"chunk_size": 10, "chunk_overlap": 10},
            ValueError,
        ),
        (
            RetrievalConfig,
            {"top_k": 0},
            ValueError,
        ),
    ],
)
def test_config_rejects_invalid_values(
    factory,
    kwargs,
    exception,
):
    with pytest.raises(exception):
        factory(**kwargs)


def test_from_mapping_rejects_unknown_keys():
    with pytest.raises(ValueError):
        AppConfig.from_mapping(
            {"unknown": {}}
        )


def test_load_config_reads_json_yaml_subset(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        json.dumps(
            {
                "retrieval": {
                    "top_k": 7,
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.retrieval.top_k == 7


def test_load_config_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_load_config_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(path)


def test_load_config_rejects_non_mapping_root(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(TypeError):
        load_config(path)
