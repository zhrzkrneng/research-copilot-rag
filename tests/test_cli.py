"""Tests for the Phase 19 professional CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from src.app import cli
from src.preprocessing.chunk import Chunk
from src.rag import RAGResponse
from src.retrieval.result import RetrievalResult


class DummyPipeline:
    """Pipeline double used by CLI tests."""

    def __init__(self, response: RAGResponse) -> None:
        self.response = response
        self.called_with = None

    def run(self, **kwargs):
        self.called_with = kwargs
        return self.response


def make_response() -> RAGResponse:
    """Build a deterministic RAG response for tests."""
    result = RetrievalResult(
        chunk=Chunk(
            text="retrieved text",
            chunk_id=0,
            source="sample.txt",
            metadata={"page": 1},
        ),
        score=0.75,
    )

    return RAGResponse(
        answer="final answer",
        prompt="final prompt",
        contexts=(result,),
    )


def test_build_parser_defaults():
    args = cli.build_parser().parse_args(
        [
            "--document",
            "sample.txt",
            "--query",
            "hello",
        ]
    )

    assert args.document == Path("sample.txt")
    assert args.query == "hello"
    assert args.top_k == 5
    assert args.max_contexts is None
    assert args.max_new_tokens == 512
    assert args.temperature == 0.0
    assert args.chunk_size == 512
    assert args.chunk_overlap == 64
    assert args.device == "auto"
    assert args.include_scores is False
    assert args.show_contexts is False
    assert args.show_prompt is False
    assert args.quiet is False


def test_build_parser_custom_values():
    args = cli.build_parser().parse_args(
        [
            "--document",
            "paper.pdf",
            "--query",
            "hello",
            "--top-k",
            "3",
            "--max-contexts",
            "2",
            "--max-new-tokens",
            "64",
            "--temperature",
            "0.5",
            "--chunk-size",
            "256",
            "--chunk-overlap",
            "32",
            "--embedding-model",
            "embed-model",
            "--generation-model",
            "generation-model",
            "--device",
            "cpu",
            "--include-scores",
            "--show-contexts",
            "--show-prompt",
            "--quiet",
        ]
    )

    assert args.document == Path("paper.pdf")
    assert args.top_k == 3
    assert args.max_contexts == 2
    assert args.max_new_tokens == 64
    assert args.temperature == pytest.approx(0.5)
    assert args.chunk_size == 256
    assert args.chunk_overlap == 32
    assert args.embedding_model == "embed-model"
    assert args.generation_model == "generation-model"
    assert args.device == "cpu"
    assert args.include_scores is True
    assert args.show_contexts is True
    assert args.show_prompt is True
    assert args.quiet is True


def test_build_config_forwards_arguments():
    args = argparse.Namespace(
        embedding_model="embed",
        generation_model="generate",
        device="cpu",
        chunk_size=200,
        chunk_overlap=20,
        include_scores=True,
    )

    config = cli.build_config(args)

    assert config.embedding_model_name == "embed"
    assert config.generation_model_name == "generate"
    assert config.device == "cpu"
    assert config.chunk_size == 200
    assert config.chunk_overlap == 20
    assert config.include_scores is True


def test_run_pipeline_forwards_arguments():
    response = make_response()
    pipeline = DummyPipeline(response)

    args = argparse.Namespace(
        query="question",
        top_k=4,
        max_contexts=3,
        max_new_tokens=128,
        temperature=0.2,
    )

    result = cli.run_pipeline(pipeline, args)

    assert result is response
    assert pipeline.called_with == {
        "query": "question",
        "top_k": 4,
        "max_contexts": 3,
        "max_new_tokens": 128,
        "temperature": 0.2,
    }


def test_format_response_preserves_legacy_output():
    output = cli.format_response(make_response())

    assert output == "Answer:\nfinal answer"


def test_format_response_includes_contexts_and_prompt():
    output = cli.format_response(
        make_response(),
        show_contexts=True,
        show_prompt=True,
    )

    assert "Retrieved contexts:" in output
    assert "source=sample.txt" in output
    assert "score=0.750000" in output
    assert "retrieved text" in output
    assert "Prompt:" in output
    assert "final prompt" in output


def test_format_professional_response_contains_sections():
    output = cli.format_professional_response(
        make_response(),
        show_contexts=True,
        show_prompt=True,
    )

    assert "Answer" in output
    assert "final answer" in output
    assert "Retrieved contexts" in output
    assert "source=sample.txt" in output
    assert "Prompt" in output
    assert "final prompt" in output


def test_read_project_version_uses_version_file(tmp_path):
    (tmp_path / "VERSION").write_text(
        "2.0.0\n",
        encoding="utf-8",
    )

    assert cli.read_project_version(tmp_path) == "2.0.0"


def test_read_project_version_uses_default_when_missing(tmp_path):
    assert cli.read_project_version(tmp_path) == cli.DEFAULT_VERSION


def test_print_banner(capsys):
    cli.print_banner(version="1.2.3")

    output = capsys.readouterr().out

    assert "Research Copilot RAG" in output
    assert "v1.2.3" in output


def test_print_step_and_success(capsys):
    cli.print_step(1, 3, "Loading")
    cli.print_success("Done")

    output = capsys.readouterr().out

    assert "[1/3] Loading" in output
    assert "✓ Done" in output


def test_print_summary(capsys):
    args = argparse.Namespace(
        document=Path("sample.txt"),
        query="question",
        top_k=3,
        embedding_model="embed",
        generation_model="generate",
        device="cpu",
    )

    cli.print_summary(
        args=args,
        response=make_response(),
    )

    output = capsys.readouterr().out

    assert "Summary" in output
    assert "sample.txt" in output
    assert "question" in output
    assert "Contexts" in output
    assert "SUCCESS" in output


def test_main_quiet_mode_builds_pipeline_and_prints_legacy_output(
    monkeypatch,
    capsys,
):
    response = make_response()
    pipeline = DummyPipeline(response)
    captured = {}

    def fake_build(document, *, config):
        captured["document"] = document
        captured["config"] = config
        return pipeline

    monkeypatch.setattr(
        cli,
        "build_indexed_pipeline",
        fake_build,
    )

    exit_code = cli.main(
        [
            "--document",
            "sample.txt",
            "--query",
            "question",
            "--quiet",
        ]
    )

    captured_output = capsys.readouterr()

    assert exit_code == 0
    assert captured["document"] == Path("sample.txt")
    assert captured_output.out == "Answer:\nfinal answer\n"
    assert captured_output.err == ""


def test_main_professional_mode_prints_banner_and_summary(
    monkeypatch,
    capsys,
):
    response = make_response()
    pipeline = DummyPipeline(response)

    monkeypatch.setattr(
        cli,
        "build_indexed_pipeline",
        lambda document, *, config: pipeline,
    )
    monkeypatch.setattr(
        cli,
        "read_project_version",
        lambda repository_root=None: "1.0.0",
    )

    exit_code = cli.main(
        [
            "--document",
            "sample.txt",
            "--query",
            "question",
            "--show-contexts",
        ]
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Research Copilot RAG" in output
    assert "[1/3]" in output
    assert "Answer" in output
    assert "Retrieved contexts" in output
    assert "Summary" in output
    assert "SUCCESS" in output


@pytest.mark.parametrize(
    "exception",
    [
        FileNotFoundError("missing"),
        TypeError("bad type"),
        ValueError("bad value"),
        RuntimeError("runtime"),
    ],
)
def test_main_returns_one_for_expected_errors(
    monkeypatch,
    capsys,
    exception,
):
    def fail(*args, **kwargs):
        raise exception

    monkeypatch.setattr(
        cli,
        "build_indexed_pipeline",
        fail,
    )

    exit_code = cli.main(
        [
            "--document",
            "sample.txt",
            "--query",
            "question",
            "--quiet",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error:" in captured.err
