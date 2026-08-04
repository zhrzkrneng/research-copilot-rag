"""Tests for the CLI."""

from __future__ import annotations

import argparse

import pytest

from src.app.cli import build_parser, main, run_pipeline


class DummyPipeline:
    def __init__(self) -> None:
        self.called_with = None

    def answer(self, **kwargs):
        self.called_with = kwargs
        return "dummy answer"


def test_build_parser_defaults():
    parser = build_parser()

    args = parser.parse_args(["--query", "hello"])

    assert args.query == "hello"
    assert args.top_k == 5
    assert args.max_contexts is None
    assert args.max_new_tokens == 512
    assert args.temperature == 0.0


def test_build_parser_custom_values():
    parser = build_parser()

    args = parser.parse_args([
        "--query", "hello",
        "--top-k", "3",
        "--max-contexts", "2",
        "--max-new-tokens", "64",
        "--temperature", "0.5",
    ])

    assert args.top_k == 3
    assert args.max_contexts == 2
    assert args.max_new_tokens == 64
    assert args.temperature == pytest.approx(0.5)


def test_run_pipeline_forwards_arguments():
    pipeline = DummyPipeline()

    args = argparse.Namespace(
        query="question",
        top_k=4,
        max_contexts=3,
        max_new_tokens=128,
        temperature=0.2,
    )

    answer = run_pipeline(pipeline, args)

    assert answer == "dummy answer"
    assert pipeline.called_with == {
        "query": "question",
        "top_k": 4,
        "max_contexts": 3,
        "max_new_tokens": 128,
        "temperature": 0.2,
    }


def test_main_not_implemented(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--query", "hello"],
    )

    with pytest.raises(NotImplementedError):
        main()
