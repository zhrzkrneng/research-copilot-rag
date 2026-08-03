"""Tests for the end-to-end RAG pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.generation.base_generator import BaseGenerator
from src.generation.prompt_builder import PromptBuilder
from src.preprocessing.chunk import Chunk
from src.rag.rag_pipeline import RAGPipeline, RAGResponse
from src.retrieval.base_retriever import BaseRetriever
from src.retrieval.result import RetrievalResult


def make_result(
    text: str,
    *,
    chunk_id: int = 0,
    score: float = 0.9,
) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(
            text=text,
            chunk_id=chunk_id,
            source="paper.pdf",
        ),
        score=score,
    )


class DummyRetriever(BaseRetriever):
    def __init__(
        self,
        results: list[RetrievalResult],
    ) -> None:
        self.results = results
        self.last_query: str | None = None
        self.last_top_k: int | None = None

    def build_index(self, chunks):
        raise NotImplementedError

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        self.last_query = query
        self.last_top_k = top_k
        return self.results


class DummyGenerator(BaseGenerator):
    def __init__(self, answer: str = "Generated answer.") -> None:
        self._answer = answer
        self.last_prompt: str | None = None
        self.last_max_new_tokens: int | None = None
        self.last_temperature: float | None = None

    @property
    def model_name(self) -> str:
        return "dummy-generator"

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        self.last_prompt = prompt
        self.last_max_new_tokens = max_new_tokens
        self.last_temperature = temperature
        return self._answer


def make_pipeline(
    *,
    results: list[RetrievalResult] | None = None,
    answer: str = "Generated answer.",
) -> tuple[RAGPipeline, DummyRetriever, DummyGenerator]:
    retriever = DummyRetriever(
        results
        if results is not None
        else [make_result("Relevant context.")]
    )
    generator = DummyGenerator(answer=answer)
    pipeline = RAGPipeline(
        retriever=retriever,
        prompt_builder=PromptBuilder(),
        generator=generator,
    )
    return pipeline, retriever, generator


def test_constructor_preserves_dependencies():
    retriever = DummyRetriever([make_result("context")])
    builder = PromptBuilder()
    generator = DummyGenerator()

    pipeline = RAGPipeline(
        retriever,
        builder,
        generator,
    )

    assert pipeline.retriever is retriever
    assert pipeline.prompt_builder is builder
    assert pipeline.generator is generator


def test_run_connects_retrieval_prompt_and_generation():
    contexts = [
        make_result("First context.", chunk_id=0, score=0.9),
        make_result("Second context.", chunk_id=1, score=0.8),
    ]
    pipeline, retriever, generator = make_pipeline(
        results=contexts,
        answer="Final answer.",
    )

    response = pipeline.run(
        "  What is RAG?  ",
        top_k=2,
        max_new_tokens=128,
        temperature=0.2,
    )

    assert isinstance(response, RAGResponse)
    assert response.answer == "Final answer."
    assert response.contexts == tuple(contexts)
    assert "First context." in response.prompt
    assert "Second context." in response.prompt
    assert "Question:\nWhat is RAG?" in response.prompt

    assert retriever.last_query == "What is RAG?"
    assert retriever.last_top_k == 2
    assert generator.last_prompt == response.prompt
    assert generator.last_max_new_tokens == 128
    assert generator.last_temperature == pytest.approx(0.2)


def test_run_respects_max_contexts():
    pipeline, _, _ = make_pipeline(
        results=[
            make_result("first", chunk_id=0),
            make_result("second", chunk_id=1),
            make_result("third", chunk_id=2),
        ]
    )

    response = pipeline.run(
        "Question?",
        top_k=3,
        max_contexts=2,
    )

    assert "first" in response.prompt
    assert "second" in response.prompt
    assert "third" not in response.prompt
    assert len(response.contexts) == 3


def test_answer_returns_only_answer_text():
    pipeline, _, _ = make_pipeline(
        answer="Only this answer."
    )

    answer = pipeline.answer("Question?")

    assert answer == "Only this answer."


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_run_rejects_empty_query(query):
    pipeline, _, _ = make_pipeline()

    with pytest.raises(ValueError, match="query cannot be empty"):
        pipeline.run(query)


def test_run_rejects_non_string_query():
    pipeline, _, _ = make_pipeline()

    with pytest.raises(TypeError, match="query must be a string"):
        pipeline.run(123)  # type: ignore[arg-type]


def test_run_rejects_empty_retrieval_results():
    pipeline, _, generator = make_pipeline(results=[])

    with pytest.raises(
        ValueError,
        match="retrieval returned no contexts",
    ):
        pipeline.run("Question?")

    assert generator.last_prompt is None


def test_retriever_error_propagates():
    retriever = MagicMock(spec=BaseRetriever)
    retriever.retrieve.side_effect = RuntimeError("retrieval failed")

    pipeline = RAGPipeline(
        retriever=retriever,
        prompt_builder=PromptBuilder(),
        generator=DummyGenerator(),
    )

    with pytest.raises(RuntimeError, match="retrieval failed"):
        pipeline.run("Question?")


def test_generator_error_propagates():
    contexts = [make_result("context")]
    retriever = DummyRetriever(contexts)
    generator = MagicMock(spec=BaseGenerator)
    generator.generate.side_effect = RuntimeError("generation failed")

    pipeline = RAGPipeline(
        retriever=retriever,
        prompt_builder=PromptBuilder(),
        generator=generator,
    )

    with pytest.raises(RuntimeError, match="generation failed"):
        pipeline.run("Question?")


def test_response_is_immutable():
    response = RAGResponse(
        answer="answer",
        prompt="prompt",
        contexts=(make_result("context"),),
    )

    with pytest.raises(Exception):
        response.answer = "changed"  # type: ignore[misc]
