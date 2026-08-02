"""Tests for PromptBuilder."""

from __future__ import annotations

import pytest

from src.generation.prompt_builder import PromptBuilder
from src.preprocessing.chunk import Chunk
from src.retrieval.result import RetrievalResult


def make_result(
    text: str,
    *,
    chunk_id: int = 0,
    source: str = "paper.pdf",
    score: float = 0.9,
) -> RetrievalResult:
    """Create a deterministic retrieval result for tests."""
    return RetrievalResult(
        chunk=Chunk(
            text=text,
            chunk_id=chunk_id,
            source=source,
        ),
        score=score,
    )


def test_build_basic_prompt():
    builder = PromptBuilder()

    prompt = builder.build(
        query="What causes pneumonia?",
        contexts=[
            make_result("Pneumonia may be caused by bacteria."),
            make_result("Viruses can also cause pneumonia.", chunk_id=1),
        ],
    )

    assert "You are a helpful research assistant." in prompt
    assert "Context:" in prompt
    assert "[1]" in prompt
    assert "[2]" in prompt
    assert "Pneumonia may be caused by bacteria." in prompt
    assert "Viruses can also cause pneumonia." in prompt
    assert "Question:\nWhat causes pneumonia?" in prompt
    assert prompt.endswith("Answer:")


def test_build_preserves_context_order():
    builder = PromptBuilder()

    prompt = builder.build(
        query="Question?",
        contexts=[
            make_result("first", chunk_id=1),
            make_result("second", chunk_id=2),
            make_result("third", chunk_id=3),
        ],
    )

    assert prompt.index("first") < prompt.index("second") < prompt.index("third")


def test_build_numbers_contexts_sequentially():
    builder = PromptBuilder()

    prompt = builder.build(
        query="Question?",
        contexts=[
            make_result("alpha"),
            make_result("beta"),
            make_result("gamma"),
        ],
    )

    assert "[1]\nalpha" in prompt
    assert "[2]\nbeta" in prompt
    assert "[3]\ngamma" in prompt


def test_build_strips_query_whitespace():
    builder = PromptBuilder()

    prompt = builder.build(
        query="   What is RAG?   ",
        contexts=[make_result("Relevant context.")],
    )

    assert "Question:\nWhat is RAG?" in prompt
    assert "   What is RAG?   " not in prompt


def test_build_strips_context_whitespace():
    builder = PromptBuilder()

    prompt = builder.build(
        query="Question?",
        contexts=[make_result("   context text   ")],
    )

    assert "[1]\ncontext text" in prompt
    assert "   context text   " not in prompt


def test_build_ignores_empty_contexts():
    builder = PromptBuilder()

    prompt = builder.build(
        query="Question?",
        contexts=[
            make_result(""),
            make_result("   ", chunk_id=1),
            make_result("usable", chunk_id=2),
        ],
    )

    assert "[1]\nusable" in prompt
    assert "[2]" not in prompt


def test_build_rejects_when_all_contexts_are_empty():
    builder = PromptBuilder()

    with pytest.raises(ValueError, match="at least one non-empty context"):
        builder.build(
            query="Question?",
            contexts=[
                make_result(""),
                make_result("   "),
            ],
        )


def test_build_respects_max_contexts():
    builder = PromptBuilder()

    prompt = builder.build(
        query="Question?",
        contexts=[
            make_result("first"),
            make_result("second"),
            make_result("third"),
        ],
        max_contexts=2,
    )

    assert "first" in prompt
    assert "second" in prompt
    assert "third" not in prompt
    assert "[3]" not in prompt


def test_max_contexts_applies_after_empty_context_filtering():
    builder = PromptBuilder()

    prompt = builder.build(
        query="Question?",
        contexts=[
            make_result(""),
            make_result("first"),
            make_result("second"),
        ],
        max_contexts=1,
    )

    assert "[1]\nfirst" in prompt
    assert "second" not in prompt


def test_include_scores_formats_scores():
    builder = PromptBuilder(include_scores=True)

    prompt = builder.build(
        query="Question?",
        contexts=[
            make_result("alpha", score=0.87654321),
        ],
    )

    assert "[1] score=0.876543" in prompt


def test_scores_are_hidden_by_default():
    builder = PromptBuilder()

    prompt = builder.build(
        query="Question?",
        contexts=[make_result("alpha", score=0.87654321)],
    )

    assert "score=" not in prompt


def test_custom_system_instruction():
    builder = PromptBuilder(
        system_instruction="Answer as a scientific assistant."
    )

    prompt = builder.build(
        query="Question?",
        contexts=[make_result("alpha")],
    )

    assert prompt.startswith("Answer as a scientific assistant.")
    assert PromptBuilder.DEFAULT_SYSTEM_INSTRUCTION not in prompt


def test_system_instruction_is_stripped():
    builder = PromptBuilder(
        system_instruction="   Custom instruction.   "
    )

    assert builder.system_instruction == "Custom instruction."


def test_rejects_empty_system_instruction():
    with pytest.raises(ValueError, match="system_instruction cannot be empty"):
        PromptBuilder(system_instruction="   ")


def test_rejects_non_string_system_instruction():
    with pytest.raises(TypeError, match="system_instruction must be a string"):
        PromptBuilder(system_instruction=123)  # type: ignore[arg-type]


def test_rejects_non_boolean_include_scores():
    with pytest.raises(TypeError, match="include_scores must be a boolean"):
        PromptBuilder(include_scores=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_rejects_empty_query(query: str):
    builder = PromptBuilder()

    with pytest.raises(ValueError, match="query cannot be empty"):
        builder.build(
            query=query,
            contexts=[make_result("alpha")],
        )


def test_rejects_non_string_query():
    builder = PromptBuilder()

    with pytest.raises(TypeError, match="query must be a string"):
        builder.build(
            query=123,  # type: ignore[arg-type]
            contexts=[make_result("alpha")],
        )


@pytest.mark.parametrize(
    "contexts",
    [
        "not-a-sequence-of-results",
        b"bytes",
        123,
        None,
    ],
)
def test_rejects_invalid_context_container(contexts):
    builder = PromptBuilder()

    with pytest.raises(TypeError, match="contexts must be a sequence"):
        builder.build(
            query="Question?",
            contexts=contexts,  # type: ignore[arg-type]
        )


def test_rejects_non_retrieval_result_items():
    builder = PromptBuilder()

    with pytest.raises(TypeError, match="all contexts must be RetrievalResult"):
        builder.build(
            query="Question?",
            contexts=[
                make_result("alpha"),
                "invalid",  # type: ignore[list-item]
            ],
        )


@pytest.mark.parametrize("max_contexts", [0, -1])
def test_rejects_non_positive_max_contexts(max_contexts: int):
    builder = PromptBuilder()

    with pytest.raises(ValueError, match="max_contexts must be greater than zero"):
        builder.build(
            query="Question?",
            contexts=[make_result("alpha")],
            max_contexts=max_contexts,
        )


@pytest.mark.parametrize("max_contexts", [1.5, "2", True])
def test_rejects_non_integer_max_contexts(max_contexts):
    builder = PromptBuilder()

    with pytest.raises(TypeError, match="max_contexts must be an integer or None"):
        builder.build(
            query="Question?",
            contexts=[make_result("alpha")],
            max_contexts=max_contexts,  # type: ignore[arg-type]
        )


def test_accepts_tuple_contexts():
    builder = PromptBuilder()

    prompt = builder.build(
        query="Question?",
        contexts=(
            make_result("alpha"),
            make_result("beta"),
        ),
    )

    assert "[1]\nalpha" in prompt
    assert "[2]\nbeta" in prompt


def test_prompt_has_expected_section_order():
    builder = PromptBuilder()

    prompt = builder.build(
        query="Question?",
        contexts=[make_result("alpha")],
    )

    instruction_index = prompt.index(builder.system_instruction)
    context_index = prompt.index("Context:")
    question_index = prompt.index("Question:")
    answer_index = prompt.index("Answer:")

    assert instruction_index < context_index < question_index < answer_index


def test_build_is_deterministic():
    builder = PromptBuilder()
    contexts = [
        make_result("alpha", score=0.8),
        make_result("beta", score=0.7),
    ]

    first = builder.build(
        query="Question?",
        contexts=contexts,
    )
    second = builder.build(
        query="Question?",
        contexts=contexts,
    )

    assert first == second
