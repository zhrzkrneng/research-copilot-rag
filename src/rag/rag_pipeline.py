"""End-to-end retrieval-augmented generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from src.generation.base_generator import BaseGenerator
from src.generation.prompt_builder import PromptBuilder
from src.retrieval.base_retriever import BaseRetriever
from src.retrieval.result import RetrievalResult


@dataclass(frozen=True, slots=True)
class RAGResponse:
    """Structured output produced by the RAG pipeline."""

    answer: str
    prompt: str
    contexts: tuple[RetrievalResult, ...]


class RAGPipeline:
    """Connect retrieval, prompt construction, and text generation."""

    def __init__(
        self,
        retriever: BaseRetriever,
        prompt_builder: PromptBuilder,
        generator: BaseGenerator,
    ) -> None:
        """Initialize the pipeline with its three core dependencies."""
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.generator = generator

    def run(
        self,
        query: str,
        *,
        top_k: int = 5,
        max_contexts: int | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> RAGResponse:
        """Run retrieval-augmented generation for one query.

        Args:
            query: Natural-language question.
            top_k: Number of retrieval results requested.
            max_contexts: Optional maximum number of retrieved contexts placed
                in the prompt.
            max_new_tokens: Maximum number of generated tokens.
            temperature: Sampling temperature passed to the generator.

        Returns:
            Structured answer, prompt, and retrieved contexts.

        Raises:
            TypeError: If ``query`` is not a string.
            ValueError: If ``query`` is empty.
            ValueError: If retrieval returns no contexts.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError("query cannot be empty.")

        contexts = self.retriever.retrieve(
            cleaned_query,
            top_k=top_k,
        )

        if not contexts:
            raise ValueError(
                "retrieval returned no contexts for the query."
            )

        prompt = self.prompt_builder.build(
            query=cleaned_query,
            contexts=contexts,
            max_contexts=max_contexts,
        )

        answer = self.generator.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        return RAGResponse(
            answer=answer,
            prompt=prompt,
            contexts=tuple(contexts),
        )

    def answer(
        self,
        query: str,
        *,
        top_k: int = 5,
        max_contexts: int | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Return only the generated answer for one query."""
        return self.run(
            query,
            top_k=top_k,
            max_contexts=max_contexts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        ).answer
