"""Prompt construction utilities for retrieval-augmented generation."""

from __future__ import annotations

from collections.abc import Sequence

from src.retrieval.result import RetrievalResult


class PromptBuilder:
    """Build grounded prompts from a query and retrieved contexts."""

    DEFAULT_SYSTEM_INSTRUCTION = (
        "You are a helpful research assistant. "
        "Answer the question using only the provided context. "
        "If the context is insufficient, say that the answer cannot be "
        "determined from the provided context."
    )

    def __init__(
        self,
        system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
        *,
        include_scores: bool = False,
    ) -> None:
        """Initialize the prompt builder.

        Args:
            system_instruction: Instruction placed at the beginning of
                every prompt.
            include_scores: Whether retrieval scores should be shown
                beside each context block.

        Raises:
            TypeError: If inputs have invalid types.
            ValueError: If the system instruction is empty.
        """
        if not isinstance(system_instruction, str):
            raise TypeError("system_instruction must be a string.")

        cleaned_instruction = system_instruction.strip()

        if not cleaned_instruction:
            raise ValueError("system_instruction cannot be empty.")

        if not isinstance(include_scores, bool):
            raise TypeError("include_scores must be a boolean.")

        self.system_instruction = cleaned_instruction
        self.include_scores = include_scores

    def build(
        self,
        query: str,
        contexts: Sequence[RetrievalResult],
        *,
        max_contexts: int | None = None,
    ) -> str:
        """Build a prompt from a query and retrieved results.

        Empty context texts are ignored while the original result order
        is preserved.

        Args:
            query: User question.
            contexts: Retrieved context results.
            max_contexts: Optional maximum number of non-empty contexts
                to include.

        Returns:
            A formatted prompt ready for a text-generation backend.

        Raises:
            TypeError: If an argument has an invalid type.
            ValueError: If the query is empty, ``max_contexts`` is invalid,
                or no usable context remains.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError("query cannot be empty.")

        if isinstance(contexts, (str, bytes)) or not isinstance(
            contexts,
            Sequence,
        ):
            raise TypeError(
                "contexts must be a sequence of RetrievalResult objects."
            )

        if max_contexts is not None:
            if (
                not isinstance(max_contexts, int)
                or isinstance(max_contexts, bool)
            ):
                raise TypeError("max_contexts must be an integer or None.")

            if max_contexts <= 0:
                raise ValueError(
                    "max_contexts must be greater than zero."
                )

        usable_contexts: list[RetrievalResult] = []

        for result in contexts:
            if not isinstance(result, RetrievalResult):
                raise TypeError(
                    "all contexts must be RetrievalResult objects."
                )

            if result.chunk.text.strip():
                usable_contexts.append(result)

        if max_contexts is not None:
            usable_contexts = usable_contexts[:max_contexts]

        if not usable_contexts:
            raise ValueError("at least one non-empty context is required.")

        context_blocks = [
            self._format_context(index, result)
            for index, result in enumerate(
                usable_contexts,
                start=1,
            )
        ]

        return (
            f"{self.system_instruction}\n\n"
            "Context:\n\n"
            f"{'\n\n'.join(context_blocks)}\n\n"
            f"Question:\n{cleaned_query}\n\n"
            "Answer:"
        )

    def _format_context(
        self,
        index: int,
        result: RetrievalResult,
    ) -> str:
        """Format one retrieval result as a numbered context block."""
        header = f"[{index}]"

        if self.include_scores:
            header = f"{header} score={result.score:.6f}"

        return f"{header}\n{result.chunk.text.strip()}"
