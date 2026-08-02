"""Abstract interface for text generation backends."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class BaseGenerator(ABC):
    """Abstract base class for all text generation backends."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the underlying model identifier."""
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate a response from a prompt.

        Args:
            prompt: Fully formatted prompt.
            max_new_tokens: Maximum number of generated tokens.
            temperature: Sampling temperature.

        Returns:
            Generated response text.
        """
        raise NotImplementedError

    def __call__(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Forward calls to :meth:`generate`."""
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
