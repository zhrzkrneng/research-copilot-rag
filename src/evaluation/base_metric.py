"""Base evaluation metric interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique metric name."""
        raise NotImplementedError

    @abstractmethod
    def compute(
        self,
        prediction: str,
        reference: str,
    ) -> float:
        """Compute the metric score for one prediction-reference pair."""
        raise NotImplementedError
