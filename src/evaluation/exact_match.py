"""Exact Match evaluation metric."""

from __future__ import annotations

from src.evaluation.base_metric import BaseMetric


class ExactMatch(BaseMetric):
    """Binary exact string match metric."""

    @property
    def name(self) -> str:
        return "exact_match"

    def compute(
        self,
        prediction: str,
        reference: str,
    ) -> float:
        if not isinstance(prediction, str):
            raise TypeError("prediction must be a string.")

        if not isinstance(reference, str):
            raise TypeError("reference must be a string.")

        return float(
            prediction.strip() == reference.strip()
        )
