"""Token-level F1 evaluation metric."""

from __future__ import annotations

from collections import Counter

from src.evaluation.base_metric import BaseMetric
from src.evaluation.normalization import normalize_answer


class TokenF1(BaseMetric):
    """Compute token-level F1 using normalized token multiplicities."""

    @property
    def name(self) -> str:
        return "token_f1"

    def compute(self, prediction: str, reference: str) -> float:
        if not isinstance(prediction, str):
            raise TypeError("prediction must be a string.")
        if not isinstance(reference, str):
            raise TypeError("reference must be a string.")

        prediction_tokens = normalize_answer(prediction).split()
        reference_tokens = normalize_answer(reference).split()

        if not prediction_tokens and not reference_tokens:
            return 1.0
        if not prediction_tokens or not reference_tokens:
            return 0.0

        overlap = sum(
            (Counter(prediction_tokens) & Counter(reference_tokens)).values()
        )
        if overlap == 0:
            return 0.0

        precision = overlap / len(prediction_tokens)
        recall = overlap / len(reference_tokens)
        return 2.0 * precision * recall / (precision + recall)
