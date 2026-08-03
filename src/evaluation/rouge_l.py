"""ROUGE-L evaluation metric."""

from __future__ import annotations

from src.evaluation.base_metric import BaseMetric
from src.evaluation.normalization import normalize_answer


class RougeL(BaseMetric):
    """Compute ROUGE-L F1 using the longest common subsequence."""

    @property
    def name(self) -> str:
        return "rouge_l"

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

        lcs_length = self._lcs_length(
            prediction_tokens,
            reference_tokens,
        )
        if lcs_length == 0:
            return 0.0

        precision = lcs_length / len(prediction_tokens)
        recall = lcs_length / len(reference_tokens)

        return 2.0 * precision * recall / (precision + recall)

    @staticmethod
    def _lcs_length(
        prediction_tokens: list[str],
        reference_tokens: list[str],
    ) -> int:
        """Return the longest common subsequence length."""
        if not prediction_tokens or not reference_tokens:
            return 0

        previous = [0] * (len(reference_tokens) + 1)

        for prediction_token in prediction_tokens:
            current = [0]
            for index, reference_token in enumerate(
                reference_tokens,
                start=1,
            ):
                if prediction_token == reference_token:
                    current.append(previous[index - 1] + 1)
                else:
                    current.append(
                        max(current[index - 1], previous[index])
                    )
            previous = current

        return previous[-1]
