"""Evaluation orchestrator."""

from __future__ import annotations

from collections.abc import Sequence

from src.evaluation.base_metric import BaseMetric


class Evaluator:
    """Evaluate predictions using one or more metrics."""

    def __init__(self, metrics: Sequence[BaseMetric]) -> None:
        if (
            not isinstance(metrics, Sequence)
            or isinstance(metrics, (str, bytes))
        ):
            raise TypeError(
                "metrics must be a sequence of BaseMetric instances."
            )

        if len(metrics) == 0:
            raise ValueError("metrics cannot be empty.")

        validated_metrics: list[BaseMetric] = []

        for metric in metrics:
            if not isinstance(metric, BaseMetric):
                raise TypeError(
                    "all metrics must inherit from BaseMetric."
                )
            validated_metrics.append(metric)

        names = [metric.name for metric in validated_metrics]

        if len(names) != len(set(names)):
            raise ValueError(
                "metric names must be unique."
            )

        self._metrics = tuple(validated_metrics)

    @property
    def metrics(self) -> tuple[BaseMetric, ...]:
        return self._metrics

    def evaluate(
        self,
        prediction: str,
        reference: str,
    ) -> dict[str, float]:
        if not isinstance(prediction, str):
            raise TypeError("prediction must be a string.")

        if not isinstance(reference, str):
            raise TypeError("reference must be a string.")

        return {
            metric.name: metric.compute(
                prediction,
                reference,
            )
            for metric in self._metrics
        }

    def evaluate_dataset(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
    ) -> dict[str, float]:
        if (
            not isinstance(predictions, Sequence)
            or isinstance(predictions, (str, bytes))
        ):
            raise TypeError(
                "predictions must be a sequence of strings."
            )

        if (
            not isinstance(references, Sequence)
            or isinstance(references, (str, bytes))
        ):
            raise TypeError(
                "references must be a sequence of strings."
            )

        if len(predictions) != len(references):
            raise ValueError(
                "predictions and references must have the same length."
            )

        if len(predictions) == 0:
            raise ValueError("dataset cannot be empty.")

        totals = {
            metric.name: 0.0
            for metric in self._metrics
        }

        for prediction, reference in zip(
            predictions,
            references,
            strict=True,
        ):
            scores = self.evaluate(prediction, reference)

            for name, score in scores.items():
                totals[name] += score

        size = len(predictions)

        return {
            name: total / size
            for name, total in totals.items()
        }
