"""Pure, text-free logical observations and label-aware evaluation metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

STATUSES = frozenset({"completed", "failed", "malformed", "missing"})


@dataclass(frozen=True)
class Observation:
    """One logical matrix cell, optionally backed by a shared physical request.

    ``request_id`` remains the stable logical observation identifier for backwards
    compatibility. ``physical_request_id`` identifies attempts, usage, cache and
    latency; a legacy row without it is its own physical request.
    """

    request_id: str
    target_id: str
    condition: str
    draw: int
    order: str
    true_label: str
    predicted_label: str | None
    status: str
    probabilities: Mapping[str, float] | None = None
    model_id: str | None = None
    usage: Mapping[str, float] | None = None
    latency_ms: float | None = None
    attempt_count: int = 1
    cache_hit: bool = False
    physical_request_id: str | None = None
    physical_request_provenance: Mapping[str, str] | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not all(isinstance(v, str) and v for v in (
            self.request_id, self.target_id, self.condition, self.order,
            self.true_label, self.status,
        )):
            raise ValueError("observation identifiers must be non-empty strings")
        if isinstance(self.draw, bool) or not isinstance(self.draw, int) or self.draw < 0:
            raise ValueError("draw must be non-negative")
        if self.predicted_label is not None and (
            not isinstance(self.predicted_label, str) or not self.predicted_label
        ):
            raise ValueError("predicted_label must be a non-empty string or None")
        if self.status not in STATUSES:
            raise ValueError("status must be completed, failed, malformed, or missing")
        if self.physical_request_id is not None and (
            not isinstance(self.physical_request_id, str) or not self.physical_request_id
        ):
            raise ValueError("physical_request_id must be a non-empty string or None")
        if self.physical_request_provenance is not None and any(
            not isinstance(key, str) or not key or not isinstance(value, str) or not value
            for key, value in self.physical_request_provenance.items()
        ):
            raise ValueError("physical request provenance must contain non-empty strings")
        if isinstance(self.attempt_count, bool) or not isinstance(self.attempt_count, int) or self.attempt_count < 0:
            raise ValueError("attempt_count must be non-negative")
        if not isinstance(self.cache_hit, bool):
            raise ValueError("cache_hit must be boolean")
        if self.latency_ms is not None and (
            not isinstance(self.latency_ms, (int, float)) or isinstance(self.latency_ms, bool)
            or not math.isfinite(self.latency_ms) or self.latency_ms < 0
        ):
            raise ValueError("latency_ms must be finite and non-negative")
        if self.confidence is not None and (
            not isinstance(self.confidence, (int, float)) or isinstance(self.confidence, bool)
            or not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("confidence must be finite and within [0, 1]")
        if self.usage is not None and any(
            not isinstance(key, str) or not key or not isinstance(value, (int, float))
            or isinstance(value, bool) or not math.isfinite(value) or value < 0
            for key, value in self.usage.items()
        ):
            raise ValueError("usage must be finite non-negative numeric values")

    @property
    def logical_id(self) -> tuple[str, int, str, str]:
        """Matrix identity; unlike request provenance, this cannot be shared."""
        return (self.condition, self.draw, self.order, self.target_id)

    @property
    def physical_id(self) -> str:
        return self.physical_request_id or self.request_id


@dataclass(frozen=True)
class MetricSummary:
    total: int
    completed: int
    failed: int
    malformed: int
    missing: int
    label_coverage: int
    accuracy: float | None
    macro_f1: float | None
    recall: Mapping[str, float]
    probability_coverage: int
    log_loss: float | None
    brier: float | None
    ece: float | None
    reliability: tuple["ReliabilityBin", ...]

    @property
    def failures(self) -> int:
        """Compatibility total for failed and malformed scored cells."""
        return self.failed + self.malformed


@dataclass(frozen=True)
class ReliabilityBin:
    index: int
    lower: float
    upper: float
    mean_confidence: float | None
    accuracy: float | None
    count: int


def validate_rows(rows: Sequence[Observation], labels: Sequence[str]) -> tuple[str, ...]:
    """Validate declared labels before any metric selects a subset of rows."""
    labels = tuple(labels)
    if not labels or len(set(labels)) != len(labels) or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("labels must be unique non-empty strings")
    declared = set(labels)
    for row in rows:
        if not isinstance(row, Observation):
            raise ValueError("rows must be Observation instances")
        if row.true_label not in declared:
            raise ValueError(f"unknown true label for logical observation {row.request_id}")
        if row.predicted_label is not None and row.predicted_label not in declared:
            raise ValueError(f"unknown predicted label for logical observation {row.request_id}")
        if row.status == "completed" and row.predicted_label is None:
            raise ValueError(f"completed observation {row.request_id} has no predicted label")
        if row.probabilities is not None and not _valid_probs(row.probabilities, labels):
            raise ValueError(f"malformed probabilities for logical observation {row.request_id}")
    return labels


def summarize(rows: Sequence[Observation], labels: Sequence[str], *, ece_bins: int = 10) -> MetricSummary:
    labels = validate_rows(rows, labels)
    if isinstance(ece_bins, bool) or not isinstance(ece_bins, int) or ece_bins < 1:
        raise ValueError("ece_bins must be positive")
    completed = [row for row in rows if row.status == "completed"]
    recalls = {
        label: _ratio(
            sum(row.predicted_label == label and row.true_label == label for row in completed),
            sum(row.true_label == label for row in completed),
        )
        for label in labels
    }
    accuracy = _ratio(sum(row.predicted_label == row.true_label for row in completed), len(completed))
    f1 = []
    for label in labels:
        true_positive = sum(row.predicted_label == label and row.true_label == label for row in completed)
        false_positive = sum(row.predicted_label == label and row.true_label != label for row in completed)
        false_negative = sum(row.predicted_label != label and row.true_label == label for row in completed)
        f1.append(_ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative) or 0.0)
    status_counts = {status: sum(row.status == status for row in rows) for status in STATUSES}
    common = dict(
        total=len(rows), completed=len(completed), failed=status_counts["failed"],
        malformed=status_counts["malformed"], missing=status_counts["missing"],
        label_coverage=len(completed), accuracy=accuracy,
        macro_f1=sum(f1) / len(labels) if completed else None, recall=recalls,
    )
    valid = [row for row in completed if row.probabilities is not None]
    if len(valid) != len(completed) or not completed:
        return MetricSummary(**common, probability_coverage=len(valid), log_loss=None, brier=None, ece=None, reliability=())
    losses = [-math.log(max(row.probabilities[row.true_label], 1e-15)) for row in valid]
    briers = [sum((row.probabilities[label] - (1.0 if label == row.true_label else 0.0)) ** 2 for label in labels) for row in valid]
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(ece_bins)]
    for row in valid:
        top = max(labels, key=lambda label: row.probabilities[label])
        confidence = row.probabilities[top]
        buckets[min(ece_bins - 1, int(confidence * ece_bins))].append((confidence, top == row.true_label))
    reliability = tuple(ReliabilityBin(index, index / ece_bins, (index + 1) / ece_bins,
                                       sum(confidence for confidence, _ in bucket) / len(bucket),
                                       sum(correct for _, correct in bucket) / len(bucket), len(bucket))
                        for index, bucket in enumerate(buckets) if bucket)
    ece = sum(abs(item.mean_confidence - item.accuracy) * item.count for item in reliability) / len(valid)
    return MetricSummary(**common, probability_coverage=len(valid), log_loss=sum(losses) / len(valid), brier=sum(briers) / len(valid), ece=ece, reliability=reliability)


def reliability_bins(rows: Sequence[Observation], labels: Sequence[str], *, bins: int = 10) -> tuple[ReliabilityBin, ...]:
    """Return named fixed-width bins, including empty bins, for auditable plots."""
    labels = validate_rows(rows, labels)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be positive")
    values: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for row in rows:
        if row.status == "completed" and row.probabilities is not None:
            top = max(labels, key=lambda label: row.probabilities[label])
            confidence = row.probabilities[top]
            values[min(bins - 1, int(confidence * bins))].append((confidence, top == row.true_label))
    return tuple(ReliabilityBin(index, index / bins, (index + 1) / bins,
                                sum(confidence for confidence, _ in value) / len(value) if value else None,
                                sum(correct for _, correct in value) / len(value) if value else None,
                                len(value)) for index, value in enumerate(values))


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _valid_probs(probabilities: Mapping[str, float] | None, labels: Sequence[str]) -> bool:
    return isinstance(probabilities, Mapping) and set(probabilities) == set(labels) and all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1
        for value in probabilities.values()
    ) and math.isclose(sum(probabilities.values()), 1, abs_tol=0.01)
