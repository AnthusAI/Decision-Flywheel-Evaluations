"""Auditable logical-cell metrics and de-duplicated physical-request totals.

These reports describe collected rows only; they make no claim that a benchmark
matrix is complete unless an exact expected logical-id manifest is supplied.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

from .metrics import MetricSummary, Observation, summarize, validate_rows


@dataclass(frozen=True)
class PhysicalSummary:
    """One-count-per-physical-request operational accounting."""

    requests: int
    cache_hits: int
    failures: int
    malformed: int
    missing: int
    attempts: int
    numeric_usage: Mapping[str, float]
    usage_coverage: int
    usage_complete: bool
    latency_coverage: int
    latency_ms_total: float | None
    latency_ms_mean: float | None
    latency_ms_max: float | None
    physical_request_ids: tuple[str, ...]
    physical_request_provenance: Mapping[str, Mapping[str, str] | None]


@dataclass(frozen=True)
class CellReport:
    condition: str
    draw: int
    order: str
    metrics: MetricSummary
    logical_cells: int
    physical: PhysicalSummary

    # Legacy field names now deliberately refer to physical rather than logical work.
    @property
    def requests(self) -> int: return self.physical.requests
    @property
    def cache_hits(self) -> int: return self.physical.cache_hits
    @property
    def failures(self) -> int: return self.physical.failures
    @property
    def missing(self) -> int: return self.physical.missing
    @property
    def attempts(self) -> int: return self.physical.attempts
    @property
    def numeric_usage(self) -> Mapping[str, float]: return self.physical.numeric_usage


@dataclass(frozen=True)
class MeanDrawMetrics:
    """Arithmetic mean of per-draw metrics, never a pooled-prediction score."""

    draws: int
    accuracy: float | None
    macro_f1: float | None
    log_loss: float | None
    brier: float | None
    ece: float | None
    recall: Mapping[str, float | None]


@dataclass(frozen=True)
class PooledReport:
    condition: str
    order: str
    draws: tuple[int, ...]
    metrics: MetricSummary
    mean_draw_metrics: MeanDrawMetrics
    logical_cells: int
    physical: PhysicalSummary


@dataclass(frozen=True)
class StudyReport:
    """The only totals safe to use across all condition/draw report groups."""

    cells: tuple[CellReport, ...]
    pooled: tuple[PooledReport, ...]
    logical_cells: int
    physical: PhysicalSummary


@dataclass(frozen=True)
class HolmCorrection:
    """Family correction only when every declared contrast has a valid p-value."""
    available: bool
    adjusted_p_values: Mapping[str, float]
    reason: str | None = None


def report_cells(
    rows: Sequence[Observation],
    labels: Sequence[str],
    *,
    expected_request_ids: Sequence[str] | None = None,
    expected_logical_ids: Sequence[str] | None = None,
    expected_matrix_ids: Sequence[tuple[str, int, str, str]] | None = None,
) -> tuple[CellReport, ...]:
    """Return one report per condition/draw/order logical matrix cell group.

    ``expected_request_ids`` is retained as the historical manifest name; both
    manifest arguments refer to the stable logical observation ``request_id``.
    """
    _validate_matrix(rows, labels, expected_request_ids, expected_logical_ids, expected_matrix_ids)
    groups: dict[tuple[str, int, str], list[Observation]] = defaultdict(list)
    for row in rows:
        groups[(row.condition, row.draw, row.order)].append(row)
    return tuple(
        CellReport(condition, draw, order, summarize(group, labels), len(group), _physical_summary(group))
        for (condition, draw, order), group in sorted(groups.items())
    )


def report_pooled(
    rows: Sequence[Observation],
    labels: Sequence[str],
    *,
    expected_request_ids: Sequence[str] | None = None,
    expected_logical_ids: Sequence[str] | None = None,
    expected_matrix_ids: Sequence[tuple[str, int, str, str]] | None = None,
) -> tuple[PooledReport, ...]:
    """Pool predictions by condition/order while retaining a mean-of-draw view."""
    cells = report_cells(rows, labels, expected_request_ids=expected_request_ids, expected_logical_ids=expected_logical_ids, expected_matrix_ids=expected_matrix_ids)
    groups: dict[tuple[str, str], list[CellReport]] = defaultdict(list)
    row_groups: dict[tuple[str, str], list[Observation]] = defaultdict(list)
    for cell in cells:
        groups[(cell.condition, cell.order)].append(cell)
    for row in rows:
        row_groups[(row.condition, row.order)].append(row)
    return tuple(
        PooledReport(condition, order, tuple(sorted(cell.draw for cell in group)),
                     summarize(row_groups[(condition, order)], labels), _mean_draw_metrics(group, labels),
                     len(row_groups[(condition, order)]), _physical_summary(row_groups[(condition, order)]))
        for (condition, order), group in sorted(groups.items())
    )


def report_study(
    rows: Sequence[Observation],
    labels: Sequence[str],
    *,
    expected_request_ids: Sequence[str] | None = None,
    expected_logical_ids: Sequence[str] | None = None,
    expected_matrix_ids: Sequence[tuple[str, int, str, str]] | None = None,
) -> StudyReport:
    """Report groups plus globally de-duplicated physical accounting."""
    cells = report_cells(rows, labels, expected_request_ids=expected_request_ids, expected_logical_ids=expected_logical_ids, expected_matrix_ids=expected_matrix_ids)
    pooled = report_pooled(rows, labels, expected_request_ids=expected_request_ids, expected_logical_ids=expected_logical_ids, expected_matrix_ids=expected_matrix_ids)
    return StudyReport(cells, pooled, len(rows), _physical_summary(rows))


def mean_draw_accuracy(cells: Sequence[CellReport]) -> dict[tuple[str, str], float | None]:
    """Compatibility helper; prefer ``PooledReport.mean_draw_metrics`` for new reports."""
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for cell in cells:
        if cell.metrics.accuracy is not None:
            grouped[(cell.condition, cell.order)].append(cell.metrics.accuracy)
    return {key: sum(values) / len(values) if values else None for key, values in grouped.items()}


def percentage_effect(treatment: float, baseline: float) -> dict[str, float | None]:
    """Keep percentage-point change distinct from relative percent change."""
    return {"percentage_points": 100 * (treatment - baseline),
            "relative_percent": None if baseline == 0 else 100 * (treatment - baseline) / baseline}


def holm_correction(p_values: Mapping[str, float | None]) -> HolmCorrection:
    """Holm-adjust exactly the supplied declared primary-comparison family."""
    if not p_values or any(not isinstance(name, str) or not name or value is None or not isinstance(value, (int, float))
                           or isinstance(value, bool) or not 0 <= value <= 1 for name, value in p_values.items()):
        return HolmCorrection(False, {}, "valid p-values are unavailable for every declared contrast")
    ordered=sorted(p_values.items(),key=lambda item:(item[1],item[0])); total=len(ordered); adjusted={}; running=0.0
    for index,(name,value) in enumerate(ordered):
        running=max(running,min(1.0,value*(total-index))); adjusted[name]=running
    return HolmCorrection(True, adjusted)


def _validate_matrix(
    rows: Sequence[Observation], labels: Sequence[str], expected_request_ids: Sequence[str] | None,
    expected_logical_ids: Sequence[str] | None, expected_matrix_ids: Sequence[tuple[str, int, str, str]] | None,
) -> None:
    validate_rows(rows, labels)
    request_ids = [row.request_id for row in rows]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("duplicate logical observation ID")
    matrix_ids = [row.logical_id for row in rows]
    if len(set(matrix_ids)) != len(matrix_ids):
        raise ValueError("duplicate logical matrix cell")
    if sum(value is not None for value in (expected_request_ids, expected_logical_ids, expected_matrix_ids)) > 1:
        raise ValueError("provide one expected logical-ID manifest")
    if expected_matrix_ids is not None:
        expected = tuple(expected_matrix_ids)
        if len(set(expected)) != len(expected) or any(
            not isinstance(value, tuple) or len(value) != 4
            or not isinstance(value[0], str) or not value[0]
            or isinstance(value[1], bool) or not isinstance(value[1], int) or value[1] < 0
            or not isinstance(value[2], str) or not value[2]
            or not isinstance(value[3], str) or not value[3]
            for value in expected
        ):
            raise ValueError("expected matrix IDs must be unique logical cell tuples")
        observed = set(matrix_ids)
        if observed - set(expected):
            raise ValueError("observations outside expected matrix")
        if set(expected) - observed:
            raise ValueError("missing expected matrix cells")
        return
    expected_ids = expected_logical_ids if expected_logical_ids is not None else expected_request_ids
    if expected_ids is None:
        return
    if any(not isinstance(value, str) or not value for value in expected_ids) or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("expected logical IDs must be unique non-empty strings")
    observed, expected = set(request_ids), set(expected_ids)
    if observed - expected:
        raise ValueError("observations outside expected matrix")
    if expected - observed:
        raise ValueError("missing expected matrix cells")


def _physical_summary(rows: Sequence[Observation]) -> PhysicalSummary:
    physical: dict[str, Observation] = {}
    for row in rows:
        previous = physical.get(row.physical_id)
        if previous is not None and _physical_signature(previous) != _physical_signature(row):
            raise ValueError("shared physical request has conflicting operational provenance")
        physical[row.physical_id] = row
    unique = tuple(physical.values())
    usage: dict[str, float] = defaultdict(float)
    for row in unique:
        if row.usage is not None:
            for name, value in row.usage.items():
                usage[name] += float(value)
    latencies = [float(row.latency_ms) for row in unique if row.latency_ms is not None]
    return PhysicalSummary(
        requests=len(unique), cache_hits=sum(row.cache_hit for row in unique),
        failures=sum(row.status in {"failed", "malformed"} for row in unique),
        malformed=sum(row.status == "malformed" for row in unique),
        missing=sum(row.status == "missing" for row in unique),
        attempts=sum(row.attempt_count for row in unique), numeric_usage=dict(usage),
        usage_coverage=sum(row.usage is not None for row in unique),
        usage_complete=all(row.usage is not None for row in unique),
        latency_coverage=len(latencies), latency_ms_total=sum(latencies) if latencies else None,
        latency_ms_mean=sum(latencies) / len(latencies) if latencies else None,
        latency_ms_max=max(latencies) if latencies else None,
        physical_request_ids=tuple(sorted(physical)),
        physical_request_provenance={key: dict(row.physical_request_provenance) if row.physical_request_provenance is not None else None
                                     for key, row in sorted(physical.items())},
    )


def _physical_signature(row: Observation) -> tuple[object, ...]:
    """Fields that must be immutable for one physical call, independent of cell."""
    return (row.target_id, row.true_label, row.status, row.predicted_label, tuple(sorted(row.probabilities.items())) if row.probabilities else None,
            row.model_id, tuple(sorted(row.usage.items())) if row.usage is not None else None,
            row.latency_ms, row.attempt_count, row.cache_hit,
            tuple(sorted(row.physical_request_provenance.items())) if row.physical_request_provenance else None)


def _mean_draw_metrics(cells: Sequence[CellReport], labels: Sequence[str]) -> MeanDrawMetrics:
    metrics = [cell.metrics for cell in cells]
    def mean(name: str) -> float | None:
        values = [getattr(metric, name) for metric in metrics]
        return None if any(value is None for value in values) else sum(values) / len(values)
    return MeanDrawMetrics(
        draws=len(metrics), accuracy=mean("accuracy"), macro_f1=mean("macro_f1"),
        log_loss=mean("log_loss"), brier=mean("brier"), ece=mean("ece"),
        recall={label: (None if any(metric.recall[label] is None for metric in metrics)
                        else sum(metric.recall[label] for metric in metrics) / len(metrics)) for label in labels},
    )
