"""Descriptive, text-free operational summaries for a frozen development pilot."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping, Sequence

from .metrics import Observation
from .pilot import PilotPlan
from .protocol import FrozenProtocol


_STATUSES = ("completed", "failed", "malformed", "missing")


def summarize_pilot_compatibility(
    protocol: FrozenProtocol,
    plan: PilotPlan,
    observations: Sequence[Observation],
) -> dict[str, object]:
    """Describe operational pilot compatibility without reading labels or scoring outputs."""
    protocol.validate()
    if not isinstance(plan, PilotPlan):
        raise ValueError("pilot compatibility summary requires a typed pilot plan")
    plan.validate()
    if plan.protocol_identity != protocol.identity:
        raise ValueError("pilot plan protocol identity does not match the frozen protocol")
    if plan.manifest_sha256 != protocol.dataset_manifest_sha256:
        raise ValueError("pilot plan manifest identity does not match the frozen protocol")
    rows = tuple(observations)
    cells = {cell.id: cell for cell in plan.cells}
    if len(rows) != len(cells):
        raise ValueError("pilot observations must exactly cover the 21-cell logical pilot matrix")

    physical: dict[str, Observation] = {}
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Observation):
            raise ValueError("pilot observations must be sanitized Observation values")
        cell = cells.get(row.request_id)
        if cell is None or row.request_id in seen:
            raise ValueError("pilot observation is outside the exact logical pilot cell matrix")
        seen.add(row.request_id)
        _validate_observation(row, cell, protocol)
        previous = physical.get(row.physical_request_id)
        if previous is not None and _physical_signature(previous) != _physical_signature(row):
            raise ValueError("shared physical pilot request has conflicting operational fields")
        physical[row.physical_request_id] = row
    if set(cells) != seen:
        raise ValueError("pilot observations are missing logical pilot cells")

    by_request_id = {row.request_id: row for row in rows}
    status_counts = _status_counts(rows)
    unique = tuple(physical.values())
    usage: dict[str, float] = defaultdict(float)
    for row in unique:
        if row.usage is not None:
            for name, value in row.usage.items():
                usage[name] += float(value)
    latencies = [float(row.latency_ms) for row in unique if row.latency_ms is not None]
    completed = tuple(row for row in rows if row.status == "completed")
    confidence_coverage = sum(row.confidence is not None for row in completed)
    probability_coverage = sum(row.probabilities is not None for row in completed)

    zero = tuple((cell, by_request_id[cell.id]) for cell in plan.cells if cell.per_label == 0)
    few = tuple((cell, by_request_id[cell.id]) for cell in plan.cells if cell.per_label > 0)
    successful = tuple((cell, by_request_id[cell.id]) for cell in plan.cells
                       if by_request_id[cell.id].status == "completed")
    largest = _largest_successful(successful)
    counters = tuple(sorted({cell.counter_identity for cell in plan.cells}))
    return {
        "schema": "decision-flywheel-evaluations/pilot-compatibility/v1",
        "scope": "development-only compatibility pilot; not a benchmark result",
        "provenance": {
            "pilot_plan_checksum": plan.checksum,
            "protocol_identity": plan.protocol_identity,
            "source_preflight_checksum": plan.source_preflight_checksum,
            "provider_identity": plan.cells[0].model,
        },
        "logical_status": {"total": len(rows), **status_counts},
        "physical": {
            "requests": len(unique),
            "attempts": sum(row.attempt_count for row in unique),
            "cache_hits": sum(row.cache_hit for row in unique),
            "status": _status_counts(unique),
            "usage": {"available": bool(usage), "coverage": sum(row.usage is not None for row in unique),
                      "numeric_usage": dict(usage)},
            "latency": {"available": bool(latencies), "coverage": len(latencies),
                        "latency_ms_total": sum(latencies) if latencies else None,
                        "latency_ms_mean": sum(latencies) / len(latencies) if latencies else None,
                        "latency_ms_max": max(latencies) if latencies else None},
        },
        "response_fields": {
            "explicit_confidence": {"available": confidence_coverage > 0, "coverage": confidence_coverage,
                                    "completed": len(completed)},
            "probability_distribution": {"available": probability_coverage > 0, "coverage": probability_coverage,
                                         "completed": len(completed)},
        },
        "paths": {
            "zero_shot": {"logical_cases": len(zero), "status": _status_counts(tuple(row for _cell, row in zero))},
            "few_shot": {
                "logical_cases": len(few), "status": _status_counts(tuple(row for _cell, row in few)),
                "groups": [
                    {"per_label": cell.per_label, "draw_seed": cell.draw_seed,
                     "status": _status_counts((row,))}
                    for cell, row in sorted(few, key=lambda item: (item[0].per_label, item[0].draw_seed or 0))
                ],
            },
        },
        "observed_context": {
            "counter_identities": list(counters),
            "largest_successful": largest,
            "scope": "successful pilot requests only; not a provider context-limit claim",
        },
    }


def _validate_observation(row: Observation, cell: object, protocol: FrozenProtocol) -> None:
    expected_condition = f"pilot:{cell.model}:{cell.selector}:{cell.per_label}"
    expected_provenance = {"model": cell.model, "dataset_revision": cell.dataset_revision,
                           "task_fingerprint": cell.task_fingerprint, "wire_fingerprint": cell.wire_fingerprint}
    if (row.condition != expected_condition or row.draw != (cell.draw_seed or 0)
            or row.order != cell.display_order or row.model_id != cell.model
            or row.target_id != cell.target_id or row.physical_request_id != cell.fingerprint):
        raise ValueError("pilot observation does not match its logical pilot cell provenance")
    if dict(row.physical_request_provenance or {}) != expected_provenance:
        raise ValueError("pilot observation physical provenance does not match its frozen cell")
    if row.status == "completed":
        if row.predicted_label not in protocol.task.labels:
            raise ValueError("completed pilot prediction is not a declared task label")
        if row.probabilities is not None:
            _validate_distribution(row.probabilities, protocol.task.labels)
    elif row.predicted_label is not None or row.probabilities is not None or row.confidence is not None:
        raise ValueError("non-completed pilot observations cannot carry response fields")


def _validate_distribution(probabilities: Mapping[str, float], labels: Sequence[str]) -> None:
    if not isinstance(probabilities, Mapping) or set(probabilities) != set(labels):
        raise ValueError("pilot probability distribution must cover every declared label")
    values = tuple(probabilities.values())
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           or not math.isfinite(value) or value < 0 or value > 1 for value in values):
        raise ValueError("pilot probability distribution must contain finite probabilities")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("pilot probability distribution must sum to one")


def _physical_signature(row: Observation) -> tuple[object, ...]:
    """Physical-call agreement check deliberately excludes protected ground truth."""
    return (
        row.target_id, row.status, row.predicted_label,
        tuple(sorted(row.probabilities.items())) if row.probabilities is not None else None,
        row.model_id, tuple(sorted(row.usage.items())) if row.usage is not None else None,
        row.latency_ms, row.confidence, row.attempt_count, row.cache_hit,
        tuple(sorted(row.physical_request_provenance.items())) if row.physical_request_provenance else None,
    )


def _status_counts(rows: Sequence[Observation]) -> dict[str, int]:
    return {status: sum(row.status == status for row in rows) for status in _STATUSES}


def _largest_successful(successful: Sequence[tuple[object, Observation]]) -> dict[str, object]:
    if not successful:
        return {"available": False}
    cell, _row = max(successful, key=lambda item: (len(item[0].example_ids), item[0].estimated_request_tokens, item[0].id))
    return {"available": True, "per_label": cell.per_label, "example_count": len(cell.example_ids),
            "estimated_request_tokens": cell.estimated_request_tokens}
