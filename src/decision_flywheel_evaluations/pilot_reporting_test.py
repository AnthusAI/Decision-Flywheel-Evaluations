import json
from dataclasses import replace

import pytest

from .metrics import Observation
from .pilot import PilotPlan, _checksum, plan_jev_pilot
from .pilot_reporting import summarize_pilot_compatibility
from .pilot_test import _fixture


class _PoisonTruth(str):
    def __eq__(self, _other):
        raise AssertionError("pilot compatibility reporting must not read ground truth")


def _observation(cell, labels, *, index, status="completed", probabilities=True, confidence=True):
    distribution = ({label: (1.0 if position == 0 else 0.0) for position, label in enumerate(labels)}
                    if probabilities else None)
    return Observation(
        request_id=cell.id, target_id=cell.target_id,
        condition=f"pilot:{cell.model}:{cell.selector}:{cell.per_label}",
        draw=cell.draw_seed or 0, order=cell.display_order,
        true_label=_PoisonTruth("private truth"),
        predicted_label=labels[0] if status == "completed" else None,
        status=status, probabilities=distribution,
        model_id=cell.model, usage={"tokens": float(index + 1)} if status == "completed" else None,
        latency_ms=float(index + 10) if status == "completed" else None,
        attempt_count=2 if status == "completed" else 1,
        cache_hit=False, physical_request_id=cell.fingerprint,
        physical_request_provenance={"model": cell.model, "dataset_revision": cell.dataset_revision,
                                     "task_fingerprint": cell.task_fingerprint, "wire_fingerprint": cell.wire_fingerprint},
        confidence=0.75 if confidence and status == "completed" else None,
    )


def _inputs():
    protocol, manifest, rows, source = _fixture()
    plan = plan_jev_pilot(protocol, manifest, rows, source)
    physical_indices = {fingerprint: index for index, fingerprint in enumerate(sorted({cell.fingerprint for cell in plan.cells}))}
    observations = tuple(_observation(cell, protocol.task.labels, index=physical_indices[cell.fingerprint])
                         for cell in plan.cells)
    return protocol, plan, observations


def _plan_with_shared_physical_request(plan):
    cells = list(plan.cells)
    first, second = cells[0], cells[1]
    cells[1] = replace(second, target_id=first.target_id, fingerprint=first.fingerprint,
                       wire_fingerprint=first.wire_fingerprint, example_ids=first.example_ids)
    cells_tuple = tuple(cells)
    return PilotPlan(plan.protocol_identity, plan.manifest_sha256, plan.source_preflight_checksum,
                     cells_tuple, _checksum(plan.protocol_identity, plan.manifest_sha256,
                                            plan.source_preflight_checksum, cells_tuple))


def test_a_pilot_compatibility_summary_binds_every_sanitized_observation_without_reading_ground_truth():
    protocol, plan, observations = _inputs()

    summary = summarize_pilot_compatibility(protocol, plan, observations)

    assert summary["scope"] == "development-only compatibility pilot; not a benchmark result"
    assert summary["provenance"] == {
        "pilot_plan_checksum": plan.checksum,
        "protocol_identity": protocol.identity,
        "source_preflight_checksum": plan.source_preflight_checksum,
        "provider_identity": plan.cells[0].model,
    }
    assert summary["logical_status"] == {"total": 21, "completed": 21, "failed": 0, "malformed": 0, "missing": 0}
    assert summary["physical"]["requests"] == plan.physical_count
    assert summary["physical"]["attempts"] == 2 * plan.physical_count
    assert summary["response_fields"] == {
        "explicit_confidence": {"available": True, "coverage": 21, "completed": 21},
        "probability_distribution": {"available": True, "coverage": 21, "completed": 21},
    }
    assert summary["paths"]["zero_shot"]["logical_cases"] == 1
    assert summary["paths"]["few_shot"]["logical_cases"] == 20
    assert len(summary["paths"]["few_shot"]["groups"]) == 20
    assert summary["observed_context"]["scope"] == "successful pilot requests only; not a provider context-limit claim"
    rendered = json.dumps(summary, sort_keys=True)
    assert "private truth" not in rendered
    assert "accuracy" not in rendered and "macro_f1" not in rendered and "predicted_label" not in rendered


def test_a_pilot_compatibility_summary_rejects_protocol_or_cell_provenance_tampering():
    protocol, plan, observations = _inputs()
    with pytest.raises(ValueError, match="protocol identity"):
        summarize_pilot_compatibility(replace(protocol, name="other"), plan, observations)
    with pytest.raises(ValueError, match="logical pilot cell"):
        summarize_pilot_compatibility(protocol, plan, (replace(observations[0], condition="pilot:wrong"), *observations[1:]))
    with pytest.raises(ValueError, match="physical provenance"):
        summarize_pilot_compatibility(protocol, plan, (replace(observations[0], physical_request_provenance={"model": "wrong"}), *observations[1:]))


def test_a_pilot_compatibility_summary_deduplicates_consistent_physical_usage_attempts_and_latency():
    protocol, plan, observations = _inputs()
    shared = _plan_with_shared_physical_request(plan)
    first = observations[0]
    duplicate = replace(
        observations[1], target_id=first.target_id, physical_request_id=first.physical_request_id,
        physical_request_provenance=first.physical_request_provenance, predicted_label=first.predicted_label,
        probabilities=first.probabilities, usage=first.usage, latency_ms=first.latency_ms,
        attempt_count=first.attempt_count, confidence=first.confidence,
    )

    summary = summarize_pilot_compatibility(protocol, shared, (first, duplicate, *observations[2:]))

    assert summary["physical"]["requests"] == shared.physical_count
    assert summary["physical"]["attempts"] == 2 * shared.physical_count
    with pytest.raises(ValueError, match="shared physical"):
        summarize_pilot_compatibility(protocol, shared, (first, replace(duplicate, latency_ms=99), *observations[2:]))


def test_a_pilot_compatibility_summary_marks_optional_fields_unavailable_and_rejects_incomplete_distributions():
    protocol, plan, observations = _inputs()
    unavailable = tuple(replace(row, probabilities=None, confidence=None) for row in observations)

    summary = summarize_pilot_compatibility(protocol, plan, unavailable)

    assert summary["response_fields"] == {
        "explicit_confidence": {"available": False, "coverage": 0, "completed": 21},
        "probability_distribution": {"available": False, "coverage": 0, "completed": 21},
    }
    invalid = replace(observations[0], probabilities={protocol.task.labels[0]: 1.0})
    with pytest.raises(ValueError, match="probability distribution"):
        summarize_pilot_compatibility(protocol, plan, (invalid, *observations[1:]))
