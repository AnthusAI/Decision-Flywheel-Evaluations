"""Specifications for descriptive reporting of a frozen ordering follow-up."""
from __future__ import annotations

from dataclasses import replace

import pytest

from .metrics import Observation
from .ordering import _checksum, plan_ordering
from .preflight import _physical_request_fingerprint
from .ordering_reporting import report_ordering
from .preflight import preflight
from .preflight_test import _protocol, _rows
from .protocol import OrderTreatment


def _inputs():
    manifest, rows = _rows()
    base = _protocol(manifest, rows, artifact=True)
    protocol = replace(base, models=(base.models[0],))
    initial = preflight(protocol, manifest=manifest, rows=rows, stage="scoreboard")
    labels = {record.id: record.label for record in manifest.scoreboard}
    initial_rows = tuple(
        Observation(
            cell.id, cell.target_id, f"scoreboard:{cell.model}:{cell.selector}:{cell.per_label}",
            cell.draw_seed or 0, cell.display_order, labels[cell.target_id],
            labels[cell.target_id], "completed", model_id=cell.model,
            physical_request_id=cell.fingerprint,
            physical_request_provenance={
                "model": cell.model, "dataset_revision": cell.dataset_revision,
                "task_fingerprint": cell.task_fingerprint, "wire_fingerprint": cell.wire_fingerprint,
            },
        )
        for cell in initial.cells
    )
    treatments = (
        OrderTreatment("canonical", 0), OrderTreatment("interleaved", 0),
        OrderTreatment("reversed", 0), OrderTreatment("shuffled", 1),
        OrderTreatment("shuffled", 2),
    )
    plan = plan_ordering(protocol, manifest, rows, initial, initial_rows,
                         "artifacts/initial-result.json", treatments)
    expected = {record.id: record.label for record in manifest.scoreboard}
    observations = tuple(
        Observation(
            cell.request.id, cell.request.target_id,
            f"scoreboard:{cell.request.model}:{cell.request.selector}:{cell.request.per_label}",
            cell.request.draw_seed or 0, cell.request.display_order,
            expected[cell.request.target_id], expected[cell.request.target_id], "completed",
            model_id=cell.request.model, usage={"tokens": 1}, latency_ms=2.0,
            physical_request_id=cell.request.fingerprint,
            physical_request_provenance={
                "model": cell.request.model, "dataset_revision": cell.request.dataset_revision,
                "task_fingerprint": cell.request.task_fingerprint,
                "wire_fingerprint": cell.request.wire_fingerprint,
            },
        )
        for cell in plan.cells
    )
    return protocol, manifest, plan, observations


def test_a_frozen_ordering_plan_reports_exact_cells_physical_totals_and_only_nominal_descriptive_intervals():
    protocol, manifest, plan, observations = _inputs()

    report = report_ordering(protocol, manifest, plan, observations, seed=7, resamples=9)

    assert report["schema"] == "decision-flywheel-evaluations/ordering-report/v1"
    assert report["scope"] == "descriptive ordering follow-up; not a selection or significance result"
    assert report["provenance"] == {
        "initial_protocol_identity": plan.initial_protocol_identity,
        "manifest_sha256": plan.manifest_sha256,
        "initial_preflight_checksum": plan.initial_preflight_checksum,
        "initial_observations_sha256": plan.initial_observations_sha256,
        "initial_result_artifact": plan.initial_result_artifact,
        "ordering_plan_checksum": plan.checksum,
    }
    assert report["study"]["logical_cells"] == plan.logical_count
    assert report["study"]["physical"]["requests"] == plan.physical_count
    assert report["study"]["physical"]["attempts"] == plan.physical_count
    assert report["inference"] == {
        "family_correction": "none; descriptive paired 95% intervals",
        "significance": {"available": False, "reason": "formal significance testing was not requested"},
    }
    assert all(value["available"] for value in report["paired_effects"].values())
    assert all(value["confidence_level"] == 0.95
               and value["interval_scope"] == "nominal per-contrast, not multiplicity adjusted"
               for value in report["paired_effects"].values())
    assert all({"baseline_mean_draw_metric", "treatment_mean_draw_metric", "percentage_effect"} <= set(value)
               for value in report["paired_effects"].values())


def test_an_ordering_report_rejects_tampered_plan_or_observation_provenance_before_metrics():
    protocol, manifest, plan, observations = _inputs()

    with pytest.raises(ValueError, match="checksum"):
        report_ordering(protocol, manifest, replace(plan, checksum="0" * 64), observations, resamples=3)
    with pytest.raises(ValueError, match="protocol"):
        report_ordering(replace(protocol, name="different"), manifest, plan, observations, resamples=3)
    with pytest.raises(ValueError, match="frozen logical"):
        report_ordering(protocol, manifest, plan,
                        (replace(observations[0], condition="ordering:wrong"), *observations[1:]), resamples=3)
    wrong_label = next(label for label in protocol.task.labels if label != observations[0].true_label)
    with pytest.raises(ValueError, match="ground truth"):
        report_ordering(protocol, manifest, plan,
                        (replace(observations[0], true_label=wrong_label), *observations[1:]), resamples=3)
    with pytest.raises(ValueError, match="physical provenance"):
        report_ordering(protocol, manifest, plan,
                        (replace(observations[0], physical_request_provenance={"model": "wrong"}), *observations[1:]), resamples=3)
    with pytest.raises(ValueError, match="typed"):
        report_ordering(protocol, manifest, plan, (*observations, object()), resamples=3)


def test_a_recomputed_ordering_checksum_cannot_bind_foreign_request_metadata_to_the_initial_study():
    protocol, manifest, plan, observations = _inputs()
    request = plan.cells[0].request
    foreign_target = next(record.id for record in manifest.scoreboard if record.id != request.target_id)
    foreign_example = manifest.scoreboard[0].id
    variants = (
        replace(request, task_fingerprint="0" * 64),
        replace(request, model="jev:foreign", fingerprint=_physical_request_fingerprint(
            "jev:foreign", request.dataset_revision, request.wire_fingerprint)),
        replace(request, dataset_revision="1" * 40, fingerprint=_physical_request_fingerprint(
            request.model, "1" * 40, request.wire_fingerprint)),
        replace(request, target_id=foreign_target),
        replace(request, example_ids=(foreign_example, *request.example_ids[1:])),
    )

    for foreign in variants:
        cells = (replace(plan.cells[0], request=foreign), *plan.cells[1:])
        unsigned = replace(plan, cells=cells, checksum="")
        recomputed = replace(unsigned, checksum=_checksum(unsigned))
        label = next(record.label for record in manifest.scoreboard if record.id == foreign.target_id)
        matching = replace(
            observations[0], request_id=foreign.id, target_id=foreign.target_id,
            condition=f"scoreboard:{foreign.model}:{foreign.selector}:{foreign.per_label}",
            draw=foreign.draw_seed or 0, order=foreign.display_order,
            true_label=label, predicted_label=label, model_id=foreign.model,
            physical_request_id=foreign.fingerprint,
            physical_request_provenance={
                "model": foreign.model, "dataset_revision": foreign.dataset_revision,
                "task_fingerprint": foreign.task_fingerprint,
                "wire_fingerprint": foreign.wire_fingerprint,
            },
        )
        with pytest.raises(ValueError):
            report_ordering(protocol, manifest, recomputed,
                            (matching, *observations[1:]), resamples=3)


def test_an_incomplete_ordering_matrix_keeps_declared_order_intervals_unavailable_instead_of_dropping_cells():
    protocol, manifest, plan, observations = _inputs()
    index = next(index for index, cell in enumerate(plan.cells) if cell.request.per_label > 0)
    physical_id = observations[index].physical_request_id
    failed_rows = tuple(
        replace(row, status="failed", predicted_label=None)
        if row.physical_request_id == physical_id else row
        for row in observations
    )

    report = report_ordering(protocol, manifest, plan, failed_rows, resamples=3)

    assert report["study"]["logical_cells"] == plan.logical_count
    assert report["study"]["physical"]["failures"] == 1
    assert any(not value["available"] for value in report["paired_effects"].values())
    assert all("reason" in value for value in report["paired_effects"].values() if not value["available"])
