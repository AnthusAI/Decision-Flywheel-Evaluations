"""Specifications for the post-scoreboard presentation-order follow-up plan."""
from dataclasses import replace

import pytest

from .metrics import Observation
from .ordering import OrderingCell, OrderingPlan, _checksum, ordering_execution, plan_ordering
from .preflight import preflight
from .preflight_test import _protocol, _rows
from .protocol import OrderTreatment
from .datasets import AG_NEWS, DatasetRow
from .manifests import ExposureStatus, prepare_official_split


def _initial_inputs():
    manifest, rows = _rows()
    protocol = _protocol(manifest, rows, artifact=True)
    protocol = replace(protocol, models=(protocol.models[0],))
    initial = preflight(protocol, manifest=manifest, rows=rows, stage="scoreboard")
    labels = {record.id: record.label for record in manifest.scoreboard}
    observations = tuple(
        Observation(
            cell.id,
            cell.target_id,
            f"scoreboard:{cell.model}:{cell.selector}:{cell.per_label}",
            cell.draw_seed or 0,
            cell.display_order,
            labels[cell.target_id],
            labels[cell.target_id],
            "completed",
            model_id=cell.model,
            physical_request_id=cell.fingerprint,
            physical_request_provenance={
                "model": cell.model,
                "dataset_revision": cell.dataset_revision,
                "task_fingerprint": cell.task_fingerprint,
                "wire_fingerprint": cell.wire_fingerprint,
            },
        )
        for cell in initial.cells
    )
    treatments = (
        OrderTreatment("canonical", 0),
        OrderTreatment("interleaved", 0),
        OrderTreatment("reversed", 0),
        OrderTreatment("shuffled", 1),
        OrderTreatment("shuffled", 2),
    )
    return protocol, manifest, rows, initial, observations, treatments


def _two_targets_per_label_inputs():
    train = tuple(
        DatasetRow(f"train-{label_index}-{index}", "train", label_index * 100_000 + index,
                   label, f"{label} candidate {index}")
        for label_index, label in enumerate(AG_NEWS.labels)
        for index in range(65)
    )
    test = tuple(
        DatasetRow(f"test-{label_index}-{index}", "test", label_index * 100_000 + index,
                   label, f"{label} scoreboard {index}")
        for label_index, label in enumerate(AG_NEWS.labels)
        for index in range(2)
    )
    manifest = prepare_official_split(
        train, test, AG_NEWS, seed=2, development_per_label=1, scoreboard_per_label=2,
        exposure_status=ExposureStatus.CONFIRMATORY_FRESH,
    )
    rows = train + test
    base_protocol = _protocol(manifest, rows, artifact=True)
    protocol = replace(base_protocol, models=(base_protocol.models[0],), scoreboard_count=8)
    initial = preflight(protocol, manifest=manifest, rows=rows, stage="scoreboard")
    labels = {record.id: record.label for record in manifest.scoreboard}
    observations = tuple(
        Observation(
            cell.id, cell.target_id, f"scoreboard:{cell.model}:{cell.selector}:{cell.per_label}",
            cell.draw_seed or 0, cell.display_order, labels[cell.target_id], labels[cell.target_id],
            "completed", model_id=cell.model, physical_request_id=cell.fingerprint,
            physical_request_provenance={"model": cell.model, "dataset_revision": cell.dataset_revision,
                                         "task_fingerprint": cell.task_fingerprint,
                                         "wire_fingerprint": cell.wire_fingerprint},
        )
        for cell in initial.cells
    )
    treatments = (OrderTreatment("canonical", 0), OrderTreatment("interleaved", 0),
                  OrderTreatment("reversed", 0), OrderTreatment("shuffled", 1),
                  OrderTreatment("shuffled", 2))
    return protocol, manifest, rows, initial, observations, treatments


def _reseal(plan, *, cells=None):
    changed = replace(plan, cells=tuple(cells if cells is not None else plan.cells), checksum="")
    return replace(changed, checksum=_checksum(changed))


def test_ordering_expands_only_positive_complete_initial_jev_cells_and_retains_one_canonical_zero_shot():
    protocol, manifest, rows, initial, observations, treatments = _initial_inputs()

    plan = plan_ordering(
        protocol,
        manifest,
        rows,
        initial,
        observations,
        "artifacts/initial-result.json",
        treatments,
    )

    assert plan.logical_count == 644
    assert len(plan.cells) == 644
    assert sum(cell.request.per_label == 0 for cell in plan.cells) == manifest.counts["scoreboard"]
    assert {cell.request.display_order for cell in plan.cells} == {
        "canonical", "interleaved", "reversed", "shuffled-1", "shuffled-2"
    }
    assert all(cell.request.id.startswith("ordering:") for cell in plan.cells)
    assert plan.physical_count == len({cell.request.fingerprint for cell in plan.cells})
    source = {cell.id: cell for cell in initial.cells}
    assert all(
        cell.request.fingerprint == source[cell.source_request_id].fingerprint
        for cell in plan.cells
        if cell.treatment.display_order == "canonical"
    )
    plan.validate()


def test_ordering_preserves_initial_membership_and_rebuilds_only_presentation_wires():
    protocol, manifest, rows, initial, observations, treatments = _initial_inputs()
    plan = plan_ordering(protocol, manifest, rows, initial, observations,
                         "artifacts/initial-result.json", treatments)
    source = {cell.id: cell for cell in initial.cells}

    for cell in plan.cells:
        original = source[cell.source_request_id]
        assert cell.request.target_id == original.target_id
        assert cell.request.selector == original.selector
        assert cell.request.per_label == original.per_label
        assert cell.request.draw_seed == original.draw_seed
        assert cell.request.task_fingerprint == original.task_fingerprint
        assert cell.request.dataset_revision == original.dataset_revision
        assert set(cell.request.example_ids) == set(original.example_ids)
        assert (cell.request.wire_fingerprint == original.wire_fingerprint) == (
            cell.request.example_ids == original.example_ids
        )

    execution = ordering_execution(protocol, manifest, rows, plan)
    assert len(execution) == plan.logical_count
    assert all(request.wire_fingerprint == context.token_accounting.serialized_request_fingerprint
               for request, _target, context in execution)
    assert plan == plan_ordering(protocol, manifest, rows, initial, observations,
                                 "artifacts/initial-result.json", treatments)


def test_ordering_requires_exact_complete_initial_scoreboard_evidence_and_safe_artifact_reference():
    protocol, manifest, rows, initial, observations, treatments = _initial_inputs()

    with pytest.raises(ValueError, match="complete initial"):
        plan_ordering(protocol, manifest, rows, initial, observations[:-1],
                      "artifacts/initial-result.json", treatments)
    with pytest.raises(ValueError, match="exact regenerated"):
        plan_ordering(protocol, manifest, rows, replace(initial, checksum="0" * 64), observations,
                      "artifacts/initial-result.json", treatments)
    with pytest.raises(ValueError, match="safe relative"):
        plan_ordering(protocol, manifest, rows, initial, observations,
                      "../initial-result.json", treatments)


def test_ordering_static_validation_rejects_tampered_request_metadata_and_duplicate_treatments():
    protocol, manifest, rows, initial, observations, treatments = _initial_inputs()
    plan = plan_ordering(protocol, manifest, rows, initial, observations,
                         "artifacts/initial-result.json", treatments)
    bad_request = replace(plan.cells[0].request, wire_fingerprint="0" * 64)
    bad_cell = replace(plan.cells[0], request=bad_request)

    with pytest.raises(ValueError):
        replace(plan, cells=(bad_cell, *plan.cells[1:])).validate()
    with pytest.raises(ValueError, match="treatments"):
        plan_ordering(protocol, manifest, rows, initial, observations,
                      "artifacts/initial-result.json", (treatments[0], treatments[0], *treatments[1:]))


def test_ordering_static_validation_requires_every_complete_source_group_even_with_a_recomputed_checksum():
    protocol, manifest, rows, initial, observations, treatments = _initial_inputs()
    plan = plan_ordering(protocol, manifest, rows, initial, observations,
                         "artifacts/initial-result.json", treatments)
    removed_source = plan.cells[0].source_request_id

    with pytest.raises(ValueError, match="complete 33"):
        _reseal(plan, cells=[cell for cell in plan.cells if cell.source_request_id != removed_source]).validate()


def test_ordering_static_validation_derives_source_and_logical_ids_not_just_the_checksum():
    protocol, manifest, rows, initial, observations, treatments = _initial_inputs()
    plan = plan_ordering(protocol, manifest, rows, initial, observations,
                         "artifacts/initial-result.json", treatments)
    arbitrary_source = replace(plan.cells[0], source_request_id="scoreboard:arbitrary")
    duplicate = replace(
        plan.cells[0],
        request=replace(plan.cells[0].request, id=f"{plan.cells[0].request.id}:duplicate"),
    )

    with pytest.raises(ValueError, match="source request ID"):
        _reseal(plan, cells=(arbitrary_source, *plan.cells[1:])).validate()
    with pytest.raises(ValueError, match="logical request ID"):
        _reseal(plan, cells=(*plan.cells, duplicate)).validate()


def test_ordering_accepts_a_cache_aware_saved_initial_preflight_when_its_immutable_matrix_matches():
    protocol, manifest, rows, initial, observations, treatments = _initial_inputs()
    physical = len({cell.fingerprint for cell in initial.cells})
    cache_aware = replace(initial, new_request_count=0, cache_hit_count=physical)

    plan = plan_ordering(protocol, manifest, rows, cache_aware, observations,
                         "artifacts/initial-result.json", treatments)

    assert plan.initial_preflight_checksum == initial.checksum


@pytest.mark.parametrize("stage", ("optimization", "pilot"))
def test_ordering_rejects_non_scoreboard_initial_preflights(stage):
    protocol, manifest, rows, initial, observations, treatments = _initial_inputs()

    with pytest.raises(ValueError, match="complete initial"):
        plan_ordering(protocol, manifest, rows, replace(initial, stage=stage), observations,
                      "artifacts/initial-result.json", treatments)


def test_ordering_rejects_noncanonical_initial_cells_and_mixed_static_task_provenance():
    protocol, manifest, rows, initial, observations, treatments = _initial_inputs()
    noncanonical = replace(initial, cells=(replace(initial.cells[0], display_order="reversed"), *initial.cells[1:]))
    with pytest.raises(ValueError, match="exact regenerated"):
        plan_ordering(protocol, manifest, rows, noncanonical, observations,
                      "artifacts/initial-result.json", treatments)

    plan = plan_ordering(protocol, manifest, rows, initial, observations,
                         "artifacts/initial-result.json", treatments)
    source_id = plan.cells[0].source_request_id
    mixed = tuple(
        replace(cell, request=replace(cell.request, task_fingerprint="0" * 64))
        if cell.source_request_id == source_id else cell
        for cell in plan.cells
    )
    with pytest.raises(ValueError, match="one JEV model, task"):
        _reseal(plan, cells=mixed).validate()


@pytest.mark.parametrize("status", ("failed", "malformed"))
def test_ordering_rejects_any_noncompleted_initial_observation(status):
    protocol, manifest, rows, initial, observations, treatments = _initial_inputs()
    changed = replace(observations[0], status=status, predicted_label=None)

    with pytest.raises(ValueError, match="complete initial"):
        plan_ordering(protocol, manifest, rows, initial, (changed, *observations[1:]),
                      "artifacts/initial-result.json", treatments)


@pytest.mark.parametrize(
    "change",
    (
        lambda row: replace(row, true_label="Sports"),
        lambda row: replace(row, model_id="jev:wrong"),
        lambda row: replace(row, physical_request_provenance={"model": "wrong"}),
    ),
)
def test_ordering_rejects_initial_truth_model_or_physical_provenance_mismatches(change):
    protocol, manifest, rows, initial, observations, treatments = _initial_inputs()

    with pytest.raises(ValueError, match="provenance"):
        plan_ordering(protocol, manifest, rows, initial, (change(observations[0]), *observations[1:]),
                      "artifacts/initial-result.json", treatments)


def test_ordering_execution_rejects_resealed_membership_or_target_tampering():
    protocol, manifest, rows, initial, observations, treatments = _initial_inputs()
    plan = plan_ordering(protocol, manifest, rows, initial, observations,
                         "artifacts/initial-result.json", treatments)
    source_id = next(cell.source_request_id for cell in plan.cells if cell.request.per_label > 0)
    group = [cell for cell in plan.cells if cell.source_request_id == source_id]
    old_id = group[0].request.example_ids[0]
    replacement_id = next(
        candidate for cell in plan.cells if cell.source_request_id != source_id
        for candidate in cell.request.example_ids
        if candidate not in group[0].request.example_ids
    )
    altered = []
    for cell in plan.cells:
        if cell.source_request_id == source_id:
            examples = tuple(replacement_id if value == old_id else value for value in cell.request.example_ids)
            altered.append(replace(cell, request=replace(cell.request, example_ids=examples)))
        else:
            altered.append(cell)
    with pytest.raises(ValueError):
        ordering_execution(protocol, manifest, rows, _reseal(plan, cells=altered))
    bad_target = replace(plan.cells[0], request=replace(plan.cells[0].request, target_id="not-in-scoreboard"))
    with pytest.raises(ValueError):
        ordering_execution(protocol, manifest, rows, _reseal(plan, cells=(bad_target, *plan.cells[1:])))


def test_ordering_supports_multiple_scoreboard_targets_per_label_and_execution_requires_the_full_manifest_set():
    protocol, manifest, rows, initial, observations, treatments = _two_targets_per_label_inputs()
    plan = plan_ordering(protocol, manifest, rows, initial, observations,
                         "artifacts/initial-result.json", treatments)

    assert manifest.counts["scoreboard"] == 8
    assert plan.logical_count == 1_288
    assert {request.target_id for request, _target, _context in ordering_execution(protocol, manifest, rows, plan)} == {
        record.id for record in manifest.scoreboard
    }

    removed_target = manifest.scoreboard[0].id
    truncated = _reseal(plan, cells=[
        cell for cell in plan.cells if cell.request.target_id != removed_target
    ])
    truncated.validate()
    with pytest.raises(ValueError, match="full scoreboard"):
        ordering_execution(protocol, manifest, rows, truncated)
