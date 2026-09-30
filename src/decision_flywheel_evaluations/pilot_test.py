"""Specifications for the bounded, development-only JEV capability pilot."""
from dataclasses import replace

import pytest

from decision_flywheel.models import DecisionTask

from .datasets import AG_NEWS, DatasetRow
from .manifests import ExposureStatus, prepare_official_split
from . import pilot as pilot_module
from .pilot import PilotPlan, plan_jev_pilot
from .preflight import manifest_sha256, preflight
from .protocol import Capability, FrozenProtocol, ModelIdentity, OptimizationSpec, Selector
from .study import Engine


def _fixture():
    train = tuple(
        DatasetRow(
            f"train-{label_index}-{index}",
            "train",
            label_index * 100_000 + index,
            label,
            f"{label} candidate {index}",
        )
        for label_index, label in enumerate(AG_NEWS.labels)
        for index in range(65)
    )
    test = tuple(
        DatasetRow(
            f"test-{label_index}-0",
            "test",
            label_index * 100_000,
            label,
            f"{label} heldout text",
        )
        for label_index, label in enumerate(AG_NEWS.labels)
    )
    manifest = prepare_official_split(
        train,
        test,
        AG_NEWS,
        seed=2,
        development_per_label=1,
        scoreboard_per_label=1,
        exposure_status=ExposureStatus.CONFIRMATORY_FRESH,
    )
    task = DecisionTask("ag-news", AG_NEWS.labels, "Classify the target text into exactly one topic.")
    protocol = FrozenProtocol(
        name="ag-news-context-matrix",
        dataset=AG_NEWS.name,
        dataset_manifest_sha256=manifest_sha256(manifest),
        candidate_count=256,
        development_count=4,
        scoreboard_count=4,
        task=task,
        models=(
            ModelIdentity(
                Engine.JEV,
                "jev-1.13.0",
                frozenset({Capability.ZERO_SHOT, Capability.FEW_SHOT}),
                64,
            ),
            ModelIdentity(Engine.LAYA, "laya-local", frozenset({Capability.ZERO_SHOT}), 0),
        ),
        selectors=(
            Selector.ZERO,
            Selector.RANDOM,
            Selector.DEVELOPMENT_SELECTED_GLOBAL,
            Selector.PROTOTYPE,
            Selector.RETRIEVAL,
        ),
        per_label_sizes=(0, 1, 4, 16, 64),
        random_draw_seeds=(0, 1, 2, 3, 4),
        display_rule="canonical",
        selector_configuration_version="core-policy-1",
        selector_search_artifact=None,
        optimization=OptimizationSpec(
            (Selector.RANDOM,), (1, 4, 16, 64), (0, 1, 2, 3, 4), 80
        ),
        metric="accuracy",
    )
    rows = train + test
    source = preflight(protocol, manifest=manifest, rows=rows, stage="optimization")
    return protocol, manifest, rows, source


@pytest.fixture(scope="module")
def pilot_plan():
    protocol, manifest, rows, source = _fixture()
    return plan_jev_pilot(protocol, manifest, rows, source)


def _reseal(plan, index, **changes):
    cells = list(plan.cells)
    cells[index] = replace(cells[index], **changes)
    frozen_cells = tuple(cells)
    return replace(
        plan,
        cells=frozen_cells,
        checksum=pilot_module._checksum(
            plan.protocol_identity,
            plan.manifest_sha256,
            plan.source_preflight_checksum,
            frozen_cells,
        ),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    tuple(
        (field, value)
        for field in (
            "id",
            "fingerprint",
            "wire_fingerprint",
            "task_fingerprint",
            "dataset_revision",
            "per_label",
            "estimated_request_tokens",
            "draw_seed",
            "example_ids",
        )
        for value in (None, True, 1.5, [])
    ),
)
def test_static_pilot_validation_turns_malformed_deserialized_cell_fields_into_value_errors(
        pilot_plan, field, value):
    with pytest.raises(ValueError):
        _reseal(pilot_plan, 0, **{field: value}).validate()


def test_static_pilot_validation_rejects_the_target_as_its_own_example(pilot_plan):
    cell = pilot_plan.cells[0]
    examples = (cell.target_id, *cell.example_ids[1:])

    with pytest.raises(ValueError, match="target"):
        _reseal(pilot_plan, 0, example_ids=examples).validate()


def test_static_pilot_validation_requires_the_declared_whitespace_estimator_identity(pilot_plan):
    with pytest.raises(ValueError, match="counter"):
        _reseal(pilot_plan, 0, counter_identity="another-safe-counter").validate()


def test_a_pilot_has_one_largest_development_request_for_each_declared_size_and_seed_plus_zero_shot():
    protocol, manifest, rows, source = _fixture()

    plan = plan_jev_pilot(protocol, manifest, rows, source)

    positive = tuple(cell for cell in plan.cells if cell.per_label > 0)
    zero = tuple(cell for cell in plan.cells if cell.per_label == 0)
    assert len(plan.cells) == 21
    assert {(cell.per_label, cell.draw_seed) for cell in positive} == {
        (size, seed) for size in (1, 4, 16, 64) for seed in range(5)
    }
    assert len(zero) == 1
    assert zero[0].phase == "pilot"
    assert zero[0].selector == "zero"
    assert zero[0].draw_seed is None
    assert zero[0].target_id == min(
        positive,
        key=lambda cell: (-cell.estimated_request_tokens, cell.id),
    ).target_id


def test_a_pilot_uses_the_largest_request_then_stable_cell_id_for_each_development_group():
    protocol, manifest, rows, source = _fixture()

    plan = plan_jev_pilot(protocol, manifest, rows, source)

    for selected in (cell for cell in plan.cells if cell.per_label > 0):
        source_group = [
            cell for cell in source.cells
            if (cell.model, cell.per_label, cell.draw_seed)
            == (selected.model, selected.per_label, selected.draw_seed)
        ]
        expected = min(source_group, key=lambda cell: (-cell.estimated_request_tokens, cell.id))
        assert (selected.target_id, selected.fingerprint, selected.wire_fingerprint,
                selected.example_ids) == (
                    expected.target_id, expected.fingerprint, expected.wire_fingerprint,
                    expected.example_ids,
                )
        assert selected.id == expected.id.replace("optimization:", "pilot:", 1)


def test_a_pilot_is_deterministic_text_free_and_derives_physical_counts_from_logical_cells():
    protocol, manifest, rows, source = _fixture()

    first = plan_jev_pilot(protocol, manifest, rows, source)
    second = plan_jev_pilot(protocol, manifest, rows, source)

    assert first == second
    assert first.checksum == second.checksum
    assert first.physical_count == len({cell.fingerprint for cell in first.cells})
    assert first.physical_count <= len(first.cells)
    assert "heldout text" not in repr(first)
    first.validate()


def test_a_pilot_counts_shared_physical_wires_once_even_when_logical_draws_differ():
    protocol, manifest, rows, source = _fixture()
    plan = plan_jev_pilot(protocol, manifest, rows, source)
    fingerprint_counts = {
        fingerprint: sum(cell.fingerprint == fingerprint for cell in plan.cells)
        for fingerprint in {cell.fingerprint for cell in plan.cells}
    }
    first_index = next(
        index for index, cell in enumerate(plan.cells)
        if cell.per_label > 0 and fingerprint_counts[cell.fingerprint] == 1
    )
    second_index = next(
        index for index, cell in enumerate(plan.cells)
        if (cell.per_label > 0
            and cell.fingerprint != plan.cells[first_index].fingerprint
            and fingerprint_counts[cell.fingerprint] == 1)
    )
    first = plan.cells[first_index]
    shared = replace(
        plan.cells[second_index],
        fingerprint=first.fingerprint,
        wire_fingerprint=first.wire_fingerprint,
        example_ids=first.example_ids,
        estimated_request_tokens=first.estimated_request_tokens,
        counter_identity=first.counter_identity,
    )
    cells = plan.cells[:second_index] + (shared,) + plan.cells[second_index + 1:]
    deduplicated = PilotPlan(
        plan.protocol_identity,
        plan.manifest_sha256,
        plan.source_preflight_checksum,
        cells,
        pilot_module._checksum(
            plan.protocol_identity,
            plan.manifest_sha256,
            plan.source_preflight_checksum,
            cells,
        ),
    )

    deduplicated.validate()
    assert deduplicated.physical_count == plan.physical_count - 1


def test_a_pilot_rejects_a_nonregenerated_or_tampered_optimization_preflight_before_deriving_cells():
    protocol, manifest, rows, source = _fixture()

    with pytest.raises(ValueError, match="exact regenerated optimization preflight"):
        plan_jev_pilot(protocol, manifest, rows, replace(source, checksum="0" * 64))

    plan = plan_jev_pilot(protocol, manifest, rows, source)
    with pytest.raises(ValueError, match="checksum"):
        replace(
            plan,
            cells=(
                replace(plan.cells[0], estimated_request_tokens=plan.cells[0].estimated_request_tokens + 1),
                *plan.cells[1:],
            ),
        ).validate()


def test_a_pilot_never_reads_heldout_source_text_when_rehydrating_optimization_rows():
    protocol, manifest, rows, source = _fixture()
    permitted = tuple(row for row in rows if row.id not in {record.id for record in manifest.scoreboard})

    class PoisonedScoreboardRow:
        id = manifest.scoreboard[0].id

        @property
        def source_split(self):
            raise AssertionError("pilot inspected scoreboard source")

        @property
        def source_index(self):
            raise AssertionError("pilot inspected scoreboard source")

        @property
        def label(self):
            raise AssertionError("pilot inspected scoreboard source")

        @property
        def text(self):
            raise AssertionError("pilot inspected scoreboard source")

    plan = plan_jev_pilot(protocol, manifest, permitted + (PoisonedScoreboardRow(),), source)

    assert len(plan.cells) == 21


@pytest.mark.parametrize(
    "models",
    (
        (ModelIdentity(Engine.JEV, "jev-1.13.0", frozenset({Capability.ZERO_SHOT}), 64),),
        (ModelIdentity(Engine.JEV, "jev-1.13.0", frozenset({Capability.ZERO_SHOT, Capability.FEW_SHOT}), 63),),
    ),
)
def test_a_pilot_requires_one_jev_with_declared_zero_and_64_example_capabilities(models):
    protocol, manifest, rows, source = _fixture()

    with pytest.raises(ValueError, match="JEV"):
        plan_jev_pilot(replace(protocol, models=models), manifest, rows, source)


def test_a_pilot_rejects_multiple_frozen_jev_models():
    protocol, manifest, rows, source = _fixture()
    second = ModelIdentity(
        Engine.JEV,
        "jev-2.0.0",
        frozenset({Capability.ZERO_SHOT, Capability.FEW_SHOT}),
        64,
    )

    with pytest.raises(ValueError, match="exactly one"):
        plan_jev_pilot(replace(protocol, models=(protocol.models[0], second)), manifest, rows, source)


def test_a_pilot_checksum_binds_the_frozen_protocol_manifest_and_source_preflight_identities():
    protocol, manifest, rows, source = _fixture()
    plan = plan_jev_pilot(protocol, manifest, rows, source)

    assert plan.protocol_identity == protocol.identity
    assert plan.manifest_sha256 == manifest_sha256(manifest)
    assert plan.source_preflight_checksum == source.checksum
    with pytest.raises(ValueError, match="checksum"):
        replace(plan, protocol_identity="0" * 64).validate()
