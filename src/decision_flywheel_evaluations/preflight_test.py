import asyncio
from dataclasses import replace

import pytest

from decision_flywheel.context import RandomBalanced
from decision_flywheel.models import DecisionResult, DecisionTask, Item, LabeledItem
from decision_flywheel.optimizer import TrialSpec, search_context_policies

from .datasets import AG_NEWS, EMOTION, DatasetRow
from .cache import ApprovedRequest, CacheStore
from .manifests import (ExposureStatus, HISTORICAL_AG_NEWS_COMMIT, PreparationMetadata,
                        prepare_official_split)
from .preflight import _selected_global_anchor, candidate_pool_fingerprint, manifest_sha256, preflight
from .protocol import (Capability, FrozenProtocol, ModelIdentity, OptimizationSpec,
                       SelectedGlobalArtifact, Selector, transport_config_fingerprint)
from .study import Engine


def _rows():
    """A real-shaped, entirely local AG News fixture: 64 candidates/class plus dev/test."""
    train = tuple(
        DatasetRow(f"train-{label_index}-{index}", "train", label_index * 100_000 + index, label,
                   f"{label} candidate {index} — NFKC Café")
        for label_index, label in enumerate(AG_NEWS.labels)
        for index in range(65)
    )
    test = tuple(
        DatasetRow(f"test-{label_index}-0", "test", label_index * 100_000, label, f"{label} scoreboard")
        for label_index, label in enumerate(AG_NEWS.labels)
    )
    manifest = prepare_official_split(train, test, AG_NEWS, seed=2, development_per_label=1,
                                      scoreboard_per_label=1,
                                      exposure_status=ExposureStatus.CONFIRMATORY_FRESH)
    return manifest, train + test


def _task():
    return DecisionTask("ag-news", AG_NEWS.labels, "Classify the target text into exactly one topic.")


def test_two_explicit_jev_transports_produce_disjoint_physical_cache_requests():
    manifest, rows = _rows()
    protocol = _protocol(manifest, rows, artifact=False)
    first_model = replace(protocol.models[0], transport_fingerprint=transport_config_fingerprint(
        base_url="https://one.example/v1", timeout_seconds=10.0, retries=0))
    second_model = replace(protocol.models[0], transport_fingerprint=transport_config_fingerprint(
        base_url="https://two.example/v1", timeout_seconds=10.0, retries=0))
    first = preflight(replace(protocol, models=(first_model, protocol.models[1])), manifest=manifest, rows=rows,
                      stage="optimization")
    second = preflight(replace(protocol, models=(second_model, protocol.models[1])), manifest=manifest, rows=rows,
                       stage="optimization")
    first_requests = {cell.fingerprint for cell in first.cells if cell.model == first_model.semantic_identity}
    second_requests = {cell.fingerprint for cell in second.cells if cell.model == second_model.semantic_identity}

    assert first_requests
    assert first_requests.isdisjoint(second_requests)


def test_a_manifest_fingerprint_commits_preparation_configuration_and_historical_inventory_digest():
    manifest, _ = _rows()
    preparation = PreparationMetadata(
        {"candidate_per_label": 64, "development_per_label": 1, "scoreboard_per_label": 1,
         "natural_official_scoreboard": False, "ladder_per_label": 64,
         "historical_source_commit": HISTORICAL_AG_NEWS_COMMIT},
        {"candidate": 256, "development": 4, "scoreboard": 4, "official_history_excluded": 0,
         "dedup_excluded": 0, "leakage_excluded": 0}, "a" * 64,
    )
    configured = replace(manifest, preparation=preparation)
    changed_configuration = replace(configured, preparation=replace(
        preparation, configuration={**preparation.configuration, "candidate_per_label": 63}))
    changed_history = replace(configured, preparation=replace(preparation, exposure_history_fingerprint="b" * 64))

    assert manifest_sha256(configured) != manifest_sha256(changed_configuration)
    assert manifest_sha256(configured) != manifest_sha256(changed_history)


def _artifact(manifest, rows, task, *, objective="accuracy"):
    by_id = {row.id: row for row in rows}
    candidates = tuple(LabeledItem(Item(record.id, {task.input_field: by_id[record.id].text}), record.label, "trusted")
                       for record in manifest.candidate)
    development = tuple(LabeledItem(Item(record.id, {task.input_field: by_id[record.id].text}), record.label, "trusted")
                        for record in manifest.development)
    chosen = tuple(item.item.id for item in RandomBalanced(3).select(
        task, _selected_global_anchor(task, candidates), candidates, per_label=1))
    labels = {row.item.id: row.label for row in development}

    class FakeDevelopmentModel:
        async def decide(self, _task, target, context):
            return DecisionResult(labels[target.id] if tuple(item.item.id for item in context) == chosen else task.labels[0])

    trials = tuple(TrialSpec(RandomBalanced(seed), size)
                   for size in (1, 4, 16, 64) for seed in (0, 1, 2, 3, 4))
    optimization = asyncio.run(search_context_policies(
        task, candidates, development, FakeDevelopmentModel(), trials, max_model_calls=len(trials) * len(development),
        model_fingerprint="jev:jev-1.13.0", objective=objective, display_order="canonical", order_seed=0,
        presentation_label_order=task.labels,
    ))
    assert optimization.winner is not None
    return SelectedGlobalArtifact.from_optimization("development-search", task, candidates, manifest.revision, optimization)


def _rechecksum(artifact):
    return artifact._checksum(artifact.reference, artifact.task_fingerprint, artifact.candidate_pool_fingerprint,
                              artifact.development_split_fingerprint, artifact.search_fingerprint,
                              artifact.objective, artifact.objective_value, artifact.model_fingerprint,
                              artifact.winner_trial_name, artifact.policy_name, artifact.policy_fingerprint,
                                  artifact.selection_seed, artifact.completed_trial_names,
                                  artifact.completed_trials_fingerprint, artifact.completed_trials,
                                  artifact.context_policy_artifact,
                                  artifact.ids_by_size)


def _protocol(manifest, rows, *, artifact=True):
    task = _task()
    selected = _artifact(manifest, rows, task) if artifact else None
    return FrozenProtocol(
        name="ag-news-context-matrix", dataset=AG_NEWS.name,
        dataset_manifest_sha256=manifest_sha256(manifest), candidate_count=256,
        development_count=4, scoreboard_count=4, task=task,
        models=(ModelIdentity(Engine.JEV, "jev-1.13.0", frozenset({Capability.ZERO_SHOT, Capability.FEW_SHOT}), 64),
                ModelIdentity(Engine.LAYA, "laya-local", frozenset({Capability.ZERO_SHOT}), 0)),
        selectors=(Selector.ZERO, Selector.RANDOM, Selector.DEVELOPMENT_SELECTED_GLOBAL,
                   Selector.PROTOTYPE, Selector.RETRIEVAL),
        per_label_sizes=(0, 1, 4, 16, 64), random_draw_seeds=(0, 1, 2, 3, 4),
        display_rule="canonical", selector_configuration_version="core-policy-1",
        selector_search_artifact=selected,
        optimization=OptimizationSpec((Selector.RANDOM,), (1, 4, 16, 64), (0, 1, 2, 3, 4), 80),
        metric="accuracy",
    )


def _emotion_rows():
    train = tuple(
        DatasetRow(f"emotion-train-{label_index}-{index}", "train", label_index * 100_000 + index, label,
                   f"{label} candidate {index} — NFKC Café")
        for label_index, label in enumerate(EMOTION.labels) for index in range(65)
    )
    test = tuple(DatasetRow(f"emotion-test-{label_index}-0", "test", label_index * 100_000, label, f"{label} scoreboard")
                 for label_index, label in enumerate(EMOTION.labels))
    manifest = prepare_official_split(train, test, EMOTION, seed=2, development_per_label=1,
                                      scoreboard_per_label=1, exposure_status=ExposureStatus.HISTORICAL_EXPOSED)
    return manifest, train + test


def test_an_offline_ag_news_preflight_enumerates_the_complete_ready_matrix_without_model_calls(tmp_path):
    manifest, rows = _rows()
    protocol = _protocol(manifest, rows)
    calls = []

    assert manifest.counts == {"candidate": 256, "development": 4, "scoreboard": 4}

    optimization = preflight(protocol, manifest=manifest, rows=rows, stage="optimization",
                             engine=lambda _: calls.append("called"))
    scoreboard = preflight(protocol, manifest=manifest, rows=rows, stage="scoreboard",
                           engine=lambda _: calls.append("called"))

    assert calls == []
    assert optimization.optimization_count == 80
    assert optimization.scoreboard_count == 0
    assert not optimization.ready  # selection is not yet complete at this stage
    assert scoreboard.scoreboard_count == 136
    assert scoreboard.optimization_count == 0
    assert scoreboard.ready
    assert len(scoreboard.cells) == 136
    assert len([cell for cell in scoreboard.cells if cell.model.startswith("jev:")]) == 132
    assert len([cell for cell in scoreboard.cells if cell.model.startswith("laya:")]) == 4
    assert sum(item.target_count for item in optimization.excluded) == 80
    assert sum(item.target_count for item in scoreboard.excluded) == 128
    assert all(item.selector != "zero" for item in scoreboard.excluded)
    approved = {cell.fingerprint: ApprovedRequest(cell.fingerprint, cell.task_fingerprint,
                                                   cell.wire_fingerprint, cell.dataset_revision, cell.model)
                for cell in scoreboard.cells}
    assert len(approved) == scoreboard.new_request_count
    with CacheStore(str(tmp_path / "ledger.sqlite"), task=protocol.task, preflight_fingerprint=scoreboard.checksum,
                    approved_requests=tuple(approved.values()), ceiling=scoreboard.new_request_count):
        pass


def test_request_cells_are_complete_and_reused_contexts_are_logical_not_new_wire_calls():
    manifest, rows = _rows()
    result = preflight(_protocol(manifest, rows), manifest=manifest, rows=rows)
    assert all(cell.wire_fingerprint and cell.task_fingerprint and cell.dataset_revision and cell.display_order
               and cell.counter_identity and cell.estimated_request_tokens >= 0 for cell in result.cells)
    assert all(cell.task_fingerprint == _task().fingerprint and cell.dataset_revision == AG_NEWS.revision
               and cell.display_order == "canonical" for cell in result.cells)
    # The core optimizer chose seed 3, and its selected-global record reuses that exact context.
    duplicate_groups = {}
    for cell in result.cells:
        duplicate_groups.setdefault(cell.fingerprint, []).append(cell)
    assert any({cell.selector for cell in cells} >= {"random", "development_selected_global"}
               and any(cell.draw_seed == 3 for cell in cells if cell.selector == "random")
               for cells in duplicate_groups.values())
    assert any({cell.selector for cell in cells} >= {"random", "prototype", "retrieval",
                                                      "development_selected_global"}
               and {cell.per_label for cell in cells} == {64}
               for cells in duplicate_groups.values())
    assert result.new_request_count == len(duplicate_groups)
    cached = preflight(_protocol(manifest, rows), manifest=manifest, rows=rows,
                       cached_fingerprints=frozenset(duplicate_groups))
    assert cached.cache_hit_count == len(duplicate_groups)
    assert cached.new_request_count == 0
    assert cached.checksum == result.checksum


def test_optimization_needs_no_selected_global_artifact_but_scoreboard_does():
    manifest, rows = _rows()
    protocol = _protocol(manifest, rows, artifact=False)
    optimization = preflight(protocol, manifest=manifest, rows=rows, stage="optimization")
    assert optimization.optimization_count == 80
    assert not optimization.ready
    with pytest.raises(ValueError, match="selected-global artifact"):
        preflight(protocol, manifest=manifest, rows=rows)


def test_preflight_rejects_tampered_candidate_provenance_and_selected_global_membership_before_calls():
    manifest, rows = _rows()
    protocol = _protocol(manifest, rows)
    calls = []
    changed_rows = tuple(replace(row, text="different text") if row.id == manifest.candidate[0].id else row for row in rows)
    with pytest.raises(ValueError, match="provenance"):
        preflight(protocol, manifest=manifest, rows=changed_rows, engine=lambda _: calls.append("called"))
    bad_ids = dict(protocol.selector_search_artifact.ids_by_size)
    bad_ids[1] = ("missing",) + bad_ids[1][1:]
    bad_artifact = replace(protocol.selector_search_artifact, ids_by_size=bad_ids)
    bad_artifact = replace(bad_artifact, checksum=_rechecksum(bad_artifact))
    with pytest.raises(ValueError, match="membership does not match"):
        preflight(replace(protocol, selector_search_artifact=bad_artifact), manifest=manifest, rows=rows,
                  engine=lambda _: calls.append("called"))
    assert calls == []


def test_selected_global_rejects_incomplete_searches_and_tampered_development_provenance():
    manifest, rows = _rows()
    task = _task()
    by_id = {row.id: row for row in rows}
    candidates = tuple(LabeledItem(Item(record.id, {task.input_field: by_id[record.id].text}), record.label, "trusted")
                       for record in manifest.candidate)
    development = tuple(LabeledItem(Item(record.id, {task.input_field: by_id[record.id].text}), record.label, "trusted")
                        for record in manifest.development)
    labels = {row.item.id: row.label for row in development}

    class FakeModel:
        async def decide(self, _task, target, _context): return DecisionResult(labels[target.id])

    incomplete = asyncio.run(search_context_policies(
        task, candidates, development, FakeModel(), (TrialSpec(RandomBalanced(3), 1),),
        max_model_calls=1, model_fingerprint="jev:jev-1.13.0"))
    with pytest.raises(ValueError, match="every declared optimizer trial to complete"):
        SelectedGlobalArtifact.from_optimization("incomplete", task, candidates, manifest.revision, incomplete)

    one_completed = asyncio.run(search_context_policies(
        task, candidates, development, FakeModel(), (TrialSpec(RandomBalanced(3), 1),),
        max_model_calls=len(development), model_fingerprint="jev:jev-1.13.0", objective="accuracy",
        display_order="canonical", order_seed=0, presentation_label_order=task.labels))
    one_trial_artifact = SelectedGlobalArtifact.from_optimization(
        "one-trial", task, candidates, manifest.revision, one_completed)
    with pytest.raises(ValueError, match="does not cover every declared optimizer trial"):
        preflight(replace(_protocol(manifest, rows), selector_search_artifact=one_trial_artifact),
                  manifest=manifest, rows=rows)

    artifact = _artifact(manifest, rows, task)
    tampered = replace(artifact, development_split_fingerprint="0" * 64)
    tampered = replace(tampered, checksum=_rechecksum(tampered))
    with pytest.raises(ValueError, match="development split"):
        preflight(replace(_protocol(manifest, rows), selector_search_artifact=tampered), manifest=manifest, rows=rows)


def test_rechecksummed_trial_integrity_fields_and_an_ineligible_optimizer_model_are_still_rejected():
    manifest, rows = _rows()
    protocol = _protocol(manifest, rows)
    artifact = protocol.selector_search_artifact
    changed_names = replace(artifact, completed_trial_names=("substituted",) + artifact.completed_trial_names[1:])
    changed_names = replace(changed_names, checksum=_rechecksum(changed_names))
    with pytest.raises(ValueError, match="completed trial records"):
        preflight(replace(protocol, selector_search_artifact=changed_names), manifest=manifest, rows=rows)

    changed_fingerprint = replace(artifact, completed_trials_fingerprint="0" * 64)
    changed_fingerprint = replace(changed_fingerprint, checksum=_rechecksum(changed_fingerprint))
    with pytest.raises(ValueError, match="completed trial records"):
        preflight(replace(protocol, selector_search_artifact=changed_fingerprint), manifest=manifest, rows=rows)

    outside_model = replace(artifact, model_fingerprint="other:outside")
    outside_model = replace(outside_model, checksum=_rechecksum(outside_model))
    with pytest.raises(ValueError, match="few-shot eligible"):
        preflight(replace(protocol, selector_search_artifact=outside_model), manifest=manifest, rows=rows)


def test_preflight_reports_missing_zero_shot_capability_instead_of_silently_omitting_it():
    manifest, rows = _rows()
    protocol = _protocol(manifest, rows)
    few_only = ModelIdentity(Engine.LAYA, "few-only", frozenset({Capability.FEW_SHOT}), 64)

    result = preflight(replace(protocol, models=(few_only,), selection_transfer_source="jev:jev-1.13.0"),
                       manifest=manifest, rows=rows)

    assert result.scoreboard_count == 128
    assert result.excluded == tuple(item for item in result.excluded if item.selector == "zero")
    assert result.excluded[0].target_count == 4
    assert result.excluded[0].reason == "zero-shot capability unsupported"


def test_a_complete_emotion_macro_f1_optimizer_artifact_rehydrates_into_scoreboard_preflight():
    manifest, rows = _emotion_rows()
    task = DecisionTask("emotion", EMOTION.labels, "Classify the target text into exactly one emotion.")
    artifact = _artifact(manifest, rows, task, objective="macro-f1")
    protocol = FrozenProtocol(
        name="emotion-context-matrix", dataset=EMOTION.name, dataset_manifest_sha256=manifest_sha256(manifest),
        candidate_count=384, development_count=6, scoreboard_count=6, task=task,
        models=(ModelIdentity(Engine.JEV, "jev-1.13.0", frozenset({Capability.ZERO_SHOT, Capability.FEW_SHOT}), 64),),
        selectors=(Selector.ZERO, Selector.RANDOM, Selector.DEVELOPMENT_SELECTED_GLOBAL,
                   Selector.PROTOTYPE, Selector.RETRIEVAL),
        per_label_sizes=(0, 1, 4, 16, 64), random_draw_seeds=(0, 1, 2, 3, 4),
        display_rule="canonical", selector_configuration_version="core-policy-1",
        selector_search_artifact=artifact,
        optimization=OptimizationSpec((Selector.RANDOM,), (1, 4, 16, 64), (0, 1, 2, 3, 4), 120),
        metric="macro_f1",
    )

    result = preflight(protocol, manifest=manifest, rows=rows)

    assert artifact.objective == "macro-f1"
    assert result.ready and result.scoreboard_count == 198
