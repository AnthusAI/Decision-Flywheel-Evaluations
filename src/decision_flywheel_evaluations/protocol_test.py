from dataclasses import replace

import pytest

from decision_flywheel.models import DecisionTask

from .protocol import (Capability, FrozenProtocol, ModelIdentity, OptimizationSpec, OrderAnchor, OrderProtocol,
                       OrderTreatment, SelectedGlobalArtifact, Selector, _core_search_fingerprint,
                       transport_config_fingerprint)
from .serialization import read_protocol, write_protocol
from .study import Engine


def _protocol():
    task = DecisionTask("emotion", ("sadness", "joy", "love", "anger", "fear", "surprise"), "Classify only target.text.")
    return FrozenProtocol(
        name="emotion-selector-size", dataset="dair-ai/emotion", dataset_manifest_sha256="a" * 64,
        candidate_count=1_000, development_count=600, scoreboard_count=2_000,
        task=task,
        models=(ModelIdentity(Engine.JEV, "jev-1.13.0", frozenset({Capability.ZERO_SHOT, Capability.FEW_SHOT}), 64),
                ModelIdentity(Engine.LAYA, "laya-local", frozenset({Capability.ZERO_SHOT}), 0)),
        selectors=(Selector.ZERO, Selector.RANDOM, Selector.DEVELOPMENT_SELECTED_GLOBAL, Selector.PROTOTYPE, Selector.RETRIEVAL),
        per_label_sizes=(0, 1, 4, 16, 64), random_draw_seeds=(0, 1, 2, 3, 4),
        display_rule="canonical", selector_configuration_version="core-policy-1",
        selector_search_artifact=None,
        optimization=OptimizationSpec((Selector.RANDOM,), (1, 4, 16, 64), (0, 1, 2, 3, 4), 12_000),
    )


def test_a_frozen_protocol_binds_exact_wording_option_order_and_manifest_hash():
    protocol = _protocol()
    protocol.validate()
    assert protocol.metric == "macro_f1"
    assert protocol.primary_contrasts == ((Selector.RETRIEVAL, Selector.RANDOM, 16), (Selector.RANDOM, Selector.RANDOM, (64, 1)))
    assert protocol.primary_family_correction == "none; descriptive paired 95% intervals"


def test_new_protocols_are_descriptive_while_legacy_holm_protocols_round_trip_unchanged(tmp_path):
    legacy = replace(_protocol(), primary_family_correction="Holm across two primary contrasts")
    path = tmp_path / "legacy-protocol.json"
    write_protocol(path, legacy)

    restored = read_protocol(path)
    assert restored == legacy and restored.identity == legacy.identity
    with pytest.raises(ValueError, match="primary family correction"):
        replace(_protocol(), primary_family_correction="unjustified p-values").validate()


def test_a_protocol_rejects_unanchored_selector_search_or_wrong_dataset_metric():
    with pytest.raises(ValueError, match="typed optimizer artifact"):
        replace(_protocol(), selector_search_artifact="not-an-artifact").validate()  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="macro_f1"):
        _protocol().__class__(**{**_protocol().__dict__, "metric": "accuracy"}).validate()


def test_a_model_without_few_shot_capability_is_explicitly_limited_not_silently_enabled():
    laya = _protocol().models[1]
    assert not laya.supports_few_shot(1)
    assert laya.supports_zero_shot()


def test_a_public_transport_configuration_has_a_stable_secret_free_fingerprint():
    first = transport_config_fingerprint(base_url="https://gateway.example/v1/", timeout_seconds=12.5,
                                         retries=0, adapter_revision="jev-adapter-2")
    repeated = transport_config_fingerprint(base_url="https://gateway.example/v1", timeout_seconds=12.5,
                                           retries=0, adapter_revision="jev-adapter-2")
    changed = transport_config_fingerprint(base_url="https://other-gateway.example/v1", timeout_seconds=12.5,
                                          retries=0, adapter_revision="jev-adapter-2")

    assert first == repeated
    assert first != changed
    with pytest.raises(ValueError) as rejected:
        transport_config_fingerprint(base_url="https://api-key:secret@gateway.example/v1?token=secret",
                                     timeout_seconds=12.5, retries=0)
    assert "secret" not in str(rejected.value)


def test_transport_changes_the_semantic_protocol_and_core_search_identity_but_not_adapter_route():
    left = ModelIdentity(Engine.JEV, "jev-1.13.0", frozenset({Capability.ZERO_SHOT, Capability.FEW_SHOT}), 64,
                         transport_config_fingerprint(base_url="https://one.example/v1", timeout_seconds=10.0, retries=0))
    right = replace(left, transport_fingerprint=transport_config_fingerprint(
        base_url="https://two.example/v1", timeout_seconds=10.0, retries=0))

    assert left.route_identity == right.route_identity == "jev:jev-1.13.0"
    assert left.semantic_identity != right.semantic_identity
    assert replace(_protocol(), models=(left, _protocol().models[1])).identity != replace(
        _protocol(), models=(right, _protocol().models[1])).identity
    assert _core_search_fingerprint(_protocol(), left.semantic_identity, "a" * 64, "b" * 64) != _core_search_fingerprint(
        _protocol(), right.semantic_identity, "a" * 64, "b" * 64)


def test_a_protocol_rejects_changed_task_options_order_or_model_before_preflight_and_ignores_legacy_claims():
    protocol = _protocol()
    with pytest.raises(ValueError, match="label names and order"):
        replace(protocol, task=DecisionTask("emotion", tuple(reversed(protocol.task.labels)), protocol.task.instructions)).validate()
    with pytest.raises(ValueError, match="display rule"):
        replace(protocol, display_rule="provider-default").validate()
    with pytest.raises(ValueError, match="typed engine"):
        replace(protocol, models=(ModelIdentity("jev", "jev-1.13.0", frozenset({Capability.ZERO_SHOT}), 0),)).validate()  # type: ignore[arg-type]
    claimed = replace(protocol, optimization=OptimizationSpec((Selector.RANDOM,), (1, 4, 16, 64),
                                                               (0, 1, 2, 3, 4), 12_000, "d" * 64))
    assert claimed.identity == protocol.identity


def test_an_order_protocol_requires_a_completed_initial_anchor_before_it_can_freeze():
    with pytest.raises(ValueError, match="completed initial"):
        OrderProtocol("order", "a" * 64, _protocol().task, None, ()).validate()
    from .preflight_test import _protocol as fixture_protocol, _rows
    manifest, rows = _rows()
    initial = fixture_protocol(manifest, rows)
    order = OrderProtocol("order", initial.dataset_manifest_sha256, initial.task,
                          OrderAnchor("results/initial.json", initial.selector_search_artifact, initial.dataset_manifest_sha256),
                          (OrderTreatment("canonical", 0), OrderTreatment("interleaved", 0), OrderTreatment("reversed", 0), OrderTreatment("shuffled", 1), OrderTreatment("shuffled", 2)))
    order.validate()
    with pytest.raises(ValueError, match="freeze canonical"):
        replace(order, treatments=(OrderTreatment("interleaved", 0), OrderTreatment("reversed", 0),
                                   OrderTreatment("shuffled", 1), OrderTreatment("shuffled", 2))).validate()


def test_an_order_anchor_rejects_an_untyped_context_and_unsafe_result_reference():
    from .preflight_test import _protocol as fixture_protocol, _rows
    manifest, rows = _rows()
    initial = fixture_protocol(manifest, rows)
    class PretendedArtifact:
        def validate(self, task): pass
    with pytest.raises(ValueError, match="typed"):
        OrderAnchor("results/initial.json", PretendedArtifact(), initial.dataset_manifest_sha256).validate(initial.task)
    for reference in (None, [], "initial", "/tmp/initial.json", "../initial.json", "results/../initial.json",
                      "https://example.test/key.json", "results/initial.json?token=secret"):
        with pytest.raises(ValueError, match="reference"):
            OrderAnchor(reference, initial.selector_search_artifact, initial.dataset_manifest_sha256).validate(initial.task)


@pytest.mark.parametrize("order,seed", ((None, 0), ([], 0), ("canonical", True), ("canonical", 1),
                                       ("interleaved", 2), ("reversed", 3), ("shuffled", 1.5)))
def test_order_treatments_reject_untyped_or_meaningless_seeds(order, seed):
    with pytest.raises(ValueError):
        OrderTreatment(order, seed).validate()


def test_order_labels_distinguish_shuffled_permutations_and_reject_duplicate_treatments():
    assert OrderTreatment("shuffled", 1).label == "shuffled-1"
    assert OrderTreatment("shuffled", 2).label == "shuffled-2"
    assert OrderTreatment("canonical", 0).label == "canonical"
    from .preflight_test import _protocol as fixture_protocol, _rows
    manifest, rows = _rows()
    initial = fixture_protocol(manifest, rows)
    duplicate = OrderTreatment("shuffled", 1)
    order = OrderProtocol("order", initial.dataset_manifest_sha256, initial.task,
                          OrderAnchor("results/initial.json", initial.selector_search_artifact, initial.dataset_manifest_sha256),
                          (OrderTreatment("canonical", 0), OrderTreatment("interleaved", 0),
                           OrderTreatment("reversed", 0), duplicate, duplicate))
    with pytest.raises(ValueError, match="unique"):
        order.validate()
