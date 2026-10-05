"""Specs for the immutable per-item stream-serving seam."""
import asyncio

import pytest

from . import unified_env

if not unified_env.jev_dependencies_available():
    pytest.skip("needs scikit-learn and Tactus (Jev-Flywheel's interpreter)", allow_module_level=True)
try:
    unified_env.ensure_decision_flywheel()
    CLONE = unified_env.put_clone_first()
except unified_env.EnvironmentProblem as problem:
    pytest.skip(f"pinned Jev-Flywheel clone unavailable: {problem}", allow_module_level=True)

# This must precede parsing/building an arm: an accidental real client is a
# test failure rather than a paid request.
unified_env.install_network_guard()

from decision_flywheel.bundle import BundleValidationError  # noqa: E402
from jev_flywheel.answers import AnswerCache  # noqa: E402
from jev_flywheel.scorecard import Score  # noqa: E402

from .unified_fake_jev import FakeJevCore, FakeResponse  # noqa: E402
from .unified_loop import RunConfig, UnifiedFlywheel  # noqa: E402
from .unified_stream import UnifiedFlywheelStreamArm  # noqa: E402
from .unified_stream_serving import FrozenStreamClassifier  # noqa: E402
from .unified_spend import CircuitOpen  # noqa: E402


def _service(tmp_path, *, core=None, max_new=1):
    core = core or FakeJevCore()
    flywheel = UnifiedFlywheel(RunConfig(clone=CLONE.path, run_dir=tmp_path / "run", rounds=1,
                                         per_round=1, dev_size=5, bootstrap_resamples=1,
                                         max_new_requests=max_new), fake_core=core)
    arm = UnifiedFlywheelStreamArm.create(flywheel, "L")
    # An empty independent cache makes the miss/hit contract observable without
    # changing the fake engine or using network data.
    flywheel.cache = AnswerCache(tmp_path / "serve-answers.jsonl")
    service = FrozenStreamClassifier(flywheel, "L", arm.state, ())
    return service, arm, core, flywheel.splits.pool[0]


def test_a_frozen_service_uses_its_reloaded_snapshot_after_live_state_changes(tmp_path):
    service, arm, core, item_id = _service(tmp_path)
    publication = service.publish()
    assert publication.path.name == publication.bundle_hash
    # The mutable stream arm becomes unusable, yet the saved/reloaded score and
    # fixed context still serve the item.  ``labels`` is deliberately removed
    # too: serving must not read a target's trusted label.
    arm.state.head = object()
    service.flywheel.labels = {}
    decision = service.classify(item_id)
    assert decision is not None and decision.label in service.labels
    assert core.calls == 1
    assert service.classify(item_id) == decision
    assert core.calls == 1, "a complete cached answer must send no second request"
    assert service.usage()["attempts"] == 1


def test_the_client_factory_is_created_in_the_request_event_loop_and_cache_hits_stay_free(tmp_path):
    service, _arm, core, item_id = _service(tmp_path)
    factory = service.flywheel.engines.zero_shot_factory

    def loop_bound_factory():
        asyncio.get_running_loop()
        return factory()

    service.client_factory = loop_bound_factory
    service.publish()
    assert service.classify(item_id) is not None and core.calls == 1
    assert service.classify(item_id) is not None and core.calls == 1


def test_serving_preserves_the_reloaded_rubrics_calibrated_decision_distribution(tmp_path):
    service, arm, _core, item_id = _service(tmp_path)
    config = arm.state.head.to_config()
    config["decision"]["calibration"] = {
        "method": "temperature", "raw_confidence": [0.0, 1.0],
        "calibrated_confidence": [0.0, 0.5],
    }
    arm.state.head = Score.from_config(config)
    service.publish()
    served = service.classify(item_id)
    answers = service.flywheel.cache.bulk_partial_answers(
        [item_id], service.bundle.rubric.questions())[item_id]
    expected = service.bundle.rubric.predict(answers)
    assert served == expected
    assert served.probabilities[served.label] == pytest.approx(served.confidence)


def test_a_tampered_content_addressed_artifact_is_refused(tmp_path):
    service, _arm, _core, _item_id = _service(tmp_path)
    publication = service.publish()
    (publication.path / "scorecard.yaml").write_text("tampered: true\n")
    with pytest.raises(BundleValidationError):
        service.reload()


def test_missing_answers_are_unavailable_not_a_zero_feature_chance_prediction(tmp_path):
    class Missing(FakeJevCore):
        def answer(self, state, questions):
            self.calls += 1
            return FakeResponse({})

    service, _arm, core, item_id = _service(tmp_path, core=Missing(), max_new=1)
    service.publish()
    assert service.predict(item_id) == (None, {}, "unavailable")
    assert core.calls == 1
    assert service.usage()["attempts"] == 1


def test_a_zero_new_request_cap_refuses_before_the_fake_engine_is_called(tmp_path):
    service, _arm, core, item_id = _service(tmp_path, max_new=0)
    service.publish()
    assert service.predict(item_id) == (None, {}, "unavailable")
    assert core.calls == 0
    assert service.usage()["attempts"] == 0


def test_a_provider_failure_that_opens_the_circuit_is_not_silently_unavailable(tmp_path):
    class Forbidden(RuntimeError):
        status = 403

    class Rejecting(FakeJevCore):
        def answer(self, state, questions):
            self.calls += 1
            raise Forbidden("no credentials")

    service, _arm, core, item_id = _service(tmp_path, core=Rejecting(), max_new=1)
    service.publish()
    with pytest.raises(CircuitOpen):
        service.predict(item_id)
    assert core.calls == 1
