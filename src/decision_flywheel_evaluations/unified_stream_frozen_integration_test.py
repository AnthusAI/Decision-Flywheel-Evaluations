"""Real fake-runtime integration specs for frozen stream serving."""
from types import SimpleNamespace

import pytest

from . import unified_env

if not unified_env.jev_dependencies_available():
    pytest.skip("needs scikit-learn and Tactus (Jev-Flywheel's interpreter)", allow_module_level=True)
try:
    unified_env.ensure_decision_flywheel()
    CLONE = unified_env.put_clone_first()
except unified_env.EnvironmentProblem as problem:
    pytest.skip(f"pinned Jev-Flywheel clone unavailable: {problem}", allow_module_level=True)
unified_env.install_network_guard()

from jev_flywheel.answers import AnswerCache  # noqa: E402
from jev_flywheel.scorecard import Score  # noqa: E402

from .unified_fake_jev import FakeJevCore, FakeResponse  # noqa: E402
from .unified_loop import RunConfig, UnifiedFlywheel  # noqa: E402
from .unified_spend import CircuitOpen  # noqa: E402
from .unified_stream import StreamConfig, StreamDriver, UnifiedFlywheelStreamArm  # noqa: E402


def _engine(tmp_path, *, core=None, max_new=1):
    core = core or FakeJevCore()
    flywheel = UnifiedFlywheel(RunConfig(clone=CLONE.path, run_dir=tmp_path / "run", rounds=1,
                                         per_round=1, dev_size=5, bootstrap_resamples=1,
                                         max_new_requests=max_new), fake_core=core)
    flywheel.cache = AnswerCache(tmp_path / "stream-cache.jsonl")
    engine = UnifiedFlywheelStreamArm.create(flywheel, "L")
    return engine, flywheel.splits.pool[0], core


def test_predict_serves_the_published_reloaded_bundle_not_the_mutable_head(tmp_path):
    engine, item_id, _core = _engine(tmp_path)
    assert engine.predict(item_id)[2] == "ok"
    frozen = engine._frozen_service
    expected = frozen.predict(item_id)
    # The publication remains meaningful even after a live arm mutates.  The
    # next arm prediction may publish a new semantic snapshot, but the already
    # published object cannot silently read this mutable state.
    engine.state.head = object()
    assert frozen.predict(item_id) == expected


def test_predict_returns_the_frozen_rubrics_calibrated_distribution(tmp_path):
    engine, item_id, _core = _engine(tmp_path)
    config = engine.state.head.to_config()
    config["decision"]["calibration"] = {
        "method": "temperature", "raw_confidence": [0.0, 1.0],
        "calibrated_confidence": [0.0, 0.5],
    }
    engine.state.head = Score.from_config(config)
    label, probabilities, status = engine.predict(item_id)
    decision = engine._frozen_service.classify(item_id)
    assert status == "ok" and (label, probabilities) == (decision.label, decision.probabilities)
    assert probabilities[label] == pytest.approx(decision.confidence)


def test_protected_heldout_items_are_rejected_before_prediction_or_feedback_mutates(tmp_path):
    engine, _item_id, _core = _engine(tmp_path)
    heldout = engine.flywheel.splits.paper600[0]
    with pytest.raises(ValueError, match="pool-only"):
        engine.predict(heldout)
    before = tuple(engine.state.workspace.feedback())
    with pytest.raises(ValueError, match="pool-only"):
        engine.record_label(heldout, engine.labels[0], propensity=1.0, explanation_mode="labels")
    assert tuple(engine.state.workspace.feedback()) == before


def test_a_cache_miss_is_one_all_question_request_through_the_frozen_arm(tmp_path):
    class Recording(FakeJevCore):
        def __init__(self):
            super().__init__()
            self.question_sets = []

        def answer(self, state, questions):
            self.question_sets.append(tuple(questions))
            return super().answer(state, questions)

    engine, item_id, core = _engine(tmp_path, core=Recording())
    assert engine.predict(item_id)[2] == "ok"
    assert core.calls == 1
    assert set(core.question_sets[0]) == set(engine._frozen_service.bundle.rubric.questions())


def test_feedback_without_a_semantic_change_reuses_the_bundle_and_driver_records_its_hash(tmp_path):
    engine, first, _core = _engine(tmp_path, max_new=2)
    second = next(item_id for item_id in engine.flywheel.splits.pool if item_id != first)
    stream = [(first, engine.flywheel.labels[first]), (second, engine.flywheel.labels[second])]
    result = StreamDriver(StreamConfig(seed=1, review_probability=1.0, batch_size=1,
                                       checkpoint_at=99)).run(
        "L", engine, stream, selection_plan=[(item_id, True, 1.0) for item_id, _ in stream])
    first_hash, second_hash = (row.bundle_hash for row in result.served)
    assert first_hash == second_hash == engine.served_bundle_hash
    assert engine._frozen_service is not None and len(engine.reviewed) == 2


def test_recording_feedback_alone_does_not_replace_the_frozen_service_object(tmp_path):
    engine, first, _core = _engine(tmp_path, max_new=2)
    second = next(item_id for item_id in engine.flywheel.splits.pool if item_id != first)
    assert engine.predict(first)[2] == "ok"
    published = engine._frozen_service
    engine.record_label(first, engine.flywheel.labels[first], propensity=1.0, explanation_mode="labels")
    assert engine._frozen_service is published
    assert engine.predict(second)[2] == "ok"
    assert engine._frozen_service is published


def test_list_context_is_not_committed_when_cached_answers_are_missing_despite_a_successful_fill(tmp_path, monkeypatch):
    engine, item_id, _core = _engine(tmp_path)
    incumbent, candidate = SimpleNamespace(fingerprint="old"), SimpleNamespace(fingerprint="new")
    head, record = engine.state.head, {"winner": "candidate", "promoted": True}
    engine.state.example_list = incumbent
    engine.state.list_record = {"winner": "incumbent"}
    monkeypatch.setattr(engine, "list_ready", lambda _reviewed: True)
    monkeypatch.setattr(engine.flywheel, "_improve_list", lambda *_args: (candidate, record))
    monkeypatch.setattr(engine.flywheel, "_list_fill", lambda *_args: {"failures": 0})
    monkeypatch.setattr(engine.flywheel.list_answers, "answers", lambda ids, *_args: {i: {} for i in ids})
    assert engine.optimize_list((item_id,)) is False
    assert engine.state.example_list is incumbent
    assert engine.state.head is head
    assert engine.state.list_record == {"winner": "incumbent"}


def test_list_promotion_commits_matching_context_head_and_fit_provenance_atomically(tmp_path, monkeypatch):
    import jev_flywheel.fit as fit
    from . import unified_loop
    from . import unified_stream as stream_module

    engine, item_id, _core = _engine(tmp_path)
    incumbent, candidate = SimpleNamespace(fingerprint="old"), SimpleNamespace(fingerprint="new")
    new_head = object()
    engine.state.example_list = incumbent
    engine.state.list_record = {"winner": "incumbent"}
    monkeypatch.setattr(engine, "list_ready", lambda _reviewed: True)
    monkeypatch.setattr(engine.flywheel, "_improve_list", lambda *_args: (candidate, {"winner": "candidate", "promoted": True}))
    monkeypatch.setattr(engine.flywheel, "_list_fill", lambda *_args: {"failures": 0})
    monkeypatch.setattr(engine.flywheel.list_answers, "answers", lambda ids, *_args: {i: {"wire": "ok"} for i in ids})
    monkeypatch.setattr(engine.flywheel, "_list_rows", lambda *_args: {item_id: {"f": 1.0}})
    monkeypatch.setattr(stream_module, "_answers_complete", lambda *_args: True)
    monkeypatch.setattr(stream_module, "_feature_row_complete", lambda *_args: True)
    training = SimpleNamespace(needs_answers=[], item_ids=[item_id], weights=[1.0])
    fitted = SimpleNamespace(fitted=True, head={}, calibration={}, provenance={"fit_id": "fit-1"})
    monkeypatch.setattr(unified_loop, "training_set", lambda *_args: training)
    monkeypatch.setattr(unified_loop, "served_summary", lambda *_args: object())
    monkeypatch.setattr(unified_loop, "head_from_fit", lambda *_args: new_head)
    monkeypatch.setattr(fit, "fit_head", lambda *_args, **_kwargs: fitted)
    monkeypatch.setattr(fit, "compare", lambda *_args, **_kwargs: SimpleNamespace(promote=True))
    assert engine.optimize_list((item_id,)) is True
    assert engine.state.example_list is candidate
    assert engine.state.head is new_head
    assert engine.state.list_record["head_fit_id"] == "fit-1"


def test_zero_shot_refit_keeps_the_head_when_nominal_fill_leaves_cached_answers_missing(tmp_path, monkeypatch):
    engine, item_id, _core = _engine(tmp_path)
    original = engine.state.head
    monkeypatch.setattr(engine.flywheel, "_fill_zero_shot", lambda *_args: {"failures": 0})
    monkeypatch.setattr(engine.flywheel.cache, "bulk_partial_answers", lambda ids, *_args: {i: {} for i in ids})
    monkeypatch.setattr(engine.flywheel, "_gate", lambda *_args: pytest.fail("partial answers must not reach _gate"))
    engine.refit((item_id,))
    assert engine.state.head is original


def test_cap_exhaustion_is_unavailable_but_a_tripped_circuit_propagates(tmp_path):
    capped, capped_id, capped_core = _engine(tmp_path / "cap", max_new=0)
    assert capped.predict(capped_id) == (None, {}, "unavailable")
    assert capped_core.calls == 0

    class Forbidden(RuntimeError):
        status = 403

    class Rejecting(FakeJevCore):
        def answer(self, state, questions):
            self.calls += 1
            raise Forbidden("refused")

    broken, broken_id, _core = _engine(tmp_path / "circuit", core=Rejecting())
    with pytest.raises(CircuitOpen):
        broken.predict(broken_id)


def test_checkpoint_heldout_path_does_not_hide_an_already_tripped_circuit(tmp_path):
    engine, _item_id, _core = _engine(tmp_path)
    reference = engine.freeze("checkpoint")

    class Forbidden(RuntimeError):
        status = 403

    engine.flywheel.ledger.failed(Forbidden("refused"))
    with pytest.raises(CircuitOpen):
        engine.heldout_from_bundle(reference)
