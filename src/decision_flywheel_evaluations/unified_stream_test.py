"""Specs for the offline, text-free streaming driver."""
from decision_flywheel_evaluations.unified_stream import StreamConfig, StreamDriver, _answers_complete


class FakeArm:
    labels = ("a", "b")

    def __init__(self):
        self.fit_calls = []
        self.steer_calls = 0

    def predict(self, item_id):
        return ("a", {"a": .8, "b": .2}, "ok")

    def record_label(self, item_id, label, *, propensity, explanation_mode):
        self.last_label = (item_id, label, propensity, explanation_mode)

    def refit(self, reviewed):
        self.fit_calls.append(tuple(reviewed))

    def steering_triggered(self, reviewed):
        return len(reviewed) >= 30

    def steer(self):
        self.steer_calls += 1

    def heldout(self):
        return [("h1", "b", "a", {"a": .8, "b": .2}, "ok")]


def test_answer_completeness_rejects_empty_missing_nan_and_malformed_native_answers():
    questions = {
        "choice": {"question_type": "choice", "criteria": {"a": None, "b": None}},
        "noul": {"question_type": "noul"},
        "score": {"question_type": "score", "criteria": {str(i): None for i in range(6)}},
    }
    valid = {"choice": {"choice": "a", "probabilities": {"a": .7, "b": .3}},
             "noul": {"noul": .2}, "score": {"score": 3, "probabilities": {str(i): (1.0 if i == 3 else 0.0) for i in range(6)}}}
    assert _answers_complete(questions, valid)
    assert not _answers_complete(questions, {})
    assert not _answers_complete(questions, {**valid, "choice": {}})
    assert not _answers_complete(questions, {**valid, "noul": {"noul": float("nan")}})
    assert not _answers_complete(questions, {**valid, "choice": {"choice": "a", "probabilities": {"a": 1}}})
    assert not _answers_complete(questions, {**valid, "score": {"score": 9, "probabilities": {str(i): 0 for i in range(6)}}})
    assert not _answers_complete(questions, {**valid, "score": {"score": 3}})


class FakeListArm(FakeArm):
    def __init__(self):
        super().__init__()
        self.list_calls = []

    def list_ready(self, reviewed):
        return len(reviewed) >= 30

    def optimize_list(self, reviewed):
        self.list_calls.append(tuple(reviewed))
        return True


class FailedFreezeArm(FakeArm):
    def freeze(self, _name):
        return None

    def heldout_from_bundle(self, _bundle):
        raise AssertionError("a failed freeze must not score the mutable head")


class RefusingSteerArm(FakeArm):
    def steer(self):
        self.steer_calls += 1
        return False


def test_a_prediction_is_recorded_before_its_sampled_feedback():
    arm = FakeArm()
    result = StreamDriver(StreamConfig(seed=3, review_probability=1.0, batch_size=1)).run(
        "E", arm, [("i1", "b")])
    served, review = result.served[0], result.reviews[0]
    assert served.predicted_label == "a"
    assert served.reviewed_count_at_prediction == 0
    assert served.review_selected is True
    assert review.label == "b"
    assert arm.last_label == ("i1", "b", 1.0, "explanation")


def test_uniform_sampling_is_seeded_and_shared_without_using_predictions():
    items = [(f"i{i}", "a") for i in range(20)]
    config = StreamConfig(seed=7, review_probability=.3, batch_size=10)
    one = StreamDriver(config).sample_plan(items)
    two = StreamDriver(config).sample_plan(items)
    assert one == two
    assert {propensity for _, selected, propensity in one} == {.3}


def test_cold_start_does_not_refit_before_thirty_reviews():
    arm = FakeArm()
    StreamDriver(StreamConfig(seed=1, review_probability=1, batch_size=10)).run(
        "L", arm, [(f"i{i}", "a") for i in range(29)])
    assert arm.fit_calls == []


def test_refits_every_thirty_reviews_and_steering_is_capped():
    arm = FakeArm()
    result = StreamDriver(StreamConfig(seed=1, review_probability=1, batch_size=10)).run(
        "L", arm, [(f"i{i}", "a") for i in range(130)])
    assert [len(rows) for rows in arm.fit_calls] == [30, 60, 90, 120]
    assert arm.steer_calls == 3
    assert result.counters["refit_rounds"] == 4
    assert result.counters["steering_rounds"] == 3


def test_fixed_list_is_optimized_on_its_own_thirty_review_schedule():
    arm = FakeListArm()
    result = StreamDriver(StreamConfig(seed=1, review_probability=1, batch_size=10)).run(
        "E", arm, [(f"i{i}", "a") for i in range(130)])
    assert [len(rows) for rows in arm.list_calls] == [30, 60, 90, 120]
    assert result.counters["list_rounds"] == 4


def test_a_refused_steering_call_is_not_counted_as_a_round():
    arm = RefusingSteerArm()
    result = StreamDriver(StreamConfig(seed=1, review_probability=1, batch_size=10)).run(
        "L", arm, [(f"i{i}", "a") for i in range(31)])
    assert arm.steer_calls == 2
    assert result.counters["steering_rounds"] == 0


def test_result_is_text_free_and_has_checkpoint_rows():
    arm = FakeArm()
    result = StreamDriver(StreamConfig(seed=1, review_probability=0, batch_size=10, checkpoint_at=3)).run(
        "B", arm, [(f"i{i}", "a") for i in range(4)])
    report = result.to_dict()
    assert report["schema"] == "decision-flywheel-evaluations/stream/v1"
    assert [row["name"] for row in report["checkpoints"]] == ["stream-3", "end"]
    assert "text" not in repr(report).lower()


def test_a_failed_freeze_has_no_heldout_outcomes():
    result = StreamDriver(StreamConfig(review_probability=0, checkpoint_at=1)).run(
        "B", FailedFreezeArm(), [("i1", "a")])
    assert result.checkpoints[0].bundle is None
    assert result.checkpoints[0].heldout == ()


def test_untrusted_engine_fields_never_reach_the_text_free_artifact():
    class Leaky(FakeArm):
        labels = ("a", "b")
        def predict(self, _item_id):
            return "not-a-label", {"api_key": "sk-private-sentinel", "a": float("nan")}, "private explanation"
    report = StreamDriver(StreamConfig(review_probability=0)).run("B", Leaky(), [("i1", "a")]).to_dict()
    row = report["served"][0]
    assert row["predicted_label"] is None and row["probabilities"] == {} and row["status"] == "malformed"
    assert "sk-private-sentinel" not in repr(report)


def test_the_first_shuffled_review_has_no_available_donor_even_with_one_comment():
    from types import SimpleNamespace
    from .unified_stream import UnifiedFlywheelStreamArm

    flywheel = SimpleNamespace(corpus=SimpleNamespace(labels=("a", "b")), cfg=SimpleNamespace(seed=1))
    engine = UnifiedFlywheelStreamArm(flywheel, None, "X", comments={"first": "its own explanation"})
    assert engine.comment_for("first", "shuffled_explanation") is None


def test_concrete_fake_runtime_keeps_heldout_out_of_feedback(tmp_path):
    """The optional expensive smoke spec uses the pinned fake Jev, never a provider."""
    from decision_flywheel_evaluations import unified_env
    if not unified_env.jev_dependencies_available():
        import pytest
        pytest.skip("needs the pinned Jev-Flywheel interpreter")
    try:
        clone = unified_env.put_clone_first()
        unified_env.install_network_guard()
        from decision_flywheel_evaluations.unified_loop import RunConfig, UnifiedFlywheel
        from decision_flywheel_evaluations.unified_stream import UnifiedFlywheelStreamArm
    except Exception as error:  # the local fast interpreter intentionally lacks these extras
        import pytest
        pytest.skip(f"pinned fake runtime unavailable: {type(error).__name__}")
    flywheel = UnifiedFlywheel(RunConfig(clone=clone.path, run_dir=tmp_path / "run", rounds=1,
                                           per_round=1, dev_size=5, bootstrap_resamples=1))
    item_id = flywheel.splits.pool[0]
    comment = {item_id: "should be the trusted label because this is a fake runtime comment"}
    engine = UnifiedFlywheelStreamArm.create(flywheel, "E", comments=comment)
    StreamDriver(StreamConfig(seed=1, review_probability=1, checkpoint_at=999)).run(
        "E", engine, [(item_id, flywheel.labels[item_id])], is_synthetic=True)
    feedback = engine.state.workspace.feedback()
    assert len(feedback) == 1 and feedback[0].edit_comment_value == comment[item_id]
    assert not set(flywheel.splits.paper600) & {row.item_id for row in feedback}


def test_concrete_partial_answers_are_unavailable_but_trusted_feedback_is_kept(tmp_path, monkeypatch):
    """A cache hole is never converted to zero-valued features or a guessed class."""
    from decision_flywheel_evaluations import unified_env
    if not unified_env.jev_dependencies_available():
        import pytest
        pytest.skip("needs the pinned Jev-Flywheel interpreter")
    try:
        clone = unified_env.put_clone_first()
        unified_env.install_network_guard()
        from decision_flywheel_evaluations.unified_loop import RunConfig, UnifiedFlywheel
        from decision_flywheel_evaluations.unified_stream import UnifiedFlywheelStreamArm
    except Exception as error:
        import pytest
        pytest.skip(f"pinned fake runtime unavailable: {type(error).__name__}")
    for with_list in (False, True):
        flywheel = UnifiedFlywheel(RunConfig(clone=clone.path, run_dir=tmp_path / str(with_list), rounds=1,
                                               per_round=1, dev_size=5, bootstrap_resamples=1))
        item_id, gold = flywheel.splits.pool[0], flywheel.labels[flywheel.splits.pool[0]]
        engine = UnifiedFlywheelStreamArm.create(flywheel, "L")
        if with_list:
            engine.state.example_list = object()
            monkeypatch.setattr(flywheel, "_list_fill", lambda *_args, **_kwargs: {"requests": 0, "failures": 0})
            monkeypatch.setattr(flywheel.list_answers, "answers", lambda *_args, **_kwargs: {item_id: {}})
        else:
            monkeypatch.setattr(flywheel, "_fill_zero_shot", lambda *_args, **_kwargs: {"requests": 0, "failures": 0})
            monkeypatch.setattr(flywheel.cache, "bulk_partial_answers", lambda *_args, **_kwargs: {item_id: {}})
        assert engine.predict(item_id) == (None, {}, "unavailable")
        engine.record_label(item_id, gold, propensity=.3, explanation_mode="labels")
        feedback = engine.state.workspace.feedback()[-1]
        assert feedback.initial_answer_value is None and feedback.final_answer_value == gold
        assert feedback.propensity == .3 and engine._plain_mismatches == 0


def test_concrete_fake_runtime_all_arms_freeze_and_account(tmp_path):
    """A shared 31-item review plan exercises the real fake B/L/E/X paths."""
    from decision_flywheel_evaluations import unified_env
    if not unified_env.jev_dependencies_available():
        import pytest
        pytest.skip("needs the pinned Jev-Flywheel interpreter")
    try:
        clone = unified_env.put_clone_first()
        unified_env.install_network_guard()
        from decision_flywheel_evaluations.unified_loop import RunConfig, UnifiedFlywheel
        from decision_flywheel_evaluations.unified_stream import UnifiedFlywheelStreamArm
    except Exception as error:
        import pytest
        pytest.skip(f"pinned fake runtime unavailable: {type(error).__name__}")
    seed_cfg = dict(clone=clone.path, rounds=1, per_round=1, dev_size=5, bootstrap_resamples=1)
    probe = UnifiedFlywheel(RunConfig(run_dir=tmp_path / "probe", **seed_cfg))
    by_label = {label: [i for i in probe.splits.pool if probe.labels[i] == label]
                for label in probe.corpus.labels}
    stream_ids = [item for label in probe.corpus.labels for item in by_label[label][:16]]
    stream_ids = stream_ids[:31]
    stream = [(item, probe.labels[item]) for item in stream_ids]
    selection = [(item, True, 1.0) for item, _ in stream]
    for arm in ("B", "L", "E", "X"):
        flywheel = UnifiedFlywheel(RunConfig(run_dir=tmp_path / arm, **seed_cfg))
        comments = {item: f"comment {index}" for index, (item, _) in enumerate(stream)}
        engine = UnifiedFlywheelStreamArm.create(flywheel, arm, comments=comments, list_per_label=1)
        result = StreamDriver(StreamConfig(seed=7, review_probability=1, batch_size=10,
                                           checkpoint_at=31, list_per_label=1)).run(
            arm, engine, stream, selection_plan=selection)
        checkpoint = result.checkpoints[0]
        assert checkpoint.bundle is not None
        assert {row.item_id for row in checkpoint.heldout} == set(flywheel.splits.paper600)
        assert result.usage["request_attempts"] == engine.usage()["attempts"]
        if arm == "B":
            assert engine.state.workspace is None
        else:
            assert result.counters["refit_rounds"] > 0
            assert result.counters["list_rounds"] > 0
        if arm == "X":
            for item_id in engine.reviewed[1:]:
                assert engine.comment_for(item_id, "shuffled_explanation") != comments[item_id]
