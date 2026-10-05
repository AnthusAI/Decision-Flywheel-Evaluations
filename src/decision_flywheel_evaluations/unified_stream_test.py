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


def test_each_served_row_records_only_a_valid_frozen_bundle_hash():
    arm = FakeArm()
    arm.served_bundle_hash = "a" * 64
    result = StreamDriver(StreamConfig(review_probability=0)).run("B", arm, [("i1", "a")])
    assert result.to_dict()["served"][0]["bundle_hash"] == "a" * 64
    arm.served_bundle_hash = "private policy or secret"
    result = StreamDriver(StreamConfig(review_probability=0)).run("B", arm, [("i1", "a")])
    assert result.to_dict()["served"][0]["bundle_hash"] is None


def test_uniform_sampling_is_seeded_and_shared_without_using_predictions():
    items = [(f"i{i}", "a") for i in range(20)]
    config = StreamConfig(seed=7, review_probability=.3, batch_size=10)
    one = StreamDriver(config).sample_plan(items)
    two = StreamDriver(config).sample_plan(items)
    assert one == two
    assert {propensity for _, selected, propensity in one} == {.3}


def test_default_batches_mix_one_through_ten_items_reproducibly_without_changing_review_sampling():
    items = [(f"i{i}", "a") for i in range(80)]

    def batches(seed):
        result = StreamDriver(StreamConfig(seed=seed, review_probability=0)).run("B", FakeArm(), items)
        return [row.batch_index for row in result.served]

    one = batches(4)
    assert one == batches(4)
    assert one != batches(5)
    sizes = [one.count(batch) for batch in sorted(set(one))]
    assert all(1 <= size <= 10 for size in sizes)
    assert len(set(sizes)) > 1
    random_driver = StreamDriver(StreamConfig(seed=4))
    fixed_driver = StreamDriver(StreamConfig(seed=4, batch_size=10))
    assert random_driver.sample_plan(items) == fixed_driver.sample_plan(items)


def test_a_fixed_batch_size_is_a_positive_integer_not_a_boolean_or_float():
    import pytest
    for value in (True, False, 1.0, 0, -1, 11):
        with pytest.raises(ValueError):
            StreamConfig(batch_size=value)


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
        unified_env.install_network_guard()
        unified_env.ensure_decision_flywheel()
        clone = unified_env.put_clone_first()
    except unified_env.EnvironmentProblem as error:
        import pytest
        pytest.skip(f"pinned fake runtime unavailable: {type(error).__name__}")
    from decision_flywheel_evaluations.unified_loop import RunConfig, UnifiedFlywheel
    from decision_flywheel_evaluations.unified_stream import UnifiedFlywheelStreamArm
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


def test_concrete_partial_answers_are_unavailable_but_trusted_feedback_is_kept(tmp_path):
    """A cache hole is never converted to zero-valued features or a guessed class."""
    from decision_flywheel_evaluations import unified_env
    if not unified_env.jev_dependencies_available():
        import pytest
        pytest.skip("needs the pinned Jev-Flywheel interpreter")
    try:
        unified_env.install_network_guard()
        unified_env.ensure_decision_flywheel()
        clone = unified_env.put_clone_first()
    except unified_env.EnvironmentProblem as error:
        import pytest
        pytest.skip(f"pinned fake runtime unavailable: {type(error).__name__}")
    from decision_flywheel_evaluations.unified_loop import RunConfig, UnifiedFlywheel
    from decision_flywheel_evaluations.unified_stream import UnifiedFlywheelStreamArm
    from .unified_fake_jev import FakeJevCore, FakeResponse
    from decision_flywheel.context import FixedExampleList
    from jev_flywheel.answers import AnswerCache

    class MissingAnswers(FakeJevCore):
        def answer(self, state, questions):
            self.calls += 1
            return FakeResponse({})

    for with_list in (False, True):
        flywheel = UnifiedFlywheel(RunConfig(clone=clone.path, run_dir=tmp_path / str(with_list), rounds=1,
                                               per_round=1, dev_size=5, bootstrap_resamples=1),
                                   fake_core=MissingAnswers())
        flywheel.cache = AnswerCache(tmp_path / f"empty-cache-{with_list}.jsonl")
        item_id, gold = flywheel.splits.pool[0], flywheel.labels[flywheel.splits.pool[0]]
        engine = UnifiedFlywheelStreamArm.create(flywheel, "L")
        if with_list:
            by_label = {label: [i for i in flywheel.splits.pool
                                if i != item_id and flywheel.labels[i] == label]
                        for label in flywheel.corpus.labels}
            examples = [ids[0] for ids in by_label.values()]
            reserves = [ids[1] for ids in by_label.values()]
            engine.state.example_list = FixedExampleList.from_items(
                flywheel.task, flywheel._labeled_rows(examples), flywheel._labeled_rows(reserves))
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
        unified_env.install_network_guard()
        unified_env.ensure_decision_flywheel()
        clone = unified_env.put_clone_first()
    except unified_env.EnvironmentProblem as error:
        import pytest
        pytest.skip(f"pinned fake runtime unavailable: {type(error).__name__}")
    from decision_flywheel_evaluations.unified_loop import RunConfig, UnifiedFlywheel
    from decision_flywheel_evaluations.unified_stream import UnifiedFlywheelStreamArm
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


def test_stream_steering_uses_a_rolling_view_and_a_sequential_round_number(monkeypatch):
    """A late stream steering round never passes its full history to legacy `_steer`."""
    from types import SimpleNamespace
    from . import unified_stream_workspace
    from .unified_stream import UnifiedFlywheelStreamArm

    class Workspace:
        def scorecard(self):
            return SimpleNamespace(score=lambda _name: "authoritative-head")

    authoritative = Workspace()
    state = SimpleNamespace(workspace=authoritative, head=None, example_list=None)
    seen = {"rounds": [], "windows": [], "cap_workspace": []}
    view = object()
    flywheel = SimpleNamespace(
        corpus=SimpleNamespace(labels=("positive", "negative"), score_name="Sentiment"),
        _steer=lambda arm, received_state, round_number: (
            seen["rounds"].append((arm, received_state.workspace, round_number)) or {"ok": True}),
        _cap_to_budget=lambda arm, received_state, labeled, out: (
            seen["cap_workspace"].append(received_state.workspace), seen["windows"].append(tuple(labeled))),
    )
    monkeypatch.setattr(unified_stream_workspace, "bounded_steering_workspace",
                        lambda workspace, reviewed, *, score_name: view)
    engine = UnifiedFlywheelStreamArm(flywheel, state, "L")
    engine.reviewed = [f"i{index}" for index in range(200)]
    engine._round = 900  # arrival count must not become an analyst steering round.

    assert engine.steer() is True
    assert seen["rounds"] == [("L", view, 1)]
    assert seen["windows"] == [tuple(f"i{index}" for index in range(50, 200))]
    assert seen["cap_workspace"] == [authoritative]
    assert state.workspace is authoritative and state.head == "authoritative-head"


def test_a_failed_stream_steering_round_restores_the_authoritative_workspace(monkeypatch):
    from types import SimpleNamespace
    import pytest
    from . import unified_stream_workspace
    from .unified_stream import UnifiedFlywheelStreamArm

    authoritative = SimpleNamespace(scorecard=lambda: None)
    state = SimpleNamespace(workspace=authoritative, head=None)
    flywheel = SimpleNamespace(
        corpus=SimpleNamespace(labels=("positive",), score_name="Sentiment"),
        _steer=lambda *_args: (_ for _ in ()).throw(RuntimeError("expected steering failure")),
    )
    monkeypatch.setattr(unified_stream_workspace, "bounded_steering_workspace",
                        lambda *_args, **_kwargs: object())
    engine = UnifiedFlywheelStreamArm(flywheel, state, "L")
    engine.reviewed = ["i1"]

    with pytest.raises(RuntimeError, match="expected steering failure"):
        engine.steer()
    assert state.workspace is authoritative


def test_fixed_list_steering_keeps_the_served_head_when_its_refit_does_not_promote(monkeypatch):
    """A zero-shot workspace fit cannot silently replace a fixed-list served head."""
    from types import SimpleNamespace
    from . import unified_stream_workspace
    from .unified_stream import UnifiedFlywheelStreamArm

    authoritative = SimpleNamespace(scorecard=lambda: SimpleNamespace(score=lambda _name: "zero-shot-head"))
    state = SimpleNamespace(workspace=authoritative, head="fixed-list-head", example_list=object())
    flywheel = SimpleNamespace(
        corpus=SimpleNamespace(labels=("positive",), score_name="Sentiment"),
        _steer=lambda *_args: {"promoted": True},
        _cap_to_budget=lambda *_args: None,
    )
    monkeypatch.setattr(unified_stream_workspace, "bounded_steering_workspace",
                        lambda *_args, **_kwargs: object())
    engine = UnifiedFlywheelStreamArm(flywheel, state, "L")
    engine.reviewed = [f"i{index}" for index in range(151)]
    calls = []
    monkeypatch.setattr(engine, "_refit_fixed_list", lambda ids: calls.append(tuple(ids)))

    assert engine.steer() is True
    assert state.head == "fixed-list-head"
    assert calls == [tuple(f"i{index}" for index in range(1, 151))]


def test_fixed_list_steering_serves_the_head_only_when_its_refit_promotes(monkeypatch):
    from types import SimpleNamespace
    from . import unified_stream_workspace
    from .unified_stream import UnifiedFlywheelStreamArm

    authoritative = SimpleNamespace(scorecard=lambda: SimpleNamespace(score=lambda _name: "zero-shot-head"))
    state = SimpleNamespace(workspace=authoritative, head="fixed-list-head", example_list=object())
    flywheel = SimpleNamespace(
        corpus=SimpleNamespace(labels=("positive",), score_name="Sentiment"),
        _steer=lambda *_args: {"promoted": True},
        _cap_to_budget=lambda *_args: None,
    )
    monkeypatch.setattr(unified_stream_workspace, "bounded_steering_workspace",
                        lambda *_args, **_kwargs: object())
    engine = UnifiedFlywheelStreamArm(flywheel, state, "L")
    engine.reviewed = ["i1"]
    monkeypatch.setattr(engine, "_refit_fixed_list", lambda _ids: setattr(state, "head", "promoted-list-head"))

    assert engine.steer() is True
    assert state.head == "promoted-list-head"


def test_real_stream_steering_after_160_reviews_uses_the_bounded_workspace(tmp_path):
    """The real offline scripted procedure receives 150 records after a long stream."""
    from decision_flywheel_evaluations import unified_env
    if not unified_env.jev_dependencies_available():
        import pytest
        pytest.skip("needs the pinned Jev-Flywheel interpreter")
    try:
        unified_env.ensure_decision_flywheel()
        clone = unified_env.put_clone_first()
        unified_env.install_network_guard()
    except unified_env.EnvironmentProblem as error:
        import pytest
        pytest.skip(f"pinned fake runtime unavailable: {error}")
    from jev_flywheel.items import FeedbackItem, LABEL_SOURCE_FINAL
    from .unified_loop import RunConfig, UnifiedFlywheel
    from .unified_stream import UnifiedFlywheelStreamArm
    flywheel = UnifiedFlywheel(RunConfig(clone=clone.path, run_dir=tmp_path / "late-steer", rounds=1,
                                           per_round=1, dev_size=5, bootstrap_resamples=1))
    engine = UnifiedFlywheelStreamArm.create(flywheel, "L")
    reviewed = list(flywheel.splits.pool[:160])
    for index, item_id in enumerate(reviewed):
        engine.state.workspace.add_feedback(FeedbackItem(
            id=f"stream-review-{index}", item_id=item_id, score_name=flywheel.corpus.score_name,
            initial_answer_value="positive", final_answer_value=flywheel.labels[item_id],
            label_source=LABEL_SOURCE_FINAL,
            metadata={"propensity": .3, "selection_policy": "stream-uniform-v1"}))
    engine.reviewed = reviewed
    actual_steer, seen = flywheel._steer, []

    def observe(arm, state, steering_round):
        seen.append((len(state.workspace.items), len(state.workspace.feedback()), steering_round,
                     {record.metadata["propensity"] for record in state.workspace.feedback()}))
        return actual_steer(arm, state, steering_round)

    flywheel._steer = observe
    assert engine.steer() is True
    assert seen == [(150, 150, 1, {.3})]
    assert engine.state.workspace.feedback()[-1].item_id == reviewed[-1]
    assert engine._steering_round == 1
