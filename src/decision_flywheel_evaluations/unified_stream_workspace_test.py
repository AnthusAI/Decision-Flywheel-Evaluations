"""Specs for the rolling steering workspace seam."""
from dataclasses import dataclass, field
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from .unified_stream_workspace import (BoundedSteeringWorkspace,
                                       BoundedWorkspaceError)


@dataclass
class _Item:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def split(self):
        return self.metadata.get("split")


@dataclass
class _Feedback:
    item_id: str
    score_name: str = "sentiment"
    final_answer_value: str = "positive"
    label: str = "positive"
    metadata: dict = field(default_factory=lambda: {"propensity": 0.3})


class _Cache:
    model = "fake"

    def __init__(self):
        self.planned = []

    def rows(self):
        yield "old", "q", "h", {"choice": "negative"}
        yield "i199", "q", "h", {"choice": "positive"}

    def plan(self, ids, questions):
        self.planned.append(tuple(ids))
        return object()

    def get(self, *_): return None
    def missing(self, *_): return {}
    def answers_for(self, *_): return {}
    def partial_answers_for(self, *_): return {}
    def bulk_partial_answers(self, ids, _questions): return {item_id: {} for item_id in ids}


class _Workspace:
    root = "/fake"
    engine = "fake"
    version = 7

    def __init__(self):
        self.cache = _Cache()
        self._items = {f"i{i}": _Item(f"i{i}", f"text {i}") for i in range(200)}
        self._items["old"] = _Item("old", "OLDER EXPLANATION SENTINEL")
        self._feedback = [_Feedback(f"i{i}", metadata={"propensity": .3, "comment": f"comment {i}"})
                          for i in range(200)]
        self.commits = []
        self._events = [
            {"kind": "fit", "score_name": "sentiment", "n_feedback": 40,
             "n_labeled": 40, "fitted": True, "log_loss": .4,
             "analyst_reply": "OLDER EXPLANATION SENTINEL"},
            {"kind": "rethink", "score_name": "sentiment", "n_feedback": 200,
             "n_labeled": 200, "analyst_reply": "future-looking reply"},
            {"kind": "fit", "score_name": "sentiment", "n_feedback": 201,
             "n_labeled": 201, "analyst_reply": "FUTURE SENTINEL"},
        ]

    def item(self, item_id): return self._items[item_id]
    def feedback(self): return list(self._feedback)
    def scorecard(self, version=None): return ("card", version)
    def commit_scorecard(self, card, *, kind, provenance=None):
        self.commits.append((card, kind, provenance)); return 8
    def lineage(self): return []
    def events(self, kind=None): return [event for event in self._events if kind is None or event["kind"] == kind]
    def log_event(self, kind, score_name, **data):
        self._events.append({"kind": kind, "score_name": score_name, "n_feedback": len(self._feedback),
                             "n_labeled": len(self._feedback), **data})
        return self._events[-1]
    def predict(self, *args): return args


def test_a_steering_workspace_exposes_exactly_the_last_150_reviewed_items_and_feedback():
    backing = _Workspace()
    view = BoundedSteeringWorkspace(backing, [f"i{i}" for i in range(200)], score_name="sentiment")

    assert [item.id for item in view.items] == [f"i{i}" for i in range(50, 200)]
    assert [record.item_id for record in view.feedback()] == [f"i{i}" for i in range(50, 200)]
    assert {record.metadata["propensity"] for record in view.feedback()} == {.3}
    assert "OLDER EXPLANATION SENTINEL" not in repr(view.feedback())


def test_a_steering_workspace_rejects_old_and_future_item_and_cache_access():
    backing = _Workspace()
    view = BoundedSteeringWorkspace(backing, [f"i{i}" for i in range(50, 200)], score_name="sentiment")

    with pytest.raises(BoundedWorkspaceError): view.item("old")
    with pytest.raises(BoundedWorkspaceError): view.cache.plan(["i49"], {})
    with pytest.raises(BoundedWorkspaceError): view.cache.partial_answers_for("i0", {})
    view.cache.plan(["i50", "i199"], {})
    assert backing.cache.planned == [("i50", "i199")]


def test_a_steering_workspace_writes_an_approved_scorecard_to_the_main_workspace():
    backing = _Workspace()
    view = BoundedSteeringWorkspace(backing, [f"i{i}" for i in range(50, 200)], score_name="sentiment")

    assert view.commit_scorecard("candidate", kind="steer", provenance={"fit": "oof-only"}) == 8
    assert backing.commits == [("candidate", "steer", {"fit": "oof-only"})]


def test_event_history_is_scoped_to_the_window_without_replies_or_future_events():
    backing = _Workspace()
    view = BoundedSteeringWorkspace(backing, [f"i{i}" for i in range(50, 200)], score_name="sentiment")

    assert view.events() == [
        {"kind": "fit", "score_name": "sentiment", "n_feedback": 0, "n_labeled": 0,
         "fitted": True, "log_loss": .4},
        {"kind": "rethink", "score_name": "sentiment", "n_feedback": 150, "n_labeled": 150},
    ]
    assert "SENTINEL" not in repr(view.events())


def test_feedback_appended_after_a_view_is_created_is_not_steering_context():
    backing = _Workspace()
    view = BoundedSteeringWorkspace(backing, [f"i{i}" for i in range(50, 200)], score_name="sentiment")
    backing._feedback.append(_Feedback("i199", metadata={"comment": "FUTURE EXPLANATION SENTINEL"}))

    assert "FUTURE EXPLANATION SENTINEL" not in repr(view.feedback())


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_a_workspace_rejects_a_non_positive_or_non_integer_review_limit(limit):
    with pytest.raises(ValueError, match="positive integer"):
        BoundedSteeringWorkspace(_Workspace(), ["i1"], score_name="sentiment", limit=limit)


def test_a_workspace_rejects_duplicate_reviews_even_when_the_duplicate_is_before_the_tail():
    with pytest.raises(ValueError, match="each reviewed item once"):
        BoundedSteeringWorkspace(_Workspace(), ["i1", "i1", *[f"i{i}" for i in range(2, 200)]],
                                 score_name="sentiment", limit=150)


@pytest.mark.skipif(not os.environ.get("RUN_STREAM_WORKSPACE_INTEGRATION"),
                    reason="optional pinned-Jev-Flywheel/Tactus integration spec")
def test_a_scripted_steering_round_tops_up_only_the_last_150_reviewed_items():
    """The real host/procedure sees the view, not merely an equivalent fake host."""
    from jev_flywheel.items import FeedbackItem
    from jev_flywheel.steer import ScriptedApprover, run_steering
    from jev_flywheel.workspace import Workspace

    class Response:
        model = "stream-window-fake"
        usage = None

        def __init__(self, answers): self.answers = answers

    class Client:
        def __init__(self): self.calls = []

        async def system_one(self, *, state, questions):
            self.calls.append(state["text"])
            return Response({name: {"noul": .5} for name, question in questions.items()
                             if question.get("question_type", question.get("type")) == "noul"})

    fixture_root = Path(os.environ.get("JEV_FLYWHEEL_FIXTURES", "/Users/home/Projects/Jev-Flywheel/fixtures"))
    with TemporaryDirectory() as directory:
        workspace = Workspace.init(Path(directory) / "workspace", fixture_root)
        reviewed = [item.id for item in workspace.items]
        for index, item_id in enumerate(reviewed):
            workspace.add_feedback(FeedbackItem(
                id=f"review-{index}", item_id=item_id, score_name="Sentiment",
                initial_answer_value="positive", final_answer_value="positive",
                metadata={"propensity": .3, "selection_policy": "stream-uniform-v1"}))
        view = BoundedSteeringWorkspace(workspace, reviewed, score_name="Sentiment")
        client = Client()
        proposal = ("{\"root_cause\":\"test\",\"add_elements\":[{\"key\":\"stream_signal\","
                    "\"question_type\":\"noul\",\"instructions\":\"Is this a streaming signal?\","
                    "\"criteria\":null}],\"retire_elements\":[],\"reword_elements\":[]}")
        outcome = run_steering(
            view, "Sentiment", provider="openai", model="fake", allow_spend=True,
            client_factory=lambda: client, hitl_handler=ScriptedApprover((), default=True),
            # ``run_steering`` installs Tactus's MockManager before parsing its agent.
            mock_replies=[proposal, proposal], max_revisions=1)
        assert {record.metadata["propensity"] for record in view.feedback()} == {.3}
        assert {record.metadata["selection_policy"] for record in view.feedback()} == {"stream-uniform-v1"}

    assert outcome.decision in {"promoted", "not_evaluable", "not_promoted"}
    assert len(client.calls) == 150
    assert set(client.calls) <= {item.text for item in view.items}
