"""A deliberately small, bounded workspace for a streaming steering round.

``jev_flywheel.host.FlywheelHost`` is intentionally written against the
workspace protocol, rather than a database.  This adapter gives it a *view* of
the most recent reviewed items while retaining the real workspace as the sole
place scorecards, answers, and events are committed.  It is not a copy: a
successful proposal still becomes the next version of the real scorecard.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


class BoundedWorkspaceError(PermissionError):
    """A steering view attempted to use an item outside its review window."""


class BoundedAnswerCache:
    """Answer-cache facade that cannot plan, read, or fill outside the window."""

    def __init__(self, cache: Any, item_ids: Iterable[str]):
        self._cache = cache
        self._item_ids = frozenset(item_ids)

    @property
    def model(self):
        return self._cache.model

    def __len__(self):
        # The count is diagnostic only; don't disclose answers outside the view.
        return sum(1 for _ in self.rows())

    def _require(self, item_id: str) -> None:
        if item_id not in self._item_ids:
            raise BoundedWorkspaceError(f"item {item_id!r} is outside the steering review window")

    def _require_many(self, item_ids: Iterable[str]) -> list[str]:
        ids = list(item_ids)
        for item_id in ids:
            self._require(item_id)
        return ids

    def get(self, item_id: str, name: str, question: Mapping[str, Any]):
        self._require(item_id)
        return self._cache.get(item_id, name, question)

    def missing(self, item_id: str, questions: Mapping[str, Mapping[str, Any]]):
        self._require(item_id)
        return self._cache.missing(item_id, questions)

    def answers_for(self, item_id: str, questions: Mapping[str, Mapping[str, Any]]):
        self._require(item_id)
        return self._cache.answers_for(item_id, questions)

    def partial_answers_for(self, item_id: str, questions: Mapping[str, Mapping[str, Any]]):
        self._require(item_id)
        return self._cache.partial_answers_for(item_id, questions)

    def bulk_partial_answers(self, item_ids: Iterable[str], questions: Mapping[str, Mapping[str, Any]]):
        return self._cache.bulk_partial_answers(self._require_many(item_ids), questions)

    def plan(self, item_ids: Iterable[str], questions: Mapping[str, Mapping[str, Any]]):
        return self._cache.plan(self._require_many(item_ids), questions)

    async def fill(self, session: Any, items: Iterable[Any], questions: Mapping[str, Mapping[str, Any]], **kwargs):
        rows = list(items)
        self._require_many(item.id for item in rows)
        return await self._cache.fill(session, rows, questions, **kwargs)

    def put(self, item_id: str, name: str, question: Mapping[str, Any], answer: Mapping[str, Any], model=None):
        self._require(item_id)
        return self._cache.put(item_id, name, question, answer, model)

    def put_hashed(self, item_id: str, name: str, qhash: str, answer: Mapping[str, Any], model=None):
        self._require(item_id)
        return self._cache.put_hashed(item_id, name, qhash, answer, model)

    def rows(self):
        for row in self._cache.rows():
            if row[0] in self._item_ids:
                yield row


class BoundedSteeringWorkspace:
    """A native workspace-shaped view over exactly the reviewed tail.

    ``reviewed_ids`` is chronological; duplicates are rejected because a review
    window must name the exact items a host is allowed to inspect.  The view
    filters feedback by those ids (and optionally score), so future feedback,
    unreviewed text, and held-out text are unavailable even if they are present
    in the backing workspace.
    """

    def __init__(self, workspace: Any, reviewed_ids: Sequence[str], *, score_name: str | None = None,
                 limit: int = 150):
        ids = tuple(reviewed_ids)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        if len(set(ids)) != len(ids):
            raise ValueError("reviewed_ids must contain each reviewed item once")
        if len(ids) > limit:
            ids = ids[-limit:]
        if not ids:
            raise ValueError("a bounded steering workspace needs at least one reviewed item")
        self._workspace = workspace
        self._ids = ids
        self._allowed = frozenset(ids)
        self._score_name = score_name
        # ``Workspace.events`` anchors events in the authoritative, append-only
        # feedback history.  Keep that coordinate long enough to distinguish a
        # past event from one appended after this view was constructed, then
        # translate it to the review-window coordinate exposed to the host.
        self._history_count = len(workspace.feedback())
        self._window_start_count = max(0, self._history_count - len(ids))
        # Snapshot records as well as the count.  An asynchronous reviewer may
        # append feedback while Tactus is running; it must not become analyst
        # context halfway through a round merely because it names an old item.
        self._feedback = tuple(record for record in workspace.feedback()
                               if record.item_id in self._allowed
                               and (score_name is None or record.score_name == score_name))
        # Resolve now: an invalid selection must fail before a host can render text.
        self._items_by_id = {item_id: workspace.item(item_id) for item_id in ids}
        self._cache = BoundedAnswerCache(workspace.cache, self._allowed)

    @property
    def root(self):
        return self._workspace.root

    @property
    def engine(self):
        return self._workspace.engine

    @property
    def version(self):
        return self._workspace.version

    @property
    def cache(self):
        return self._cache

    @property
    def items(self):
        # Preserve review order: the host's balanced tail samples then remain deterministic.
        return [self._items_by_id[item_id] for item_id in self._ids]

    def item(self, item_id: str):
        if item_id not in self._allowed:
            raise BoundedWorkspaceError(f"item {item_id!r} is outside the steering review window")
        return self._items_by_id[item_id]

    def split(self, name: str):
        return [item for item in self.items if item.split == name]

    def feedback(self):
        return list(self._feedback)

    def require(self):
        self._workspace.require()
        return self

    def labeled_ids(self, score_name: str):
        return {record.item_id for record in self.feedback()
                if record.score_name == score_name}

    def n_labeled(self, score_name: str):
        latest = {}
        for record in self.feedback():
            if record.score_name == score_name:
                latest[record.item_id] = record
        return sum(record.label is not None for record in latest.values())

    # Scorecard mutations are intentionally forwarded.  The host cannot provide
    # weights/calibration/provenance: it can only commit its own evaluated fit.
    def scorecard(self, version=None):
        return self._workspace.scorecard(version)

    def commit_scorecard(self, card, *, kind: str, provenance=None):
        return self._workspace.commit_scorecard(card, kind=kind, provenance=provenance)

    def lineage(self):
        return self._workspace.lineage()

    def events(self, kind=None):
        """Return safe, window-relative event anchors from the frozen history.

        The backing workspace may contain old analyst replies (and, after this
        view is made, future feedback/events).  A host needs only fit metrics
        and counts, so expose that small projection and never the free-form
        event payload.  Anchoring before projection means an old fit remains a
        valid baseline at zero labels rather than producing a negative
        ``labels_since_fit``.
        """
        safe = []
        fields = {"kind", "score_name", "version", "n_labeled", "fitted", "log_loss",
                  "oof_accuracy", "oof_brier"}
        for event in self._workspace.events(kind):
            global_count = event.get("n_feedback", 0)
            if not isinstance(global_count, int) or global_count > self._history_count:
                continue
            scoped_count = min(len(self._ids), max(0, global_count - self._window_start_count))
            projected = {key: event[key] for key in fields if key in event}
            projected["n_feedback"] = scoped_count
            # A persisted event can have a global n_labeled count.  The host's
            # arithmetic must use the same bounded coordinate as n_feedback.
            projected["n_labeled"] = min(self.n_labeled(event.get("score_name", "")), scoped_count)
            safe.append(projected)
        return safe

    def log_event(self, kind: str, score_name: str, **data):
        # Persist through the authoritative workspace.  The event is post-round
        # audit data, not analyst input; proposal fitting still used this view.
        self._workspace.log_event(kind, score_name, **data)
        # Return the same redacted/scoped shape a subsequent host call sees.
        events = self.events(kind)
        return events[-1] if events else {"kind": kind, "score_name": score_name,
                                          "n_feedback": len(self.feedback()),
                                          "n_labeled": self.n_labeled(score_name)}

    def predict(self, item_id: str, score_name: str, card=None):
        self.item(item_id)
        return self._workspace.predict(item_id, score_name, card)

    def steering_state(self, score_name: str, n_effective: float = 0.0):
        # Delegate no global state: construct the pinned state type from bounded
        # feedback and bounded event history, retaining comments and provenance.
        from jev_flywheel.items import normalize_label
        from jev_flywheel.steering import SteeringState

        feedback = self.feedback()
        n_labeled = self.n_labeled(score_name)
        fits = [event for event in self.events("fit") if event.get("score_name") == score_name]
        rethinks = [event for event in self.events("rethink") if event.get("score_name") == score_name]
        last_fit = fits[-1]["n_labeled"] if fits else 0
        last_rethink = rethinks[-1] if rethinks else None
        since = feedback[last_rethink["n_feedback"]:] if last_rethink else feedback
        commented = sum(bool(record.edit_comment_value) and record.label is not None
                        and normalize_label(record.initial_answer_value) != normalize_label(record.final_answer_value)
                        for record in since if record.score_name == score_name)
        fitted = [event for event in fits if event.get("fitted")]
        return SteeringState(n_labeled=n_labeled, n_effective=n_effective or float(n_labeled),
                             labels_since_fit=n_labeled - last_fit,
                             labels_since_rethink=n_labeled - (last_rethink["n_labeled"] if last_rethink else 0),
                             commented_mismatches_since_rethink=commented,
                             fit_log_losses=[event["log_loss"] for event in fitted if event.get("log_loss") is not None],
                             oof_accuracy=fitted[-1].get("oof_accuracy") if fitted else None)


def bounded_steering_workspace(workspace: Any, reviewed_ids: Sequence[str], *, score_name: str,
                               limit: int = 150) -> BoundedSteeringWorkspace:
    """Return the review-tail-only workspace for one native steering invocation."""
    return BoundedSteeringWorkspace(workspace, reviewed_ids, score_name=score_name, limit=limit)
