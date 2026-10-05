"""Frozen per-item serving for the unified stream.

This is deliberately a narrow seam: publishing snapshots the current arm as a
core bundle, writes a second content-addressed private copy, and immediately
reloads that copy.  Every subsequent prediction uses that reloaded object;
mutable workspace/head state is never consulted by the serving path.

Import this module only after :func:`unified_env.put_clone_first`; its runtime
dependencies are the same pinned Jev-Flywheel dependencies as ``unified_loop``.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from decision_flywheel.models import Item

from .unified_bundles import CachedJevClient, ScoreRubric, text_free


class ServingUnavailable(RuntimeError):
    """A response cannot safely become features for a served decision."""


class _ValidatedCachedJevClient(CachedJevClient):
    """Cache client which refuses incomplete native answer wires.

    ``Score.feature_vector`` intentionally gives absent answers zero-valued
    features.  That is useful during fitting but is unsafe at the serving
    boundary, so validate after both cache and fresh-response paths.
    """

    def __init__(self, *args, rubric: ScoreRubric, **kwargs):
        super().__init__(*args, **kwargs)
        self.rubric = rubric

    async def system_one(self, *, state, questions):
        response = await super().system_one(state=state, questions=questions)
        # Kept lazy to avoid importing the stream module's optional runtime at
        # module import time, while retaining one shared wire definition.
        from .unified_stream import _answers_complete, _feature_row_complete

        if not _answers_complete(questions, response.answers):
            raise ServingUnavailable("cached or provider response lacks complete native answers")
        features = self.rubric.score.feature_vector(response.answers)
        if not _feature_row_complete(self.rubric.score.decision.features, features):
            raise ServingUnavailable("complete answers do not cover every served head feature")
        return response


@dataclass(frozen=True)
class FrozenServingPublication:
    """Text-free identity returned by :meth:`FrozenStreamClassifier.publish`."""

    arm: str
    artifact_id: str
    bundle_hash: str
    path: Path
    manifest: Mapping[str, Any]

    def metadata(self) -> dict[str, Any]:
        return {"artifact_id": self.artifact_id, "path": str(self.path), **text_free(self.manifest)}


class FrozenStreamClassifier:
    """Serve one item at a time through an immutable, verified bundle.

    ``flywheel`` is intentionally duck-typed so this seam remains usable by
    the stream arm without adding another loop implementation.  Publishing is
    allowed to use its existing ``_freeze`` routine for the authoritative
    Score/list/label-row construction, but never relies on that routine's
    conventional round directory: the returned bundle is saved under its hash
    and reloaded from there.
    """

    def __init__(self, flywheel: Any, arm: str, state: Any, labeled: Sequence[str], *,
                 client_factory: Optional[Any] = None):
        self.flywheel = flywheel
        self.arm = str(arm)
        self.state = state
        self.labeled = tuple(str(item_id) for item_id in labeled)
        self.client_factory = client_factory
        self.publication: Optional[FrozenServingPublication] = None
        self.bundle: Any = None
        self._serve_round = 0

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.flywheel.corpus.labels)

    def publish(self) -> FrozenServingPublication:
        """Persist and reload a content-addressed snapshot of the current arm."""
        from decision_flywheel.bundle import load_bundle

        saved = self.flywheel._freeze(self.arm, self.state, self.labeled)
        digest = saved.bundle_hash
        root = Path(self.flywheel.run_dir) / "stream-serving" / self.arm.replace("+", "plus")
        target = root / digest
        # ClassifierBundle.save refuses an accidental different payload at an
        # existing location; it also materializes a deep scorecard/list copy.
        saved.save(target)
        reloaded = load_bundle(target, self.flywheel.task, ScoreRubric.parse,
                               configured_model=self.flywheel.cfg.provider_model)
        if reloaded.bundle_hash != digest:
            raise ServingUnavailable("reloaded bundle identity does not match published hash")
        manifest = reloaded.manifest()
        publication = FrozenServingPublication(self.arm, f"{self.arm}:{digest}", digest, target, manifest)
        self.publication, self.bundle = publication, reloaded
        return publication

    def reload(self, publication: Optional[FrozenServingPublication] = None) -> FrozenServingPublication:
        """Verify a persisted publication again, refusing any tampered file."""
        from decision_flywheel.bundle import load_bundle

        publication = publication or self.publication
        if publication is None:
            raise ServingUnavailable("no published bundle")
        bundle = load_bundle(publication.path, self.flywheel.task, ScoreRubric.parse,
                             configured_model=self.flywheel.cfg.provider_model)
        if bundle.bundle_hash != publication.bundle_hash:
            raise ServingUnavailable("reloaded bundle identity does not match published hash")
        self.publication, self.bundle = publication, bundle
        return publication

    def classify(self, item_id: str):
        """Return the core calibrated ``DecisionResult``, or ``None`` if unavailable.

        A miss makes exactly one bundle request containing every missing rubric
        question and the immutable bundle context.  A complete cache hit makes
        none.  This method reads item text only; it never reads a target label
        or a held-out partition.
        """
        if self.bundle is None:
            raise ServingUnavailable("publish a bundle before serving")
        item_id = str(item_id)
        try:
            target = self.flywheel.splits.items[item_id]
        except KeyError as error:
            raise ValueError(f"unknown stream item {item_id!r}") from error
        self._serve_round += 1
        self.flywheel.ledger.set_scope(self.arm, self._serve_round, "frozen-stream-predict")
        async def one():
            # Some SDK clients bind themselves to the currently-running event
            # loop, so constructing the factory outside this coroutine is not
            # safe even though a complete cache hit makes no request.
            inner = (self.client_factory or self.flywheel.engines.zero_shot_factory)()
            client = _ValidatedCachedJevClient(
                inner, {target.text: item_id}, zero_shot=self.flywheel.cache,
                lists=self.flywheel.list_answers, fixed=self.bundle.examples,
                rubric=self.bundle.rubric,
            )
            return await self.bundle.classify(Item(item_id, {"text": target.text}), client)

        try:
            result = asyncio.run(one())
        except ServingUnavailable:
            return None
        except Exception:
            # Provider/cache failures must not be converted into a chance
            # prediction.  The counting client has already accounted for an
            # attempted provider request where applicable.  Propagate a
            # tripped circuit, but retain ordinary max-new/ceiling exhaustion
            # as an unavailable item.
            self.flywheel.ledger.raise_if_tripped()
            return None
        # A response failure may have tripped a global circuit.  That is a
        # hard stop, unlike a local cache hole or a max-new ceiling refusal.
        self.flywheel.ledger.raise_if_tripped()
        return result

    def predict(self, item_id: str):
        """StreamDriver-compatible projection of :meth:`classify`."""
        result = self.classify(item_id)
        if result is None:
            return None, {}, "unavailable"
        probabilities = dict(result.probabilities or {})
        return result.label, probabilities, "ok" if result.label in probabilities else "unavailable"

    def usage(self) -> dict[str, int]:
        """Only this service's ledger scopes, suitable for text-free reporting."""
        scoped = self.flywheel.ledger.summary().get("by_arm_round", {}).get(self.arm, {})
        counts = {"attempts": 0, "failures": 0, "input_tokens": 0, "output_tokens": 0}
        for kinds in scoped.values():
            for kind, values in kinds.items():
                if kind != "frozen-stream-predict":
                    continue
                for key in counts:
                    counts[key] += int(values.get(key, 0))
        return counts
