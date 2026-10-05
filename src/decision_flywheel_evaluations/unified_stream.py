"""Offline, prequential stream runner shared by the Amazon-review study.

This module deliberately owns scheduling and the text-free audit trail, not an
optimizer.  A supplied arm owns predictions, trusted-label ingestion, refitting
and optional steering.  That small seam lets the real arm compose
``UnifiedFlywheel``/Jev-Flywheel helpers while specs exercise every arm with a
fake and no model client.
"""
from __future__ import annotations

import json
import random
import hashlib
import shutil
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


SCHEMA = "decision-flywheel-evaluations/stream/v1"
LEARNING_ARMS = frozenset(("L", "E", "X"))


def _answers_complete(questions: Mapping[str, Mapping[str, Any]], answers: Mapping[str, Any]) -> bool:
    """Strictly validate native answer wires before feature extraction.

    Cache-key presence alone is insufficient: Jev feature extraction deliberately
    fills absent values with zeros, which must never turn a malformed response
    into a served prediction.
    """
    if not isinstance(answers, Mapping):
        return False
    for name, question in questions.items():
        answer = answers.get(name)
        if not isinstance(answer, Mapping):
            return False
        kind = question.get("question_type", question.get("type"))
        if kind == "noul":
            value = answer.get("noul")
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                return False
        elif kind == "choice":
            criteria = question.get("criteria") or ()
            options = tuple(criteria.keys()) if isinstance(criteria, Mapping) else tuple(criteria)
            probabilities = answer.get("probabilities")
            if (not options or answer.get("choice") not in options or not isinstance(probabilities, Mapping)
                    or set(probabilities) != set(options)):
                return False
            values = list(probabilities.values())
            if any(not isinstance(value, (int, float)) or isinstance(value, bool)
                   or not math.isfinite(value) or not 0 <= value <= 1 for value in values):
                return False
            if abs(sum(values) - 1.0) > 1e-6:
                return False
        elif kind == "score":
            value = answer.get("score")
            criteria = question.get("criteria") or ()
            options = tuple(criteria.keys()) if isinstance(criteria, Mapping) else tuple(criteria)
            probabilities = answer.get("probabilities")
            low, high = 0, len(options) - 1
            if (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
                    or int(value) != value or value < low or value > high
                    or not isinstance(probabilities, Mapping)):
                return False
            expected = {str(i) for i in range(len(options))}
            if set(probabilities) != expected:
                return False
            values = list(probabilities.values())
            if (any(not isinstance(item, (int, float)) or isinstance(item, bool)
                    or not math.isfinite(item) or not 0 <= item <= 1 for item in values)
                    or abs(sum(values) - 1.0) > 1e-6):
                return False
        else:
            return False
    return True


def _feature_row_complete(features: Sequence[str], row: Mapping[str, Any]) -> bool:
    return all(name in row and isinstance(row[name], (int, float)) and not isinstance(row[name], bool)
               and math.isfinite(row[name]) for name in features)


@dataclass(frozen=True)
class StreamConfig:
    seed: int = 1
    review_probability: float = .3
    batch_size: Optional[int] = None
    cold_start_reviews: int = 30
    refit_every_reviews: int = 30
    max_refits: int = 4
    max_steering_rounds: int = 3
    checkpoint_at: int = 300
    list_per_label: int = 4

    def __post_init__(self) -> None:
        if not 0 <= self.review_probability <= 1:
            raise ValueError("review_probability must be in [0, 1]")
        if self.batch_size is not None and (isinstance(self.batch_size, bool)
                                          or not isinstance(self.batch_size, int)
                                          or self.batch_size not in range(1, 11)):
            raise ValueError("batch_size must be from 1 through 10")
        if self.cold_start_reviews < 1 or self.refit_every_reviews < 1 or self.list_per_label < 1:
            raise ValueError("review thresholds must be positive")


@dataclass(frozen=True)
class ServedPrediction:
    # One-based arrival index: stable for reports and never an array offset.
    stream_index: int
    batch_index: int
    item_id: str
    true_label: str
    predicted_label: Optional[str]
    probabilities: Mapping[str, float]
    status: str
    review_selected: bool
    review_propensity: float
    reviewed_count_at_prediction: int
    bundle_hash: Optional[str] = None


@dataclass(frozen=True)
class ReviewRecord:
    item_id: str
    label: Optional[str]
    mode: str
    selected: bool
    propensity: float
    status: str = "accepted"
    reason_sha256: Optional[str] = None


@dataclass(frozen=True)
class HeldoutPrediction:
    item_id: str
    true_label: str
    predicted_label: Optional[str]
    probabilities: Mapping[str, float]
    status: str


@dataclass(frozen=True)
class Checkpoint:
    name: str
    served_count: int
    reviewed_count: int
    heldout: tuple[HeldoutPrediction, ...]
    bundle: Optional[Mapping[str, Any]] = None


@dataclass
class StreamResult:
    arm: str
    seed: int
    is_synthetic: bool
    served: list[ServedPrediction] = field(default_factory=list)
    reviews: list[ReviewRecord] = field(default_factory=list)
    checkpoints: list[Checkpoint] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """A reporting boundary: only identifiers, labels, hashes and measurements."""
        return {"schema": SCHEMA, "arm": self.arm, "seed": self.seed,
                "is_synthetic": self.is_synthetic, "served": [asdict(x) for x in self.served],
                # ``stream`` keeps the step-5 reporter compatible with its first draft.
                "stream": [asdict(x) for x in self.served],
                "reviews": [asdict(x) for x in self.reviews], "usage": dict(self.usage),
                "checkpoints": [asdict(x) for x in self.checkpoints], "counters": dict(self.counters)}

    as_dict = to_dict


class StreamDriver:
    """Serve first, then possibly reveal feedback, using a shared uniform draw plan.

    ``engine.predict(id)`` returns ``(label, probabilities, status)``.
    Learning engines implement ``record_label`` and ``refit``; an engine can
    implement ``steering_triggered(reviewed)`` and ``steer``.  Keeping that
    interface structural avoids importing the expensive pinned runtime merely
    to load reports or run unit specs.
    """
    def __init__(self, config: StreamConfig):
        self.config = config

    def sample_plan(self, items: Sequence[tuple[str, str]]) -> list[tuple[str, bool, float]]:
        """The arm-independent Bernoulli plan. Never inspect a prediction here."""
        rng = random.Random(f"stream-review-v1:{self.config.seed}")
        p = self.config.review_probability
        return [(item_id, rng.random() < p, p) for item_id, _ in items]

    def batch_indices(self, n: int) -> list[int]:
        """Independent seeded batches, or the explicitly requested fixed size."""
        rng = random.Random(f"stream-batches-v1:{self.config.seed}")
        out: list[int] = []
        batch = 0
        while len(out) < n:
            batch += 1
            size = self.config.batch_size or rng.randint(1, 10)
            out.extend([batch] * min(size, n - len(out)))
        return out

    def run(self, arm: str, engine: Any, stream: Sequence[tuple[str, str]], *,
            selection_plan: Optional[Sequence[tuple[str, bool, float]]] = None,
            is_synthetic: bool = True) -> StreamResult:
        if arm not in ("B", "L", "E", "X"):
            raise ValueError("stream arm must be one of B, L, E, X")
        items = [(str(item_id), str(label)) for item_id, label in stream]
        plan = list(selection_plan or self.sample_plan(items))
        if len({i for i, _ in items}) != len(items):
            raise ValueError("stream item IDs must be unique")
        if [row[0] for row in plan] != [row[0] for row in items]:
            raise ValueError("selection plan must align exactly with the stream")
        if any(not isinstance(selected, bool) for _, selected, _ in plan):
            raise ValueError("selection plan selections must be booleans")
        if any(p != self.config.review_probability for _, _, p in plan):
            raise ValueError("stream reviews must retain the uniform configured propensity")
        result = StreamResult(arm, self.config.seed, is_synthetic,
                              counters={"refit_rounds": 0, "steering_rounds": 0, "list_rounds": 0,
                                        "request_attempts": 0, "request_failures": 0})
        reviewed: list[str] = []
        next_refit = self.config.cold_start_reviews
        next_list = self.config.cold_start_reviews
        checkpoint_done = False
        batches = self.batch_indices(len(items))
        for index, ((item_id, truth), (_, selected, propensity)) in enumerate(zip(items, plan), start=1):
            batch = batches[index - 1]
            label, probabilities, status = self._prediction(engine, item_id)
            digest = getattr(engine, "served_bundle_hash", None)
            digest = (digest if isinstance(digest, str) and len(digest) == 64
                      and all(char in "0123456789abcdef" for char in digest) else None)
            # This append must precede feedback: its count makes leakage auditable.
            result.served.append(ServedPrediction(index, batch, item_id, truth, label, probabilities, status,
                                                   selected, propensity, len(reviewed), digest))
            if selected:
                mode = {"B": "none", "L": "labels", "E": "explanation", "X": "shuffled_explanation"}[arm]
                digest = self._review_digest(engine, item_id, mode)
                review = ReviewRecord(item_id, truth, mode, True, propensity, reason_sha256=digest)
                result.reviews.append(review)
                if arm in LEARNING_ARMS:
                    engine.record_label(item_id, truth, propensity=propensity, explanation_mode=mode)
                    reviewed.append(item_id)
            end_of_batch = index == len(items) or batches[index] != batch
            if end_of_batch and arm in LEARNING_ARMS and len(reviewed) >= next_refit and result.counters["refit_rounds"] < self.config.max_refits:
                engine.refit(tuple(reviewed))
                result.counters["refit_rounds"] += 1
                next_refit += self.config.refit_every_reviews
            if (end_of_batch and arm in LEARNING_ARMS and len(reviewed) >= next_list
                    and result.counters["list_rounds"] < self.config.max_refits
                    and self._list_ready(engine, reviewed)):
                if bool(getattr(engine, "optimize_list", lambda *_: False)(tuple(reviewed))):
                    result.counters["list_rounds"] += 1
                next_list += self.config.refit_every_reviews
            if (end_of_batch and len(reviewed) >= self.config.cold_start_reviews and arm in LEARNING_ARMS and result.counters["steering_rounds"] < self.config.max_steering_rounds
                    and self._steering_triggered(engine, reviewed)):
                completed = engine.steer()
                if completed is not False:
                    result.counters["steering_rounds"] += 1
            if index == self.config.checkpoint_at:
                result.checkpoints.append(self._checkpoint(f"stream-{self.config.checkpoint_at}", index, len(reviewed), engine))
                checkpoint_done = True
        result.checkpoints.append(self._checkpoint("end", len(items), len(reviewed), engine))
        self._usage(engine, result.counters)
        result.usage = {key: result.counters.get(key, 0) for key in
                        ("request_attempts", "request_failures", "input_tokens", "output_tokens")}
        return result

    @staticmethod
    def _prediction(engine: Any, item_id: str) -> tuple[Optional[str], Mapping[str, float], str]:
        outcome = engine.predict(item_id)
        if isinstance(outcome, Mapping):
            label, probabilities, status = outcome.get("label"), outcome.get("probabilities"), outcome.get("status")
        else:
            label, probabilities, status = outcome
        labels = set(getattr(engine, "labels", ()))
        label = label if label in labels else None
        clean = {str(key): float(value) for key, value in dict(probabilities or {}).items()
                 if key in labels and isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= 0}
        status = str(status) if status in {"ok", "unavailable", "malformed"} else "malformed"
        if label is None:
            status = "unavailable" if status == "ok" else status
        return label, clean, status

    @staticmethod
    def _steering_triggered(engine: Any, reviewed: Sequence[str]) -> bool:
        callback = getattr(engine, "steering_triggered", None)
        return bool(callback(tuple(reviewed))) if callback else False

    def _checkpoint(self, name: str, served: int, reviewed: int, engine: Any) -> Checkpoint:
        freezer = getattr(engine, "freeze", None)
        bundle = freezer(name) if freezer else None
        rows = []
        # A concrete flywheel checkpoint is only scoreable through its frozen,
        # reloadable bundle.  Falling back to the mutable in-memory head would
        # make a failed save look like a valid frozen checkpoint.
        if freezer:
            source = getattr(engine, "heldout_from_bundle", lambda _bundle: ())
            outcomes = source(bundle) if bundle is not None else ()
        else:
            outcomes = getattr(engine, "heldout", lambda: ())()
        for item_id, truth, label, probabilities, status in outcomes:
            labels = set(getattr(engine, "labels", ()))
            clean = {str(key): float(value) for key, value in dict(probabilities or {}).items()
                     if key in labels and isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= 0}
            label = label if label in labels else None
            status = str(status) if status in {"ok", "unavailable", "malformed"} else "malformed"
            if label is None:
                status = "unavailable" if status == "ok" else status
            rows.append(HeldoutPrediction(str(item_id), str(truth), label, clean, status))
        return Checkpoint(name, served, reviewed, tuple(rows), bundle)

    @staticmethod
    def _usage(engine: Any, counters: dict[str, int]) -> None:
        usage = getattr(engine, "usage", None)
        usage = usage() if callable(usage) else usage
        if isinstance(usage, Mapping):
            counters["request_attempts"] = int(usage.get("attempts", usage.get("requests", 0)))
            counters["request_failures"] = int(usage.get("failures", 0))
            counters["input_tokens"] = int(usage.get("input_tokens", 0))
            counters["output_tokens"] = int(usage.get("output_tokens", 0))

    @staticmethod
    def _review_digest(engine: Any, item_id: str, mode: str) -> Optional[str]:
        comment = getattr(engine, "comment_for", lambda *_: None)(item_id, mode)
        return hashlib.sha256(comment.encode()).hexdigest() if comment else None

    @staticmethod
    def _list_ready(engine: Any, reviewed: Sequence[str]) -> bool:
        callback = getattr(engine, "list_ready", None)
        return bool(callback(tuple(reviewed))) if callback else False


class _FixtureArm:
    """A deterministic, no-network engine for the offline CLI fixture format."""
    def __init__(self, labels: Sequence[str], predictions: Mapping[str, Any], heldout: Sequence[Mapping[str, Any]]):
        self.predictions, self.rows = predictions, heldout
        self.labels = tuple(labels)
        self.recorded_labels: list[tuple[str, str, float, str]] = []
        self.fits = self.steers = 0

    def predict(self, item_id: str):
        row = self.predictions.get(item_id, {})
        row = {"label": row} if isinstance(row, str) else row
        return row.get("label"), row.get("probabilities", {}), row.get("status", "ok")

    def record_label(self, item_id, label, *, propensity, explanation_mode):
        self.recorded_labels.append((item_id, label, propensity, explanation_mode))

    def refit(self, reviewed): self.fits += 1
    def steering_triggered(self, reviewed): return len(reviewed) >= 30
    def steer(self): self.steers += 1
    def heldout(self):
        return [(r["id"], r["label"], self.predict(r["id"])[0], self.predict(r["id"])[1], self.predict(r["id"])[2]) for r in self.rows]


class UnifiedFlywheelStreamArm:
    """Concrete bridge to the existing fake-capable ``UnifiedFlywheel`` runtime.

    It is intentionally constructed only after ``unified_env.put_clone_first`` and
    the offline network guard.  The bridge does not fit its own model: labels go
    through pinned ``record_label`` (including propensity), refits through the
    harness's existing `_gate`, and steering through `_steer`/`_cap_to_budget`.
    """
    def __init__(self, flywheel: Any, state: Any, arm: str, *, comments: Optional[Mapping[str, str]] = None,
                 list_per_label: int = 4):
        self.flywheel, self.state, self.arm = flywheel, state, arm
        self.labels = tuple(flywheel.corpus.labels)
        self.comments = dict(comments or {})
        self.reviewed: list[str] = []
        self._questions: dict[str, Any] = {}
        self._round = 0
        self.list_per_label = list_per_label
        self._last_list_record: Optional[Mapping[str, Any]] = None
        self._hard_demos: list[tuple[float, str]] = []
        self._plain_mismatches = 0
        self.fit_window_ids: tuple[str, ...] = ()
        self._checkpoint_paths: dict[str, Path] = {}
        self._steering_round = 0
        self._frozen_service = None
        self._frozen_semantic_key: Optional[tuple[str, Optional[str], Optional[str]]] = None

    @classmethod
    def create(cls, flywheel: Any, arm: str, *, comments: Optional[Mapping[str, str]] = None,
               list_per_label: int = 4):
        """Build an arm state from the existing harness's seed scorecard.

        The caller must have installed the network guard before constructing the
        supplied offline ``UnifiedFlywheel``.
        """
        from . import unified_env
        if flywheel.cfg.live or not unified_env.network_guard_installed():
            raise RuntimeError("UnifiedFlywheelStreamArm is offline-only and requires the network guard")
        from jev_flywheel.scorecard import Score
        from .unified_loop import ArmState
        seed = flywheel.v1.score(flywheel.corpus.score_name)
        workspace = flywheel._workspace(arm) if arm in LEARNING_ARMS else None
        return cls(flywheel, ArmState(arm, Score.from_config(seed.to_config()), workspace), arm, comments=comments,
                   list_per_label=list_per_label)

    def _require_pool_item(self, item_id: str) -> str:
        """Reject any dev/held-out ID before a stream operation can mutate state."""
        item_id = str(item_id)
        if item_id not in set(self.flywheel.splits.pool):
            raise ValueError("stream prediction and feedback are pool-only")
        return item_id

    def _serving_key(self) -> tuple[str, Optional[str], Optional[str]]:
        """Only a served score/context/card change earns a new publication."""
        from .unified_bundles import ScoreRubric

        list_fingerprint = getattr(self.state.example_list, "fingerprint", None)
        card_fingerprint = None
        if self.state.workspace is not None:
            card = self.state.workspace.scorecard().score(self.flywheel.corpus.score_name)
            card_fingerprint = ScoreRubric.dump(card)
        return ScoreRubric.dump(self.state.head), list_fingerprint, card_fingerprint

    def _serving(self):
        """Return the current immutable snapshot, publishing lazily when needed."""
        key = self._serving_key()
        if self._frozen_service is None or self._frozen_semantic_key != key:
            from .unified_stream_serving import FrozenStreamClassifier

            service = FrozenStreamClassifier(self.flywheel, self.arm, self.state, tuple(self.reviewed))
            service.publish()
            self._frozen_service, self._frozen_semantic_key = service, key
        return self._frozen_service

    @property
    def bundle_hash(self) -> Optional[str]:
        """Hash of the immutable classifier that served the latest prediction."""
        publication = getattr(self._frozen_service, "publication", None)
        return publication.bundle_hash if publication is not None else None

    @property
    def served_bundle_hash(self) -> Optional[str]:
        """Text-free provenance exposed to ``StreamDriver`` for this prediction."""
        digest = self.bundle_hash
        return digest if (isinstance(digest, str) and len(digest) == 64
                          and all(char in "0123456789abcdef" for char in digest)) else None

    def predict(self, item_id: str):
        item_id = self._require_pool_item(item_id)
        self._round += 1
        result = self._serving().classify(item_id)
        label = None if result is None else result.label
        confidence = 0.0 if result is None else float(result.confidence or 0.0)
        probabilities = {} if result is None else dict(result.probabilities or {})
        if self.state.workspace is not None:
            self._questions[item_id] = self._question(item_id, label, confidence)
        return label, probabilities, "ok" if label in probabilities else "unavailable"

    def _question(self, item_id: str, label: Optional[str], confidence: float):
        # Pinned record_label expects a Question so it can persist selection
        # provenance.  Our stream is uniformly sampled, hence a synthetic
        # Selection with configured propensity is substituted just before record.
        from jev_flywheel.loop import Question
        from jev_flywheel.scoring import ScoreResult
        from jev_flywheel.selection import Candidate, Selection
        result = ScoreResult(self.flywheel.corpus.score_name, label, confidence)
        candidate = Candidate(item_id, result, {}, {}, 0.0)
        selection = Selection(candidate, 1.0, "stream-uniform-placeholder", 1)
        return Question(self.flywheel._jev_item(item_id), result, selection, self.state.workspace.version,
                        classes=list(self.state.head.decision.classes))

    def record_label(self, item_id, label, *, propensity, explanation_mode):
        item_id = self._require_pool_item(item_id)
        from jev_flywheel.loop import AGREE, DISAGREE, record_label
        question = self._questions[item_id]
        question.selection.propensity = propensity
        question.selection.policy = "stream-uniform-v1"
        # Comments are retained only in the workspace feedback record; the stream
        # result publishes mode/hash/counts, never SME prose.
        comment = self.comment_for(item_id, explanation_mode)
        if question.result.value is None:
            from jev_flywheel.items import FeedbackItem, LABEL_SOURCE_FINAL
            self.state.workspace.add_feedback(FeedbackItem(
                id=f"stream-unavailable-{item_id}-{self._round}", item_id=item_id,
                score_name=self.flywheel.corpus.score_name, initial_answer_value=None,
                final_answer_value=label, edit_comment_value=comment or None,
                is_agreement=False, editor_name="scripted-sme", label_source=LABEL_SOURCE_FINAL,
                metadata={"propensity": propensity, "selection_policy": "stream-uniform-v1",
                          "prediction_status": "unavailable"}))
            self.reviewed.append(item_id)
            return
        verdict = AGREE if question.result.value == label else DISAGREE
        if verdict == DISAGREE and question.result.value is not None:
            self._hard_demos.append((float(question.result.confidence or 0.0), item_id))
            self._plain_mismatches += 1
        record_label(self.state.workspace, question, verdict, correct_label=label, comment=comment,
                     editor="scripted-sme")
        self.reviewed.append(item_id)

    def comment_for(self, item_id, explanation_mode):
        if explanation_mode in ("labels", "none"):
            return None
        if explanation_mode == "shuffled_explanation":
            # A deterministic reviewed-prefix shuffle keeps explanation content but
            # never lets an item retain its own explanation.  Crucially its
            # donor is already reviewed: future SME feedback is unavailable.
            donors = sorted((set(self.reviewed) & set(self.comments)) - {item_id})
            if not donors:
                return None
            random.Random(f"stream-x-v1:{self.flywheel.cfg.seed}:{item_id}:{len(donors)}").shuffle(donors)
            return self.comments[donors[0]]
        return self.comments.get(item_id)

    def refit(self, reviewed):
        if self.state.example_list is not None:
            self._refit_fixed_list(reviewed)
            return
        from .unified_loop import candidate_template, with_fit
        base = self.state.workspace.scorecard().score(self.flywheel.corpus.score_name)
        template = candidate_template(base, fewshot=False, knn=False, corpus=self.flywheel.corpus)
        # The stream contract bounds expensive feature top-ups and fitting to a
        # declared rolling window.  Older trusted labels remain audit records,
        # never silently become missing-feature rows in an all-history fit.
        fit_ids = tuple(reviewed[-150:])
        self.fit_window_ids = fit_ids
        self.flywheel.ledger.set_scope(self.arm, self._round, "stream-zero-window-coverage")
        filled = self.flywheel._fill_zero_shot(fit_ids, template.questions())
        if filled.get("failures"):
            return
        answers = self.flywheel.cache.bulk_partial_answers(fit_ids, template.questions())
        if any(not _answers_complete(template.questions(), answers.get(item_id, {})) for item_id in fit_ids):
            return
        rows = self.flywheel._rows(template, fit_ids, fit_ids)
        if any(not _feature_row_complete(template.decision.features, rows.get(item_id, {})) for item_id in fit_ids):
            return
        self.state.head, _gate, fit = self.flywheel._gate(self.arm, self._round, template, self.state.head, fit_ids)
        if fit is not None:
            self.state.workspace.commit_scorecard(with_fit(self.state.workspace.scorecard(),
                                                            self.flywheel.corpus.score_name, fit), kind="fit",
                                                 provenance=fit.provenance)

    def list_ready(self, reviewed: Sequence[str]) -> bool:
        if self.arm not in LEARNING_ARMS:
            return False
        counts = {label: 0 for label in self.flywheel.corpus.labels}
        for item_id in reviewed:
            counts[self.flywheel.labels[item_id]] = counts.get(self.flywheel.labels[item_id], 0) + 1
        # FixedExampleList requires a reserve in addition to each shown example.
        return all(count >= self.list_per_label + 1 for count in counts.values())

    def optimize_list(self, reviewed: Sequence[str]) -> bool:
        """Atomically promote a context together with its fitted/calibrated head.

        The optimizer reads the context-keyed ``ListAnswers`` cache and makes one
        request per target/list trial through its existing fill path.  The
        driver refits before calling this method, so assigning a new list here
        without fitting it would serve an old head against a new context.
        """
        if not self.list_ready(reviewed):
            return False
        from jev_flywheel.fit import compare, fit_head
        from jev_flywheel.ladder import LadderRefusal
        from .unified_loop import candidate_template, head_from_fit, served_summary, training_set

        base = self.state.workspace.scorecard().score(self.flywheel.corpus.score_name)
        template = candidate_template(base, fewshot=False, knn=False, corpus=self.flywheel.corpus)
        self.flywheel.ledger.set_scope(self.arm, self._round, "stream-list-optimize")
        hard_demos = [item_id for _, item_id in sorted(self._hard_demos, key=lambda x: (-x[0], x[1]))]
        chosen, record = self.flywheel._improve_list(self.state, reviewed, hard_demos, template.questions(), self._round)
        self._last_list_record = record
        incumbent = self.state.example_list
        if incumbent is not None and chosen.fingerprint == incumbent.fingerprint:
            # Search completed but retained the already coherent classifier.
            self.state.list_record = dict(record)
            return True

        # The context and head must cross the same full-feature fit boundary.
        # Labels outside this declared horizon remain trusted audit provenance,
        # but cannot become partial rows in a fit merely because a list changed.
        fit_ids = tuple(reviewed[-150:])
        self.fit_window_ids = fit_ids
        filled = self.flywheel._list_fill(chosen, fit_ids, template.questions())
        if filled.get("failures"):
            return False
        answers = self.flywheel.list_answers.answers(fit_ids, template.questions(), chosen)
        if any(not _answers_complete(template.questions(), answers.get(item_id, {})) for item_id in fit_ids):
            return False
        rows = self.flywheel._list_rows(template, fit_ids, chosen)
        if any(not _feature_row_complete(template.decision.features, rows.get(item_id, {})) for item_id in fit_ids):
            return False
        training = training_set(template, fit_ids, self.flywheel.labels, rows,
                                f"example-list:{chosen.fingerprint}")
        if training.needs_answers:
            return False
        try:
            fitted = fit_head(training, template, folds=self.flywheel.cfg.folds, seed=self.flywheel.cfg.fit_seed)
        except LadderRefusal:
            return False
        if not fitted.fitted:
            return False
        # Even with no incumbent list, the current zero-shot head is measured
        # on this candidate context; no calibration or fit is fabricated.
        incumbent_rows = self.flywheel._list_rows(self.state.head, training.item_ids, chosen)
        incumbent_score = served_summary(self.state.head, training.item_ids, incumbent_rows,
                                         self.flywheel.labels, training.weights)
        if not compare(fitted, incumbent_score, min_brier_gain=self.flywheel.cfg.min_brier_gain).promote:
            # A fully evaluated attempt is still consumed by the scheduler;
            # keeping the incumbent avoids retrying the identical list every
            # batch while preserving its matching head.
            self._last_list_record = {**record, "served": "rejected-head-gate"}
            return True
        self.state.example_list = chosen
        self.state.head = head_from_fit(template, fitted)
        self.state.list_record = {**record, "head_fit_id": fitted.provenance.get("fit_id"),
                                  "fit_window_n": len(training.item_ids)}
        return True

    def _refit_fixed_list(self, reviewed: Sequence[str]) -> None:
        """Fit only after every trusted label has the chosen list's full features."""
        from jev_flywheel.fit import compare, fit_head
        from jev_flywheel.ladder import LadderRefusal
        from .unified_loop import candidate_template, head_from_fit, served_summary, training_set
        base = self.state.workspace.scorecard().score(self.flywheel.corpus.score_name)
        template = candidate_template(base, fewshot=False, knn=False, corpus=self.flywheel.corpus)
        # The serving/list cache is intentionally topped up only for the last
        # 150 reviews.  Older labels remain trusted provenance but are outside
        # this explicitly recorded fit window; we never issue an unbounded
        # retroactive top-up then misdescribe the result as an all-label fit.
        fit_ids = tuple(reviewed[-150:])
        self.fit_window_ids = fit_ids
        self.flywheel.ledger.set_scope(self.arm, self._round, "stream-list-window-coverage")
        filled = self.flywheel._list_fill(self.state.example_list, fit_ids, template.questions())
        if filled.get("failures"):
            return  # no partial-label fit after a failed top-up
        answers = self.flywheel.list_answers.answers(fit_ids, template.questions(), self.state.example_list)
        if any(not _answers_complete(template.questions(), answers.get(item_id, {})) for item_id in fit_ids):
            return
        rows = self.flywheel._list_rows(template, fit_ids, self.state.example_list)
        if any(not _feature_row_complete(template.decision.features, rows.get(item_id, {})) for item_id in fit_ids):
            return
        training = training_set(template, fit_ids, self.flywheel.labels, rows,
                                f"example-list:{self.state.example_list.fingerprint}")
        if training.needs_answers:
            return
        try:
            fitted = fit_head(training, template, folds=self.flywheel.cfg.folds, seed=self.flywheel.cfg.fit_seed)
        except LadderRefusal:
            return
        if fitted.fitted:
            incumbent_rows = self.flywheel._list_rows(self.state.head, training.item_ids, self.state.example_list)
            incumbent = served_summary(self.state.head, training.item_ids, incumbent_rows,
                                       self.flywheel.labels, training.weights)
            if compare(fitted, incumbent, min_brier_gain=self.flywheel.cfg.min_brier_gain).promote:
                self.state.head = head_from_fit(template, fitted)

    def freeze(self, checkpoint_name: str):
        """Freeze the currently served bundle; private bundle files stay under run_dir."""
        try:
            saved = self.flywheel._freeze(self.arm, self.state, self.reviewed)
            from .unified_bundles import bundle_dir, text_free, ScoreRubric
            source = bundle_dir(self.flywheel.run_dir, self.arm, len(self.flywheel.batches))
            artifact_id = f"{self.arm}:{checkpoint_name}:{saved.bundle_hash}"
            target = self.flywheel.run_dir / "stream-checkpoints" / self.arm / artifact_id.replace(":", "-")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, target)
            from decision_flywheel.bundle import load_bundle
            bundle = load_bundle(target, self.flywheel.task, ScoreRubric.parse,
                                 configured_model=self.flywheel.cfg.provider_model)
            if bundle.bundle_hash != saved.bundle_hash:
                return None
            self._checkpoint_paths[artifact_id] = target
        except Exception:  # artifact remains unavailable rather than reporting a fake hash
            self.flywheel.ledger.raise_if_tripped()
            return None
        manifest = bundle.manifest()
        return {"checkpoint": checkpoint_name, "artifact_id": artifact_id, **text_free(manifest)}

    def steering_triggered(self, reviewed):
        from dataclasses import replace
        from jev_flywheel.loop import status
        from jev_flywheel.steering import SteeringPolicy, evaluate
        policy = SteeringPolicy("steering-v1")
        current = status(self.state.workspace, self.flywheel.corpus.score_name, policy)
        if self.arm != "L":
            return current.triggers["rethink"].fire
        state = self.state.workspace.steering_state(self.flywheel.corpus.score_name, current.n_effective)
        return evaluate(replace(state, commented_mismatches_since_rethink=self._plain_mismatches), policy)["rethink"].fire

    def heldout_from_bundle(self, bundle_reference):
        """Classify through a reloaded bundle, not the mutable state head."""
        if bundle_reference is None:
            return ()
        try:
            artifact_id = bundle_reference["artifact_id"]
            path = self._checkpoint_paths[artifact_id]
            from decision_flywheel.bundle import load_bundle
            from .unified_bundles import ScoreRubric
            bundle = load_bundle(path, self.flywheel.task, ScoreRubric.parse,
                                 configured_model=self.flywheel.cfg.provider_model)
            if bundle.bundle_hash != bundle_reference["bundle_hash"]:
                return ()
            done = self.flywheel.classify_with_bundle(bundle, self.flywheel.splits.paper600)
        except Exception:
            self.flywheel.ledger.raise_if_tripped()
            return ()
        rows = []
        for item_id in self.flywheel.splits.paper600:
            decision = done["results"].get(item_id)
            if decision is None:
                rows.append((item_id, self.flywheel.splits.items[item_id].reference_label,
                             None, {}, "unavailable"))
                continue
            probabilities = dict(getattr(decision, "probabilities", None) or {})
            rows.append((item_id, self.flywheel.splits.items[item_id].reference_label, decision.label,
                         probabilities, "ok" if decision.label in probabilities else "unavailable"))
        return rows

    def usage(self):
        """Only this arm's counted fake/provider activity, including token totals."""
        scoped = self.flywheel.ledger.summary().get("by_arm_round", {}).get(self.arm, {})
        counts = {"attempts": 0, "failures": 0, "input_tokens": 0, "output_tokens": 0}
        for kinds in scoped.values():
            for values in kinds.values():
                for key in counts:
                    counts[key] += int(values.get(key, 0))
        return counts

    def steer(self):
        from .unified_stream_workspace import bounded_steering_workspace

        authoritative = self.state.workspace
        window = tuple(self.reviewed[-150:])
        view = bounded_steering_workspace(authoritative, window,
                                          score_name=self.flywheel.corpus.score_name)
        # Stream arrival numbers are deliberately unrelated to analyst rounds:
        # recorded steering replies are numbered 1, 2, 3 regardless of how many
        # requests preceded the review trigger.
        steering_round = getattr(self, "_steering_round", 0) + 1
        try:
            self.state.workspace = view
            out = {"steering": self.flywheel._steer(self.arm, self.state, steering_round)}
        finally:
            self.state.workspace = authoritative
        self._steering_round = steering_round
        self.flywheel._cap_to_budget(self.arm, self.state, window, out)
        if self.state.example_list is None:
            self.state.head = self.state.workspace.scorecard().score(self.flywheel.corpus.score_name)
        else:
            # The workspace candidate was fit from zero-shot answers.  A fixed
            # example list is a different serving context, so keep its current
            # head until its own latest-window coverage/refit/gate promotes one.
            self._refit_fixed_list(window)
        if self.arm == "L":
            self._plain_mismatches = 0
        return True

def run_fake_fixture(document: Mapping[str, Any], run_dir: Path, *, seed: int = 1,
                     review_probability: float = .3, max_new_requests: Optional[int] = None) -> dict[str, Any]:
    """Run all fake arms from an explicit synthetic fixture and save text-free JSON.

    ``max_new_requests`` is accepted for CLI parity; fixture execution makes no requests.
    """
    del max_new_requests
    if document.get("synthetic") is not True:
        raise ValueError("offline stream fixtures must explicitly set synthetic: true")
    stream = [(str(row["id"]), str(row["label"])) for row in document["stream"]]
    heldout = list(document.get("heldout", ()))
    outputs = {}
    for arm in ("B", "L", "E", "X"):
        engine = _FixtureArm(document.get("labels") or (), (document.get("predictions") or {}).get(arm, {}), heldout)
        outputs[arm] = StreamDriver(StreamConfig(seed=seed, review_probability=review_probability)).run(
            arm, engine, stream, is_synthetic=True).to_dict()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"schema": SCHEMA, "is_synthetic": True, "labels": list(document.get("labels") or ()),
               "arms": outputs}
    (run_dir / "stream-results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
