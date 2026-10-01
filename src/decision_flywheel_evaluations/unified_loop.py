"""The unified flywheel loop: arms 0, A, B-local, B and A+B (studies/UNIFIED_FLYWHEEL_PLAN.md).

Import this only after ``unified_env.put_clone_first()``: it imports ``jev_flywheel`` from
the pinned clone (Workspace, AnswerCache, fit_head, compare, steer, ...) and needs
scikit-learn (fitting) and Tactus (steering) in the interpreter.

One round, for every arm, on the same fixed label order (plan section 2):

1. **Feedback.** The next ``per_round`` items of the seeded order; the scripted labeler gives
   the corpus reference label.
2. **Lever A (arms A, A+B).** One ``jev_flywheel.steer`` round on the arm's own workspace. The
   analyst may add an element; the host fits it out of fold and promotes it only if Brier
   improves without losing accuracy. Before the round, existing elements are topped up for
   the newly labeled items so the analyst and the gate see every label with full features.
3. **Lever B (arms B, A+B).** The labeled set is the pool. ``PerLabelLexicalRetrieval`` picks
   4 examples per label for each item; Jev answers the score question with
   ``state = {labeled_examples, target}`` through Decision-Flywheel's ``JevAdapter``. The
   answer is stored as element ``sentiment.fewshot`` under a context-aware hash
   (``AnswerCache.put_hashed``) and read back as feature ``fewshot.clr.positive``.
4. **Lever B-local (arm B-local).** ``knn.*`` features from unbalanced lexical neighbours
   (``unified_knn``), free.
5. **Refit and gate.** Every arm refits its head on all labels with ``fit_head`` (out-of-fold
   temperature) and promotes it over its incumbent only through ``jev_flywheel.fit.compare``.
   The candidate's out-of-fold metrics for the gate are *nested per fold*: each outer fold
   rebuilds every pool-dependent feature from the training fold alone (free for ``knn.*``),
   so B-local's leave-one-out features are not computed once over the whole labeled set. The
   few-shot answer cannot be rebuilt per fold without new requests, so it stays
   leave-one-out over the full labeled set (the plan's listed risk, recorded in the output).

Leakage rules: labels come from the pool only; retrieval and kNN pools are the labeled set
only; a target is excluded from its own context by ID and by normalized text. Scoring uses
dev-100 unless ``final`` is set, and paper-600 is never touched otherwise.

Outputs are text-free: metrics, intervals, request counts, fingerprints and element keys.
The analyst's raw replies (which can quote dataset text) are kept only in the run
directory's ``analyst-replies.jsonl`` under the gitignored ``var/``, for replay.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from decision_flywheel.adapters.jev import JevAdapter, JevConfiguration
from decision_flywheel.budget import ContextBudget, build_context_plan
from decision_flywheel.context import PerLabelLexicalRetrieval
from decision_flywheel.models import DecisionTask
from decision_flywheel.models import Item as DFItem
from decision_flywheel.models import LabeledItem
from jev_flywheel.answers import AnswerCache, import_answers_jsonl, question_hash
from jev_flywheel.evaluate import summarize as jev_summarize
from jev_flywheel.fit import TrainingSet, compare, fit_head, with_fit
from jev_flywheel.items import LABEL_SOURCE_FINAL, FeedbackItem, agrees
from jev_flywheel.items import Item as JevItem
from jev_flywheel.jev import JevSession, fingerprint
from jev_flywheel.ladder import LadderRefusal
from jev_flywheel.proposal import ProposalError, parse_proposal
from jev_flywheel.sampling import inverse_propensity_weights
from jev_flywheel.scorecard import ElementSpec, Score, Scorecard, build_question
from jev_flywheel.scoring import predict
from jev_flywheel.workspace import Workspace

from . import unified_env
from .unified_fake_jev import FAKE_MODEL, FakeJevAsync, FakeJevCore, FakeJevSync, text_key
from .unified_knn import (KNN_FEATURES, PoolEntry, context_fingerprint, knn_policy_fingerprint,
                          knn_rows, retrieval_policy_fingerprint)
from .unified_spend import (CountingAsyncClient, CountingSyncClient, SpendLedger,
                            concurrency_slots)
from .unified_splits import (LABELS, Splits, assert_labels_from_pool, assert_retrieval_pool_clean,
                             assert_target_excluded, fingerprint_ids, label_order, load_splits,
                             round_batches)
from .unified_stats import ItemResult, contrasts, summarize

HARNESS_VERSION = "unified-flywheel-1"
ARMS = ("0", "A", "B-local", "B", "A+B")
WORKSPACE_ARMS = frozenset({"A", "A+B"})
FEWSHOT_ARMS = frozenset({"B", "A+B"})
KNN_ARMS = frozenset({"B-local"})
SCORE_NAME = "Sentiment"
PLAN_REQUEST_CEILING = 9500
DEFAULT_ANALYST_PROVIDER = "openai"
DEFAULT_ANALYST_MODEL = "gpt-6-luna"
TASK = DecisionTask(SCORE_NAME, LABELS, "What is the overall sentiment of this text?")
FEWSHOT_KEY = "fewshot"
FEWSHOT_WIRE = "sentiment.fewshot"
FEWSHOT_FEATURE = "fewshot.clr.positive"
FEWSHOT_PER_LABEL = 4
DISPLAY_ORDER = "canonical"
FEWSHOT_QUESTION = build_question(question_type="choice", instructions=TASK.instructions,
                                  criteria={label: None for label in LABELS})

# Offline analyst replies. Round 1 replays the recorded Kimi-K3 reply that Jev-Flywheel's
# simulated-labeler recording promoted (it proposes `topic_domain`); later rounds use fixed
# fake replies so the loop exercises both an add and a no-op. None of these came from
# gpt-6-luna; the configured live analyst is recorded separately in the output.
FAKE_ROUND_REPLIES = {
    2: json.dumps({"root_cause": "Offline fake analyst: hedged wording may flip the label.",
                   "add_elements": [{"key": "hedged", "question_type": "noul",
                                     "instructions": "Does the text hedge or qualify its overall judgement?",
                                     "criteria": None}],
                   "retire_elements": [], "reword_elements": []}),
    3: json.dumps({"root_cause": "Offline fake analyst: no change.", "add_elements": [],
                   "retire_elements": [], "reword_elements": []}),
}


class HarnessError(RuntimeError):
    pass


# ---- configuration --------------------------------------------------------------------------

@dataclass
class RunConfig:
    clone: Path
    run_dir: Path
    seed: int = 1
    rounds: int = 3
    per_round: int = 100
    arms: Tuple[str, ...] = ARMS
    final: bool = False
    live: bool = False
    dev_size: int = 100
    request_ceiling: int = PLAN_REQUEST_CEILING
    max_new_requests: Optional[int] = None
    spend_ledger: Optional[Path] = None
    max_concurrency: int = 8
    max_consecutive_failures: int = 25
    analyst_provider: str = DEFAULT_ANALYST_PROVIDER
    analyst_model: str = DEFAULT_ANALYST_MODEL
    analyst_replies: Optional[Mapping[int, str]] = None
    provider_model: str = FAKE_MODEL
    base_url: Optional[str] = None
    timeout_seconds: float = 30.0
    knn_k: int = 8
    knn_top: int = 4
    folds: int = 5
    fit_seed: int = 0
    min_brier_gain: float = 0.005
    bootstrap_resamples: int = 1000
    bootstrap_seed: int = 0
    fake_signal_strength: float = 0.15

    def validate(self) -> None:
        unknown = set(self.arms) - set(ARMS)
        if unknown or not self.arms:
            raise HarnessError(f"unknown arms {sorted(unknown)}; choose from {ARMS}")
        if self.rounds < 1 or self.per_round < 1:
            raise HarnessError("rounds and per_round must be positive")
        if self.request_ceiling > PLAN_REQUEST_CEILING:
            raise HarnessError(f"the plan caps the study at {PLAN_REQUEST_CEILING} requests")
        if self.live and self.spend_ledger is None:
            raise HarnessError("live mode needs a durable --spend-ledger for the cumulative ceiling")


@dataclass
class Engines:
    zero_shot_factory: Callable[[], Any]
    fewshot_adapter: Any
    identity: str
    mode: str


def offline_engines(cfg: RunConfig, splits: Splits, ledger: SpendLedger,
                    core: Optional[FakeJevCore] = None) -> Engines:
    """The fake Jev behind the same counting clients a live run uses."""
    if core is None:
        planted = {text_key(item.text): item.reference_label for item in splits.items.values()}
        core = FakeJevCore(planted=planted, strength=cfg.fake_signal_strength)
    slots = concurrency_slots(cfg.max_concurrency)
    adapter = JevAdapter(CountingSyncClient(lambda: FakeJevSync(core), ledger, slots),
                         configuration=JevConfiguration(model=FAKE_MODEL))
    return Engines(lambda: CountingAsyncClient(lambda: FakeJevAsync(core), ledger, slots),
                   adapter, f"fake:{FAKE_MODEL}:signal-{core.strength}", "offline-fake")


def live_engines(cfg: RunConfig, ledger: SpendLedger) -> Engines:  # pragma: no cover - never run offline
    """Real Jev clients. Constructed lazily, inside the counting clients, after every gate.

    Retries are disabled in both SDK clients: every retry would be a billed request the
    ledger never saw. Credentials come from the environment (the gitignored ``.env`` via
    python-dotenv, as Decision-Flywheel's ``JevAdapter.from_environment`` does); nothing here
    reads, prints or stores them.
    """
    slots = concurrency_slots(cfg.max_concurrency)
    configuration = JevConfiguration(model=cfg.provider_model, base_url=cfg.base_url,
                                     timeout_seconds=cfg.timeout_seconds, max_retries=0)

    def zero_shot_inner():
        from dotenv import load_dotenv
        from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

        load_dotenv(override=False)
        return AsyncTypeSafeClient(model=cfg.provider_model, base_url=cfg.base_url,
                                   timeout=cfg.timeout_seconds, retry=RetryPolicy(max_retries=0))

    def fewshot_inner():
        return JevAdapter.from_environment(configuration=configuration).client

    adapter = JevAdapter(CountingSyncClient(fewshot_inner, ledger, slots), configuration=configuration)
    return Engines(lambda: CountingAsyncClient(zero_shot_inner, ledger, slots), adapter,
                   configuration.model_identity, "live")


# ---- heads and features ---------------------------------------------------------------------

def zero_shot_questions(score: Score) -> Dict[str, Dict[str, Any]]:
    """The questions Jev answers zero-shot for this score: everything but the few-shot element."""
    return {name: q for name, q in score.questions().items() if name != FEWSHOT_WIRE}


def uses_fewshot(score: Score) -> bool:
    return any(spec.key == FEWSHOT_KEY for spec in score.elements)


def uses_knn(score: Score) -> bool:
    return bool(score.decision) and any(f.startswith("knn.") for f in score.decision.features)


def candidate_template(base: Score, *, fewshot: bool, knn: bool) -> Score:
    """The arm's candidate head: the base score's features plus the arm's injected ones.

    Built from config without ``validate``: ``knn.*`` names do not refer to a Jev element, so
    a Jev-Flywheel scorecard would reject them. ``fit_head`` and ``predict`` never validate,
    which is why the injected features pass through them unchanged.
    """
    template = Score.from_config(base.to_config())
    features = [f for f in template.decision.features
                if not f.startswith("knn.") and f != FEWSHOT_FEATURE]
    if fewshot:
        template.elements = [e for e in template.elements if e.key != FEWSHOT_KEY] + [
            ElementSpec(FEWSHOT_KEY, "choice", TASK.instructions, {label: None for label in LABELS})]
        features.append(FEWSHOT_FEATURE)
    if knn:
        features.extend(KNN_FEATURES)
    decision = template.decision
    decision.features = features
    decision.model, decision.weights = "multinomial_logistic", {}
    decision.positive_class, decision.threshold = None, 0.0
    decision.calibration = decision.provenance = None
    return template


def head_from_fit(template: Score, result) -> Score:
    """``with_fit`` for one score, without the scorecard validation ``knn.*`` would fail."""
    head = Score.from_config(template.to_config())
    decision = head.decision
    decision.model = result.head["model"]
    decision.classes = list(result.head["classes"])
    decision.weights = result.head["weights"]
    decision.positive_class, decision.threshold = None, 0.0
    decision.calibration = result.calibration
    decision.provenance = result.provenance
    return head


def feature_row(score: Score, answers: Mapping[str, Any], injected: Optional[Mapping[str, float]]) -> Dict[str, float]:
    jev_features = [f for f in score.decision.features if not f.startswith("knn.")]
    shadow = replace(score, decision=replace(score.decision, features=jev_features))
    row = shadow.feature_vector(answers)
    for name in score.decision.features:
        if name.startswith("knn.") and injected and name in injected:
            row[name] = float(injected[name])
    return row


def training_set(score: Score, ids: Sequence[str], labels: Mapping[str, str],
                 rows: Mapping[str, Mapping[str, float]], context: str) -> TrainingSet:
    """What ``build_training_set`` assembles, from rows that may carry injected features."""
    training = TrainingSet(question_set_fingerprint=fingerprint(
        {"questions": score.questions(), "features": list(score.decision.features), "context": context}))
    for item_id in ids:
        row = rows.get(item_id) or {}
        if any(name not in row for name in score.decision.features):
            training.needs_answers.append(item_id)
            continue
        training.item_ids.append(item_id)
        training.rows.append(dict(row))
        training.labels.append(labels[item_id])
        training.cells.append(None)
    # A uniform, seeded label order: every propensity is equal, so the weights are all one.
    training.weights = inverse_propensity_weights([1.0] * len(training.labels))
    return training


def serve(score: Score, row: Mapping[str, float]) -> Tuple[str, float]:
    result = predict(score, {}, features=row)
    return result.value, float(result.confidence or 0.0)


def served_summary(score: Score, ids: Sequence[str], rows: Mapping[str, Mapping[str, float]],
                   labels: Mapping[str, str], weights: Sequence[float]):
    confidences, correct = [], []
    for item_id in ids:
        value, confidence = serve(score, rows.get(item_id) or {})
        confidences.append(confidence)
        correct.append(int(agrees(value, labels[item_id])))
    return jev_summarize(confidences, correct, list(weights))


# ---- the run --------------------------------------------------------------------------------

@dataclass
class ArmState:
    name: str
    head: Score
    workspace: Optional[Workspace] = None
    gates: List[Dict[str, Any]] = field(default_factory=list)


class UnifiedFlywheel:
    def __init__(self, cfg: RunConfig, *, engines: Optional[Engines] = None,
                 ledger: Optional[SpendLedger] = None, fake_core: Optional[FakeJevCore] = None):
        cfg.validate()
        self.cfg = cfg
        self.fixtures = Path(cfg.clone) / "fixtures"
        self.decision_flywheel = unified_env.ensure_decision_flywheel()
        self.splits = load_splits(self.fixtures, dev_size=cfg.dev_size)
        self.slice_name, self.eval_ids = self.splits.evaluation_slice(final=cfg.final)
        self.order = label_order(self.splits.pool, cfg.seed)
        self.batches = round_batches(self.order, rounds=cfg.rounds, per_round=cfg.per_round)
        self.labels = {i: self.splits.items[i].reference_label for i in self.splits.pool}
        self.ledger = ledger or SpendLedger(cfg.spend_ledger, cfg.request_ceiling, max_new=cfg.max_new_requests,
                                            max_consecutive_failures=cfg.max_consecutive_failures,
                                            run_label=f"seed{cfg.seed}")
        if cfg.live:
            if engines is None:
                engines = live_engines(cfg, self.ledger)
        else:
            if not unified_env.network_guard_installed():
                raise HarnessError("offline runs must install the network guard first")
            engines = engines or offline_engines(cfg, self.splits, self.ledger, fake_core)
        self.engines = engines
        self.run_dir = Path(cfg.run_dir)
        self._prepare_run_dir()
        self.cache = self._shared_cache()
        self.fewshot_cache = AnswerCache(self.run_dir / "fewshot-answers.jsonl")
        self.v1 = Scorecard.from_yaml((self.fixtures / "scorecards" / "v1.yaml").read_text())
        self.events: List[Dict[str, Any]] = []
        self.results: Dict[int, Dict[str, List[ItemResult]]] = {}
        self.fewshot_diagnostics: Dict[int, Dict[str, Any]] = {}

    # ---- setup ---------------------------------------------------------------------------

    def _prepare_run_dir(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.run_dir / "run-manifest.json"
        identity = {"harness": HARNESS_VERSION, "mode": self.engines.mode, "engine": self.engines.identity,
                    "jev_flywheel_commit": unified_env.JEV_FLYWHEEL_COMMIT}
        if manifest.exists():
            stored = json.loads(manifest.read_text())
            if stored != identity:
                raise HarnessError("this run directory holds answers from a different engine or mode; "
                                   "use a fresh --run-dir rather than mixing fake and real answers")
        else:
            manifest.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")

    def _shared_cache(self) -> AnswerCache:
        path = self.run_dir / "answers.jsonl"
        if not path.exists():
            reference = Scorecard.from_yaml((self.fixtures / "scorecards" / "reference_full.yaml").read_text())
            staging = AnswerCache(path.with_suffix(".staging"))
            import_answers_jsonl(self.fixtures / "answers.jsonl.gz", reference.questions(), staging)
            os.replace(path.with_suffix(".staging"), path)
        return AnswerCache(path)

    def _workspace(self, arm: str) -> Workspace:
        root = self.run_dir / "workspaces" / arm.replace("+", "plus")
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        shutil.copyfile(self.fixtures / "items.jsonl", root / "items.jsonl")
        (root / "workspace.json").write_text(json.dumps({"engine": "jev", "answers": "shared"}) + "\n")
        (root / "answers.jsonl").symlink_to(self.run_dir / "answers.jsonl")
        workspace = Workspace(root)
        # One AnswerCache object for every arm: answers are keyed per question body, so arms
        # that ask the same question share one answer (and one request) instead of two.
        workspace._cache = self.cache
        seed = Scorecard.from_yaml((self.fixtures / "scorecards" / "v1.yaml").read_text())
        workspace.commit_scorecard(seed, kind="seed", provenance={"source": "fixtures/v1.yaml"})
        return workspace

    # ---- answers and rows ----------------------------------------------------------------

    def _jev_item(self, item_id: str) -> JevItem:
        return JevItem(id=item_id, text=self.splits.items[item_id].text)

    def _fill_zero_shot(self, item_ids: Sequence[str], questions: Mapping[str, Any]) -> Dict[str, int]:
        plan = self.cache.plan(item_ids, questions)
        if not plan.requests:
            return {"requests": 0, "failures": 0}
        session = JevSession(client_factory=self.engines.zero_shot_factory)
        items = [self._jev_item(i) for i in plan.items_needing]
        report = asyncio.run(self.cache.fill(session, items, questions, concurrency=self.cfg.max_concurrency))
        self.ledger.raise_if_tripped()
        return {"requests": report.requested + report.failures, "failures": report.failures}

    def _fewshot_context(self, labeled: Sequence[str]) -> str:
        return context_fingerprint(retrieval_policy_fingerprint(), labeled, per_label=FEWSHOT_PER_LABEL,
                                   display_order=DISPLAY_ORDER, task=TASK.fingerprint,
                                   engine=self.engines.identity)

    @staticmethod
    def _fewshot_key(context: str) -> Dict[str, Any]:
        return {**FEWSHOT_QUESTION, "context_fingerprint": context}

    def _ensure_fewshot(self, targets: Sequence[str], labeled: Sequence[str]) -> Dict[str, int]:
        assert_labels_from_pool(labeled, self.splits)
        assert_retrieval_pool_clean(labeled, self.splits)
        context = self._fewshot_context(labeled)
        key = self._fewshot_key(context)
        qhash = question_hash(key)
        missing = [t for t in targets if self.fewshot_cache.get(t, FEWSHOT_WIRE, key) is None]
        if not missing:
            return {"requests": 0, "failures": 0}
        candidates = [LabeledItem(DFItem(i, {"text": self.splits.items[i].text}), self.labels[i])
                      for i in sorted(labeled)]
        policy = PerLabelLexicalRetrieval()
        failures = 0

        async def one(target_id: str) -> None:
            nonlocal failures
            target = DFItem(target_id, {"text": self.splits.items[target_id].text})
            plan = build_context_plan(TASK, target, candidates, policy,
                                      budget=ContextBudget(per_label=FEWSHOT_PER_LABEL),
                                      display_order=DISPLAY_ORDER, order_seed=0,
                                      presentation_label_order=TASK.labels)
            assert_target_excluded(target_id, plan.example_ids)
            assert_retrieval_pool_clean(plan.example_ids, self.splits)
            try:
                result = await self.engines.fewshot_adapter.decide(TASK, target, plan.examples)
            except Exception:  # noqa: BLE001 - counted by the ledger; a gap is reported, not fatal
                failures += 1
                return
            probabilities = dict(result.probabilities or {})
            answer = {"type": "choice", "choice": result.label, "probabilities": probabilities,
                      "confidence": result.confidence if result.confidence is not None
                      else probabilities.get(result.label)}
            self.fewshot_cache.put_hashed(target_id, FEWSHOT_WIRE, qhash, answer, result.model)

        async def all_of_them() -> None:
            await asyncio.gather(*(one(t) for t in missing))

        asyncio.run(all_of_them())
        self.ledger.raise_if_tripped()
        return {"requests": len(missing), "failures": failures}

    def _answers(self, ids: Sequence[str], score: Score, labeled: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        answers = self.cache.bulk_partial_answers(ids, zero_shot_questions(score))
        if uses_fewshot(score):
            key = self._fewshot_key(self._fewshot_context(labeled))
            for item_id in ids:
                found = self.fewshot_cache.get(item_id, FEWSHOT_WIRE, key)
                if found is not None:
                    answers[item_id][FEWSHOT_WIRE] = found
        return answers

    def _knn(self, targets: Sequence[str], pool_ids: Sequence[str]) -> Dict[str, Dict[str, float]]:
        assert_retrieval_pool_clean(pool_ids, self.splits)
        pool = [PoolEntry(i, self.splits.items[i].text, self.labels[i]) for i in sorted(pool_ids)]
        return knn_rows({t: self.splits.items[t].text for t in targets}, pool,
                        k=self.cfg.knn_k, top=self.cfg.knn_top)

    def _rows(self, score: Score, ids: Sequence[str], labeled: Sequence[str],
              knn_pool: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, float]]:
        """Feature rows; ``knn_pool`` (default: every label) is where neighbours come from."""
        answers = self._answers(ids, score, labeled)
        injected = self._knn(ids, knn_pool if knn_pool is not None else labeled) if uses_knn(score) else {}
        return {i: feature_row(score, answers[i], injected.get(i)) for i in ids}

    # ---- the gate ------------------------------------------------------------------------

    def _nested_oof(self, template: Score, training: TrainingSet, labeled: Sequence[str]):
        """Out-of-fold metrics where pool-dependent features see only the training fold."""
        import numpy as np
        from sklearn.model_selection import StratifiedKFold

        ids, y = list(training.item_ids), list(training.labels)
        counts = {label: y.count(label) for label in set(y)}
        splitter = StratifiedKFold(n_splits=min(self.cfg.folds, min(counts.values())), shuffle=True,
                                   random_state=self.cfg.fit_seed)
        confidences, correct = [0.0] * len(ids), [0] * len(ids)
        for train_index, test_index in splitter.split(np.zeros(len(ids)), y):
            train_ids = [ids[i] for i in train_index]
            test_ids = [ids[i] for i in test_index]
            rows = self._rows(template, train_ids + test_ids, labeled, knn_pool=train_ids)
            fold = training_set(template, train_ids, self.labels, rows, "nested-fold")
            result = fit_head(fold, template, folds=self.cfg.folds, seed=self.cfg.fit_seed)
            if not result.fitted:
                return None, result.reason
            head = head_from_fit(template, result)
            for index, item_id in zip(test_index, test_ids):
                value, confidence = serve(head, rows[item_id])
                confidences[index] = confidence
                correct[index] = int(agrees(value, self.labels[item_id]))
        return jev_summarize(confidences, correct, list(training.weights)), None

    def _gate(self, arm: str, round_number: int, template: Score, incumbent: Score,
              labeled: Sequence[str]) -> Tuple[Score, Dict[str, Any], Optional[Any]]:
        context = self._fewshot_context(labeled) if uses_fewshot(template) else "zero-shot"
        rows = self._rows(template, labeled, labeled)
        training = training_set(template, labeled, self.labels, rows, context)
        record: Dict[str, Any] = {"arm": arm, "round": round_number, "features": list(template.decision.features),
                                  "n_train": training.n, "missing_features": len(training.needs_answers)}
        try:
            result = fit_head(training, template, folds=self.cfg.folds, seed=self.cfg.fit_seed)
        except LadderRefusal as refusal:
            return incumbent, {**record, "decision": "refused", "reasons": [str(refusal)]}, None
        if not result.fitted:
            return incumbent, {**record, "decision": "held", "reasons": [result.reason]}, None
        try:
            nested, problem = self._nested_oof(template, training, labeled)
        except LadderRefusal as refusal:
            nested, problem = None, str(refusal)
        if nested is None:
            return incumbent, {**record, "decision": "rejected", "reasons": [f"nested out-of-fold fit failed: {problem}"]}, None
        incumbent_rows = self._rows(incumbent, training.item_ids, labeled)
        incumbent_summary = served_summary(incumbent, training.item_ids, incumbent_rows, self.labels, training.weights)
        comparison = compare(replace(result, metrics=nested), incumbent_summary,
                             min_brier_gain=self.cfg.min_brier_gain)
        record.update({
            "decision": "promoted" if comparison.promote else "rejected",
            "reasons": comparison.reasons, "fit_id": result.provenance.get("fit_id"),
            "regularization_c": result.chosen_c, "calibration": (result.calibration or {}).get("method"),
            "candidate_nested_oof": _summary_dict(nested),
            "candidate_fit_head_oof": _summary_dict(result.metrics),
            "incumbent_served": _summary_dict(incumbent_summary),
            "pool_features_per_fold": uses_knn(template),
            "fewshot_features_leave_one_out_once": uses_fewshot(template),
        })
        if comparison.promote:
            return head_from_fit(template, result), record, result
        return incumbent, record, None

    # ---- lever A -------------------------------------------------------------------------

    def _record_feedback(self, workspace: Workspace, batch: Sequence[str], round_number: int) -> None:
        card = workspace.scorecard()
        score = card.score(SCORE_NAME)
        questions = card.questions()
        for item_id in batch:
            result = predict(score, self.cache.partial_answers_for(item_id, questions))
            label = self.labels[item_id]
            workspace.add_feedback(FeedbackItem(
                id=f"fb-{item_id}-s{self.cfg.seed}-r{round_number}", item_id=item_id, score_name=SCORE_NAME,
                initial_answer_value=result.value, final_answer_value=label,
                is_agreement=agrees(result.value, label), editor_name="scripted-labeler",
                cache_key=f"{item_id}:{SCORE_NAME}:v{workspace.version}", label_source=LABEL_SOURCE_FINAL,
                metadata={"propensity": 1.0, "selection_policy": "seeded-uniform-order",
                          "shown_confidence": result.confidence, "round": round_number}))

    def _replies_path(self) -> Path:
        return self.run_dir / "analyst-replies.jsonl"

    def _recorded_reply(self, arm: str, round_number: int) -> Optional[str]:
        path = self._replies_path()
        if not path.exists():
            return None
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if (row["arm"], row["seed"], row["round"]) == (arm, self.cfg.seed, round_number):
                return row["reply"]
        return None

    def _offline_reply(self, round_number: int) -> str:
        if self.cfg.analyst_replies and round_number in self.cfg.analyst_replies:
            return self.cfg.analyst_replies[round_number]
        if round_number == 1:
            script = json.loads((self.fixtures / "recordings" / "simulated-labeler" / "script.json").read_text())
            return next(step["analyst_reply"] for step in script["steps"] if step["op"] == "steer")
        return FAKE_ROUND_REPLIES.get(round_number, FAKE_ROUND_REPLIES[3])

    def _steer(self, arm: str, state: ArmState, round_number: int) -> Dict[str, Any]:
        from jev_flywheel.steer import ScriptedApprover, run_steering

        workspace = state.workspace
        if self.cfg.live:
            reply = self._recorded_reply(arm, round_number)
            source = "replayed" if reply is not None else "live"
        else:
            reply, source = self._offline_reply(round_number), "offline-fixed"
        before = workspace.scorecard().score(SCORE_NAME)
        if self.cfg.analyst_provider == "openai":
            from .unified_openai import allow_gpt6_token_parameter
            allow_gpt6_token_parameter()
        outcome = run_steering(
            workspace, SCORE_NAME, provider=self.cfg.analyst_provider, model=self.cfg.analyst_model,
            allow_spend=True, client_factory=self.engines.zero_shot_factory,
            # The scripted human approves what the gate already passed; spend is bounded by
            # the ledger's ceiling, not by the procedure's price prompt.
            hitl_handler=ScriptedApprover((), default=True),
            mock_replies=None if reply is None else [reply, reply],
            max_auto_requests=10 ** 6, max_revisions=1)
        self.ledger.raise_if_tripped()
        raw = outcome.analyst_reply if reply is None else reply
        if self.cfg.live and source == "live" and raw:
            with self._replies_path().open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"arm": arm, "seed": self.cfg.seed, "round": round_number,
                                         "provider": self.cfg.analyst_provider, "model": self.cfg.analyst_model,
                                         "reply": raw}) + "\n")
        after = workspace.scorecard().score(SCORE_NAME)
        return {"decision": outcome.decision, "analyst_source": source,
                "analyst_reply_sha256": hashlib.sha256((raw or "").encode()).hexdigest(),
                "proposed": _proposed_elements(raw),
                "elements_before": [e.key for e in before.elements],
                "elements_after": [e.key for e in after.elements],
                "scorecard_version": workspace.version}

    # ---- one round -----------------------------------------------------------------------

    def _step(self, arm: str, round_number: int, kind: str, name: str, fn: Callable[[], Any]) -> Any:
        self.ledger.set_scope(arm, round_number, kind)
        used = self.ledger.used
        out = fn()
        self.events.append({"round": round_number, "arm": arm, "step": name, "kind": kind,
                            "requests": self.ledger.used - used})
        return out

    def run(self) -> Dict[str, Any]:
        arms = [a for a in ARMS if a in self.cfg.arms]   # B before A+B: A+B reuses B's few-shot cache
        states: Dict[str, ArmState] = {}
        v1_score = self.v1.score(SCORE_NAME)
        for arm in arms:
            workspace = self._workspace(arm) if arm in WORKSPACE_ARMS else None
            states[arm] = ArmState(arm, Score.from_config(v1_score.to_config()), workspace)
        labeled: List[str] = []
        rounds_out = []
        for round_number, batch in enumerate(self.batches, start=1):
            labeled = labeled + list(batch)
            assert_labels_from_pool(labeled, self.splits)
            assert_retrieval_pool_clean(labeled, self.splits)
            round_record: Dict[str, Any] = {"round": round_number, "n_labeled": len(labeled),
                                            "labeled_fingerprint": fingerprint_ids(labeled),
                                            "fewshot_context": self._fewshot_context(labeled),
                                            "knn_context": context_fingerprint(knn_policy_fingerprint(self.cfg.knn_k, self.cfg.knn_top), labeled),
                                            "arms": {}}
            self.results[round_number] = {}
            for arm in arms:
                round_record["arms"][arm] = self._run_arm(arm, states[arm], round_number, batch, labeled)
            round_record["contrasts"] = contrasts(self.results[round_number], resamples=self.cfg.bootstrap_resamples,
                                                  seed=self.cfg.bootstrap_seed)
            if round_number in self.fewshot_diagnostics:
                round_record["fewshot_vs_zero_shot"] = self.fewshot_diagnostics[round_number]
            rounds_out.append(round_record)
        return self._finish(rounds_out)

    def _run_arm(self, arm: str, state: ArmState, round_number: int, batch: Sequence[str],
                 labeled: Sequence[str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if arm in WORKSPACE_ARMS:
            workspace = state.workspace
            existing = workspace.scorecard().questions()
            self._step(arm, round_number, "zero-shot", "top-up-new-labels",
                       lambda: self._fill_zero_shot(batch, existing))
            self._record_feedback(workspace, batch, round_number)
            out["steering"] = self._step(arm, round_number, "zero-shot", "steer",
                                         lambda: self._steer(arm, state, round_number))
            ws_score = workspace.scorecard().score(SCORE_NAME)
            ws_head, ws_gate, ws_fit = self._gate(arm, round_number, candidate_template(ws_score, fewshot=False, knn=False),
                                                  ws_score, labeled)
            if ws_fit is not None:
                workspace.commit_scorecard(with_fit(workspace.scorecard(), SCORE_NAME, ws_fit), kind="fit",
                                           provenance=ws_fit.provenance)
            out["workspace_gate"] = ws_gate
            ws_score = workspace.scorecard().score(SCORE_NAME)
            self._step(arm, round_number, "zero-shot", "fill-evaluation-slice",
                       lambda: self._fill_zero_shot(self.eval_ids, workspace.scorecard().questions()))
            if arm == "A":
                state.head = ws_score
        if arm in FEWSHOT_ARMS:
            out["fewshot_requests"] = self._step(
                arm, round_number, "few-shot", "few-shot-answers",
                lambda: self._ensure_fewshot(list(labeled) + list(self.eval_ids), labeled))
            if round_number not in self.fewshot_diagnostics:
                self.fewshot_diagnostics[round_number] = self._fewshot_diagnostic(labeled)
        if arm != "A":
            base = state.workspace.scorecard().score(SCORE_NAME) if state.workspace else self.v1.score(SCORE_NAME)
            template = candidate_template(base, fewshot=arm in FEWSHOT_ARMS, knn=arm in KNN_ARMS)
            state.head, gate, _ = self._gate(arm, round_number, template, state.head, labeled)
            out["gate"] = gate
        out["head_features"] = list(state.head.decision.features)
        results, coverage = self._score(state.head, labeled)
        self.results[round_number][arm] = results
        out["metrics"] = summarize(results)
        out["evaluation_rows_missing_features"] = coverage
        self.ledger.raise_if_tripped()
        return out

    def _score(self, head: Score, labeled: Sequence[str]) -> Tuple[List[ItemResult], int]:
        rows = self._rows(head, self.eval_ids, labeled)
        results, missing = [], 0
        for item_id in self.eval_ids:
            row = rows[item_id]
            if any(name not in row for name in head.decision.features):
                missing += 1
            value, confidence = serve(head, row)
            correct = int(agrees(value, self.splits.items[item_id].reference_label))
            results.append(ItemResult(item_id, confidence, correct))
        return results, missing

    def _fewshot_diagnostic(self, labeled: Sequence[str]) -> Dict[str, Any]:
        """D0's check that few-shot answers differ from zero-shot: text-free rates only."""
        key = self._fewshot_key(self._fewshot_context(labeled))
        holistic = self.v1.questions()[SCORE_NAME]
        changed, gaps, n = 0, [], 0
        for item_id in list(labeled) + list(self.eval_ids):
            few = self.fewshot_cache.get(item_id, FEWSHOT_WIRE, key)
            zero = self.cache.get(item_id, SCORE_NAME, holistic)
            if few is None or zero is None:
                continue
            n += 1
            changed += int(few.get("choice") != zero.get("choice"))
            gaps.append(abs(float((few.get("probabilities") or {}).get(LABELS[0], 0.0))
                            - float((zero.get("probabilities") or {}).get(LABELS[0], 0.0))))
        return {"n": n, "choice_differs_rate": round(changed / n, 4) if n else None,
                "mean_abs_p_positive_difference": round(sum(gaps) / n, 4) if n else None}

    def _finish(self, rounds_out: List[Dict[str, Any]]) -> Dict[str, Any]:
        summary = {
            "harness": HARNESS_VERSION,
            "jev_flywheel": {"clone": "var/" + Path(self.cfg.clone).name, "commit": unified_env.JEV_FLYWHEEL_COMMIT},
            "mode": self.engines.mode, "engine": self.engines.identity,
            "analyst": {"provider": self.cfg.analyst_provider, "model": self.cfg.analyst_model,
                        "replies": "live (recorded for replay)" if self.cfg.live else
                        "offline: round 1 = recorded Kimi-K3 reply from the clone's simulated-labeler recording; later rounds = fixed fake replies"},
            "seed": self.cfg.seed, "rounds": self.cfg.rounds, "per_round": self.cfg.per_round,
            "arms": [a for a in ARMS if a in self.cfg.arms],
            "evaluation_slice": {"name": self.slice_name, "n": len(self.eval_ids),
                                 "fingerprint": fingerprint_ids(self.eval_ids)},
            "label_order_sha256": hashlib.sha256(
                json.dumps(list(self.order[:self.cfg.rounds * self.cfg.per_round])).encode()).hexdigest(),
            "decision_flywheel": self.decision_flywheel,
            "policies": {"fewshot": {"policy": "per-label-lexical-retrieval", "fingerprint": retrieval_policy_fingerprint(),
                                     "per_label": FEWSHOT_PER_LABEL, "display_order": DISPLAY_ORDER},
                         "knn": {"fingerprint": knn_policy_fingerprint(self.cfg.knn_k, self.cfg.knn_top),
                                 "k": self.cfg.knn_k, "top": self.cfg.knn_top},
                         "gate": {"min_brier_gain": self.cfg.min_brier_gain, "folds": self.cfg.folds,
                                  "fit_seed": self.cfg.fit_seed, "candidate_metrics": "nested per-fold out-of-fold"},
                         "bootstrap": {"resamples": self.cfg.bootstrap_resamples, "seed": self.cfg.bootstrap_seed}},
            "per_round": rounds_out,
            "requests": self.ledger.summary(),
        }
        _assert_text_free(summary, self.splits)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        (self.run_dir / "run-log.json").write_text(json.dumps(
            {"events": self.events, "requests": self.ledger.summary()}, indent=2, sort_keys=True) + "\n")
        with (self.run_dir / "predictions.jsonl").open("w") as handle:
            for round_number, by_arm in sorted(self.results.items()):
                for arm, rows in by_arm.items():
                    for row in rows:
                        handle.write(json.dumps({"round": round_number, "arm": arm, "item_id": row.item_id,
                                                 "confidence": round(row.confidence, 6), "correct": row.correct}) + "\n")
        return summary


def _summary_dict(summary) -> Optional[Dict[str, float]]:
    if summary is None:
        return None
    return {"n": summary.n, "accuracy": round(summary.accuracy, 6), "brier": round(summary.brier, 6),
            "ece": round(summary.ece, 6)}


def _proposed_elements(reply: Optional[str]) -> List[Dict[str, Any]]:
    """Text-free: element keys and types, plus a hash of each question's wording."""
    if not reply:
        return []
    try:
        proposal = parse_proposal(reply)
    except ProposalError:
        return [{"invalid": True}]
    return [{"key": a.key, "question_type": a.question_type,
             "instructions_sha256": hashlib.sha256(a.instructions.encode()).hexdigest()}
            for a in proposal.add] + [{"retire": key} for key in proposal.retire] + [
        {"reword": r.key} for r in proposal.reword]


def _assert_text_free(summary: Mapping[str, Any], splits: Splits) -> None:
    """Refuse to write an output that contains any corpus text."""
    blob = json.dumps(summary)
    for item in splits.items.values():
        if len(item.text) >= 24 and item.text in blob:
            raise HarnessError("refusing to write an output that contains dataset text")
