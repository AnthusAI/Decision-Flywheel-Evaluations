"""The unified flywheel loop: arms 0, A, B-local, B and A+B (studies/UNIFIED_FLYWHEEL_PLAN.md),
plus the fixed-list arms F, F-rand, A-c and A-c+F (studies/DECISION_FLYWHEEL_DESIGN.md).

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

**Fixed-list arms** (F, F-rand, A-c+F) follow the owner's round, in the simplest request shape
(one Jev request per item with the example list in the state and every rubric question; see
``unified_lists``):

1. predict the new labels with the served classifier (round 1: Jev zero-shot), and take the
   feedback; the confidently wrong items are the hard demos;
2. A-c+F only: one steering round, with the labeler's comments in the feedback (A-c as well);
3. F and A-c+F: ``improve_example_list`` tries the incumbent, a hard-demo swap and a random
   control against the labels so far and keeps the incumbent on ties; F-rand redraws a
   random list each round instead;
4. answer every label and the evaluation slice with the chosen list and refit the head on
   those answers (no separate few-shot feature, no extra gate).

A-c is arm A with the labeler's comments recorded in the feedback. Until an explanation
labeler supplies comments (``RunConfig.comments``, see ``unified_labeler``), A-c behaves exactly
like A. Each steering record notes how many comments the arm's feedback carried and the hashes of
those the analyst's reply quoted.

**Bundles.** At the end of every run the arms 0, A, A-c, F, F-rand and A-c+F are frozen as core
``ClassifierBundle`` directories (``run_dir/bundles/<arm>/round-<n>``; private, under ``var/``) and
the summary records their hashes. ``final=True`` runs no rounds: it reloads the bundles of the
last round from the run directory and scores paper-600 *through them* (one request per item,
answered from cache when possible), so the held-out slice never informs any choice.

**Arm D (optional dynamic retrieval, final only).** ``--final --arms D --retriever <variant>`` runs
``unified_retrieval.run_final_d``: A-c's frozen rubric questions, each item's own retrieved examples
from the run's labels (leave-one-out for labeled items) in one request per item, a refit head, and
paper-600 scored from those answers. Its outputs go to ``run_dir/final-d/<variant>/``; it is not bundled.

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
from decision_flywheel.bundle import ClassifierBundle, load_bundle
from decision_flywheel.budget import ContextBudget, build_context_plan
from decision_flywheel.context import FixedExampleList, PerLabelLexicalRetrieval
from decision_flywheel.example_list import improve_example_list, plan_example_list_round
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
from .unified_bundles import BUNDLE_ARMS, CachedJevClient, ScoreRubric, bundle_dir, text_free
from .unified_fake_jev import FAKE_MODEL, FakeJevAsync, FakeJevCore, FakeJevSync, text_key
from .unified_lists import ListAnswers, ListModel, random_list
from .unified_retrieval import D_ARM, RETRIEVERS
from .unified_knn import (PoolEntry, context_fingerprint, knn_policy_fingerprint,
                          knn_rows, retrieval_policy_fingerprint)
from .unified_spend import (CountingAsyncClient, CountingSyncClient, SpendLedger,
                            concurrency_slots)
from .unified_corpus import PLANTED, Corpus
from .unified_splits import (Splits, assert_labels_from_pool, assert_retrieval_pool_clean,
                             assert_target_excluded, fingerprint_ids, label_order, load_splits,
                             round_batches)
from .unified_stats import ItemResult, contrasts, summarize

HARNESS_VERSION = "unified-flywheel-1"
UNIFIED_ARMS = ("0", "A", "B-local", "B", "A+B")     # the first study; still the default
ARMS = UNIFIED_ARMS + ("F", "F-rand", "A-c", "A-c+F")
FINAL_ONLY_ARMS = (D_ARM,)   # optional dynamic retrieval (unified_retrieval); never part of the rounds
F_RAND_SEED_OFFSET = 1000   # F-rand draws independently of F's own random-control trial
WORKSPACE_ARMS = frozenset({"A", "A+B", "A-c", "A-c+F"})
COMMENT_ARMS = frozenset({"A-c", "A-c+F"})
LIST_ARMS = frozenset({"F", "F-rand", "A-c+F"})
FEWSHOT_ARMS = frozenset({"B", "A+B"})
KNN_ARMS = frozenset({"B-local"})
# The planted corpus's values, kept under their old names for callers and specs that predate
# ``unified_corpus``; a run reads its own ``cfg.corpus`` instead.
SCORE_NAME = PLANTED.score_name
PLAN_REQUEST_CEILING = 9500
DEFAULT_ANALYST_PROVIDER = "openai"
DEFAULT_ANALYST_MODEL = "gpt-6-luna"
TASK = PLANTED.task()
FEWSHOT_KEY = PLANTED.fewshot_key
FEWSHOT_WIRE = PLANTED.fewshot_wire
FEWSHOT_FEATURE = PLANTED.fewshot_feature
FEWSHOT_PER_LABEL = 4
DISPLAY_ORDER = "canonical"
FEWSHOT_QUESTION = build_question(question_type="choice", instructions=TASK.instructions,
                                  criteria={label: None for label in PLANTED.labels})

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
    arms: Tuple[str, ...] = UNIFIED_ARMS
    final: bool = False
    live: bool = False
    replay: bool = False
    dev_size: int = 100
    corpus: Corpus = PLANTED
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
    comments: Optional[Mapping[str, str]] = None   # item_id -> the labeler's explanation (A-c arms)
    list_per_label: int = 4
    list_dev_max: Optional[int] = None             # None: every label outside the trial lists
    retriever: Optional[str] = None                # arm D only: one of unified_retrieval.RETRIEVERS
    embedding_cache: Optional[Path] = None         # arm D embedding: default run_dir/embeddings.jsonl

    def validate(self) -> None:
        self.corpus.require_ready("a flywheel run")
        unknown = set(self.arms) - set(ARMS + FINAL_ONLY_ARMS)
        if unknown or not self.arms:
            raise HarnessError(f"unknown arms {sorted(unknown)}; choose from {ARMS + FINAL_ONLY_ARMS}")
        if D_ARM in self.arms:
            if not self.final or tuple(self.arms) != (D_ARM,):
                raise HarnessError("arm D is final-only and runs alone: --final --arms D after the rounds exist")
            if self.retriever not in RETRIEVERS:
                raise HarnessError(f"arm D needs a retriever from {RETRIEVERS}")
        elif self.retriever is not None or self.embedding_cache is not None:
            raise HarnessError("a retriever applies only to arm D")
        if self.rounds < 1 or self.per_round < 1:
            raise HarnessError("rounds and per_round must be positive")
        if self.request_ceiling > PLAN_REQUEST_CEILING:
            raise HarnessError(f"the plan caps the study at {PLAN_REQUEST_CEILING} requests")
        if self.live and self.spend_ledger is None:
            raise HarnessError("live mode needs a durable --spend-ledger for the cumulative ceiling")
        if self.replay and self.live:
            raise HarnessError("replay is offline: it may not be combined with live mode")
        if self.final and set(self.arms) - set(BUNDLE_ARMS) - {D_ARM}:
            raise HarnessError(f"the final run scores frozen bundles; choose arms from {BUNDLE_ARMS}")


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
        core = FakeJevCore(planted=planted, positive_label=cfg.corpus.fake_jev_positive_label,
                           strength=cfg.fake_signal_strength, cues=dict(cfg.corpus.fake_jev_cues))
    slots = concurrency_slots(cfg.max_concurrency)
    adapter = JevAdapter(CountingSyncClient(lambda: FakeJevSync(core), ledger, slots),
                         configuration=JevConfiguration(model=FAKE_MODEL))
    return Engines(lambda: CountingAsyncClient(lambda: FakeJevAsync(core), ledger, slots),
                   adapter, f"fake:{FAKE_MODEL}:signal-{core.strength}", "offline-fake")


class ReplayMiss(RuntimeError):
    """A replay needed an answer that is not cached. Nothing was sent."""


class _RefusingClient:
    def system_one(self, *, state, questions):
        raise ReplayMiss("replay needed an uncached Jev answer")


class _RefusingAsyncClient:
    async def system_one(self, *, state, questions):
        raise ReplayMiss("replay needed an uncached Jev answer")


def replay_engines(cfg: RunConfig, ledger: SpendLedger) -> Engines:
    """The live engine's identity, but clients that refuse: every answer must come from cache.

    The ledger still counts each refused attempt, so a run that needed anything uncached
    reports it (``UnifiedFlywheel.run`` then fails) instead of silently scoring gaps.
    """
    slots = concurrency_slots(cfg.max_concurrency)
    configuration = JevConfiguration(model=cfg.provider_model)
    adapter = JevAdapter(CountingSyncClient(_RefusingClient, ledger, slots), configuration=configuration)
    return Engines(lambda: CountingAsyncClient(_RefusingAsyncClient, ledger, slots), adapter,
                   configuration.model_identity, "live")


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

def zero_shot_questions(score: Score, corpus: Corpus = PLANTED) -> Dict[str, Dict[str, Any]]:
    """The questions Jev answers zero-shot for this score: everything but the few-shot element."""
    return {name: q for name, q in score.questions().items() if name != corpus.fewshot_wire}


def uses_fewshot(score: Score, corpus: Corpus = PLANTED) -> bool:
    return any(spec.key == corpus.fewshot_key for spec in score.elements)


def uses_knn(score: Score) -> bool:
    return bool(score.decision) and any(f.startswith("knn.") for f in score.decision.features)


def candidate_template(base: Score, *, fewshot: bool, knn: bool, corpus: Corpus = PLANTED) -> Score:
    """The arm's candidate head: the base score's features plus the arm's injected ones.

    Built from config without ``validate``: ``knn.*`` names do not refer to a Jev element, so
    a Jev-Flywheel scorecard would reject them. ``fit_head`` and ``predict`` never validate,
    which is why the injected features pass through them unchanged.
    """
    template = Score.from_config(base.to_config())
    features = [f for f in template.decision.features
                if not f.startswith("knn.") and f not in corpus.fewshot_features]
    if fewshot:
        template.elements = [e for e in template.elements if e.key != corpus.fewshot_key] + [
            ElementSpec(corpus.fewshot_key, "choice", corpus.instructions, {label: None for label in corpus.labels})]
        features.extend(corpus.fewshot_features)
    if knn:
        features.extend(corpus.knn_features)
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

def fewshot_diagnostic_summary(pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]],
                               labels: Sequence[str]) -> Dict[str, Any]:
    """Text-free rates comparing few-shot with zero-shot answers, ``pairs`` being ``(few, zero)`` answers.

    Two labels keep the original keys (the top label differs; mean absolute difference of
    P(first label)). More labels report top-label agreement and the mean total variation distance
    between the two probability vectors.
    """
    n = len(pairs)
    changed = sum(int(few.get("choice") != zero.get("choice")) for few, zero in pairs)
    rate = round(changed / n, 4) if n else None
    if len(labels) == 2:
        gaps = [abs(float((few.get("probabilities") or {}).get(labels[0], 0.0))
                    - float((zero.get("probabilities") or {}).get(labels[0], 0.0))) for few, zero in pairs]
        return {"n": n, "choice_differs_rate": rate,
                "mean_abs_p_positive_difference": round(sum(gaps) / n, 4) if n else None}
    tv = [0.5 * sum(abs(float((few.get("probabilities") or {}).get(label, 0.0))
                       - float((zero.get("probabilities") or {}).get(label, 0.0))) for label in labels)
          for few, zero in pairs]
    return {"n": n, "choice_differs_rate": rate,
            "top_label_agreement_rate": round(1 - changed / n, 4) if n else None,
            "mean_total_variation": round(sum(tv) / n, 4) if n else None}


@dataclass
class ArmState:
    name: str
    head: Score
    workspace: Optional[Workspace] = None
    gates: List[Dict[str, Any]] = field(default_factory=list)
    example_list: Optional[FixedExampleList] = None   # list arms; None = Jev zero-shot
    steering: List[Dict[str, Any]] = field(default_factory=list)
    list_record: Optional[Dict[str, Any]] = None


class UnifiedFlywheel:
    def __init__(self, cfg: RunConfig, *, engines: Optional[Engines] = None,
                 ledger: Optional[SpendLedger] = None, fake_core: Optional[FakeJevCore] = None):
        cfg.validate()
        self.cfg = cfg
        self.fixtures = Path(cfg.clone) / "fixtures"
        self.decision_flywheel = unified_env.ensure_decision_flywheel()
        self.corpus = cfg.corpus
        self.task = self.corpus.task()
        self.splits = self.corpus.load(self.fixtures, dev_size=cfg.dev_size)
        self.slice_name, self.eval_ids = self.splits.evaluation_slice(final=cfg.final)
        self.order = self.corpus.label_order_fn(self.splits, cfg.seed)
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
            if cfg.replay:
                engines = engines or replay_engines(cfg, self.ledger)
            engines = engines or offline_engines(cfg, self.splits, self.ledger, fake_core)
        self.engines = engines
        self.run_dir = Path(cfg.run_dir)
        self._prepare_run_dir()
        self.cache = self._shared_cache()
        self.fewshot_cache = AnswerCache(self.run_dir / "fewshot-answers.jsonl")
        self.list_answers = ListAnswers(self.run_dir / "list-answers.jsonl", self.task,
                                        {i: item.text for i, item in self.splits.items.items()}, self.labels,
                                        self.engines.zero_shot_factory, concurrency=cfg.max_concurrency)
        self.v1 = Scorecard.from_yaml((self.fixtures / "scorecards" / "v1.yaml").read_text())
        self.events: List[Dict[str, Any]] = []
        self.results: Dict[int, Dict[str, List[ItemResult]]] = {}
        self.fewshot_diagnostics: Dict[int, Dict[str, Any]] = {}
        self.bundles: Dict[str, Dict[str, Any]] = {}
        self._labeled_so_far: set = set()

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
                                   display_order=DISPLAY_ORDER, task=self.task.fingerprint,
                                   engine=self.engines.identity)

    def _fewshot_key(self, context: str) -> Dict[str, Any]:
        question = FEWSHOT_QUESTION if self.corpus is PLANTED else build_question(
            question_type="choice", instructions=self.corpus.instructions,
            criteria={label: None for label in self.corpus.labels})
        return {**question, "context_fingerprint": context}

    def _ensure_fewshot(self, targets: Sequence[str], labeled: Sequence[str]) -> Dict[str, int]:
        assert_labels_from_pool(labeled, self.splits)
        assert_retrieval_pool_clean(labeled, self.splits)
        context = self._fewshot_context(labeled)
        key = self._fewshot_key(context)
        qhash = question_hash(key)
        missing = [t for t in targets if self.fewshot_cache.get(t, self.corpus.fewshot_wire, key) is None]
        if not missing:
            return {"requests": 0, "failures": 0}
        candidates = [LabeledItem(DFItem(i, {"text": self.splits.items[i].text}), self.labels[i])
                      for i in sorted(labeled)]
        policy = PerLabelLexicalRetrieval()
        failures = 0

        async def one(target_id: str) -> None:
            nonlocal failures
            target = DFItem(target_id, {"text": self.splits.items[target_id].text})
            plan = build_context_plan(self.task, target, candidates, policy,
                                      budget=ContextBudget(per_label=FEWSHOT_PER_LABEL),
                                      display_order=DISPLAY_ORDER, order_seed=0,
                                      presentation_label_order=self.task.labels)
            assert_target_excluded(target_id, plan.example_ids)
            assert_retrieval_pool_clean(plan.example_ids, self.splits)
            try:
                result = await self.engines.fewshot_adapter.decide(self.task, target, plan.examples)
            except Exception:  # noqa: BLE001 - counted by the ledger; a gap is reported, not fatal
                failures += 1
                return
            probabilities = dict(result.probabilities or {})
            answer = {"type": "choice", "choice": result.label, "probabilities": probabilities,
                      "confidence": result.confidence if result.confidence is not None
                      else probabilities.get(result.label)}
            self.fewshot_cache.put_hashed(target_id, self.corpus.fewshot_wire, qhash, answer, result.model)

        async def all_of_them() -> None:
            await asyncio.gather(*(one(t) for t in missing))

        asyncio.run(all_of_them())
        self.ledger.raise_if_tripped()
        return {"requests": len(missing), "failures": failures}

    def _answers(self, ids: Sequence[str], score: Score, labeled: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        answers = self.cache.bulk_partial_answers(ids, zero_shot_questions(score, self.corpus))
        if uses_fewshot(score, self.corpus):
            key = self._fewshot_key(self._fewshot_context(labeled))
            for item_id in ids:
                found = self.fewshot_cache.get(item_id, self.corpus.fewshot_wire, key)
                if found is not None:
                    answers[item_id][self.corpus.fewshot_wire] = found
        return answers

    def _knn(self, targets: Sequence[str], pool_ids: Sequence[str]) -> Dict[str, Dict[str, float]]:
        assert_retrieval_pool_clean(pool_ids, self.splits)
        pool = [PoolEntry(i, self.splits.items[i].text, self.labels[i]) for i in sorted(pool_ids)]
        return knn_rows({t: self.splits.items[t].text for t in targets}, pool,
                        k=self.cfg.knn_k, top=self.cfg.knn_top, corpus=self.corpus)

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
        context = self._fewshot_context(labeled) if uses_fewshot(template, self.corpus) else "zero-shot"
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
            "fewshot_features_leave_one_out_once": uses_fewshot(template, self.corpus),
        })
        if comparison.promote:
            return head_from_fit(template, result), record, result
        return incumbent, record, None

    # ---- lever A -------------------------------------------------------------------------

    def _record_feedback(self, workspace: Workspace, batch: Sequence[str], round_number: int, *,
                         predictions: Optional[Mapping[str, Tuple[str, float]]] = None,
                         with_comments: bool = False) -> int:
        """Feedback for the new labels; returns how many carried a labeler comment."""
        card = workspace.scorecard()
        score = card.score(self.corpus.score_name)
        questions = card.questions()
        commented = 0
        for item_id in batch:
            if predictions is not None:
                value, confidence = predictions[item_id]
            else:
                result = predict(score, self.cache.partial_answers_for(item_id, questions))
                value, confidence = result.value, result.confidence
            label = self.labels[item_id]
            comment = (self.cfg.comments or {}).get(item_id) if with_comments else None
            commented += int(bool(comment))
            workspace.add_feedback(FeedbackItem(
                id=f"fb-{item_id}-s{self.cfg.seed}-r{round_number}", item_id=item_id, score_name=self.corpus.score_name,
                initial_answer_value=value, final_answer_value=label, edit_comment_value=comment or None,
                is_agreement=agrees(value, label), editor_name="scripted-labeler",
                cache_key=f"{item_id}:{self.corpus.score_name}:v{workspace.version}", label_source=LABEL_SOURCE_FINAL,
                metadata={"propensity": 1.0, "selection_policy": "seeded-uniform-order",
                          "shown_confidence": confidence, "round": round_number}))
        return commented

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
        if self.cfg.live or self.cfg.replay:
            reply = self._recorded_reply(arm, round_number)
            source = "replayed" if reply is not None else "live"
            if reply is None and self.cfg.replay:
                raise HarnessError(f"replay has no recorded analyst reply for arm {arm} round {round_number}")
        else:
            reply, source = self._offline_reply(round_number), "offline-fixed"
        before = workspace.scorecard().score(self.corpus.score_name)
        if self.cfg.analyst_provider == "openai":
            from .unified_openai import allow_gpt6_token_parameter
            allow_gpt6_token_parameter()
        outcome = run_steering(
            workspace, self.corpus.score_name, provider=self.cfg.analyst_provider, model=self.cfg.analyst_model,
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
        after = workspace.scorecard().score(self.corpus.score_name)
        comments = [c for i, c in sorted((self.cfg.comments or {}).items())
                    if arm in COMMENT_ARMS and i in self._labeled_so_far] if self.cfg.comments else []
        cited = sorted({hashlib.sha256(c.encode()).hexdigest() for c in comments
                        if len(c) >= 12 and c in (raw or "")})
        record = {"round": round_number, "comments_in_feedback": len(comments), "reply_cites_comments": cited}
        state.steering.append({**record, "reply_sha256": hashlib.sha256((raw or "").encode()).hexdigest(),
                               "elements_after": [e.key for e in after.elements]})
        return {**record, "decision": outcome.decision, "analyst_source": source,
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
        if self.cfg.final:
            if D_ARM in self.cfg.arms:
                from .unified_retrieval import run_final_d
                return run_final_d(self)
            return self.run_final()
        arms = [a for a in ARMS if a in self.cfg.arms]   # B before A+B: A+B reuses B's few-shot cache
        states: Dict[str, ArmState] = {}
        v1_score = self.v1.score(self.corpus.score_name)
        for arm in arms:
            workspace = self._workspace(arm) if arm in WORKSPACE_ARMS else None
            states[arm] = ArmState(arm, Score.from_config(v1_score.to_config()), workspace)
        labeled: List[str] = []
        rounds_out = []
        for round_number, batch in enumerate(self.batches, start=1):
            labeled = labeled + list(batch)
            self._labeled_so_far = set(labeled)
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
        if self.cfg.replay and self.ledger.new_attempts:
            raise HarnessError(f"replay needed {self.ledger.new_attempts} uncached answers; it is not a pure replay")
        self.bundles = {arm: text_free(self._freeze(arm, states[arm], labeled).manifest())
                        for arm in arms if arm in BUNDLE_ARMS}
        return self._finish(rounds_out)

    def _run_arm(self, arm: str, state: ArmState, round_number: int, batch: Sequence[str],
                 labeled: Sequence[str]) -> Dict[str, Any]:
        if arm in LIST_ARMS:
            return self._run_list_arm(arm, state, round_number, batch, labeled)
        out: Dict[str, Any] = {}
        if arm in WORKSPACE_ARMS:
            workspace = state.workspace
            existing = workspace.scorecard().questions()
            self._step(arm, round_number, "zero-shot", "top-up-new-labels",
                       lambda: self._fill_zero_shot(batch, existing))
            out["comments"] = self._record_feedback(workspace, batch, round_number,
                                                    with_comments=arm in COMMENT_ARMS)
            out["steering"] = self._step(arm, round_number, "zero-shot", "steer",
                                         lambda: self._steer(arm, state, round_number))
            ws_score = workspace.scorecard().score(self.corpus.score_name)
            ws_head, ws_gate, ws_fit = self._gate(arm, round_number, candidate_template(ws_score, fewshot=False, knn=False, corpus=self.corpus),
                                                  ws_score, labeled)
            if ws_fit is not None:
                workspace.commit_scorecard(with_fit(workspace.scorecard(), self.corpus.score_name, ws_fit), kind="fit",
                                           provenance=ws_fit.provenance)
            out["workspace_gate"] = ws_gate
            ws_score = workspace.scorecard().score(self.corpus.score_name)
            self._step(arm, round_number, "zero-shot", "fill-evaluation-slice",
                       lambda: self._fill_zero_shot(self.eval_ids, workspace.scorecard().questions()))
            if arm in ("A", "A-c"):
                state.head = ws_score
        if arm in FEWSHOT_ARMS:
            out["fewshot_requests"] = self._step(
                arm, round_number, "few-shot", "few-shot-answers",
                lambda: self._ensure_fewshot(list(labeled) + list(self.eval_ids), labeled))
            if round_number not in self.fewshot_diagnostics:
                self.fewshot_diagnostics[round_number] = self._fewshot_diagnostic(labeled)
        if arm not in ("A", "A-c"):
            base = state.workspace.scorecard().score(self.corpus.score_name) if state.workspace else self.v1.score(self.corpus.score_name)
            template = candidate_template(base, fewshot=arm in FEWSHOT_ARMS, knn=arm in KNN_ARMS, corpus=self.corpus)
            state.head, gate, _ = self._gate(arm, round_number, template, state.head, labeled)
            out["gate"] = gate
        out["head_features"] = list(state.head.decision.features)
        results, coverage = self._score(state.head, labeled)
        self.results[round_number][arm] = results
        out["metrics"] = summarize(results)
        out["evaluation_rows_missing_features"] = coverage
        self.ledger.raise_if_tripped()
        return out

    # ---- fixed-list arms (F, F-rand, A-c+F) ------------------------------------------------

    def _list_fill(self, fixed: Optional[FixedExampleList], ids: Sequence[str],
                   questions: Mapping[str, Any]) -> Dict[str, int]:
        if fixed is None:
            return self._fill_zero_shot(ids, questions)
        assert_labels_from_pool(fixed.example_ids + fixed.reserve_ids, self.splits)
        out = self.list_answers.fill(ids, questions, fixed)
        self.ledger.raise_if_tripped()
        return out

    def _list_rows(self, score: Score, ids: Sequence[str],
                   fixed: Optional[FixedExampleList]) -> Dict[str, Dict[str, float]]:
        if fixed is None:
            answers = self.cache.bulk_partial_answers(ids, score.questions())
        else:
            answers = self.list_answers.answers(ids, score.questions(), fixed)
        return {i: feature_row(score, answers[i], None) for i in ids}

    def _labeled_rows(self, labeled: Sequence[str]) -> List[LabeledItem]:
        return [LabeledItem(DFItem(i, {"text": self.splits.items[i].text}), self.labels[i]) for i in sorted(labeled)]

    def _improve_list(self, state: ArmState, labeled: Sequence[str], hard_demos: Sequence[str],
                      questions: Mapping[str, Any], round_number: int) -> Tuple[FixedExampleList, Dict[str, Any]]:
        rows = self._labeled_rows(labeled)
        settings = dict(incumbent=state.example_list, hard_demo_ids=hard_demos, per_label=self.cfg.list_per_label,
                        seed=round_number, dev_max=self.cfg.list_dev_max)
        plan = plan_example_list_round(self.task, rows, **settings)
        development = [row.item.id for row in plan.development]
        for _, fixed in plan.trials:   # prefetch concurrently; the search then reads the cache
            self._list_fill(fixed, development, questions)
        model = ListModel(self.list_answers, questions, self.corpus.score_name, [fixed for _, fixed in plan.trials],
                          self.engines.identity)
        result = asyncio.run(improve_example_list(
            self.task, rows, model, max_model_calls=len(plan.trials) * len(development),
            min_brier_gain=self.cfg.min_brier_gain, protected_ids=self.splits.test,
            presentation_label_order=self.task.labels, **settings))   # the order ListAnswers presents in
        self.ledger.raise_if_tripped()
        record = {"winner": result.winner_trial, "promoted": result.promoted, "reason": result.reason,
                  "incumbent_source": plan.incumbent_source, "n_development": len(development),
                  "scores": {name: {k: round(v, 6) for k, v in score.items()} for name, score in result.scores.items()},
                  "trial_fingerprints": {name: fixed.fingerprint for name, fixed in plan.trials},
                  "trial_status": {t.trial_name: {"status": t.status, "failures": list(t.failure_reasons)}
                                   for t in result.optimization.trials},
                  "requests_not_prefetched": model.requests}
        return result.winner, record

    def _run_list_arm(self, arm: str, state: ArmState, round_number: int, batch: Sequence[str],
                      labeled: Sequence[str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        # 1. Predict the new labels with the served classifier; its confident mistakes are hard demos.
        #    F-rand uses neither, so it skips these requests.
        predictions, wrong = None, []
        if arm != "F-rand":
            self._step(arm, round_number, "list", "predict-new-labels",
                       lambda: self._list_fill(state.example_list, batch, state.head.questions()))
            served_rows = self._list_rows(state.head, batch, state.example_list)
            predictions = {i: serve(state.head, served_rows[i]) for i in batch}
            wrong = sorted((i for i in batch if not agrees(predictions[i][0], self.labels[i])),
                           key=lambda i: (-predictions[i][1], i))
            out["new_labels_wrong"] = len(wrong)
        # 2. A-c+F: the analyst reads the feedback (with comments) and may add a question.
        base = self.v1.score(self.corpus.score_name)
        if state.workspace is not None:
            workspace = state.workspace
            existing = workspace.scorecard().questions()
            self._step(arm, round_number, "zero-shot", "top-up-new-labels",
                       lambda: self._fill_zero_shot(batch, existing))
            out["comments"] = self._record_feedback(workspace, batch, round_number, predictions=predictions,
                                                    with_comments=arm in COMMENT_ARMS)
            out["steering"] = self._step(arm, round_number, "zero-shot", "steer",
                                         lambda: self._steer(arm, state, round_number))
            base = workspace.scorecard().score(self.corpus.score_name)
        template = candidate_template(base, fewshot=False, knn=False, corpus=self.corpus)
        questions = template.questions()
        # 3. Choose the example list against the labels so far.
        if arm == "F-rand":
            chosen = random_list(self.task, self._labeled_rows(labeled), per_label=self.cfg.list_per_label,
                                 seed=F_RAND_SEED_OFFSET + round_number)
            out["list"] = {"winner": "random", "seed": F_RAND_SEED_OFFSET + round_number}
        else:
            chosen, out["list"] = self._step(arm, round_number, "list", "improve-list",
                                             lambda: self._improve_list(state, labeled, wrong, questions, round_number))
        out["list"].update({"fingerprint": chosen.fingerprint, "example_ids": list(chosen.example_ids),
                            "reserve_ids": list(chosen.reserve_ids)})
        state.list_record = {k: out["list"].get(k) for k in ("winner", "promoted", "seed", "fingerprint", "n_development",
                                                              "trial_fingerprints") if k in out["list"]}
        # 4. Answer every label and the evaluation slice with that list; retrain the head.
        out["fill"] = self._step(arm, round_number, "list", "fill-labels-and-evaluation",
                                 lambda: self._list_fill(chosen, list(labeled) + list(self.eval_ids), questions))
        rows = self._list_rows(template, labeled, chosen)
        training = training_set(template, labeled, self.labels, rows, f"example-list:{chosen.fingerprint}")
        record: Dict[str, Any] = {"n_train": training.n, "missing_features": len(training.needs_answers)}
        try:
            result = fit_head(training, template, folds=self.cfg.folds, seed=self.cfg.fit_seed)
        except LadderRefusal as refusal:
            result, record["reasons"] = None, [str(refusal)]
        if result is not None and result.fitted:
            state.head, state.example_list = head_from_fit(template, result), chosen
            record.update({"decision": "refit", "fit_id": result.provenance.get("fit_id"),
                           "fit_head_oof": _summary_dict(result.metrics)})
        else:
            record.setdefault("reasons", [getattr(result, "reason", "not fitted")])
            record["decision"] = "kept previous"
        out["head"] = record
        out["list_vs_zero_shot"] = self._list_diagnostic(list(labeled) + list(self.eval_ids), chosen)
        out["head_features"] = list(state.head.decision.features)
        results, coverage = self._score(state.head, labeled, fixed=state.example_list, list_arm=True)
        self.results[round_number][arm] = results
        out["metrics"] = summarize(results)
        out["evaluation_rows_missing_features"] = coverage
        self.ledger.raise_if_tripped()
        return out

    def _list_diagnostic(self, ids: Sequence[str], fixed: FixedExampleList) -> Dict[str, Any]:
        """S1's check: how often the holistic answer with the list differs from zero-shot (text-free)."""
        holistic = {self.corpus.score_name: self.v1.questions()[self.corpus.score_name]}
        with_list = self.list_answers.answers(ids, holistic, fixed)
        zero = self.cache.bulk_partial_answers(ids, holistic)
        pairs = [(with_list[i][self.corpus.score_name], zero[i][self.corpus.score_name]) for i in ids
                 if self.corpus.score_name in with_list[i] and self.corpus.score_name in zero[i]]
        changed = sum(a.get("choice") != b.get("choice") for a, b in pairs)
        return {"n": len(pairs), "choice_differs_rate": round(changed / len(pairs), 4) if pairs else None}

    def _score(self, head: Score, labeled: Sequence[str], *, fixed: Optional[FixedExampleList] = None,
               list_arm: bool = False) -> Tuple[List[ItemResult], int]:
        rows = (self._list_rows(head, self.eval_ids, fixed) if list_arm
                else self._rows(head, self.eval_ids, labeled))
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
        holistic = self.v1.questions()[self.corpus.score_name]
        pairs = []
        for item_id in list(labeled) + list(self.eval_ids):
            few = self.fewshot_cache.get(item_id, self.corpus.fewshot_wire, key)
            zero = self.cache.get(item_id, self.corpus.score_name, holistic)
            if few is not None and zero is not None:
                pairs.append((few, zero))
        return fewshot_diagnostic_summary(pairs, self.corpus.labels)

    # ---- bundles (design section 1.2) ----------------------------------------------------------

    def _freeze(self, arm: str, state: ArmState, labeled: Sequence[str]) -> ClassifierBundle:
        """The arm's served classifier as a core bundle, saved under ``run_dir/bundles``."""
        fixed = state.example_list
        rows = [] if fixed is None else self._labeled_rows(fixed.example_ids + fixed.reserve_ids)
        decision = state.head.decision
        provenance = decision.provenance or {}
        bundle = ClassifierBundle(
            self.task, ScoreRubric.dump(state.head), ScoreRubric(state.head), examples=fixed, example_rows=rows,
            engine={"adapter": "jev", "configured_model": self.cfg.provider_model,
                    "model_identity": self.engines.identity, "reported_model": self.cfg.provider_model},
            head={"features": list(decision.features), "classes": list(decision.classes), "model": decision.model,
                  "fit_id": provenance.get("fit_id"), "calibration": (decision.calibration or {}).get("method"),
                  "n_labels": len(labeled), "labels_fingerprint": fingerprint_ids(labeled)},
            provenance={"analyst_provider": self.cfg.analyst_provider, "analyst_model": self.cfg.analyst_model,
                        "comments_supplied": arm in COMMENT_ARMS and bool(self.cfg.comments),
                        "steering": state.steering} if state.workspace is not None else {},
            fewshot_audit=state.list_record or {},
            lineage={"harness": HARNESS_VERSION, "arm": arm, "seed": self.cfg.seed, "round": len(self.batches),
                     "labels_fingerprint": fingerprint_ids(labeled), "label_order_sha256": self._order_sha(),
                     "decision_flywheel_commit": self.decision_flywheel.get("commit"),
                     "jev_flywheel_commit": unified_env.JEV_FLYWHEEL_COMMIT, "parent": None})
        path = bundle_dir(self.run_dir, arm, len(self.batches))
        manifest = path / "bundle.json"
        if manifest.exists():
            old = json.loads(manifest.read_text()).get("bundle_hash", "unknown")
            if old != bundle.bundle_hash:   # a rerun changed it (e.g. filled failed requests): keep both
                path.rename(path.with_name(f"{path.name}.superseded-{old[:12]}"))
        bundle.save(path)
        return bundle

    def _order_sha(self) -> str:
        return hashlib.sha256(json.dumps(list(self.order[:self.cfg.rounds * self.cfg.per_round])).encode()).hexdigest()

    def load_arm_bundle(self, arm: str) -> ClassifierBundle:
        path = bundle_dir(self.run_dir, arm, self.cfg.rounds)
        if not path.exists():
            raise HarnessError(f"no frozen bundle for arm {arm} at round {self.cfg.rounds}; run the rounds first")
        return load_bundle(path, self.task, ScoreRubric.parse, configured_model=self.cfg.provider_model)

    def classify_with_bundle(self, bundle: ClassifierBundle, ids: Sequence[str],
                             client_factory: Optional[Callable[[], Any]] = None) -> Dict[str, Any]:
        """``bundle.classify`` for every item (one request each unless cached); results by ID."""
        by_text = {self.splits.items[i].text: i for i in ids}
        results: Dict[str, Any] = {}
        failures = 0

        async def all_of_them() -> int:
            nonlocal failures
            inner = (client_factory or self.engines.zero_shot_factory)()
            client = CachedJevClient(inner, by_text, zero_shot=self.cache, lists=self.list_answers,
                                     fixed=bundle.examples)
            semaphore = asyncio.Semaphore(self.cfg.max_concurrency)

            async def one(item_id: str) -> None:
                nonlocal failures
                async with semaphore:
                    try:
                        results[item_id] = await bundle.classify(DFItem(item_id, {"text": self.splits.items[item_id].text}), client)
                    except ReplayMiss:
                        raise
                    except Exception:  # noqa: BLE001 - counted by the ledger; a gap is reported, not fatal
                        failures += 1

            await asyncio.gather(*(one(i) for i in ids))
            return client.sent

        sent = asyncio.run(all_of_them())
        self.ledger.raise_if_tripped()
        return {"results": results, "requests": sent, "failures": failures}

    def run_final(self) -> Dict[str, Any]:
        """Score the evaluation slice (paper-600) through the reloaded bundles of the last round."""
        arms = [a for a in ARMS if a in self.cfg.arms]
        final: Dict[str, List[ItemResult]] = {}
        out: Dict[str, Any] = {}
        for arm in arms:
            bundle = self.load_arm_bundle(arm)
            done = self._step(arm, "final", "bundle", "classify-through-bundle",
                              lambda: self.classify_with_bundle(bundle, self.eval_ids))
            rows = []
            for item_id in self.eval_ids:
                decision = done["results"].get(item_id)
                if decision is None:
                    continue
                correct = int(agrees(decision.label, self.splits.items[item_id].reference_label))
                rows.append(ItemResult(item_id, float(decision.confidence or 0.0), correct))
            final[arm] = rows
            out[arm] = {"bundle": text_free(bundle.manifest()), "metrics": summarize(rows),
                        "unscored": len(self.eval_ids) - len(rows), "requests": done["requests"],
                        "failures": done["failures"]}
        if self.cfg.replay and self.ledger.new_attempts:
            raise HarnessError(f"replay needed {self.ledger.new_attempts} uncached answers; it is not a pure replay")
        complete = {arm: rows for arm, rows in final.items() if len(rows) == len(self.eval_ids)}
        self.results = {self.cfg.rounds: final}
        self.bundles = {arm: entry["bundle"] for arm, entry in out.items()}
        record = {"round": "final", "scored_through": "reloaded bundles", "arms": out,
                  "contrasts": contrasts(complete, resamples=self.cfg.bootstrap_resamples, seed=self.cfg.bootstrap_seed)}
        return self._finish([record])

    def _finish(self, rounds_out: List[Dict[str, Any]]) -> Dict[str, Any]:
        summary = {
            "harness": HARNESS_VERSION,
            "jev_flywheel": {"clone": "var/" + Path(self.cfg.clone).name, "commit": unified_env.JEV_FLYWHEEL_COMMIT},
            "mode": self.engines.mode, "replay": self.cfg.replay, "engine": self.engines.identity,
            "analyst": {"provider": self.cfg.analyst_provider, "model": self.cfg.analyst_model,
                        "replies": "live (recorded for replay)" if self.cfg.live else
                        "offline: round 1 = recorded Kimi-K3 reply from the clone's simulated-labeler recording; later rounds = fixed fake replies"},
            "seed": self.cfg.seed, "rounds": self.cfg.rounds, "per_round": self.cfg.per_round,
            "arms": [a for a in ARMS if a in self.cfg.arms],
            "evaluation_slice": {"name": self.slice_name, "n": len(self.eval_ids),
                                 "fingerprint": fingerprint_ids(self.eval_ids)},
            "label_order_sha256": self._order_sha(), "final": self.cfg.final,
            "decision_flywheel": self.decision_flywheel,
            "policies": {"fewshot": {"policy": "per-label-lexical-retrieval", "fingerprint": retrieval_policy_fingerprint(),
                                     "per_label": FEWSHOT_PER_LABEL, "display_order": DISPLAY_ORDER},
                         "knn": {"fingerprint": knn_policy_fingerprint(self.cfg.knn_k, self.cfg.knn_top),
                                 "k": self.cfg.knn_k, "top": self.cfg.knn_top},
                         "gate": {"min_brier_gain": self.cfg.min_brier_gain, "folds": self.cfg.folds,
                                  "fit_seed": self.cfg.fit_seed, "candidate_metrics": "nested per-fold out-of-fold"},
                         "fixed_list": {"policy": "fixed-example-list", "per_label": self.cfg.list_per_label,
                                        "request": "one Jev request per item: examples in state + every rubric question",
                                        "optimizer": "improve_example_list (incumbent, hard-swap, random-control)",
                                        "development": "labels so far outside every trial list",
                                        "dev_max": self.cfg.list_dev_max, "min_brier_gain": self.cfg.min_brier_gain,
                                        "head": "refit each round on the chosen list's answers (no extra gate)"},
                         "comments": {"supplied": len(self.cfg.comments or {}),
                                      "source": "explanation labeler" if self.cfg.comments else "none (A-c runs as A)"},
                         "bootstrap": {"resamples": self.cfg.bootstrap_resamples, "seed": self.cfg.bootstrap_seed}},
            "per_round": rounds_out,
            "bundles": self.bundles,
            "requests": self.ledger.summary(),
        }
        _assert_text_free(summary, self.splits)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        prefix = "final-" if self.cfg.final else ""   # a final run never overwrites the rounds' outputs
        (self.run_dir / f"{prefix}summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        (self.run_dir / f"{prefix}run-log.json").write_text(json.dumps(
            {"events": self.events, "requests": self.ledger.summary()}, indent=2, sort_keys=True) + "\n")
        with (self.run_dir / f"{prefix}predictions.jsonl").open("w") as handle:
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
