"""The corpus the unified-flywheel harness runs on: labels, task wording and naming, in one frozen record.

The harness was written for one corpus, the planted sports-versus-workplace sentiment corpus. A
``Corpus`` gathers everything that corpus contributed as module constants (the label set, the task
question, the few-shot feature names, the fixtures loader, the labeler and fake-Jev hook names) so a
second corpus can be added without touching the loop. ``PLANTED`` holds exactly the values the
harness used before this module existed; running it must change nothing.

N-label conventions (two labels reproduce the old binary names exactly):

* the few-shot head features are ``<fewshot_feature_prefix>.<label>`` for every label EXCEPT THE LAST
  in ``labels`` (the *reference* label, ``fewshot_dropped_label``): centered-log-ratio features sum
  to zero, so the last one is implied and would only split the regularized weight;
* the kNN head features are ``knn.share.<label>`` for the same N-1 labels (the last share is
  ``1 - sum``) plus ``knn.top4.<label>`` for every label;
* the offline fake Jev can answer an N-option choice question from ``fake_jev_cues``, a per-label keyword
  lexicon (a test double: its numbers mean nothing about Jev).

``fomc`` (rubric-dataset initiative) starts from a one-line rubric, has a ceiling arm (``CEIL``, the full
guideline) and generates its explanations inside the loop from a simulated stakeholder.

``CORPORA`` is the registry behind ``--corpus``. ``planted`` is the default; ``emotion`` runs LABELS-ONLY (no
comments are generated or read) but refuses anything that needs the explanation labeler until it exists
(see ``labeler_unsupported_reason``). Further corpora are added by registering them here.

Readiness has two levels: ``unsupported_reason`` blocks every run (``require_ready``), and
``labeler_unsupported_reason`` blocks only what needs the explanation labeler, i.e. ``comments`` generation
and a run that is handed comments (``require_labeler``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from .unified_emotion import (EMOTION_CACHED_ZERO_SHOT, EMOTION_INSTRUCTIONS, EMOTION_LABELS, EMOTION_TASK_NAME,
                              emotion_label_order, emotion_seed_score_config, load_emotion_corpus,
                              study_shaped_factory, study_zero_shot_state)
from .unified_fomc import (FOMC_FINAL_SIZE, FOMC_LABELS, FOMC_TASK_NAME, ceiling_score_config, fomc_label_order,
                           guideline_text, load_fomc_corpus, seed_score_config, starting_rubric)
from . import unified_reviews as reviews
from .unified_splits import Splits, label_order, load_splits


class UnknownCorpus(ValueError):
    """The requested corpus is not registered."""


def planted_label_order(splits: Splits, seed: int) -> Tuple[str, ...]:
    """The planted corpus's order: the plain seeded shuffle of the pool (unchanged)."""
    return label_order(splits.pool, seed)


@dataclass(frozen=True)
class Corpus:
    name: str
    labels: Tuple[str, ...]
    score_name: str                 # the Jev score the labels are scored under
    instructions: str               # the task wording: the zero-shot / few-shot question
    fewshot_key: Optional[str]      # the element key of the few-shot feature (None: not built for this corpus yet)
    fewshot_wire: Optional[str]     # that element's name on the wire (``<score>.<key>`` lower-cased)
    fewshot_feature_prefix: Optional[str]   # the few-shot features are ``<prefix>.<label>`` for every label but the last
    knn_share_feature: Optional[str]        # the first kNN share feature, ``knn.share.<labels[0]>`` (None: not built yet)
    load_splits: Callable[..., Splits]   # fixtures loader: ``load_splits(fixtures, dev_size=...)``
    labeler_style: Optional[str]    # which explanation-labeler prompt the corpus uses
    fake_labeler_hook: Optional[str]     # which offline labeler fake the corpus uses
    fake_jev_positive_label: Optional[str]   # the label the offline fake Jev leans toward
    label_order_fn: Callable[[Splits, int], Tuple[str, ...]] = planted_label_order   # the fixed labeling order
    unsupported_reason: Optional[str] = None   # set while the corpus lacks plumbing a run needs; runs refuse with it
    labeler_unsupported_reason: Optional[str] = None   # set while there is no explanation labeler: comments and comment-fed runs refuse
    multiclass_metrics: bool = False   # also report macro-F1, per-class recall and the confusion matrix (planted: no, byte-identical)
    workspace_items_from_pool: bool = False   # arm workspaces hold this corpus's POOL items, not the pinned fixtures' items.jsonl
    fake_analyst_round1_reply: Optional[str] = None   # offline fake analyst's round-1 reply (None: the clone's planted recording)
    fake_jev_cues: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()   # (label, keywords) pairs the offline fake Jev leans on; () = binary planted fake
    # The pinned steering procedure speaks of "the positive and negative examples" and "sentiment"; a corpus
    # that is not about sentiment sets ``reword_steering_prompt`` and the harness swaps in a reworded copy
    # for the run (see ``unified_steering_prompt``). PLANTED keeps the pinned file byte for byte.
    reword_steering_prompt: bool = False
    steer_examples_phrase: str = "the examples of each label"   # replaces "the positive and negative examples"
    steer_subject_phrase: str = "the labeled property"          # replaces "sentiment" in "not about sentiment"
    # Reusing answers a static study already paid for (see ``unified_import``). All four are None for PLANTED,
    # which keeps the pinned fixture scorecard, the plain ``{"text": t}`` zero-shot state and no import.
    seed_score: Optional[Callable[[], Dict[str, Any]]] = None   # the seed score's config, in place of fixtures/v1.yaml
    zero_shot_state: Optional[Callable[[str], Dict[str, Any]]] = None   # the state a zero-shot request carries
    zero_shot_client: Optional[Callable[[Callable], Callable]] = None   # wraps the zero-shot client factory to send that state
    cached_zero_shot: Optional[Path] = None   # repo-relative sqlite of the static study's zero-shot answers
    # Rubric corpora (FOMC). All default to the old behaviour.
    final_size: int = 600                      # the held-out FINAL slice's size (the request bound needs it before loading)
    fill_seed_answers: bool = False            # no shipped seed answers: arm 0 asks the seed question of labels + slice
    ceiling_score: Optional[Callable[[], Dict[str, Any]]] = None   # arm CEIL's question (the full guideline)
    stakeholder_guideline: Optional[Callable[[], str]] = None      # the rubric stakeholder's guideline (in-loop comments)

    def __post_init__(self) -> None:
        if self.knn_share_feature is not None and self.knn_share_feature != f"knn.share.{self.labels[0]}":
            raise ValueError(f"knn_share_feature {self.knn_share_feature!r} must be 'knn.share.<first label>' "
                             f"('knn.share.{self.labels[0]}')")

    def require_ready(self, what: str = "this command") -> None:
        if self.unsupported_reason:
            raise NotImplementedError(f"corpus {self.name!r} cannot run {what} yet: {self.unsupported_reason}")

    def require_labeler(self, what: str = "this command") -> None:
        """Refuse what needs the explanation labeler (comments) when the corpus has none yet."""
        self.require_ready(what)
        if self.labeler_unsupported_reason:
            raise NotImplementedError(f"corpus {self.name!r} cannot run {what} yet: {self.labeler_unsupported_reason}")

    def _need(self, value: Any, what: str) -> Any:
        if value is None:
            self.require_ready(what)
            raise NotImplementedError(f"corpus {self.name!r} defines no {what}")
        return value

    @property
    def fewshot_dropped_label(self) -> str:
        """The reference label whose few-shot feature is dropped: always the last label (deterministic)."""
        return self.labels[-1]

    @property
    def fewshot_features(self) -> Tuple[str, ...]:
        prefix = self._need(self.fewshot_feature_prefix, "few-shot feature")
        return tuple(f"{prefix}.{label}" for label in self.labels[:-1])

    @property
    def fewshot_feature(self) -> str:
        """The first few-shot feature (the only one for two labels)."""
        return self.fewshot_features[0]

    @property
    def knn_share_features(self) -> Tuple[str, ...]:
        self._need(self.knn_share_feature, "kNN features")
        return tuple(f"knn.share.{label}" for label in self.labels[:-1])

    @property
    def knn_top_features(self) -> Tuple[str, ...]:
        self._need(self.knn_share_feature, "kNN features")
        return tuple(f"knn.top4.{label}" for label in self.labels)

    @property
    def knn_features(self) -> Tuple[str, ...]:
        return (*self.knn_share_features, *self.knn_top_features)

    def task(self) -> Any:
        """The Decision-Flywheel task for this corpus (imported lazily: the core may not be on the path yet)."""
        from decision_flywheel.context import DecisionTask

        return DecisionTask(self.score_name, self.labels, self.instructions)

    def seed_scorecard(self) -> Any:
        """The scorecard the run starts from when the corpus defines its own seed score (else None)."""
        if self.seed_score is None:
            return None
        from jev_flywheel.scorecard import Scorecard

        return Scorecard.from_config({"name": self.score_name, "version": 1, "scores": [self.seed_score()]})

    def ceiling_scorecard(self) -> Any:
        from jev_flywheel.scorecard import Scorecard

        config = self._need(self.ceiling_score, "ceiling question")()
        return Scorecard.from_config({"name": self.score_name, "version": 1, "scores": [config]})

    def load(self, fixtures: Path, *, dev_size: int) -> Splits:
        return self.load_splits(fixtures, dev_size=dev_size)


PLANTED = Corpus(
    name="planted",
    labels=("positive", "negative"),
    score_name="Sentiment",
    instructions="What is the overall sentiment of this text?",
    fewshot_key="fewshot",
    fewshot_wire="sentiment.fewshot",
    fewshot_feature_prefix="fewshot.clr",
    knn_share_feature="knn.share.positive",
    load_splits=load_splits,
    labeler_style="hidden-convention",
    fake_labeler_hook="sports-office-lexicon",
    fake_jev_positive_label="positive",
)

# The multi-class few-shot / kNN naming, probabilities and fake-Jev cues exist (plan step S3); the
# labeler (S5) does not, so comment generation and runs handed comments refuse Emotion with that reason,
# while labels-only runs (no comments) work. The steering-prompt override and the feature-budget cap (S4)
# exist (unified_steering_prompt, unified_budget).
EMOTION_FAKE_ANALYST_ROUND1 = json.dumps({
    "root_cause": "Offline fake analyst: strong bodily reactions may separate fear and surprise from sadness.",
    "add_elements": [{"key": "bodily_reaction", "question_type": "noul",
                      "instructions": "Does the text describe a strong bodily reaction such as trembling, tears or a racing heart?",
                      "criteria": None}],
    "retire_elements": [], "reword_elements": []})
EMOTION_FAKE_JEV_CUES = (
    ("sadness", ("sad", "depressed", "miserable", "lonely", "grief", "heartbroken", "unhappy", "hopeless", "gloomy", "hurt")),
    ("joy", ("happy", "glad", "delighted", "cheerful", "thrilled", "excited", "joyful", "pleased", "blessed", "proud")),
    ("love", ("love", "adore", "loving", "affection", "caring", "sweet", "fond", "tender", "cherish", "devoted")),
    ("anger", ("angry", "furious", "mad", "rage", "annoyed", "irritated", "hate", "outraged", "resentful", "bitter")),
    ("fear", ("afraid", "scared", "terrified", "anxious", "worried", "nervous", "frightened", "panic", "dread", "threatened")),
    ("surprise", ("surprised", "amazed", "astonished", "shocked", "stunned", "startled", "unexpected", "wow", "speechless", "unbelievable")),
)
EMOTION_CORPUS = Corpus(
    name="emotion",
    labels=EMOTION_LABELS,
    score_name=EMOTION_TASK_NAME,   # "emotion": the static study's task name, so its zero-shot requests are reproducible
    instructions=EMOTION_INSTRUCTIONS,
    fewshot_key="fewshot", fewshot_wire="emotion.fewshot", fewshot_feature_prefix="fewshot.clr",
    knn_share_feature="knn.share.sadness",
    load_splits=load_emotion_corpus,
    labeler_style=None, fake_labeler_hook=None, fake_jev_positive_label=None,
    label_order_fn=emotion_label_order,
    labeler_unsupported_reason="the explanation labeler is not built yet (plan step S5)",
    multiclass_metrics=True, workspace_items_from_pool=True, fake_analyst_round1_reply=EMOTION_FAKE_ANALYST_ROUND1,
    fake_jev_cues=EMOTION_FAKE_JEV_CUES,
    reword_steering_prompt=True, steer_examples_phrase="the examples of each label", steer_subject_phrase="the emotion",
    seed_score=emotion_seed_score_config, zero_shot_state=study_zero_shot_state,
    zero_shot_client=study_shaped_factory, cached_zero_shot=EMOTION_CACHED_ZERO_SHOT,
)

# FOMC (rubric-dataset initiative): starts from the one-line rubric S; CEIL asks the full guideline F;
# the explanation arms get the in-loop rubric stakeholder (``unified_labeler.RubricStakeholder``), so
# ``comments`` files are not used. Zero-shot requests go out in the R3 screen's state shape.
FOMC_FAKE_ANALYST_ROUND1 = json.dumps({
    "root_cause": "Offline fake analyst: the direction of inflation or prices may separate hawkish from dovish.",
    "add_elements": [{"key": "prices_rising", "question_type": "noul",
                      "instructions": "Does the sentence say that inflation, energy prices or house prices are rising?",
                      "criteria": None}],
    "retire_elements": [], "reword_elements": []})
FOMC_FAKE_JEV_CUES = (
    ("dovish", ("easing", "accommodative", "lower", "decline", "declined", "weak", "weaker", "slack", "subdued", "decrease")),
    ("hawkish", ("tightening", "inflation", "higher", "rising", "increase", "increased", "strong", "stronger", "firming", "pressures")),
    ("neutral", ("unchanged", "maintained", "mixed", "moderate", "reaffirmed", "sustained", "stable", "balanced", "members", "committee")),
)
FOMC_CORPUS = Corpus(
    name="fomc",
    labels=FOMC_LABELS,
    score_name=FOMC_TASK_NAME,
    instructions=starting_rubric(),
    fewshot_key="fewshot", fewshot_wire="fomc.fewshot", fewshot_feature_prefix="fewshot.clr",
    knn_share_feature="knn.share.dovish",
    load_splits=load_fomc_corpus,
    labeler_style="rubric-stakeholder", fake_labeler_hook="guideline-keyword-rule", fake_jev_positive_label=None,
    label_order_fn=fomc_label_order,
    multiclass_metrics=True, workspace_items_from_pool=True, fake_analyst_round1_reply=FOMC_FAKE_ANALYST_ROUND1,
    fake_jev_cues=FOMC_FAKE_JEV_CUES,
    reword_steering_prompt=True, steer_examples_phrase="the examples of each label",
    steer_subject_phrase="the monetary policy stance",
    seed_score=seed_score_config, zero_shot_state=study_zero_shot_state, zero_shot_client=study_shaped_factory,
    final_size=FOMC_FINAL_SIZE, fill_seed_answers=True, ceiling_score=ceiling_score_config,
    stakeholder_guideline=guideline_text,
)

# Amazon review moderation (streaming SME plan, step 3): gold labels are the simulated SME's, S is the
# committed one-liner, F (ceiling and stakeholder guideline) is the private policy read only when needed.
# ``reviews-merged`` is the plan's class merge (abusive + promotional -> remove_other); which one the data
# allows is decided from the SME labels by ``unified_reviews.should_merge``, and each refuses the other's data.
REVIEWS_FAKE_ANALYST_ROUND1 = json.dumps({
    "root_cause": "Offline fake analyst: whether the review is mainly about delivery or the seller may separate seller_shipping from approve.",
    "add_elements": [{"key": "about_delivery", "question_type": "noul",
                      "instructions": "Is the review mainly about delivery, packaging or the seller rather than the product?",
                      "criteria": None}],
    "retire_elements": [], "reword_elements": []})
_REVIEWS_CUES = {
    "approve": ("love", "great", "works", "quality", "recommend", "sturdy", "cute", "perfect", "easy", "fits"),
    "abusive": ("crap", "idiot", "idiots", "stupid", "damn", "moron", "liar", "liars", "pissed", "shit"),
    "promotional": ("http", "www", "exchange", "free", "sample", "discounted", "instead", "website", "email", "promo"),
    "seller_shipping": ("seller", "shipping", "shipped", "refund", "arrived", "package", "delivery", "returned", "box", "late"),
    "price_availability": ("cheaper", "sale", "stock", "price", "priced", "deal", "walmart", "target", "available", "dropped"),
}


def _reviews_corpus(merged: bool) -> Corpus:
    labels = reviews.reviews_labels(merged)
    cues = dict(_REVIEWS_CUES)
    if merged:
        cues[reviews.MERGED_LABEL] = cues["abusive"] + cues["promotional"]
    return Corpus(
        name="reviews-merged" if merged else "reviews",
        labels=labels,
        score_name=reviews.REVIEWS_TASK_NAME,
        instructions=reviews.starting_rubric(merged),
        fewshot_key="fewshot", fewshot_wire="reviews.fewshot", fewshot_feature_prefix="fewshot.clr",
        knn_share_feature=f"knn.share.{labels[0]}",
        load_splits=partial(reviews.load_reviews_corpus, merged=merged),
        labeler_style=None, fake_labeler_hook=None, fake_jev_positive_label=None,
        label_order_fn=partial(reviews.reviews_label_order, merged=merged),
        multiclass_metrics=True, workspace_items_from_pool=True, fake_analyst_round1_reply=REVIEWS_FAKE_ANALYST_ROUND1,
        fake_jev_cues=tuple((label, cues[label]) for label in labels),
        reword_steering_prompt=True, steer_examples_phrase="the examples of each label",
        steer_subject_phrase="the moderation decision",
        seed_score=partial(reviews.seed_score_config, merged), zero_shot_state=study_zero_shot_state,
        zero_shot_client=study_shaped_factory,
        final_size=reviews.REVIEWS_FINAL_SIZE, fill_seed_answers=True,
        ceiling_score=partial(reviews.ceiling_score_config, merged),
        stakeholder_guideline=partial(reviews.guideline_text, merged),
    )


REVIEWS_CORPUS = _reviews_corpus(False)
REVIEWS_MERGED_CORPUS = _reviews_corpus(True)

CORPORA: Dict[str, Corpus] = {PLANTED.name: PLANTED, EMOTION_CORPUS.name: EMOTION_CORPUS, FOMC_CORPUS.name: FOMC_CORPUS,
                              REVIEWS_CORPUS.name: REVIEWS_CORPUS, REVIEWS_MERGED_CORPUS.name: REVIEWS_MERGED_CORPUS}
DEFAULT_CORPUS = PLANTED.name
CORPUS_CHOICES = tuple(CORPORA)


def get_corpus(name: str) -> Corpus:
    try:
        return CORPORA[name]
    except KeyError:
        raise UnknownCorpus(f"unknown corpus {name!r}; choose from {', '.join(CORPUS_CHOICES)}") from None
