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

``CORPORA`` is the registry behind ``--corpus``. ``planted`` is the default; ``emotion`` is registered with its loader and stratified order but still
refuses runs (see ``unsupported_reason``). Further corpora are added by registering them here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from .unified_emotion import (EMOTION_CACHED_ZERO_SHOT, EMOTION_INSTRUCTIONS, EMOTION_LABELS, EMOTION_TASK_NAME,
                              emotion_label_order, emotion_seed_score_config, load_emotion_corpus,
                              study_shaped_factory, study_zero_shot_state)
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

    def __post_init__(self) -> None:
        if self.knn_share_feature is not None and self.knn_share_feature != f"knn.share.{self.labels[0]}":
            raise ValueError(f"knn_share_feature {self.knn_share_feature!r} must be 'knn.share.<first label>' "
                             f"('knn.share.{self.labels[0]}')")

    def require_ready(self, what: str = "this command") -> None:
        if self.unsupported_reason:
            raise NotImplementedError(f"corpus {self.name!r} cannot run {what} yet: {self.unsupported_reason}")

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
# labeler (S5) does not, so runs and comment generation still refuse Emotion with that reason. The
# steering-prompt override and the feature-budget cap (S4) exist (unified_steering_prompt, unified_budget).
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
    unsupported_reason="the explanation labeler is not built yet (plan step S5)",
    fake_jev_cues=EMOTION_FAKE_JEV_CUES,
    reword_steering_prompt=True, steer_examples_phrase="the examples of each label", steer_subject_phrase="the emotion",
    seed_score=emotion_seed_score_config, zero_shot_state=study_zero_shot_state,
    zero_shot_client=study_shaped_factory, cached_zero_shot=EMOTION_CACHED_ZERO_SHOT,
)

CORPORA: Dict[str, Corpus] = {PLANTED.name: PLANTED, EMOTION_CORPUS.name: EMOTION_CORPUS}
DEFAULT_CORPUS = PLANTED.name
CORPUS_CHOICES = tuple(CORPORA)


def get_corpus(name: str) -> Corpus:
    try:
        return CORPORA[name]
    except KeyError:
        raise UnknownCorpus(f"unknown corpus {name!r}; choose from {', '.join(CORPUS_CHOICES)}") from None
