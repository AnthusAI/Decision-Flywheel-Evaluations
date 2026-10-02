"""The corpus the unified-flywheel harness runs on: labels, task wording and naming, in one frozen record.

The harness was written for one corpus, the planted sports-versus-workplace sentiment corpus. A
``Corpus`` gathers everything that corpus contributed as module constants (the label set, the task
question, the few-shot feature names, the fixtures loader, the labeler and fake-Jev hook names) so a
second corpus can be added without touching the loop. ``PLANTED`` holds exactly the values the
harness used before this module existed; running it must change nothing.

``CORPORA`` is the registry behind ``--corpus``. ``planted`` is the default; ``emotion`` is registered with its loader and stratified order but still
refuses runs (see ``unsupported_reason``). Further corpora are added by registering them here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from .unified_emotion import EMOTION_INSTRUCTIONS, EMOTION_LABELS, emotion_label_order, load_emotion_corpus
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
    fewshot_feature_prefix: Optional[str]   # the few-shot feature is ``<prefix>.<label>`` for the first label
    knn_share_feature: Optional[str]        # the kNN share feature (label share of the first label)
    load_splits: Callable[..., Splits]   # fixtures loader: ``load_splits(fixtures, dev_size=...)``
    labeler_style: Optional[str]    # which explanation-labeler prompt the corpus uses
    fake_labeler_hook: Optional[str]     # which offline labeler fake the corpus uses
    fake_jev_positive_label: Optional[str]   # the label the offline fake Jev leans toward
    label_order_fn: Callable[[Splits, int], Tuple[str, ...]] = planted_label_order   # the fixed labeling order
    unsupported_reason: Optional[str] = None   # set while the corpus lacks plumbing a run needs; runs refuse with it

    def require_ready(self, what: str = "this command") -> None:
        if self.unsupported_reason:
            raise NotImplementedError(f"corpus {self.name!r} cannot run {what} yet: {self.unsupported_reason}")

    def _need(self, value: Any, what: str) -> Any:
        if value is None:
            self.require_ready(what)
            raise NotImplementedError(f"corpus {self.name!r} defines no {what}")
        return value

    @property
    def fewshot_feature(self) -> str:
        return f"{self._need(self.fewshot_feature_prefix, 'few-shot feature')}.{self.labels[0]}"

    @property
    def knn_top_features(self) -> Tuple[str, ...]:
        self._need(self.knn_share_feature, "kNN features")
        return tuple(f"knn.top4.{label}" for label in self.labels)

    @property
    def knn_features(self) -> Tuple[str, ...]:
        return (self._need(self.knn_share_feature, "kNN features"), *self.knn_top_features)

    def task(self) -> Any:
        """The Decision-Flywheel task for this corpus (imported lazily: the core may not be on the path yet)."""
        from decision_flywheel.context import DecisionTask

        return DecisionTask(self.score_name, self.labels, self.instructions)

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

# The multi-class few-shot / kNN features, probabilities, fake Jev and labeler are not built for
# Emotion yet (plan steps S3-S5); runs and comment generation refuse it with this reason.
EMOTION_CORPUS = Corpus(
    name="emotion",
    labels=EMOTION_LABELS,
    score_name="Emotion",
    instructions=EMOTION_INSTRUCTIONS,
    fewshot_key=None, fewshot_wire=None, fewshot_feature_prefix=None, knn_share_feature=None,
    load_splits=load_emotion_corpus,
    labeler_style=None, fake_labeler_hook=None, fake_jev_positive_label=None,
    label_order_fn=emotion_label_order,
    unsupported_reason="the multi-class few-shot/kNN features, probabilities, fake Jev and labeler are not built yet (plan steps S3-S5)",
)

CORPORA: Dict[str, Corpus] = {PLANTED.name: PLANTED, EMOTION_CORPUS.name: EMOTION_CORPUS}
DEFAULT_CORPUS = PLANTED.name
CORPUS_CHOICES = tuple(CORPORA)


def get_corpus(name: str) -> Corpus:
    try:
        return CORPORA[name]
    except KeyError:
        raise UnknownCorpus(f"unknown corpus {name!r}; choose from {', '.join(CORPUS_CHOICES)}") from None
