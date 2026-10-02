"""The corpus the unified-flywheel harness runs on: labels, task wording and naming, in one frozen record.

The harness was written for one corpus, the planted sports-versus-workplace sentiment corpus. A
``Corpus`` gathers everything that corpus contributed as module constants (the label set, the task
question, the few-shot feature names, the fixtures loader, the labeler and fake-Jev hook names) so a
second corpus can be added without touching the loop. ``PLANTED`` holds exactly the values the
harness used before this module existed; running it must change nothing.

``CORPORA`` is the registry behind ``--corpus``. Only ``planted`` is registered; further corpora are
added by registering them here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from .unified_splits import Splits, load_splits


class UnknownCorpus(ValueError):
    """The requested corpus is not registered."""


@dataclass(frozen=True)
class Corpus:
    name: str
    labels: Tuple[str, ...]
    score_name: str                 # the Jev score the labels are scored under
    instructions: str               # the task wording: the zero-shot / few-shot question
    fewshot_key: str                # the element key of the few-shot feature
    fewshot_wire: str               # that element's name on the wire (``<score>.<key>`` lower-cased)
    fewshot_feature_prefix: str     # the few-shot feature is ``<prefix>.<label>`` for the first label
    knn_share_feature: str          # the kNN share feature (label share of the first label)
    load_splits: Callable[..., Splits]   # fixtures loader: ``load_splits(fixtures, dev_size=...)``
    labeler_style: str              # which explanation-labeler prompt the corpus uses
    fake_labeler_hook: str          # which offline labeler fake the corpus uses
    fake_jev_positive_label: str    # the label the offline fake Jev leans toward

    @property
    def fewshot_feature(self) -> str:
        return f"{self.fewshot_feature_prefix}.{self.labels[0]}"

    @property
    def knn_top_features(self) -> Tuple[str, ...]:
        return tuple(f"knn.top4.{label}" for label in self.labels)

    @property
    def knn_features(self) -> Tuple[str, ...]:
        return (self.knn_share_feature, *self.knn_top_features)

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

CORPORA: Dict[str, Corpus] = {PLANTED.name: PLANTED}
DEFAULT_CORPUS = PLANTED.name
CORPUS_CHOICES = tuple(CORPORA)


def get_corpus(name: str) -> Corpus:
    try:
        return CORPORA[name]
    except KeyError:
        raise UnknownCorpus(f"unknown corpus {name!r}; choose from {', '.join(CORPUS_CHOICES)}") from None
