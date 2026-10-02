"""The FOMC corpus for the unified-flywheel harness (rubric-dataset initiative, step R5).

gtfintechlab "Trillion Dollar Words" (CC BY-NC 4.0), read from the local CSVs in
``.data/rubric-screen/fomc`` (never downloaded here). Labels 0/1/2 are dovish/hawkish/neutral, and
the options are always in that fixed order.

* the labeled **pool** is every TRAIN row except repeats of an earlier row's normalized text (1,984 rows,
  43 repeats, so 1,941 items; the core refuses duplicate example texts); labels come from nowhere else;
* every evaluation slice comes from the TEST split (496 rows): **dev-100** is a fixed-seed,
  class-stratified sample, and **FINAL** is every remaining test row (held in ``Splits.paper600``,
  the field the harness reads behind ``final=True``). A test row whose normalized text repeats a
  pool row or an earlier test row is dropped from both slices.

Item ids are ``fomc-train-<row>`` and ``fomc-test-<row>`` (0-based CSV row). The dataset's ``index``
column is NOT unique (745 repeated values in train, 69 in test), so the R3 screen's ``fomc-<index>``
ids cannot key a corpus.

The seed question asks the one-line rubric ``S.txt`` (the starting point); the ceiling question asks
``F.txt`` (S plus the published guideline excerpt). Both reproduce Jev's own distribution through a
hand-written head, as the Emotion seed does.
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

from .datasets import normalized_text_hash
from .unified_emotion import assert_no_duplicate_text, stratified_label_order
from .unified_splits import CorpusItem, Splits, assert_partitions_disjoint

FOMC_LABELS = ("dovish", "hawkish", "neutral")
FOMC_LABEL_BY_CODE = {"0": "dovish", "1": "hawkish", "2": "neutral"}
FOMC_TASK_NAME = "fomc"   # the R3 screen's DecisionTask name, so the wire question is the screen's
FOMC_DEV_SIZE = 100
FOMC_DEV_SLICE_SEED = "unified-flywheel-fomc-dev100-v1"
FOMC_LABEL_ORDER_SEED = "unified-flywheel-fomc-label-order-v1"
FOMC_REPO_ROOT = Path(__file__).resolve().parents[2]
FOMC_TRAIN_CSV = Path(".data") / "rubric-screen" / "fomc" / "hf_train.csv"
FOMC_TEST_CSV = Path(".data") / "rubric-screen" / "fomc" / "hf_test.csv"
FOMC_PROMPTS = Path("studies") / "rubric_screen" / "prompts" / "fomc"
# The size of FINAL with the real CSVs (496 test rows - 100 dev - 19 exact duplicates of pool or earlier test text). The CLI needs it
# for the request bound before any data is read; the loader checks it.
FOMC_FINAL_SIZE = 377

Row = Tuple[str, str, str]   # (id, label, text)


def starting_rubric() -> str:
    """S: the one-line rubric the classifier starts from (exactly the committed ``S.txt``)."""
    return (FOMC_REPO_ROOT / FOMC_PROMPTS / "S.txt").read_text(encoding="utf-8").strip()


def guideline_text() -> str:
    """F: S plus the published annotation-guide excerpt (the ceiling and the stakeholder's guideline)."""
    return (FOMC_REPO_ROOT / FOMC_PROMPTS / "F.txt").read_text(encoding="utf-8").strip()


def _holistic_config(instructions: str) -> dict:
    features = [f"self.holistic.clr.{label}" for label in FOMC_LABELS[:-1]]
    weights = {label: {feature: 1.0} for label, feature in zip(FOMC_LABELS[:-1], features)}
    weights[FOMC_LABELS[-1]] = {feature: -1.0 for feature in features}
    return {"name": FOMC_TASK_NAME, "key": FOMC_TASK_NAME, "question_type": "choice",
            "instructions": instructions, "criteria": {label: None for label in FOMC_LABELS},
            "decision": {"model": "multinomial_logistic", "classes": list(FOMC_LABELS), "features": features,
                         "parameters": {"weights": weights}}}


def seed_score_config() -> dict:
    """The seed score: Jev's holistic answer under S, nothing else."""
    return _holistic_config(starting_rubric())


def ceiling_score_config() -> dict:
    """The CEILING reference: the same holistic question under F (one extra question, never refit)."""
    return _holistic_config(guideline_text())


@dataclass(frozen=True)
class FomcSplitReport:
    """Text-free counts from building the splits."""
    pool_size: int
    test_size: int
    duplicate_test_items_dropped: int
    pool_duplicates_dropped: int
    final_size: int


def _stratified_sample(ids: Sequence[str], labels: dict, size: int, seed: str) -> Tuple[str, ...]:
    """A fixed-seed sample of ``size`` whose class counts follow the population (largest remainder)."""
    by_label = defaultdict(list)
    for item_id in sorted(ids):
        by_label[labels[item_id]].append(item_id)
    quotas = {label: size * len(members) / len(ids) for label, members in by_label.items()}
    counts = {label: int(quota) for label, quota in quotas.items()}
    for label in sorted(quotas, key=lambda l: (counts[l] - quotas[l], l))[:size - sum(counts.values())]:
        counts[label] += 1
    rng = random.Random(seed)
    return tuple(sorted(i for label in sorted(by_label) for i in rng.sample(by_label[label], counts[label])))


def build_fomc_splits(train: Sequence[Row], test: Sequence[Row], *,
                      dev_size: int = FOMC_DEV_SIZE) -> Tuple[Splits, FomcSplitReport]:
    """Splits from (id, label, text) rows: pool = train, dev and FINAL = disjoint parts of test."""
    ids = [row[0] for row in (*train, *test)]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate item id in the FOMC rows")
    items, pool_texts, pool_dupes = {}, set(), 0
    for item_id, label, text in train:
        key = normalized_text_hash(text)
        if key in pool_texts:   # a repeated pool text is left out (first row kept)
            pool_dupes += 1
            continue
        pool_texts.add(key)
        items[item_id] = CorpusItem(item_id, "pool", label, text)
    for item_id, label, text in test:
        items[item_id] = CorpusItem(item_id, "test", label, text)
    seen, eligible, dropped = set(pool_texts), [], 0
    for item_id, _, text in sorted(test):
        key = normalized_text_hash(text)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        eligible.append(item_id)
    labels = {i: items[i].reference_label for i in eligible}
    dev = _stratified_sample(eligible, labels, dev_size, FOMC_DEV_SLICE_SEED)
    final = tuple(sorted(set(eligible) - set(dev)))
    splits = Splits(items, tuple(sorted(i for i, it in items.items() if it.split == "pool")),
                    tuple(sorted(i for i, it in items.items() if it.split == "test")), final, dev,
                    final_name=f"final-{len(final)}")
    assert_partitions_disjoint(splits)
    assert_no_duplicate_text(splits)
    return splits, FomcSplitReport(len(splits.pool), len(splits.test), dropped, pool_dupes, len(final))


def read_fomc_rows(path: Path, split: str) -> list:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return [(f"fomc-{split}-{n}", FOMC_LABEL_BY_CODE[row["label"]], row["sentence"])
                for n, row in enumerate(csv.DictReader(handle))]


def load_fomc_splits(repo_root: Optional[Path] = None, *,
                     dev_size: int = FOMC_DEV_SIZE) -> Tuple[Splits, FomcSplitReport]:
    root = Path(repo_root) if repo_root is not None else FOMC_REPO_ROOT
    return build_fomc_splits(read_fomc_rows(root / FOMC_TRAIN_CSV, "train"),
                             read_fomc_rows(root / FOMC_TEST_CSV, "test"), dev_size=dev_size)


def load_fomc_corpus(fixtures: Path, *, dev_size: int = FOMC_DEV_SIZE) -> Splits:
    """The ``Corpus.load_splits`` hook (``fixtures`` unused). FINAL must have the size the request bound assumed."""
    splits, report = load_fomc_splits(dev_size=dev_size)
    if dev_size == FOMC_DEV_SIZE and report.final_size != FOMC_FINAL_SIZE:
        raise ValueError(f"FINAL has {report.final_size} items, not the {FOMC_FINAL_SIZE} the request bound assumes")
    return splits


def fomc_label_order(splits: Splits, seed: int) -> Tuple[str, ...]:
    """Seeded shuffle within each class, then round-robin: a round of 100 holds 34/33/33."""
    return stratified_label_order(splits, seed, FOMC_LABELS, order_seed=FOMC_LABEL_ORDER_SEED)

