"""The Emotion corpus for the unified-flywheel harness: loader, slices, stratified label order, leakage rules.

Built from the committed text-free manifest (``studies/manifests/emotion.json``) and the pinned local
Hugging Face Arrow cache (``.data/huggingface/dair-ai___emotion``), read through the repository's
strict cache reader; nothing here downloads or calls a network.

* the labeled **pool** is the manifest's 1,536 ``candidate`` items only (256 per label), ids ``train-N``;
* every evaluation slice is drawn only from the manifest's 2,000 ``scoreboard`` items, ids ``test-N``:
  **dev-100** is a fixed-seed sample of 100 and the **FINAL** slice (held in ``Splits.paper600``, the
  field the harness already reads behind ``final=True``) is a fixed-seed disjoint sample of 600;
* the manifest's 600 ``development`` items are NOT used anywhere in this experiment (they were
  the earlier static study's selection split and are never loaded here);
* the ids are exactly the earlier static study's, because cached Jev answers are keyed by them.

The scoreboard was exposed in earlier project work, so this is exploratory, not confirmatory.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple

from .cached_rows import load_cached_manifest_rows
from .datasets import EMOTION, normalized_text_hash
from .manifests import read_manifest
from .unified_splits import CorpusItem, LeakageError, Splits, assert_partitions_disjoint

EMOTION_LABELS = tuple(EMOTION.labels)
EMOTION_DEV_SLICE_SEED = "unified-flywheel-emotion-dev100-v1"
EMOTION_FINAL_SLICE_SEED = "unified-flywheel-emotion-final600-v1"
EMOTION_LABEL_ORDER_SEED = "unified-flywheel-emotion-label-order-v1"
EMOTION_DEV_SIZE = 100
EMOTION_FINAL_SIZE = 600
EMOTION_REPO_ROOT = Path(__file__).resolve().parents[2]
EMOTION_MANIFEST = Path("studies") / "manifests" / "emotion.json"
EMOTION_CACHE = Path(".data") / "huggingface"
# The task wording of the earlier static study (``study_setup._TASKS`` and the ``task.instructions`` of
# ``studies/selected/emotion.protocol.json``); a spec asserts they stay equal. The cached zero-shot
# answers (``.data/dev/emotion.scoreboard2.sqlite``) were asked under the static study's question.
EMOTION_INSTRUCTIONS = (
    "Choose exactly one emotion label. Use labeled_examples as examples of the intended categories. "
    "Classify only target.text; do not classify the examples themselves.")

Row = Tuple[str, str, str]   # (id, reference label, text)


@dataclass(frozen=True)
class EmotionSplitReport:
    """Text-free counts from building the splits."""
    pool_size: int
    scoreboard_size: int
    duplicate_scoreboard_items_dropped: int   # scoreboard items excluded: normalized text equals a pool item or an earlier scoreboard item
    pool_internal_duplicates: int             # extra pool copies of an already-seen normalized text (reported, not changed)
    eligible_scoreboard_size: int


def build_emotion_splits(candidates: Sequence[Row], scoreboard: Sequence[Row], *, dev_size: int = EMOTION_DEV_SIZE,
                         final_size: int = EMOTION_FINAL_SIZE) -> Tuple[Splits, EmotionSplitReport]:
    """Splits from (id, label, text) rows. Exact (normalized) duplicates never reach an evaluation slice."""
    items = {}
    pool_texts, pool_dupes = set(), 0
    for item_id, label, text in candidates:
        items[item_id] = CorpusItem(item_id, "pool", label, text)
        key = normalized_text_hash(text)
        pool_dupes += key in pool_texts
        pool_texts.add(key)
    seen, eligible, dropped = set(pool_texts), [], 0
    for item_id, label, text in sorted(scoreboard, key=lambda row: row[0]):
        items[item_id] = CorpusItem(item_id, "test", label, text)
        key = normalized_text_hash(text)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        eligible.append(item_id)
    if len(eligible) < dev_size + final_size:
        raise ValueError(f"only {len(eligible)} usable scoreboard items for slices of {dev_size} and {final_size}")
    dev = tuple(sorted(random.Random(EMOTION_DEV_SLICE_SEED).sample(eligible, dev_size)))
    taken = set(dev)
    rest = [i for i in eligible if i not in taken]
    final = tuple(sorted(random.Random(EMOTION_FINAL_SLICE_SEED).sample(rest, final_size)))
    splits = Splits(items, tuple(sorted(i for i, it in items.items() if it.split == "pool")),
                    tuple(sorted(i for i, it in items.items() if it.split == "test")), final, dev)
    assert_partitions_disjoint(splits)
    assert_no_duplicate_text(splits)
    report = EmotionSplitReport(len(splits.pool), len(splits.test), dropped, pool_dupes, len(eligible))
    return splits, report


def assert_no_duplicate_text(splits: Splits) -> None:
    """No normalized text appears in two places among pool, dev-100 and FINAL (an item never duplicates itself)."""
    pool_hashes = {normalized_text_hash(splits.items[i].text) for i in splits.pool}
    seen = set()
    for name, ids in (("dev-100", splits.dev100), ("final", splits.paper600)):
        for item_id in ids:
            key = normalized_text_hash(splits.items[item_id].text)
            if key in pool_hashes or key in seen:
                raise LeakageError(f"duplicate normalized text between {name} item {item_id} and the pool or another slice")
            seen.add(key)


def _pyarrow_reader(path: Path) -> Iterable[Mapping[str, object]]:
    """Read one pinned Arrow stream file with pyarrow (no ``datasets`` import, no network)."""
    if not path.is_file():
        raise OSError("missing pinned Arrow file")
    import pyarrow as pa

    with pa.memory_map(str(path)) as source:
        try:
            table = pa.ipc.open_stream(source).read_all()
        except pa.ArrowInvalid:
            table = pa.ipc.open_file(source).read_all()
    return table.to_pylist()


def load_emotion_splits(repo_root: Optional[Path] = None, *, dev_size: int = EMOTION_DEV_SIZE,
                        final_size: int = EMOTION_FINAL_SIZE,
                        reader: Optional[Callable] = None) -> Tuple[Splits, EmotionSplitReport]:
    """Splits and the text-free report from the real manifest and cache (roles candidate and scoreboard only)."""
    root = Path(repo_root) if repo_root is not None else EMOTION_REPO_ROOT
    manifest = read_manifest(root / EMOTION_MANIFEST)
    rows = load_cached_manifest_rows(manifest, cache_root=root / EMOTION_CACHE, reader=reader or _pyarrow_reader,
                                     roles=("candidate", "scoreboard"))
    role = {record.id: record.role for record in manifest.records}
    candidates = [(r.id, r.label, r.text) for r in rows if role[r.id] == "candidate"]
    scoreboard = [(r.id, r.label, r.text) for r in rows if role[r.id] == "scoreboard"]
    return build_emotion_splits(candidates, scoreboard, dev_size=dev_size, final_size=final_size)


def load_emotion_corpus(fixtures: Path, *, dev_size: int = EMOTION_DEV_SIZE) -> Splits:
    """The ``Corpus.load_splits`` hook: ``fixtures`` (a Jev-Flywheel clone path) is unused; Emotion reads this repo's manifest."""
    return load_emotion_splits(dev_size=dev_size)[0]


def stratified_label_order(splits: Splits, seed: int, labels: Sequence[str], *,
                           order_seed: str = EMOTION_LABEL_ORDER_SEED) -> Tuple[str, ...]:
    """Seeded shuffle within each label, then round-robin across labels.

    Every complete cycle takes one item per label, so a round whose size is a multiple of the label
    count (150 over 6 labels) holds exactly 25 per class, and any prefix differs by at most one
    between classes (while every class still has items).
    """
    by_label = defaultdict(list)
    for item_id in sorted(splits.pool):
        by_label[splits.items[item_id].reference_label].append(item_id)
    unknown = set(by_label) - set(labels)
    if unknown:
        raise ValueError(f"pool has labels outside the corpus labels: {sorted(unknown)}")
    queues = []
    for label in labels:
        ids = by_label.get(label, [])
        random.Random(f"{order_seed}:{seed}:{label}").shuffle(ids)
        queues.append(ids)
    order = []
    for rank in range(max((len(q) for q in queues), default=0)):
        order.extend(q[rank] for q in queues if rank < len(q))
    return tuple(order)


def emotion_label_order(splits: Splits, seed: int) -> Tuple[str, ...]:
    return stratified_label_order(splits, seed, EMOTION_LABELS)


def label_counts(splits: Splits, ids: Iterable[str]) -> Counter:
    return Counter(splits.items[i].reference_label for i in ids)
