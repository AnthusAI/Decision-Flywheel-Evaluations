"""On-demand, text-free preparation of pinned evaluation manifests."""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .datasets import AG_NEWS, EMOTION, DatasetRow, DatasetSpec, normalized_text_hash
from .manifests import (DatasetManifest, DuplicateExclusion, ExposureStatus, ManifestRecord,
                        HISTORICAL_AG_NEWS_COMMIT, PreparationMetadata)


_HEX = re.compile(r"^[0-9a-f]{64}$")


class PreparationInfeasible(ValueError):
    """Requested bounded selection cannot be made from the pinned source rows."""


@dataclass(frozen=True)
class HistoricalExposure:
    source_indices: frozenset[int]
    legacy_text_hashes: frozenset[str]
    fingerprint: str
    source_commit: str


@dataclass(frozen=True)
class PreparationPlan:
    candidate_per_label: int
    development_per_label: int
    scoreboard_per_label: int | None
    natural_official_scoreboard: bool
    seed: int
    exposure_status: ExposureStatus


AG_NEWS_PLAN = PreparationPlan(512, 100, 500, False, 20261001, ExposureStatus.CONFIRMATORY_FRESH)
EMOTION_PLAN = PreparationPlan(256, 100, None, True, 20261001, ExposureStatus.EXPLORATORY)


def read_historical_ag_news_exposure(source: str | Path | bytes, *, commit: str) -> HistoricalExposure:
    """Read the committed 2,000-row Jev scoreboard without retaining article text."""
    if commit != HISTORICAL_AG_NEWS_COMMIT:
        raise ValueError("historical AG News provenance must use the reviewed source commit")
    raw = source if isinstance(source, bytes) else Path(source).read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line]
    if len(rows) != 2_000 or any(not isinstance(row, dict) or set(row) != {"id", "label", "source_index", "source_split", "text_sha256"}
                                 for row in rows):
        raise ValueError("historical AG News exposure manifest must be the exact 2,000-row text-free schema")
    indices = frozenset(row["source_index"] for row in rows)
    hashes = frozenset(row["text_sha256"] for row in rows)
    if (len(indices) != 2_000 or len(hashes) != 2_000 or any(not isinstance(index, int) or index < 0 for index in indices)
            or any(not isinstance(value, str) or not _HEX.fullmatch(value) for value in hashes)
            or any(row["source_split"] != "test" or row["id"] != f"test-{row['source_index']}" for row in rows)):
        raise ValueError("historical AG News exposure manifest has invalid source provenance")
    return HistoricalExposure(indices, hashes, hashlib.sha256(raw).hexdigest(), commit)


def prepare_pinned_study(spec: DatasetSpec, train_rows: Iterable[DatasetRow], official_rows: Iterable[DatasetRow],
                         plan: PreparationPlan, *, history: HistoricalExposure | None = None) -> DatasetManifest:
    """Select bounded train roles and a safe official scoreboard without model calls."""
    if spec == AG_NEWS:
        expected_plan = AG_NEWS_PLAN
    elif spec == EMOTION:
        expected_plan = EMOTION_PLAN
    else:
        raise ValueError("pinned preparation supports only the recorded AG News and Emotion specifications")
    if plan != expected_plan:
        raise ValueError("pinned preparation requires the declared plan and count invariants")
    if spec == AG_NEWS:
        _validate_historical_exposure(history)
    elif history is not None:
        raise ValueError("Emotion preparation has no AG News historical exposure inventory")
    train = _validate_rows(train_rows, spec, "train")
    official = _validate_rows(official_rows, spec, "test")
    history_excluded = []
    if history is not None:
        historical_rows = tuple(row for row in official if row.source_index in history.source_indices)
        if {row.source_index for row in historical_rows} != history.source_indices:
            raise ValueError("pinned official rows do not contain every reviewed historical source index")
        historical_normalized_hashes = {normalized_text_hash(row.text) for row in historical_rows}
        selected = []
        for row in official:
            if (row.source_index in history.source_indices or _legacy_text_hash(row.text) in history.legacy_text_hashes
                    or normalized_text_hash(row.text) in historical_normalized_hashes):
                history_excluded.append(row)
            else:
                selected.append(row)
        official = tuple(selected)
    official_clean, official_exclusions = _deduplicate(official)
    scoreboard = _official_scoreboard(official_clean, spec, plan)
    scoreboard_hashes = {normalized_text_hash(row.text) for row in scoreboard}
    train_without_scoreboard = tuple(row for row in train if normalized_text_hash(row.text) not in scoreboard_hashes)
    train_clean, train_exclusions = _deduplicate(train_without_scoreboard)
    leakage_exclusions = tuple(DuplicateExclusion(row.id, row.source_split, row.source_index, row.label,
                                                  normalized_text_hash(row.text),
                                                  next(item.id for item in scoreboard if normalized_text_hash(item.text) == normalized_text_hash(row.text)),
                                                  "overlaps_scoreboard_text")
                                      for row in train if normalized_text_hash(row.text) in scoreboard_hashes)
    candidates, development = [], []
    for offset, label in enumerate(spec.labels):
        group = [row for row in train_clean if row.label == label]
        need = plan.candidate_per_label + plan.development_per_label
        if len(group) < need:
            raise PreparationInfeasible(f"{spec.name}/{label} needs {need} safe train rows; has {len(group)}")
        shuffled = group.copy(); random.Random(plan.seed + offset).shuffle(shuffled)
        development.extend(shuffled[:plan.development_per_label])
        candidates.extend(shuffled[plan.development_per_label:need])
    records = tuple(_record(row, "candidate") for row in sorted(candidates, key=_identity)) \
        + tuple(_record(row, "development") for row in sorted(development, key=_identity)) \
        + tuple(_record(row, "scoreboard") for row in sorted(scoreboard, key=_identity))
    retained_ids = {record.id for record in records}
    # A same-label duplicate can be documented only when its representative is
    # itself retained.  Groups wholly outside the bounded sample do not affect
    # the manifest; conflicting-label groups remain documented as a group.
    dedup_exclusions = tuple(item for item in (*official_exclusions, *train_exclusions)
                             if item.reason == "ambiguous_normalized_text" or item.duplicate_of_id in retained_ids)
    exclusions = tuple(sorted((*dedup_exclusions, *leakage_exclusions), key=lambda item: (item.source_split, item.source_index, item.id)))
    metadata = PreparationMetadata(
        {"candidate_per_label": plan.candidate_per_label, "development_per_label": plan.development_per_label,
         "scoreboard_per_label": plan.scoreboard_per_label, "natural_official_scoreboard": plan.natural_official_scoreboard,
         "ladder_per_label": 64, "historical_source_commit": history.source_commit if history else None},
        {"candidate": len(candidates), "development": len(development), "scoreboard": len(scoreboard),
         "official_history_excluded": len(history_excluded), "dedup_excluded": len(dedup_exclusions),
         "leakage_excluded": len(leakage_exclusions)},
        history.fingerprint if history else None,
    )
    manifest = DatasetManifest(spec.name, spec.revision, plan.seed,
                               {"candidate": len(candidates), "development": len(development), "scoreboard": len(scoreboard)},
                               plan.exposure_status, records, exclusions, metadata)
    manifest.validate()
    return manifest


def _validate_historical_exposure(history: HistoricalExposure | None) -> None:
    if history is None:
        raise ValueError("AG News cannot claim a fresh scoreboard without the historical exposure manifest")
    if (not isinstance(history, HistoricalExposure) or history.source_commit != HISTORICAL_AG_NEWS_COMMIT
            or not _HEX.fullmatch(history.fingerprint)
            or any(not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in history.source_indices)
            or any(not isinstance(text_hash, str) or not _HEX.fullmatch(text_hash) for text_hash in history.legacy_text_hashes)):
        raise ValueError("AG News historical provenance must use the reviewed source commit and text-hash inventory")


def _official_scoreboard(rows: tuple[DatasetRow, ...], spec: DatasetSpec, plan: PreparationPlan) -> tuple[DatasetRow, ...]:
    if plan.natural_official_scoreboard:
        return tuple(sorted(rows, key=_identity))
    if plan.scoreboard_per_label is None:
        raise ValueError("balanced official scoreboard needs a per-label target")
    selected = []
    for offset, label in enumerate(spec.labels):
        group = [row for row in rows if row.label == label]
        if len(group) < plan.scoreboard_per_label:
            raise PreparationInfeasible(f"{spec.name}/{label} needs {plan.scoreboard_per_label} safe official rows; has {len(group)}")
        selected.extend(random.Random(plan.seed + offset).sample(group, plan.scoreboard_per_label))
    return tuple(sorted(selected, key=_identity))


def _validate_rows(rows: Iterable[DatasetRow], spec: DatasetSpec, expected_split: str | None) -> tuple[DatasetRow, ...]:
    result = tuple(rows)
    if any((expected_split is not None and row.source_split != expected_split) or row.label not in spec.labels or not isinstance(row.text, str)
           or not isinstance(row.source_index, int) or isinstance(row.source_index, bool) or row.source_index < 0
           for row in result):
        raise ValueError("pinned rows have invalid split, identity, label, or text")
    if len({(row.source_split, row.source_index) for row in result}) != len(result):
        raise ValueError("source split/index identities must be unique")
    if len({row.id for row in result}) != len(result):
        raise ValueError("source row IDs must be unique")
    return tuple(sorted(result, key=_identity))


def _deduplicate(rows: Iterable[DatasetRow]) -> tuple[tuple[DatasetRow, ...], tuple[DuplicateExclusion, ...]]:
    groups: dict[str, list[DatasetRow]] = {}
    for row in rows: groups.setdefault(normalized_text_hash(row.text), []).append(row)
    retained, exclusions = [], []
    for text_hash, group in sorted(groups.items()):
        group = sorted(group, key=_identity); anchor = group[0]
        if len({row.label for row in group}) > 1:
            exclusions.extend(DuplicateExclusion(row.id, row.source_split, row.source_index, row.label, text_hash,
                                                 anchor.id, "ambiguous_normalized_text") for row in group)
        else:
            retained.append(anchor)
            exclusions.extend(DuplicateExclusion(row.id, row.source_split, row.source_index, row.label, text_hash,
                                                 anchor.id, "duplicate_normalized_text") for row in group[1:])
    return tuple(retained), tuple(exclusions)


def _legacy_text_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.strip().casefold().split()).encode()).hexdigest()


def _record(row: DatasetRow, role: str) -> ManifestRecord:
    return ManifestRecord(row.id, row.source_split, row.source_index, row.label, normalized_text_hash(row.text), role)


def _identity(row: DatasetRow) -> tuple[str, int, str]:
    return row.source_split, row.source_index, row.id
