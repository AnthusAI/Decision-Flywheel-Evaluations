"""Text-free deterministic split manifests with prior-exposure protection."""
from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from .datasets import DatasetRow, DatasetSpec, normalized_text_hash


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
HISTORICAL_AG_NEWS_COMMIT = "59548114ecf45e817613c946af2cab6ceb675804"
_PREPARATION_CONFIGURATION_FIELDS = {
    "candidate_per_label", "development_per_label", "scoreboard_per_label",
    "natural_official_scoreboard", "ladder_per_label", "historical_source_commit",
}
_PREPARATION_COUNT_FIELDS = {
    "candidate", "development", "scoreboard", "official_history_excluded",
    "dedup_excluded", "leakage_excluded",
}


class ExposureStatus(str, Enum):
    CONFIRMATORY_FRESH = "confirmatory_fresh"
    EXPLORATORY = "exploratory"
    HISTORICAL_EXPOSED = "historical_exposed"


@dataclass(frozen=True)
class ManifestRecord:
    id: str
    source_split: str
    source_index: int
    label: str
    normalized_text_sha256: str
    role: str


@dataclass(frozen=True)
class DuplicateExclusion:
    """Text-free provenance for a duplicate excluded before role assignment."""

    id: str
    source_split: str
    source_index: int
    label: str
    normalized_text_sha256: str
    duplicate_of_id: str
    reason: str = "duplicate_normalized_text"


@dataclass(frozen=True)
class PreparationMetadata:
    """Text-free, deterministic selection and exposure provenance."""

    configuration: dict[str, object]
    sample_counts: dict[str, int]
    exposure_history_fingerprint: str | None


@dataclass(frozen=True)
class DatasetManifest:
    dataset: str
    revision: str
    seed: int
    counts: dict[str, int]
    exposure_status: ExposureStatus
    records: tuple[ManifestRecord, ...]
    exclusions: tuple[DuplicateExclusion, ...]
    preparation: PreparationMetadata | None = None

    def for_role(self, role: str) -> tuple[ManifestRecord, ...]:
        return tuple(record for record in self.records if record.role == role)

    @property
    def candidate(self) -> tuple[ManifestRecord, ...]: return self.for_role("candidate")

    @property
    def development(self) -> tuple[ManifestRecord, ...]: return self.for_role("development")

    @property
    def scoreboard(self) -> tuple[ManifestRecord, ...]: return self.for_role("scoreboard")

    def validate(self) -> None:
        if (not isinstance(self.dataset, str) or not self.dataset or not isinstance(self.revision, str)
                or not _REVISION.fullmatch(self.revision) or not isinstance(self.seed, int)
                or isinstance(self.seed, bool) or self.seed < 0):
            raise ValueError("manifest needs dataset, revision, and seed")
        roles = {"candidate", "development", "scoreboard"}
        if not isinstance(self.counts, dict) or set(self.counts) != roles:
            raise ValueError("manifest counts must name candidate, development, and scoreboard")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in self.counts.values()):
            raise ValueError("manifest counts must be nonnegative integers")
        if not isinstance(self.exposure_status, ExposureStatus):
            raise ValueError("manifest needs an explicit valid exposure status")
        if self.preparation is not None:
            configuration = self.preparation.configuration if isinstance(self.preparation, PreparationMetadata) else None
            sample_counts = self.preparation.sample_counts if isinstance(self.preparation, PreparationMetadata) else None
            if (not isinstance(self.preparation, PreparationMetadata)
                    or not isinstance(configuration, dict) or set(configuration) != _PREPARATION_CONFIGURATION_FIELDS
                    or not isinstance(sample_counts, dict) or set(sample_counts) != _PREPARATION_COUNT_FIELDS
                    or any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                           for value in sample_counts.values())
                    or sample_counts["candidate"] != self.counts["candidate"]
                    or sample_counts["development"] != self.counts["development"]
                    or sample_counts["scoreboard"] != self.counts["scoreboard"]
                    or any(not isinstance(configuration[field], int) or isinstance(configuration[field], bool)
                           or configuration[field] < 0
                           for field in ("candidate_per_label", "development_per_label", "ladder_per_label"))
                    or (configuration["scoreboard_per_label"] is not None
                        and (not isinstance(configuration["scoreboard_per_label"], int)
                             or isinstance(configuration["scoreboard_per_label"], bool)
                             or configuration["scoreboard_per_label"] < 1))
                    or not isinstance(configuration["natural_official_scoreboard"], bool)
                    or configuration["ladder_per_label"] != 64
                    or (configuration["historical_source_commit"] is not None
                        and (not isinstance(configuration["historical_source_commit"], str)
                             or not _REVISION.fullmatch(configuration["historical_source_commit"])))
                    or ((configuration["historical_source_commit"] is None)
                        != (self.preparation.exposure_history_fingerprint is None))
                    or (self.preparation.exposure_history_fingerprint is not None
                        and not _SHA256.fullmatch(self.preparation.exposure_history_fingerprint))
                    or (self.dataset == "fancyzhx/ag_news"
                        and configuration["historical_source_commit"] != HISTORICAL_AG_NEWS_COMMIT)
                    or (self.dataset == "dair-ai/emotion"
                        and configuration["historical_source_commit"] is not None)):
                raise ValueError("manifest preparation metadata must be text-free and valid")
        if any(not isinstance(record, ManifestRecord) for record in self.records):
            raise ValueError("manifest records have invalid types")
        if any(not isinstance(item, DuplicateExclusion) for item in self.exclusions):
            raise ValueError("manifest exclusions have invalid types")
        if any(record.role not in roles for record in self.records):
            raise ValueError("manifest record has an unknown role")
        if any(not isinstance(record.id, str) or not record.id or not isinstance(record.source_split, str)
               or not record.source_split or not isinstance(record.source_index, int) or isinstance(record.source_index, bool)
               or record.source_index < 0 or not isinstance(record.label, str) or not record.label
               for record in self.records):
            raise ValueError("manifest records need valid source identity fields")
        if any(not isinstance(record.normalized_text_sha256, str) or not _SHA256.fullmatch(record.normalized_text_sha256)
               for record in self.records):
            raise ValueError("manifest records need normalized SHA-256 hashes")
        actual = {role: len(self.for_role(role)) for role in roles}
        if actual != self.counts:
            raise ValueError("manifest counts do not match records")
        ids = [record.id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("split IDs must be disjoint")
        identities = [(record.source_split, record.source_index) for record in self.records]
        if len(identities) != len(set(identities)):
            raise ValueError("manifest records must have unique source split/index identities")
        by_role = {role: {record.normalized_text_sha256 for record in self.for_role(role)} for role in roles}
        if any(len(by_role[role]) != len(self.for_role(role)) for role in roles):
            raise ValueError("normalized text hashes must be unique within each split role")
        if any(by_role[left] & by_role[right] for left in roles for right in roles if left < right):
            raise ValueError("normalized text hashes must be disjoint across split roles")
        excluded_ids = {record.id for record in self.exclusions}
        if excluded_ids & set(ids) or len(excluded_ids) != len(self.exclusions):
            raise ValueError("duplicate exclusions must not overlap manifest records")
        allowed_exclusion_reasons = {"duplicate_normalized_text", "overlaps_scoreboard_text", "ambiguous_normalized_text"}
        if any(not isinstance(item.id, str) or not item.id or not isinstance(item.source_split, str) or not item.source_split
               or not isinstance(item.source_index, int) or isinstance(item.source_index, bool) or item.source_index < 0
               or not isinstance(item.label, str) or not item.label or not isinstance(item.duplicate_of_id, str)
               or not item.duplicate_of_id or item.reason not in allowed_exclusion_reasons
               or not isinstance(item.normalized_text_sha256, str) or not _SHA256.fullmatch(item.normalized_text_sha256)
               for item in self.exclusions):
            raise ValueError("duplicate exclusions must have text-free duplicate provenance")
        all_identities = identities + [(item.source_split, item.source_index) for item in self.exclusions]
        if len(all_identities) != len(set(all_identities)):
            raise ValueError("manifest records and exclusions must have unique source split/index identities")
        records_by_id = {record.id: record for record in self.records}
        for item in self.exclusions:
            reference = records_by_id.get(item.duplicate_of_id)
            if item.reason == "ambiguous_normalized_text":
                continue
            if reference is None or reference.normalized_text_sha256 != item.normalized_text_sha256 \
                    or (item.reason == "duplicate_normalized_text" and reference.label != item.label):
                raise ValueError("duplicate exclusion must reference a matching manifest record")


def _record(row: DatasetRow, role: str) -> ManifestRecord:
    return ManifestRecord(row.id, row.source_split, row.source_index, row.label, normalized_text_hash(row.text), role)


def _exposed_keys(prior_exposed: Iterable[ManifestRecord]) -> tuple[set[str], set[str]]:
    records = tuple(prior_exposed)
    return {record.id for record in records}, {record.normalized_text_sha256 for record in records}


def _deduplicate(rows: Iterable[DatasetRow]) -> tuple[tuple[DatasetRow, ...], tuple[DuplicateExclusion, ...]]:
    """Retain a deterministic representative for same-label duplicate source text."""
    by_hash: dict[str, list[DatasetRow]] = {}
    for row in rows:
        by_hash.setdefault(normalized_text_hash(row.text), []).append(row)
    retained, exclusions = [], []
    for text_hash, group in sorted(by_hash.items()):
        ordered = sorted(group, key=lambda row: (row.source_split, row.source_index, row.id))
        if len({row.label for row in ordered}) != 1:
            exclusions.extend(DuplicateExclusion(row.id, row.source_split, row.source_index, row.label, text_hash,
                                                 ordered[0].id, "ambiguous_normalized_text") for row in ordered)
            continue
        retained.append(ordered[0])
        exclusions.extend(DuplicateExclusion(row.id, row.source_split, row.source_index, row.label, text_hash,
                                             ordered[0].id) for row in ordered[1:])
    return tuple(sorted(retained, key=lambda row: (row.source_split, row.source_index, row.id))), tuple(exclusions)


def _validate_rows(rows: Iterable[DatasetRow], spec: DatasetSpec) -> tuple[DatasetRow, ...]:
    source = tuple(rows)
    if any(not isinstance(row.id, str) or not row.id or not isinstance(row.source_split, str) or not row.source_split
           or not isinstance(row.source_index, int) or isinstance(row.source_index, bool) or row.source_index < 0
           or not isinstance(row.label, str) or row.label not in spec.labels or not isinstance(row.text, str)
           for row in source):
        raise ValueError("rows need valid nonnegative source identity and canonical labels")
    if len({row.id for row in source}) != len(source):
        raise ValueError("source rows must have unique IDs")
    if len({(row.source_split, row.source_index) for row in source}) != len(source):
        raise ValueError("source rows must have unique source split/index identities")
    return tuple(sorted(source, key=lambda row: (row.source_split, row.source_index, row.id)))


def select_official_scoreboard(
    rows: Iterable[DatasetRow], spec: DatasetSpec, *, source_split: str, seed: int, per_label: int,
) -> tuple[DatasetRow, ...]:
    """Select an exact, deterministic scoreboard from one named official split."""
    if not isinstance(source_split, str) or not source_split:
        raise ValueError("official scoreboard source split is required")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(per_label, int) or isinstance(per_label, bool) or per_label < 1:
        raise ValueError("official scoreboard target must be positive")
    source, _ = _deduplicate(_validate_rows((row for row in rows if row.source_split == source_split), spec))
    selected = []
    for offset, label in enumerate(spec.labels):
        group = [row for row in source if row.label == label]
        if len(group) < per_label:
            raise ValueError(f"{source_split}/{label} needs {per_label} rows; has {len(group)}")
        sample = random.Random(seed + offset).sample(group, per_label)
        selected.extend(sorted(sample, key=lambda row: (row.source_index, row.id)))
    return tuple(selected)


def prepare_split(
    rows: Iterable[DatasetRow], spec: DatasetSpec, *, seed: int,
    development_per_label: int, scoreboard_per_label: int,
    exposure_status: ExposureStatus,
    prior_exposed: Iterable[ManifestRecord] = (),
) -> DatasetManifest:
    """Create documented label-stratified roles without inspecting model outcomes."""
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(development_per_label, int) or isinstance(development_per_label, bool) \
            or not isinstance(scoreboard_per_label, int) or isinstance(scoreboard_per_label, bool):
        raise ValueError("split counts must be integers")
    if development_per_label < 0 or scoreboard_per_label < 1:
        raise ValueError("development target must be nonnegative and scoreboard target must be positive")
    if not isinstance(exposure_status, ExposureStatus):
        raise ValueError("an explicit valid exposure status is required")
    source, exclusions = _deduplicate(_validate_rows(rows, spec))
    if spec.name == "dair-ai/emotion" and exposure_status is ExposureStatus.CONFIRMATORY_FRESH and \
            {row.source_split for row in source} & {"test", "validation"}:
        raise ValueError("Emotion official test and validation splits are historically exposed")
    exposed_ids, exposed_hashes = _exposed_keys(prior_exposed)
    selected: list[ManifestRecord] = []
    need = development_per_label + scoreboard_per_label
    for offset, label in enumerate(spec.labels):
        group = [row for row in source if row.label == label]
        if len(group) < need:
            raise ValueError(f"{label} needs {need} rows for development plus scoreboard; has {len(group)}")
        shuffled = group.copy()
        random.Random(seed + offset).shuffle(shuffled)
        selected.extend(_record(row, "development") for row in shuffled[:development_per_label])
        selected.extend(_record(row, "scoreboard") for row in shuffled[development_per_label:need])
        selected.extend(_record(row, "candidate") for row in shuffled[need:])
    if exposure_status is ExposureStatus.CONFIRMATORY_FRESH:
        protected = [record for record in selected if record.role in {"development", "scoreboard"}]
        selected_ids = {record.id for record in protected}
        selected_hashes = {record.normalized_text_sha256 for record in protected}
        if selected_ids & exposed_ids or selected_hashes & exposed_hashes:
            raise ValueError("confirmatory fresh split overlaps prior exposed IDs or normalized text hashes")
    ordered = tuple(record for role in ("candidate", "development", "scoreboard") for record in selected if record.role == role)
    manifest = DatasetManifest(spec.name, spec.revision, seed,
                               {role: sum(record.role == role for record in ordered) for role in ("candidate", "development", "scoreboard")},
                               exposure_status, ordered, exclusions)
    manifest.validate()
    return manifest


def prepare_official_split(
    train_rows: Iterable[DatasetRow], official_rows: Iterable[DatasetRow], spec: DatasetSpec, *, seed: int,
    development_per_label: int, scoreboard_per_label: int, exposure_status: ExposureStatus,
    prior_exposed: Iterable[ManifestRecord] = (),
) -> DatasetManifest:
    """Build a role-safe study manifest with an explicit official scoreboard source."""
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(development_per_label, int) or isinstance(development_per_label, bool) \
            or not isinstance(scoreboard_per_label, int) or isinstance(scoreboard_per_label, bool):
        raise ValueError("split counts must be integers")
    if development_per_label < 0 or scoreboard_per_label < 1:
        raise ValueError("development target must be nonnegative and scoreboard target must be positive")
    if not isinstance(exposure_status, ExposureStatus):
        raise ValueError("an explicit valid exposure status is required")
    train = _validate_rows(train_rows, spec)
    official = _validate_rows(official_rows, spec)
    if any(row.source_split != "train" for row in train):
        raise ValueError("train rows must all use source split 'train'")
    splits = {row.source_split for row in official}
    if len(splits) != 1:
        raise ValueError("official scoreboard rows must name exactly one source split")
    official_split = next(iter(splits))
    if official_split == "train":
        raise ValueError("official scoreboard split must differ from train")
    scoreboard_source, _ = _deduplicate(official)
    scoreboard_rows = select_official_scoreboard(scoreboard_source, spec, source_split=official_split,
                                                 seed=seed, per_label=scoreboard_per_label)
    scoreboard_hashes = {normalized_text_hash(row.text) for row in scoreboard_rows}
    surviving_train, overlap_exclusions = [], []
    for row in train:
        text_hash = normalized_text_hash(row.text)
        if text_hash in scoreboard_hashes:
            reference = next(item for item in scoreboard_rows if normalized_text_hash(item.text) == text_hash)
            overlap_exclusions.append(DuplicateExclusion(row.id, row.source_split, row.source_index, row.label,
                                                         text_hash, reference.id, "overlaps_scoreboard_text"))
        else:
            surviving_train.append(row)
    deduplicated_train, train_duplicates = _deduplicate(surviving_train)
    candidate_rows, development_rows = [], []
    for offset, label in enumerate(spec.labels):
        group = [row for row in deduplicated_train if row.label == label]
        if len(group) < development_per_label:
            raise ValueError(f"{label} needs {development_per_label} development rows; has {len(group)}")
        shuffled = group.copy()
        random.Random(seed + offset).shuffle(shuffled)
        development_rows.extend(_record(row, "development") for row in shuffled[:development_per_label])
        candidate_rows.extend(_record(row, "candidate") for row in shuffled[development_per_label:])
    records = tuple(candidate_rows + development_rows + [_record(row, "scoreboard") for row in scoreboard_rows])
    all_exclusions = tuple(overlap_exclusions) + train_duplicates
    manifest = DatasetManifest(spec.name, spec.revision, seed,
                               {"candidate": len(candidate_rows), "development": len(development_rows),
                                "scoreboard": len(scoreboard_rows)}, exposure_status, records, all_exclusions)
    if spec.name == "dair-ai/emotion" and official_split in {"test", "validation"} \
            and exposure_status is ExposureStatus.CONFIRMATORY_FRESH:
        raise ValueError("Emotion official test and validation splits are historically exposed")
    if exposure_status is ExposureStatus.CONFIRMATORY_FRESH:
        exposed_ids, exposed_hashes = _exposed_keys(prior_exposed)
        selected_ids = {record.id for record in manifest.scoreboard}
        selected_hashes = {record.normalized_text_sha256 for record in manifest.scoreboard}
        if selected_ids & exposed_ids or selected_hashes & exposed_hashes:
            raise ValueError("confirmatory fresh scoreboard overlaps prior exposed IDs or normalized text hashes")
    manifest.validate()
    return manifest


def _payload(manifest: DatasetManifest) -> dict:
    manifest.validate()
    payload = {"dataset": manifest.dataset, "revision": manifest.revision, "seed": manifest.seed,
            "counts": manifest.counts, "exposure_status": manifest.exposure_status.value,
            "records": [asdict(record) for record in manifest.records], "exclusions": [asdict(record) for record in manifest.exclusions]}
    if manifest.preparation is not None:
        payload["preparation"] = asdict(manifest.preparation)
    return payload


def write_manifest(path: str | Path, manifest: DatasetManifest) -> None:
    """Write the whitelisted, text-free manifest representation."""
    Path(path).write_text(json.dumps(_payload(manifest), sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def read_manifest(path: str | Path) -> DatasetManifest:
    """Load and validate only the text-free manifest schema."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object")
    if set(payload) not in ({"dataset", "revision", "seed", "counts", "exposure_status", "records", "exclusions"},
                            {"dataset", "revision", "seed", "counts", "exposure_status", "records", "exclusions", "preparation"}):
        raise ValueError("manifest has unsupported fields")
    allowed_record = {"id", "source_split", "source_index", "label", "normalized_text_sha256", "role"}
    if not isinstance(payload["records"], list) or any(not isinstance(record, dict) or set(record) != allowed_record
                                                        for record in payload["records"]):
        raise ValueError("manifest record has unsupported fields")
    allowed_exclusion = {"id", "source_split", "source_index", "label", "normalized_text_sha256", "duplicate_of_id", "reason"}
    if not isinstance(payload["exclusions"], list) or any(not isinstance(item, dict) or set(item) != allowed_exclusion
                                                           for item in payload["exclusions"]):
        raise ValueError("manifest exclusion has unsupported fields")
    preparation = payload.get("preparation")
    if preparation is not None and (not isinstance(preparation, dict)
                                    or set(preparation) != {"configuration", "sample_counts", "exposure_history_fingerprint"}):
        raise ValueError("manifest preparation has unsupported fields")
    manifest = DatasetManifest(payload["dataset"], payload["revision"], payload["seed"], payload["counts"],
                               ExposureStatus(payload["exposure_status"]),
                               tuple(ManifestRecord(**record) for record in payload["records"]),
                               tuple(DuplicateExclusion(**item) for item in payload["exclusions"]),
                               PreparationMetadata(**preparation) if preparation is not None else None)
    manifest.validate()
    return manifest
