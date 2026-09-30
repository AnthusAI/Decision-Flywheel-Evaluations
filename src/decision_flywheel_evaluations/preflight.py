"""Pure matrix enumeration: it creates no client and sends no request."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

from decision_flywheel.budget import ContextBudget, build_context_plan
from decision_flywheel.context import PerLabelLexicalRetrieval, PolicyMetadata, PrototypeBalanced, RandomBalanced
from decision_flywheel.models import Item, LabeledItem

from .datasets import DatasetRow, normalized_text_hash
from .manifests import DatasetManifest
from .protocol import FrozenProtocol, Selector


class _FrozenGlobal:
    name = "development-selected-global"
    def __init__(self, ids: tuple[str, ...], checksum: str): self.ids, self.checksum = ids, checksum
    @property
    def metadata(self): return PolicyMetadata(self.name, "1", "fixed-global", {"artifact_checksum": self.checksum})
    @property
    def fingerprint(self): return self.metadata.fingerprint
    def select(self, task, target, candidates, *, per_label):
        by_id = {candidate.item.id: candidate for candidate in candidates}
        if any(identifier not in by_id for identifier in self.ids): raise ValueError("selected-global artifact IDs are not in candidate pool")
        return [by_id[identifier] for identifier in self.ids]


@dataclass(frozen=True)
class RequestCell:
    id: str
    fingerprint: str
    phase: str
    model: str
    selector: str
    per_label: int
    draw_seed: int | None
    target_id: str
    example_ids: tuple[str, ...] = ()
    estimated_request_tokens: int = 0
    counter_identity: str = ""
    wire_fingerprint: str = ""
    task_fingerprint: str = ""
    dataset_revision: str = ""
    display_order: str = ""


@dataclass(frozen=True)
class ExcludedCell:
    phase: str
    model: str
    selector: str
    per_label: int
    draw_seed: int | None
    target_count: int
    reason: str


@dataclass(frozen=True)
class PreflightResult:
    cells: tuple[RequestCell, ...]
    excluded: tuple[ExcludedCell, ...]
    blockers: tuple[str, ...]
    new_request_count: int
    cache_hit_count: int
    optimization_count: int
    scoreboard_count: int
    checksum: str = ""
    stage: str = ""
    selected_stage_complete: bool = False

    @property
    def ready(self) -> bool:
        return self.stage == "scoreboard" and self.selected_stage_complete and not self.blockers


def manifest_sha256(manifest: DatasetManifest) -> str:
    """Hash the complete text-free manifest payload used to rehydrate a study."""
    manifest.validate()
    payload = {"dataset": manifest.dataset, "revision": manifest.revision, "seed": manifest.seed,
               "counts": manifest.counts, "exposure_status": manifest.exposure_status.value,
               "records": [record.__dict__ for record in manifest.records],
               "exclusions": [record.__dict__ for record in manifest.exclusions]}
    if manifest.preparation is not None:
        payload["preparation"] = asdict(manifest.preparation)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def candidate_pool_fingerprint(task, candidates: Iterable[LabeledItem]) -> str:
    """Fingerprint the exact core optimizer pool: sorted ID, canonical label, NFKC text hash."""
    rows = []
    for candidate in candidates:
        if not isinstance(candidate, LabeledItem) or not isinstance(candidate.item.id, str) or not candidate.item.id:
            raise ValueError("candidate pool needs labeled items with stable IDs")
        text = candidate.item.values.get(task.input_field)
        if not isinstance(text, str):
            raise ValueError("candidate pool item lacks the frozen task input")
        rows.append((candidate.item.id, task.validate_label(candidate.label), normalized_text_hash(text)))
    if len({item[0] for item in rows}) != len(rows):
        raise ValueError("candidate pool IDs must be unique")
    return hashlib.sha256(json.dumps(sorted(rows), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _physical_request_fingerprint(model: str, dataset_revision: str, wire_fingerprint: str) -> str:
    """The physical cache identity deliberately excludes logical selector cell names."""
    payload = {"model": model, "dataset_revision": dataset_revision, "wire_fingerprint": wire_fingerprint}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _selected_global_anchor(task, candidates: Iterable[LabeledItem]) -> Item:
    """Make a validation target guaranteed not to trigger the target/candidate firewall."""
    candidate_values = {candidate.item.values.get(task.input_field) for candidate in candidates}
    index = 0
    while True:
        text = f"__selected_global_validation_anchor_{index}__"
        if text not in candidate_values:
            return Item(f"__selected_global_validation_anchor_{index}__", {task.input_field: text})
        index += 1


def rehydrate_manifest(
    manifest: DatasetManifest,
    rows: Iterable[DatasetRow],
    protocol: FrozenProtocol,
    *,
    stage: str = "scoreboard",
):
    """Rehydrate source rows needed for one frozen stage.

    Optimization admits full caller-owned fixtures for compatibility, but reads
    labels and text only for candidate/development records.  Scoreboard source
    rows remain required and fully verified for scoreboard collection.
    """
    if stage not in {"optimization", "scoreboard"}:
        raise ValueError("stage must be optimization or scoreboard")
    manifest.validate()
    protocol.validate()
    if manifest_sha256(manifest) != protocol.dataset_manifest_sha256:
        raise ValueError("supplied manifest does not match frozen protocol hash")
    if manifest.dataset != protocol.dataset or manifest.counts != {"candidate": protocol.candidate_count, "development": protocol.development_count, "scoreboard": protocol.scoreboard_count}:
        raise ValueError("manifest dataset or declared counts do not match protocol")
    source_rows = tuple(rows)
    stage_records = (
        manifest.candidate + manifest.development
        if stage == "optimization" else manifest.records
    )
    records = {record.id: record for record in stage_records}
    permitted_ids = set(records) | ({record.id for record in manifest.scoreboard}
                                    if stage == "optimization" else set())
    supplied = {}
    for row in source_rows:
        row_id = row.id
        if row_id not in permitted_ids:
            raise ValueError("rehydrated rows must exactly match manifest IDs")
        if row_id not in records:
            # A complete caller fixture may include a held-out row.  Do not
            # inspect its source label or text in optimization mode.
            continue
        if row_id in supplied:
            raise ValueError("rehydrated rows must have unique IDs")
        supplied[row_id] = row
    if set(supplied) != set(records):
        raise ValueError("rehydrated rows must exactly match manifest IDs")
    for key, record in records.items():
        row = supplied[key]
        if (row.source_split, row.source_index, row.label, normalized_text_hash(row.text)) != (record.source_split, record.source_index, record.label, record.normalized_text_sha256):
            raise ValueError("rehydrated row does not match manifest provenance")
        protocol.task.validate_label(row.label)
    item = lambda record: Item(record.id, {protocol.task.input_field: supplied[record.id].text})
    candidates = tuple(LabeledItem(item(record), record.label, "trusted") for record in manifest.candidate)
    development = tuple(LabeledItem(item(record), record.label, "trusted") for record in manifest.development)
    scoreboard = (
        tuple(LabeledItem(item(record), record.label, "trusted") for record in manifest.scoreboard)
        if stage == "scoreboard" else ()
    )
    return candidates, development, scoreboard


def core_plan_fingerprints(protocol: FrozenProtocol, manifest: DatasetManifest, rows: Iterable[DatasetRow], *, stage: str = "scoreboard") -> tuple[RequestCell, ...]:
    """Build core ContextPlans offline; returned cells expose only IDs/fingerprints/estimates."""
    if stage not in {"optimization", "scoreboard"}: raise ValueError("stage must be optimization or scoreboard")
    candidates, development, scoreboard = rehydrate_manifest(manifest, rows, protocol, stage=stage)
    pool = candidate_pool_fingerprint(protocol.task, candidates)
    artifact = protocol.selector_search_artifact
    if stage == "scoreboard":
        if artifact is None:
            raise ValueError("scoreboard preflight requires a selected-global artifact")
        artifact.validate_for(protocol, candidates, development, manifest.revision)
    policies = {Selector.RANDOM: lambda seed, size: RandomBalanced(seed), Selector.PROTOTYPE: lambda _, size: PrototypeBalanced(),
                Selector.RETRIEVAL: lambda _, size: PerLabelLexicalRetrieval()}
    def cell(*, phase: str, model: str, selector: Selector, size: int, seed: int | None,
             target_id: str, plan) -> RequestCell:
        wire = plan.token_accounting.serialized_request_fingerprint
        return RequestCell(
            id=f"{phase}:{model}:{selector.value}:{size}:{seed}:{target_id}",
            fingerprint=_physical_request_fingerprint(model, manifest.revision, wire), phase=phase,
            model=model, selector=selector.value, per_label=size, draw_seed=seed, target_id=target_id,
            example_ids=plan.example_ids, estimated_request_tokens=plan.token_accounting.estimated_request_tokens,
            counter_identity=plan.token_accounting.counter_identity, wire_fingerprint=wire,
            task_fingerprint=protocol.task.fingerprint, dataset_revision=manifest.revision,
            display_order=protocol.display_rule,
        )

    cells = []
    targets = development if stage == "optimization" else scoreboard
    for model in protocol.models:
        for target in targets:
            if stage == "optimization":
                for selector, factory in policies.items():
                    if selector not in protocol.optimization.selectors: continue
                    for size in protocol.optimization.per_label_sizes:
                        for seed in protocol.optimization.draw_seeds if selector is Selector.RANDOM else (None,):
                            if not model.supports_few_shot(size): continue
                            plan = build_context_plan(protocol.task, target.item, candidates, factory(seed or 0, size), budget=ContextBudget(per_label=size), display_order=protocol.display_rule, order_seed=0, presentation_label_order=protocol.task.labels)
                            cells.append(cell(phase="optimization", model=model.semantic_identity, selector=selector,
                                              size=size, seed=seed, target_id=target.item.id, plan=plan))
                continue
            if model.supports_zero_shot():
                plan = build_context_plan(protocol.task, target.item, candidates, RandomBalanced(0), budget=ContextBudget(per_label=0), display_order=protocol.display_rule, order_seed=0, presentation_label_order=protocol.task.labels)
                cells.append(cell(phase="scoreboard", model=model.semantic_identity, selector=Selector.ZERO,
                                  size=0, seed=None, target_id=target.item.id, plan=plan))
            for selector, factory in policies.items():
                for size in (1, 4, 16, 64):
                    seeds = protocol.random_draw_seeds if selector is Selector.RANDOM else (None,)
                    for seed in seeds:
                        if not model.supports_few_shot(size): continue
                        plan = build_context_plan(protocol.task, target.item, candidates, factory(seed or 0, size),
                                                  budget=ContextBudget(per_label=size), display_order=protocol.display_rule,
                                                  order_seed=0, presentation_label_order=protocol.task.labels)
                        cells.append(cell(phase="scoreboard", model=model.semantic_identity, selector=selector,
                                          size=size, seed=seed, target_id=target.item.id, plan=plan))
            for size, ids in artifact.ids_by_size.items():
                if not model.supports_few_shot(size): continue
                plan = build_context_plan(protocol.task, target.item, candidates, _FrozenGlobal(ids, artifact.checksum), budget=ContextBudget(per_label=size), display_order=protocol.display_rule, order_seed=0, presentation_label_order=protocol.task.labels)
                cells.append(cell(phase="scoreboard", model=model.semantic_identity,
                                  selector=Selector.DEVELOPMENT_SELECTED_GLOBAL, size=size, seed=None,
                                  target_id=target.item.id, plan=plan))
    return tuple(cells)


def preflight(protocol: FrozenProtocol, *, development_ids: Iterable[str] = (), scoreboard_ids: Iterable[str] = (), stage: str = "scoreboard",
              cached_fingerprints: frozenset[str] = frozenset(), engine: Callable[[RequestCell], object] | None = None,
              manifest: DatasetManifest | None = None, rows: Iterable[DatasetRow] | None = None) -> PreflightResult:
    """Enumerate fixed cells; ``engine`` is accepted only to prove it remains unused."""
    del engine
    protocol.validate()
    if stage not in {"optimization", "scoreboard"}:
        raise ValueError("stage must be optimization or scoreboard")
    if manifest is None or rows is None:
        return PreflightResult((), (), ("a validated DatasetManifest and rehydrated rows are required for core preflight",),
                               0, 0, 0, 0, stage=stage)
    expected_development = {record.id for record in manifest.development}
    expected_scoreboard = {record.id for record in manifest.scoreboard}
    supplied_development, supplied_scoreboard = tuple(development_ids), tuple(scoreboard_ids)
    if supplied_development and set(supplied_development) != expected_development:
        raise ValueError("development IDs must exactly match the manifest development partition")
    if supplied_scoreboard and set(supplied_scoreboard) != expected_scoreboard:
        raise ValueError("scoreboard IDs must exactly match the manifest protected partition")
    runtime = core_plan_fingerprints(protocol, manifest, rows, stage=stage)
    # Physical calls are keyed solely by complete model/wire fingerprints; the
    # logical selector/draw cells retain their own provenance IDs.
    physical = {cell.fingerprint for cell in runtime}
    hits = sum(key in cached_fingerprints for key in physical)
    targets = protocol.development_count if stage == "optimization" else protocol.scoreboard_count
    requested = protocol.optimization.selectors if stage == "optimization" else (Selector.ZERO, Selector.RANDOM, Selector.DEVELOPMENT_SELECTED_GLOBAL, Selector.PROTOTYPE, Selector.RETRIEVAL)
    sizes = protocol.optimization.per_label_sizes if stage == "optimization" else protocol.per_label_sizes
    excluded = _excluded_cells(protocol, stage, requested, sizes, targets)
    checksum_payload = {"protocol": protocol.frozen_payload(), "stage": stage,
                        "cells": [asdict(cell) for cell in runtime], "excluded": [asdict(item) for item in excluded],
                        "physical": sorted(physical)}
    checksum = hashlib.sha256(json.dumps(checksum_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    selected_stage_complete = stage == "scoreboard" and artifact_complete(runtime)
    return PreflightResult(runtime, excluded, (), len(physical) - hits, hits,
                           sum(cell.phase == "optimization" for cell in runtime),
                           sum(cell.phase == "scoreboard" for cell in runtime), checksum, stage,
                           selected_stage_complete)


def artifact_complete(cells: Iterable[RequestCell]) -> bool:
    """Only a scoreboard containing actual selected-global requests may be ready to collect."""
    return any(cell.selector == Selector.DEVELOPMENT_SELECTED_GLOBAL.value for cell in cells)


def _excluded_cells(protocol: FrozenProtocol, stage: str, requested: tuple[Selector, ...],
                    sizes: tuple[int, ...], targets: int) -> tuple[ExcludedCell, ...]:
    exclusions = []
    for model in protocol.models:
        for selector in requested:
            if selector is Selector.ZERO:
                if not model.supports_zero_shot():
                    exclusions.append(ExcludedCell(stage, model.semantic_identity, selector.value, 0, None, targets,
                                                    "zero-shot capability unsupported"))
                continue
            for size in sizes:
                if size == 0 or model.supports_few_shot(size):
                    continue
                for seed in protocol.random_draw_seeds if selector is Selector.RANDOM else (None,):
                    exclusions.append(ExcludedCell(stage, model.semantic_identity, selector.value, size, seed, targets,
                                                    "few-shot capability or context limit unsupported"))
    return tuple(exclusions)
