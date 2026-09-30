"""Pure planning for a frozen post-scoreboard presentation-order study."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

from decision_flywheel.budget import ContextBudget, ContextPlan, build_context_plan
from decision_flywheel.context import RandomBalanced
from decision_flywheel.models import Item

from .datasets import DatasetRow
from .manifests import DatasetManifest
from .metrics import Observation
from .preflight import (PreflightResult, RequestCell, _FrozenGlobal,
                        _physical_request_fingerprint, manifest_sha256,
                        preflight, rehydrate_manifest)
from .protocol import FrozenProtocol, OrderTreatment, Selector, validate_result_reference
from .reporting import report_study


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,511}$")
_COUNTER_IDENTITY = "whitespace-request-estimate-v1"
_POSITIVE_SIZES = (1, 4, 16, 64)
_DRAW_SEEDS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class OrderingCell:
    """One logical ordering treatment derived from an initial request cell."""

    source_request_id: str
    treatment: OrderTreatment
    request: RequestCell


@dataclass(frozen=True)
class OrderingPlan:
    """Text-free, follow-up plan anchored to complete canonical evidence."""

    initial_protocol_identity: str
    manifest_sha256: str
    initial_preflight_checksum: str
    initial_observations_sha256: str
    initial_result_artifact: str
    treatments: tuple[OrderTreatment, ...]
    cells: tuple[OrderingCell, ...]
    checksum: str

    @property
    def physical_count(self) -> int:
        self.validate()
        return len({cell.request.fingerprint for cell in self.cells})

    @property
    def logical_count(self) -> int:
        self.validate()
        return len(self.cells)

    def validate(self) -> None:
        for value in (
            self.initial_protocol_identity,
            self.manifest_sha256,
            self.initial_preflight_checksum,
            self.initial_observations_sha256,
            self.checksum,
        ):
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValueError("ordering plan identity fields must be SHA-256 values")
        validate_result_reference(self.initial_result_artifact)
        if (not isinstance(self.treatments, tuple) or not self.treatments
                or any(type(item) is not OrderTreatment for item in self.treatments)):
            raise ValueError("ordering plan treatments must be typed and non-empty")
        for treatment in self.treatments:
            treatment.validate()
        identities = tuple((item.display_order, item.seed) for item in self.treatments)
        if len(set(identities)) != len(identities):
            raise ValueError("ordering plan treatments must be unique")
        _validate_treatment_family(self.treatments)
        if not isinstance(self.cells, tuple) or not self.cells:
            raise ValueError("ordering plan cells must be a non-empty tuple")
        if any(type(cell) is not OrderingCell for cell in self.cells):
            raise ValueError("ordering plan cells must be OrderingCell values")
        for cell in self.cells:
            _validate_ordering_cell(cell)
        request_ids = [cell.request.id for cell in self.cells]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("ordering plan logical request IDs must be unique")

        expected_treatments = set(identities)
        by_source: dict[str, list[OrderingCell]] = {}
        for cell in self.cells:
            by_source.setdefault(cell.source_request_id, []).append(cell)
        models: set[str] = set()
        tasks: set[str] = set()
        revisions: set[str] = set()
        conditions_by_target: dict[str, set[tuple[str, int, int | None]]] = {}
        label_counts: set[int] = set()
        for source_id, group in by_source.items():
            baseline = group[0].request
            expected_source_id = (
                f"scoreboard:{baseline.model}:{baseline.selector}:{baseline.per_label}:"
                f"{baseline.draw_seed}:{baseline.target_id}"
            )
            if source_id != expected_source_id:
                raise ValueError("ordering source request ID does not match frozen request metadata")
            if any(
                (cell.request.model, cell.request.selector, cell.request.per_label,
                 cell.request.draw_seed, cell.request.target_id, frozenset(cell.request.example_ids),
                 cell.request.task_fingerprint, cell.request.dataset_revision,
                 cell.request.counter_identity)
                != (baseline.model, baseline.selector, baseline.per_label,
                    baseline.draw_seed, baseline.target_id, frozenset(baseline.example_ids),
                    baseline.task_fingerprint, baseline.dataset_revision,
                    baseline.counter_identity)
                for cell in group
            ):
                raise ValueError("ordering treatments cannot change frozen source membership")
            if any(cell.request.id != f"ordering:{source_id}:{cell.treatment.label}" for cell in group):
                raise ValueError("ordering logical request ID does not match its source and treatment")
            group_treatments = {(cell.treatment.display_order, cell.treatment.seed) for cell in group}
            if baseline.per_label == 0:
                if len(group) != 1 or group_treatments != {("canonical", 0)}:
                    raise ValueError("ordering zero-shot requests appear only once in canonical order")
            elif len(group) != len(self.treatments) or group_treatments != expected_treatments:
                raise ValueError("ordering positive source requests must cover every treatment")
            models.add(baseline.model)
            tasks.add(baseline.task_fingerprint)
            revisions.add(baseline.dataset_revision)
            conditions_by_target.setdefault(baseline.target_id, set()).add(
                (baseline.selector, baseline.per_label, baseline.draw_seed)
            )
            if baseline.per_label > 0:
                if len(baseline.example_ids) % baseline.per_label:
                    raise ValueError("ordering frozen memberships do not have a whole label count")
                label_counts.add(len(baseline.example_ids) // baseline.per_label)
        if (len(models) != 1 or len(tasks) != 1 or len(revisions) != 1
                or not next(iter(models)).startswith("jev:")):
            raise ValueError("ordering plan must use one JEV model, task, and dataset revision")
        if len(label_counts) != 1 or next(iter(label_counts)) not in {4, 6}:
            raise ValueError("ordering plan cannot infer a supported fixed label count")
        expected_conditions = {(Selector.ZERO.value, 0, None)}
        expected_conditions |= {(Selector.RANDOM.value, size, seed)
                                for size in _POSITIVE_SIZES for seed in _DRAW_SEEDS}
        expected_conditions |= {(selector, size, None)
                                for selector in (Selector.DEVELOPMENT_SELECTED_GLOBAL.value,
                                                 Selector.PROTOTYPE.value, Selector.RETRIEVAL.value)
                                for size in _POSITIVE_SIZES}
        if any(conditions != expected_conditions
               for conditions in conditions_by_target.values()):
            raise ValueError("ordering plan must retain every complete 33-condition source group")
        if self.checksum != _checksum(self):
            raise ValueError("ordering plan checksum does not match its text-free cells")


def plan_ordering(
    protocol: FrozenProtocol,
    manifest: DatasetManifest,
    rows: Iterable[DatasetRow],
    initial_preflight: PreflightResult,
    initial_observations: Sequence[Observation],
    initial_result_artifact: str,
    treatments: tuple[OrderTreatment, ...],
) -> OrderingPlan:
    """Freeze a presentation-only study after complete canonical JEV scoreboard evidence."""
    protocol.validate()
    validate_result_reference(initial_result_artifact)
    if not isinstance(treatments, tuple):
        raise ValueError("ordering treatments must be an explicit tuple")
    _validate_treatments(treatments)
    source_rows = tuple(rows)
    _validate_initial_preflight(protocol, manifest, source_rows, initial_preflight)
    source_cells = _validate_initial_matrix(protocol, initial_preflight)
    observations = tuple(initial_observations)
    _validate_initial_observations(protocol, manifest, source_cells, observations)
    candidates, _development, scoreboard = rehydrate_manifest(
        manifest, source_rows, protocol, stage="scoreboard"
    )
    targets = {item.item.id: item.item for item in scoreboard}
    cells: list[OrderingCell] = []
    for source in source_cells:
        applicable = ((OrderTreatment("canonical", 0),)
                      if source.per_label == 0 else treatments)
        for treatment in applicable:
            target = targets.get(source.target_id)
            if target is None:
                raise ValueError("initial ordering source cell is outside the scoreboard")
            request = _ordered_request(protocol, manifest, candidates, source, treatment, target)
            cells.append(OrderingCell(source.id, treatment, request))
    plan = OrderingPlan(
        protocol.identity,
        manifest_sha256(manifest),
        initial_preflight.checksum,
        _observations_checksum(observations),
        initial_result_artifact,
        treatments,
        tuple(cells),
        "",
    )
    plan = replace(plan, checksum=_checksum(plan))
    plan.validate()
    return plan


def ordering_execution(
    protocol: FrozenProtocol,
    manifest: DatasetManifest,
    rows: Iterable[DatasetRow],
    plan: OrderingPlan,
) -> tuple[tuple[RequestCell, Item, ContextPlan], ...]:
    """Rebuild and verify every frozen treatment plan without a provider factory."""
    protocol.validate()
    if not isinstance(plan, OrderingPlan):
        raise ValueError("ordering execution requires an OrderingPlan")
    plan.validate()
    if (plan.initial_protocol_identity != protocol.identity
            or plan.manifest_sha256 != manifest_sha256(manifest)):
        raise ValueError("ordering plan does not match the frozen protocol and manifest")
    candidates, _development, scoreboard = rehydrate_manifest(
        manifest, tuple(rows), protocol, stage="scoreboard"
    )
    jev_models = tuple(model for model in protocol.models if model.engine.value == "jev")
    if len(jev_models) != 1:
        raise ValueError("ordering execution requires exactly one frozen JEV model")
    model_identity = jev_models[0].semantic_identity
    label_count = len(protocol.task.labels)
    targets = {item.item.id: item.item for item in scoreboard}
    if {cell.request.target_id for cell in plan.cells} != set(targets):
        raise ValueError("ordering plan does not cover the full scoreboard target set")
    output = []
    for cell in plan.cells:
        request = cell.request
        if (request.model != model_identity
                or request.task_fingerprint != protocol.task.fingerprint
                or request.dataset_revision != manifest.revision
                or len(request.example_ids) != request.per_label * label_count):
            raise ValueError("ordering request does not match the frozen execution contract")
        target = targets.get(request.target_id)
        if target is None:
            raise ValueError("ordering request target is outside the scoreboard")
        context = _build_context(protocol, candidates, request, cell.treatment, target)
        if (context.example_ids != request.example_ids
                or context.token_accounting.serialized_request_fingerprint != request.wire_fingerprint
                or context.token_accounting.estimated_request_tokens != request.estimated_request_tokens
                or context.token_accounting.counter_identity != request.counter_identity):
            raise ValueError("ordering request does not match its rehydrated frozen context")
        output.append((request, target, context))
    return tuple(output)


def _validate_treatments(treatments: tuple[OrderTreatment, ...]) -> None:
    if not treatments or any(type(item) is not OrderTreatment for item in treatments):
        raise ValueError("ordering treatments must be typed and non-empty")
    for treatment in treatments:
        treatment.validate()
    identities = {(item.display_order, item.seed) for item in treatments}
    if len(identities) != len(treatments):
        raise ValueError("ordering treatments must be unique")
    _validate_treatment_family(treatments)


def _validate_treatment_family(treatments: tuple[OrderTreatment, ...]) -> None:
    identities = {(item.display_order, item.seed) for item in treatments}
    if not {("canonical", 0), ("interleaved", 0), ("reversed", 0)} <= identities:
        raise ValueError("ordering treatments require canonical, interleaved, and reversed seed-zero arms")
    shuffled = {seed for order, seed in identities if order == "shuffled"}
    if len(shuffled) < 2:
        raise ValueError("ordering treatments require at least two distinct shuffled seeds")


def _validate_initial_preflight(protocol: FrozenProtocol, manifest: DatasetManifest,
                                rows: tuple[DatasetRow, ...], initial: PreflightResult) -> None:
    if (not isinstance(initial, PreflightResult) or not initial.ready
            or initial.stage != "scoreboard" or initial.blockers):
        raise ValueError("a complete initial canonical scoreboard preflight is required")
    regenerated = preflight(protocol, manifest=manifest, rows=rows, stage="scoreboard")
    physical_count = len({cell.fingerprint for cell in regenerated.cells})
    immutable_matches = (
        initial.cells == regenerated.cells
        and initial.excluded == regenerated.excluded
        and initial.blockers == regenerated.blockers
        and initial.checksum == regenerated.checksum
        and initial.stage == regenerated.stage
        and initial.selected_stage_complete == regenerated.selected_stage_complete
        and initial.optimization_count == regenerated.optimization_count
        and initial.scoreboard_count == regenerated.scoreboard_count
    )
    counter_types_valid = all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in (initial.new_request_count, initial.cache_hit_count)
    )
    if (not immutable_matches or not counter_types_valid
            or initial.new_request_count + initial.cache_hit_count != physical_count):
        raise ValueError("ordering requires the exact regenerated initial preflight")


def _validate_initial_matrix(protocol: FrozenProtocol, initial: PreflightResult) -> tuple[RequestCell, ...]:
    jev = tuple(model for model in protocol.models if model.engine.value == "jev")
    if len(jev) != 1:
        raise ValueError("ordering requires exactly one frozen JEV model")
    source = tuple(initial.cells)
    if any(cell.phase != "scoreboard" or cell.model != jev[0].semantic_identity
           or cell.display_order != "canonical" for cell in source):
        raise ValueError("initial ordering matrix must be a single-JEV canonical scoreboard")
    expected = {(Selector.ZERO.value, 0, None)}
    expected |= {(Selector.RANDOM.value, size, seed) for size in _POSITIVE_SIZES for seed in _DRAW_SEEDS}
    expected |= {(selector, size, None)
                 for selector in (Selector.DEVELOPMENT_SELECTED_GLOBAL.value, Selector.PROTOTYPE.value,
                                  Selector.RETRIEVAL.value)
                 for size in _POSITIVE_SIZES}
    by_target: dict[str, set[tuple[str, int, int | None]]] = {}
    for cell in source:
        by_target.setdefault(cell.target_id, set()).add((cell.selector, cell.per_label, cell.draw_seed))
    if len(source) != protocol.scoreboard_count * len(expected) or any(group != expected for group in by_target.values()):
        raise ValueError("initial JEV scoreboard does not contain the complete 33-condition matrix")
    return source


def _validate_initial_observations(protocol: FrozenProtocol, manifest: DatasetManifest,
                                   source: tuple[RequestCell, ...], rows: tuple[Observation, ...]) -> None:
    try:
        report_study(rows, protocol.task.labels, expected_logical_ids=tuple(cell.id for cell in source))
    except ValueError as error:
        raise ValueError("complete initial observations must exactly cover the frozen scoreboard") from error
    records = {record.id: record for record in manifest.scoreboard}
    by_id = {row.request_id: row for row in rows}
    for cell in source:
        row = by_id[cell.id]
        record = records.get(cell.target_id)
        expected_provenance = {"model": cell.model, "dataset_revision": cell.dataset_revision,
                               "task_fingerprint": cell.task_fingerprint, "wire_fingerprint": cell.wire_fingerprint}
        if (record is None or row.status != "completed" or row.target_id != cell.target_id
                or row.true_label != record.label or row.model_id != cell.model
                or row.condition != f"scoreboard:{cell.model}:{cell.selector}:{cell.per_label}"
                or row.draw != (cell.draw_seed or 0) or row.order != "canonical"
                or row.physical_request_id != cell.fingerprint
                or dict(row.physical_request_provenance or {}) != expected_provenance):
            raise ValueError("complete initial observations do not match frozen scoreboard provenance")


def _ordered_request(protocol: FrozenProtocol, manifest: DatasetManifest, candidates, source: RequestCell,
                     treatment: OrderTreatment, target: Item) -> RequestCell:
    context = _build_context(protocol, candidates, source, treatment, target)
    wire = context.token_accounting.serialized_request_fingerprint
    request = RequestCell(
        id=f"ordering:{source.id}:{treatment.label}",
        fingerprint=_physical_request_fingerprint(source.model, manifest.revision, wire),
        phase="scoreboard", model=source.model, selector=source.selector,
        per_label=source.per_label, draw_seed=source.draw_seed, target_id=source.target_id,
        example_ids=context.example_ids,
        estimated_request_tokens=context.token_accounting.estimated_request_tokens,
        counter_identity=context.token_accounting.counter_identity,
        wire_fingerprint=wire, task_fingerprint=source.task_fingerprint,
        dataset_revision=source.dataset_revision, display_order=treatment.label,
    )
    if frozenset(request.example_ids) != frozenset(source.example_ids):
        raise ValueError("ordering treatment changed frozen initial context membership")
    if treatment.display_order == "canonical" and (
            request.wire_fingerprint != source.wire_fingerprint
            or request.fingerprint != source.fingerprint):
        raise ValueError("canonical ordering treatment does not exactly reproduce the initial physical request")
    return request


def _build_context(protocol: FrozenProtocol, candidates, source: RequestCell,
                   treatment: OrderTreatment, target: Item) -> ContextPlan:
    policy = (RandomBalanced(0) if source.per_label == 0
              else _FrozenGlobal(source.example_ids, source.fingerprint))
    return build_context_plan(
        protocol.task, target, candidates, policy,
        budget=ContextBudget(per_label=source.per_label),
        display_order=treatment.display_order, order_seed=treatment.seed,
        presentation_label_order=protocol.task.labels,
    )


def _validate_ordering_cell(cell: OrderingCell) -> None:
    if not isinstance(cell.source_request_id, str) or not _SAFE_ID.fullmatch(cell.source_request_id):
        raise ValueError("ordering source request ID must be safe")
    if type(cell.treatment) is not OrderTreatment or type(cell.request) is not RequestCell:
        raise ValueError("ordering cell has invalid typed metadata")
    cell.treatment.validate()
    request = cell.request
    identifiers = (request.id, request.model, request.selector, request.target_id, request.counter_identity)
    if (any(not isinstance(value, str) or not _SAFE_ID.fullmatch(value) for value in identifiers)
            or request.phase != "scoreboard" or not request.id.startswith("ordering:")
            or request.display_order != cell.treatment.label
            or request.counter_identity != _COUNTER_IDENTITY
            or not isinstance(request.per_label, int) or isinstance(request.per_label, bool)
            or request.per_label < 0 or not isinstance(request.estimated_request_tokens, int)
            or isinstance(request.estimated_request_tokens, bool) or request.estimated_request_tokens < 0
            or not isinstance(request.example_ids, tuple)
            or any(not isinstance(value, str) or not _SAFE_ID.fullmatch(value) for value in request.example_ids)
            or request.target_id in request.example_ids):
        raise ValueError("ordering request has invalid text-free metadata")
    if len(set(request.example_ids)) != len(request.example_ids):
        raise ValueError("ordering request examples must be unique")
    if request.per_label == 0:
        if (request.selector != Selector.ZERO.value or request.draw_seed is not None
                or request.example_ids):
            raise ValueError("ordering zero-shot request has invalid frozen metadata")
    elif request.per_label not in _POSITIVE_SIZES:
        raise ValueError("ordering request has an unsupported frozen context size")
    elif request.selector == Selector.RANDOM.value:
        if (not isinstance(request.draw_seed, int) or isinstance(request.draw_seed, bool)
                or request.draw_seed not in _DRAW_SEEDS):
            raise ValueError("ordering random request has an invalid frozen draw")
    elif request.selector not in {
        Selector.DEVELOPMENT_SELECTED_GLOBAL.value,
        Selector.PROTOTYPE.value,
        Selector.RETRIEVAL.value,
    } or request.draw_seed is not None:
        raise ValueError("ordering request has invalid frozen selector metadata")
    for value, pattern in ((request.fingerprint, _SHA256), (request.wire_fingerprint, _SHA256),
                           (request.task_fingerprint, _SHA256), (request.dataset_revision, _SHA1)):
        if not isinstance(value, str) or not pattern.fullmatch(value):
            raise ValueError("ordering request has invalid fingerprint metadata")
    if request.fingerprint != _physical_request_fingerprint(request.model, request.dataset_revision,
                                                            request.wire_fingerprint):
        raise ValueError("ordering request physical fingerprint does not match its wire")


def _observations_checksum(rows: Sequence[Observation]) -> str:
    payload = []
    for row in sorted(rows, key=lambda item: item.request_id):
        payload.append({
            "request_id": row.request_id, "target_id": row.target_id, "condition": row.condition,
            "draw": row.draw, "order": row.order, "true_label": row.true_label,
            "predicted_label": row.predicted_label, "status": row.status,
            "probabilities": dict(row.probabilities) if row.probabilities is not None else None,
            "model_id": row.model_id, "usage": dict(row.usage) if row.usage is not None else None,
            "latency_ms": row.latency_ms, "attempt_count": row.attempt_count, "cache_hit": row.cache_hit,
            "physical_request_id": row.physical_request_id,
            "physical_request_provenance": (dict(row.physical_request_provenance)
                                            if row.physical_request_provenance is not None else None),
            "confidence": row.confidence,
        })
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _checksum(plan: OrderingPlan) -> str:
    payload = {
        "initial_protocol_identity": plan.initial_protocol_identity,
        "manifest_sha256": plan.manifest_sha256,
        "initial_preflight_checksum": plan.initial_preflight_checksum,
        "initial_observations_sha256": plan.initial_observations_sha256,
        "initial_result_artifact": plan.initial_result_artifact,
        "treatments": [(item.display_order, item.seed) for item in plan.treatments],
        "cells": [
            {"source_request_id": cell.source_request_id,
             "treatment": (cell.treatment.display_order, cell.treatment.seed),
             "request": {name: getattr(cell.request, name) for name in RequestCell.__dataclass_fields__}}
            for cell in plan.cells
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
