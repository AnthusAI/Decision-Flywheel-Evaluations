"""Pure derivation of a small, development-only JEV capability pilot."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from typing import Iterable

from decision_flywheel.budget import ContextBudget, build_context_plan
from decision_flywheel.context import RandomBalanced

from .datasets import DatasetRow
from .manifests import DatasetManifest
from .preflight import (PreflightResult, RequestCell, _physical_request_fingerprint,
                        manifest_sha256, preflight, rehydrate_manifest)
from .protocol import FrozenProtocol, Selector


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,511}$")
_PILOT_SIZES = (1, 4, 16, 64)
_PILOT_SEEDS = (0, 1, 2, 3, 4)
_DISPLAY_ORDERS = frozenset({"canonical", "reversed", "interleaved", "shuffled"})
_COUNTER_IDENTITY = "whitespace-request-estimate-v1"


@dataclass(frozen=True)
class PilotPlan:
    """A text-free, integrity-checked set of development pilot requests."""

    protocol_identity: str
    manifest_sha256: str
    source_preflight_checksum: str
    cells: tuple[RequestCell, ...]
    checksum: str

    @property
    def physical_count(self) -> int:
        """Distinct physical requests, derived from (not assumed by) logical cells."""
        self.validate()
        return len({cell.fingerprint for cell in self.cells})

    def validate(self) -> None:
        """Validate static, text-free plan integrity for serialization consumers.

        Regenerating the source preflight and validating that these cells are
        development-only remains the responsibility of :func:`plan_jev_pilot`.
        """
        for value in (
            self.protocol_identity,
            self.manifest_sha256,
            self.source_preflight_checksum,
            self.checksum,
        ):
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValueError("pilot plan identity fields must be SHA-256 values")
        if not isinstance(self.cells, tuple) or len(self.cells) != 21:
            raise ValueError("pilot plan must contain exactly 21 logical cells")
        if any(type(cell) is not RequestCell for cell in self.cells):
            raise ValueError("pilot plan cells must be RequestCell values")
        for cell in self.cells:
            _validate_cell_shape(cell)
        if len({cell.id for cell in self.cells}) != len(self.cells):
            raise ValueError("pilot plan logical cell IDs must be unique")

        models: set[str] = set()
        tasks: set[str] = set()
        revisions: set[str] = set()
        orders: set[str] = set()
        positive_groups: set[tuple[int, int]] = set()
        zero_count = 0
        for cell in self.cells:
            models.add(cell.model)
            tasks.add(cell.task_fingerprint)
            revisions.add(cell.dataset_revision)
            orders.add(cell.display_order)
            if cell.per_label == 0:
                zero_count += 1
            else:
                positive_groups.add((cell.per_label, cell.draw_seed))
        if len(models) != 1 or len(tasks) != 1 or len(revisions) != 1 or len(orders) != 1:
            raise ValueError("pilot plan cells must have one model, task, revision, and display order")
        expected_groups = {(size, seed) for size in _PILOT_SIZES for seed in _PILOT_SEEDS}
        if zero_count != 1 or positive_groups != expected_groups:
            raise ValueError("pilot plan must contain one zero-shot cell and every size/seed group")
        if self.checksum != _checksum(
            self.protocol_identity,
            self.manifest_sha256,
            self.source_preflight_checksum,
            self.cells,
        ):
            raise ValueError("pilot plan checksum does not match its text-free cells")


def plan_jev_pilot(
    protocol: FrozenProtocol,
    manifest: DatasetManifest,
    rows: Iterable[DatasetRow],
    optimization_preflight: PreflightResult,
) -> PilotPlan:
    """Derive exactly 20 worst-case few-shot cells and one matching zero-shot cell.

    This only rehydrates candidate and development source rows.  A supplied
    optimization preflight must exactly equal a fresh, cache-free regeneration.
    """
    protocol.validate()
    source_rows = tuple(rows)
    jev = _exact_pilot_jev(protocol)
    _validate_source_preflight(protocol, manifest, source_rows, optimization_preflight)
    candidates, development, _ = rehydrate_manifest(
        manifest,
        source_rows,
        protocol,
        stage="optimization",
    )
    source_cells = tuple(cell for cell in optimization_preflight.cells if cell.model == jev.semantic_identity)
    _validate_source_matrix(source_cells, development, jev.semantic_identity)

    chosen = []
    for size in _PILOT_SIZES:
        for seed in _PILOT_SEEDS:
            group = tuple(
                cell for cell in source_cells
                if cell.per_label == size and cell.draw_seed == seed
            )
            selected = min(group, key=lambda cell: (-cell.estimated_request_tokens, cell.id))
            chosen.append(_pilot_copy(selected))

    zero_target_id = min(chosen, key=lambda cell: (-cell.estimated_request_tokens, cell.id)).target_id
    by_target = {item.item.id: item for item in development}
    target = by_target.get(zero_target_id)
    if target is None:
        raise ValueError("selected pilot target is outside the development split")
    zero = _zero_shot_cell(protocol, manifest, candidates, target.item.id, target.item)
    plan_cells = tuple(chosen) + (zero,)
    _validate_builder_cells(plan_cells, protocol, manifest, jev.semantic_identity)
    plan = PilotPlan(
        protocol.identity,
        manifest_sha256(manifest),
        optimization_preflight.checksum,
        plan_cells,
        _checksum(protocol.identity, manifest_sha256(manifest), optimization_preflight.checksum, plan_cells),
    )
    plan.validate()
    return plan


def _exact_pilot_jev(protocol: FrozenProtocol):
    jev_models = tuple(model for model in protocol.models if model.engine.value == "jev")
    if len(jev_models) != 1:
        raise ValueError("pilot requires exactly one frozen JEV model")
    model = jev_models[0]
    if not model.supports_zero_shot() or not model.supports_few_shot(64):
        raise ValueError("pilot JEV model must declare zero-shot and 64-example capabilities")
    return model


def _validate_source_preflight(
    protocol: FrozenProtocol,
    manifest: DatasetManifest,
    rows: tuple[DatasetRow, ...],
    source: PreflightResult,
) -> None:
    if (not isinstance(source, PreflightResult)
            or source.stage != "optimization"
            or source.blockers):
        raise ValueError("an unblocked optimization preflight is required")
    regenerated = preflight(protocol, manifest=manifest, rows=rows, stage="optimization")
    if source != regenerated:
        raise ValueError("pilot requires the exact regenerated optimization preflight")


def _validate_source_matrix(source_cells, development, model_identity: str) -> None:
    expected_targets = {item.item.id for item in development}
    expected_groups = {(size, seed) for size in _PILOT_SIZES for seed in _PILOT_SEEDS}
    if len(source_cells) != len(expected_targets) * len(expected_groups):
        raise ValueError("JEV optimization preflight does not cover the complete pilot matrix")
    if any(cell.phase != "optimization" or cell.selector != Selector.RANDOM.value
           or cell.model != model_identity or cell.target_id not in expected_targets
           for cell in source_cells):
        raise ValueError("JEV optimization preflight has invalid development pilot cells")
    groups = {(cell.per_label, cell.draw_seed) for cell in source_cells}
    if groups != expected_groups:
        raise ValueError("JEV optimization preflight does not cover every declared size and seed")
    for size, seed in expected_groups:
        group_targets = {
            cell.target_id for cell in source_cells
            if (cell.per_label, cell.draw_seed) == (size, seed)
        }
        if group_targets != expected_targets:
            raise ValueError("JEV optimization preflight does not cover every development target")


def _pilot_copy(cell: RequestCell) -> RequestCell:
    return replace(cell, id=cell.id.replace("optimization:", "pilot:", 1), phase="pilot")


def _zero_shot_cell(protocol: FrozenProtocol, manifest: DatasetManifest, candidates, target_id: str, target) -> RequestCell:
    plan = build_context_plan(
        protocol.task,
        target,
        candidates,
        RandomBalanced(0),
        budget=ContextBudget(per_label=0),
        display_order=protocol.display_rule,
        order_seed=0,
        presentation_label_order=protocol.task.labels,
    )
    wire = plan.token_accounting.serialized_request_fingerprint
    model = _exact_pilot_jev(protocol).semantic_identity
    return RequestCell(
        id=f"pilot:{model}:{Selector.ZERO.value}:0:None:{target_id}",
        fingerprint=_physical_request_fingerprint(model, manifest.revision, wire),
        phase="pilot",
        model=model,
        selector=Selector.ZERO.value,
        per_label=0,
        draw_seed=None,
        target_id=target_id,
        example_ids=plan.example_ids,
        estimated_request_tokens=plan.token_accounting.estimated_request_tokens,
        counter_identity=plan.token_accounting.counter_identity,
        wire_fingerprint=wire,
        task_fingerprint=protocol.task.fingerprint,
        dataset_revision=manifest.revision,
        display_order=protocol.display_rule,
    )


def _validate_builder_cells(cells: tuple[RequestCell, ...], protocol: FrozenProtocol,
                            manifest: DatasetManifest, model_identity: str) -> None:
    label_count = len(protocol.task.labels)
    for cell in cells:
        if (cell.model != model_identity
                or cell.task_fingerprint != protocol.task.fingerprint
                or cell.dataset_revision != manifest.revision
                or cell.display_order != protocol.display_rule
                or len(cell.example_ids) != cell.per_label * label_count):
            raise ValueError("pilot cells do not preserve the frozen core request contract")


def _validate_cell_shape(cell: RequestCell) -> None:
    identifiers = (cell.id, cell.model, cell.selector, cell.target_id)
    if any(not isinstance(value, str) or not _SAFE_ID.fullmatch(value) for value in identifiers):
        raise ValueError("pilot plan cell has invalid text-free metadata")
    if cell.phase != "pilot" or not cell.id.startswith("pilot:"):
        raise ValueError("pilot plan cell has invalid text-free metadata")
    for value, pattern in (
        (cell.fingerprint, _SHA256),
        (cell.wire_fingerprint, _SHA256),
        (cell.task_fingerprint, _SHA256),
        (cell.dataset_revision, _SHA1),
    ):
        if not isinstance(value, str) or not pattern.fullmatch(value):
            raise ValueError("pilot plan cell has invalid text-free metadata")
    if (not isinstance(cell.display_order, str)
            or cell.display_order not in _DISPLAY_ORDERS
            or not isinstance(cell.estimated_request_tokens, int)
            or isinstance(cell.estimated_request_tokens, bool)
            or cell.estimated_request_tokens < 0
            or not isinstance(cell.per_label, int)
            or isinstance(cell.per_label, bool)
            or cell.per_label < 0
            or not isinstance(cell.example_ids, tuple)):
        raise ValueError("pilot plan cell has invalid text-free metadata")
    if cell.counter_identity != _COUNTER_IDENTITY:
        raise ValueError("pilot plan cell has an unexpected token counter identity")
    if any(not isinstance(value, str) or not _SAFE_ID.fullmatch(value) for value in cell.example_ids):
        raise ValueError("pilot plan cell has invalid text-free metadata")
    if len(set(cell.example_ids)) != len(cell.example_ids):
        raise ValueError("pilot plan cell has invalid text-free metadata")
    if cell.target_id in cell.example_ids:
        raise ValueError("pilot plan target must not appear in its examples")
    if cell.fingerprint != _physical_request_fingerprint(
        cell.model,
        cell.dataset_revision,
        cell.wire_fingerprint,
    ):
        raise ValueError("pilot plan physical request fingerprint does not match its wire metadata")
    if cell.per_label == 0:
        if cell.selector != Selector.ZERO.value or cell.draw_seed is not None or cell.example_ids:
            raise ValueError("pilot zero-shot cell must have no examples or draw")
        return
    if (cell.selector != Selector.RANDOM.value
            or cell.per_label not in _PILOT_SIZES
            or not isinstance(cell.draw_seed, int)
            or isinstance(cell.draw_seed, bool)
            or cell.draw_seed not in _PILOT_SEEDS):
        raise ValueError("pilot few-shot cell has an invalid size, selector, or seed")


def _checksum(protocol_identity: str, manifest_identity: str, source_preflight_checksum: str,
              cells: tuple[RequestCell, ...]) -> str:
    payload = {
        "protocol_identity": protocol_identity,
        "manifest_sha256": manifest_identity,
        "source_preflight_checksum": source_preflight_checksum,
        "cells": [asdict(cell) for cell in cells],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
