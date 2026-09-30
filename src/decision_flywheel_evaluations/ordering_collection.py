"""Bounded collection for a frozen post-scoreboard ordering plan."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .cache import ApprovedRequest, CacheStore
from .collector import (CollectionApproval, CollectionOptions, CollectionResult,
                        CommittedChecker, EngineFactory, _collect_execution,
                        git_preregistration_is_committed)
from .datasets import DatasetRow
from .manifests import DatasetManifest
from .metrics import Observation
from .ordering import OrderingPlan, ordering_execution, plan_ordering
from .preflight import PreflightResult, manifest_sha256
from .protocol import FrozenProtocol


async def collect_ordering(
    protocol: FrozenProtocol,
    manifest: DatasetManifest,
    rows: Iterable[DatasetRow],
    initial_preflight: PreflightResult,
    initial_observations: Sequence[Observation],
    ordering_plan: OrderingPlan,
    cache: CacheStore,
    approval: CollectionApproval,
    engine_factory: EngineFactory,
    *,
    committed_checker: CommittedChecker = None,  # type: ignore[assignment]
    options: CollectionOptions = CollectionOptions(),
) -> CollectionResult:
    """Collect only the exact frozen ordering matrix in its dedicated ledger."""
    checker = committed_checker or git_preregistration_is_committed
    source_rows = tuple(rows)
    _validate_ordering_gates(protocol, manifest, source_rows, initial_preflight,
                             initial_observations, ordering_plan, cache, approval,
                             checker, options)
    execution = ordering_execution(protocol, manifest, source_rows, ordering_plan)
    return await _collect_execution(protocol, manifest, execution, cache, engine_factory, options)


def validate_ordering_preregistration(
    protocol: FrozenProtocol,
    initial_preflight: PreflightResult,
    ordering_plan: OrderingPlan,
    approval: CollectionApproval,
    committed_checker: CommittedChecker,
) -> None:
    """Validate the committed ordering authorization before opening a ledger."""
    approval.validate()
    preregistration = Path(approval.preregistration_path)
    try:
        content = preregistration.read_bytes()
    except OSError as error:
        raise ValueError("approved preregistration file is unavailable") from error
    if (hashlib.sha256(content).hexdigest() != approval.preregistration_sha256
            or protocol.identity.encode() not in content
            or initial_preflight.checksum.encode() not in content
            or ordering_plan.initial_observations_sha256.encode() not in content
            or ordering_plan.checksum.encode() not in content
            or not committed_checker(str(preregistration), approval.preregistration_sha256)):
        raise ValueError("approved preregistration is not an exact committed ordering file")


def validate_ordering_evidence(
    protocol: FrozenProtocol,
    manifest: DatasetManifest,
    source_rows: tuple[DatasetRow, ...],
    initial_preflight: PreflightResult,
    initial_observations: Sequence[Observation],
    ordering_plan: OrderingPlan,
) -> None:
    """Validate the complete initial anchor and regenerate the plan without any ledger."""
    protocol.validate()
    manifest.validate()
    if not isinstance(ordering_plan, OrderingPlan):
        raise ValueError("ordering collection requires a typed frozen ordering plan")
    ordering_plan.validate()
    if (ordering_plan.initial_protocol_identity != protocol.identity
            or ordering_plan.manifest_sha256 != manifest_sha256(manifest)
            or ordering_plan.initial_preflight_checksum != getattr(initial_preflight, "checksum", None)):
        raise ValueError("ordering plan does not bind the supplied initial provenance")
    expected = plan_ordering(protocol, manifest, source_rows, initial_preflight,
                             initial_observations, ordering_plan.initial_result_artifact,
                             ordering_plan.treatments)
    if ordering_plan != expected:
        raise ValueError("ordering plan does not match the exact regenerated initial evidence")


def _validate_ordering_gates(
    protocol: FrozenProtocol,
    manifest: DatasetManifest,
    source_rows: tuple[DatasetRow, ...],
    initial_preflight: PreflightResult,
    initial_observations: Sequence[Observation],
    ordering_plan: OrderingPlan,
    cache: CacheStore,
    approval: CollectionApproval,
    committed_checker: CommittedChecker,
    options: CollectionOptions,
) -> None:
    options.validate()
    if options.max_new_attempts is None:
        raise ValueError("ordering collection requires an explicit maximum-new attempt bound")
    approval.validate()
    if not approval.confirmed:
        raise ValueError("explicit collection confirmation is required")
    validate_ordering_evidence(protocol, manifest, source_rows, initial_preflight,
                               initial_observations, ordering_plan)
    if approval.protocol_hash != protocol.identity or approval.preflight_checksum != ordering_plan.checksum:
        raise ValueError("collection approval does not bind the frozen ordering plan")
    validate_ordering_preregistration(protocol, initial_preflight, ordering_plan,
                                      approval, committed_checker)
    expected_approved = [item.as_manifest_entry() for item in _approved_requests(ordering_plan)]
    config = cache.frozen_configuration
    stored_approved = config.get("approved_requests")
    stored_base = ([{key: value for key, value in item.items() if key != "provider_model_identity"}
                    for item in stored_approved]
                   if isinstance(stored_approved, list) and all(isinstance(item, Mapping) for item in stored_approved)
                   else None)
    models = tuple(model for model in protocol.models
                   if model.semantic_identity == ordering_plan.cells[0].request.model
                   and model.route_identity.startswith("jev:"))
    if len(models) != 1:
        raise ValueError("ordering plan does not bind one frozen JEV route")
    provider_identity = models[0].route_identity.split(":", 1)[1]
    if (isinstance(stored_approved, list)
            and any(item.get("provider_model_identity") is not None
                    and item.get("provider_model_identity") != provider_identity
                    for item in stored_approved if isinstance(item, Mapping))):
        raise ValueError("ordering cache provider identity does not match the frozen JEV route")
    if (cache.task.fingerprint != protocol.task.fingerprint
            or config.get("preflight_fingerprint") != ordering_plan.checksum
            or stored_base != expected_approved
            or cache.ceiling != approval.attempt_ceiling
            or cache.attempts_used > cache.ceiling
            or cache.ceiling > ordering_plan.physical_count * (1 + options.max_retries_per_request)
            or options.max_new_attempts > cache.ceiling - cache.attempts_used):
        raise ValueError("ordering cache ceiling does not match the frozen ordering approval")
    expected_mapping = {cell.request.id: cell.request.fingerprint for cell in ordering_plan.cells}
    snapshot = cache.snapshot()
    if any(expected_mapping.get(cell_id) != request_id
           for cell_id, request_id in snapshot.logical_mapping.items()):
        raise ValueError("ordering ledger contains logical cells outside the frozen matrix")


def _approved_requests(ordering_plan: OrderingPlan) -> tuple[ApprovedRequest, ...]:
    """Deduplicate only exact physical ordering requests for the dedicated ledger."""
    physical = {cell.request.fingerprint: cell.request for cell in ordering_plan.cells}
    return tuple(
        ApprovedRequest(request.fingerprint, request.task_fingerprint, request.wire_fingerprint,
                        request.dataset_revision, request.model)
        for request in sorted(physical.values(), key=lambda item: item.fingerprint)
    )
