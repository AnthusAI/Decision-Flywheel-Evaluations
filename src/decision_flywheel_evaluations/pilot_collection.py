"""Bounded collection for a frozen development-only compatibility pilot."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping

from .cache import ApprovedRequest, CacheStore
from .collector import (CollectionApproval, CollectionOptions, CollectionResult,
                        CommittedChecker, EngineFactory, _collect_execution,
                        _execution_plans, git_preregistration_is_committed)
from .datasets import DatasetRow
from .manifests import DatasetManifest
from .pilot import PilotPlan, plan_jev_pilot
from .preflight import PreflightResult, manifest_sha256
from .protocol import FrozenProtocol


async def collect_pilot(
    protocol: FrozenProtocol,
    manifest: DatasetManifest,
    rows: Iterable[DatasetRow],
    source_preflight: PreflightResult,
    pilot_plan: PilotPlan,
    cache: CacheStore,
    approval: CollectionApproval,
    engine_factory: EngineFactory,
    *,
    committed_checker: CommittedChecker = None,  # type: ignore[assignment]
    options: CollectionOptions = CollectionOptions(),
) -> CollectionResult:
    """Collect a small frozen pilot without permitting benchmark conclusions.

    The pilot is restricted to candidate and development source rows, an exact
    regenerated plan, and a no-retry physical ceiling.  Returned observations
    retain the regular structural shape solely for ledger compatibility.
    """
    checker = committed_checker or git_preregistration_is_committed
    source_rows = tuple(rows)
    _validate_pilot_gates(protocol, manifest, source_rows, source_preflight, pilot_plan,
                          cache, approval, checker, options)
    execution = _execution_plans(protocol, manifest, source_rows, pilot_plan.cells)
    return await _collect_execution(protocol, manifest, execution, cache, engine_factory, options)


def _validate_pilot_gates(
    protocol: FrozenProtocol,
    manifest: DatasetManifest,
    source_rows: tuple[DatasetRow, ...],
    source_preflight: PreflightResult,
    pilot_plan: PilotPlan,
    cache: CacheStore,
    approval: CollectionApproval,
    committed_checker: CommittedChecker,
    options: CollectionOptions,
) -> None:
    options.validate()
    if (options.max_new_attempts is None or options.retry_failed or options.recover_uncertain
            or options.max_retries_per_request != 0):
        raise ValueError("pilot collection requires an explicit no-retry attempt bound")
    approval.validate()
    if not approval.confirmed:
        raise ValueError("explicit collection confirmation is required")
    protocol.validate()
    manifest.validate()
    scoreboard_ids = {record.id for record in manifest.scoreboard}
    if any(row.id in scoreboard_ids for row in source_rows):
        raise ValueError("pilot source rows must exclude the protected scoreboard")
    if not isinstance(pilot_plan, PilotPlan):
        raise ValueError("pilot collection requires a typed frozen pilot plan")
    pilot_plan.validate()
    if (pilot_plan.protocol_identity != protocol.identity
            or pilot_plan.manifest_sha256 != manifest_sha256(manifest)
            or pilot_plan.source_preflight_checksum != getattr(source_preflight, "checksum", None)):
        raise ValueError("pilot plan does not bind the supplied protocol, manifest, and source preflight")
    # The source preflight and all pilot cell provenance are regenerated before
    # any factory can be reached.
    expected = plan_jev_pilot(protocol, manifest, source_rows, source_preflight)
    if pilot_plan != expected:
        raise ValueError("pilot plan does not match the exact regenerated development plan")
    if approval.protocol_hash != protocol.identity or approval.preflight_checksum != pilot_plan.checksum:
        raise ValueError("collection approval does not bind the frozen pilot plan")
    validate_pilot_preregistration(protocol, source_preflight, pilot_plan, approval, committed_checker)
    expected_approved = [item.as_manifest_entry() for item in _approved_requests(pilot_plan)]
    config = cache.frozen_configuration
    stored_approved = config.get("approved_requests")
    stored_base = ([{key: value for key, value in item.items() if key != "provider_model_identity"}
                    for item in stored_approved]
                   if isinstance(stored_approved, list) and all(isinstance(item, Mapping) for item in stored_approved)
                   else None)
    pilot_models = tuple(model for model in protocol.models
                         if model.semantic_identity == pilot_plan.cells[0].model
                         and model.route_identity.startswith("jev:"))
    if len(pilot_models) != 1:
        raise ValueError("pilot plan does not bind one frozen JEV route")
    provider_identity = pilot_models[0].route_identity.split(":", 1)[1]
    if (isinstance(stored_approved, list)
            and any(item.get("provider_model_identity") is not None
                    and item.get("provider_model_identity") != provider_identity
                    for item in stored_approved if isinstance(item, Mapping))):
        raise ValueError("pilot cache provider identity does not match the frozen JEV route")
    if (cache.task.fingerprint != protocol.task.fingerprint
            or config.get("preflight_fingerprint") != pilot_plan.checksum
            or stored_base != expected_approved
            or cache.ceiling != approval.attempt_ceiling
            or cache.ceiling != pilot_plan.physical_count
            or cache.attempts_used > cache.ceiling
            or options.max_new_attempts > cache.ceiling - cache.attempts_used):
        raise ValueError("pilot cache ceiling does not match the frozen pilot approval")
    if any(attempts > 1 for attempts in cache.snapshot().physical_attempts.values()):
        raise ValueError("pilot cache has historical retries despite its no-retry approval")


def validate_pilot_preregistration(
    protocol: FrozenProtocol, source_preflight: PreflightResult, pilot_plan: PilotPlan,
    approval: CollectionApproval, committed_checker: CommittedChecker,
) -> None:
    """Shared commit gate, also checked before a CLI can create its ledger."""
    approval.validate()
    preregistration = Path(approval.preregistration_path)
    try:
        content = preregistration.read_bytes()
    except OSError as error:
        raise ValueError("approved preregistration file is unavailable") from error
    if (hashlib.sha256(content).hexdigest() != approval.preregistration_sha256
            or protocol.identity.encode() not in content
            or source_preflight.checksum.encode() not in content
            or pilot_plan.checksum.encode() not in content
            or not committed_checker(str(preregistration), approval.preregistration_sha256)):
        raise ValueError("approved preregistration is not an exact committed pilot file")


def _approved_requests(pilot_plan: PilotPlan) -> tuple[ApprovedRequest, ...]:
    """Derive the exact physical cache whitelist from the frozen logical cells."""
    physical = {cell.fingerprint: cell for cell in pilot_plan.cells}
    return tuple(
        ApprovedRequest(cell.fingerprint, cell.task_fingerprint, cell.wire_fingerprint,
                        cell.dataset_revision, cell.model)
        for cell in sorted(physical.values(), key=lambda item: item.fingerprint)
    )
